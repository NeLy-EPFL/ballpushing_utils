from pathlib import Path
import os
import json
import gc
import psutil
from datetime import datetime

import threading

import concurrent.futures
from tqdm import tqdm  # For progress bar
import time
import pandas as pd
import yaml  # Import PyYAML for parsing YAML files
import argparse  # For command-line arguments
from concurrent.futures import ThreadPoolExecutor
import logging  # For improved logging
import Ballpushing_utils
from Ballpushing_utils import utilities, config

# ==================================================================
# INTRODUCTION AND EXAMPLE USAGE
# ==================================================================

# This script processes datasets for ball-pushing experiments, generating
# datasets for experiments.

# Available datasets:
# - coordinates : Full coordinates of ball and fly positions over time
# - event_metrics : individual event metrics for each fly
# - summary : Summary of all events for each fly
# - standardized_contacts : contact data reworked to be time standardized and fly centric
# - transposed: standardized contacts + associated statistics transposed to be one row per event
# - contact_data : skeleton data associated with individual events (redundant with Skeleton_contacts) => Should probably be reworked


# Example usage:
# Process experiment folders:
# python dataset_builder.py --mode experiment --yaml config.yaml
#
# Process individual fly directories from a combined YAML file:
# python dataset_builder.py --mode flies --yaml combined_flies.yaml
#
# Process experiments with filter (no YAML):
# python dataset_builder.py --mode experiment
#
# Verify existing datasets and create missing pooled datasets:
# python dataset_builder.py --verify /path/to/existing/dataset/directory
#
# Verify with completeness checking against YAML:
# python dataset_builder.py --verify /path/to/existing/dataset/directory --yaml folders.yaml --mode experiment

# ==================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES TO MODIFY BEHAVIOR
# ==================================================================
# Data root is where to find the raw videos and tracking data
# dataset_dir is where to save the processed datasets
# excluded_folders is a list of folders from data root to exclude from processing
# config_path is the path to the configuration file for the experiment. It just sets the name of the json file which will be saved along with the experiment files.
# experiment_filter can be used to select the experiments to process based on a substring in the folder name (e.g. TNT_Fine)
# metrics is a list of metrics to process.

CONFIG = {
    "PATHS": {
        "data_root": [Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/")],
        "dataset_dir": Path("/mnt/upramdya_data/MD/Ballpushing_Balltypes/Datasets"),
        "excluded_folders": [],
        "output_summary_dir": None,  # "250419_transposed_control_folders"  # Optional output directory for summary files, should be a Path object
        "config_path": "config.json",
    },
    "PROCESSING": {
        "experiment_filter": "",  # Filter for a specific experiment folder to test
        "metrics": [
            "standardized_contacts",
            "summary",  # Re-enabled for testing the optimization
            # "coordinates",
        ],  # Metrics to process (add/remove as needed)
    },
}

# Print the current ball pushing configurations
current_config = config.Config()
print("Current ball pushing configurations:")
current_config.print_config()

# ==================================================================
# LOGGING CONFIGURATION
# ==================================================================
logging.basicConfig(
    level=logging.INFO,  # Change this to WARNING or ERROR to reduce log verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


# ==================================================================
# UTILITY FUNCTIONS
# ==================================================================


def load_yaml_config(yaml_path):
    """
    Load directories from a YAML file and update the CONFIG dictionary.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file.

    Returns
    -------
    list
        List of directories specified in the YAML file.
    """
    try:
        with open(yaml_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        directories = yaml_data.get("directories", [])
        logging.info(f"Loaded {len(directories)} directories from YAML file: {yaml_path}")
        return [Path(directory) for directory in directories]
    except FileNotFoundError:
        logging.error(f"YAML file not found: {yaml_path}")
        return []
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return []


def log_memory_usage(label):
    """
    Log current memory usage.

    Parameters
    ----------
    label : str
        Label to identify the memory usage log.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logging.info(f"Memory usage at {label}: {memory_info.rss / 1024 / 1024:.2f} MB")


def process_experiment(folder, metrics, output_data):
    """
    Process an experiment folder in experiment mode.

    Parameters
    ----------
    folder : Path
        Path to the experiment folder.
    metrics : list
        List of metrics to process.
    output_data : Path
        Output directory for datasets.
    """
    logging.info(f"Processing experiment folder: {folder.name}")

    try:
        # Check if all datasets already exist
        all_datasets_exist = True
        for metric in metrics:
            dataset_path = output_data / metric / f"{folder.name}_{metric}.feather"
            if not dataset_path.exists():
                all_datasets_exist = False
                break

        if all_datasets_exist:
            logging.info(f"All datasets for {folder.name} already exist. Skipping.")
            return

        # Check if the Experiment object exists or needs to be created
        experiment = None
        for metric in metrics:
            dataset_path = output_data / metric / f"{folder.name}_{metric}.feather"

            if dataset_path.exists():
                logging.info(f"Dataset {dataset_path} already exists. Skipping.")
                continue

            # Create the Experiment object only if needed
            if experiment is None:
                experiment = Ballpushing_utils.Experiment(folder)

            # Process the dataset
            dataset = Ballpushing_utils.Dataset(experiment, dataset_type=metric)
            if dataset.data is not None and not dataset.data.empty:
                dataset_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
                dataset.data.to_feather(dataset_path)
                logging.info(f"Saved {metric} dataset for {folder.name} to {dataset_path}")
            else:
                logging.warning(f"No data available for {folder.name} with metric {metric}")

            # Clear caches between experiments to prevent memory accumulation
            # For experiment-level processing, we need to clear caches for each fly
            if experiment is not None and hasattr(experiment, "flies"):
                for fly in experiment.flies:
                    # Clear all caches on the Fly object (includes BallPushingMetrics, FlyTrackingData, etc.)
                    if hasattr(fly, "clear_caches"):
                        fly.clear_caches()
                        print(f"Cleared all caches for fly: {fly.metadata.name}")

            # Force garbage collection
            gc.collect()

    except Exception as e:
        logging.error(f"Error processing experiment {folder.name}: {str(e)}")


def process_fly_directory(fly_dir, metrics, output_data):
    """
    Process an individual fly directory in flies mode.

    Parameters
    ----------
    fly_dir : Path
        Path to the fly directory (e.g., /path/to/experiment/arena1/corridor1).
    metrics : list
        List of metrics to process.
    output_data : Path
        Output directory for datasets.
    """
    logging.info(f"Processing fly directory: {fly_dir}")

    try:
        # Extract meaningful name from fly directory path
        # e.g., /path/to/240111_TNT_Fine_1_Videos_Tracked/arena6/corridor1
        # becomes 240111_TNT_Fine_1_Videos_Tracked_arena6_corridor1
        experiment_folder = fly_dir.parent.parent  # experiment folder (where Metadata.json is located)
        experiment_name = experiment_folder.name  # experiment folder name
        arena_name = fly_dir.parent.name  # arena folder name
        corridor_name = fly_dir.name  # corridor folder name
        fly_name = f"{experiment_name}_{arena_name}_{corridor_name}"

        # Verify that the experiment folder exists and contains Metadata.json
        metadata_path = experiment_folder / "Metadata.json"
        if not metadata_path.exists():
            logging.error(f"Metadata.json not found in experiment folder: {experiment_folder}")
            return

        # Check if all datasets already exist
        all_datasets_exist = True
        for metric in metrics:
            dataset_path = output_data / metric / f"{fly_name}_{metric}.feather"
            if not dataset_path.exists():
                all_datasets_exist = False
                break

        if all_datasets_exist:
            logging.info(f"All datasets for {fly_name} already exist. Skipping.")
            return

        # Create individual Fly object with as_individual=True to avoid loading full experiment
        fly = Ballpushing_utils.Fly(fly_dir, as_individual=True)

        # Process each metric for this specific fly
        for metric in metrics:
            dataset_path = output_data / metric / f"{fly_name}_{metric}.feather"

            if dataset_path.exists():
                logging.info(f"Dataset {dataset_path} already exists. Skipping.")
                continue

            # Create dataset for this specific fly (pass the fly object directly)
            dataset = Ballpushing_utils.Dataset(fly, dataset_type=metric)

            if dataset.data is not None and not dataset.data.empty:
                # No need to filter since we're working with individual fly data
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset.data.to_feather(dataset_path)
                logging.info(f"Saved {metric} dataset for {fly_name} to {dataset_path} ({len(dataset.data)} rows)")
            else:
                logging.warning(f"No data available for {fly_name} with metric {metric}")

        # Clear caches and force garbage collection after processing each fly
        if hasattr(fly, "clear_caches"):
            fly.clear_caches()
            print(f"Cleared all caches for fly: {fly.metadata.name}")

        # Clear the fly object
        del fly

        # Force garbage collection
        gc.collect()
        logging.debug(f"Cleared caches and performed garbage collection after processing {fly_name}")

    except Exception as e:
        logging.error(f"Error processing fly directory {fly_dir}: {str(e)}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")


def extract_experiment_folders_from_fly_dirs(fly_dirs):
    """
    Extract unique experiment folders from a list of fly directories.

    Parameters
    ----------
    fly_dirs : list
        List of fly directory paths.

    Returns
    -------
    list
        List of unique experiment folder paths.
    """
    experiment_folders = set()
    for fly_dir in fly_dirs:
        # Assume fly_dir structure is: /path/to/experiment/arena/corridor
        experiment_folder = fly_dir.parent.parent
        experiment_folders.add(experiment_folder)
    return list(experiment_folders)


def verify_and_complete_datasets(verify_path, yaml_path=None, mode="experiment"):
    """
    Verify existing datasets and create missing pooled datasets.

    Parameters
    ----------
    verify_path : str or Path
        Path to the dataset directory to verify (should contain metric subdirectories).
    yaml_path : str or Path, optional
        Path to YAML file containing expected directories for completeness checking.
    mode : str
        Processing mode ("experiment" or "flies") to determine how to interpret YAML entries.
    """
    verify_path = Path(verify_path)

    if not verify_path.exists():
        logging.error(f"Verification path does not exist: {verify_path}")
        return

    logging.info(f"Verifying datasets in: {verify_path}")

    # Load expected items from YAML if provided
    expected_items = []
    if yaml_path:
        yaml_dirs = load_yaml_config(yaml_path)
        if mode == "experiment":
            expected_items = [d.name for d in yaml_dirs]
        elif mode == "flies":
            # For flies mode, create expected fly names from directory paths
            expected_items = [f"{d.parent.parent.name}_{d.parent.name}_{d.name}" for d in yaml_dirs]
        logging.info(f"Loaded {len(expected_items)} expected items from YAML file for completeness checking")

    # Find all metric subdirectories
    metric_dirs = [d for d in verify_path.iterdir() if d.is_dir()]

    if not metric_dirs:
        logging.warning(f"No metric subdirectories found in {verify_path}")
        return

    logging.info(f"Found metric directories: {[d.name for d in metric_dirs]}")

    for metric_dir in metric_dirs:
        metric_name = metric_dir.name
        logging.info(f"Checking metric: {metric_name}")

        # Find all individual dataset files (exclude pooled files)
        individual_files = [f for f in metric_dir.glob("*.feather") if not f.name.startswith("pooled_")]
        pooled_file = metric_dir / f"pooled_{metric_name}.feather"

        logging.info(f"  Found {len(individual_files)} individual files")

        # Check completeness if YAML provided
        if expected_items:
            # Extract item names from existing files
            existing_items = set()
            for f in individual_files:
                # Remove the metric suffix to get the item name
                # e.g., "experiment_name_summary.feather" -> "experiment_name"
                item_name = f.stem.replace(f"_{metric_name}", "")
                existing_items.add(item_name)

            missing_items = set(expected_items) - existing_items
            extra_items = existing_items - set(expected_items)

            logging.info(f"  Expected {len(expected_items)} items, found {len(existing_items)} items")

            if missing_items:
                logging.warning(f"  ⚠ Missing {len(missing_items)} expected items:")
                for item in sorted(missing_items):
                    logging.warning(f"    - {item}")

            if extra_items:
                logging.info(f"  ℹ Found {len(extra_items)} extra items not in YAML:")
                for item in sorted(extra_items):
                    logging.info(f"    + {item}")

            if not missing_items:
                logging.info(f"  ✓ All expected items are present for metric {metric_name}")

        if individual_files:
            # Check if pooled dataset exists
            if pooled_file.exists():
                logging.info(f"  ✓ Pooled dataset already exists: {pooled_file.name}")

                # Optionally check if pooled dataset is up-to-date
                pooled_mtime = pooled_file.stat().st_mtime
                newest_individual_mtime = max(f.stat().st_mtime for f in individual_files)

                if newest_individual_mtime > pooled_mtime:
                    logging.warning(f"  ⚠ Pooled dataset is older than some individual files")
                    create_pooled_dataset(metric_dir, metric_name, individual_files, force=True)
                else:
                    logging.info(f"  ✓ Pooled dataset is up-to-date")
            else:
                logging.info(f"  ❌ Missing pooled dataset: {pooled_file.name}")
                create_pooled_dataset(metric_dir, metric_name, individual_files)
        else:
            logging.warning(f"  ⚠ No individual files found for metric {metric_name}")

    # Summary
    if yaml_path:
        logging.info("\n" + "=" * 60)
        logging.info("VERIFICATION SUMMARY")
        logging.info("=" * 60)
        for metric_dir in metric_dirs:
            metric_name = metric_dir.name
            individual_files = [f for f in metric_dir.glob("*.feather") if not f.name.startswith("pooled_")]
            pooled_file = metric_dir / f"pooled_{metric_name}.feather"

            existing_items = set()
            for f in individual_files:
                item_name = f.stem.replace(f"_{metric_name}", "")
                existing_items.add(item_name)

            missing_count = len(set(expected_items) - existing_items)
            pooled_status = "✓" if pooled_file.exists() else "❌"

            logging.info(
                f"{metric_name}: {len(existing_items)}/{len(expected_items)} items, "
                f"{missing_count} missing, pooled: {pooled_status}"
            )


def create_pooled_dataset(metric_dir, metric_name, individual_files, force=False):
    """
    Create a pooled dataset from individual files.

    Parameters
    ----------
    metric_dir : Path
        Directory containing the metric files.
    metric_name : str
        Name of the metric.
    individual_files : list
        List of individual dataset files to pool.
    force : bool
        Whether to overwrite existing pooled dataset.
    """
    pooled_path = metric_dir / f"pooled_{metric_name}.feather"

    if pooled_path.exists() and not force:
        logging.info(f"Pooled dataset already exists: {pooled_path.name}")
        return

    try:
        if force and pooled_path.exists():
            logging.info(f"Overwriting existing pooled dataset: {pooled_path.name}")

        # Use chunking for pooling to avoid loading all files at once
        chunk_size = 5  # Adjust based on file sizes
        total_chunks = (len(individual_files) + chunk_size - 1) // chunk_size

        logging.info(f"Creating pooled dataset for '{metric_name}' from {len(individual_files)} files")

        first_chunk = True
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(individual_files))
            chunk_files = individual_files[start_idx:end_idx]

            logging.info(f"  Processing chunk {chunk_idx + 1}/{total_chunks}")

            try:
                chunk_dfs = []
                for f in chunk_files:
                    try:
                        df = pd.read_feather(f)
                        chunk_dfs.append(df)
                    except Exception as e:
                        logging.warning(f"  Failed to read {f.name}: {str(e)}")

                if not chunk_dfs:
                    logging.warning(f"  No valid data in chunk {chunk_idx + 1}")
                    continue

                chunk_df = pd.concat(chunk_dfs, ignore_index=True)

                if first_chunk:
                    # First chunk: create the file
                    chunk_df.to_feather(pooled_path)
                    first_chunk = False
                    total_rows = len(chunk_df)
                else:
                    # Subsequent chunks: append to existing
                    existing_df = pd.read_feather(pooled_path)
                    combined_df = pd.concat([existing_df, chunk_df], ignore_index=True)
                    combined_df.to_feather(pooled_path)
                    total_rows = len(combined_df)
                    del existing_df, combined_df

                del chunk_df, chunk_dfs
                gc.collect()

            except Exception as e:
                logging.error(f"  Error processing chunk {chunk_idx + 1}: {str(e)}")

        if pooled_path.exists():
            final_df = pd.read_feather(pooled_path)
            logging.info(f"  ✓ Created pooled dataset: {pooled_path.name} ({len(final_df)} total rows)")
        else:
            logging.error(f"  ❌ Failed to create pooled dataset: {pooled_path.name}")

    except Exception as e:
        logging.error(f"Error creating pooled dataset for {metric_name}: {str(e)}")


# ==================================================================
# MAIN PROCESSING SCRIPT
# ==================================================================
if __name__ == "__main__":
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dataset Builder Script")
    parser.add_argument("--yaml", type=str, help="Path to the YAML file specifying directories", default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["experiment", "flies"],
        default="experiment",
        help="Processing mode: 'experiment' for experiment folders, 'flies' for individual fly directories (default: experiment)",
    )
    parser.add_argument(
        "--verify",
        type=str,
        help="Verification mode: path to existing dataset directory to check completeness and create missing pooled datasets",
        default=None,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Handle verification mode
    if args.verify:
        logging.info("Running in verification mode")
        verify_and_complete_datasets(args.verify, yaml_path=args.yaml, mode=args.mode)
        end_time = time.time()
        runtime = end_time - start_time
        logging.info(f"Verification completed in {runtime:.2f} seconds")
        exit(0)

    if CONFIG["PATHS"]["output_summary_dir"]:
        # Use the output directory if provided
        output_summary = Path(CONFIG["PATHS"]["output_summary_dir"])
        logging.info(f"Using optional output directory: {output_summary}")
    else:
        # Determine the output_summary_dir based on the provided arguments
        today_date = datetime.now().strftime("%y%m%d_%H")
        dataset_type = CONFIG["PROCESSING"]["metrics"][0]  # Use the first metric as the dataset type

        if args.yaml:
            yaml_stem = Path(args.yaml).stem  # Extract the stem (filename without extension)
            CONFIG["PATHS"]["output_summary_dir"] = f"{today_date}_{dataset_type}_{yaml_stem}"
        elif CONFIG["PROCESSING"]["experiment_filter"]:
            experiment_filter = CONFIG["PROCESSING"]["experiment_filter"]
            CONFIG["PATHS"]["output_summary_dir"] = f"{today_date}_{dataset_type}_{experiment_filter}"
        else:
            raise ValueError(
                "Either --yaml must be provided or an experiment_filter must be set in the configuration. "
                "Processing the entire dataset is not allowed."
            )

    # Automatically set output_data_dir based on output_summary_dir
    CONFIG["PATHS"]["output_data_dir"] = f"{CONFIG['PATHS']['output_summary_dir']}_Data"

    # Build derived paths from configuration
    output_summary = CONFIG["PATHS"]["dataset_dir"] / CONFIG["PATHS"]["output_summary_dir"]
    output_data = CONFIG["PATHS"]["dataset_dir"] / CONFIG["PATHS"]["output_data_dir"]
    output_summary.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)

    # Save the configuration used for this dataset generation
    config_save_path = output_data / CONFIG["PATHS"]["config_path"]
    try:
        # Include both the CONFIG dictionary and the ball pushing config
        config_to_save = {
            "dataset_builder_config": CONFIG,
            "ballpushing_config": current_config.__dict__,
            "processing_info": {
                "mode": args.mode,
                "yaml_file": args.yaml,
                "threads": args.threads,
                "log_level": args.log_level,
                "timestamp": datetime.now().isoformat(),
            },
        }
        with open(config_save_path, "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)  # default=str handles Path objects
        logging.info(f"Configuration saved to: {config_save_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration to {config_save_path}: {str(e)}")

    # Get list of folders to process based on mode
    processing_items = []

    if args.mode == "experiment":
        # Original experiment mode logic
        Exp_folders = []
        if args.yaml:
            yaml_dirs = load_yaml_config(args.yaml)
            Exp_folders.extend(yaml_dirs)
        else:
            for root in CONFIG["PATHS"]["data_root"]:
                Exp_folders.extend(
                    folder
                    for folder in root.iterdir()
                    if folder.is_dir() and CONFIG["PROCESSING"]["experiment_filter"] in folder.name
                )

        if CONFIG["PATHS"]["excluded_folders"]:
            Exp_folders = [folder for folder in Exp_folders if folder.name not in CONFIG["PATHS"]["excluded_folders"]]

        if not Exp_folders:
            raise ValueError("No experiment folders found to process. Check your YAML file or experiment filter.")

        logging.info(f"Experiment folders to analyze: {[f.name for f in Exp_folders]}")
        processing_items = Exp_folders

    elif args.mode == "flies":
        # New flies mode logic
        if not args.yaml:
            raise ValueError("--yaml argument is required when using --mode flies")

        fly_dirs = load_yaml_config(args.yaml)

        # Validate that all directories exist
        missing_dirs = [d for d in fly_dirs if not d.exists()]
        if missing_dirs:
            logging.warning(f"The following directories do not exist and will be skipped: {missing_dirs}")
            fly_dirs = [d for d in fly_dirs if d.exists()]

        if not fly_dirs:
            raise ValueError("No valid fly directories found in YAML file.")

        logging.info(f"Fly directories to analyze: {len(fly_dirs)} directories")
        logging.info(f"Sample directories: {[str(d) for d in fly_dirs[:5]]}")
        if len(fly_dirs) > 5:
            logging.info(f"... and {len(fly_dirs) - 5} more directories")

        processing_items = fly_dirs

    # Create metric subdirectories
    for metric in CONFIG["PROCESSING"]["metrics"]:
        (output_data / metric).mkdir(exist_ok=True)

    # Main processing loop
    checkpoint_file = output_summary / "processing_checkpoint.json"

    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            processed_items = set(json.load(f))
        logging.info(f"Resuming from checkpoint, {len(processed_items)} items already processed")
    else:
        processed_items = set()

    for item in processing_items:
        # Initialize variables for each item
        item_name = ""
        log_label = ""

        if args.mode == "experiment":
            item_name = item.name
            log_label = f"experiment {item_name}"
        elif args.mode == "flies":
            # Create a unique name for the fly directory
            item_name = f"{item.parent.parent.name}_{item.parent.name}_{item.name}"
            log_label = f"fly {item_name}"

        if item_name in processed_items:
            logging.info(f"Skipping already processed {log_label}")
            continue

        log_memory_usage(f"Before processing {log_label}")

        try:
            if args.mode == "experiment":
                process_experiment(item, CONFIG["PROCESSING"]["metrics"], output_data)
            elif args.mode == "flies":
                process_fly_directory(item, CONFIG["PROCESSING"]["metrics"], output_data)

            processed_items.add(item_name)

            # Save checkpoint after each successful processing
            with open(checkpoint_file, "w") as f:
                json.dump(list(processed_items), f)

        except Exception as e:
            logging.error(f"Error processing {log_label}: {str(e)}")

        log_memory_usage(f"After processing {log_label}")

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logging.info(f"Removed checkpoint file: {checkpoint_file}")

    logging.info("Processing complete!")

    # ==================================================================
    # Create pooled datasets
    # ==================================================================
    for metric in CONFIG["PROCESSING"]["metrics"]:
        pooled_path = output_data / metric / f"pooled_{metric}.feather"

        if not pooled_path.exists():
            try:
                metric_files = list((output_data / metric).glob("*.feather"))
                if not metric_files:
                    logging.warning(f"No {metric} files found for pooling. Skipping.")
                    continue

                # Use chunking for pooling to avoid loading all files at once
                chunk_size = 5  # Adjust based on file sizes
                total_chunks = (len(metric_files) + chunk_size - 1) // chunk_size

                logging.info(f"Pooling {len(metric_files)} files for metric '{metric}' into {pooled_path.name}")
                first_chunk = True
                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, len(metric_files))
                    chunk_files = metric_files[start_idx:end_idx]

                    logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} for metric '{metric}'")
                    chunk_df = pd.concat([pd.read_feather(f) for f in chunk_files])

                    if first_chunk:
                        # First chunk: create the file
                        chunk_df.to_feather(pooled_path)
                        first_chunk = False
                    else:
                        # Subsequent chunks: append to existing
                        existing_df = pd.read_feather(pooled_path)
                        combined_df = pd.concat([existing_df, chunk_df])
                        combined_df.to_feather(pooled_path)
                        del existing_df, combined_df

                    del chunk_df
                    gc.collect()
                    log_memory_usage(f"After pooling chunk {chunk_idx + 1} for metric '{metric}'")

                logging.info(f"Created pooled {metric} dataset: {pooled_path.name}")
            except Exception as e:
                logging.error(f"Error pooling {metric}: {str(e)}")
        else:
            logging.info(f"Pooled {metric} dataset already exists: {pooled_path.name}")

    logging.info("Pooled dataset creation complete!")

    end_time = time.time()
    runtime = end_time - start_time
    logging.info(f"Script completed in {runtime:.2f} seconds")
