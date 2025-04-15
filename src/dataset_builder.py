from pathlib import Path
import os
import json
import gc
import psutil
from datetime import datetime

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
# datasets for individual flies or entire experiments. It can be run in
# two modes: "experiment" or "fly". In "experiment" mode, it processes
# all flies in a given experiment folder, while in "fly" mode, it processes
# individual flies. The script can also create pooled datasets for
# easier analysis.

# Available datasets:
# - coordinates : Full coordinates of ball and fly positions over time
# - event_metrics : individual event metrics for each fly
# - summary : Summary of all events for each fly
# - standardized_contacts : contact data reworked to be time standardized and fly centric
# - transposed: standardized contacts + associated statistics transposed to be one row per event
# - contact_data : skeleton data associated with individual events (redundant with Skeleton_contacts) => Should probably be reworked


# Example usage:
# python dataset_builder.py --mode experiment --yaml config.yaml
# python dataset_builder.py --mode fly --yaml config.yaml

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
        "data_root": Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/"),
        "dataset_dir": Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"),
        "excluded_folders": [],
        "output_summary_dir": Path("250411_Transposed_Ctrls"),
        "config_path": "config.json",
    },
    "PROCESSING": {
        "experiment_filter": "",  # Filter for experiment folders
        "metrics": ["standardized_contacts"],  # Metrics to process (add/remove as needed)
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


def process_fly(fly_path, metric, output_data):
    """
    Process an individual fly and save its dataset.

    Parameters
    ----------
    fly_path : Path
        Path to the fly's .h5 file.
    metric : str
        Metric to process.
    output_data : Path
        Output directory for the dataset.

    Returns
    -------
    Path or None
        Path to the saved dataset or None if processing failed.
    """
    try:
        logging.info(f"Processing fly: {fly_path.name} for metric: {metric}")
        # Load the fly data
        fly = Ballpushing_utils.Fly(fly_path, as_individual=True)
        dataset = Ballpushing_utils.Dataset(fly, dataset_type=metric)

        if dataset.data is not None and not dataset.data.empty:
            # Save the dataset
            dataset_path = output_data / metric / f"{fly.metadata.name}_{metric}.feather"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
            dataset.data.to_feather(dataset_path)
            logging.info(f"Saved {metric} dataset for {fly.metadata.name} to {dataset_path}")
            return dataset_path
        else:
            logging.warning(f"No data available for {fly.metadata.name} with metric {metric}")
            return None
    except Exception as e:
        logging.error(f"Error processing {fly_path} for metric {metric}: {str(e)}")
        return None


def process_experiment(folder, metrics, output_data):
    """
    Process an experiment folder in fly mode.

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

    # Find all directories containing .h5 files (corridor folders)
    corridor_folders = [
        subfolder for subfolder in folder.rglob("*") if subfolder.is_dir() and any(subfolder.glob("*.h5"))
    ]
    if not corridor_folders:
        logging.warning(f"No corridor folders with .h5 files found in {folder.name}")
        return

    # Prepare tasks for parallel processing
    tasks = []
    for fly_path in corridor_folders:
        for metric in metrics:
            tasks.append((fly_path, metric))

    # Process each fly with each metric in parallel
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_fly, fly_path, metric, output_data): (fly_path, metric)
            for fly_path, metric in tasks
        }

        # Use tqdm to track progress
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {folder.name}"
        ):
            try:
                future.result()
            except Exception as e:
                fly_path, metric = futures[future]
                logging.error(f"Error processing fly {fly_path} for metric {metric}: {str(e)}")

    # Filter out None results and concatenate the datasets by metric
    for metric in metrics:
        metric_files = list((output_data / metric).glob("*.feather"))
        valid_files = []

        # Validate Feather files
        for file in metric_files:
            try:
                # Attempt to read the file to ensure it's valid
                df = pd.read_feather(file)
                if not df.empty:
                    valid_files.append(file)
                else:
                    logging.warning(f"Empty Feather file skipped: {file}")
            except Exception as e:
                logging.error(f"Invalid Feather file skipped: {file}, Error: {e}")

        if valid_files:
            # Concatenate all valid datasets for the current metric
            combined_df = pd.concat([pd.read_feather(f) for f in valid_files], ignore_index=True)
            combined_path = output_data / metric / f"{folder.name}_{metric}.feather"
            combined_df.to_feather(combined_path)
            logging.info(f"Saved combined {metric} dataset for {folder.name} to {combined_path}")

            # Optionally delete individual files to save disk space
            for file in valid_files:
                file.unlink()
        else:
            logging.warning(f"No valid datasets found for {metric} in {folder.name}")


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
        choices=["experiment", "fly"],
        default="experiment",
        help="Processing mode: 'experiment' (default) or 'fly'",
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

    if CONFIG["PATHS"]["output_summary_dir"]:
        # Use the output directory if provided
        output_summary = Path(CONFIG["PATHS"]["output_summary_dir"])
        logging.info(f"Using optional output directory: {output_summary}")
    else:
        # Determine the output_summary_dir based on the provided arguments
        today_date = datetime.now().strftime("%y%m%d")
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

    # Create output directories
    output_summary.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)

    # Get list of experiment folders to process
    Exp_folders = []
    if args.yaml:
        # Load directories from YAML file
        yaml_dirs = load_yaml_config(args.yaml)
        Exp_folders.extend(yaml_dirs)
    else:
        for root in CONFIG["PATHS"]["data_root"]:
            Exp_folders.extend(
                folder
                for folder in root.iterdir()
                if folder.is_dir() and CONFIG["PROCESSING"]["experiment_filter"] in folder.name
            )

    # Exclude folders based on config list
    if CONFIG["PATHS"]["excluded_folders"]:
        Exp_folders = [folder for folder in Exp_folders if folder.name not in CONFIG["PATHS"]["excluded_folders"]]

    if not Exp_folders:
        raise ValueError("No experiment folders found to process. Check your YAML file or experiment filter.")

    logging.info(f"Folders to analyze: {[f.name for f in Exp_folders]}")

    # Create metric subdirectories
    for metric in CONFIG["PROCESSING"]["metrics"]:
        (output_data / metric).mkdir(exist_ok=True)

    if args.mode == "experiment":
        # Main processing loop
        checkpoint_file = output_summary / "processing_checkpoint.json"

        # Load checkpoint if exists
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                processed_folders = set(json.load(f))
            logging.info(f"Resuming from checkpoint, {len(processed_folders)} folders already processed")
        else:
            processed_folders = set()

        # Process each folder
        for folder in Exp_folders:
            exp_name = folder.name
            experiment_pkl_path = output_summary / f"{exp_name}.pkl"
            experiment = None  # Initialize experiment to avoid unbound variable issues

            # Skip if already processed
            if exp_name in processed_folders:
                logging.info(f"Skipping already processed experiment: {exp_name}")
                continue

            log_memory_usage(f"Before processing {exp_name}")

            try:
                # Check if the experiment exists
                if experiment_pkl_path.exists():
                    logging.info(f"Experiment {exp_name} already exists.")
                    # Check if all datasets for this experiment exist
                    all_datasets_exist = True
                    for metric in CONFIG["PROCESSING"]["metrics"]:
                        dataset_path = output_data / metric / f"{exp_name}_{metric}.feather"
                        if not dataset_path.exists():
                            all_datasets_exist = False
                            break

                    if all_datasets_exist:
                        logging.info(f"All datasets for {exp_name} already exist. Skipping.")
                        continue
                    else:
                        logging.info(f"Some datasets for {exp_name} are missing. Loading experiment.")
                        experiment = utilities.load_object(experiment_pkl_path)
                else:
                    # If the experiment doesn't exist, create it
                    logging.info(f"Creating new experiment: {exp_name}")
                    experiment = Ballpushing_utils.Experiment(folder)
                    utilities.save_object(experiment, experiment_pkl_path)
                    logging.info(f"Saved new experiment: {exp_name}")

                    # Save config if first experiment
                    if not (output_summary / CONFIG["PATHS"]["config_path"]).exists():
                        config_dict = (
                            experiment.config if isinstance(experiment.config, dict) else vars(experiment.config)
                        )
                        config_json_path = output_summary / CONFIG["PATHS"]["config_path"]
                        with open(config_json_path, "w") as config_file:
                            json.dump(config_dict, config_file, indent=4)
                        logging.info(f"Saved config for {exp_name} to {config_json_path}")

                # Generate and save datasets
                all_metrics_processed = True
                for metric in CONFIG["PROCESSING"]["metrics"]:
                    dataset_path = output_data / metric / f"{exp_name}_{metric}.feather"

                    if dataset_path.exists():
                        logging.info(f"Dataset {dataset_path} already exists. Skipping.")
                        continue

                    try:
                        dataset = Ballpushing_utils.Dataset(experiment, dataset_type=metric)
                        if dataset.data is not None and not dataset.data.empty:
                            dataset.data.to_feather(dataset_path)
                            logging.info(f"Saved {metric} dataset for {exp_name}")
                        else:
                            logging.warning(f"Empty {metric} data for {exp_name}")
                    except Exception as e:
                        logging.error(f"Error generating {metric} for {exp_name}: {str(e)}")
                        all_metrics_processed = False

                # Mark as processed only if all metrics were processed
                if all_metrics_processed:
                    processed_folders.add(exp_name)
                    # Save checkpoint after each successful experiment
                    with open(checkpoint_file, "w") as f:
                        json.dump(list(processed_folders), f)

            except Exception as e:
                logging.error(f"Error processing {exp_name}: {str(e)}")

            # Explicit cleanup
            del experiment
            gc.collect()
            log_memory_usage(f"After processing {exp_name}")

    elif args.mode == "fly":
        # Main processing loop for fly mode
        checkpoint_file = output_summary / "fly_processing_checkpoint.json"

        # Load checkpoint if exists
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                processed_experiments = set(json.load(f))
            logging.info(f"Resuming from checkpoint, {len(processed_experiments)} experiments already processed")
        else:
            processed_experiments = set()

        for folder in Exp_folders:
            exp_name = folder.name
            experiment_pkl_path = output_summary / f"{exp_name}.pkl"

            # Skip if the experiment has already been processed
            if exp_name in processed_experiments:
                logging.info(f"Skipping already processed experiment: {exp_name}")
                continue

            # Check if the experiment-level datasets already exist
            all_datasets_exist = True
            for metric in CONFIG["PROCESSING"]["metrics"]:
                dataset_path = output_data / metric / f"{exp_name}_{metric}.feather"
                if not dataset_path.exists():
                    all_datasets_exist = False
                    break

            if all_datasets_exist:
                logging.info(f"All datasets for {exp_name} already exist. Skipping.")
                processed_experiments.add(exp_name)
                # Save checkpoint after skipping
                with open(checkpoint_file, "w") as f:
                    json.dump(list(processed_experiments), f)
                continue

            # Process the experiment folder
            logging.info(f"Processing experiment folder: {exp_name}")
            try:
                process_experiment(folder, CONFIG["PROCESSING"]["metrics"], output_data)
                processed_experiments.add(exp_name)
                # Save checkpoint after successful processing
                with open(checkpoint_file, "w") as f:
                    json.dump(list(processed_experiments), f)
            except Exception as e:
                logging.error(f"Error processing experiment {exp_name}: {str(e)}")

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
