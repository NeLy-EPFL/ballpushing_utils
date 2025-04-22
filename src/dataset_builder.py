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
# python dataset_builder.py --mode experiment --yaml config.yaml

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
        "dataset_dir": Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/"),
        "excluded_folders": [],
        "output_summary_dir": Path(
            "250419_transposed_control_folders"
        ),  # Optional output directory for summary files, should be a Path object
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

    except Exception as e:
        logging.error(f"Error processing experiment {folder.name}: {str(e)}")


# ==================================================================
# MAIN PROCESSING SCRIPT
# ==================================================================
if __name__ == "__main__":
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dataset Builder Script")
    parser.add_argument("--yaml", type=str, help="Path to the YAML file specifying directories", default=None)
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
    output_summary.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)

    # Get list of experiment folders to process
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

    logging.info(f"Folders to analyze: {[f.name for f in Exp_folders]}")

    # Create metric subdirectories
    for metric in CONFIG["PROCESSING"]["metrics"]:
        (output_data / metric).mkdir(exist_ok=True)

    # Main processing loop
    checkpoint_file = output_summary / "processing_checkpoint.json"

    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            processed_folders = set(json.load(f))
        logging.info(f"Resuming from checkpoint, {len(processed_folders)} folders already processed")
    else:
        processed_folders = set()

    for folder in Exp_folders:
        exp_name = folder.name

        if exp_name in processed_folders:
            logging.info(f"Skipping already processed experiment: {exp_name}")
            continue

        log_memory_usage(f"Before processing {exp_name}")

        try:
            process_experiment(folder, CONFIG["PROCESSING"]["metrics"], output_data)
            processed_folders.add(exp_name)

            # Save checkpoint after each successful experiment
            with open(checkpoint_file, "w") as f:
                json.dump(list(processed_folders), f)

        except Exception as e:
            logging.error(f"Error processing {exp_name}: {str(e)}")

        log_memory_usage(f"After processing {exp_name}")

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
