from pathlib import Path
import os
import json
import gc
import psutil
import pandas as pd
import yaml  # Import PyYAML for parsing YAML files
import argparse  # For command-line arguments
import Ballpushing_utils
from Ballpushing_utils import utilities


# ==================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES TO MODIFY BEHAVIOR
# ==================================================================
CONFIG = {
    "PATHS": {
        "data_root": Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/"),
        "dataset_dir": Path("/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets"),
        "output_summary_dir": "250403_BallpushingMetrics",
        "output_data_dir": "250403_BallpushingMetrics_Data",
        "excluded_folders": [],
        "config_path": "config.json",
    },
    "PROCESSING": {
        "experiment_filter": "",  # Filter for experiment folders
        "pooled_prefix": "250403_pooled",  # Base name for combined datasets
        "metrics": ["summary"],  # Metrics to process (add/remove as needed)
    },
}


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
    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    directories = yaml_data.get("directories", [])
    return [Path(directory) for directory in directories]


def log_memory_usage(label):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage at {label}: {memory_info.rss / 1024 / 1024:.2f} MB")


# ==================================================================
# MAIN PROCESSING SCRIPT
# ==================================================================
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dataset Builder Script")
    parser.add_argument("--yaml", type=str, help="Path to the YAML file specifying directories", default=None)
    args = parser.parse_args()

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
            # Iterate over subdirectories in each root directory
            Exp_folders.extend(
                folder
                for folder in root.iterdir()
                if folder.is_dir() and CONFIG["PROCESSING"]["experiment_filter"] in folder.name
            )

    # Exclude folders based on config list
    if CONFIG["PATHS"]["excluded_folders"]:
        Exp_folders = [folder for folder in Exp_folders if folder.name not in CONFIG["PATHS"]["excluded_folders"]]

    print(f"Folders to analyze: {[f.name for f in Exp_folders]}")

    # Create metric subdirectories
    for metric in CONFIG["PROCESSING"]["metrics"]:
        (output_data / metric).mkdir(exist_ok=True)

    # Main processing loop
    checkpoint_file = output_summary / "processing_checkpoint.json"

    # Load checkpoint if exists
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            processed_folders = set(json.load(f))
        print(f"Resuming from checkpoint, {len(processed_folders)} folders already processed")
    else:
        processed_folders = set()

    # Process each folder
    for folder in Exp_folders:
        exp_name = folder.name
        experiment_pkl_path = output_summary / f"{exp_name}.pkl"
        experiment = None  # Initialize experiment to avoid unbound variable issues

        # Skip if already processed
        if exp_name in processed_folders:
            print(f"Skipping already processed experiment: {exp_name}")
            continue

        log_memory_usage(f"Before processing {exp_name}")

        try:
            # Check if the experiment exists
            if experiment_pkl_path.exists():
                print(f"Experiment {exp_name} already exists.")
                # Check if all datasets for this experiment exist
                all_datasets_exist = True
                for metric in CONFIG["PROCESSING"]["metrics"]:
                    dataset_path = output_data / metric / f"{exp_name}_{metric}.feather"
                    if not dataset_path.exists():
                        all_datasets_exist = False
                        break

                if all_datasets_exist:
                    print(f"All datasets for {exp_name} already exist. Skipping.")
                    continue
                else:
                    print(f"Some datasets for {exp_name} are missing. Loading experiment.")
                    experiment = utilities.load_object(experiment_pkl_path)
            else:
                # If the experiment doesn't exist, create it
                print(f"Creating new experiment: {exp_name}")
                experiment = Ballpushing_utils.Experiment(folder)
                utilities.save_object(experiment, experiment_pkl_path)
                print(f"Saved new experiment: {exp_name}")

                # Save config if first experiment
                if not (output_summary / CONFIG["PATHS"]["config_path"]).exists():
                    config_dict = experiment.config if isinstance(experiment.config, dict) else vars(experiment.config)
                    config_json_path = output_summary / CONFIG["PATHS"]["config_path"]
                    with open(config_json_path, "w") as config_file:
                        json.dump(config_dict, config_file, indent=4)
                    print(f"Saved config for {exp_name} to {config_json_path}")

            # Generate and save datasets
            all_metrics_processed = True
            for metric in CONFIG["PROCESSING"]["metrics"]:
                dataset_path = output_data / metric / f"{exp_name}_{metric}.feather"

                if dataset_path.exists():
                    print(f"Dataset {dataset_path} already exists. Skipping.")
                    continue

                try:
                    dataset = Ballpushing_utils.Dataset(experiment, dataset_type=metric)
                    if dataset.data is not None and not dataset.data.empty:
                        dataset.data.to_feather(dataset_path)
                        print(f"Saved {metric} dataset for {exp_name}")
                    else:
                        print(f"Empty {metric} data for {exp_name}")
                except Exception as e:
                    print(f"Error generating {metric} for {exp_name}: {str(e)}")
                    all_metrics_processed = False

            # Mark as processed only if all metrics were processed
            if all_metrics_processed:
                processed_folders.add(exp_name)
                # Save checkpoint after each successful experiment
                with open(checkpoint_file, "w") as f:
                    json.dump(list(processed_folders), f)

        except Exception as e:
            print(f"Error processing {exp_name}: {str(e)}")

        # Explicit cleanup
        del experiment
        gc.collect()
        log_memory_usage(f"After processing {exp_name}")

    # ==================================================================
    # Create pooled datasets
    # ==================================================================
    for metric in CONFIG["PROCESSING"]["metrics"]:
        pooled_path = output_data / metric / f"{CONFIG['PROCESSING']['pooled_prefix']}_{metric}.feather"

        if not pooled_path.exists():
            try:
                metric_files = list((output_data / metric).glob("*.feather"))
                if not metric_files:
                    print(f"No {metric} files found for pooling")
                    continue

                # Use chunking for pooling to avoid loading all files at once
                chunk_size = 5  # Adjust based on file sizes
                total_chunks = (len(metric_files) + chunk_size - 1) // chunk_size

                first_chunk = True
                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, len(metric_files))
                    chunk_files = metric_files[start_idx:end_idx]

                    print(f"Processing chunk {chunk_idx+1}/{total_chunks} for {metric}")
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
                    log_memory_usage(f"After pooling chunk {chunk_idx+1} for {metric}")

                print(f"Created pooled {metric} dataset: {pooled_path.name}")
            except Exception as e:
                print(f"Error pooling {metric}: {str(e)}")
        else:
            print(f"Pooled {metric} dataset already exists")

    print("Processing complete!")
