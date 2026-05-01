from pathlib import Path
import os
import sys
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
import ballpushing_utils
from ballpushing_utils import utilities, config

# ==================================================================
# INTRODUCTION AND EXAMPLE USAGE
# ==================================================================

# This script processes datasets for ball-pushing experiments, generating
# datasets for experiments.
#
# PERFORMANCE OPTIMIZATIONS:
# - Conditional cache clearing: Only clears caches when memory usage exceeds threshold
# - Batch processing: For flies mode, processes multiple flies before clearing caches
# - Memory monitoring: Tracks memory usage and provides detailed logging
# - Configurable thresholds: Memory threshold and batch size can be adjusted

# Available datasets (full list also exposed in ELIGIBLE_DATASETS below):
# - coordinates           : Full coordinates of ball and fly positions over time
# - fly_positions         : Raw tracking positions for all keypoints (fly, ball, skeleton)
# - event_metrics         : Individual event metrics for each fly
# - summary               : Summary of all events for each fly
# - standardized_contacts : Contact data reworked to be time-standardized and fly-centric
# - transposed            : Standardized contacts transposed to be one row per event
# - contact_data          : Skeleton data associated with individual events
# - F1_coordinates        : F1-only — coordinates re-baselined to corridor-exit time
# - F1_checkpoints        : F1-only — distance-based checkpoints with adjusted times
#
# Experiment type is auto-detected from the path (F1_Tracks → F1,
# MagnetBlock → MagnetBlock, MultiMazeRecorder → TNT). Use
# ``--experiment-type`` to override for experiments still in the
# default catch-all directory. Asking for an F1-only dataset on a
# non-F1 experiment is a no-op with a warning, never silent.


# Example usage:
# Build a single dataset for a YAML of experiment folders (auto type detection):
# python dataset_builder.py --yaml config.yaml --datasets summary
#
# Multiple datasets in one pass:
# python dataset_builder.py --yaml config.yaml --datasets summary coordinates F1_coordinates
#
# Override experiment type (e.g. a TNT screen still living in the
# MultiMazeRecorder catch-all that the rig hasn't moved yet):
# python dataset_builder.py --yaml config.yaml --datasets summary --experiment-type TNT
#
# Process individual fly directories from a combined YAML file:
# python dataset_builder.py --mode flies --yaml combined_flies.yaml --datasets summary
#
# Re-run the full pipeline against a Dataverse-layout subtree
# (<root>/<condition>/<date>/arenaN/corridorM/{ball,fly,skeleton}.h5):
# python dataset_builder.py --dataverse-root /data/Screen \
#       --experiment-type TNT --datasets summary
# (the condition folder name becomes the value of arena_metadata[Genotype]
#  for TNT, [Magnet] for MagnetBlock, [Pretraining] for F1; override with
#  --condition-field if your subtree uses a different schema.)
#
# Tune cache management (batch size + memory threshold):
# python dataset_builder.py --mode flies --yaml flies.yaml --datasets summary \
#       --batch-size 10 --memory-threshold 4096
#
# Verify existing datasets and create missing pooled datasets:
# python dataset_builder.py --verify /path/to/existing/dataset/directory
#
# Verify with completeness checking against YAML:
# python dataset_builder.py --verify /path/to/existing/dataset/directory \
#       --yaml folders.yaml --mode experiment --datasets summary

# ==================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES TO MODIFY BEHAVIOR
# ==================================================================
# Data root is where to find the raw videos and tracking data
# dataset_dir is where to save the processed datasets
# excluded_folders is a list of folders from data root to exclude from processing
# config_path is the path to the configuration file for the experiment. It just sets the name of the json file which will be saved along with the experiment files.
# experiment_filter can be used to select the experiments to process based on a substring in the folder name (e.g. TNT_Fine)
# Datasets are now selected via the mandatory ``--datasets`` CLI flag instead
# of a hardcoded list inside CONFIG; experiment type comes from path detection
# (or ``--experiment-type`` override). See ELIGIBLE_DATASETS below.

CONFIG = {
    "PATHS": {
        "data_root": [Path("/mnt/upramdya_data/MD/F1_Tracks/Videos")],
        "dataset_dir": Path("/mnt/upramdya_data/MD/F1_Tracks/Datasets"),
        "excluded_folders": [],
        "output_summary_dir": None,  # "250419_transposed_control_folders"  # Optional output directory for summary files, should be a Path object
        "config_path": "config.json",
    },
    "PROCESSING": {
        "experiment_filter": "",  # Filter for a specific experiment folder to test
        "memory_threshold_mb": 4096,  # Memory threshold in MB for conditional cache clearing
    },
}


# ==================================================================
# EXPERIMENT TYPE & DATASET ELIGIBILITY
# ==================================================================
# The recording rig writes each experiment to a top-level directory
# named after the paradigm. Path-based detection looks for one of these
# segments anywhere in the experiment folder's parts; the first match
# wins. ``--experiment-type`` overrides this entirely (useful for
# experiments still living in the catch-all MultiMazeRecorder tree
# that haven't been moved to a dedicated directory yet).
EXPERIMENT_TYPE_FROM_PATH = {
    "F1_Tracks": "F1",
    "MagnetBlock": "MagnetBlock",
    "MultiMazeRecorder": "TNT",  # default catch-all rig output
}

# Canonical dataset output directory for each experiment type.
# Used when CONFIG["PATHS"]["dataset_dir"] is still set to the F1 default
# (i.e. the user hasn't overridden it) so TNT/MagnetBlock/Learning runs
# don't end up inside F1_Tracks/Datasets.
DATASET_DIR_FROM_TYPE = {
    "F1": Path("/mnt/upramdya_data/MD/F1_Tracks/Datasets"),
    "MagnetBlock": Path("/mnt/upramdya_data/MD/MagnetBlock/Datasets"),
    "TNT": Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets"),
    "Learning": Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets"),
    "TNT_DDC": Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets"),
}

# Fallback output root: repo_root/outputs/ — always present, gitignored.
# Used when neither --output-dir is given nor the NAS is reachable.
_REPO_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def _resolve_dataset_dir(experiment_type: str | None, cli_output_dir: str | None) -> Path:
    """Return the base directory where output feathers should be written.

    Priority:
    1. ``--output-dir`` (explicit CLI override)
    2. ``DATASET_DIR_FROM_TYPE`` — only if the NAS parent is mounted
    3. ``<repo_root>/outputs/`` — always available, gitignored fallback
    """
    if cli_output_dir:
        return Path(cli_output_dir).expanduser()
    canonical = DATASET_DIR_FROM_TYPE.get(experiment_type) if experiment_type else None
    if canonical is not None and canonical.parent.exists():
        return canonical
    logging.info(
        f"NAS not reachable (or experiment type unknown); " f"writing output to repo fallback: {_REPO_OUTPUTS_DIR}"
    )
    _REPO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return _REPO_OUTPUTS_DIR


# Which dataset types each experiment type can produce. Asking for an
# ineligible dataset → loud error during compatibility filtering, never
# a silently-empty feather. Adding a new dataset means: extend the
# Dataset class in ballpushing_utils/dataset.py AND add it here.
ELIGIBLE_DATASETS = {
    "F1": {
        "summary",
        "coordinates",
        "F1_coordinates",
        "F1_checkpoints",
        "fly_positions",
        "event_metrics",
        "standardized_contacts",
        "transposed",
        "contact_data",
    },
    "MagnetBlock": {
        "summary",
        "coordinates",
        "fly_positions",
        "event_metrics",
        "standardized_contacts",
        "transposed",
        "contact_data",
    },
    "TNT": {
        "summary",
        "coordinates",
        "fly_positions",
        "event_metrics",
        "standardized_contacts",
        "transposed",
        "contact_data",
    },
    "Learning": {
        "summary",
        "coordinates",
        "fly_positions",
        "event_metrics",
        "standardized_contacts",
        "transposed",
        "contact_data",
    },
}

# Flat list of every dataset name we know how to build. Used to
# constrain ``--datasets`` choices so typos fail at parse time.
ALL_DATASET_TYPES = sorted({d for ds in ELIGIBLE_DATASETS.values() for d in ds})


def detect_experiment_type(folder):
    """Infer experiment type from a folder's path.

    Walks the folder's parts and returns the first match in
    ``EXPERIMENT_TYPE_FROM_PATH``. Returns ``None`` if no segment
    matches, so the caller can decide whether to raise or fall back to
    a CLI override.
    """
    folder = Path(folder)
    for part in folder.parts:
        if part in EXPERIMENT_TYPE_FROM_PATH:
            return EXPERIMENT_TYPE_FROM_PATH[part]
    return None


def resolve_experiment_type(folder, override=None):
    """Resolve the experiment type for ``folder`` (override > path).

    Raises ``ValueError`` if neither the override nor path detection
    yields a known type — never silently default, since the wrong
    experiment_type changes time-range cropping and metric selection
    inside the package.
    """
    if override is not None:
        return override
    detected = detect_experiment_type(folder)
    if detected is None:
        raise ValueError(
            f"Could not auto-detect experiment type from path: {folder}\n"
            f"  No path segment matched any of {sorted(EXPERIMENT_TYPE_FROM_PATH)}.\n"
            f"  Pass --experiment-type to set it explicitly. "
            f"Valid choices: {sorted(ELIGIBLE_DATASETS)}."
        )
    return detected


def filter_datasets_for_type(requested, experiment_type, label=""):
    """Drop datasets that the resolved experiment type can't produce.

    Logs a clear warning for each skipped dataset (e.g. asking for
    ``F1_coordinates`` on a MagnetBlock experiment) and returns the
    eligible subset, preserving the caller's order.
    """
    eligible = ELIGIBLE_DATASETS.get(experiment_type, set())
    if not eligible:
        raise ValueError(
            f"Unknown experiment type {experiment_type!r}. " f"Valid choices: {sorted(ELIGIBLE_DATASETS)}."
        )
    valid = [d for d in requested if d in eligible]
    skipped = [d for d in requested if d not in eligible]
    if skipped:
        prefix = f"[{label}] " if label else ""
        logging.warning(
            f"{prefix}Datasets {skipped} not compatible with experiment type "
            f"{experiment_type!r} — skipping. Eligible for {experiment_type}: "
            f"{sorted(eligible)}."
        )
    return valid


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


def get_memory_usage_mb():
    """
    Get current memory usage in MB.

    Returns
    -------
    float
        Memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024


def should_clear_caches(memory_threshold_mb=None, baseline_memory_mb=None):
    """
    Check if cache clearing is needed based on memory growth from baseline.

    Parameters
    ----------
    memory_threshold_mb : int, optional
        Memory growth threshold in MB above which caches should be cleared.
        If None, uses CONFIG value.
    baseline_memory_mb : float, optional
        Baseline memory usage in MB. If provided, checks memory growth from baseline.
        If None, uses absolute memory threshold (old behavior).

    Returns
    -------
    bool
        True if caches should be cleared, False otherwise.
    """
    if memory_threshold_mb is None:
        memory_threshold_mb = CONFIG["PROCESSING"]["memory_threshold_mb"]

    current_memory = get_memory_usage_mb()

    if baseline_memory_mb is not None:
        # Check memory growth from baseline
        memory_growth = current_memory - baseline_memory_mb
        return memory_growth > memory_threshold_mb
    else:
        # Old behavior: check absolute memory
        return current_memory > memory_threshold_mb


def clear_experiment_caches_batch(experiment, force=False, baseline_memory_mb=None):
    """
    Clear caches for all flies in an experiment efficiently.

    Parameters
    ----------
    experiment : Experiment
        The experiment object containing flies to clear caches for.
    force : bool
        If True, clear caches regardless of memory usage. If False, only clear if needed.
    baseline_memory_mb : float, optional
        Baseline memory to check growth against. If provided, clears when growth exceeds threshold.

    Returns
    -------
    bool
        True if caches were cleared, False if clearing was skipped.
    """
    if not force and not should_clear_caches(baseline_memory_mb=baseline_memory_mb):
        current_memory = get_memory_usage_mb()
        if baseline_memory_mb is not None:
            memory_growth = current_memory - baseline_memory_mb
            logging.debug(f"Skipping cache clearing - memory growth {memory_growth:.1f}MB below threshold")
        else:
            logging.debug("Skipping cache clearing - memory usage below threshold")
        return False

    memory_before = get_memory_usage_mb()
    start_time = time.time()

    if experiment is not None and hasattr(experiment, "flies"):
        # Clear caches for all flies at once
        for fly in experiment.flies:
            if hasattr(fly, "clear_caches"):
                fly.clear_caches()

        # Force garbage collection once for all flies
        gc.collect()

        memory_after = get_memory_usage_mb()
        elapsed_time = time.time() - start_time
        memory_freed = memory_before - memory_after

        logging.info(
            f"Batch cleared caches for {len(experiment.flies)} flies in {elapsed_time:.2f}s. "
            f"Memory freed: {memory_freed:.1f}MB ({memory_before:.1f}MB → {memory_after:.1f}MB)"
        )

    return True


def clear_fly_caches_conditional(fly, force=False):
    """
    Conditionally clear caches for a single fly based on memory usage.

    Parameters
    ----------
    fly : Fly
        The fly object to clear caches for.
    force : bool
        If True, clear caches regardless of memory usage. If False, only clear if needed.

    Returns
    -------
    bool
        True if caches were cleared, False if clearing was skipped.
    """
    if not force and not should_clear_caches():
        logging.debug(f"Skipping cache clearing for {fly.metadata.name} - memory usage below threshold")
        return False

    memory_before = get_memory_usage_mb()
    start_time = time.time()

    if hasattr(fly, "clear_caches"):
        fly.clear_caches()

    # Force garbage collection
    gc.collect()

    memory_after = get_memory_usage_mb()
    elapsed_time = time.time() - start_time
    memory_freed = memory_before - memory_after

    logging.debug(
        f"Cleared caches for {fly.metadata.name} in {elapsed_time:.2f}s. "
        f"Memory freed: {memory_freed:.1f}MB ({memory_before:.1f}MB → {memory_after:.1f}MB)"
    )

    return True


def estimate_cache_clearing_time(num_flies, individual_time_ms=100, batch_time_ms=500):
    """
    Estimate time savings from batch cache clearing.

    Parameters
    ----------
    num_flies : int
        Number of flies to process.
    individual_time_ms : float
        Average time in milliseconds to clear cache for one fly individually.
    batch_time_ms : float
        Average time in milliseconds to clear cache for a batch of flies.

    Returns
    -------
    tuple
        (individual_total_ms, batch_total_ms, savings_ms, savings_percent)
    """
    individual_total = num_flies * individual_time_ms
    num_batches = (num_flies + 4) // 5  # Assuming batch size of 5
    batch_total = num_batches * batch_time_ms
    savings = individual_total - batch_total
    savings_percent = (savings / individual_total) * 100 if individual_total > 0 else 0

    return individual_total, batch_total, savings, savings_percent


def process_flies_batch(fly_dirs, metrics, output_data, batch_size=5, experiment_type_override=None):
    """
    Process multiple fly directories in batches with efficient cache management.

    Parameters
    ----------
    fly_dirs : list
        List of fly directory paths.
    metrics : list
        List of dataset types requested by the user (will be filtered
        per-fly against the eligibility table for the resolved
        experiment type).
    output_data : Path
        Output directory for datasets.
    batch_size : int
        Number of flies to process before clearing caches.
    experiment_type_override : str, optional
        If set, force this experiment type for every fly instead of
        detecting it from the path.

    Returns
    -------
    list
        List of successfully processed fly names.
    """
    processed_flies = []
    batch_flies = []

    for i, fly_dir in enumerate(fly_dirs):
        try:
            # Extract fly name
            experiment_name = fly_dir.parent.parent.name
            arena_name = fly_dir.parent.name
            corridor_name = fly_dir.name
            fly_name = f"{experiment_name}_{arena_name}_{corridor_name}"

            logging.info(f"Processing fly directory: {fly_dir} ({i+1}/{len(fly_dirs)})")

            # Resolve experiment type from path (or CLI override) and
            # filter the requested datasets down to ones this paradigm
            # can actually produce.
            experiment_type = resolve_experiment_type(fly_dir, override=experiment_type_override)
            fly_metrics = filter_datasets_for_type(metrics, experiment_type, label=fly_name)
            if not fly_metrics:
                logging.warning(f"No compatible datasets for {fly_name} (type={experiment_type}). Skipping.")
                continue

            # Check if all datasets already exist
            all_datasets_exist = True
            for metric in fly_metrics:
                dataset_path = output_data / metric / f"{fly_name}_{metric}.feather"
                if not dataset_path.exists():
                    all_datasets_exist = False
                    break

            if all_datasets_exist:
                logging.info(f"All datasets for {fly_name} already exist. Skipping.")
                processed_flies.append(fly_name)
                continue

            # Create and process fly. ``custom_config`` overrides
            # ``experiment_type`` on the fly's Config (and the shared
            # Experiment.config it points at), so paradigm-specific
            # branches (F1 exit time, MagnetBlock 1h cutoff, …) take
            # the right path without anyone editing config.py.
            custom_config = {"experiment_type": experiment_type}
            fly = ballpushing_utils.Fly(fly_dir, as_individual=True, custom_config=custom_config)
            batch_flies.append((fly, fly_name))

            # Process each metric for this specific fly
            for metric in fly_metrics:
                dataset_path = output_data / metric / f"{fly_name}_{metric}.feather"

                if dataset_path.exists():
                    logging.info(f"Dataset {dataset_path} already exists. Skipping.")
                    continue

                # Create dataset for this specific fly
                dataset = ballpushing_utils.Dataset(fly, dataset_type=metric)

                if dataset.data is not None and not dataset.data.empty:
                    dataset_path.parent.mkdir(parents=True, exist_ok=True)
                    dataset.data.to_feather(dataset_path)
                    logging.info(f"Saved {metric} dataset for {fly_name} to {dataset_path} ({len(dataset.data)} rows)")
                else:
                    logging.warning(f"No data available for {fly_name} with metric {metric}")

            processed_flies.append(fly_name)

            # Clear caches in batches
            if len(batch_flies) >= batch_size or i == len(fly_dirs) - 1:
                memory_before = get_memory_usage_mb()
                start_time = time.time()

                # Clear all flies in the batch
                for fly, name in batch_flies:
                    if hasattr(fly, "clear_caches"):
                        fly.clear_caches()
                    del fly

                # Force garbage collection once for the whole batch
                gc.collect()

                memory_after = get_memory_usage_mb()
                elapsed_time = time.time() - start_time
                memory_freed = memory_before - memory_after

                logging.info(
                    f"Batch cleared caches for {len(batch_flies)} flies in {elapsed_time:.2f}s. "
                    f"Memory freed: {memory_freed:.1f}MB ({memory_before:.1f}MB → {memory_after:.1f}MB)"
                )

                batch_flies = []  # Reset batch

        except Exception as e:
            logging.error(f"Error processing fly directory {fly_dir}: {str(e)}")
            import traceback

            logging.error(f"Traceback: {traceback.format_exc()}")

    return processed_flies


def process_experiment(folder, metrics, output_data, baseline_memory_mb=None, experiment_type_override=None):
    """
    Process an experiment folder in experiment mode.

    Parameters
    ----------
    folder : Path
        Path to the experiment folder.
    metrics : list
        List of dataset types requested by the user (will be filtered
        against the eligibility table for the resolved experiment type
        before any work happens).
    output_data : Path
        Output directory for datasets.
    baseline_memory_mb : float, optional
        Baseline memory usage for delta-based cache clearing.
    experiment_type_override : str, optional
        If set, force this experiment type instead of detecting it
        from ``folder``'s path.

    Returns
    -------
    float
        Updated baseline memory after cache clearing (if cleared), otherwise None.
    """
    logging.info(f"Processing experiment folder: {folder.name}")

    try:
        # Resolve experiment type once for the whole folder; this
        # determines which datasets we'll attempt to build AND seeds
        # the per-fly Config via custom_config below.
        experiment_type = resolve_experiment_type(folder, override=experiment_type_override)
        logging.info(f"  Resolved experiment type: {experiment_type}")
        exp_metrics = filter_datasets_for_type(metrics, experiment_type, label=folder.name)
        if not exp_metrics:
            logging.warning(f"No compatible datasets for {folder.name} (type={experiment_type}). Skipping.")
            return None

        # Check if all datasets already exist
        all_datasets_exist = True
        for metric in exp_metrics:
            dataset_path = output_data / metric / f"{folder.name}_{metric}.feather"
            if not dataset_path.exists():
                all_datasets_exist = False
                break

        if all_datasets_exist:
            logging.info(f"All datasets for {folder.name} already exist. Skipping.")
            return

        # Check if the Experiment object exists or needs to be created
        experiment = None
        for metric in exp_metrics:
            dataset_path = output_data / metric / f"{folder.name}_{metric}.feather"

            if dataset_path.exists():
                logging.info(f"Dataset {dataset_path} already exists. Skipping.")
                continue

            # Create the Experiment object only if needed.
            # ``custom_config`` propagates experiment_type into every
            # fly's shared Config — the package's paradigm-specific
            # branches read from ``fly.config.experiment_type``.
            if experiment is None:
                experiment = ballpushing_utils.Experiment(
                    folder,
                    custom_config={"experiment_type": experiment_type},
                )

            # Process the dataset
            dataset = ballpushing_utils.Dataset(experiment, dataset_type=metric)
            if dataset.data is not None and not dataset.data.empty:
                dataset_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
                dataset.data.to_feather(dataset_path)
                logging.info(f"Saved {metric} dataset for {folder.name} to {dataset_path}")
            else:
                logging.warning(f"No data available for {folder.name} with metric {metric}")

        # Clear caches after processing all metrics for this experiment (batch clearing)
        # Use conditional clearing based on memory growth from baseline
        cleared = clear_experiment_caches_batch(experiment, force=False, baseline_memory_mb=baseline_memory_mb)
        if cleared:
            logging.info(f"Cleared caches for {folder.name} - memory growth exceeded threshold")
            # Reset baseline after clearing
            new_baseline = get_memory_usage_mb()
            logging.info(f"New baseline memory: {new_baseline:.1f}MB")
            return new_baseline
        elif experiment is not None:
            logging.debug(f"Skipped cache clearing for {folder.name} - memory growth acceptable")

        return None

    except Exception as e:
        import traceback

        logging.error(f"Error processing experiment {folder.name}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")


def process_fly_directory(fly_dir, metrics, output_data, force_cache_clear=False, experiment_type_override=None):
    """
    Process an individual fly directory in flies mode.

    Parameters
    ----------
    fly_dir : Path
        Path to the fly directory (e.g., /path/to/experiment/arena1/corridor1).
    metrics : list
        List of dataset types requested by the user (will be filtered
        against the eligibility table for the resolved experiment type).
    output_data : Path
        Output directory for datasets.
    force_cache_clear : bool
        If True, always clear caches regardless of memory usage.
    experiment_type_override : str, optional
        If set, force this experiment type instead of detecting it
        from ``fly_dir``'s path.
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

        # Resolve experiment type and prune incompatible datasets.
        experiment_type = resolve_experiment_type(fly_dir, override=experiment_type_override)
        fly_metrics = filter_datasets_for_type(metrics, experiment_type, label=fly_name)
        if not fly_metrics:
            logging.warning(f"No compatible datasets for {fly_name} (type={experiment_type}). Skipping.")
            return

        # Check if all datasets already exist
        all_datasets_exist = True
        for metric in fly_metrics:
            dataset_path = output_data / metric / f"{fly_name}_{metric}.feather"
            if not dataset_path.exists():
                all_datasets_exist = False
                break

        if all_datasets_exist:
            logging.info(f"All datasets for {fly_name} already exist. Skipping.")
            return

        # Create individual Fly object with as_individual=True to avoid
        # loading full experiment. ``custom_config`` overrides
        # experiment_type on the Config used by this Fly.
        fly = ballpushing_utils.Fly(
            fly_dir,
            as_individual=True,
            custom_config={"experiment_type": experiment_type},
        )

        # Process each metric for this specific fly
        for metric in fly_metrics:
            dataset_path = output_data / metric / f"{fly_name}_{metric}.feather"

            if dataset_path.exists():
                logging.info(f"Dataset {dataset_path} already exists. Skipping.")
                continue

            # Create dataset for this specific fly (pass the fly object directly)
            dataset = ballpushing_utils.Dataset(fly, dataset_type=metric)

            if dataset.data is not None and not dataset.data.empty:
                # No need to filter since we're working with individual fly data
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset.data.to_feather(dataset_path)
                logging.info(f"Saved {metric} dataset for {fly_name} to {dataset_path} ({len(dataset.data)} rows)")
            else:
                logging.warning(f"No data available for {fly_name} with metric {metric}")

        # Conditionally clear caches based on memory usage or force flag
        cleared = clear_fly_caches_conditional(fly, force=force_cache_clear)
        if cleared:
            logging.debug(f"Cleared caches for fly: {fly.metadata.name}")

        # Clear the fly object
        del fly

    except Exception as e:
        logging.error(f"Error processing fly directory {fly_dir}: {str(e)}")
        import traceback

        logging.error(f"Traceback: {traceback.format_exc()}")


def process_dataverse_fly(
    dataverse_fly,
    metrics,
    output_data,
    experiment_type,
    fields,
    force_cache_clear=False,
):
    """Build per-fly feathers from one Dataverse-layout corridor folder.

    Mirrors :func:`process_fly_directory` but for the Dataverse archive,
    where there is no parent ``Metadata.json``: per-arena metadata is
    synthesised from a pre-parsed ``(experiment_type, fields)`` recipe
    (typically the output of
    :func:`ballpushing_utils.dataverse.parse_archive_name` applied to
    the archive folder name).

    Parameters
    ----------
    dataverse_fly : ballpushing_utils.DataverseFly
        Record describing one corridor under
        ``<root>/<archive>/<date>/arenaN/corridorM``.
    metrics : list[str]
        Requested dataset types (filtered against the experiment type's
        eligibility table before any work happens).
    output_data : Path
        Output root containing one subdirectory per metric.
    experiment_type : str
        Resolved experiment type — drives metric eligibility and the
        paradigm-specific branches inside the package (e.g. F1 exit-time
        adjustment, MagnetBlock 3600 s cutoff).
    fields : dict
        Flat ``{column: value}`` mapping baked into the synthetic
        ``arena_metadata``. Multi-column paradigms (F1 → Genotype +
        Pretraining + F1_condition; TNT-olfaction-dark → Genotype +
        Light) use a multi-key dict; single-column paradigms use one.
    force_cache_clear : bool
        Pass-through to :func:`clear_fly_caches_conditional`.
    """
    fly_dir = dataverse_fly.directory
    fly_name = f"{dataverse_fly.date_folder.name}_{dataverse_fly.arena}_{dataverse_fly.corridor}"
    logging.info(f"Processing Dataverse fly: condition={dataverse_fly.condition!r} -> {fly_name}")

    try:
        fly_metrics = filter_datasets_for_type(metrics, experiment_type, label=fly_name)
        if not fly_metrics:
            logging.warning(f"No compatible datasets for {fly_name} (type={experiment_type}). Skipping.")
            return

        all_datasets_exist = all(
            (output_data / metric / f"{fly_name}_{metric}.feather").exists() for metric in fly_metrics
        )
        if all_datasets_exist:
            logging.info(f"All datasets for {fly_name} already exist. Skipping.")
            return

        fly = ballpushing_utils.Fly(
            fly_dir,
            as_individual=True,
            custom_config={"experiment_type": experiment_type},
            dataverse_condition={"fields": fields},
        )

        for metric in fly_metrics:
            dataset_path = output_data / metric / f"{fly_name}_{metric}.feather"
            if dataset_path.exists():
                logging.info(f"Dataset {dataset_path} already exists. Skipping.")
                continue

            dataset = ballpushing_utils.Dataset(fly, dataset_type=metric)
            if dataset.data is not None and not dataset.data.empty:
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                dataset.data.to_feather(dataset_path)
                logging.info(f"Saved {metric} dataset for {fly_name} to {dataset_path} " f"({len(dataset.data)} rows)")
            else:
                logging.warning(f"No data available for {fly_name} with metric {metric}")

        clear_fly_caches_conditional(fly, force=force_cache_clear)
        del fly

    except Exception as e:
        logging.error(f"Error processing Dataverse fly {fly_dir}: {str(e)}")
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


def process_complement_mode(yaml_path, complement_path, metrics):
    """
    Process complement mode: identify missing feather files and process only those.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML file containing fly directories or experiment folders.
    complement_path : str or Path
        Path to the existing dataset directory.
    metrics : list
        List of metrics to process.

    Returns
    -------
    tuple
        (processing_items, output_data, is_experiment_mode)
        - processing_items: items to process based on detected mode
        - output_data: path to the complement data directory
        - is_experiment_mode: boolean indicating if YAML contains experiment folders
    """
    complement_path = Path(complement_path)
    logging.info(f"Running in complement mode")
    logging.info(f"  YAML file: {yaml_path}")
    logging.info(f"  Complement path: {complement_path}")

    if not complement_path.exists():
        raise ValueError(f"Complement path does not exist: {complement_path}")

    # Load directories from YAML (could be experiment folders or fly directories)
    yaml_dirs = load_yaml_config(yaml_path)
    if not yaml_dirs:
        raise ValueError("No valid directories found in YAML file.")

    logging.info(f"Loaded {len(yaml_dirs)} directories from YAML")

    # Determine if we have experiment folders or fly directories
    is_experiment_mode = any((path / "Metadata.json").exists() for path in yaml_dirs)

    if is_experiment_mode:
        # Process as experiment folders
        # Identify missing files for each experiment
        experiments_to_process = []
        for exp_path in yaml_dirs:
            if (exp_path / "Metadata.json").exists():
                exp_name = exp_path.name
                missing_metrics = []

                # Check which metrics are missing for this experiment
                for metric in metrics:
                    feather_path = complement_path / metric / f"{exp_name}_{metric}.feather"
                    if not feather_path.exists():
                        missing_metrics.append(metric)

                if missing_metrics:
                    experiments_to_process.append(exp_path)
                    logging.info(f"  {exp_name}: missing {', '.join(missing_metrics)}")

        if not experiments_to_process:
            logging.info(f"All experiment files already exist in complement path. No processing needed.")
            return [], complement_path, True

        logging.info(f"Found missing files in {len(experiments_to_process)} experiments")
        return experiments_to_process, complement_path, True
    else:
        # Process as fly directories
        fly_dirs = yaml_dirs

        # Identify missing feather files
        flies_to_process = []
        missing_count = 0

        for fly_dir in fly_dirs:
            # Extract names: experiment_arena_corridor
            # fly_dir structure: /path/to/experiment/arena/corridor
            experiment_name = fly_dir.parent.parent.name
            arena_name = fly_dir.parent.name
            corridor_name = fly_dir.name
            fly_name = f"{experiment_name}_{arena_name}_{corridor_name}"

            missing_metrics = []

            # Check each metric for missing files
            for metric in metrics:
                feather_path = complement_path / metric / f"{fly_name}_{metric}.feather"
                if not feather_path.exists():
                    missing_metrics.append(metric)
                    missing_count += 1

            # Add to processing list if any metrics are missing
            if missing_metrics:
                flies_to_process.append(fly_dir)
                logging.info(f"  {fly_name}: missing {', '.join(missing_metrics)}")

        if not flies_to_process:
            logging.info(f"All files already exist in complement path. No processing needed.")
            return [], complement_path, False

        logging.info(f"Found {missing_count} missing feather files across {len(flies_to_process)} flies")
        logging.info(f"Will process {len(flies_to_process)} fly directories")

        return flies_to_process, complement_path, False


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
        "--complement",
        type=str,
        help="Complement mode: path to existing dataset directory; when used with --yaml, only builds missing feather files and redoes pooling",
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
    parser.add_argument(
        "--memory-threshold",
        type=int,
        default=2048,
        help="Memory threshold in MB for conditional cache clearing (default: 2048MB)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of flies to process before clearing caches in batch mode (default: 5)",
    )
    parser.add_argument(
        "--force-cache-clear",
        action="store_true",
        help="Always clear caches regardless of memory usage (original behavior)",
    )
    parser.add_argument(
        "--skip-pooling",
        nargs="+",
        default=[],
        metavar="DATASET",
        help="Datasets to exclude from pooling (e.g. --skip-pooling coordinates). Useful for heavy datasets that would crash RAM.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=ALL_DATASET_TYPES,
        metavar="NAME",
        help=(
            "Dataset types to build (one or more). Each is validated "
            "against the resolved experiment type; incompatible "
            "combinations (e.g. F1_coordinates on a non-F1 experiment) "
            "are skipped with a warning, never silently produced. "
            "Choices: " + ", ".join(ALL_DATASET_TYPES)
        ),
    )
    parser.add_argument(
        "--experiment-type",
        choices=sorted(ELIGIBLE_DATASETS),
        default=None,
        help=(
            "Override path-based experiment type detection. By default "
            "the type is read from the experiment's path (F1_Tracks → "
            "F1, MagnetBlock → MagnetBlock, MultiMazeRecorder → TNT). "
            "Use this for experiments saved outside their canonical "
            "directory (typically still in the MultiMazeRecorder catch-all)."
        ),
    )
    parser.add_argument(
        "--dataverse-root",
        type=str,
        default=None,
        help=(
            "Build feathers from a Dataverse-layout subtree "
            "(<root>/<condition>/<date>/arenaN/corridorM/{ball,fly,skeleton}.h5). "
            "Requires --experiment-type so the synthetic metadata uses "
            "the right paradigm. The condition folder name becomes the "
            "value of the column selected by --condition-field (default: "
            "depends on experiment_type). Mutually exclusive with "
            "--yaml/--mode."
        ),
    )
    parser.add_argument(
        "--condition-field",
        type=str,
        default=None,
        help=(
            "Arena-metadata column the Dataverse condition folder name "
            "maps to (e.g. Genotype, Magnet, Pretraining). Defaults to "
            "the per-experiment-type value from "
            "ballpushing_utils.dataverse.DEFAULT_CONDITION_FIELD."
        ),
    )
    parser.add_argument(
        "--filter-dates",
        type=str,
        nargs="+",
        default=None,
        metavar="YYMMDD",
        help=(
            "Restrict Dataverse processing to these date folders "
            "(e.g. --filter-dates 251008 251009). Useful for quick tests "
            "without processing the full archive."
        ),
    )
    parser.add_argument(
        "--keep-individual",
        action="store_true",
        default=False,
        help=(
            "Keep the per-fly intermediate feathers after pooling. "
            "By default they are removed once a pooled_*.feather has "
            "been successfully written, leaving a single clean output "
            "file per dataset type. Has no effect when --skip-pooling "
            "is used for a given metric."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where output feathers are written. "
            "Overrides the default location derived from "
            'CONFIG["PATHS"]["dataset_dir"] / DATASET_DIR_FROM_TYPE. '
            "Useful for Dataverse users who don't have the lab NAS "
            "mounted: pass any local path, e.g. ~/Downloads/datasets. "
            "The directory is created if it does not exist."
        ),
    )
    args = parser.parse_args()

    # Update CONFIG with command line arguments
    CONFIG["PROCESSING"]["memory_threshold_mb"] = args.memory_threshold

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Log dataset / experiment-type resolution up front so the run's
    # intent is captured in the log even if it crashes mid-fly.
    logging.info(f"Datasets to build: {args.datasets}")
    if args.experiment_type:
        logging.info(f"Experiment type: forced to {args.experiment_type!r} (CLI override)")
    else:
        logging.info("Experiment type: auto-detect from path " f"({EXPERIMENT_TYPE_FROM_PATH})")

    # Log optimization settings
    logging.info(f"Cache management configuration:")
    logging.info(f"  - Memory threshold: {args.memory_threshold}MB")
    logging.info(f"  - Batch size (flies mode): {args.batch_size}")
    logging.info(f"  - Force cache clearing: {args.force_cache_clear}")
    if args.mode == "flies" and args.batch_size > 1 and not args.force_cache_clear:
        logging.info(f"  - Will use batch processing for optimal cache management")
    elif args.force_cache_clear:
        logging.info(f"  - Will use original cache clearing behavior")
    else:
        logging.info(f"  - Will use conditional cache clearing based on memory usage")

    # Handle complement mode
    skip_folder_discovery = False
    if args.complement:
        if not args.yaml:
            raise ValueError("--complement requires --yaml to be specified")
        logging.info("Running in complement mode")
        processing_items, output_data, is_experiment_mode = process_complement_mode(
            args.yaml, args.complement, args.datasets
        )
        if not processing_items:
            logging.info("No missing files to process. Exiting.")
            end_time = time.time()
            runtime = end_time - start_time
            logging.info(f"Complement mode completed in {runtime:.2f} seconds")
            exit(0)
        # Set mode based on detected YAML content
        if is_experiment_mode:
            logging.info("Detected experiment folders in YAML, using experiment mode")
            args.mode = "experiment"
        else:
            logging.info("Detected fly directories in YAML, using flies mode")
            args.mode = "flies"
        # Create metric subdirectories
        for metric in args.datasets:
            (output_data / metric).mkdir(exist_ok=True)
        # Skip the normal folder discovery and go straight to processing
        skip_folder_discovery = True

    # Handle verification mode
    if args.verify:
        logging.info("Running in verification mode")
        verify_and_complete_datasets(args.verify, yaml_path=args.yaml, mode=args.mode)
        end_time = time.time()
        runtime = end_time - start_time
        logging.info(f"Verification completed in {runtime:.2f} seconds")
        exit(0)

    # Handle Dataverse mode (self-contained: discovery, processing, pooling).
    if args.dataverse_root:
        from ballpushing_utils.dataverse import (
            CONDITION_TRANSFORMERS,
            DEFAULT_CONDITION_FIELD,
            detect_dataverse_experiment_type,
            expand_condition,
            iter_dataverse_flies,
            parse_archive_name,
        )

        dv_root = Path(args.dataverse_root).expanduser()
        if not dv_root.is_dir():
            from ballpushing_utils.paths import missing_data_message

            logging.error(missing_data_message(dv_root, context="Dataverse root"))
            sys.exit(2)

        # Output dir mirrors the YAML-mode convention but keys off the
        # Dataverse subtree name so two separate runs against different
        # subtrees don't clobber each other.
        if CONFIG["PATHS"]["output_summary_dir"]:
            output_dir_name = CONFIG["PATHS"]["output_summary_dir"]
        else:
            today_date = datetime.now().strftime("%y%m%d_%H")
            output_dir_name = f"{today_date}_{args.datasets[0]}_dataverse_{dv_root.name}"
        _dataset_dir = _resolve_dataset_dir(args.experiment_type, args.output_dir)
        output_data = _dataset_dir / f"{output_dir_name}_Data"
        output_data.mkdir(parents=True, exist_ok=True)
        for metric in args.datasets:
            (output_data / metric).mkdir(exist_ok=True)

        # Persist the run config alongside the feathers, same as the
        # YAML/experiment paths do, so the resulting tree can be shipped
        # back to the Dataverse if needed.
        config_save_path = output_data / CONFIG["PATHS"]["config_path"]
        try:
            config_to_save = {
                "dataset_builder_config": CONFIG,
                "ballpushing_config": current_config.__dict__,
                "processing_info": {
                    "mode": "dataverse",
                    "dataverse_root": str(dv_root),
                    "experiment_type": args.experiment_type,
                    "condition_field": condition_field,
                    "datasets_requested": args.datasets,
                    "timestamp": datetime.now().isoformat(),
                },
            }
            with open(config_save_path, "w") as f:
                json.dump(config_to_save, f, indent=2, default=str)
            logging.info(f"Configuration saved to: {config_save_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")

        dataverse_flies = list(iter_dataverse_flies(dv_root))
        if args.filter_dates:
            filter_set = set(args.filter_dates)
            dataverse_flies = [f for f in dataverse_flies if f.date_folder.name in filter_set]
            logging.info(
                f"--filter-dates applied: keeping {len(dataverse_flies)} flies " f"from dates {sorted(filter_set)}"
            )
        if not dataverse_flies:
            from ballpushing_utils.paths import detect_layout, missing_data_message

            detected = detect_layout(dv_root)
            if detected == "server":
                logging.error(
                    f"{dv_root} looks like a server-style tree (Metadata.json "
                    f"detected) — call dataset_builder.py with --yaml instead "
                    f"of --dataverse-root."
                )
            else:
                logging.error(
                    f"No Dataverse-layout flies found under {dv_root}.\n"
                    f"Expected: <root>/<condition>/<date>/arenaN/corridorM/*ball*.h5\n\n"
                    + missing_data_message(dv_root, context="Dataverse root")
                )
            sys.exit(2)
        logging.info(
            f"Found {len(dataverse_flies)} flies across "
            f"{len(set(f.condition for f in dataverse_flies))} conditions in {dv_root}"
        )

        # Per-archive recipe lookup: parse_archive_name decodes the
        # archive folder name into (experiment_type, fields). CLI flags
        # override on a global basis (--experiment-type forces all
        # archives to a paradigm; --condition-field overrides the
        # column for paradigms without a registered transformer).
        unique_conditions = sorted({f.condition for f in dataverse_flies})
        archive_recipes: dict[str, tuple[str, dict]] = {}
        for cond in unique_conditions:
            parsed_etype, parsed_fields = parse_archive_name(cond)
            etype = args.experiment_type or parsed_etype
            if args.experiment_type and parsed_etype != args.experiment_type:
                logging.warning(
                    f"Archive {cond!r} parses as {parsed_etype} but "
                    f"--experiment-type={args.experiment_type} forces an override."
                )
            # Apply --condition-field override only if the paradigm has
            # no registered multi-column transformer (otherwise the
            # parser already produced the correct multi-key fields).
            fields = dict(parsed_fields)
            if args.condition_field and etype not in CONDITION_TRANSFORMERS:
                # User wants a different column for the primary value.
                # Find the single value in parsed_fields and rebind it.
                if len(parsed_fields) == 1:
                    [(_, v)] = parsed_fields.items()
                    fields = {args.condition_field: v}
            archive_recipes[cond] = (etype, fields)

        # Show the recipe table up front so the user can sanity-check
        # before kicking off a long run.
        logging.info("Archive → recipe map:")
        for cond, (etype, fields) in archive_recipes.items():
            logging.info(f"  {cond!r:55s} → ({etype}, {fields})")

        for dv_fly in tqdm(dataverse_flies, desc="Dataverse flies"):
            etype, fields = archive_recipes[dv_fly.condition]
            process_dataverse_fly(
                dv_fly,
                args.datasets,
                output_data,
                experiment_type=etype,
                fields=fields,
                force_cache_clear=args.force_cache_clear,
            )

        # Pool per-metric feathers (skip whatever the user excluded).
        for metric in args.datasets:
            if metric in args.skip_pooling:
                logging.info(f"Skipping pooling for metric '{metric}'")
                continue
            metric_dir = output_data / metric
            files = [f for f in metric_dir.glob("*.feather") if not f.name.startswith("pooled_")]
            if files:
                create_pooled_dataset(metric_dir, metric, files, force=False)
                if not args.keep_individual:
                    pooled_ok = (metric_dir / f"pooled_{metric}.feather").exists()
                    if pooled_ok:
                        for f in files:
                            f.unlink()
                        logging.info(f"Removed {len(files)} individual feathers for '{metric}' (pooled file kept)")

        end_time = time.time()
        runtime = end_time - start_time
        logging.info(f"Dataverse build completed in {runtime:.2f} seconds")
        exit(0)

    # Determine output directories (skip if in complement mode since we already have them)
    if not skip_folder_discovery:
        if CONFIG["PATHS"]["output_summary_dir"]:
            # Use the output directory if provided
            output_summary = Path(CONFIG["PATHS"]["output_summary_dir"])
            logging.info(f"Using optional output directory: {output_summary}")
        else:
            # Determine the output_summary_dir based on the provided arguments
            today_date = datetime.now().strftime("%y%m%d_%H")
            dataset_type = args.datasets[0]  # Use the first metric as the dataset type

            if args.yaml:
                yaml_stem = Path(args.yaml).stem  # Extract the stem (filename without extension)
                CONFIG["PATHS"]["output_summary_dir"] = f"{today_date}_{dataset_type}_{yaml_stem}"
            elif CONFIG["PROCESSING"]["experiment_filter"]:
                experiment_filter = CONFIG["PROCESSING"]["experiment_filter"]
                CONFIG["PATHS"]["output_summary_dir"] = f"{today_date}_{dataset_type}_{experiment_filter}"
            else:
                from ballpushing_utils.paths import missing_data_message

                logging.error(
                    "Pass one of --yaml (server / on-rig layout), "
                    "--dataverse-root (Dataverse archive layout), or set "
                    "experiment_filter in CONFIG. Processing the entire "
                    "data root in one shot is not allowed.\n\n" + missing_data_message(context="data root")
                )
                sys.exit(2)

        # Automatically set output_data_dir based on output_summary_dir
        CONFIG["PATHS"]["output_data_dir"] = f"{CONFIG['PATHS']['output_summary_dir']}_Data"

        # Use the canonical dataset dir for this experiment type so TNT/
        # MagnetBlock runs don't land inside F1_Tracks/Datasets.
        # --output-dir overrides; NAS fallback → repo outputs/.
        _dataset_dir = _resolve_dataset_dir(args.experiment_type, args.output_dir)

        # Build derived paths from configuration
        output_summary = _dataset_dir / CONFIG["PATHS"]["output_summary_dir"]
        output_data = _dataset_dir / CONFIG["PATHS"]["output_data_dir"]
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
                "datasets_requested": args.datasets,
                "experiment_type_override": args.experiment_type,
                "experiment_type_path_map": EXPERIMENT_TYPE_FROM_PATH,
                "timestamp": datetime.now().isoformat(),
            },
        }
        with open(config_save_path, "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)  # default=str handles Path objects
        logging.info(f"Configuration saved to: {config_save_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration to {config_save_path}: {str(e)}")

    # Get list of folders to process based on mode (skip if in complement mode)
    if not skip_folder_discovery:
        processing_items = []

    if not skip_folder_discovery and args.mode == "experiment":
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

        # Filter out missing dirs upfront so the breadcrumb fires once
        # if everything is gone, rather than logging per-fly later.
        if Exp_folders:
            missing_exp = [f for f in Exp_folders if not f.exists()]
            if missing_exp:
                logging.warning(
                    f"{len(missing_exp)} experiment folder(s) from "
                    f"{args.yaml or 'CONFIG'} not found on disk; will be skipped."
                )
                Exp_folders = [f for f in Exp_folders if f.exists()]

        if not Exp_folders:
            from ballpushing_utils.paths import missing_data_message

            logging.error(
                "No experiment folders found to process. The YAML / "
                "experiment_filter resolved zero existing directories.\n\n"
                + missing_data_message(context="experiment data")
            )
            sys.exit(2)

        logging.info(f"Experiment folders to analyze: {[f.name for f in Exp_folders]}")
        processing_items = Exp_folders

    elif not skip_folder_discovery and args.mode == "flies":
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
            from ballpushing_utils.paths import missing_data_message

            logging.error(
                f"No valid fly directories found in {args.yaml}. None of "
                f"the entries resolved on disk.\n\n" + missing_data_message(context="fly tracks")
            )
            sys.exit(2)

        logging.info(f"Fly directories to analyze: {len(fly_dirs)} directories")
        logging.info(f"Sample directories: {[str(d) for d in fly_dirs[:5]]}")
        if len(fly_dirs) > 5:
            logging.info(f"... and {len(fly_dirs) - 5} more directories")

        processing_items = fly_dirs

    if not skip_folder_discovery:
        # Create metric subdirectories (already done in complement mode)
        for metric in args.datasets:
            (output_data / metric).mkdir(exist_ok=True)

    # Main processing loop
    # In complement mode, we don't use checkpoints since we process only missing files
    if skip_folder_discovery:
        checkpoint_file = None
    else:
        checkpoint_file = output_summary / "processing_checkpoint.json"

    # Load checkpoint if exists
    if checkpoint_file is not None and checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            processed_items = set(json.load(f))
        logging.info(f"Resuming from checkpoint, {len(processed_items)} items already processed")
    else:
        processed_items = set()

    # Track baseline memory for delta-based cache clearing
    baseline_memory_mb = get_memory_usage_mb()
    logging.info(f"Baseline memory usage: {baseline_memory_mb:.1f}MB")

    # Choose processing approach based on mode and options
    if args.mode == "flies" and args.batch_size > 1 and not args.force_cache_clear:
        # Use batch processing for flies mode to optimize cache clearing
        logging.info(f"Using batch processing for flies with batch size: {args.batch_size}")

        # Filter out already processed flies
        flies_to_process = []
        for fly_dir in processing_items:
            item_name = f"{fly_dir.parent.parent.name}_{fly_dir.parent.name}_{fly_dir.name}"
            if item_name not in processed_items:
                flies_to_process.append(fly_dir)
            else:
                logging.info(f"Skipping already processed fly {item_name}")

        log_memory_usage("Before batch processing flies")

        # Process flies in batches
        successfully_processed = process_flies_batch(
            flies_to_process,
            args.datasets,
            output_data,
            batch_size=args.batch_size,
            experiment_type_override=args.experiment_type,
        )

        # Update checkpoint with all processed flies (skip in complement mode)
        if checkpoint_file is not None:
            processed_items.update(successfully_processed)
            with open(checkpoint_file, "w") as f:
                json.dump(list(processed_items), f)

        log_memory_usage("After batch processing flies")

    else:
        # Use original individual processing
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
                    new_baseline = process_experiment(
                        item,
                        args.datasets,
                        output_data,
                        baseline_memory_mb,
                        experiment_type_override=args.experiment_type,
                    )
                    if new_baseline is not None:
                        baseline_memory_mb = new_baseline
                elif args.mode == "flies":
                    # Use individual processing with configurable cache clearing
                    process_fly_directory(
                        item,
                        args.datasets,
                        output_data,
                        force_cache_clear=args.force_cache_clear,
                        experiment_type_override=args.experiment_type,
                    )

                processed_items.add(item_name)

                # Save checkpoint after each successful processing (skip in complement mode)
                if checkpoint_file is not None:
                    with open(checkpoint_file, "w") as f:
                        json.dump(list(processed_items), f)

            except Exception as e:
                logging.error(f"Error processing {log_label}: {str(e)}")

            log_memory_usage(f"After processing {log_label}")

    if checkpoint_file is not None and checkpoint_file.exists():
        checkpoint_file.unlink()
        logging.info(f"Removed checkpoint file: {checkpoint_file}")

    logging.info("Processing complete!")

    # ==================================================================
    # Create pooled datasets
    # ==================================================================
    if args.skip_pooling:
        logging.info(f"Skipping pooling for metrics: {args.skip_pooling}")
    for metric in args.datasets:
        if metric in args.skip_pooling:
            logging.info(f"Skipping pooling for metric '{metric}' (excluded via --skip-pooling)")
            continue
        pooled_path = output_data / metric / f"pooled_{metric}.feather"

        # In complement mode, always redo pooling since we added new files
        if args.complement and pooled_path.exists():
            try:
                logging.info(f"Removing existing pooled dataset for re-pooling: {pooled_path.name}")
                pooled_path.unlink()
            except Exception as e:
                logging.error(f"Error removing pooled dataset {pooled_path.name}: {str(e)}")

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

    # Remove individual per-fly feathers unless the user asked to keep them.
    # Only applies to flies mode (in experiment mode the per-experiment files
    # are meaningful outputs in their own right).
    if args.mode == "flies" and not args.keep_individual:
        for metric in args.datasets:
            if metric in args.skip_pooling:
                continue
            metric_dir = output_data / metric
            pooled_ok = (metric_dir / f"pooled_{metric}.feather").exists()
            if not pooled_ok:
                continue
            individual_files = [f for f in metric_dir.glob("*.feather") if not f.name.startswith("pooled_")]
            for f in individual_files:
                f.unlink()
            if individual_files:
                logging.info(f"Removed {len(individual_files)} individual feathers for '{metric}' (pooled file kept)")

    end_time = time.time()
    runtime = end_time - start_time
    logging.info(f"Script completed in {runtime:.2f} seconds")
