#!/usr/bin/env python3
"""
F1 Ball Identity Assignment Script

This script processes SLEAP tracking files to assign consistent ball identities based on x-position.
It finds all ball tracking .slp files in the specified directory and reassigns track identities
such that track_1 is always the leftmost ball (lower x) and track_2 is the rightmost ball (higher x).

Usage:
    python F1_assign_ball_id.py [--dry-run] [directory_filter1] [directory_filter2] ...

Arguments:
    --dry-run: Show what would be processed without making changes
    directory_filter: Optional filters to process only directories containing these strings

Examples:
    python F1_assign_ball_id.py --dry-run
    python F1_assign_ball_id.py arena2 arena4
    python F1_assign_ball_id.py 240924
"""

import sleap_io
import argparse
import sys
from pathlib import Path
import json
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import tqdm
import yaml

from ballpushing_utils.paths import dataset


def load_metadata(metadata_file, arena_number):
    """
    Load metadata for a specific arena from the metadata JSON file.

    Args:
        metadata_file (Path): Path to the metadata JSON file
        arena_number (int): Arena number to extract metadata for

    Returns:
        dict: Dictionary containing metadata for the specified arena
    """
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            variables = metadata["Variable"]
            metadata_dict = {}
            arena_key = f"Arena{arena_number}"

            if arena_key not in metadata:
                return {}

            for var in variables:
                var_index = variables.index(var)
                if var_index < len(metadata[arena_key]):
                    metadata_dict[var] = metadata[arena_key][var_index]

            # Make all keys lowercase for consistency
            metadata_dict = {k.lower(): v for k, v in metadata_dict.items()}
            return metadata_dict
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        return {}


def normalize_processed_filenames(data_dir, verbose=False, dry_run=False):
    """
    Normalize filenames with multiple '_processed' suffixes to have exactly one.
    Also consolidates duplicate H5 files to ensure only one H5 file per SLP file.

    Args:
        data_dir (Path): Directory to search for files
        verbose (bool): Print detailed information
        dry_run (bool): Show what would be normalized without making changes

    Returns:
        list: List of tuples (old_path, new_path) for normalized files
    """
    normalized_files = []
    data_path = Path(data_dir)

    if verbose:
        print(f"Normalizing processed filenames in: {data_path}")

    # First pass: Handle SLP files and files with multiple '_processed' suffixes
    for file_pattern in ["**/*.slp", "**/*.h5"]:
        for file_path in data_path.glob(file_pattern):
            filename = file_path.name
            stem = file_path.stem
            suffix = file_path.suffix

            # Count occurrences of '_processed'
            processed_count = stem.count("_processed")

            if processed_count > 1:
                # Handle H5 files with complex naming (e.g., file.000_Left.analysis.h5)
                if suffix == ".h5" and ".analysis" in stem:
                    # Split at .analysis and handle the parts separately
                    parts = stem.split(".analysis")
                    if len(parts) == 2:
                        main_part = parts[
                            0
                        ]  # e.g., "Left_tracked_ball_processed_processed_processed.000_Left"
                        analysis_suffix = parts[1]  # e.g., ""

                        # Remove all '_processed' from main part and add exactly one after base name
                        clean_main = main_part.replace("_processed", "")
                        # For H5 files, put _processed after the base name but before the .000_Side part
                        if ".000_" in clean_main:
                            base_part, side_part = clean_main.split(".000_", 1)
                            new_filename = (
                                base_part
                                + "_processed.000_"
                                + side_part
                                + ".analysis"
                                + analysis_suffix
                                + suffix
                            )
                        else:
                            new_filename = (
                                clean_main
                                + "_processed.analysis"
                                + analysis_suffix
                                + suffix
                            )
                    else:
                        # Fallback for unexpected H5 naming
                        base_name = stem.replace("_processed", "")
                        new_filename = base_name + "_processed" + suffix
                else:
                    # Regular files (SLP, etc.)
                    base_name = stem.replace("_processed", "")
                    new_filename = base_name + "_processed" + suffix

                new_path = file_path.with_name(new_filename)

                if dry_run:
                    print(f"  Would normalize: {file_path.name} -> {new_filename}")
                    normalized_files.append((str(file_path), str(new_path)))
                else:
                    # Check if target file already exists
                    if new_path.exists():
                        if verbose:
                            print(
                                f"  Target already exists, removing duplicate: {file_path.name}"
                            )
                        try:
                            file_path.unlink()  # Remove the duplicate
                            normalized_files.append(
                                (str(file_path), "removed_duplicate")
                            )
                        except Exception as e:
                            print(f"  Error removing duplicate {file_path}: {e}")
                    else:
                        # Rename the file
                        try:
                            if verbose:
                                print(
                                    f"  Normalizing: {file_path.name} -> {new_filename}"
                                )
                            file_path.rename(new_path)
                            normalized_files.append((str(file_path), str(new_path)))
                        except Exception as e:
                            print(f"  Error normalizing {file_path}: {e}")

    # Second pass: Find and consolidate duplicate H5 files
    if not dry_run:
        # Re-scan for H5 files after first pass
        h5_groups = {}
        for file_path in data_path.glob("**/*.h5"):
            stem = file_path.stem
            if "_processed" in stem and ".analysis" in stem:
                # Extract base name for grouping
                # Handle both formats: base_processed.000_Side.analysis and base.000_Side_processed.analysis
                if "_processed.000_" in stem:
                    # Correct format: base_processed.000_Side.analysis
                    base_name = stem.split("_processed.000_")[0]
                elif ".000_" in stem and stem.endswith("_processed.analysis"):
                    # Incorrect format: base.000_Side_processed.analysis
                    base_name = stem.split(".000_")[0]
                else:
                    continue

                if base_name not in h5_groups:
                    h5_groups[base_name] = []
                h5_groups[base_name].append(file_path)

        # For each group with multiple H5 files, keep only the correctly formatted one
        for base_name, h5_files in h5_groups.items():
            if len(h5_files) > 1:
                correct_files = []
                incorrect_files = []

                for h5_file in h5_files:
                    if "_processed.000_" in h5_file.stem:
                        correct_files.append(h5_file)
                    else:
                        incorrect_files.append(h5_file)

                # Remove incorrect format files
                for incorrect_file in incorrect_files:
                    try:
                        if verbose:
                            print(
                                f"  Removing incorrectly formatted H5: {incorrect_file.name}"
                            )
                        incorrect_file.unlink()
                        normalized_files.append(
                            (str(incorrect_file), "removed_incorrect_format")
                        )
                    except Exception as e:
                        print(f"  Error removing incorrect H5 {incorrect_file}: {e}")

    if verbose and normalized_files:
        print(f"Normalized {len(normalized_files)} files")
    elif verbose:
        print("No files needed normalization")

    return normalized_files


def cleanup_h5_files(slp_file, verbose=False, dry_run=False):
    """
    Clean up existing H5 files for a given SLP file location.
    Only deletes H5 files that correspond to the original (unprocessed) SLP file.

    Args:
        slp_file (Path): Path to the SLP file (processed or unprocessed)
        verbose (bool): Print detailed information
        dry_run (bool): Show what would be deleted without making changes
    """
    directory = slp_file.parent

    # Get the base name of the SLP file (without _processed suffix)
    slp_base_name = slp_file.stem.replace("_processed", "")

    # Only delete H5 files that match the original SLP file name (not processed versions)
    h5_files_to_delete = set()
    for h5_pattern in [f"{slp_base_name}*.h5", f"{slp_base_name}*.analysis.h5"]:
        for h5_file in directory.glob(h5_pattern):
            # Skip H5 files that have "_processed" in their name
            if "_processed" not in h5_file.stem and h5_file.exists():
                h5_files_to_delete.add(h5_file)

    # Delete the collected H5 files
    for h5_file in h5_files_to_delete:
        if dry_run:
            print(f"    Would delete H5 file: {h5_file}")
        else:
            try:
                if h5_file.exists():  # Double-check before deletion
                    os.remove(h5_file)
                    if verbose:
                        print(f"    Deleted existing H5 file: {h5_file}")
            except OSError as e:
                if verbose:
                    print(f"    Warning: Could not delete {h5_file}: {e}")


def prune_instances_tracks(slp_file, verbose=False, dry_run=False, reprocess=False):
    """
    Process a single SLP file to assign consistent ball identities using video-wide context.

    Args:
        slp_file (Path): Path to the SLP file
        verbose (bool): Print detailed processing information
        dry_run (bool): Show what would be processed without making changes
        reprocess (bool): Force reprocessing even if file was already processed
    """
    if verbose:
        print(f"\nProcessing: {slp_file}")

    # Always clean up H5 files first, even if already processed
    cleanup_h5_files(slp_file, verbose, dry_run)

    # Check if file has already been processed (unless reprocessing)
    if "_processed" in slp_file.stem and not reprocess:
        if verbose:
            print("  File already processed, skipping SLP processing")
        return

    if dry_run:
        print(f"  Would process: {slp_file}")
        return

    try:
        # Load the SLP file
        labels = sleap_io.load_slp(slp_file)

        if verbose:
            print(f"  Loaded {len(labels)} frames")

        # Reset tracks if there are too many
        if len(labels.tracks) > 2:
            if verbose:
                print("  Too many tracks defined, removing all")
            for frame in labels:
                for instance in frame.instances:
                    instance.track = None
            labels.tracks = []

        # Ensure we have exactly 2 tracks with proper names
        if len(labels.tracks) == 0:
            labels.tracks.append(sleap_io.Track(name="training_ball"))
            labels.tracks.append(sleap_io.Track(name="test_ball"))
        elif len(labels.tracks) == 1:
            labels.tracks.append(sleap_io.Track(name="test_ball"))
        else:
            # Update existing track names
            labels.tracks[0].name = "training_ball"
            labels.tracks[1].name = "test_ball"

        # Phase 1: Clean up instances and establish reference positions
        frames_with_multiple_instances = 0
        reference_positions = None  # Will store (left_x, right_x) reference positions

        # First pass: clean up instances per frame and collect good reference frames
        good_frames_data = (
            []
        )  # Store (frame_idx, left_x, right_x) for frames with 2 good detections

        for frame in labels:
            # Sort instances by confidence (highest first)
            frame.instances.sort(key=lambda x: x.score, reverse=True)

            # Keep only the 2 highest confidence instances that are sufficiently far apart
            kept_instances = []
            for instance in frame.instances:
                if len(kept_instances) == 0:
                    kept_instances.append(instance)
                elif len(kept_instances) == 1:
                    # Get x positions
                    first_x = kept_instances[0].points[0][0][0]
                    inst_x = instance.points[0][0][0]

                    # Only keep if far enough apart (90 pixels threshold)
                    if abs(first_x - inst_x) > 90:
                        kept_instances.append(instance)
                        break

            # Update frame instances
            frame.instances = kept_instances

            if len(frame.instances) == 2:
                frames_with_multiple_instances += 1
                # Sort by x position and record for reference
                x_positions = [inst.points[0][0][0] for inst in frame.instances]
                x_positions.sort()
                good_frames_data.append(
                    (frame.frame_idx, x_positions[0], x_positions[1])
                )

        if verbose:
            print(f"  Frames with 2 instances: {frames_with_multiple_instances}")
            print(
                f"  Found {len(good_frames_data)} frames with good detections for reference"
            )

        # Phase 2: Establish reference positions from early frames with good detections
        if (
            len(good_frames_data) >= 5
        ):  # Need at least 5 good frames to establish reference
            # Use the first 20% of good frames to establish reference positions
            early_frames = good_frames_data[: max(5, len(good_frames_data) // 5)]

            # Calculate median positions for stability
            left_positions = [data[1] for data in early_frames]  # left x positions
            right_positions = [data[2] for data in early_frames]  # right x positions

            import statistics

            ref_left_x = statistics.median(left_positions)
            ref_right_x = statistics.median(right_positions)
            reference_positions = (ref_left_x, ref_right_x)

            if verbose:
                print(
                    f"  Established reference positions: left={ref_left_x:.1f}, right={ref_right_x:.1f}"
                )
        else:
            if verbose:
                print(
                    "  Warning: Not enough good frames to establish reference positions, using frame-by-frame sorting"
                )

        # Phase 3: Assign consistent identities using reference positions
        identity_swaps_corrected = 0

        for frame in labels:
            if len(frame.instances) == 2:
                if reference_positions is not None:
                    # Use reference positions to maintain consistent identity
                    ref_left_x, ref_right_x = reference_positions

                    # Get current instance positions
                    inst1, inst2 = frame.instances
                    x1, x2 = inst1.points[0][0][0], inst2.points[0][0][0]

                    # Calculate distances to reference positions
                    # Option 1: inst1=left, inst2=right
                    dist1_left = abs(x1 - ref_left_x)
                    dist2_right = abs(x2 - ref_right_x)
                    option1_cost = dist1_left + dist2_right

                    # Option 2: inst1=right, inst2=left
                    dist1_right = abs(x1 - ref_right_x)
                    dist2_left = abs(x2 - ref_left_x)
                    option2_cost = dist1_right + dist2_left

                    # Choose the assignment with lower total distance
                    if option1_cost <= option2_cost:
                        # inst1 = training_ball (left), inst2 = test_ball (right)
                        frame.instances = [inst1, inst2]
                        if x1 > x2:  # Instances were actually swapped
                            identity_swaps_corrected += 1
                    else:
                        # inst2 = training_ball (left), inst1 = test_ball (right)
                        frame.instances = [inst2, inst1]
                        if x2 > x1:  # Instances were actually swapped
                            identity_swaps_corrected += 1
                else:
                    # Fallback: sort by x position (left to right)
                    frame.instances.sort(key=lambda x: x.points[0][0][0])

                # Assign tracks: training_ball (index 0) = left, test_ball (index 1) = right
                frame.instances[0].track = labels.tracks[0]  # training_ball
                frame.instances[1].track = labels.tracks[1]  # test_ball

            elif len(frame.instances) == 1:
                # Single instance: assign based on proximity to reference positions
                instance = frame.instances[0]
                x_pos = instance.points[0][0][0]

                if reference_positions is not None:
                    ref_left_x, ref_right_x = reference_positions

                    # Assign to the closer reference position
                    if abs(x_pos - ref_left_x) <= abs(x_pos - ref_right_x):
                        instance.track = labels.tracks[0]  # training_ball (left)
                    else:
                        instance.track = labels.tracks[1]  # test_ball (right)
                else:
                    # No reference, assign based on absolute position (assume left if < center)
                    # This is a fallback and might not be accurate
                    instance.track = (
                        labels.tracks[0] if x_pos < 500 else labels.tracks[1]
                    )

        if verbose:
            print(f"  Identity swaps corrected: {identity_swaps_corrected}")

        # Save the updated labels
        sleap_io.save_slp(labels, slp_file)

        # Rename the file to indicate it has been processed (only if not already processed)
        slp_file_path = Path(slp_file)
        if "_processed" not in slp_file_path.stem:
            processed_filename = (
                slp_file_path.stem + "_processed" + slp_file_path.suffix
            )
            processed_path = slp_file_path.with_name(processed_filename)
            try:
                os.rename(slp_file, processed_path)
                if verbose:
                    print(f"  Renamed file to {processed_path}")
            except OSError as e:
                print(f"  Error renaming file: {e}")
        else:
            if verbose:
                print(
                    f"  File already has '_processed' suffix, keeping name: {slp_file_path}"
                )

    except Exception as e:
        print(f"Error processing {slp_file}: {e}")


def reassign_tracks_based_on_x(slp_file, verbose=False, dry_run=False, reprocess=False):
    """
    Reassign track identities using video-wide context for existing tracks.

    Args:
        slp_file (Path): Path to the SLP file
        verbose (bool): Print detailed processing information
        dry_run (bool): Show what would be processed without making changes
        reprocess (bool): Force reprocessing even if file was already processed
    """
    # Always clean up H5 files first
    cleanup_h5_files(slp_file, verbose, dry_run)

    if dry_run:
        print(f"  Would reassign tracks: {slp_file}")
        return

    try:
        labels = sleap_io.load_slp(slp_file)

        if len(labels.tracks) != 2:
            if verbose:
                print(
                    f"  Expected 2 tracks, found {len(labels.tracks)}, skipping reassignment"
                )
            return

        # Update track names
        labels.tracks[0].name = "training_ball"
        labels.tracks[1].name = "test_ball"

        # Establish reference positions from frames with 2 instances
        good_frames_data = []

        for frame in labels:
            if len(frame.instances) == 2:
                # Get x positions and sort them
                x_positions = [inst.points[0][0][0] for inst in frame.instances]
                x_positions.sort()
                good_frames_data.append(
                    (frame.frame_idx, x_positions[0], x_positions[1])
                )

        if verbose:
            print(
                f"  Found {len(good_frames_data)} frames with 2 instances for reference"
            )

        # Establish reference positions if we have enough data
        reference_positions = None
        if len(good_frames_data) >= 5:
            # Use early frames to establish reference
            early_frames = good_frames_data[: max(5, len(good_frames_data) // 5)]

            import statistics

            left_positions = [data[1] for data in early_frames]
            right_positions = [data[2] for data in early_frames]

            ref_left_x = statistics.median(left_positions)
            ref_right_x = statistics.median(right_positions)
            reference_positions = (ref_left_x, ref_right_x)

            if verbose:
                print(
                    f"  Reference positions: left={ref_left_x:.1f}, right={ref_right_x:.1f}"
                )

        # Reassign tracks using reference positions
        identity_swaps_corrected = 0

        for frame in labels:
            if len(frame.instances) == 2:
                if reference_positions is not None:
                    # Use reference-based assignment
                    ref_left_x, ref_right_x = reference_positions

                    inst1, inst2 = frame.instances
                    x1, x2 = inst1.points[0][0][0], inst2.points[0][0][0]

                    # Calculate assignment costs
                    option1_cost = abs(x1 - ref_left_x) + abs(x2 - ref_right_x)
                    option2_cost = abs(x1 - ref_right_x) + abs(x2 - ref_left_x)

                    if option1_cost <= option2_cost:
                        frame.instances = [inst1, inst2]
                        if x1 > x2:
                            identity_swaps_corrected += 1
                    else:
                        frame.instances = [inst2, inst1]
                        if x2 > x1:
                            identity_swaps_corrected += 1
                else:
                    # Fallback: sort by x position
                    frame.instances.sort(key=lambda x: x.points[0][0][0])

                # Assign tracks
                frame.instances[0].track = labels.tracks[0]  # training_ball (left)
                frame.instances[1].track = labels.tracks[1]  # test_ball (right)

            elif len(frame.instances) == 1:
                # Single instance assignment
                instance = frame.instances[0]
                x_pos = instance.points[0][0][0]

                if reference_positions is not None:
                    ref_left_x, ref_right_x = reference_positions
                    if abs(x_pos - ref_left_x) <= abs(x_pos - ref_right_x):
                        instance.track = labels.tracks[0]  # training_ball
                    else:
                        instance.track = labels.tracks[1]  # test_ball
                else:
                    # Fallback
                    instance.track = (
                        labels.tracks[0] if x_pos < 500 else labels.tracks[1]
                    )

        if verbose:
            print(f"  Identity swaps corrected: {identity_swaps_corrected}")

        # Save updated labels
        sleap_io.save_slp(labels, slp_file)

        # Rename the file to indicate it has been processed (only if not already processed)
        slp_file_path = Path(slp_file)
        if "_processed" not in slp_file_path.stem:
            processed_filename = (
                slp_file_path.stem + "_processed" + slp_file_path.suffix
            )
            processed_path = slp_file_path.with_name(processed_filename)
            try:
                os.rename(slp_file, processed_path)
                if verbose:
                    print(f"  Reassigned tracks and renamed to {processed_path}")
            except OSError as e:
                print(f"  Error renaming file: {e}")
        else:
            if verbose:
                print(
                    f"  File already has '_processed' suffix, keeping name: {slp_file_path}"
                )

    except Exception as e:
        print(f"Error reassigning tracks for {slp_file}: {e}")


def convert_slp_to_h5(slp_file, conda_env="sleap", dry_run=False):
    """
    Convert SLP file to H5 format using sleap-convert.

    Args:
        slp_file (Path): Path to the SLP file
        conda_env (str): Name of the conda environment with SLEAP
        dry_run (bool): Show what would be processed without making changes
    """
    if dry_run:
        print(f"  Would convert to H5: {slp_file}")
        return

    # Check if H5 file already exists
    h5_file = slp_file.with_suffix(".h5")
    analysis_h5_file = slp_file.with_suffix(".analysis.h5")

    if h5_file.exists() or analysis_h5_file.exists():
        print(f"  H5 file already exists for {slp_file}, skipping conversion")
        return

    sleap_convert_cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {conda_env} && \
    sleap-convert "{slp_file}" --format analysis
    """

    try:
        subprocess.run(
            sleap_convert_cmd, shell=True, check=True, executable="/bin/bash"
        )
        print(f"  Converted {slp_file} to H5 format")
    except subprocess.CalledProcessError as e:
        print(f"  Error converting {slp_file} to H5: {e}")
    except OSError as e:
        print(f"  OS error converting {slp_file}: {e}")


def process_control_experiments(
    slp_file, verbose=False, dry_run=False, reprocess=False
):
    """
    Special processing for control experiments (should have only 1 ball).

    Args:
        slp_file (Path): Path to the SLP file
        verbose (bool): Print detailed processing information
        dry_run (bool): Show what would be processed without making changes
        reprocess (bool): Force reprocessing even if file was already processed
    """
    # Always clean up H5 files first
    cleanup_h5_files(slp_file, verbose, dry_run)

    if dry_run:
        print(f"  Would process control experiment: {slp_file}")
        return

    try:
        labels = sleap_io.load_slp(slp_file)

        # Reset tracks for control experiments
        if len(labels.tracks) > 1:
            if verbose:
                print("  Control experiment: removing extra tracks")
            for frame in labels:
                for instance in frame.instances:
                    instance.track = None
            labels.tracks = [sleap_io.Track(name="test_ball")]
        elif len(labels.tracks) == 0:
            labels.tracks = [sleap_io.Track(name="test_ball")]
        else:
            # Update existing single track name
            labels.tracks[0].name = "test_ball"

        # Remove low confidence instances and keep only 1 per frame
        for frame in labels:
            # Remove instances with score < 0.3
            frame.instances = [
                instance for instance in frame.instances if instance.score >= 0.3
            ]

            # Keep only the highest confidence instance
            if len(frame.instances) > 1:
                frame.instances.sort(key=lambda x: x.score, reverse=True)
                frame.instances = frame.instances[:1]

            # Assign test_ball to the remaining instance
            for instance in frame.instances:
                instance.track = labels.tracks[0]

        # Save updated labels
        sleap_io.save_slp(labels, slp_file)

        # Rename the file to indicate it has been processed (only if not already processed)
        slp_file_path = Path(slp_file)
        if "_processed" not in slp_file_path.stem:
            processed_filename = (
                slp_file_path.stem + "_processed" + slp_file_path.suffix
            )
            processed_path = slp_file_path.with_name(processed_filename)
            try:
                os.rename(slp_file, processed_path)
                if verbose:
                    print(
                        f"  Processed control experiment and renamed to {processed_path}"
                    )
            except OSError as e:
                print(f"  Error renaming file: {e}")
        else:
            if verbose:
                print(
                    f"  File already has '_processed' suffix, keeping name: {slp_file_path}"
                )

    except Exception as e:
        print(f"Error processing control experiment {slp_file}: {e}")


def find_ball_slp_files(
    data_dir,
    directory_filters=None,
    verbose=False,
    include_processed=False,
    reprocess=False,
):
    """
    Find all ball tracking SLP files in the data directory.

    Args:
        data_dir (Path): Root directory to search
        directory_filters (list): Optional list of strings to filter directories
        verbose (bool): Print search information
        include_processed (bool): Include already processed files
        reprocess (bool): Include processed files for reprocessing (ignore _processed suffix)

    Returns:
        list: List of Path objects for found SLP files
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return []

    # First, find all ball SLP files in the entire directory tree
    all_ball_slps = list(data_dir.rglob("*ball*.slp"))

    if verbose:
        print(f"Found {len(all_ball_slps)} ball SLP files in total")

    # Filter directories if filters provided
    if directory_filters:
        filtered_slps = []
        for dir_filter in directory_filters:
            for slp_file in all_ball_slps:
                if dir_filter in str(slp_file):
                    filtered_slps.append(slp_file)
        all_ball_slps = filtered_slps

        if verbose:
            print(
                f"Filtered to {len(all_ball_slps)} files matching: {directory_filters}"
            )

    # Check if we're in a processed directory or subdirectory of one
    # Look for "_Checked" in the path hierarchy
    processed_slps = []
    for slp_file in all_ball_slps:
        # Check if any parent directory contains "_Checked"
        is_processed = any("_Checked" in parent.name for parent in slp_file.parents)

        if is_processed:
            processed_slps.append(slp_file)
        elif verbose:
            print(f"  Skipping {slp_file} (not in processed directory)")

    if verbose:
        print(f"Found {len(processed_slps)} ball SLP files in processed directories")

    # Remove already processed files (unless we want to include them for cleanup or reprocessing)
    if include_processed or reprocess:
        final_slps = processed_slps
        if verbose:
            if reprocess:
                print(
                    f"Found {len(final_slps)} ball SLP files (including processed for reprocessing)"
                )
            else:
                print(f"Found {len(final_slps)} ball SLP files (including processed)")
    else:
        final_slps = [f for f in processed_slps if "_processed" not in f.stem]
        if verbose:
            print(f"Found {len(final_slps)} unprocessed ball SLP files")

    if verbose:
        for slp_file in final_slps:
            print(f"  - {slp_file}")

    return final_slps


def main():
    parser = argparse.ArgumentParser(
        description="Assign ball identities in SLEAP tracking files based on x-position",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be processed without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed processing information",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=str(dataset("F1_Tracks/Videos")),
        help="Root directory containing tracking data (default: $BALLPUSHING_DATA_ROOT/F1_Tracks/Videos)",
    )
    parser.add_argument(
        "--no-convert-h5",
        action="store_true",
        help="Skip converting processed SLP files to H5 format (conversion is done by default)",
    )
    parser.add_argument(
        "--cleanup-h5",
        action="store_true",
        help="Clean up existing H5 files even for already processed SLP files",
    )
    parser.add_argument(
        "--reprocess",
        "--force",
        action="store_true",
        help="Reprocess files even if they have already been processed (overwrite processed files)",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Only normalize filenames with multiple '_processed' suffixes, don't do other processing",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Process files in parallel (faster but less verbose)",
    )
    parser.add_argument(
        "--yaml-file",
        "-y",
        type=str,
        help="Path to YAML file containing list of directories to process (alternative to directory_filters)",
    )
    parser.add_argument(
        "directory_filters",
        nargs="*",
        help="Filter directories to process (e.g., 'arena2', '240924')",
    )

    args = parser.parse_args()

    # Convert to h5 by default, unless --no-convert-h5 is specified
    args.convert_h5 = not args.no_convert_h5

    # Load directories from YAML file if provided
    directory_filters = args.directory_filters
    if args.yaml_file:
        try:
            with open(args.yaml_file, "r") as f:
                yaml_data = yaml.safe_load(f)
                yaml_directories = yaml_data.get("directories", [])
                if yaml_directories:
                    # Extract just the directory names from full paths to use as filters
                    directory_filters = [Path(d).name for d in yaml_directories]
                    print(
                        f"Loaded {len(directory_filters)} directories from YAML file: {args.yaml_file}"
                    )
                else:
                    print(
                        f"Warning: No directories found in YAML file: {args.yaml_file}"
                    )
        except Exception as e:
            print(f"Error loading YAML file {args.yaml_file}: {e}")
            sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN MODE - No actual processing will occur ===")
        if args.reprocess:
            print("REPROCESSING MODE: Will overwrite already processed files")

    # Find all ball SLP files
    # Include processed files if we're doing cleanup, conversion, reprocessing, or normalize-only
    include_processed = (
        args.cleanup_h5 or args.convert_h5 or args.reprocess or args.normalize_only
    )
    slp_files = find_ball_slp_files(
        args.data_dir,
        directory_filters if directory_filters else None,
        args.verbose,
        include_processed,
        args.reprocess,
    )

    if not slp_files and not args.normalize_only:
        if include_processed:
            print("No ball SLP files found")
        else:
            print("No unprocessed ball SLP files found to process")
        return

    # First, normalize any files with multiple '_processed' suffixes
    if args.verbose or args.dry_run:
        print(f"\n=== NORMALIZING FILENAMES ===")

    normalized_files = normalize_processed_filenames(
        args.data_dir, verbose=args.verbose, dry_run=args.dry_run
    )

    if normalized_files and not args.dry_run:
        # Re-find files after normalization since paths may have changed
        slp_files = find_ball_slp_files(
            args.data_dir,
            directory_filters if directory_filters else None,
            verbose=False,  # Suppress verbose output for re-find
            include_processed=include_processed,
            reprocess=args.reprocess,
        )

    # If normalize-only mode, stop here
    if args.normalize_only:
        if normalized_files:
            print(f"Normalization complete! Processed {len(normalized_files)} files.")
        else:
            print("No files needed normalization.")
        return

    # Check if we're only doing H5 conversion with all processed files (and not reprocessing)
    unprocessed_files = [f for f in slp_files if "_processed" not in f.stem]
    if args.convert_h5 and len(unprocessed_files) == 0 and not args.reprocess:
        print(f"\n=== H5 CONVERSION MODE ===")
        print(f"Converting {len(slp_files)} processed files to H5 format...")

        if args.dry_run:
            print("Would convert processed files to H5:")
            for slp_file in slp_files:
                print(f"  - {slp_file}")
                print(
                    f"    Would convert to H5: {slp_file.with_suffix('.analysis.h5')}"
                )
        else:
            for slp_file in slp_files:
                if args.verbose:
                    print(f"\nConverting: {slp_file}")
                convert_slp_to_h5(slp_file, dry_run=args.dry_run)

        print("H5 conversion complete!")
        return

    # Check if we're only doing cleanup with all processed files (and not reprocessing)
    if args.cleanup_h5 and len(unprocessed_files) == 0 and not args.reprocess:
        print(f"\n=== H5 CLEANUP MODE ===")
        print(f"Cleaning up old H5 files for {len(slp_files)} processed files...")

        if args.dry_run:
            print("Would perform H5 cleanup:")
            for slp_file in slp_files:
                print(f"  - {slp_file}")
                cleanup_h5_files(slp_file, args.verbose, args.dry_run)
        else:
            for slp_file in slp_files:
                if args.verbose:
                    print(f"\nCleaning up old H5 files for: {slp_file}")
                cleanup_h5_files(slp_file, args.verbose, args.dry_run)

        print("H5 cleanup complete!")
        return

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total files to process: {len(slp_files)}")

    if args.dry_run:
        print("Files that would be processed:")
        for slp_file in slp_files:
            # Get metadata for this file
            metadata_file = slp_file.parent.parent.parent / "Metadata.json"

            # Try to extract arena number from path
            arena_number = 1  # default
            for part in slp_file.parts:
                if "arena" in part.lower():
                    try:
                        arena_number = int(part.lower().replace("arena", ""))
                        break
                    except ValueError:
                        pass

            metadata = load_metadata(metadata_file, arena_number)

            # Determine experiment type
            is_control = (
                "pretraining" in metadata
                and "n" in str(metadata["pretraining"]).lower()
            ) or (
                "pretrained" in metadata and "n" in str(metadata["pretrained"]).lower()
            )

            exp_type = (
                "CONTROL" if is_control else "PRETRAINED" if metadata else "OTHER"
            )

            print(f"  - {slp_file}")
            print(f"    Type: {exp_type}")
            print(f"    Metadata file: {metadata_file}")
            if metadata:
                print(f"    Metadata: {metadata}")
            else:
                print(f"    Metadata: No metadata found")

            if is_control:
                print(
                    f"    → Will process as CONTROL (remove extra tracks, single 'test_ball' track)"
                )
            elif metadata:
                print(
                    f"    → Will process as PRETRAINED (assign 'training_ball' left, 'test_ball' right)"
                )
            else:
                print(
                    f"    → Will process as OTHER (assign 'training_ball' left, 'test_ball' right)"
                )
            print()
        print("=== END DRY RUN ===")
        return

    # Categorize files by experiment type
    control_files = []
    pretrained_files = []
    other_files = []

    for slp_file in slp_files:
        metadata_file = slp_file.parent.parent.parent / "Metadata.json"

        # Try to extract arena number from path (e.g., "arena1", "arena2")
        arena_number = 1  # default
        for part in slp_file.parts:
            if "arena" in part.lower():
                try:
                    arena_number = int(part.lower().replace("arena", ""))
                    break
                except ValueError:
                    pass

        metadata = load_metadata(metadata_file, arena_number)

        # Check if this is a control experiment
        is_control = (
            "pretraining" in metadata and "n" in str(metadata["pretraining"]).lower()
        ) or ("pretrained" in metadata and "n" in str(metadata["pretrained"]).lower())

        if is_control:
            control_files.append(slp_file)
        elif metadata:  # Has metadata but not control
            pretrained_files.append(slp_file)
        else:
            other_files.append(slp_file)

    print(f"Control experiments: {len(control_files)}")
    if control_files and args.verbose:
        for cf in control_files:
            print(f"  - {cf}")
    print(f"Pretrained experiments: {len(pretrained_files)}")
    if pretrained_files and args.verbose:
        for pf in pretrained_files:
            print(f"  - {pf}")
    print(f"Other files: {len(other_files)}")
    if other_files and args.verbose:
        for of in other_files:
            print(f"  - {of}")

    # Process files
    if args.parallel and not args.verbose:
        # Process in parallel (less verbose)
        print("\nProcessing files in parallel...")

        with ThreadPoolExecutor() as executor:
            # Process control experiments
            if control_files:
                print("Processing control experiments...")
                futures = [
                    executor.submit(
                        process_control_experiments, f, False, False, args.reprocess
                    )
                    for f in control_files
                ]
                for future in tqdm.tqdm(futures, desc="Control experiments"):
                    future.result()

            # Process pretrained experiments
            if pretrained_files:
                print("Processing pretrained experiments...")
                futures = [
                    executor.submit(
                        prune_instances_tracks, f, False, False, args.reprocess
                    )
                    for f in pretrained_files
                ]
                for future in tqdm.tqdm(futures, desc="Pretrained experiments"):
                    future.result()

            # Process other files
            if other_files:
                print("Processing other files...")
                futures = [
                    executor.submit(
                        prune_instances_tracks, f, False, False, args.reprocess
                    )
                    for f in other_files
                ]
                for future in tqdm.tqdm(futures, desc="Other files"):
                    future.result()
    else:
        # Process sequentially (more verbose)
        print("\nProcessing files sequentially...")

        # Process control experiments
        if control_files:
            print(f"\nProcessing {len(control_files)} control experiments...")
            for slp_file in control_files:
                process_control_experiments(
                    slp_file, args.verbose, False, args.reprocess
                )

        # Process pretrained experiments
        if pretrained_files:
            print(f"\nProcessing {len(pretrained_files)} pretrained experiments...")
            for slp_file in pretrained_files:
                prune_instances_tracks(slp_file, args.verbose, False, args.reprocess)

        # Process other files
        if other_files:
            print(f"\nProcessing {len(other_files)} other files...")
            for slp_file in other_files:
                prune_instances_tracks(slp_file, args.verbose, False, args.reprocess)

    # Convert to H5 if requested
    if args.convert_h5:
        print(f"\nConverting processed files to H5 format...")
        # Find all processed SLP files that currently exist (after normalization)
        processed_files = []
        data_path = Path(args.data_dir)

        # Search for all SLP files with "_processed" in their name
        for slp_pattern in ["**/*_processed*.slp"]:
            for slp_file in data_path.glob(slp_pattern):
                # Check if it's in a processed directory (_Checked)
                is_processed_dir = any(
                    "_Checked" in parent.name for parent in slp_file.parents
                )
                if is_processed_dir and "_processed" in slp_file.stem:
                    # Apply directory filters if provided
                    if directory_filters:
                        if any(
                            dir_filter in str(slp_file)
                            for dir_filter in directory_filters
                        ):
                            processed_files.append(slp_file)
                    else:
                        processed_files.append(slp_file)

        print(f"Found {len(processed_files)} processed files to convert...")
        for processed_file in processed_files:
            if args.verbose:
                print(f"Converting: {processed_file}")
            convert_slp_to_h5(processed_file, dry_run=False)

    print("\n=== PROCESSING COMPLETE ===")


if __name__ == "__main__":
    main()
