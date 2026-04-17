import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.signal import find_peaks, savgol_filter

brain_regions_path = "/mnt/upramdya_data/MD/Region_map_250116.csv"


def save_object(obj, filename):
    """Save a custom object as a pickle file.

    Args:
        obj (object): The object to be saved.
        filename (Path): The path where to save the object. No need to add the .pkl extension.
    """
    # Ensure filename is a Path object
    filename = Path(filename)

    # If the filename does not end with .pkl, add it
    if filename.suffix != ".pkl":
        filename = filename.with_suffix(".pkl")

    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load a custom object from a pickle file.

    Args:
        filename (Pathlib path): the path to the object. No need to add the .pkl extension.
    """

    # Ensure filename is a Path object
    filename = Path(filename)

    # If the filename does not end with .pkl, add it
    if filename.suffix != ".pkl":
        filename = filename.with_suffix(".pkl")

    with open(filename, "rb") as input:
        obj = pickle.load(input)

    return obj


def find_interaction_events(
    protagonist1,
    protagonist2,
    nodes1=["Lfront", "Rfront"],
    nodes2=["x_centre", "y_centre"],
    threshold=[0, 11],
    gap_between_events=None,
    event_min_length=None,
    fps=29,
):
    """
    Finds interaction events as runs of frames where the distance between nodes is within threshold.
    Optionally merges runs separated by gap_between_events (in seconds), and/or filters by event_min_length (in seconds).
    """
    # Compute distances for all node pairs
    distances = []
    for node1 in nodes1:
        for node2 in nodes2:
            distances_node = np.sqrt(
                (protagonist1[f"x_{node1}"] - protagonist2[f"x_{node2}"]) ** 2
                + (protagonist1[f"y_{node1}"] - protagonist2[f"y_{node2}"]) ** 2
            )
            distances.append(distances_node)
    combined_distances = np.min(distances, axis=0)
    interaction_mask = (np.array(combined_distances) > threshold[0]) & (np.array(combined_distances) < threshold[1])
    interaction_frames = np.where(interaction_mask)[0]
    if len(interaction_frames) == 0:
        return []

    # Find consecutive runs of interaction frames
    runs = []
    start = interaction_frames[0]
    prev = interaction_frames[0]
    for f in interaction_frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append([start, prev])
            start = f
            prev = f
    runs.append([start, prev])

    # Merge runs if gap_between_events is set
    if gap_between_events is not None:
        gap_frames = int(round(gap_between_events * fps))
        merged_runs = []
        cur_start, cur_end = runs[0]
        for s, e in runs[1:]:
            if s - cur_end - 1 <= gap_frames:
                cur_end = e
            else:
                merged_runs.append([cur_start, cur_end])
                cur_start, cur_end = s, e
        merged_runs.append([cur_start, cur_end])
        runs = merged_runs

    # Filter by min length if set
    if event_min_length is not None:
        min_length_frames = int(round(event_min_length * fps))
        runs = [[s, e] for s, e in runs if (e - s + 1) >= min_length_frames]

    # If neither gap nor min length is set, return each frame as its own event
    if gap_between_events is None and event_min_length is None:
        return [[f, f, 1] for f in interaction_frames]
    else:
        return [[s, e, e - s + 1] for s, e in runs]


def find_interaction_boundaries(
    data,
    distance_col,
    frame_col,
    threshold_multiplier=1.5,
    window_size=30,
    min_plateau_length=50,
    peak_prominence=1,
    peak_window_size=10,
):
    """
    Find both the start and end of an interaction based on distance data.

    Returns:
        tuple: (start_index, end_index)
    """
    # Work on a copy to avoid modifying original data
    data = data.copy()

    # Smoothing and derivative calculation
    data["smoothed_distance"] = savgol_filter(data[distance_col], window_length=11, polyorder=3)
    data["smoothed_diff"] = data["smoothed_distance"].diff()
    data["smoothed_accel"] = data["smoothed_diff"].diff()  # Second derivative

    # Dynamic threshold for plateaus based on rolling standard deviation
    rolling_std = data["smoothed_diff"].rolling(window=window_size).std()
    dynamic_threshold = rolling_std.mean() * threshold_multiplier  # Adjust multiplier as needed

    # Plateau detection with dynamic threshold
    plateau_mask = data["smoothed_diff"].abs() < dynamic_threshold
    plateau_groups = (plateau_mask != plateau_mask.shift()).cumsum()

    # Initialize plateau markers
    data["plateau_start"] = 0
    data["plateau_end"] = 0
    valid_plateaus = data[plateau_mask].groupby(plateau_groups).filter(lambda x: len(x) >= min_plateau_length)
    if not valid_plateaus.empty:
        start_indices = valid_plateaus.groupby(plateau_groups).head(1).index
        end_indices = valid_plateaus.groupby(plateau_groups).tail(1).index
        data.loc[start_indices, "plateau_start"] = 1
        data.loc[end_indices, "plateau_end"] = 1

    # Peak detection with stricter prominence
    peaks, _ = find_peaks(-data["smoothed_distance"], prominence=peak_prominence, width=3)
    troughs, _ = find_peaks(data["smoothed_distance"], prominence=peak_prominence, width=3)

    # Refine peak detection for better alignment
    refined_peaks = []
    for peak in peaks:
        if peak > 0 and peak < len(data) - 1:
            local_region = data.iloc[max(0, peak - peak_window_size) : min(len(data), peak + peak_window_size)]
            true_peak_idx = local_region[distance_col].idxmin()  # Find true minimum in this region
            refined_peaks.append(true_peak_idx)

    refined_troughs = []
    for trough in troughs:
        if trough > 0 and trough < len(data) - 1:
            local_region = data.iloc[max(0, trough - peak_window_size) : min(len(data), trough + peak_window_size)]
            true_trough_idx = local_region[distance_col].idxmax()  # Find true maximum in this region
            refined_troughs.append(true_trough_idx)

    # Combine plateau and refined peak detections
    plateau_start_indices = data[data["plateau_start"] == 1].index
    plateau_end_indices = data[data["plateau_end"] == 1].index

    start_candidates = sorted(list(plateau_start_indices) + refined_peaks)
    end_candidates = sorted(list(plateau_end_indices) + refined_troughs)

    # Fallback to minimum/maximum distance if no markers found
    start_index = start_candidates[0] if start_candidates else data[distance_col].idxmin()
    end_index = end_candidates[-1] if end_candidates else data[distance_col].idxmax()

    # Ensure end_index is after start_index
    if end_index <= start_index:
        # Find the next valid end candidate after the start_index
        end_candidates_after_start = [idx for idx in end_candidates if idx > start_index]
        if end_candidates_after_start:
            end_index = end_candidates_after_start[0]
        else:
            # If no valid end candidate exists, fallback to the last frame
            end_index = data.index[-1]

    return start_index, end_index
