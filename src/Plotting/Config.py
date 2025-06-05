from pathlib import Path
import json
import pyarrow
import math
import re
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from scipy import stats
from scipy.stats import gaussian_kde
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import holoviews as hv
from holoviews import opts

hv.extension("bokeh")

from utils_behavior import (
    Utils,
    Processing,
)

import importlib
from statsmodels.stats.multitest import multipletests


from tqdm import tqdm

### Configuration for the analysis

datapath = Utils.get_data_server()

# Import the Split registry
SplitRegistry = pd.read_csv(datapath / "MD/Region_map_250130.csv")

# Downsampling if any

downsampling_factor = None

# Statistics

show_signif = True

# Data binning

data_bins = 10

# Permutation test parameters

permutation_method = "scipy"

# permutation_metric = "avg_distance"

n_permutations = 1000

error_logs = True


def map_split_registry(df):
    # Add the Split column from the SplitRegistry to the Disp_Data, merging based on the Genotype column, only keeping the Simplified Nickname and Split columns
    df = df.merge(
        SplitRegistry[["Genotype", "Simplified Nickname", "Split"]],
        on="Genotype",
        how="left",
    )

    # Check for NA values in the Split column
    if df["Split"].isna().sum() > 0:
        print(f"The Nicknames with NA values in the Split column are: {df[df['Split'].isna()]['Nickname'].unique()}")

    return df


def cleanup_data(df):
    # Remove the non-TNT data from the Disp_Data
    df = df[~df["Genotype"].isin(["M6", "M7", "PR", "CS"])]
    df = df[df["Brain region"] != "None"]
    return df


# Prepare the control data for the analysis
## Manually create a color dictionary for the brain regions
color_dict = {
    "MB": "#1f77b4",  # Blue
    "Vision": "#ff7f0e",  # Orange
    "LH": "#2ca02c",  # Green
    "Neuropeptide": "#d62728",  # Red
    "Olfaction": "#9467bd",  # Purple
    "MB extrinsic neurons": "#8c564b",  # Brown
    "CX": "#e377c2",  # Pink
    "Control": "#7f7f7f",  # Gray
    "None": "#bcbd22",  # Yellow-green
    "fchON": "#17becf",  # Cyan
    "JON": "#ffbb78",  # Light orange
    "DN": "#c5b0d5",  # Light purple
}


def prepare_registries(SplitRegistry):
    brain_regions = SplitRegistry["Simplified region"].unique()

    # Define the control region and get unique nicknames
    control_region = "Control"
    control_nicknames = SplitRegistry[SplitRegistry["Simplified region"] == control_region]["Nickname"].unique()
    nicknames = SplitRegistry["Nickname"].unique()
    nicknames = [nickname for nickname in nicknames if nickname not in control_nicknames]

    # Combine all nicknames for consistent coloring
    all_nicknames = list(control_nicknames) + nicknames
    control_nicknames_dict = {"y": "Empty-Split", "n": "Empty-Gal4", "m": "TNTxPR"}
    nicknames = SplitRegistry["Nickname"].unique()
    nicknames = [nickname for nickname in nicknames if nickname not in control_nicknames_dict.values()]

    # Make a dictionary with the generated variables
    registries = {
        "brain_regions": brain_regions,
        "control_region": control_region,
        "control_nicknames": control_nicknames,
        "nicknames": nicknames,
        "all_nicknames": all_nicknames,
        "control_nicknames_dict": control_nicknames_dict,
    }

    return registries


registries = prepare_registries(SplitRegistry)


def get_subset_data(df, col="Nickname", value="random"):

    control_nicknames_dict = registries["control_nicknames_dict"]

    if value == "random":
        # Pick one random Nickname and get a subset of it
        nicknames = df["Nickname"].unique()
        nickname = np.random.choice(nicknames)
    else:
        nickname = value

    print(f"Nickname selected: {nickname}")

    # Check if 'Split' column exists
    if "Split" not in df.columns:
        print("Error: 'Split' column not found in dataframe.")
        return pd.DataFrame()

    # Check if the nickname exists in the dataframe
    if nickname not in df["Nickname"].values:
        print(f"Nickname {nickname} not found in dataframe.")
        return pd.DataFrame()

    # Get the associated control
    split_value = SplitRegistry[SplitRegistry["Nickname"] == nickname]["Split"].iloc[0]
    associated_control = control_nicknames_dict.get(split_value)

    if not associated_control:
        print(f"No associated control found for split value {split_value}.")
        return pd.DataFrame()

    print(f"Associated control is: {associated_control}")

    # Get the subset of the data for the random Nickname
    subset_data = df[df["Nickname"] == nickname]

    # Get the subset of the data for the associated control
    control_data = df[df["Nickname"] == associated_control]

    # Check if either subset is empty and handle accordingly
    if subset_data.empty:
        print(f"No data found for nickname {nickname}.")
    if control_data.empty:
        print(f"No data found for associated control {associated_control}.")

    # Combine the nickname data with the relevant control data
    subset_data = pd.concat([subset_data, control_data])

    return subset_data


def load_datasets_for_brain_region(brain_region, data_path, registries, downsample_factor=None):
    # Load the dataset for the brain region
    brain_region_file = data_path / f"{brain_region}.feather"
    if not brain_region_file.exists():
        print(f"Dataset for brain region {brain_region} not found.")
        return pd.DataFrame()

    brain_region_data = pd.read_feather(brain_region_file)

    # Load the dataset for the control region
    control_region = registries["control_region"]
    control_region_file = data_path / f"{control_region}.feather"
    if not control_region_file.exists():
        print(f"Dataset for control region {control_region} not found.")
        return pd.DataFrame()

    control_region_data = pd.read_feather(control_region_file)

    if brain_region_data.empty:
        raise ValueError(f"Empty dataset for {brain_region}")
    if control_region_data.empty:
        raise ValueError(f"Empty control dataset")

    # Combine the datasets
    combined_data = pd.concat([brain_region_data, control_region_data], ignore_index=True)

    # Downsample the data if downsample_factor is provided
    if downsample_factor:
        combined_data = combined_data.iloc[::downsample_factor, :]

    return combined_data


# def preprocess_data(data, bins=10):
#     """
#     Prepare a simplified dataset by splitting the data into n time bins and computing the average and median distance for each bin. It should be done individually for each fly.

#     Args:
#         data (pd.DataFrame): The input data containing 'time', 'Brain region', 'fly', and 'distance_ball_0'.
#         bins (int, optional): The number of time bins to split the data into. Default is 10.

#     Returns:
#         pd.DataFrame: A DataFrame containing the average and median distance for each time bin, brain region, and fly.
#     """
#     # Create a new column 'time_bin' that indicates the time bin each 'time' value belongs to
#     data['time_bin'] = pd.cut(data['time'], bins=bins, labels=False)

#     # TODO: Implement bins = None to use the original time values

#     # Group by 'time_bin', 'Brain region', and 'fly' and compute the average and median distance for each group
#     avg_distance = data.groupby(['time_bin', 'Brain region', 'fly'])['distance_ball_0'].mean().reset_index()
#     avg_distance.columns = ['time_bin', 'Brain region', 'fly', 'avg_distance']

#     median_distance = data.groupby(['time_bin', 'Brain region', 'fly'])['distance_ball_0'].median().reset_index()
#     median_distance.columns = ['time_bin', 'Brain region', 'fly', 'median_distance']

#     # Merge the average and median distance dataframes
#     processed_data = avg_distance.merge(median_distance, on=['time_bin', 'Brain region', 'fly'])

#     return processed_data

# def compute_permutation_test(data, metric, n_permutations=1000, show_progress=False, verbose=False):
#     """
#     Improved permutation test using individual datapoints
#     """
#     # Split data without pre-averaging
#     focal_raw = data[data["Brain region"] != "Control"]
#     control_raw = data[data["Brain region"] == "Control"]

#     # Initialize storage for results
#     p_values = []
#     observed_diffs = []
#     time_bins = sorted(data['time_bin'].unique())

#     for time_bin in time_bins:
#         # Get all individual measurements for this time bin
#         focal = focal_raw[focal_raw['time_bin'] == time_bin][metric].values
#         control = control_raw[control_raw['time_bin'] == time_bin][metric].values

#         # Calculate observed difference
#         obs_diff = np.mean(focal) - np.mean(control)
#         observed_diffs.append(obs_diff)

#         # Handle edge cases
#         if len(focal) == 0 or len(control) == 0:
#             p_values.append(1.0)
#             continue

#         # Combined data pool
#         combined = np.concatenate([focal, control])
#         n_focal = len(focal)

#         # Permutation test
#         extreme_count = 0
#         for _ in range(n_permutations):
#             np.random.shuffle(combined)
#             perm_diff = np.mean(combined[:n_focal]) - np.mean(combined[n_focal:])
#             if np.abs(perm_diff) >= np.abs(obs_diff):
#                 extreme_count += 1

#         p_values.append(extreme_count / n_permutations)

#     # Multiple testing correction
#     _, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')

#     # Format results
#     results = {
#         'observed_diff': np.array(observed_diffs),
#         'p_values': np.array(p_values),
#         'p_values_corrected': p_values_corrected,
#         'significant_timepoints_corrected': np.where(p_values_corrected < 0.05)[0],
#         'time_bins': time_bins
#     }

#     if verbose:
#         print("Time Bin | Observed Diff | Raw p-value | Corrected p-value")
#         for i, bin in enumerate(time_bins):
#             print(f"{bin:8} | {observed_diffs[i]:+0.3f} | {p_values[i]:0.4f} | {p_values_corrected[i]:0.4f}")

#     return results


def create_and_save_plot(
    data, nicknames, brain_region, output_path, registries, show_signif=False, show_progress=False, test_nickname=None
):
    # For ID card, we expect a single nickname
    if test_nickname:
        nicknames = [test_nickname]

    # Only one nickname for ID card
    nickname = nicknames[0]
    nickname_data = data[data["Nickname"] == nickname]
    split_value = nickname_data["Split"].iloc[0]
    associated_control = registries["control_nicknames_dict"][split_value]
    control_data = data[data["Nickname"] == associated_control]
    subset_data = pd.concat([nickname_data, control_data])

    # Preprocess data
    processed = Processing.preprocess_data(
        subset_data,
        time_col="time",
        value_col="distance_ball_0",
        group_col="Brain region",
        subject_col="fly",
        bins=12,  # or use data_bins if you want to keep it configurable
    )

    # Permutation test
    if show_signif:
        permutation = Processing.compute_permutation_test(
            processed,
            metric="avg_distance_ball_0",
            group_col="Brain region",
            control_group="Control",
            n_permutations=1000,
            progress=show_progress,
        )
        # Mark significant bins
        processed["Significant"] = processed["time_bin"].isin(
            permutation["time_bins"][permutation["significant_timepoints"]]
        )
    else:
        permutation = None
        processed["Significant"] = False

    # Plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.lineplot(
        data=subset_data,
        x="time",
        y="distance_ball_0",
        hue="Brain region",
        palette=color_dict,
        ax=ax,
    )

    # Draw vertical dotted lines for each time bin
    time_min = subset_data["time"].min()
    time_max = subset_data["time"].max()
    n_bins = processed["time_bin"].nunique()
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)
    for edge in bin_edges:
        ax.axvline(edge, color="gray", linestyle="dotted", alpha=0.5)

    # Annotate significance levels with asterisks
    if show_signif and permutation is not None:
        for idx, tb in enumerate(permutation["time_bins"]):
            bin_start = bin_edges[idx]
            bin_end = bin_edges[idx + 1]
            x_pos = bin_start + (bin_end - bin_start) / 2
            y_pos = subset_data["distance_ball_0"].max() * 0.95

            if idx in permutation["significant_timepoints"]:
                p_value = permutation["p_values_corrected"][idx]
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = ""
                if significance:
                    ax.annotate(significance, xy=(x_pos, y_pos), ha="center", va="bottom", fontsize=16, color="red")

    plt.title(f"Trajectory: {nickname} vs {associated_control}")
    plt.xlabel("Time (s)")
    plt.ylabel("Ball distance from start (px)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_control_plot(data, control_nicknames, output_path):
    """
    Create and save a plot for all control nicknames together.

    Args:
    data (pd.DataFrame): The input data containing 'Brain region', 'time', and 'distance_ball_0'.
    control_nicknames (list): List of control nicknames.
    output_path (str): Path to save the output plot.
    """
    # Subset data for all control nicknames
    control_data = data[data["Nickname"].isin(control_nicknames)]

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=control_data, x="time", y="distance_ball_0", hue="Nickname")

    plt.title("Control Brain Region")
    plt.xlabel("Time (s)")
    plt.ylabel("Ball distance from start (px)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory


# Function to create and save KDE and ECDF plots
def create_and_save_kde_ecdf_plot(data, nicknames, brain_region, output_path, registries):
    """
    Create and save a KDE and ECDF plot for a single nickname and its control.
    Always produces two subplots: one for KDE, one for ECDF.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # For ID card, we expect a single nickname
    nickname = nicknames[0]
    nickname_data = data[data["Nickname"] == nickname]
    split_value = nickname_data["Split"].iloc[0]
    associated_control = registries["control_nicknames_dict"][split_value]
    control_data = data[data["Nickname"] == associated_control]
    subset_data = pd.concat([nickname_data, control_data])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # KDE Plot
    sns.histplot(
        data=subset_data,
        x="start_time",
        hue="Brain region",
        ax=axes[0],
        kde=True,
        element="step",
        palette=color_dict,
    )
    axes[0].set_title(f"KDE: {nickname} vs {associated_control}", fontsize=12)
    axes[0].set_xlim(0, 3600)
    axes[0].tick_params(labelsize=10)

    # ECDF Plot
    sns.histplot(
        data=subset_data,
        x="start_time",
        hue="Brain region",
        ax=axes[1],
        palette=color_dict,
        cumulative=True,
        element="step",
        kde=True,
    )
    axes[1].set_title(f"Cumulative: {nickname} vs {associated_control}", fontsize=12)
    axes[1].set_xlim(0, 3600)
    axes[1].tick_params(labelsize=10)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_and_save_hist_kde_rawdisp_plot(data, nicknames, brain_region, output_path, registries, color_dict):
    # Filter data to include only raw_displacement values above 1
    data = data[data["raw_displacement"] > 5]

    if brain_region == "Control":
        # Special case: Plot all controls together
        fig, ax = plt.subplots(figsize=(10, 6))

        # Subset data for all control nicknames
        control_data = data[data["Nickname"].isin(nicknames)]

        # Histogram and KDE Plot
        sns.histplot(
            data=control_data,
            x="raw_displacement",
            hue="Nickname",
            bins=100,
            kde=True,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title("Raw Displacement for All Controls", fontsize=12)
        ax.set_xlabel("Raw Displacement (px)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 50)
        ax.tick_params(labelsize=10)

    else:
        # Default behavior for other brain regions
        pairs_per_row = 3
        n_nicknames = len(nicknames)
        n_rows = math.ceil(n_nicknames / pairs_per_row)

        subplot_size = (10, 6)  # Adjusted size for each subplot
        fig_width, fig_height = subplot_size[0] * pairs_per_row, subplot_size[1] * n_rows

        fig, axes = plt.subplots(n_rows, pairs_per_row, figsize=(fig_width, fig_height), squeeze=False)

        for i, nickname in enumerate(nicknames):
            row = i // pairs_per_row
            col = i % pairs_per_row

            nickname_data = data[data["Nickname"] == nickname]
            split_value = nickname_data["Split"].iloc[0]
            associated_control = registries["control_nicknames_dict"][split_value]
            control_data = data[data["Nickname"] == associated_control]
            subset_data = pd.concat([nickname_data, control_data])

            # Histogram and KDE Plot
            sns.histplot(
                data=subset_data,
                x="raw_displacement",
                hue="Brain region",
                palette=color_dict,
                bins=100,
                kde=True,
                stat="density",
                common_norm=False,
                element="step",
                ax=axes[row, col],
            )
            axes[row, col].set_title(f"Raw Displacement: {nickname}\nvs {associated_control}", fontsize=10)
            axes[row, col].set_xlabel("Raw Displacement (px)")
            axes[row, col].set_ylabel("Frequency")
            axes[row, col].set_xlim(0, 50)
            axes[row, col].tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory


def create_and_save_contact_rate_plot(data, nicknames, brain_region, output_path, registries, color_dict):
    # Define the time window (e.g., 100 seconds)
    time_window = 100

    # Determine the maximum time in the dataset
    max_time = data["start"].max()

    # Create bins for the time windows
    bins = np.arange(0, max_time + time_window, time_window)

    # Create a new column 'time_window' that indicates the time window each 'start' time belongs to
    data["time_window"] = pd.cut(data["start"], bins=bins, right=False, labels=bins[:-1])

    if brain_region == "Control":
        # Special case: Plot all controls together
        fig, ax = plt.subplots(figsize=(12, 6))

        # Subset data for all control nicknames
        control_data = data[data["Nickname"].isin(nicknames)]

        # Group by 'time_window', 'Nickname', and 'fly' and count the number of unique contacts in each time window
        contact_rate = control_data.groupby(["time_window", "Nickname", "fly"])["identifier"].nunique().reset_index()
        contact_rate.columns = ["time_window", "Nickname", "fly", "contact_count"]

        # Plot the contact rate over time with confidence intervals
        sns.lineplot(data=contact_rate, x="time_window", y="contact_count", hue="Nickname", marker="o", ax=ax, ci="sd")
        ax.set_title("Contact Rate Over Time: All Controls", fontsize=12)
        ax.set_xlabel("Time Window (s)")
        ax.set_ylabel("Number of Unique Contacts")
        ax.tick_params(labelsize=10)

    else:
        # Default behavior for other brain regions
        pairs_per_row = 3
        n_nicknames = len(nicknames)
        n_rows = math.ceil(n_nicknames / pairs_per_row)

        subplot_size = (10, 6)  # Adjusted size for each subplot
        fig_width, fig_height = subplot_size[0] * pairs_per_row, subplot_size[1] * n_rows

        fig, axes = plt.subplots(n_rows, pairs_per_row, figsize=(fig_width, fig_height), squeeze=False)

        for i, nickname in enumerate(nicknames):
            row = i // pairs_per_row
            col = i % pairs_per_row

            nickname_data = data[data["Nickname"] == nickname]
            split_value = nickname_data["Split"].iloc[0]
            associated_control = registries["control_nicknames_dict"][split_value]
            control_data = data[data["Nickname"] == associated_control]
            subset_data = pd.concat([nickname_data, control_data])

            # Group by 'time_window', 'Brain region', and 'fly' and count the number of unique contacts in each time window
            contact_rate = (
                subset_data.groupby(["time_window", "Brain region", "fly"])["identifier"].nunique().reset_index()
            )
            contact_rate.columns = ["time_window", "Brain region", "fly", "contact_count"]

            # Plot the contact rate over time with confidence intervals
            sns.lineplot(
                data=contact_rate,
                x="time_window",
                y="contact_count",
                hue="Brain region",
                marker="o",
                palette=color_dict,
                ci="sd",
                ax=axes[row, col],
            )
            axes[row, col].set_title(f"{nickname} vs {associated_control}", fontsize=10)
            axes[row, col].set_xlabel("Time Window (s)")
            axes[row, col].set_ylabel("Number of Unique Contacts")
            axes[row, col].tick_params(labelsize=8)
            axes[row, col].legend(title="Brain region", fontsize=8, title_fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory
