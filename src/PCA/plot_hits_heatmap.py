import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
import argparse
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Ballpushing_utils")))
from utilities import brain_regions_path

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate hits heatmap with control mode selection")
parser.add_argument("output_dir", nargs="?", default=".", help="Output directory for results")
parser.add_argument(
    "--control-mode",
    type=str,
    choices=["tailored", "emptysplit", "tnt_pr"],
    default="tailored",
    help="Control selection mode (default: tailored)",
)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
CONTROL_MODE = args.control_mode
DATA_FILES_DIR = os.path.join(OUTPUT_DIR, "data_files")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

print(f"🎯 Using output directory: {OUTPUT_DIR}")
print(f"📁 Data files directory: {DATA_FILES_DIR}")
print(f"🎨 Plots directory: {PLOTS_DIR}")
if CONTROL_MODE == "tailored":
    ctrl_desc = "Tailored controls (split-based)"
elif CONTROL_MODE == "emptysplit":
    ctrl_desc = "Empty-Split (universal control)"
elif CONTROL_MODE == "tnt_pr":
    ctrl_desc = "TNTxPR (compare all genotypes vs TNTxPR)"
else:
    ctrl_desc = CONTROL_MODE

print(f"🎛️  Control mode: {ctrl_desc}")

# Configuration: Choose which analysis types to include
# Options: "both", "static", "temporal"
ANALYSIS_TYPE = "static"  # Change this to "both" or "temporal" as needed


def detect_most_recent_pca_method():
    """Detect the most recent PCA method used based on file timestamps"""
    # Determine control suffix based on control mode
    if CONTROL_MODE == "tailored":
        control_suffix = "tailored*"
    elif CONTROL_MODE == "emptysplit":
        control_suffix = "emptysplit*"
    elif CONTROL_MODE == "tnt_pr":
        control_suffix = "tnt_pr*"
    else:
        control_suffix = "tailored*"

    # Try data_files subdirectory first
    pca_files = glob.glob(os.path.join(DATA_FILES_DIR, f"static_pca_stats_results_allmethods*{control_suffix}.csv"))
    sparsepca_files = glob.glob(
        os.path.join(DATA_FILES_DIR, f"static_sparsepca_stats_results_allmethods*{control_suffix}.csv")
    )

    # Fallback to OUTPUT_DIR root if not found in data_files
    if not pca_files and not sparsepca_files:
        pca_files = glob.glob(os.path.join(OUTPUT_DIR, f"static_pca_stats_results_allmethods*{control_suffix}.csv"))
        sparsepca_files = glob.glob(
            os.path.join(OUTPUT_DIR, f"static_sparsepca_stats_results_allmethods*{control_suffix}.csv")
        )

    all_files = [(f, "pca") for f in pca_files] + [(f, "sparsepca") for f in sparsepca_files]

    if not all_files:
        print(f"No PCA result files found for control mode: {CONTROL_MODE}")
        return "pca", []

    # Sort by modification time (most recent first)
    all_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    most_recent_file, method = all_files[0]

    print(f"Most recent PCA method detected: {method.upper()}")
    print(f"Using file: {most_recent_file}")

    # Return all files of the most recent method
    method_files = [f for f, m in all_files if m == method]
    return method, method_files


# Detect most recent PCA method and get appropriate files
pca_method, method_files = detect_most_recent_pca_method()

# Find CSVs for static and temporal based on detected method and control mode
if CONTROL_MODE == "tailored":
    control_suffix = "tailored*"
elif CONTROL_MODE == "emptysplit":
    control_suffix = "emptysplit*"
elif CONTROL_MODE == "tnt_pr":
    control_suffix = "tnt_pr*"
else:
    control_suffix = "tailored*"
if pca_method == "sparsepca":
    pattern = os.path.join(DATA_FILES_DIR, f"*_sparsepca_*allmethods*{control_suffix}.csv")
    # Fallback to OUTPUT_DIR root
    if not glob.glob(pattern):
        pattern = os.path.join(OUTPUT_DIR, f"*_sparsepca_*allmethods*{control_suffix}.csv")
else:
    pattern = os.path.join(DATA_FILES_DIR, f"*_pca_*allmethods*{control_suffix}.csv")
    # Fallback to OUTPUT_DIR root
    if not glob.glob(pattern):
        pattern = os.path.join(OUTPUT_DIR, f"*_pca_*allmethods*{control_suffix}.csv")

tailored_csvs = glob.glob(pattern)
print(f"Found CSV files matching pattern '{pattern}': {tailored_csvs}")

# Filter based on ANALYSIS_TYPE configuration
if ANALYSIS_TYPE == "static":
    filtered_csvs = [f for f in tailored_csvs if "static" in f.lower()]
elif ANALYSIS_TYPE == "temporal":
    filtered_csvs = [f for f in tailored_csvs if "temporal" in f.lower()]
elif ANALYSIS_TYPE == "both":
    filtered_csvs = [f for f in tailored_csvs if ("static" in f.lower() or "temporal" in f.lower())]
else:
    raise ValueError("ANALYSIS_TYPE must be 'static', 'temporal', or 'both'")

print(f"Filtered CSV files for analysis type '{ANALYSIS_TYPE}': {filtered_csvs}")


# Map for pretty column names
def get_pca_label(fname):
    if "static" in fname.lower():
        return "Static"
    elif "temporal" in fname.lower():
        return "Temporal"
    else:
        return os.path.basename(fname)


# Collect all unique Nicknames
all_nicknames = set()
for fname in filtered_csvs:
    df = pd.read_csv(fname)
    all_nicknames.update(df["Nickname"].tolist())
all_nicknames = sorted(all_nicknames)

# Build DataFrame: rows=Nicknames, columns=Static/Temporal
columns = [get_pca_label(f) for f in filtered_csvs]
heatmap_df = pd.DataFrame(index=all_nicknames, columns=columns)

for fname in filtered_csvs:
    col_label = get_pca_label(fname)
    df = pd.read_csv(fname)
    df = df.set_index("Nickname")
    for nickname in all_nicknames:
        val = df["Permutation_pval"].get(nickname, 1.0)
        if pd.isna(val):
            val = 1.0
        heatmap_df.loc[nickname, col_label] = val

heatmap_df = heatmap_df.astype(float)

print(heatmap_df.columns)

print(heatmap_df.head())

# Remove duplicate columns by name, keeping only the first occurrence
heatmap_df = heatmap_df.loc[:, ~heatmap_df.columns.duplicated()]
print("Columns after removing duplicates by name:", heatmap_df.columns)
print(heatmap_df.head())
print(heatmap_df.shape)

# Load region map and build nickname to simplified mapping
region_map = pd.read_csv(brain_regions_path)
# No mapping to simplified nicknames; keep regular Nicknames as index
print("Heatmap DataFrame with regular Nicknames:")
print(heatmap_df.head())

# --- Print missing nicknames from registry before plotting ---
try:
    from Config import registries, SplitRegistry, color_dict

    registry_nicknames = set(registries["nicknames"])
    heatmap_nicknames = set(heatmap_df.index)
    missing_nicknames = registry_nicknames - heatmap_nicknames
    if missing_nicknames:
        print("Missing nicknames from registry (not included in plot):", missing_nicknames)
    else:
        print("All registry nicknames are included in the plot.")
    # Build a mapping from Nickname to Brain region
    nickname_to_brainregion = dict(zip(SplitRegistry["Nickname"], SplitRegistry["Simplified region"]))
except Exception as e:
    print(f"Could not check missing nicknames from registry or build color mapping: {e}")
    color_dict = {}
    nickname_to_brainregion = {}

# Sort by ascending p-value based on available columns
if "Static" in heatmap_df.columns:
    heatmap_df = heatmap_df.sort_values(by=["Static"], ascending=[True])
elif "Temporal" in heatmap_df.columns:
    heatmap_df = heatmap_df.sort_values(by=["Temporal"], ascending=[True])

# Categorize flies based on significance and available analysis types
no_hit = []
static_only = []
temporal_only = []
both = []

for idx, row in heatmap_df.iterrows():
    static_sig = False
    temporal_sig = False

    if "Static" in heatmap_df.columns:
        try:
            static_val = heatmap_df.at[idx, "Static"]
            static_sig = static_val < 0.05 and not pd.isna(static_val)
        except:
            static_sig = False

    if "Temporal" in heatmap_df.columns:
        try:
            temporal_val = heatmap_df.at[idx, "Temporal"]
            temporal_sig = temporal_val < 0.05 and not pd.isna(temporal_val)
        except:
            temporal_sig = False

    if ANALYSIS_TYPE == "both":
        if static_sig and temporal_sig:
            both.append(idx)
        elif static_sig:
            static_only.append(idx)
        elif temporal_sig:
            temporal_only.append(idx)
        else:
            no_hit.append(idx)
    elif ANALYSIS_TYPE == "static":
        if static_sig:
            static_only.append(idx)
        else:
            no_hit.append(idx)
    elif ANALYSIS_TYPE == "temporal":
        if temporal_sig:
            temporal_only.append(idx)
        else:
            no_hit.append(idx)

# Filter all subsets to only include Nicknames present in the DataFrame
all_valid_nicknames = set(heatmap_df.index)
both = [n for n in both if n in all_valid_nicknames]
static_only = [n for n in static_only if n in all_valid_nicknames]
temporal_only = [n for n in temporal_only if n in all_valid_nicknames]
no_hit = [n for n in no_hit if n in all_valid_nicknames]

# Sort the 'activity hits' (temporal_only) subset by ascending Temporal p-value
if "Temporal" in heatmap_df.columns:
    temporal_only_sorted = sorted(
        temporal_only,
        key=lambda n: float(heatmap_df.at[n, "Temporal"]) if not pd.isna(heatmap_df.at[n, "Temporal"]) else 1.0,
    )
else:
    temporal_only_sorted = temporal_only
# Update the subset for plotting based on analysis type
subsets = []  # Initialize subsets
if ANALYSIS_TYPE == "both":
    subsets = [
        (both, "Ball pushing & activity hits"),
        (static_only, "Ball pushing hits"),
        (temporal_only_sorted, "Activity hits"),
        (no_hit, "No hit"),
    ]
elif ANALYSIS_TYPE == "static":
    subsets = [
        (static_only, "Ball pushing hits"),
        (no_hit, "No hit"),
    ]
elif ANALYSIS_TYPE == "temporal":
    subsets = [
        (temporal_only_sorted, "Activity hits"),
        (no_hit, "No hit"),
    ]

# Adjust subplot layout based on number of subsets
n_subsets = len(subsets)
if n_subsets <= 2:
    n_rows, n_cols = 1, 2
else:
    n_rows, n_cols = 2, 2
fig1, axes1 = plt.subplots(
    n_rows,
    n_cols,
    figsize=(8.5, max(6, max(len(sub) for sub, _ in subsets) * 0.3)),  # slightly wider for cbar
    sharex=True,
    constrained_layout=True,
)
axes1 = axes1.flatten()
for ax, (subset, title) in zip(axes1, subsets):
    if subset:
        hm = sns.heatmap(
            heatmap_df.loc[subset],
            cmap="viridis_r",
            cbar=False,  # No colorbar for individual axes
            linewidths=0.5,
            linecolor="gray",
            vmin=0,
            vmax=1,
            annot=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Nickname")
        for y, nickname in enumerate(heatmap_df.loc[subset].index):
            for x, col in enumerate(heatmap_df.columns):
                pval = heatmap_df.at[nickname, col]
                try:
                    pval_float = float(pval) if not pd.isna(pval) else 1.0
                except:
                    pval_float = 1.0
                if pval_float < 0.001:
                    annotation = "***"
                elif pval_float < 0.01:
                    annotation = "**"
                elif pval_float < 0.05:
                    annotation = "*"
                else:
                    annotation = ""
                if annotation:
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        annotation,
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=10,
                        fontweight="bold",
                    )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        # Color yticklabels by brain region
        for ticklabel in ax.get_yticklabels():
            nickname = ticklabel.get_text()
            region = nickname_to_brainregion.get(nickname, None)
            if region and region in color_dict:
                ticklabel.set_color(color_dict[region])
    else:
        ax.axis("off")
# Add a single colorbar for all subplots (regular scale)
sm1 = plt.cm.ScalarMappable(cmap="viridis_r", norm=Normalize(vmin=0, vmax=1))
sm1.set_array([])
cbar1 = fig1.colorbar(sm1, ax=axes1, orientation="vertical", fraction=0.03, pad=0.04)
cbar1.set_label("p-value", fontsize=12, fontweight="bold")
cbar1.set_ticks([0, 0.01, 0.05, 0.1, 0.5, 1])
cbar1.set_ticklabels(["0", "0.01", "0.05", "0.1", "0.5", "1"])
fig1.text(0.5, 0.01, "PCA Type", ha="center", va="bottom", fontsize=12)  # lower x-axis title
# Reduce ytick label fontsize
for ax in axes1:
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

# Generate output filename based on analysis type
filename_suffix = ANALYSIS_TYPE if ANALYSIS_TYPE != "both" else "static_temporal"
png_path = os.path.join(PLOTS_DIR, f"mannwhitney_{filename_suffix}_tailored_split.png")
pdf_path = os.path.join(PLOTS_DIR, f"mannwhitney_{filename_suffix}_tailored_split.pdf")
fig1.savefig(png_path, dpi=200)
fig1.savefig(pdf_path, dpi=200)
plt.close(fig1)

# --- Log scale plot ---
pval_min = 0.001  # set minimum p-value for log scale to avoid large white space
log_heatmap_df = -np.log10(heatmap_df.clip(lower=pval_min))
# Ensure log_heatmap_df is a DataFrame
if not isinstance(log_heatmap_df, pd.DataFrame):
    log_heatmap_df = pd.DataFrame(log_heatmap_df, index=heatmap_df.index, columns=heatmap_df.columns)

pval_ticks = [1, 0.5, 0.1, 0.05, 0.01, 0.001]
log_ticks = [-np.log10(p) for p in pval_ticks]
log_ticklabels = [str(p) for p in pval_ticks]

fig2, axes2 = plt.subplots(
    n_rows,
    n_cols,
    figsize=(8.5, max(6, max(len(sub) for sub, _ in subsets) * 0.3)),
    sharex=True,
    constrained_layout=True,
)
axes2 = axes2.flatten()
for ax, (subset, title) in zip(axes2, subsets):
    if subset:
        hm = sns.heatmap(
            log_heatmap_df.loc[subset],
            cmap="viridis",
            cbar=False,  # No colorbar for individual axes
            linewidths=0.5,
            linecolor="gray",
            vmin=0,
            vmax=-np.log10(pval_min),
            annot=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Nickname")
        for y, nickname in enumerate(heatmap_df.loc[subset].index):
            for x, col in enumerate(heatmap_df.columns):
                pval = heatmap_df.at[nickname, col]
                try:
                    pval_float = float(pval) if not pd.isna(pval) else 1.0
                except:
                    pval_float = 1.0
                if pval_float < 0.001:
                    annotation = "***"
                elif pval_float < 0.01:
                    annotation = "**"
                elif pval_float < 0.05:
                    annotation = "*"
                else:
                    annotation = ""
                if annotation:
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        annotation,
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=10,
                        fontweight="bold",
                    )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        # Color yticklabels by brain region
        for ticklabel in ax.get_yticklabels():
            nickname = ticklabel.get_text()
            region = nickname_to_brainregion.get(nickname, None)
            if region and region in color_dict:
                ticklabel.set_color(color_dict[region])
    else:
        ax.axis("off")
# Add a single colorbar for all subplots (log scale)
sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=-np.log10(pval_min)))
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=axes2, orientation="vertical", fraction=0.03, pad=0.04)
cbar2.set_label("-log10(p-value)", fontsize=12, fontweight="bold")
cbar2.set_ticks(log_ticks)
cbar2.set_ticklabels(log_ticklabels)
fig2.text(0.5, 0.01, "PCA Type", ha="center", va="bottom", fontsize=12)
for ax in axes2:
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
png_log_path = os.path.join(PLOTS_DIR, f"mannwhitney_{filename_suffix}_tailored_split_log.png")
pdf_log_path = os.path.join(PLOTS_DIR, f"mannwhitney_{filename_suffix}_tailored_split_log.pdf")
fig2.savefig(png_log_path, dpi=200)
fig2.savefig(pdf_log_path, dpi=200)
plt.close(fig2)
