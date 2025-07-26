import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Ballpushing_utils")))
from utilities import brain_regions_path

# Find tailored CSVs for static and temporal only
tailored_csvs = glob.glob("*_allmethods_tailored*.csv")
filtered_csvs = [f for f in tailored_csvs if ("static" in f.lower() or "temporal" in f.lower())]


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

# Sort by ascending Static p-value only
if "Static" in heatmap_df.columns:
    heatmap_df = heatmap_df.sort_values(by=["Static"], ascending=[True])

# Categorize flies based on significance
no_hit = []
static_only = []
temporal_only = []
both = []
for idx, row in heatmap_df.iterrows():
    static_sig = row["Static"] < 0.05 if "Static" in row else False
    temporal_sig = row["Temporal"] < 0.05 if "Temporal" in row else False
    if static_sig and temporal_sig:
        both.append(idx)
    elif static_sig:
        static_only.append(idx)
    elif temporal_sig:
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
        temporal_only, key=lambda n: float(pd.to_numeric(heatmap_df.loc[n, "Temporal"], errors="coerce"))
    )
else:
    temporal_only_sorted = temporal_only
# Update the subset for plotting
subsets = [
    (both, "Ball pushing & activity hits"),
    (static_only, "Ball pushing hits"),
    (temporal_only_sorted, "Activity hits"),
    (no_hit, "No hit"),
]
# --- Regular scale plot ---
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
                pval = heatmap_df.loc[nickname, col]
                pval_float = pd.to_numeric(pval, errors="coerce")
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
fig1.savefig("mannwhitney_static_temporal_tailored_split.png", dpi=200)
fig1.savefig("mannwhitney_static_temporal_tailored_split.pdf", dpi=200)
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
                pval = heatmap_df.loc[nickname, col]
                pval_float = pd.to_numeric(pval, errors="coerce")
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
fig2.savefig("mannwhitney_static_temporal_tailored_split_log.png", dpi=200)
fig2.savefig("mannwhitney_static_temporal_tailored_split_log.pdf", dpi=200)
plt.close(fig2)
