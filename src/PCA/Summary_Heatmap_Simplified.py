import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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

# Load region map and build nickname to simplified mapping
region_map = pd.read_csv(brain_regions_path)
nickname_to_simplified = dict(zip(region_map["Nickname"], region_map["Simplified Nickname"]))

# Replace heatmap_df index with Simplified Nicknames (fallback to original if not found)
heatmap_df.index = pd.Index([nickname_to_simplified.get(n, n) for n in heatmap_df.index])

# Custom sort: ascending for Static, descending for Temporal
if "Static" in heatmap_df.columns and "Temporal" in heatmap_df.columns:
    heatmap_df = heatmap_df.sort_values(by=["Static", "Temporal"], ascending=[True, False])

# Order Nicknames by ascending sum of p-values
nickname_order = heatmap_df.sum(axis=1).sort_values().index
heatmap_df = heatmap_df.loc[nickname_order]

plt.figure(figsize=(4, max(6, len(heatmap_df) * 0.3)))
ax = sns.heatmap(heatmap_df, cmap="viridis_r", cbar=True, linewidths=0.5, linecolor="gray", vmin=0, vmax=1, annot=False)
plt.title("Mann-Whitney p-values (Static & Temporal, Tailored Controls)")
plt.xlabel("PCA Type")
plt.ylabel("Nickname")

# Annotate significant p-values
for y, nickname in enumerate(heatmap_df.index):
    for x, col in enumerate(heatmap_df.columns):
        pval = heatmap_df.iloc[y, x]
        if pval < 0.001:
            annotation = "***"
        elif pval < 0.01:
            annotation = "**"
        elif pval < 0.05:
            annotation = "*"
        else:
            annotation = ""
        if annotation:
            ax.text(x + 0.5, y + 0.5, annotation, ha="center", va="center", color="red", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("mannwhitney_static_temporal_tailored.png", dpi=200)
plt.savefig("mannwhitney_static_temporal_tailored.pdf", dpi=200)
plt.close()
