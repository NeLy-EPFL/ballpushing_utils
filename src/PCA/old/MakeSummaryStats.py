import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import Config

# Load SplitRegistry from Config
split_registry = Config.SplitRegistry

# Find all relevant CSVs
csv_files = glob.glob("*results_allmethods*.csv")
if not csv_files:
    print("No CSV files found matching the pattern '*results_allmethods*.csv'.")
    exit(1)
print(f"Found {len(csv_files)} CSV files to process.")

# Read all CSVs into a list of DataFrames, adding a column for the method
dfs = []
for fname in csv_files:
    df = pd.read_csv(fname)
    base = os.path.splitext(os.path.basename(fname))[0]
    suffix = base.replace("results_allmethods", "").replace("_", "")
    cols_to_rename = {col: f"{col}_{suffix}" for col in df.columns if col not in ["Nickname", "Control"]}
    df = df.rename(columns=cols_to_rename)
    dfs.append(df)

# Merge all DataFrames on Nickname and Control
from functools import reduce
summary_df = reduce(lambda left, right: pd.merge(left, right, on=["Nickname", "Control"], how="outer"), dfs)

# Find all columns that indicate significance
sig_cols = [col for col in summary_df.columns if "_FDR_significant" in col or "MannWhitney_any_dim_significant" in col]
# Ensure all significance columns are boolean (convert if needed)
for col in sig_cols:
    summary_df[col] = summary_df[col].astype(bool)

# Compute significance score: count number of True across all significance columns for each row
summary_df["Significance_score"] = summary_df[sig_cols].sum(axis=1)

# For each Nickname, keep the row with the highest significance score (if duplicate Nicknames)
summary_df = summary_df.sort_values("Significance_score", ascending=False)
summary_df = summary_df.groupby("Nickname", as_index=False).first()

# Map Nickname to Brain region
nickname_to_brainregion = split_registry.set_index("Nickname")["Simplified region"].to_dict()
summary_df["Brain region"] = summary_df["Nickname"].map(nickname_to_brainregion)

# Save summary
summary_df.to_csv("summary_hits_allmethods.csv", index=False)
print(f"Summary saved as summary_hits_allmethods.csv")

# Use Config.color_dict for coloring
color_dict = Config.color_dict

plot_df = summary_df[["Nickname", "Significance_score", "Brain region"]].copy()
plot_df = plot_df.sort_values("Significance_score", ascending=False)

plt.figure(figsize=(12, max(6, len(plot_df) * 0.4)))
bar_colors = plot_df["Brain region"].map(color_dict).fillna("#888888")

bars = plt.barh(
    plot_df["Nickname"],
    plot_df["Significance_score"],
    color=bar_colors
)
plt.xlabel("Significance score (number of significant tests)")
plt.ylabel("Nickname")
plt.title("Significance score per Nickname (colored by Brain region)")
plt.gca().invert_yaxis()
plt.tight_layout()

# Add legend for brain regions
from matplotlib.patches import Patch
handles = [Patch(color=color, label=region) for region, color in color_dict.items() if region in plot_df["Brain region"].unique()]
plt.legend(handles=handles, title="Brain region", bbox_to_anchor=(1.01, 1), loc="upper left")

plt.savefig("nickname_significance_score_barplot_brainregion.png", dpi=200, bbox_inches="tight")
plt.show()