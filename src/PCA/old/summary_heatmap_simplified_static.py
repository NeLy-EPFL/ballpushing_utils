import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize

# Load the simplified static PCA results
df = pd.read_csv("static_pca_stats_simplified_static_tailoredctrls.csv")

# Extract minimum corrected p-value across all PCs for each genotype
pc_pval_cols = [col for col in df.columns if "pval_corrected" in col]
print(f"Found corrected p-value columns: {pc_pval_cols}")

# Calculate minimum p-value across all PCs for each genotype
df["min_pval"] = df[pc_pval_cols].min(axis=1)

# Create a DataFrame for the heatmap with genotype as index
heatmap_df = pd.DataFrame({"Static (Simplified)": df["min_pval"].values}, index=df["genotype"])

print(f"Heatmap DataFrame shape: {heatmap_df.shape}")
print(f"Number of significant genotypes (p < 0.05): {(heatmap_df['Static (Simplified)'] < 0.05).sum()}")

# Try to load Config for brain region coloring (optional)
try:
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Ballpushing_utils")))
    from PCA.Config import registries, SplitRegistry, color_dict

    # Try to find genotype to nickname mapping
    genotype_to_nickname = {}
    if "Genotype" in SplitRegistry.columns and "Nickname" in SplitRegistry.columns:
        genotype_to_nickname = dict(zip(SplitRegistry["Genotype"], SplitRegistry["Nickname"]))

    # Map genotypes to nicknames where possible
    nickname_index = []
    for genotype in heatmap_df.index:
        nickname = genotype_to_nickname.get(genotype, genotype)
        nickname_index.append(nickname)

    # Create new DataFrame with nickname index
    heatmap_df = pd.DataFrame({"Static (Simplified)": heatmap_df["Static (Simplified)"].values}, index=nickname_index)

    # Build a mapping from Nickname to Brain region for coloring
    nickname_to_brainregion = dict(zip(SplitRegistry["Nickname"], SplitRegistry["Simplified region"]))

    print(f"Successfully loaded Config mappings. Color dict has {len(color_dict)} regions.")

except Exception as e:
    print(f"Could not load Config mappings: {e}")
    color_dict = {}
    nickname_to_brainregion = {}

# Sort by ascending p-value
heatmap_df = heatmap_df.sort_values(by=["Static (Simplified)"], ascending=[True])

# Categorize flies based on significance
significant = []
no_hit = []

for idx, row in heatmap_df.iterrows():
    static_val = heatmap_df.at[idx, "Static (Simplified)"]
    static_sig = static_val < 0.05 and not pd.isna(static_val)

    if static_sig:
        significant.append(idx)
    else:
        no_hit.append(idx)

print(f"Significant hits: {len(significant)}")
print(f"No hits: {len(no_hit)}")

# Create subsets for plotting
subsets = [
    (significant, "Ball pushing hits (Simplified)"),
    (no_hit, "No hit"),
]

# Create the plot
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, max(6, max(len(sub) for sub, _ in subsets) * 0.3)),
    sharex=True,
    constrained_layout=True,
)

for ax, (subset, title) in zip(axes, subsets):
    if subset:
        hm = sns.heatmap(
            heatmap_df.loc[subset],
            cmap="viridis_r",
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
            vmin=0,
            vmax=1,
            annot=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Genotype/Nickname")

        # Add significance stars
        for y, nickname in enumerate(heatmap_df.loc[subset].index):
            pval = heatmap_df.at[nickname, "Static (Simplified)"]
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
                    0.5,  # x position (single column)
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
cbar1 = fig.colorbar(sm1, ax=axes, orientation="vertical", fraction=0.03, pad=0.04)
cbar1.set_label("Minimum corrected p-value", fontsize=12, fontweight="bold")
cbar1.set_ticks([0, 0.01, 0.05, 0.1, 0.5, 1])
cbar1.set_ticklabels(["0", "0.01", "0.05", "0.1", "0.5", "1"])

fig.text(0.5, 0.01, "PCA Type", ha="center", va="bottom", fontsize=12)

# Reduce ytick label fontsize
for ax in axes:
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

# Save the plot
fig.savefig("mannwhitney_simplified_static_tailored_split.png", dpi=200, bbox_inches="tight")
fig.savefig("mannwhitney_simplified_static_tailored_split.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)

# --- Log scale plot ---
pval_min = 0.001
log_heatmap_df = -np.log10(heatmap_df.clip(lower=pval_min))

pval_ticks = [1, 0.5, 0.1, 0.05, 0.01, 0.001]
log_ticks = [-np.log10(p) for p in pval_ticks]
log_ticklabels = [str(p) for p in pval_ticks]

fig2, axes2 = plt.subplots(
    1,
    2,
    figsize=(10, max(6, max(len(sub) for sub, _ in subsets) * 0.3)),
    sharex=True,
    constrained_layout=True,
)

for ax, (subset, title) in zip(axes2, subsets):
    if subset:
        hm = sns.heatmap(
            log_heatmap_df.loc[subset],
            cmap="viridis",
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
            vmin=0,
            vmax=-np.log10(pval_min),
            annot=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Genotype/Nickname")

        # Add significance stars
        for y, nickname in enumerate(heatmap_df.loc[subset].index):
            pval = heatmap_df.at[nickname, "Static (Simplified)"]
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
                    0.5,
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
cbar2.set_label("-log10(minimum corrected p-value)", fontsize=12, fontweight="bold")
cbar2.set_ticks(log_ticks)
cbar2.set_ticklabels(log_ticklabels)

fig2.text(0.5, 0.01, "PCA Type", ha="center", va="bottom", fontsize=12)

for ax in axes2:
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

fig2.savefig("mannwhitney_simplified_static_tailored_split_log.png", dpi=200, bbox_inches="tight")
fig2.savefig("mannwhitney_simplified_static_tailored_split_log.pdf", dpi=200, bbox_inches="tight")
plt.close(fig2)

print("Summary heatmaps saved:")
print("- mannwhitney_simplified_static_tailored_split.png")
print("- mannwhitney_simplified_static_tailored_split.pdf")
print("- mannwhitney_simplified_static_tailored_split_log.png")
print("- mannwhitney_simplified_static_tailored_split_log.pdf")

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"Total genotypes analyzed: {len(heatmap_df)}")
print(f"Significant hits (p < 0.05): {len(significant)}")
print(f"Non-significant: {len(no_hit)}")
print(f"Percentage significant: {len(significant)/len(heatmap_df)*100:.1f}%")

# Print the most significant hits
print(f"\nTop 10 most significant hits:")
top_hits = heatmap_df.head(10)
for i, (genotype, row) in enumerate(top_hits.iterrows(), 1):
    pval_value = row["Static (Simplified)"]
    if hasattr(pval_value, "iloc"):
        pval_float = float(pval_value.iloc[0])
    else:
        pval_float = float(pval_value)
    stars = "***" if pval_float < 0.001 else "**" if pval_float < 0.01 else "*" if pval_float < 0.05 else ""
    print(f"{i:2d}. {genotype:<40} p={pval_float:.6f} {stars}")
