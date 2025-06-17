import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Find all relevant CSVs
tailored_csvs = glob.glob("*_allmethods_tailored*.csv")
emptysplit_csvs = glob.glob("*_allmethods_emptysplit*.csv")
all_csvs = tailored_csvs + emptysplit_csvs

# test_types = [
#     ("MannWhitney_any_dim_significant", "Mann-Whitney"),
#     ("Permutation_FDR_significant", "Permutation"),
#     ("Mahalanobis_FDR_significant", "Mahalanobis"),
# ]

test_types = [
    ("Permutation_pval", "Mann-Whitney (p)"),
    ("Permutation_FDR_pval", "Permutation (FDR p)"),
    ("Mahalanobis_FDR_pval", "Mahalanobis (FDR p)"),
]

# Collect all unique Nicknames
all_nicknames = set()
for fname in all_csvs:
    df = pd.read_csv(fname)
    all_nicknames.update(df["Nickname"].tolist())
all_nicknames = sorted(all_nicknames)

# Build a MultiIndex DataFrame
columns = []
for fname in all_csvs:
    csv_label = os.path.basename(fname)
    for _, test_label in test_types:
        columns.append((csv_label, test_label))
multi_columns = pd.MultiIndex.from_tuples(columns, names=["CSV", "Test"])

heatmap_df = pd.DataFrame(index=all_nicknames, columns=multi_columns)

# For the empty controls, fill with 1.0 (indicating no significant p-values)
heatmap_df = heatmap_df.fillna(1.0)

for fname in all_csvs:
    csv_label = os.path.basename(fname)
    df = pd.read_csv(fname)
    df = df.set_index("Nickname")
    for test_col, test_label in test_types:
        for nickname in all_nicknames:
            # If nickname not in df, set to 1.0
            if nickname not in df.index:
                val = 1.0
            else:
                val = df[test_col].get(nickname, 1.0)
                # If still NaN, set to 1.0
                if pd.isna(val):
                    val = 1.0
            heatmap_df.loc[nickname, (csv_label, test_label)] = val

# # Convert to boolean for plotting
# heatmap_df = heatmap_df.fillna(False).astype(bool)

# Convert to float for plotting
heatmap_df = heatmap_df.astype(float)

# Identify block boundaries
columns = heatmap_df.columns
csv_names = [col[0] for col in columns]

# Find where tailored ends and emptysplit starts
split_idx = 0
for i, name in enumerate(csv_names):
    if "emptysplit" in name:
        split_idx = i
        break


# Find PCA boundaries within each block
def get_pca_boundaries(csv_names_block):
    boundaries = []
    last = None
    for i, name in enumerate(csv_names_block):
        pca = name.replace("_allmethods_tailored_ctrls.csv", "").replace("_allmethods_emptysplitctrl.csv", "")
        if last is not None and pca != last:
            boundaries.append(i)
        last = pca
    return boundaries


tailored_boundaries = get_pca_boundaries(csv_names[:split_idx])
emptysplit_boundaries = get_pca_boundaries(csv_names[split_idx:])

# Order Nicknames by ascending sum of p-values
nickname_order = heatmap_df.sum(axis=1).sort_values().index
heatmap_df = heatmap_df.loc[nickname_order]

plt.figure(figsize=(2 + len(heatmap_df.columns) * 0.5, max(8, len(heatmap_df) * 0.3)))
ax = sns.heatmap(
    heatmap_df,
    cmap="viridis_r",  # reversed so low p is dark
    cbar=True,
    linewidths=0.5,
    linecolor="gray",
    vmin=0,
    vmax=1,
)

plt.title("PCA Test p-values Heatmap (all CSVs)")
plt.xlabel("CSV / Test")
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

# Draw thick separator between tailored and emptysplit
if split_idx > 0:
    plt.axvline(split_idx, color="black", linewidth=3)

# Draw thin separators for each PCA within tailored
for b in tailored_boundaries:
    plt.axvline(b, color="black", linewidth=2, linestyle="--")

# Draw thin separators for each PCA within emptysplit
for b in emptysplit_boundaries:
    plt.axvline(split_idx + b, color="black", linewidth=2, linestyle="--")

plt.tight_layout()
plt.savefig("pca_pvalue_heatmap_combined.png", dpi=200)
# plt.show()

# Save a pdf version
plt.savefig("pca_pvalue_heatmap_combined.pdf", dpi=200)
plt.close()
