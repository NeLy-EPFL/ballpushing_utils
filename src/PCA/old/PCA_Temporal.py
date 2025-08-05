# Updated PCA Analysis with Multiblock Integration and Robustness

# 1. Import Enhanced PCA Modules
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import RobustScaler  # Better for biological data
from prince import MFA  # Multiblock analysis
import numpy as np
import pandas as pd
import PCA.Config as Config
import subprocess
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA

from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

from scipy.spatial import distance


import matplotlib.pyplot as plt


def permutation_test(group1, group2, n_permutations=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    observed = np.linalg.norm(group1.mean(axis=0) - group2.mean(axis=0))
    combined = np.vstack([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        stat = np.linalg.norm(perm_group1.mean(axis=0) - perm_group2.mean(axis=0))
        if stat >= observed:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return observed, pval


def mahalanobis_distance(group1, group2):
    mean1 = group1.mean(axis=0)
    mean2 = group2.mean(axis=0)
    pooled = np.vstack([group1, group2])
    cov = np.cov(pooled, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    dist = distance.mahalanobis(mean1, mean2, inv_cov)
    return dist


def mahalanobis_permutation_test(group1, group2, n_permutations=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    observed = mahalanobis_distance(group1, group2)
    combined = np.vstack([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        stat = mahalanobis_distance(perm_group1, perm_group2)
        if stat >= observed:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return observed, pval


# 2. Data Segmentation
dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250520_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# dataset = Config.map_split_registry(dataset)

controls = ["Empty-Split", "Empty-Gal4", "TNTxPR"]

dataset = Config.cleanup_data(dataset)

exclude_nicknames = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS", "854 (OK107-Gal4)"]

dataset = dataset[~dataset["Nickname"].isin(exclude_nicknames)]


# Convert boolean metrics to numeric (1 for True, 0 for False)
for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)


# Configure which are the metrics and what are the metadata

metric_names = []

# Find all columns with the desired prefixes
binned_slope_cols = [col for col in dataset.columns if col.startswith("binned_slope_")]
interaction_rate_cols = [col for col in dataset.columns if col.startswith("interaction_rate_bin_")]
binned_auc_cols = [col for col in dataset.columns if col.startswith("binned_auc_")]

# Add them to metric_names list
metric_names += binned_slope_cols + interaction_rate_cols + binned_auc_cols

# For each column, handle missing values gracefully

# dataset["max_event"] = dataset["max_event"].fillna(-1)
# dataset["first_significant_event"] = dataset["first_significant_event"].fillna(-1)
# dataset["major_event"] = dataset["major_event"].fillna(-1)
# dataset["final_event"] = dataset["final_event"].fillna(-1)
# dataset["max_event_time"] = dataset["max_event_time"].fillna(3600)
# dataset["first_significant_event_time"] = dataset["first_significant_event_time"].fillna(3600)
# dataset["first_major_event_time"] = dataset["first_major_event_time"].fillna(3600)
# dataset["final_event_time"] = dataset["final_event_time"].fillna(3600)

# Check which metrics have NA values
na_metrics = dataset[metric_names].isna().sum()
print("Metrics with NA values:")
print(na_metrics[na_metrics > 0])

# Keep only columns without NaN values
valid_metric_names = [col for col in metric_names if col not in na_metrics[na_metrics > 0].index]
print(f"Valid metric columns (no NaNs): {valid_metric_names}")

# Define metadata columns as any column not in metric_names
metadata_names = [col for col in dataset.columns if col not in metric_names]

print(f"Metadata columns: {metadata_names}")
print(f"Metric columns: {metric_names}")

metrics_data = dataset[valid_metric_names]  # Select only the valid metric columns
metadata_data = dataset[metadata_names]  # Select only the metadata columns

# Split metrics into temporal vs static blocks
temporal_metrics = [
    col for col in valid_metric_names if col.startswith(("binned_slope_", "interaction_rate_bin_", "binned_auc_"))
]
static_metrics = [col for col in valid_metric_names if col not in temporal_metrics]

# Prepare temporal data (samples x time bins)
temporal_data = metrics_data[temporal_metrics].to_numpy()

# Assume each column is a time bin; create a grid for time points
n_time_bins = len(temporal_metrics)
time_points = np.arange(n_time_bins)

# Create FDataGrid object
fd = FDataGrid(data_matrix=temporal_data, grid_points=time_points)

# Run Functional PCA
fpca = FPCA(n_components=min(10, n_time_bins))
fpca.fit(fd)
fpca_scores = fpca.transform(fd)

# Scree plot for FPCA
plt.figure()
plt.plot(np.cumsum(fpca.explained_variance_ratio_ * 100))
plt.xlabel("FPCA Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("FPCA on Temporal Metrics")
plt.show()

# Optionally, save FPCA scores for downstream analysis
fpca_scores_df = pd.DataFrame(fpca_scores, columns=[f"FPCA{i+1}" for i in range(fpca_scores.shape[1])])
fpca_scores_df.to_csv("fpca_temporal_scores.csv", index=False)

# Keep PCs up to % variance

explained_variance = 90

cumulative_variance = np.cumsum(fpca.explained_variance_ratio_ * 100)

n_dims = np.searchsorted(cumulative_variance, explained_variance) + 1

print(f"Using first {n_dims} FPCA components to cover {explained_variance}% variance")

print("fpca.components_ shape:", fpca.components_.shape)
# Save FPCA loadings (principal directions) as CSV
# fpca.components_ is a list of FDataGrid objects, one per component
comps = np.vstack([comp.data_matrix.ravel() for comp in fpca.components_])
n_grid_points = comps.shape[1]
if len(temporal_metrics) == n_grid_points:
    columns = temporal_metrics
else:
    columns = [f"time_bin_{i}" for i in range(n_grid_points)]
fpca_loadings_df = pd.DataFrame(comps, columns=columns, index=[f"FPCA{i+1}" for i in range(comps.shape[0])])
fpca_loadings_df.to_csv("fpca_temporal_loadings.csv")

# Optionally, set force_control here (None or "Empty-Split")
force_control = None  # or "Empty-Split" to force control group

# Compare Nicknames with Controls
results = []

# Combine FPCA scores with metadata
fpca_scores_with_meta = pd.concat([metadata_data.reset_index(drop=True), fpca_scores_df], axis=1)

# Determine suffix for output files
suffix = "_emptysplit" if force_control == "Empty-Split" else "_tailoredctrls"

# Save as feather for downstream use (ID card)
fpca_scores_with_meta.to_feather(f"fpca_temporal_with_metadata{suffix}.feather")

# Select FPCA components up to 90% variance
selected_dims = fpca_scores_df.columns[:n_dims]

results = []
all_nicknames = fpca_scores_with_meta["Nickname"].unique()

for nickname in all_nicknames:
    subset = Config.get_subset_data(
        fpca_scores_with_meta,
        col="Nickname",
        value=nickname,
        force_control=force_control,  # pass force_control to subset function
    )
    if subset.empty or (subset["Nickname"] == nickname).sum() == 0:
        continue
    control_names = subset["Nickname"].unique()
    control_names = [n for n in control_names if n != nickname]
    if not control_names:
        continue
    control_name = control_names[0]

    # Mann-Whitney U per FPCA component + FDR
    mannwhitney_pvals = []
    dims_tested = []
    for dim in selected_dims:
        group_scores = subset[subset["Nickname"] == nickname][dim]
        control_scores = subset[subset["Nickname"] == control_name][dim]
        if group_scores.empty or control_scores.empty:
            mannwhitney_pvals.append(np.nan)
            continue
        stat, pval = stats.mannwhitneyu(group_scores, control_scores, alternative="two-sided")
        mannwhitney_pvals.append(pval)
        dims_tested.append(dim)
    mannwhitney_pvals_np = np.array([p for p in mannwhitney_pvals if not np.isnan(p)])
    dims_tested_np = [d for i, d in enumerate(dims_tested) if not np.isnan(mannwhitney_pvals[i])]
    if len(mannwhitney_pvals_np) > 0:
        rejected, pvals_corr, _, _ = multipletests(mannwhitney_pvals_np, alpha=0.05, method="fdr_bh")
        significant_dims = [dims_tested_np[i] for i, rej in enumerate(rejected) if rej]
        mannwhitney_any = any(rejected)
    else:
        significant_dims = []
        mannwhitney_any = False

    # Permutation test (multivariate)
    group_matrix = subset[subset["Nickname"] == nickname][selected_dims].values
    control_matrix = subset[subset["Nickname"] == control_name][selected_dims].values
    if group_matrix.size == 0 or control_matrix.size == 0:
        continue
    perm_stat, perm_pval = permutation_test(group_matrix, control_matrix)
    maha_stat, maha_pval = mahalanobis_permutation_test(group_matrix, control_matrix)

    results.append(
        {
            "Nickname": nickname,
            "Control": control_name,
            "MannWhitney_any_dim_significant": mannwhitney_any,
            "MannWhitney_significant_dims": significant_dims,
            "Permutation_stat": perm_stat,
            "Permutation_pval": perm_pval,
            "Mahalanobis_stat": maha_stat,
            "Mahalanobis_pval": maha_pval,
        }
    )

results_df = pd.DataFrame(results)
for col in ["Permutation_pval", "Mahalanobis_pval"]:
    rejected, pvals_corrected, _, _ = multipletests(results_df[col], alpha=0.05, method="fdr_bh")
    results_df[col.replace("_pval", "_FDR_significant")] = rejected
    results_df[col.replace("_pval", "_FDR_pval")] = pvals_corrected

# Save results CSV with appropriate suffix
results_df.to_csv(f"fpca_temporal_stats_results_allmethods{suffix}.csv", index=False)
print("Significant hits for all methods:")
print(
    results_df[
        (results_df["MannWhitney_any_dim_significant"].astype(bool))
        & (results_df["Permutation_FDR_significant"].astype(bool))
        & (results_df["Mahalanobis_FDR_significant"].astype(bool))
    ]
)
