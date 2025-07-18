# PCA analysis for static metrics using RobustScaler and SparsePCA

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import RobustScaler
from scipy.spatial import distance

import matplotlib.pyplot as plt
import Config

from statsmodels.stats.multitest import multipletests
import scipy.stats as stats


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


# Load and preprocess data
dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250520_summary_TNT_screen_Data/summary/pooled_summary.feather"
)
dataset = Config.cleanup_data(dataset)
exclude_nicknames = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS", "854 (OK107-Gal4)"]
dataset = dataset[~dataset["Nickname"].isin(exclude_nicknames)]

# Convert boolean metrics to numeric
for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)

# Define metric columns (same as in PCA_Temporal.py)
metric_names = [
    "nb_events",
    "max_event",
    "max_event_time",
    "max_distance",
    "final_event",
    "final_event_time",
    "nb_significant_events",
    "significant_ratio",
    "major_event",
    "major_event_time",
    "pulled",
    "pulling_ratio",
    "interaction_persistence",
    "distance_moved",
    "distance_ratio",
    "chamber_exit_time",
    # "normalized_velocity",
    "auc",
    "overall_slope",
    "overall_interaction_rate",
]
binned_slope_cols = [col for col in dataset.columns if col.startswith("binned_slope_")]
interaction_rate_cols = [col for col in dataset.columns if col.startswith("interaction_rate_bin_")]
binned_auc_cols = [col for col in dataset.columns if col.startswith("binned_auc_")]
metric_names += binned_slope_cols + interaction_rate_cols + binned_auc_cols

# Handle missing values
# dataset["max_event"] = dataset["max_event"].fillna(-1)
# dataset["first_significant_event"] = dataset["first_significant_event"].fillna(-1)
# dataset["major_event"] = dataset["major_event"].fillna(-1)
# dataset["final_event"] = dataset["final_event"].fillna(-1)
# dataset["max_event_time"] = dataset["max_event_time"].fillna(3600)
# dataset["first_significant_event_time"] = dataset["first_significant_event_time"].fillna(3600)
# dataset["major_event_time"] = dataset["major_event_time"].fillna(3600)
# dataset["final_event_time"] = dataset["final_event_time"].fillna(3600)

na_metrics = dataset[metric_names].isna().sum()
valid_metric_names = [col for col in metric_names if col not in na_metrics[na_metrics > 0].index]
metrics_data = dataset[valid_metric_names]

# Identify static metrics
temporal_metrics = [
    col for col in valid_metric_names if col.startswith(("binned_slope_", "interaction_rate_bin_", "binned_auc_"))
]
static_metrics = [col for col in valid_metric_names if col not in temporal_metrics]

# Prepare static data
static_data = metrics_data[static_metrics].to_numpy()

# Robust scaling
scaler = RobustScaler()
static_data_scaled = scaler.fit_transform(static_data)

# Standard PCA
pca = PCA(n_components=min(10, len(static_metrics)))
pca_scores = pca.fit_transform(static_data_scaled)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("PCA Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Static Metrics PCA")
plt.show()

# Save PCA scores and loadings
pca_scores_df = pd.DataFrame(pca_scores, columns=[f"PCA{i+1}" for i in range(pca_scores.shape[1])])
pca_scores_df.to_csv("static_pca_scores.csv", index=False)
pca_loadings_df = pd.DataFrame(
    pca.components_, columns=static_metrics, index=[f"PCA{i+1}" for i in range(pca.components_.shape[0])]
)
pca_loadings_df.to_csv("static_pca_loadings.csv")

# Sparse PCA for feature selection
sparse_pca = SparsePCA(n_components=min(10, len(static_metrics)), random_state=42)
sparse_scores = sparse_pca.fit_transform(static_data_scaled)
sparse_loadings = sparse_pca.components_

# Save SparsePCA scores and loadings
sparse_scores_df = pd.DataFrame(sparse_scores, columns=[f"SparsePCA{i+1}" for i in range(sparse_scores.shape[1])])
sparse_scores_df.to_csv("static_sparsepca_scores.csv", index=False)
sparse_loadings_df = pd.DataFrame(
    sparse_loadings, columns=static_metrics, index=[f"SparsePCA{i+1}" for i in range(sparse_loadings.shape[0])]
)
sparse_loadings_df.to_csv("static_sparsepca_loadings.csv")

# Optionally, set force_control here (None or "Empty-Split")
force_control = None  # or "Empty-Split" to force control group

# Combine PCA scores with metadata
metadata_names = [col for col in dataset.columns if col not in metric_names]
metadata_data = dataset[metadata_names]
pca_scores_with_meta = pd.concat([metadata_data.reset_index(drop=True), pca_scores_df], axis=1)

# Determine suffix for output files
suffix = "_emptysplit" if force_control == "Empty-Split" else "_tailoredctrls"

# Save as feather for downstream use (ID card)
pca_scores_with_meta.to_feather(f"static_pca_with_metadata{suffix}.feather")

# Select PCA components up to % variance
Explained_variance = 95

# Print explained variance of each component
print("Explained variance by each PCA component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PCA{i+1}: {var:.4f}")

cumulative_variance = np.cumsum(pca.explained_variance_ratio_ * 100)
n_dims = np.searchsorted(cumulative_variance, Explained_variance) + 1
print(f"Using first {n_dims} PCA components to cover {Explained_variance}% variance")
selected_dims = pca_scores_df.columns[:n_dims]

# --- Statistical comparison for PCA components with three methods ---
results = []
all_nicknames = pca_scores_with_meta["Nickname"].unique()

for nickname in all_nicknames:
    subset = Config.get_subset_data(pca_scores_with_meta, col="Nickname", value=nickname, force_control=force_control)
    if subset.empty or (subset["Nickname"] == nickname).sum() == 0:
        continue
    control_names = subset["Nickname"].unique()
    control_names = [n for n in control_names if n != nickname]
    if not control_names:
        continue
    control_name = control_names[0]

    # Mann-Whitney U per dimension + FDR
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
        pvals_corr = []

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
            "MannWhitney_corrected_pvals": [float(p) for p in pvals_corr] if len(mannwhitney_pvals_np) > 0 else [],
            "MannWhitney_raw_pvals": [float(p) for p in mannwhitney_pvals_np] if len(mannwhitney_pvals_np) > 0 else [],
            "MannWhitney_dims_tested": dims_tested_np if len(mannwhitney_pvals_np) > 0 else [],
            "Permutation_stat": float(perm_stat),
            "Permutation_pval": float(perm_pval),
            "Mahalanobis_stat": float(maha_stat),
            "Mahalanobis_pval": float(maha_pval),
        }
    )

results_df = pd.DataFrame(results)
for col in ["Permutation_pval", "Mahalanobis_pval"]:
    rejected, pvals_corrected, _, _ = multipletests(results_df[col], alpha=0.05, method="fdr_bh")
    results_df[col.replace("_pval", "_FDR_significant")] = rejected
    results_df[col.replace("_pval", "_FDR_pval")] = pvals_corrected

# Save results CSV with appropriate suffix
results_df.to_csv(f"static_pca_stats_results_allmethods{suffix}.csv", index=False)
print("Significant hits for all methods:")
print(
    results_df[
        (results_df["MannWhitney_any_dim_significant"].astype(bool))
        & (results_df["Permutation_FDR_significant"].astype(bool))
        & (results_df["Mahalanobis_FDR_significant"].astype(bool))
    ]
)
