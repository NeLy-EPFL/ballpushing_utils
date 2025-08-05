# Updated PCA Analysis with Multiblock Integration and Robustness

# 1. Import Enhanced PCA Modules
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import RobustScaler  # Better for biological data
from prince import MFA  # Multiblock analysis
import numpy as np
import pandas as pd
import PCA.Config as Config
import subprocess
import os

import matplotlib.pyplot as plt

# 2. Data Segmentation
dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250520_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# dataset = Config.map_split_registry(dataset)

controls = ["Empty-Split", "Empty-Gal4", "TNTxPR"]

dataset = Config.cleanup_data(dataset)

exclude_nicknames = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS"]

dataset = dataset[~dataset["Nickname"].isin(exclude_nicknames)]


# Convert boolean metrics to numeric (1 for True, 0 for False)
for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)


# Configure which are the metrics and what are the metadata

metric_names = [
    "nb_events",
    "max_event",
    "max_event_time",
    "max_distance",
    "final_event",
    "final_event_time",
    "nb_significant_events",
    "significant_ratio",
    # "first_significant_event",
    # "first_significant_event_time",
    "major_event",
    "first_major_event_time",
    # "major_event_first",
    # "insight_effect",
    # "insight_effect_log",
    # "cumulated_breaks_duration",
    # "chamber_time",
    # "chamber_ratio",
    # "pushed",
    "pulled",
    "pulling_ratio",
    # "success_direction",
    # "interaction_proportion",
    "interaction_persistence",
    "distance_moved",
    "distance_ratio",
    # "exit_time",
    "chamber_exit_time",
    # "number_of_pauses",
    # "total_pause_duration",
    # "learning_slope",
    # "learning_slope_r2",
    # "logistic_L",
    # "logistic_k",
    # "logistic_t0",
    # "logistic_r2",
    # "avg_displacement_after_success",
    # "avg_displacement_after_failure",
    # "influence_ratio",
    "normalized_velocity",
    # "velocity_during_interactions",
    # "velocity_trend",
    "auc",
    "overall_slope",
    "overall_interaction_rate",
]

# Find all columns with the desired prefixes
binned_slope_cols = [col for col in dataset.columns if col.startswith("binned_slope_")]
interaction_rate_cols = [col for col in dataset.columns if col.startswith("interaction_rate_bin_")]
binned_auc_cols = [col for col in dataset.columns if col.startswith("binned_auc_")]

# Add them to metric_names list
metric_names += binned_slope_cols + interaction_rate_cols + binned_auc_cols

# For each column, handle missing values gracefully

dataset["max_event"] = dataset["max_event"].fillna(-1)
dataset["first_significant_event"] = dataset["first_significant_event"].fillna(-1)
dataset["major_event"] = dataset["major_event"].fillna(-1)
dataset["final_event"] = dataset["final_event"].fillna(-1)
dataset["max_event_time"] = dataset["max_event_time"].fillna(3600)
dataset["first_significant_event_time"] = dataset["first_significant_event_time"].fillna(3600)
dataset["first_major_event_time"] = dataset["first_major_event_time"].fillna(3600)
dataset["final_event_time"] = dataset["final_event_time"].fillna(3600)

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

# 3. Robust Scaling per Block
scaler_temporal = RobustScaler(quantile_range=(25, 75))  # Resistant to outliers
metrics_temporal_scaled = scaler_temporal.fit_transform(metrics_data[temporal_metrics])

scaler_static = RobustScaler()
metrics_static_scaled = scaler_static.fit_transform(metrics_data[static_metrics])

# 4. Multiblock PCA Implementation
mfa = MFA(
    n_components=min(30, len(temporal_metrics) + len(static_metrics)),
    copy=True,
    random_state=42,
)

# Combine scaled data preserving block structure
combined_scaled = np.hstack((metrics_temporal_scaled, metrics_static_scaled))
combined_scaled_df = pd.DataFrame(combined_scaled, columns=temporal_metrics + static_metrics)

# Optionally, set force_control here (None or "Empty-Split")
force_control = None  # or "Empty-Split" to force control group
# Determine suffix for output files
suffix = "_emptysplit" if force_control == "Empty-Split" else "_tailoredctrls"

# Save scaled data for R
combined_scaled_df.to_csv(f"mfa_input{suffix}.csv", index=False)

# Write group sizes (number of columns in each block) BEFORE calling R
with open(f"mfa_groups{suffix}.txt", "w") as f:
    f.write(f"{len(temporal_metrics)} {len(static_metrics)}\n")

# Call R script to run MFA with FactoMineR
# Ensure R_LIBS_USER is set so R can find local libraries
r_libs = "/home/matthias/R/pc-linux-gnu-library/FactoMineR"
env = os.environ.copy()
env["R_LIBS_USER"] = r_libs
subprocess.run(["Rscript", "run_mfa.R", suffix], check=True, env=env)

# Read R results
mfa_scores = pd.read_csv(f"mfa_scores{suffix}.csv")
loadings_df = pd.read_csv(f"mfa_loadings{suffix}.csv", index_col=0)
contrib_df = pd.read_csv(f"mfa_block_contrib{suffix}.csv", index_col=0)

# Combine with metadata
final_dataset = pd.concat([mfa_scores, metadata_data.reset_index(drop=True)], axis=1)
final_dataset.to_feather(f"mfa_with_metadata{suffix}.feather")

# Save block-aware loadings (already saved by R, but you can re-save if needed)
loadings_df.to_csv(f"mfa_block_loadings{suffix}.csv")
contrib_df.to_csv(f"mfa_block_contributions{suffix}.csv")

# Read explained variance from R
eig_df = pd.read_csv(f"mfa_eigenvalues{suffix}.csv")
plt.plot(np.cumsum(eig_df["percentage of variance"]))
plt.xlabel("Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("MFA Cumulative Explained Variance (FactoMineR)")
plt.show()

# --- Posthoc statistical analysis for MFA (integrated, standardized for IdCard) ---
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from scipy.spatial import distance


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


# --- Standardized posthoc for IdCard ---
cumulative_variance = np.cumsum(eig_df["percentage of variance"])
n_dims = np.searchsorted(cumulative_variance, 90) + 1
print(f"[MFA] Using first {n_dims} dimensions to cover 90% variance")

mfa_scores = pd.read_csv(f"mfa_scores{suffix}.csv")
final_dataset = pd.read_feather(f"mfa_with_metadata{suffix}.feather")
selected_dims = mfa_scores.columns[:n_dims]
scores_with_meta = final_dataset[["Nickname", "Split", "Brain region"] + list(selected_dims)]

results = []
all_nicknames = scores_with_meta["Nickname"].unique()
for nickname in all_nicknames:
    # Use tailored controls if not force_control, else force_control
    subset = None
    if force_control == "Empty-Split":
        subset = Config.get_subset_data(scores_with_meta, col="Nickname", value=nickname, force_control="Empty-Split")
    else:
        subset = Config.get_subset_data(scores_with_meta, col="Nickname", value=nickname)
    if subset.empty or (subset["Nickname"] == nickname).sum() == 0:
        continue
    control_names = subset["Nickname"].unique()
    control_names = [n for n in control_names if n != nickname]
    if not control_names:
        continue
    control_name = control_names[0]
    # 1. Mann-Whitney U per dimension + FDR
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
    # 2. Permutation test (multivariate)
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
results_df.to_csv(f"mfa_stats_results_allmethods{suffix}.csv", index=False)
print("[MFA] Significant hits for all methods:")
print(
    results_df[
        (results_df["MannWhitney_any_dim_significant"].astype(bool))
        & (results_df["Permutation_FDR_significant"].astype(bool))
        & (results_df["Mahalanobis_FDR_significant"].astype(bool))
    ]
)
