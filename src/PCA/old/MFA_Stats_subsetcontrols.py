import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial import distance

from statsmodels.stats.multitest import multipletests
import PCA.Config as Config

eig_df = pd.read_csv("mfa_eigenvalues.csv")
cumulative_variance = np.cumsum(eig_df["percentage of variance"])
n_dims = np.searchsorted(cumulative_variance, 90) + 1
print(f"Using first {n_dims} dimensions to cover 90% variance")

mfa_scores = pd.read_csv("mfa_scores.csv")
final_dataset = pd.read_feather("mfa_with_metadata.feather")

selected_dims = mfa_scores.columns[:n_dims]
scores_with_meta = final_dataset[["Nickname", "Split", "Brain region"] + list(selected_dims)]

results = []
all_nicknames = scores_with_meta["Nickname"].unique()


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


results = []
for nickname in all_nicknames:
    subset = Config.get_subset_data(scores_with_meta, col="Nickname", value=nickname, force_control="Empty-Split")
    if subset.empty or (subset["Nickname"] == nickname).sum() == 0:
        continue
    control_names = subset["Nickname"].unique()
    control_names = [n for n in control_names if n != nickname]
    if not control_names:
        continue
    control_name = control_names[0]

    # 1. Mann-Whitney U sur chaque dimension + correction FDR
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
    # Correction FDR sur les p-values Mann-Whitney
    mannwhitney_pvals_np = np.array([p for p in mannwhitney_pvals if not np.isnan(p)])
    dims_tested_np = [d for i, d in enumerate(dims_tested) if not np.isnan(mannwhitney_pvals[i])]
    if len(mannwhitney_pvals_np) > 0:
        rejected, pvals_corr, _, _ = multipletests(mannwhitney_pvals_np, alpha=0.05, method="fdr_bh")
        significant_dims = [dims_tested_np[i] for i, rej in enumerate(rejected) if rej]
        mannwhitney_any = any(rejected)
    else:
        significant_dims = []
        mannwhitney_any = False

    # 2. Permutation test multivariate (euclidienne)
    group_scores = subset[subset["Nickname"] == nickname][selected_dims].values
    control_scores = subset[subset["Nickname"] == control_name][selected_dims].values
    if group_scores.size == 0 or control_scores.size == 0:
        continue
    perm_stat, perm_pval = permutation_test(group_scores, control_scores)

    # 3. Mahalanobis + permutation
    maha_stat, maha_pval = mahalanobis_permutation_test(group_scores, control_scores)

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

results_df.to_csv("mfa_stats_results_allmethods.csv", index=False)
print("Significant hits for Mann whitney and permutation test:")
print(results_df[(results_df["MannWhitney_any_dim_significant"]) & (results_df["Permutation_FDR_significant"])])
