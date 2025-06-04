import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


# eig_df: DataFrame with "percentage of variance" column from R
eig_df = pd.read_csv("mfa_eigenvalues.csv")
cumulative_variance = np.cumsum(eig_df["percentage of variance"])
n_dims = np.searchsorted(cumulative_variance, 90) + 1  # +1 because index starts at 0
print(f"Using first {n_dims} dimensions to cover 90% variance")

mfa_scores = pd.read_csv("mfa_scores.csv")
final_dataset = pd.read_feather("mfa_with_metadata.feather")

# mfa_scores: DataFrame with MFA scores (from R)
# final_dataset: DataFrame with scores and metadata
selected_dims = mfa_scores.columns[:n_dims]
scores_with_meta = final_dataset[["Nickname"] + list(selected_dims)]


results = []
control = "Empty-Split"
nicknames = scores_with_meta["Nickname"].unique()
nicknames = [n for n in nicknames if n != control]

for dim in selected_dims:
    control_scores = scores_with_meta[scores_with_meta["Nickname"] == control][dim]
    for nickname in nicknames:
        group_scores = scores_with_meta[scores_with_meta["Nickname"] == nickname][dim]
        # Use t-test (or Mann-Whitney if not normal)
        stat, pval = stats.ttest_ind(group_scores, control_scores, nan_policy="omit", equal_var=False)
        results.append({"Nickname": nickname, "Dimension": dim, "pval": pval})

results_df = pd.DataFrame(results)

# FDR correction (Benjamini-Hochberg)
rejected, pvals_corrected, _, _ = multipletests(results_df["pval"], alpha=0.05, method="fdr_bh")
results_df["FDR_significant"] = rejected
results_df["FDR_pval"] = pvals_corrected

# Save results
results_df.to_csv("mfa_stats_results.csv", index=False)

# Print summary of significant results
significant_results = results_df[results_df["FDR_significant"]]
print("Significant results (FDR):")
print(significant_results)
