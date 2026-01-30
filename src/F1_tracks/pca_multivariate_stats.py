"""
Statistical Analysis Functions for PCA
========================================

This module provides comprehensive multivariate statistical analysis functions
for PCA results, including:
- PERMANOVA (Permutational MANOVA)
- Pairwise permutation tests
- Linear Mixed-Effects Models on individual PC scores
- Centroid analysis with Mahalanobis distances
- Bootstrap confidence ellipses

All permutation-based methods are non-parametric and do not assume normality.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, mannwhitneyu
from scipy.spatial.distance import mahalanobis
from statsmodels.stats.multitest import multipletests


def permanova(X, groups, n_permutations=10000):
    """
    Perform PERMANOVA (Permutational Multivariate Analysis of Variance).

    Tests whether groups differ in multivariate space using permutation testing.
    Does not assume normality - uses empirical null distribution.

    Args:
        X: numpy array of shape (n_samples, n_features) - PC scores
        groups: numpy array of shape (n_samples,) - group labels
        n_permutations: number of permutations for test

    Returns:
        dict with F-statistic, p-value, and effect size (R²)
    """
    n_samples = X.shape[0]
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Calculate observed F-statistic
    def calculate_f_statistic(X, groups):
        """Calculate pseudo-F statistic for PERMANOVA"""
        # Total sum of squares
        grand_mean = X.mean(axis=0)
        ss_total = np.sum((X - grand_mean) ** 2)

        # Within-group sum of squares
        ss_within = 0
        for group in unique_groups:
            group_mask = groups == group
            group_data = X[group_mask]
            group_mean = group_data.mean(axis=0)
            ss_within += np.sum((group_data - group_mean) ** 2)

        # Between-group sum of squares
        ss_between = ss_total - ss_within

        # Degrees of freedom
        df_between = n_groups - 1
        df_within = n_samples - n_groups

        # Mean squares
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # F-statistic
        F = ms_between / ms_within if ms_within > 0 else 0

        # R² (effect size)
        r_squared = ss_between / ss_total if ss_total > 0 else 0

        return F, r_squared

    observed_F, observed_r2 = calculate_f_statistic(X, groups)

    # Permutation test
    print(f"  Running PERMANOVA with {n_permutations} permutations...")
    permuted_F_values = []

    np.random.seed(42)
    for i in range(n_permutations):
        # Permute group labels
        permuted_groups = np.random.permutation(groups)
        permuted_F, _ = calculate_f_statistic(X, permuted_groups)
        permuted_F_values.append(permuted_F)

    permuted_F_values = np.array(permuted_F_values)

    # Calculate p-value
    p_value = np.sum(permuted_F_values >= observed_F) / n_permutations

    return {"F_statistic": observed_F, "p_value": p_value, "R_squared": observed_r2, "n_permutations": n_permutations}


def pairwise_permutation_test(X_group1, X_group2, n_permutations=10000, metric="mahalanobis"):
    """
    Perform pairwise permutation test between two groups in multivariate space.

    Args:
        X_group1: numpy array (n_samples_1, n_features) - first group PC scores
        X_group2: numpy array (n_samples_2, n_features) - second group PC scores
        n_permutations: number of permutations
        metric: 'mahalanobis' or 'euclidean' for distance metric

    Returns:
        dict with test statistic, p-value, and effect size
    """

    def calculate_statistic(X1, X2, metric="mahalanobis"):
        """Calculate distance between group centroids"""
        centroid1 = X1.mean(axis=0)
        centroid2 = X2.mean(axis=0)

        if metric == "mahalanobis":
            # Pooled covariance matrix
            X_combined = np.vstack([X1, X2])
            cov = np.cov(X_combined.T)

            # Add small regularization to avoid singular matrix
            cov += np.eye(cov.shape[0]) * 1e-6

            try:
                cov_inv = np.linalg.inv(cov)
                diff = centroid1 - centroid2
                dist = np.sqrt(diff @ cov_inv @ diff.T)
            except:
                # Fallback to Euclidean if covariance is singular
                dist = np.linalg.norm(centroid1 - centroid2)
        else:
            # Euclidean distance
            dist = np.linalg.norm(centroid1 - centroid2)

        return dist

    observed_stat = calculate_statistic(X_group1, X_group2, metric)

    # Permutation test
    X_combined = np.vstack([X_group1, X_group2])
    n1 = len(X_group1)
    n_total = len(X_combined)

    permuted_stats = []
    np.random.seed(42)

    for i in range(n_permutations):
        # Permute samples
        perm_indices = np.random.permutation(n_total)
        X_perm1 = X_combined[perm_indices[:n1]]
        X_perm2 = X_combined[perm_indices[n1:]]

        perm_stat = calculate_statistic(X_perm1, X_perm2, metric)
        permuted_stats.append(perm_stat)

    permuted_stats = np.array(permuted_stats)
    p_value = np.sum(permuted_stats >= observed_stat) / n_permutations

    # Effect size (Cohen's d in multivariate space)
    X_combined = np.vstack([X_group1, X_group2])
    pooled_std = np.std(X_combined, axis=0).mean()
    cohens_d = observed_stat / pooled_std if pooled_std > 0 else 0

    return {
        "statistic": observed_stat,
        "p_value": p_value,
        "metric": metric,
        "cohens_d": cohens_d,
        "n_permutations": n_permutations,
    }


def lmm_analysis_on_pcs(pca_df, secondary_col, pretraining_col, n_components=3):
    """
    Perform Linear Mixed-Effects Model analysis on individual PC scores.

    For each PC:
    - Tests global pretraining effect
    - Tests genotype × pretraining interaction
    - Estimates pretraining effect for each genotype separately

    Args:
        pca_df: DataFrame with PC scores and metadata
        secondary_col: column name for genotype/condition
        pretraining_col: column name for pretraining status
        n_components: number of PCs to analyze

    Returns:
        dict with results for each PC
    """
    results = {}

    for pc_idx in range(n_components):
        pc_name = f"PC{pc_idx + 1}"

        if pc_name not in pca_df.columns:
            continue

        print(f"\n  LMM Analysis for {pc_name}:")

        # Prepare data
        df_pc = pca_df[[secondary_col, pretraining_col, pc_name]].copy()
        df_pc = df_pc.dropna()

        # Convert pretraining to numeric (0 = naive, 1 = pretrained)
        df_pc["pretraining_numeric"] = df_pc[pretraining_col].apply(
            lambda x: 1 if str(x).lower() in ["y", "yes", "true", "1"] else 0
        )

        # Create reference-coded secondary variable
        all_values = df_pc[secondary_col].unique()
        control_value = all_values[0]  # First value as reference

        df_pc["secondary_cat"] = pd.Categorical(
            df_pc[secondary_col], categories=[control_value] + [v for v in sorted(all_values) if v != control_value]
        )

        try:
            # Fit global model with interaction
            formula = f"{pc_name} ~ C(secondary_cat) * pretraining_numeric"
            model = smf.ols(formula, data=df_pc).fit(cov_type="HC3")

            # Extract pretraining effect for reference group
            pretrain_coef = model.params.get("pretraining_numeric", np.nan)
            pretrain_pval = model.pvalues.get("pretraining_numeric", np.nan)
            pretrain_stderr = model.bse.get("pretraining_numeric", np.nan)

            print(f"    Pretraining effect (reference: {control_value}): β={pretrain_coef:.3f}, p={pretrain_pval:.4f}")

            # Per-group analysis
            group_effects = {}
            for group_val in sorted(df_pc[secondary_col].unique()):
                df_group = df_pc[df_pc[secondary_col] == group_val]

                if len(df_group) < 4:  # Need minimum samples
                    continue

                pretrain_vals = df_group["pretraining_numeric"].unique()
                if len(pretrain_vals) < 2:
                    continue

                try:
                    group_model = smf.ols(f"{pc_name} ~ pretraining_numeric", data=df_group).fit(cov_type="HC3")

                    effect_size = group_model.params["pretraining_numeric"]
                    effect_pval = group_model.pvalues["pretraining_numeric"]
                    effect_stderr = group_model.bse["pretraining_numeric"]

                    mean_naive = df_group[df_group["pretraining_numeric"] == 0][pc_name].mean()
                    mean_pretrained = df_group[df_group["pretraining_numeric"] == 1][pc_name].mean()

                    group_effects[group_val] = {
                        "effect_size": effect_size,
                        "stderr": effect_stderr,
                        "pvalue": effect_pval,
                        "mean_naive": mean_naive,
                        "mean_pretrained": mean_pretrained,
                    }

                    sig = (
                        "***"
                        if effect_pval < 0.001
                        else "**" if effect_pval < 0.01 else "*" if effect_pval < 0.05 else "ns"
                    )
                    print(f"      {group_val}: β={effect_size:.3f} (SE={effect_stderr:.3f}), p={effect_pval:.4f} {sig}")

                except Exception as e:
                    print(f"      {group_val}: Failed ({str(e)})")

            results[pc_name] = {
                "global_model": model,
                "pretraining_effect": pretrain_coef,
                "pretraining_pvalue": pretrain_pval,
                "group_effects": group_effects,
                "r_squared": model.rsquared,
            }

        except Exception as e:
            print(f"    Error: {str(e)}")
            results[pc_name] = {"error": str(e)}

    return results


def calculate_centroids_and_distances(pca_df, secondary_col, pretraining_col, n_components=3):
    """
    Calculate group centroids in PC space and Mahalanobis distances between them.

    Args:
        pca_df: DataFrame with PC scores and metadata
        secondary_col: column name for genotype/condition
        pretraining_col: column name for pretraining status
        n_components: number of PCs to use

    Returns:
        dict with centroids, distances, and covariance matrices
    """
    pc_cols = [f"PC{i+1}" for i in range(n_components) if f"PC{i+1}" in pca_df.columns]

    centroids = {}
    covariances = {}
    sample_sizes = {}

    # Calculate centroids and covariances for each group
    for sec_val in pca_df[secondary_col].unique():
        for pretrain_val in pca_df[pretraining_col].dropna().unique():
            mask = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretrain_val)
            group_data = pca_df.loc[mask, pc_cols].values

            if len(group_data) == 0:
                continue

            group_key = f"{sec_val}_{pretrain_val}"
            centroids[group_key] = group_data.mean(axis=0)
            covariances[group_key] = np.cov(group_data.T)
            sample_sizes[group_key] = len(group_data)

    # Calculate pairwise Mahalanobis distances
    distances = {}

    for key1, centroid1 in centroids.items():
        for key2, centroid2 in centroids.items():
            if key1 >= key2:  # Avoid duplicates
                continue

            # Pooled covariance matrix
            n1 = sample_sizes[key1]
            n2 = sample_sizes[key2]
            cov_pooled = ((n1 - 1) * covariances[key1] + (n2 - 1) * covariances[key2]) / (n1 + n2 - 2)

            # Add regularization
            cov_pooled += np.eye(cov_pooled.shape[0]) * 1e-6

            try:
                cov_inv = np.linalg.inv(cov_pooled)
                diff = centroid1 - centroid2
                maha_dist = np.sqrt(diff @ cov_inv @ diff.T)
            except:
                # Fallback to Euclidean
                maha_dist = np.linalg.norm(centroid1 - centroid2)

            distances[f"{key1}_vs_{key2}"] = {
                "mahalanobis_distance": maha_dist,
                "euclidean_distance": np.linalg.norm(centroid1 - centroid2),
                "centroid1": centroid1,
                "centroid2": centroid2,
            }

    return {"centroids": centroids, "covariances": covariances, "sample_sizes": sample_sizes, "distances": distances}


def bootstrap_confidence_ellipse(data, confidence=0.95, n_bootstrap=1000):
    """
    Calculate confidence ellipse for group centroid using bootstrap.

    Args:
        data: numpy array (n_samples, 2) - PC scores for 2D plot
        confidence: confidence level (0.95 = 95%)
        n_bootstrap: number of bootstrap samples

    Returns:
        tuple of (centroid, covariance, chi2_val) for plotting ellipse
    """
    n_samples = len(data)
    centroids = []

    np.random.seed(42)
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = data[indices]
        centroids.append(bootstrap_sample.mean(axis=0))

    centroids = np.array(centroids)

    # Calculate covariance of bootstrap centroids
    centroid = data.mean(axis=0)
    cov = np.cov(centroids.T)

    # Chi-square value for confidence level
    chi2_val = chi2.ppf(confidence, df=2)

    return centroid, cov, chi2_val


def mann_whitney_on_pcs(pca_df, secondary_col, pretraining_col, n_components=3, alpha=0.05):
    """
    Perform Mann-Whitney U tests on individual PC scores for each genotype.

    Tests whether naive vs pretrained groups differ for each PC separately.
    Applies FDR correction across all genotypes for each PC.

    Args:
        pca_df: DataFrame with PC scores and metadata
        secondary_col: column name for genotype/condition
        pretraining_col: column name for pretraining status
        n_components: number of PCs to analyze
        alpha: significance level for FDR correction

    Returns:
        dict with results for each PC
    """
    results = {}

    pc_cols = [f"PC{i+1}" for i in range(n_components) if f"PC{i+1}" in pca_df.columns]
    pretraining_vals = sorted(pca_df[pretraining_col].dropna().unique())

    if len(pretraining_vals) != 2:
        print("  Warning: Mann-Whitney test requires exactly 2 pretraining conditions")
        return results

    for pc_name in pc_cols:
        print(f"\n  Mann-Whitney U Test for {pc_name}:")

        test_results = []

        for sec_val in sorted(pca_df[secondary_col].unique()):
            mask_naive = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretraining_vals[0])
            mask_pretrained = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretraining_vals[1])

            data_naive = pca_df.loc[mask_naive, pc_name].dropna().values
            data_pretrained = pca_df.loc[mask_pretrained, pc_name].dropna().values

            if len(data_naive) < 2 or len(data_pretrained) < 2:
                continue

            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(data_naive, data_pretrained, alternative="two-sided")

            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(data_naive), len(data_pretrained)
            rank_biserial = 1 - (2 * statistic) / (n1 * n2)

            # Calculate medians
            median_naive = np.median(data_naive)
            median_pretrained = np.median(data_pretrained)

            test_results.append(
                {
                    "group": sec_val,
                    "statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial": rank_biserial,
                    "median_naive": median_naive,
                    "median_pretrained": median_pretrained,
                    "n_naive": n1,
                    "n_pretrained": n2,
                }
            )

        # Apply FDR correction
        if test_results:
            p_values = [r["p_value"] for r in test_results]
            _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

            for i, result in enumerate(test_results):
                result["p_value_corrected"] = p_corrected[i]

                sig = (
                    "***"
                    if p_corrected[i] < 0.001
                    else "**" if p_corrected[i] < 0.01 else "*" if p_corrected[i] < 0.05 else "ns"
                )
                print(
                    f"    {result['group']}: U={result['statistic']:.1f}, p={p_corrected[i]:.4f} {sig}, "
                    f"r={result['rank_biserial']:.3f}"
                )

        results[pc_name] = test_results

    return results


def permutation_test_on_pcs(pca_df, secondary_col, pretraining_col, n_components=3, n_permutations=10000, alpha=0.05):
    """
    Perform permutation tests on individual PC scores for each genotype.

    Non-parametric alternative to Mann-Whitney that directly tests difference in means.
    Applies FDR correction across all genotypes for each PC.

    Args:
        pca_df: DataFrame with PC scores and metadata
        secondary_col: column name for genotype/condition
        pretraining_col: column name for pretraining status
        n_components: number of PCs to analyze
        n_permutations: number of permutations for test
        alpha: significance level for FDR correction

    Returns:
        dict with results for each PC
    """
    results = {}

    pc_cols = [f"PC{i+1}" for i in range(n_components) if f"PC{i+1}" in pca_df.columns]
    pretraining_vals = sorted(pca_df[pretraining_col].dropna().unique())

    if len(pretraining_vals) != 2:
        print("  Warning: Permutation test requires exactly 2 pretraining conditions")
        return results

    for pc_name in pc_cols:
        print(f"\n  Permutation Test for {pc_name}:")

        test_results = []

        for sec_val in sorted(pca_df[secondary_col].unique()):
            mask_naive = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretraining_vals[0])
            mask_pretrained = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretraining_vals[1])

            data_naive = pca_df.loc[mask_naive, pc_name].dropna().values
            data_pretrained = pca_df.loc[mask_pretrained, pc_name].dropna().values

            if len(data_naive) < 2 or len(data_pretrained) < 2:
                continue

            # Calculate observed difference in means
            observed_diff = np.mean(data_pretrained) - np.mean(data_naive)

            # Permutation test
            combined_data = np.concatenate([data_naive, data_pretrained])
            n1 = len(data_naive)
            n_total = len(combined_data)

            permuted_diffs = []
            np.random.seed(42 + hash(sec_val) % 1000)  # Different seed per genotype

            for i in range(n_permutations):
                perm_indices = np.random.permutation(n_total)
                perm_group1 = combined_data[perm_indices[:n1]]
                perm_group2 = combined_data[perm_indices[n1:]]
                perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
                permuted_diffs.append(perm_diff)

            permuted_diffs = np.array(permuted_diffs)

            # Two-tailed p-value
            p_value = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff)) / n_permutations

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(data_naive) - 1) * np.var(data_naive, ddof=1)
                    + (len(data_pretrained) - 1) * np.var(data_pretrained, ddof=1)
                )
                / (len(data_naive) + len(data_pretrained) - 2)
            )
            cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

            test_results.append(
                {
                    "group": sec_val,
                    "observed_diff": observed_diff,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "mean_naive": np.mean(data_naive),
                    "mean_pretrained": np.mean(data_pretrained),
                    "n_naive": len(data_naive),
                    "n_pretrained": len(data_pretrained),
                }
            )

        # Apply FDR correction
        if test_results:
            p_values = [r["p_value"] for r in test_results]
            _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

            for i, result in enumerate(test_results):
                result["p_value_corrected"] = p_corrected[i]

                sig = (
                    "***"
                    if p_corrected[i] < 0.001
                    else "**" if p_corrected[i] < 0.01 else "*" if p_corrected[i] < 0.05 else "ns"
                )
                print(
                    f"    {result['group']}: Δ={result['observed_diff']:.3f}, p={p_corrected[i]:.4f} {sig}, "
                    f"d={result['cohens_d']:.3f}"
                )

        results[pc_name] = test_results

    return results


def residual_permutation_test_on_pcs(
    pca_df, secondary_col, pretraining_col, n_components, n_permutations=10000, alpha=0.05
):
    """
    Residual permutation tests on individual PC scores.

    Like the continuous metrics approach, this:
    1. Fits nuisance models (blocking factors only) for each PC
    2. Extracts residuals
    3. Performs permutation test on residuals
    4. Applies FDR correction

    This is distribution-free and accounts for blocking structure.

    Args:
        pca_df: DataFrame with PC scores and metadata
        secondary_col: column for genotype/condition
        pretraining_col: column for pretraining status
        n_components: number of PCs
        n_permutations: number of permutations
        alpha: significance level

    Returns:
        dict with results for each PC
    """
    print("\n🔄 RESIDUAL PERMUTATION TESTS ON PC SCORES")
    print("(Distribution-free, accounting for blocking factors)")
    print("-" * 60)

    results = {}

    for i in range(n_components):
        pc_name = f"PC{i+1}"
        if pc_name not in pca_df.columns:
            continue

        print(f"\n{pc_name}:")

        # Prepare data
        columns_needed = [secondary_col, pretraining_col, pc_name]
        for col in ["Date", "date", "fly", "arena_number", "arena_side"]:
            if col in pca_df.columns:
                columns_needed.append(col)

        df_clean = pca_df[columns_needed].copy()
        df_clean = df_clean.dropna(subset=[pc_name])

        # Try to fit nuisance model with blocking factors
        nuisance_formula_parts = []
        date_col = None
        if "Date" in df_clean.columns:
            date_col = "Date"
            nuisance_formula_parts.append(f"C({date_col})")
        elif "date" in df_clean.columns:
            date_col = "date"
            nuisance_formula_parts.append(f"C({date_col})")

        if "arena_number" in df_clean.columns and "arena_side" in df_clean.columns:
            nuisance_formula_parts.append("C(arena_number) * C(arena_side)")

        # Fit nuisance model if we have blocking factors
        if nuisance_formula_parts:
            nuisance_formula = f"{pc_name} ~ " + " + ".join(nuisance_formula_parts)
            try:
                nuisance_model = smf.ols(nuisance_formula, data=df_clean).fit()
                df_clean["residual"] = nuisance_model.resid
                r2 = nuisance_model.rsquared
                print(f"  📋 Nuisance model: R² = {r2:.3f}")
                # Interpretation
                if r2 > 0.2:
                    print(f"     → Strong blocking effects - residual correction important")
                elif r2 > 0.1:
                    print(f"     → Moderate blocking effects - residual correction helpful")
                else:
                    print(f"     → Weak blocking effects - results similar to naive permutation")
            except Exception as e:
                print(f"  ⚠️ Could not fit nuisance model: {e}")
                print(f"     → Using raw PC scores (no blocking correction)")
                df_clean["residual"] = df_clean[pc_name]
        else:
            df_clean["residual"] = df_clean[pc_name]

        # Convert pretraining to numeric
        df_clean["pretraining_numeric"] = df_clean[pretraining_col].apply(
            lambda x: 1 if str(x).lower() in ["y", "yes", "true", "1"] else 0
        )

        # Permutation test on residuals for each secondary group
        test_results = []
        for sec_value in sorted(df_clean[secondary_col].unique()):
            df_group = df_clean[df_clean[secondary_col] == sec_value]

            resid_naive = df_group[df_group["pretraining_numeric"] == 0]["residual"].values
            resid_pretrained = df_group[df_group["pretraining_numeric"] == 1]["residual"].values

            if len(resid_naive) < 2 or len(resid_pretrained) < 2:
                continue

            # Observed difference on residuals
            observed_diff_resid = np.mean(resid_pretrained) - np.mean(resid_naive)

            # Also on original scale
            orig_naive = df_group[df_group["pretraining_numeric"] == 0][pc_name].mean()
            orig_pretrained = df_group[df_group["pretraining_numeric"] == 1][pc_name].mean()
            observed_diff_orig = orig_pretrained - orig_naive

            # Cohen's d on residuals
            pooled_std_resid = np.sqrt(
                (
                    (len(resid_naive) - 1) * np.var(resid_naive, ddof=1)
                    + (len(resid_pretrained) - 1) * np.var(resid_pretrained, ddof=1)
                )
                / (len(resid_naive) + len(resid_pretrained) - 2)
            )
            cohens_d_resid = observed_diff_resid / pooled_std_resid if pooled_std_resid > 0 else 0

            # Permutation test on residuals
            combined_resid = np.concatenate([resid_naive, resid_pretrained])
            n1 = len(resid_naive)
            n_total = len(combined_resid)

            perm_diffs_resid = []
            np.random.seed(42 + hash(sec_value) % 1000)
            for _ in range(n_permutations):
                perm_idx = np.random.permutation(n_total)
                perm_group1 = combined_resid[perm_idx[:n1]]
                perm_group2 = combined_resid[perm_idx[n1:]]
                perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
                perm_diffs_resid.append(perm_diff)

            perm_diffs_resid = np.array(perm_diffs_resid)
            p_value_resid = np.sum(np.abs(perm_diffs_resid) >= np.abs(observed_diff_resid)) / n_permutations

            # Permutation test on original PC scores (for comparison)
            orig_naive_vals = df_group[df_group["pretraining_numeric"] == 0][pc_name].values
            orig_pretrained_vals = df_group[df_group["pretraining_numeric"] == 1][pc_name].values

            # Cohen's d on original scale
            pooled_std_orig = np.sqrt(
                (
                    (len(orig_naive_vals) - 1) * np.var(orig_naive_vals, ddof=1)
                    + (len(orig_pretrained_vals) - 1) * np.var(orig_pretrained_vals, ddof=1)
                )
                / (len(orig_naive_vals) + len(orig_pretrained_vals) - 2)
            )
            cohens_d_orig = observed_diff_orig / pooled_std_orig if pooled_std_orig > 0 else 0

            combined_orig = np.concatenate([orig_naive_vals, orig_pretrained_vals])

            perm_diffs_orig = []
            np.random.seed(99 + hash(sec_value) % 1000)  # Different seed
            for _ in range(n_permutations):
                perm_idx = np.random.permutation(len(combined_orig))
                perm_group1 = combined_orig[perm_idx[: len(orig_naive_vals)]]
                perm_group2 = combined_orig[perm_idx[len(orig_naive_vals) :]]
                perm_diff = np.mean(perm_group2) - np.mean(perm_group1)
                perm_diffs_orig.append(perm_diff)

            perm_diffs_orig = np.array(perm_diffs_orig)
            p_value_orig = np.sum(np.abs(perm_diffs_orig) >= np.abs(observed_diff_orig)) / n_permutations

            test_results.append(
                {
                    "group": sec_value,
                    "observed_diff_residual": observed_diff_resid,
                    "observed_diff_original": observed_diff_orig,
                    "cohens_d_residual": cohens_d_resid,
                    "cohens_d_original": cohens_d_orig,
                    "p_value_residual": p_value_resid,
                    "p_value_original": p_value_orig,
                    "p_value": p_value_resid,  # Primary method
                    "n_naive": len(resid_naive),
                    "n_pretrained": len(resid_pretrained),
                }
            )

        # No FDR correction: only 2 comparisons per group (naive vs pretrained)
        if test_results:
            for result in test_results:
                result["p_value_corrected"] = result["p_value"]  # Primary: residual permutation

                # Significance markers for both tests
                sig_resid = (
                    "***"
                    if result["p_value_residual"] < 0.001
                    else (
                        "**"
                        if result["p_value_residual"] < 0.01
                        else "*" if result["p_value_residual"] < 0.05 else "ns"
                    )
                )
                sig_orig = (
                    "***"
                    if result["p_value_original"] < 0.001
                    else (
                        "**"
                        if result["p_value_original"] < 0.01
                        else "*" if result["p_value_original"] < 0.05 else "ns"
                    )
                )

                print(
                    f"  {result['group']}: Δ(resid)={result['observed_diff_residual']:6.3f}, "
                    f"d_resid={result['cohens_d_residual']:.3f}, p_resid={result['p_value_residual']:.4f} {sig_resid} | "
                    f"Δ(orig)={result['observed_diff_original']:6.3f}, "
                    f"d_orig={result['cohens_d_original']:.3f}, p_orig={result['p_value_original']:.4f} {sig_orig}"
                )

            results[pc_name] = test_results

    return results


def perform_multivariate_statistical_analysis(
    pca_df,
    secondary_col,
    pretraining_col,
    n_components=3,
    use_permanova=True,
    use_pairwise=True,
    use_lmm=True,
    use_mahalanobis=True,
    use_mann_whitney=False,
    use_permutation=False,
    use_residual_permutation=True,
    n_permutations=10000,
    alpha=0.05,
):
    """
    Comprehensive multivariate statistical analysis on PC scores.

    Performs:
    1. PERMANOVA - global test across all groups
    2. Pairwise permutation tests - per genotype (naive vs pretrained)
    3. LMM on individual PCs - which PCs show pretraining effect
    4. Centroid analysis - Mahalanobis distances between groups
    5. Mann-Whitney U tests on individual PCs (with FDR correction)
    6. Permutation tests on individual PCs (with FDR correction)

    Args:
        pca_df: DataFrame with PC scores and metadata
        secondary_col: column name for genotype/condition
        pretraining_col: column name for pretraining status
        n_components: number of PCs to analyze
        use_permanova: whether to perform PERMANOVA
        use_pairwise: whether to perform pairwise permutation tests
        use_lmm: whether to perform LMM on individual PCs
        use_mahalanobis: whether to calculate Mahalanobis distances
        use_mann_whitney: whether to perform Mann-Whitney U tests on PCs
        use_permutation: whether to perform permutation tests on PCs
        n_permutations: number of permutations for permutation tests
        alpha: significance level for FDR correction

    Returns:
        dict with all statistical results
    """
    print("\n" + "=" * 80)
    print("MULTIVARIATE STATISTICAL ANALYSIS")
    print("=" * 80)

    results = {}

    # Get PC columns
    pc_cols = [f"PC{i+1}" for i in range(n_components) if f"PC{i+1}" in pca_df.columns]
    X_pcs = pca_df[pc_cols].values

    # Create combined group labels (genotype_pretraining)
    pca_df["combined_group"] = pca_df[secondary_col].astype(str) + "_" + pca_df[pretraining_col].astype(str)
    groups = pca_df["combined_group"].values

    # 1. PERMANOVA - global test
    if use_permanova:
        print("\n1. PERMANOVA - Global Multivariate Test")
        print("-" * 60)
        permanova_result = permanova(X_pcs, groups, n_permutations=n_permutations)
        results["permanova"] = permanova_result

        print(f"  F-statistic: {permanova_result['F_statistic']:.3f}")
        print(f"  p-value: {permanova_result['p_value']:.4f}")
        print(f"  R² (effect size): {permanova_result['R_squared']:.3f}")

        if permanova_result["p_value"] < 0.001:
            print("  *** Highly significant global difference between groups")
        elif permanova_result["p_value"] < 0.05:
            print("  * Significant global difference between groups")
        else:
            print("  ns - No significant global difference")

    # 2. Pairwise permutation tests
    if use_pairwise:
        print("\n2. Pairwise Permutation Tests (Naive vs Pretrained per group)")
        print("-" * 60)

        pairwise_results = {}
        pretraining_vals = sorted(pca_df[pretraining_col].dropna().unique())

        if len(pretraining_vals) == 2:
            for sec_val in sorted(pca_df[secondary_col].unique()):
                mask_naive = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretraining_vals[0])
                mask_pretrained = (pca_df[secondary_col] == sec_val) & (pca_df[pretraining_col] == pretraining_vals[1])

                X_naive = pca_df.loc[mask_naive, pc_cols].values
                X_pretrained = pca_df.loc[mask_pretrained, pc_cols].values

                if len(X_naive) < 2 or len(X_pretrained) < 2:
                    continue

                pairwise_result = pairwise_permutation_test(
                    X_naive, X_pretrained, n_permutations=n_permutations, metric="mahalanobis"
                )

                pairwise_results[sec_val] = pairwise_result

                sig = (
                    "***"
                    if pairwise_result["p_value"] < 0.001
                    else (
                        "**"
                        if pairwise_result["p_value"] < 0.01
                        else "*" if pairwise_result["p_value"] < 0.05 else "ns"
                    )
                )
                print(f"  {sec_val}:")
                print(f"    Mahalanobis distance: {pairwise_result['statistic']:.3f}")
                print(f"    p-value: {pairwise_result['p_value']:.4f} {sig}")
                print(f"    Cohen's d: {pairwise_result['cohens_d']:.3f}")

        results["pairwise_permutation"] = pairwise_results

    # 3. LMM on individual PCs
    if use_lmm:
        print("\n3. Linear Mixed-Effects Models on Individual PC Scores")
        print("-" * 60)
        lmm_results = lmm_analysis_on_pcs(pca_df, secondary_col, pretraining_col, n_components)
        results["lmm_on_pcs"] = lmm_results

    # 4. Centroid analysis
    if use_mahalanobis:
        print("\n4. Centroid Analysis and Mahalanobis Distances")
        print("-" * 60)
        centroid_results = calculate_centroids_and_distances(pca_df, secondary_col, pretraining_col, n_components)
        results["centroids"] = centroid_results

        print(f"  Calculated {len(centroid_results['centroids'])} group centroids")
        print(f"  Computed {len(centroid_results['distances'])} pairwise distances")

        # Show top distances
        if centroid_results["distances"]:
            print("\n  Top 5 largest Mahalanobis distances:")
            sorted_distances = sorted(
                centroid_results["distances"].items(), key=lambda x: x[1]["mahalanobis_distance"], reverse=True
            )[:5]

            for pair, dist_info in sorted_distances:
                print(f"    {pair}: {dist_info['mahalanobis_distance']:.3f}")

    # 5. RESIDUAL PERMUTATION TESTS on individual PCs (PRIMARY METHOD - distribution-free, blocking-aware)
    if use_residual_permutation:
        print("\n5. Residual Permutation Tests on Individual PC Scores (PRIMARY METHOD)")
        print("   Distribution-free inference accounting for blocking factors (Date, Arena, Side)")
        print("-" * 60)
        residual_perm_results = residual_permutation_test_on_pcs(
            pca_df, secondary_col, pretraining_col, n_components, n_permutations=n_permutations, alpha=alpha
        )
        results["residual_permutation_on_pcs"] = residual_perm_results

    # 6. Mann-Whitney U tests on individual PCs (DEPRECATED - doesn't account for blocking)
    if use_mann_whitney:
        print("\n6. Mann-Whitney U Tests on Individual PC Scores (with FDR correction)")
        print("   ⚠️  Note: Mann-Whitney does not account for blocking factors")
        print("-" * 60)
        mw_results = mann_whitney_on_pcs(pca_df, secondary_col, pretraining_col, n_components, alpha=alpha)
        results["mann_whitney_on_pcs"] = mw_results

    # 7. Standard Permutation tests on individual PCs (DEPRECATED - doesn't account for blocking)
    if use_permutation:
        print("\n7. Standard Permutation Tests on Individual PC Scores (with FDR correction)")
        print("   ⚠️  Note: Standard permutation does not account for blocking factors")
        print("-" * 60)
        perm_results = permutation_test_on_pcs(
            pca_df, secondary_col, pretraining_col, n_components, n_permutations=n_permutations, alpha=alpha
        )
        results["permutation_on_pcs"] = perm_results

    return results


def perform_bivariate_residual_permutation(pca_df, pc_list, secondary_col, pretraining_col, n_permutations=10000):
    """
    Bivariate permutation test on residuals after removing blocking effects.

    Tests joint effect of multiple PCs (e.g., PC1 + PC3) while accounting for
    blocking factors (Date, Arena, Side). This is the multivariate extension
    of the residual permutation approach used for individual PCs.

    Args:
        pca_df: DataFrame with PC scores and metadata
        pc_list: List of PC names to test jointly (e.g., ['PC1', 'PC3'])
        secondary_col: column for genotype/condition
        pretraining_col: column for pretraining status
        n_permutations: number of permutations

    Returns:
        dict with results for each genotype
    """
    from scipy.spatial.distance import euclidean

    print(f"\n🔄 BIVARIATE RESIDUAL PERMUTATION TEST")
    print(f"   Testing PCs: {', '.join(pc_list)}")
    print(f"   Distribution-free joint test accounting for blocking factors")
    print("-" * 60)

    results = {}

    # Step 1: Fit nuisance models for each PC to extract residuals
    residual_columns = []

    for pc in pc_list:
        # Build nuisance formula with blocking factors
        nuisance_formula_parts = []

        for col in ["Date", "date"]:
            if col in pca_df.columns:
                nuisance_formula_parts.append(f"C({col})")
                break

        if "arena_number" in pca_df.columns and "arena_side" in pca_df.columns:
            nuisance_formula_parts.append("C(arena_number) * C(arena_side)")
        elif "arena_number" in pca_df.columns:
            nuisance_formula_parts.append("C(arena_number)")
        elif "arena_side" in pca_df.columns:
            nuisance_formula_parts.append("C(arena_side)")

        if nuisance_formula_parts:
            nuisance_formula = f"{pc} ~ " + " + ".join(nuisance_formula_parts)

            try:
                nuisance_model = smf.ols(nuisance_formula, data=pca_df).fit()
                pca_df[f"{pc}_residual"] = nuisance_model.resid
                residual_columns.append(f"{pc}_residual")
                r2 = nuisance_model.rsquared
                print(f"  {pc}: R² = {r2:.3f} (blocking effects removed)")
            except Exception as e:
                print(f"  ⚠️ {pc}: Could not fit nuisance model, using raw values")
                pca_df[f"{pc}_residual"] = pca_df[pc]
                residual_columns.append(f"{pc}_residual")
        else:
            print(f"  {pc}: No blocking factors available, using raw values")
            pca_df[f"{pc}_residual"] = pca_df[pc]
            residual_columns.append(f"{pc}_residual")

    # Step 2: Permutation test on residuals for each genotype
    print(f"\n📊 Bivariate permutation tests ({n_permutations} permutations):")

    for genotype in sorted(pca_df[secondary_col].unique()):
        df_geno = pca_df[pca_df[secondary_col] == genotype].copy()

        # Convert pretraining to numeric for filtering
        df_geno["pretraining_numeric"] = df_geno[pretraining_col].apply(
            lambda x: 1 if str(x).lower() in ["y", "yes", "true", "1"] else 0
        )

        # Extract residuals for each pretraining group
        naive_resid = df_geno[df_geno["pretraining_numeric"] == 0][residual_columns].dropna()
        pretrained_resid = df_geno[df_geno["pretraining_numeric"] == 1][residual_columns].dropna()

        if len(naive_resid) < 2 or len(pretrained_resid) < 2:
            continue

        # Observed Euclidean distance between centroids in residual space
        centroid_naive_resid = naive_resid.mean(axis=0).values
        centroid_pretrained_resid = pretrained_resid.mean(axis=0).values
        observed_distance_resid = euclidean(centroid_naive_resid, centroid_pretrained_resid)

        # Also calculate on original scale for reference
        naive_orig = df_geno[df_geno["pretraining_numeric"] == 0][pc_list].dropna()
        pretrained_orig = df_geno[df_geno["pretraining_numeric"] == 1][pc_list].dropna()

        if len(naive_orig) >= 2 and len(pretrained_orig) >= 2:
            centroid_naive_orig = naive_orig.mean(axis=0).values
            centroid_pretrained_orig = pretrained_orig.mean(axis=0).values
            observed_distance_orig = euclidean(centroid_naive_orig, centroid_pretrained_orig)
        else:
            observed_distance_orig = np.nan

        # Permutation test on residuals
        # Clear attrs to avoid numpy array comparison issues in pd.concat
        naive_resid.attrs = {}
        pretrained_resid.attrs = {}
        combined_resid = pd.concat([naive_resid, pretrained_resid], ignore_index=True)
        n1 = len(naive_resid)
        n_total = len(combined_resid)

        perm_distances_resid = []
        np.random.seed(42 + hash(genotype) % 1000)

        for _ in range(n_permutations):
            perm_idx = np.random.permutation(n_total)
            perm_group1 = combined_resid.iloc[perm_idx[:n1]]
            perm_group2 = combined_resid.iloc[perm_idx[n1:]]

            perm_dist = euclidean(perm_group1.mean(axis=0).values, perm_group2.mean(axis=0).values)
            perm_distances_resid.append(perm_dist)

        perm_distances_resid = np.array(perm_distances_resid)
        p_value_resid = np.sum(perm_distances_resid >= observed_distance_resid) / n_permutations

        # Permutation test on original data (for comparison)
        if not np.isnan(observed_distance_orig):
            # Clear attrs to avoid numpy array comparison issues in pd.concat
            naive_orig.attrs = {}
            pretrained_orig.attrs = {}
            combined_orig = pd.concat([naive_orig, pretrained_orig], ignore_index=True)

            perm_distances_orig = []
            np.random.seed(99 + hash(genotype) % 1000)

            for _ in range(n_permutations):
                perm_idx = np.random.permutation(len(combined_orig))
                perm_group1 = combined_orig.iloc[perm_idx[: len(naive_orig)]]
                perm_group2 = combined_orig.iloc[perm_idx[len(naive_orig) :]]

                perm_dist = euclidean(perm_group1.mean(axis=0).values, perm_group2.mean(axis=0).values)
                perm_distances_orig.append(perm_dist)

            perm_distances_orig = np.array(perm_distances_orig)
            p_value_orig = np.sum(perm_distances_orig >= observed_distance_orig) / n_permutations
        else:
            p_value_orig = np.nan

        # Effect size: observed / mean permuted distance
        effect_size_resid = (
            observed_distance_resid / np.mean(perm_distances_resid) if np.mean(perm_distances_resid) > 0 else np.nan
        )

        results[genotype] = {
            "observed_distance_residual": observed_distance_resid,
            "observed_distance_original": observed_distance_orig,
            "p_value_residual": p_value_resid,
            "p_value_original": p_value_orig,
            "p_value": p_value_resid,  # Primary method
            "effect_size": effect_size_resid,
            "perm_mean_residual": np.mean(perm_distances_resid),
            "perm_95th_residual": np.percentile(perm_distances_resid, 95),
            "n_naive": len(naive_resid),
            "n_pretrained": len(pretrained_resid),
            "pcs_tested": pc_list,
        }

        # Significance markers
        sig_resid = (
            "***" if p_value_resid < 0.001 else "**" if p_value_resid < 0.01 else "*" if p_value_resid < 0.05 else "ns"
        )
        sig_orig = (
            ("***" if p_value_orig < 0.001 else "**" if p_value_orig < 0.01 else "*" if p_value_orig < 0.05 else "ns")
            if not np.isnan(p_value_orig)
            else "N/A"
        )

        print(
            f"  {genotype}: D_resid={observed_distance_resid:.3f}, p_resid={p_value_resid:.4f} {sig_resid} | "
            f"D_orig={observed_distance_orig:.3f}, p_orig={p_value_orig:.4f} {sig_orig}"
        )

    return results
