#!/usr/bin/env python3

"""Factorial design analysis for BallScents experiments - IMPROVED VERSION

Key improvements:
- Mixed-effects models with random effects for Date, Arena, VideoID
- Z-score normalization for cross-metric comparability
- Enhanced visual clarity with better heatmaps
- Standard significance notation (*, **, ***)
- Summary visualizations for quick interpretation
- Better control of known sources of variability
"""

import sys
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm
import warnings

warnings.filterwarnings("ignore")

# Add src to path for reusing helpers
sys.path.append(str(Path(__file__).parent.parent))

try:
    from PCA.plot_detailed_metric_statistics import METRIC_DISPLAY_NAMES as PCA_METRIC_DISPLAY_NAMES
    from PCA.plot_detailed_metric_statistics import METRICS_PATH as PCA_METRICS_PATH
except Exception:
    PCA_METRIC_DISPLAY_NAMES = {}
    PCA_METRICS_PATH = None


def get_display_name(metric_name):
    """Get PCA display name for a metric if available."""
    return PCA_METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


def load_canonical_metric_order():
    """Load the canonical metric order from PCA metric lists."""
    base = Path(__file__).parent.parent / "PCA" / "metrics_lists"
    candidates = [base / "canonical_metrics_order.txt"]
    if PCA_METRICS_PATH:
        candidates.append(Path(PCA_METRICS_PATH))
    candidates += [base / "final_metrics_for_pca.txt", base / "final_metrics_for_pca_list.txt"]

    for p in candidates:
        try:
            if p and p.exists():
                with p.open("r") as fh:
                    lines = [l.strip() for l in fh if l.strip()]
                    if lines:
                        print(f"  ✓ Loaded canonical metric order from: {p.name}")
                        return lines
        except Exception:
            continue
    print("  ⚠ Could not load canonical metric order - using all available metrics")
    return []


def encode_factorial_design(data, group_col="BallScent"):
    """
    Add factorial design columns to the dataset with validation.

    Returns: DataFrame with Is_Washed, Is_New, Pre_Exposed columns (0/1)
    """
    design_map = {
        "Ctrl": {"Is_Washed": 0, "Is_New": 0, "Pre_Exposed": 0},
        "CtrlScent": {"Is_Washed": 0, "Is_New": 0, "Pre_Exposed": 1},
        "Washed": {"Is_Washed": 1, "Is_New": 0, "Pre_Exposed": 0},
        "Scented": {"Is_Washed": 1, "Is_New": 0, "Pre_Exposed": 1},
        "New": {"Is_Washed": 0, "Is_New": 1, "Pre_Exposed": 0},
        "NewScent": {"Is_Washed": 0, "Is_New": 1, "Pre_Exposed": 1},
    }

    data = data.copy()

    # Encode factorial design
    data["Is_Washed"] = data[group_col].map(lambda x: design_map.get(x, {}).get("Is_Washed", np.nan))
    data["Is_New"] = data[group_col].map(lambda x: design_map.get(x, {}).get("Is_New", np.nan))
    data["Pre_Exposed"] = data[group_col].map(lambda x: design_map.get(x, {}).get("Pre_Exposed", np.nan))

    # Report encoding
    print(f"\\n{'='*60}")
    print("FACTORIAL DESIGN ENCODING")
    print(f"{'='*60}")

    for cond in sorted(design_map.keys()):
        counts = (data[group_col] == cond).sum()
        factors = design_map[cond]
        print(
            f"  {cond:12s} (n={counts:4d}): W={factors['Is_Washed']} N={factors['Is_New']} P={factors['Pre_Exposed']}"
        )

    # Check for unmapped values
    unmapped = data[data["Is_Washed"].isna()][group_col].unique()
    if len(unmapped) > 0:
        n_unmapped = data[data["Is_Washed"].isna()].shape[0]
        print(f"\\n  ⚠ Unmapped values found: {list(unmapped)} ({n_unmapped} rows)")
        print(f"  → These will be excluded from analysis")
        data = data.dropna(subset=["Is_Washed", "Is_New", "Pre_Exposed"])

    # Ensure integer encoding
    data["Is_Washed"] = data["Is_Washed"].astype(int)
    data["Is_New"] = data["Is_New"].astype(int)
    data["Pre_Exposed"] = data["Pre_Exposed"].astype(int)

    print(f"\\n  ✓ Final dataset: {data.shape[0]} rows encoded")

    return data


def fit_factorial_model(data, metric):
    """
    Fit mixed-effects model with random effects for Date, Arena, and VideoID.

    Fixed effects: Full factorial design (Is_Washed, Is_New, Pre_Exposed + interactions)
    Random effects: Date, Arena, VideoID (nested within Date and Arena)

    Returns: dict with coefficients, p-values, and diagnostics
    """
    # Identify required columns for random effects
    random_cols = []
    if "Date" in data.columns:
        random_cols.append("Date")
    if "Arena" in data.columns:
        random_cols.append("Arena")
    if "VideoID" in data.columns:
        random_cols.append("VideoID")

    # Prepare data with fixed and random effects
    required_cols = ["Is_Washed", "Is_New", "Pre_Exposed", metric] + random_cols
    model_data = data[required_cols].dropna()

    # More stringent sample size check
    if len(model_data) < 20:
        return None

    # Check for sufficient variation in the metric
    if model_data[metric].std() == 0:
        return None

    try:
        # Build formula for fixed effects (full factorial)
        formula = f"{metric} ~ Is_Washed + Is_New + Pre_Exposed + Is_Washed:Is_New + Is_Washed:Pre_Exposed + Is_New:Pre_Exposed + Is_Washed:Is_New:Pre_Exposed"

        # Try mixed-effects model with Date as random effect (simpler, more stable)
        if "Date" in random_cols:
            model_data = model_data.copy()
            groups = model_data["Date"]
            n_groups = len(groups.unique())

            # Only use mixed model if we have enough groups (at least 3)
            if n_groups >= 3:
                try:
                    model = mixedlm(formula, data=model_data, groups=groups).fit(method="lbfgs", maxiter=200, reml=True)
                    model_type = "mixed_effects"
                except:
                    # Fallback to OLS on any error
                    model = ols(formula, data=model_data).fit()
                    model_type = "ols_fallback"
            else:
                model = ols(formula, data=model_data).fit()
                model_type = "ols"
        else:
            # No random effects available, use OLS
            model = ols(formula, data=model_data).fit()
            model_type = "ols"

        # Extract results
        results = {
            "metric": metric,
            "n_samples": len(model_data),
            "model_type": model_type,
        }

        # Add R-squared if available (OLS only)
        if hasattr(model, "rsquared"):
            results["r_squared"] = model.rsquared
            results["r_squared_adj"] = model.rsquared_adj
        else:
            results["r_squared"] = np.nan
            results["r_squared_adj"] = np.nan

        # Add F-statistic if available
        if hasattr(model, "fvalue"):
            results["f_statistic"] = model.fvalue
            results["f_pvalue"] = model.f_pvalue
        else:
            results["f_statistic"] = np.nan
            results["f_pvalue"] = np.nan

        # Extract term statistics
        for term in model.params.index:
            if term == "Intercept" or term == "Group Var":
                results["Intercept"] = model.params.get("Intercept", np.nan)
                continue

            clean_term = term.replace(":", "_x_")
            results[f"{clean_term}_coef"] = model.params[term]
            results[f"{clean_term}_pval"] = model.pvalues[term]
            results[f"{clean_term}_stderr"] = model.bse[term]
            results[f"{clean_term}_tstat"] = model.tvalues[term]

        return results

    except Exception as e:
        # Try fallback to OLS if mixed model fails
        try:
            model = ols(formula, data=model_data).fit()
            results = {
                "metric": metric,
                "n_samples": len(model_data),
                "r_squared": model.rsquared,
                "r_squared_adj": model.rsquared_adj,
                "f_statistic": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "model_type": "ols_fallback",
            }

            for term in model.params.index:
                if term == "Intercept":
                    results["Intercept"] = model.params[term]
                    continue
                clean_term = term.replace(":", "_x_")
                results[f"{clean_term}_coef"] = model.params[term]
                results[f"{clean_term}_pval"] = model.pvalues[term]
                results[f"{clean_term}_stderr"] = model.bse[term]
                results[f"{clean_term}_tstat"] = model.tvalues[term]

            return results
        except Exception as e2:
            print(f"  ⚠ Failed to fit {metric}: {e2}")


def analyze_factorial_design(data, metrics, output_dir):
    """Run factorial analysis for all metrics."""
    print(f"\\n{'='*60}")
    print("FACTORIAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Model: metric ~ W + N + P + W:N + W:P + N:P + W:N:P")
    print(f"Analyzing {len(metrics)} metrics...")

    all_results = []

    for i, metric in enumerate(metrics, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Progress: {i}/{len(metrics)}", end="\\r")

        result = fit_factorial_model(data, metric)
        if result:
            all_results.append(result)

    print(f"\\n  ✓ Successfully fit {len(all_results)}/{len(metrics)} models")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("\n  ✗ No valid results obtained")
        return results_df

    # Report model types used
    if "model_type" in results_df.columns:
        model_type_counts = results_df["model_type"].value_counts()
        print(f"\n  Model types used:")
        for model_type, count in model_type_counts.items():
            print(f"    {model_type}: {count} metrics")

    # Apply FDR correction
    print(f"\\n{'='*60}")
    print("FDR CORRECTION (Benjamini-Hochberg)")
    print(f"{'='*60}")

    pval_cols = [col for col in results_df.columns if col.endswith("_pval") and not col.startswith("f_")]

    for pval_col in pval_cols:
        term_name = pval_col.replace("_pval", "")
        pvals = results_df[pval_col].values

        rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

        results_df[f"{term_name}_pval_fdr"] = pvals_corrected
        results_df[f"{term_name}_significant_fdr"] = rejected

        n_sig_raw = (pvals < 0.05).sum()
        n_sig_fdr = rejected.sum()
        print(f"  {term_name.replace('_x_', ':'):30s}: {n_sig_raw:3d} → {n_sig_fdr:3d} significant (FDR)")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_file = output_dir / "factorial_analysis_statistics.csv"
    results_df.to_csv(stats_file, index=False)
    print(f"\\n  ✓ Statistics saved: {stats_file.name}")

    # Also save to canonical location
    try:
        canonical_dir = Path("/mnt/upramdya_data/MD/Ball_scents/Plots/summaries/Factorial_Analysis")
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_file = canonical_dir / "factorial_analysis_statistics.csv"
        results_df.to_csv(canonical_file, index=False)
        print(f"  ✓ Also saved to: {canonical_file}")
    except Exception:
        pass

    return results_df


def create_improved_heatmaps(results_df, output_dir, pass_results_df=None):
    """
    Create enhanced heatmaps with z-score normalization for better readability.
    """
    print(f"\\n{'='*60}")
    print("GENERATING ENHANCED HEATMAPS")
    print(f"{'='*60}")

    output_dir = Path(output_dir)

    # Define factorial terms
    main_effects = ["Is_Washed", "Is_New", "Pre_Exposed"]
    two_way = ["Is_Washed_x_Is_New", "Is_Washed_x_Pre_Exposed", "Is_New_x_Pre_Exposed"]
    three_way = ["Is_Washed_x_Is_New_x_Pre_Exposed"]
    all_terms = main_effects + two_way + three_way

    metrics = results_df["metric"].values

    # Build raw effect matrix
    effect_matrix = pd.DataFrame(index=all_terms, columns=metrics, dtype=float)
    significance_matrix = pd.DataFrame(index=all_terms, columns=metrics, dtype=bool)

    for term in all_terms:
        coef_col = f"{term}_coef"
        sig_col = f"{term}_significant_fdr"

        if coef_col in results_df.columns:
            effect_matrix.loc[term, :] = results_df[coef_col].values
        if sig_col in results_df.columns:
            significance_matrix.loc[term, :] = results_df[sig_col].values

    effect_matrix = effect_matrix.astype(float)

    # Build p-value lookup for significance annotation
    pval_lookup = {}
    for term in all_terms:
        pval_col = f"{term}_pval_fdr"
        if pval_col in results_df.columns:
            for i, row in results_df.iterrows():
                metric = row["metric"]
                pval = row[pval_col]
                pval_lookup[(term, metric)] = pval

    # Normalize significance matrix NaNs -> False so checks are reliable
    significance_matrix = significance_matrix.fillna(False).astype(bool)

    # Diagnostic: show how many p-values were found
    n_pvals = sum(1 for v in pval_lookup.values() if np.isfinite(v))
    print(f"\n  p-value lookup entries: {len(pval_lookup)} (finite: {n_pvals})")
    # Show a small sample of p-values for debugging
    sample_items = list(pval_lookup.items())[:6]
    if sample_items:
        print("  Sample pvals:")
        for (t, m), pv in sample_items:
            print(f"    ({t}, {m}): {pv}")

    # Print diagnostics
    print(f"\n  Effect matrix shape: {effect_matrix.shape}")
    print(f"  Effect size range: [{effect_matrix.min().min():.3f}, {effect_matrix.max().max():.3f}]")
    print(f"  Number of significant effects: {significance_matrix.sum().sum()}")
    print(f"  Significant effects by term:")
    for term in all_terms:
        n_sig = significance_matrix.loc[term, :].sum()
        print(f"    {term}: {n_sig}/{len(metrics)}")

    # Remove terms that were never estimated (all NaN) to avoid empty rows
    valid_terms = [t for t in all_terms if not effect_matrix.loc[t].isnull().all()]
    if len(valid_terms) != len(all_terms):
        removed = sorted(set(all_terms) - set(valid_terms))
        print(f"  Info: removed untested terms: {removed}")
        effect_matrix = effect_matrix.loc[valid_terms]
        significance_matrix = significance_matrix.loc[valid_terms]
        # update term groups
        main_effects = [t for t in main_effects if t in effect_matrix.index]
        two_way = [t for t in two_way if t in effect_matrix.index]
        three_way = [t for t in three_way if t in effect_matrix.index]
        all_terms = valid_terms

    # Apply canonical metric ordering and re-evaluate metric/term lists
    ordered_metrics = _apply_canonical_ordering(effect_matrix.columns)
    effect_matrix = effect_matrix[ordered_metrics]
    significance_matrix = significance_matrix[ordered_metrics]

    # Recompute metric and term lists from the final matrices to avoid stale values
    metrics = list(effect_matrix.columns)
    all_terms = list(effect_matrix.index)

    # ═══════════════════════════════════════════════════════════
    # 1. Z-SCORED HEATMAP (main visualization for readability)
    # ═══════════════════════════════════════════════════════════
    print("\\n  Creating Z-scored heatmap (normalized by metric)...")

    # Z-score each metric (column) independently
    effect_matrix_zscore = effect_matrix.apply(lambda col: (col - col.mean()) / col.std(), axis=0)

    # Create figure
    n_metrics = len(metrics)
    n_terms = len(all_terms)
    fig_width = max(16, n_metrics * 0.5)
    fig_height = max(8, n_terms * 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use diverging colormap centered at 0
    sns.heatmap(
        effect_matrix_zscore,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={"label": "Z-score (effect size normalized by metric)", "shrink": 0.8},
        linewidths=0.3,
        linecolor="lightgray",
        robust=True,
    )

    # Add significance stars with standard notation (*, **, ***) using FDR p-values
    for i, term in enumerate(effect_matrix_zscore.index):
        for j, metric in enumerate(effect_matrix_zscore.columns):
            # Prefer FDR p-values from the lookup; fallback to significance matrix
            pval = pval_lookup.get((term, metric), np.nan)
            is_sig = False
            if np.isfinite(pval):
                is_sig = pval < 0.05
            else:
                # fallback: use boolean significance matrix if lookup missing
                try:
                    is_sig = bool(significance_matrix.loc[term, metric])
                except Exception:
                    is_sig = False

            if is_sig:
                if np.isfinite(pval):
                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    else:
                        stars = "*"
                else:
                    stars = "*"

                z_val = effect_matrix_zscore.loc[term, metric]
                color = "white" if (pd.notnull(z_val) and abs(z_val) > 1.5) else "black"
                ax.text(j + 0.5, i + 0.5, stars, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # Format labels
    display_names = [get_display_name(m) for m in effect_matrix_zscore.columns]
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)

    # Clean term names
    term_labels = [t.replace("_x_", " × ").replace("Is_", "").replace("_", " ") for t in all_terms]
    ax.set_yticklabels(term_labels, rotation=0, fontsize=11)

    ax.set_xlabel("Metric", fontsize=13, fontweight="bold")
    ax.set_ylabel("Factorial Term", fontsize=13, fontweight="bold")

    title = "Factorial Analysis: Standardized Effect Sizes (Mixed-Effects Model)\n"
    title += "Z-scores show relative strength within each metric | *p<0.05, **p<0.01, ***p<0.001 (FDR-corrected)"
    ax.set_title(title, fontsize=14, pad=20, fontweight="bold")

    plt.tight_layout()

    out_png = output_dir / "factorial_effects_ZSCORE_heatmap.png"
    out_pdf = output_dir / "factorial_effects_ZSCORE_heatmap.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"    ✓ {out_png.name}")
    print(f"    ✓ {out_pdf.name}")

    # ═══════════════════════════════════════════════════════════
    # 2. RAW COEFFICIENTS HEATMAP (for absolute effect sizes)
    # ═══════════════════════════════════════════════════════════
    print("\\n  Creating raw coefficient heatmap...")

    max_abs = np.abs(effect_matrix.values).max()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        effect_matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        cbar_kws={"label": "β coefficient (raw effect size)", "shrink": 0.8},
        linewidths=0.3,
        linecolor="lightgray",
    )

    # Add significance with standard notation
    for i, term in enumerate(effect_matrix.index):
        for j, metric in enumerate(effect_matrix.columns):
            # Determine significance via pval_lookup (preferred) or significance_matrix
            pval = pval_lookup.get((term, metric), np.nan)
            sig = False
            if np.isfinite(pval):
                sig = pval < 0.05
            else:
                sig = bool(significance_matrix.loc[term, metric])

            if sig:
                if np.isfinite(pval):
                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    else:
                        stars = "*"
                else:
                    stars = "*"

                bg_value = effect_matrix.loc[term, metric]
                color = "white" if abs(bg_value) > 0.5 * max_abs else "black"
                ax.text(j + 0.5, i + 0.5, stars, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(term_labels, rotation=0, fontsize=11)
    ax.set_xlabel("Metric", fontsize=13, fontweight="bold")
    ax.set_ylabel("Factorial Term", fontsize=13, fontweight="bold")
    ax.set_title(
        "Factorial Analysis: Raw Effect Sizes (β coefficients)\n*p<0.05, **p<0.01, ***p<0.001 (FDR-corrected)",
        fontsize=14,
        pad=20,
        fontweight="bold",
    )

    plt.tight_layout()

    out_png_raw = output_dir / "factorial_effects_RAW_heatmap.png"
    out_pdf_raw = output_dir / "factorial_effects_RAW_heatmap.pdf"
    fig.savefig(out_png_raw, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf_raw, bbox_inches="tight")
    plt.close()

    print(f"    ✓ {out_png_raw.name}")

    # ═══════════════════════════════════════════════════════════
    # 3. MAIN EFFECTS ONLY (cleaner view)
    # ═══════════════════════════════════════════════════════════
    print("\\n  Creating main effects heatmap...")

    # Recompute available main effects (some may have been removed)
    main_effects = [t for t in main_effects if t in effect_matrix_zscore.index]
    main_effects_zscore = effect_matrix_zscore.loc[main_effects, :]
    main_sig = significance_matrix.loc[main_effects, :]

    fig_height_main = max(5, len(main_effects) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height_main))

    sns.heatmap(
        main_effects_zscore,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={"label": "Z-score (normalized effect)", "shrink": 0.9},
        linewidths=0.5,
        linecolor="white",
        robust=True,
    )

    # Add stars with standard notation
    for i, term in enumerate(main_effects_zscore.index):
        for j, metric in enumerate(main_effects_zscore.columns):
            if main_sig.loc[term, metric]:
                # Get p-value and determine significance level
                pval = pval_lookup.get((term, metric), 1.0)
                if pval < 0.001:
                    stars = "***"
                elif pval < 0.01:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                else:
                    stars = ""

                if stars:
                    z_val = main_effects_zscore.loc[term, metric]
                    color = "white" if abs(z_val) > 1.5 else "black"
                    ax.text(
                        j + 0.5, i + 0.5, stars, ha="center", va="center", color=color, fontsize=12, fontweight="bold"
                    )

    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(["Washed", "New", "Pre-Exposed"], rotation=0, fontsize=13, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=13, fontweight="bold")
    ax.set_ylabel("Treatment Factor", fontsize=13, fontweight="bold")
    ax.set_title(
        "Main Effects: Treatment Impact (Z-scored)\n*p<0.05, **p<0.01, ***p<0.001 (FDR-corrected)",
        fontsize=15,
        pad=20,
        fontweight="bold",
    )

    plt.tight_layout()

    out_png_main = output_dir / "factorial_MAIN_EFFECTS_zscore.png"
    out_pdf_main = output_dir / "factorial_MAIN_EFFECTS_zscore.pdf"
    fig.savefig(out_png_main, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf_main, bbox_inches="tight")
    plt.close()

    print(f"    ✓ {out_png_main.name}")

    # ═══════════════════════════════════════════════════════════
    # 4. SUMMARY BAR CHART (overview of significant effects)
    # ═══════════════════════════════════════════════════════════
    print("\\n  Creating summary overview...")

    _create_summary_chart(significance_matrix, output_dir)

    # Print statistics summary
    _print_effect_summary(effect_matrix, significance_matrix, metrics)


def _apply_canonical_ordering(metrics):
    """Apply canonical metric order if available."""
    canonical = load_canonical_metric_order()
    if canonical:
        ordered = [m for m in canonical if m in metrics]
        remaining = [m for m in metrics if m not in ordered]
        return ordered + remaining
    return list(metrics)


def _create_summary_chart(significance_matrix, output_dir):
    """Create a summary bar chart showing count of significant effects per term."""

    sig_counts = significance_matrix.sum(axis=1)
    total_metrics = significance_matrix.shape[1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e74c3c" if "x" not in idx else "#3498db" for idx in sig_counts.index]

    bars = ax.barh(range(len(sig_counts)), sig_counts.values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels
    for i, (idx, val) in enumerate(sig_counts.items()):
        pct = 100 * val / total_metrics
        ax.text(val + 0.5, i, f"{val} ({pct:.0f}%)", va="center", fontsize=11, fontweight="bold")

    # Format
    term_labels = [t.replace("_x_", " × ").replace("Is_", "").replace("_", " ") for t in sig_counts.index]
    ax.set_yticks(range(len(sig_counts)))
    ax.set_yticklabels(term_labels, fontsize=12)
    ax.set_xlabel("Number of Significant Metrics (FDR q<0.05)", fontsize=13, fontweight="bold")
    ax.set_title(f"Summary: Significant Effects Across {total_metrics} Metrics", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max(sig_counts.values) * 1.15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_file = output_dir / "factorial_SUMMARY_significant_counts.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"    ✓ {out_file.name}")


def _print_effect_summary(effect_matrix, significance_matrix, metrics):
    """Print summary statistics."""
    print(f"\\n{'='*60}")
    print("EFFECT SIZE SUMMARY")
    print(f"{'='*60}")

    for term in effect_matrix.index:
        n_sig = significance_matrix.loc[term, :].sum()
        pct_sig = 100 * n_sig / len(metrics)
        mean_abs = np.abs(effect_matrix.loc[term, :]).mean()
        median_abs = np.abs(effect_matrix.loc[term, :]).median()

        term_display = term.replace("_x_", ":").replace("Is_", "")
        print(
            f"  {term_display:30s}: {n_sig:3d}/{len(metrics):3d} sig ({pct_sig:5.1f}%) | mean|β|={mean_abs:.3f}, med|β|={median_abs:.3f}"
        )


def get_continuous_metrics(data):
    """Identify continuous metrics suitable for linear modeling."""
    exclude_patterns = [
        "BallScent",
        "Genotype",
        "Nickname",
        "Driver",
        "Date",
        "Arena",
        "Experiment",
        "Split",
        "FeedingState",
        "Brain region",
        "Is_Washed",
        "Is_New",
        "Pre_Exposed",
        "binned_",
        "_bin_",
        "r2",
        "slope",
        "logistic_",
    ]

    candidates = []
    for col in data.columns:
        if any(pat in col for pat in exclude_patterns):
            continue
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue

        # Skip binary metrics
        unique_vals = data[col].dropna().unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            continue

        candidates.append(col)

    print(f"\\n  Found {len(candidates)} continuous metrics")

    # Filter to canonical PCA metrics
    canonical_metrics = load_canonical_metric_order()
    if canonical_metrics:
        filtered = [m for m in candidates if m in canonical_metrics]
        print(f"  Filtered to {len(filtered)} canonical PCA metrics")
        candidates = filtered

    return candidates


def load_dataset(test_mode=False):
    """Load and prepare the BallScents dataset."""
    dataset_path = Path(
        "/mnt/upramdya_data/MD/Ball_scents/Datasets/251103_10_summary_ballscents_Data/summary/pooled_summary.feather"
    )

    print(f"\\nLoading dataset from: {dataset_path.name}")
    try:
        data = pd.read_feather(dataset_path)
        print(f"  ✓ Main dataset loaded: {data.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load additional Ctrl cohorts
    print("\\nLoading additional Ctrl cohorts...")
    ctrl_paths = [
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250815_18_summary_control_folders_Data/summary/230704_FeedingState_1_AM_Videos_Tracked_summary.feather",
        "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250815_18_summary_control_folders_Data/summary/230705_FeedingState_2_AM_Videos_Tracked_summary.feather",
    ]

    ctrl_dfs = []
    for ctrl_path in ctrl_paths:
        try:
            ctrl_df = pd.read_feather(ctrl_path)
            if "FeedingState" in ctrl_df.columns:
                ctrl_df = ctrl_df[ctrl_df["FeedingState"] == "starved_noWater"].copy()
            ctrl_df["BallScent"] = "Ctrl"
            ctrl_dfs.append(ctrl_df)
            print(f"  ✓ Loaded Ctrl cohort: {ctrl_df.shape}")
        except Exception as e:
            print(f"  ⚠ Could not load {Path(ctrl_path).name}: {e}")

    if ctrl_dfs:
        data = pd.concat([data] + ctrl_dfs, ignore_index=True, sort=False)
        print(f"  ✓ Combined dataset: {data.shape}")

    if test_mode:
        print(f"\\n  🧪 TEST MODE: Sampling 500 rows")
        data = data.sample(n=min(500, len(data)), random_state=42)

    return data


def main(test_mode=False):
    """Main analysis function."""
    print(f"\\n{'='*70}")
    print("FACTORIAL DESIGN ANALYSIS - IMPROVED VERSION")
    print(f"{'='*70}")
    print("\\nDesign: 2×2×2 factorial")
    print("  • Is_Washed (ball washed)")
    print("  • Is_New (ball new/unused)")
    print("  • Pre_Exposed (ball pre-exposed to flies)")
    print("\\nKey improvements:")
    print("  ✓ Z-score normalization for cross-metric comparability")
    print("  ✓ Enhanced visual clarity with better heatmaps")
    print("  ✓ Summary charts for quick interpretation")

    # Load and prepare data
    data = load_dataset(test_mode=test_mode)
    data = encode_factorial_design(data, group_col="BallScent")

    # Get metrics
    metrics = get_continuous_metrics(data)

    if test_mode:
        print(f"\\n  🧪 TEST MODE: Limiting to first 10 metrics")
        metrics = metrics[:10]

    print(f"\\nAnalyzing {len(metrics)} metrics")

    # Set up output
    output_dir = Path("/mnt/upramdya_data/MD/Ball_scents/Plots/factorial_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    results_df = analyze_factorial_design(data, metrics, output_dir)

    if not results_df.empty:
        create_improved_heatmaps(results_df, output_dir)

        print(f"\\n{'='*70}")
        print("✓ ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"  Output directory: {output_dir}")
        print(f"  Analyzed: {len(results_df)} metrics")
        print(f"  Mean R²: {results_df['r_squared'].mean():.3f}")
    else:
        print("\\n  ✗ No results obtained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factorial design analysis for BallScents (improved)")
    parser.add_argument("--test", action="store_true", help="Test mode: sample data and limit metrics")
    args = parser.parse_args()
    main(test_mode=args.test)
