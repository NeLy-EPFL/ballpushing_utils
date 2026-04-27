#!/usr/bin/env python3
"""
Linear Mixed-Effects analysis per metric (without Mann-Whitney)

This script:
- Loads a dataset of behavioral metrics with categorical factors (Light, FeedingState, Period, Date) and a fly identifier.
- For each continuous metric, fits a linear mixed-effects model:
    outcome ~ C(Light) + C(FeedingState) + C(Period)
  with a random intercept for Fly (and a sensitivity model with Date as the grouping factor).
- Applies transformation heuristics for skewed outcomes (log1p/log).
- Outputs:
    - CSV of raw fixed-effects estimates per metric (coef, SE, z, p, CI)
    - CSV with FDR-corrected q-values per factor family
    - Markdown report summarizing significant terms

Usage:
  python run_lmm_metrics.py
  python run_lmm_metrics.py --test --no-overwrite
"""

import sys
import time
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns


import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests
from io import StringIO

# =========================
# Configuration (EDIT ME)
# =========================

# TODO: set your dataset path (feather or parquet highly recommended for speed)
DATASET_PATH = "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/summary/pooled_summary.feather"

# Output base directory
BASE_OUTPUT_DIR = "/mnt/upramdya_data/MD/Ballpushing_Exploration/Plots/Summary_metrics/250815_LMM_Models"

# Identify the column names in your dataset
LIGHT_COL = "Light"
FEEDING_COL = "FeedingState"
PERIOD_COL = "Period"  # keep this column; do not drop it during cleaning
DATE_COL = "Date"  # as factor (category) or as grouping for sensitivity model
FLY_COL = "fly"  # TODO: change to your actual fly ID column (e.g., "Nickname", "FlyID", etc.)

# Fixed effects to include (only included if present in the data)
FIXED_EFFECTS = [LIGHT_COL, FEEDING_COL, PERIOD_COL]

# Random effects: primary model grouping (Fly) and optional sensitivity (Date)
GROUP_PRIMARY = FLY_COL
GROUP_SENSITIVITY = DATE_COL  # set to None to skip sensitivity model

# Minimum observations required to fit a model for a metric
MIN_OBS_PER_MODEL = 50

# Transformation strategy heuristic for outcomes:
# Automatically chooses log1p/log when right-skewed and non-negative/positive
ENABLE_TRANSFORM = True

# Multiple testing
FDR_METHOD = "fdr_bh"
ALPHA = 0.05

# Test mode (subset rows and metrics to run fast)
DEFAULT_TEST_MODE = False
TEST_SAMPLE_SIZE = 200
TEST_MAX_METRICS = 3


# =========================
# Utilities
# =========================


def load_dataset(test_mode=False, test_sample_size=200):
    """Load dataset from DATASET_PATH and perform minimal cleaning
    preserving the needed modeling columns."""
    print(f"Loading dataset from: {DATASET_PATH}")
    p = Path(DATASET_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # Load based on extension
    if p.suffix == ".feather":
        df = pd.read_feather(p)
    elif p.suffix in [".parquet"]:
        df = pd.read_parquet(p)
    elif p.suffix in [".csv", ".tsv"]:
        df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_csv(p, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    print(f"✅ Loaded dataset: shape={df.shape}")

    # Minimal cleaning:
    # - Ensure Light is not empty string
    if LIGHT_COL in df.columns:
        # Remove empty string Light values if any
        if df[LIGHT_COL].dtype == object:
            df = df[df[LIGHT_COL].astype(str) != ""].copy()
        else:
            df = df[~df[LIGHT_COL].isna()].copy()

    # Keep needed modeling columns; if absent, we'll warn later
    for cat_col in [LIGHT_COL, FEEDING_COL, PERIOD_COL, DATE_COL, FLY_COL]:
        if cat_col in df.columns:
            # Coerce to category where appropriate
            try:
                df[cat_col] = df[cat_col].astype("category")
            except Exception:
                # If Date is datetime64, you can optionally convert to string categories
                if cat_col == DATE_COL and np.issubdtype(df[DATE_COL].dtype, np.datetime64):
                    df[DATE_COL] = df[DATE_COL].dt.strftime("%Y-%m-%d").astype("category")

    # Convert boolean columns to int to avoid issues with design matrices
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # In test mode, sample rows for speed
    if test_mode and len(df) > test_sample_size:
        df = df.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
        print(f"🧪 TEST MODE: sampled {len(df)} rows")

    print(f"Columns after load: {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")
    return df


def find_continuous_metrics(df, excluded_patterns=None, max_preview=30):
    """Heuristically find continuous metric columns (numeric, >2 unique non-NaN values)."""
    if excluded_patterns is None:
        excluded_patterns = [
            "binned_",
            "r2",
            "slope",
            "_bin_",
            "logistic_",
            "learning_",
            "interaction_rate_bin",
            "binned_auc",
            "binned_slope",
        ]
    candidates = []
    for col in df.columns:
        if col in [LIGHT_COL, FEEDING_COL, PERIOD_COL, DATE_COL, FLY_COL]:
            continue
        if any(pat in col.lower() for pat in excluded_patterns):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            non_na = df[col].dropna()
            if non_na.nunique() >= 3:
                candidates.append(col)
    print(f"Detected {len(candidates)} continuous metric candidates.")
    if candidates:
        print("Preview:", candidates[:max_preview])
    return candidates


def transform_outcome(series):
    """Heuristic transformation for right-skewed outcomes.
    Returns transformed series and a string tag."""
    s = pd.Series(series)
    tag = "none"
    if not ENABLE_TRANSFORM:
        return s, tag
    # Only transform if numeric
    if not pd.api.types.is_numeric_dtype(s):
        return s, tag
    # Safe heuristics
    try:
        skew = s.dropna().skew()
    except Exception:
        skew = 0.0
    if s.dropna().shape[0] < 20:
        return s, tag
    try:
        minval = s.min()
        try:
            minval_f: float = float(minval)
            if minval_f >= 0 and skew > 1.0:
                return np.log1p(s), "log1p"
            if minval_f > 0 and skew > 1.0:
                return np.log(s), "log"
        except Exception:
            pass
    except Exception:
        return s, tag
    return s, tag


def tidy_mixedlm_results(mdf, metric, grouping, n_obs):
    """Extract fixed-effects table as a tidy DataFrame with standard names."""
    # Statsmodels MixedLM summary table[1] is the fixed effects coef table
    tbl = mdf.summary().tables[1]
    # If tbl is a DataFrame, use to_html; if not, use as_html
    if hasattr(tbl, "to_html"):
        df_html = tbl.to_html()
    else:
        df_html = tbl.as_html()
    df = pd.read_html(StringIO(df_html), header=0, index_col=0)[0].reset_index()
    # Normalize column names
    rename_map = {
        "index": "term",
        "coef": "coef",
        "std err": "std_err",
        "z": "z",
        "P>|z|": "pval",
        "[0.025": "ci2.5",
        "0.975]": "ci97.5",
    }
    df.columns = [rename_map.get(c, c) for c in df.columns]
    df.insert(0, "metric", metric)
    df.insert(1, "grouping", grouping)
    # n_obs is a tuple (n_rows, n_cols); use n_obs[0] and broadcast as scalar
    df["n"] = n_obs[0] if isinstance(n_obs, tuple) else n_obs
    return df


def fit_mixedlm_for_metric(df, metric, fixed_effects, group_var, transform=True, max_iter=200):
    """Fit MixedLM for a single grouping factor."""
    # Keep needed columns
    cols_needed = [metric] + [fe for fe in fixed_effects if fe in df.columns]
    if group_var is not None and group_var in df.columns:
        cols_needed.append(group_var)
    # Add Date for variance component if present
    add_date_vc = False
    if DATE_COL in df.columns and DATE_COL not in cols_needed:
        cols_needed.append(DATE_COL)
        add_date_vc = True
    data = df[cols_needed].dropna().copy()

    # Enough data?
    if data.shape[0] < MIN_OBS_PER_MODEL or data[metric].dropna().nunique() < 3:
        return None, "insufficient_data", data.shape

    # Transform outcome
    y = data[metric]
    y_t, tag = transform_outcome(y) if transform else (y.copy(), "none")
    data["_y_"] = y_t

    # Build formula: C() around categorical fixed effects
    terms = []
    for fe in fixed_effects:
        if fe not in data.columns:
            continue
        if isinstance(data[fe].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(data[fe]):
            terms.append(f"C({fe})")
        else:
            terms.append(fe)
    rhs = " + ".join(terms) if terms else "1"
    formula = f"_y_ ~ {rhs}"

    # Add Date as variance component (random effect)
    vc_formula = None
    if add_date_vc:
        vc_formula = {"Date": f"0 + C({DATE_COL})"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            if group_var:
                md = smf.mixedlm(formula, data=data, groups=data[group_var], vc_formula=vc_formula)
            else:
                md = smf.mixedlm(formula, data=data, groups=np.arange(len(data)), vc_formula=vc_formula)
            mdf = md.fit(method="lbfgs", maxiter=max_iter, disp=False)
        except Exception as e:
            return None, f"fit_error: {e}", data.shape

    return mdf, "ok", data.shape


def fdr_correct_by_factor_family(results_df, alpha=0.05, method="fdr_bh"):
    """Apply FDR within factor families (e.g., all Light levels together) separately,
    per grouping (Fly/Date). Intercept not corrected."""
    out = results_df.copy()
    if out.empty:
        out["qval"] = np.nan
        out["significant_fdr"] = False
        return out

    # Identify grouping models
    groupings = out["grouping"].dropna().unique()
    out["qval"] = np.nan
    out["significant_fdr"] = False

    # Extract factor family from term, e.g., C(Light)[T.on] -> Light
    def factor_of(term):
        t = str(term)
        if t.startswith("C("):
            # Extract the factor name between 'C(' and ')'
            return t.split("C(")[1].split(")")[0]
        return "other"

    for g in groupings:
        sub_idx = out["grouping"] == g
        sub = out.loc[sub_idx].copy()
        # Exclude Intercept and failed placeholders
        sub = sub[(sub["term"] != "Intercept") & (sub["term"] != "MODEL_FAILED")].copy()
        if sub.empty:
            continue
        sub["factor"] = sub["term"].apply(factor_of)

        frames = []
        for fac, block in sub.groupby("factor"):
            block_valid = block[block["pval"].notna()].copy()
            if block_valid.empty:
                frames.append(block.assign(qval=np.nan, significant_fdr=False))
                continue
            rej, qvals, _, _ = multipletests(block_valid["pval"].values, alpha=alpha, method=method)
            block_valid.loc[:, "qval"] = qvals
            block_valid.loc[:, "significant_fdr"] = rej
            # Merge valid back into block
            merged = block.merge(block_valid[["term", "qval", "significant_fdr"]], on="term", how="left")
            frames.append(merged)
        corrected = pd.concat(frames, ignore_index=True)
        # Push corrected values back to out
        for _, row in corrected.iterrows():
            mask = (out["grouping"] == g) & (out["metric"] == row["metric"]) & (out["term"] == row["term"])
            if "qval" in corrected.columns:
                out.loc[mask, "qval"] = row.get("qval", np.nan)
            if "significant_fdr" in corrected.columns:
                out.loc[mask, "significant_fdr"] = bool(row.get("significant_fdr", False))

    return out


def make_report(corrected_df, report_path, alpha=0.05):
    """Generate a markdown report summarizing significant effects per metric."""
    lines = []
    lines.append("# Linear Mixed-Effects Results")
    lines.append("")
    lines.append(f"- Fixed effects: {', '.join(FIXED_EFFECTS)}")
    lines.append(f"- Random intercept models: groups={GROUP_PRIMARY} (Date as random effect)")
    lines.append(f"- Significance threshold: α={alpha}")
    lines.append("")

    # Summary per metric
    any_sig = False
    for metric, g in corrected_df.groupby("metric"):
        # Keep significant terms only (exclude Intercept)
        g_sig = g[(g["term"] != "Intercept") & (g["pval"].notna()) & (g["pval"] < alpha)].copy()
        if g_sig.empty:
            continue
        any_sig = True
        lines.append(f"## {metric}")
        # Sort by available columns
        sort_cols = [col for col in ["grouping", "pval", "coef"] if col in g_sig.columns]
        if sort_cols:
            g_sig = g_sig.sort_values(sort_cols, ascending=[True] * len(sort_cols))

        # Add plain-language summary sentences for each significant effect
        for _, r in g_sig.iterrows():
            # Build line robustly in case coef/ci columns are missing
            coef_str = f"coef={r['coef']:.3g} " if "coef" in r and pd.notna(r["coef"]) else ""
            ci_str = (
                f"[{r['ci2.5']:.3g}, {r['ci97.5']:.3g}], "
                if "ci2.5" in r and "ci97.5" in r and pd.notna(r["ci2.5"]) and pd.notna(r["ci97.5"])
                else ""
            )
            lines.append(f"- [{r['grouping']}] {r['term']}: {coef_str}{ci_str}p={r['pval']:.3g} (n={int(r['n'])})")

            # Plain-language summary
            term = str(r["term"])
            coef = r["coef"] if "coef" in r and pd.notna(r["coef"]) else None
            if coef is not None:
                if term.startswith("C(") and "[T." in term:
                    # e.g. C(Light)[T.on]
                    factor = term.split("C(")[1].split(")")[0]
                    level = term.split("[T.")[-1].rstrip("]")
                    direction = "higher" if coef > 0 else "lower"
                    abs_coef = abs(coef)
                    mult = np.exp(coef) if abs(coef) < 5 else None
                    if mult is not None and 0.1 < mult < 10:
                        mult_str = f"on average {mult:.2f}× the {metric}"
                    else:
                        mult_str = f"by {coef:.2g} units"
                    lines.append(
                        f"    → {factor} had a significant effect on {metric}: flies with {factor}={level} had {direction} {metric} ({mult_str}, p={r['pval']:.3g})."
                    )
                elif term.startswith("C("):
                    # e.g. C(Factor)[level] (fallback)
                    factor = term.split("C(")[1].split(")")[0]
                    direction = "higher" if coef > 0 else "lower"
                    lines.append(
                        f"    → {factor} had a significant effect on {metric}: effect size {coef:.2g} ({direction}, p={r['pval']:.3g})."
                    )
                else:
                    # Continuous or other effect
                    direction = "increase" if coef > 0 else "decrease"
                    lines.append(
                        f"    → {term} had a significant effect on {metric}: each unit increase in {term} led to a {direction} of {metric} by {abs(coef):.2g} units (p={r['pval']:.3g})."
                    )
        lines.append("")

    if not any_sig:
        lines.append("No significant fixed-effect terms detected (p < α).")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Report saved: {report_path}")


# =========================
# Main pipeline
# =========================


def run_pipeline(overwrite=True, test_mode=False):
    start = time.time()

    # Prepare output dirs
    base_out = Path(BASE_OUTPUT_DIR)
    base_out.mkdir(parents=True, exist_ok=True)
    out_dir = base_out / "lmm_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load data
    df = load_dataset(test_mode=test_mode, test_sample_size=TEST_SAMPLE_SIZE)

    # Validate required model columns
    required_cols = [LIGHT_COL, FLY_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print("Provide a fly identifier column (e.g., Fly) and ensure Light exists.")
        return

    # Ensure categorical dtypes for fixed effects
    for c in [LIGHT_COL, FEEDING_COL, PERIOD_COL, DATE_COL, FLY_COL]:
        if c in df.columns and not isinstance(df[c].dtype, pd.CategoricalDtype):
            try:
                if c == DATE_COL and np.issubdtype(df[c].dtype, np.datetime64):
                    df[c] = df[c].dt.strftime("%Y-%m-%d").astype("category")
                else:
                    df[c] = df[c].astype("category")
            except Exception:
                pass

    # Discover metrics
    all_metrics = find_continuous_metrics(df)
    if test_mode:
        all_metrics = all_metrics[:TEST_MAX_METRICS]
        print(f"🧪 TEST MODE: limiting to {len(all_metrics)} metrics")

    if not all_metrics:
        print("No continuous metrics detected. Nothing to model.")
        return

    # Build list of fixed effects present in the data
    fixed_effects = [fe for fe in FIXED_EFFECTS if fe in df.columns]
    if not fixed_effects:
        print("No fixed effects found in dataset. Add at least one of Light, FeedingState, Period.")
        return

    # Fit models per metric (Fly as group, Date as random effect)
    rows = []
    print(f"Running MixedLM for {len(all_metrics)} metrics...")
    for i, m in enumerate(all_metrics, 1):
        print(f"[{i}/{len(all_metrics)}] Fitting LMM for metric: {m}")

        mdf, status, nobs = fit_mixedlm_for_metric(
            df=df,
            metric=m,
            fixed_effects=fixed_effects,
            group_var=GROUP_PRIMARY,
            transform=ENABLE_TRANSFORM,
            max_iter=200,
        )
        if mdf is not None and status == "ok":
            fe_df = tidy_mixedlm_results(mdf, m, GROUP_PRIMARY, nobs)
            rows.append(fe_df)
        else:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "metric": m,
                            "grouping": GROUP_PRIMARY,
                            "term": "MODEL_FAILED",
                            "coef": np.nan,
                            "std_err": np.nan,
                            "z": np.nan,
                            "pval": np.nan,
                            "ci2.5": np.nan,
                            "ci97.5": np.nan,
                            "n": nobs,
                        }
                    ]
                )
            )
            if status != "insufficient_data":
                print(f"  ⚠️  {m} ({GROUP_PRIMARY}) status: {status}")

    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    raw_file = out_dir / "lmm_fixed_effects_raw.csv"
    results.to_csv(raw_file, index=False)
    print(f"💾 Saved raw fixed-effects results to: {raw_file}")

    # Generate markdown report (no FDR correction)
    report_path = out_dir / "lmm_significant_report.md"
    make_report(results, report_path, alpha=ALPHA)

    # Console summary
    sig_total = (results["pval"] < ALPHA).sum()
    print(f"✅ Done. Terms with p < {ALPHA}: {sig_total}")
    print(f"⏱️  Elapsed: {time.time() - start:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Per-metric Linear Mixed-Effects analysis (no Mann-Whitney)")
    parser.add_argument("--no-overwrite", action="store_true", help="Kept for compatibility; not used")
    parser.add_argument("--test", action="store_true", help="Run in test mode (sample rows, few metrics)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(overwrite=not args.no_overwrite, test_mode=args.test)
