#!/usr/bin/env python3
"""Jitterboxplot comparison of PCA metrics for the GtACR TH-test experiment.

Two sets of comparisons per test genotype (OGS32xMB247, OGS32xTH):
  1. vs EmptySplit control  (both BlueLight='on')
  2. vs same genotype with BlueLight='off'  (optogenetic silencing control)

Metrics come from the PCA metrics list (final_metrics_for_pca.txt).
Missing or unavailable metrics are silently skipped.

Plots are jitterboxplots with scatter points colour-coded by the test
genotype's brain region (MB247 → MB blue, TH → Neuropeptide red,
EmptySplit → Control grey).

FDR correction (Benjamini–Hochberg) is applied across all available
metrics within each comparison.

Usage:
    python plot_gtacr_th_metrics.py [--n-permutations N] [--output-dir PATH] [--test]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ballpushing_utils import dataset, read_feather
from ballpushing_utils.plotting import paired_boxplot, set_illustrator_style
from ballpushing_utils.plotting.significance import draw_significance_bar
from ballpushing_utils.stats import cohens_d, permutation_test

set_illustrator_style()

# ── Paths ──────────────────────────────────────────────────────────────────────

SUMMARY_PATH = dataset(
    "Ballpushing_Gtacr/Datasets/260520_15_summary_Gtacr_TH_Test_Data/summary/pooled_summary.feather"
)
METRICS_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "Screen_analysis"
    / "metrics_lists"
    / "final_metrics_for_pca.txt"
)
DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Ballpushing_Gtacr/Plots/GtACR_TH_test")

N_PERMUTATIONS = 10_000

# ── Genotypes ──────────────────────────────────────────────────────────────────

EMPTYSPLIT = "OGS32xEmptySplit"
TEST_GENOTYPES = ["OGS32xMB247", "OGS32xTH"]

# Scatter-dot colours per genotype, keyed to the brain-region palette.
# MB247 → Mushroom body blue; TH → Neuropeptide/dopaminergic red;
# EmptySplit → Control grey.
GENOTYPE_COLORS: dict[str, str] = {
    EMPTYSPLIT: "#7f7f7f",  # Control
    "OGS32xMB247": "#1f77b4",  # MB
    "OGS32xTH": "#d62728",  # Neuropeptide / dopaminergic
}

GENOTYPE_LABELS: dict[str, str] = {
    EMPTYSPLIT: "EmptySplit",
    "OGS32xMB247": "MB247",
    "OGS32xTH": "TH",
}

# ── Metric name remapping (PCA list uses legacy velocity names) ────────────────

METRIC_REMAP: dict[str, str] = {
    "normalized_velocity": "normalized_speed",
    "velocity_during_interactions": "speed_during_interactions",
    "velocity_trend": "speed_trend",
}
METRIC_ALT: dict[str, str] = {
    "number_of_pauses": "nb_pauses",
}

# ── Figure layout ──────────────────────────────────────────────────────────────

NCOLS = 6
PANEL_W_IN = 2.2
PANEL_H_IN = 2.8


# ── Helpers ────────────────────────────────────────────────────────────────────


def load_pca_metrics() -> list[str]:
    """Return the raw metric names from the PCA metrics list file."""
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")
    with METRICS_FILE.open() as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def resolve_metrics(columns: list[str], pca_metrics: list[str]) -> list[str]:
    """Return the available metrics after name remapping, in PCA list order."""
    col_set = set(columns)
    seen: set[str] = set()
    out: list[str] = []
    for m in pca_metrics:
        candidate = METRIC_REMAP.get(m, m)
        if candidate in col_set:
            resolved = candidate
        elif (alt := METRIC_ALT.get(m)) and alt in col_set:
            resolved = alt
        elif m in col_set:
            resolved = m
        else:
            print(f"  [skip] metric not found: {m!r}")
            continue
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def load_dataset(test_mode: bool = False) -> pd.DataFrame:
    """Load and filter the pooled summary feather."""
    print(f"Loading dataset from: {SUMMARY_PATH}")
    df = read_feather(SUMMARY_PATH)
    print(f"  Loaded: {df.shape}")

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    for col in ["insight_effect", "insight_effect_log", "exit_time", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df[df["Genotype"].isin([EMPTYSPLIT] + TEST_GENOTYPES)].copy()
    df = df[df["BlueLight"].isin(["on", "off"])].copy()
    print(f"  After genotype/BlueLight filter: {df.shape}")

    for gt in sorted(df["Genotype"].unique()):
        for bl in sorted(df["BlueLight"].unique()):
            n = len(df[(df["Genotype"] == gt) & (df["BlueLight"] == bl)])
            if n:
                print(f"    {gt} / BlueLight={bl}: {n} flies")

    if test_mode:
        df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
        print(f"  TEST MODE: sampled {len(df)} rows")

    return df


def run_comparison(
    df: pd.DataFrame,
    metrics: list[str],
    control_mask: pd.Series,
    test_mask: pd.Series,
    control_label: str,
    test_label: str,
    n_permutations: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation tests for all metrics in one comparison; FDR-corrected.

    Returns a DataFrame with one row per metric:
    metric, n_control, n_test, median_control, median_test,
    observed_diff, cohens_d, p_raw, p_fdr.
    """
    rng = np.random.RandomState(seed)

    rows = []
    for metric in metrics:
        ctrl = df.loc[control_mask, metric].dropna().to_numpy()
        test = df.loc[test_mask, metric].dropna().to_numpy()
        if len(ctrl) < 2 or len(test) < 2:
            continue
        result = permutation_test(
            ctrl,
            test,
            statistic="median",
            n_permutations=n_permutations,
            rng=rng,
        )
        d = cohens_d(ctrl, test)
        rows.append(
            {
                "metric": metric,
                "control": control_label,
                "test": test_label,
                "n_control": len(ctrl),
                "n_test": len(test),
                "median_control": float(np.median(ctrl)),
                "median_test": float(np.median(test)),
                "observed_diff": result.observed_diff,
                "cohens_d": d,
                "p_raw": result.p_value,
            }
        )

    if not rows:
        return pd.DataFrame()

    stats = pd.DataFrame(rows)
    _, p_fdr, _, _ = multipletests(stats["p_raw"].values, method="fdr_bh")
    stats["p_fdr"] = p_fdr
    return stats


def plot_metrics_figure(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    control_mask: pd.Series,
    test_mask: pd.Series,
    control_color: str,
    test_color: str,
    control_label: str,
    test_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Multi-panel jitterboxplot figure for all metrics in *stats*."""
    metrics = stats["metric"].tolist()
    n = len(metrics)
    if n == 0:
        print(f"  [skip] no metrics for: {title}")
        return

    ncols = min(NCOLS, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(PANEL_W_IN * ncols, PANEL_H_IN * nrows)
    )
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, row in enumerate(stats.itertuples()):
        metric = row.metric
        ax = axes_flat[idx]

        ctrl = df.loc[control_mask, metric].dropna().to_numpy()
        test = df.loc[test_mask, metric].dropna().to_numpy()

        paired_boxplot(ax, [ctrl, test], colors=[control_color, test_color])
        draw_significance_bar(ax, 0, 1, p_value=row.p_fdr)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"{control_label}\n(n={row.n_control})",
                f"{test_label}\n(n={row.n_test})",
            ],
            fontsize=5,
        )
        ax.set_title(metric, fontsize=5, pad=2)
        ax.tick_params(axis="both", labelsize=5, length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.grid(False)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=7, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics list and data
    pca_metrics = load_pca_metrics()
    df = load_dataset(test_mode=args.test)
    metrics = resolve_metrics(list(df.columns), pca_metrics)
    print(f"\nAvailable PCA metrics ({len(metrics)}): {metrics}")

    n_perm = args.n_permutations
    all_stats: list[pd.DataFrame] = []

    # ── Comparisons 1: each test genotype vs EmptySplit (BlueLight on) ──────────
    print("\n" + "=" * 60)
    print("COMPARISON 1: test genotype vs EmptySplit  (BlueLight='on')")
    print("=" * 60)

    emptysplit_mask = (df["Genotype"] == EMPTYSPLIT) & (df["BlueLight"] == "on")

    for genotype in TEST_GENOTYPES:
        test_mask = (df["Genotype"] == genotype) & (df["BlueLight"] == "on")
        ctrl_label = GENOTYPE_LABELS[EMPTYSPLIT]
        test_label = GENOTYPE_LABELS[genotype]
        title = f"{test_label} vs {ctrl_label}  (BlueLight on)"
        print(f"\n  {title}")

        stats = run_comparison(
            df,
            metrics,
            control_mask=emptysplit_mask,
            test_mask=test_mask,
            control_label=ctrl_label,
            test_label=test_label,
            n_permutations=n_perm,
        )
        if stats.empty:
            print("  [warn] no stats computed – skipping")
            continue

        slug = genotype.replace("OGS32x", "").lower()
        out_pdf = output_dir / f"metrics_{slug}_vs_emptysplit_blulight_on.pdf"
        plot_metrics_figure(
            df,
            stats,
            control_mask=emptysplit_mask,
            test_mask=test_mask,
            control_color=GENOTYPE_COLORS[EMPTYSPLIT],
            test_color=GENOTYPE_COLORS[genotype],
            control_label=ctrl_label,
            test_label=test_label,
            title=title,
            output_path=out_pdf,
        )
        stats["comparison_type"] = "vs_emptysplit"
        all_stats.append(stats)

    # ── Comparisons 2: each test genotype BlueLight on vs off ───────────────────
    print("\n" + "=" * 60)
    print("COMPARISON 2: test genotype  BlueLight='on' vs BlueLight='off'")
    print("=" * 60)

    for genotype in TEST_GENOTYPES:
        off_mask = (df["Genotype"] == genotype) & (df["BlueLight"] == "off")
        on_mask = (df["Genotype"] == genotype) & (df["BlueLight"] == "on")
        ctrl_label = f"{GENOTYPE_LABELS[genotype]} off"
        test_label = f"{GENOTYPE_LABELS[genotype]} on"
        title = f"{GENOTYPE_LABELS[genotype]}  BlueLight on vs off"
        print(f"\n  {title}")

        stats = run_comparison(
            df,
            metrics,
            control_mask=off_mask,
            test_mask=on_mask,
            control_label=ctrl_label,
            test_label=test_label,
            n_permutations=n_perm,
        )
        if stats.empty:
            print("  [warn] no stats computed – skipping")
            continue

        slug = genotype.replace("OGS32x", "").lower()
        out_pdf = output_dir / f"metrics_{slug}_bluelight_on_vs_off.pdf"
        plot_metrics_figure(
            df,
            stats,
            control_mask=off_mask,
            test_mask=on_mask,
            control_color="#aaaaaa",  # same genotype off → lighter grey
            test_color=GENOTYPE_COLORS[genotype],
            control_label=ctrl_label,
            test_label=test_label,
            title=title,
            output_path=out_pdf,
        )
        stats["comparison_type"] = "vs_bluelight_off"
        all_stats.append(stats)

    # ── Save combined stats CSV ──────────────────────────────────────────────────
    if all_stats:
        stats_csv = output_dir / "gtacr_th_metrics_statistics.csv"
        pd.concat(all_stats, ignore_index=True).to_csv(
            stats_csv, index=False, float_format="%.6f"
        )
        print(f"\nStatistics saved to: {stats_csv}")

    print(f"\nDone. Outputs in: {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GtACR TH-test: jitterboxplot comparison of PCA metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_gtacr_th_metrics.py
  python plot_gtacr_th_metrics.py --n-permutations 1000 --test
  python plot_gtacr_th_metrics.py --output-dir /tmp/gtacr_test
        """,
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=f"Number of permutations for statistical testing (default: {N_PERMUTATIONS})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: subsample data to 200 rows for quick verification",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
