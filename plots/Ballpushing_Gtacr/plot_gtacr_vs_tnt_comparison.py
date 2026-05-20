#!/usr/bin/env python3
"""Cross-experiment jitterboxplot comparison: GtACR (light-on) vs TNT screen.

Compares:
  - OGS32xMB247 (BlueLight=on)       vs  50742 (MB247-GAL4)         (TNT screen)
  - OGS32xTH    (BlueLight=on)       vs  Ple-Gal4.F a.k.a TH-Gal4  (TNT screen)

Goal: verify that GtACR and TNT silencing of the same neurons yield similar
behavioural phenotypes (ideally no significant difference across metrics).

Metrics come from final_metrics_for_pca.txt with velocity→speed name remapping.
nb_freeze and number_of_pauses are skipped (absent from the GtACR dataset).

FDR correction (Benjamini–Hochberg) is applied per comparison.
Output: one PDF+PNG per genotype pair in DEFAULT_OUTPUT_DIR.

Usage:
    python plot_gtacr_vs_tnt_comparison.py [--n-permutations N] [--output-dir PATH] [--test]
"""

from __future__ import annotations

import argparse
import importlib.util
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

GTACR_PATH = dataset(
    "Ballpushing_Gtacr/Datasets/260520_15_summary_Gtacr_TH_Test_Data/summary/pooled_summary.feather"
)
TNT_PATH = dataset(
    "Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
)
METRICS_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "Screen_analysis"
    / "metrics_lists"
    / "final_metrics_for_pca.txt"
)
DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/Ballpushing_Gtacr/Plots/GtACR_vs_TNT")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "Plotting" / "Config.py"
_spec = importlib.util.spec_from_file_location("Config", _CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not locate Config module at {_CONFIG_PATH}")
Config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Config)

N_PERMUTATIONS = 10_000

# ── Genotype mappings ──────────────────────────────────────────────────────────

# Each entry defines a cross-experiment pair to compare.
# GtACR genotype (BlueLight=on) vs TNT nickname.
GENOTYPE_PAIRS = [
    {
        "gtacr_nickname": "OGS32xMB247",
        "tnt_nickname": "50742 (MB247-GAL4)",
        "display": "MB247",
        "gtacr_color": "#1f77b4",  # MB blue
        "tnt_color": "#aec7e8",  # light MB blue
    },
    {
        "gtacr_nickname": "OGS32xTH",
        "tnt_nickname": "Ple-Gal4.F a.k.a TH-Gal4",
        "display": "TH",
        "gtacr_color": "#d62728",  # Neuropeptide red
        "tnt_color": "#f4a582",  # light salmon
    },
]

# ── Metric remapping (PCA list uses legacy velocity names) ─────────────────────

METRIC_REMAP: dict[str, str] = {
    "normalized_velocity": "normalized_speed",
    "velocity_during_interactions": "speed_during_interactions",
    "velocity_trend": "speed_trend",
    "number_of_pauses": "nb_pauses",  # TNT uses number_of_pauses, GtACR uses nb_pauses
}
METRIC_ALT: dict[str, str] = {}  # all aliases covered by METRIC_REMAP

# ── Figure layout ──────────────────────────────────────────────────────────────

NCOLS = 6
PANEL_W_IN = 2.2
PANEL_H_IN = 2.8


# ── Data loading ───────────────────────────────────────────────────────────────


def load_pca_metrics() -> list[str]:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")
    with METRICS_FILE.open() as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def resolve_metrics_for_columns(
    columns: list[str], pca_metrics: list[str]
) -> list[str]:
    """Return metrics available in *columns* after name remapping, in PCA list order."""
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
            continue
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def load_gtacr(test_mode: bool = False) -> pd.DataFrame:
    print(f"Loading GtACR dataset from: {GTACR_PATH}")
    df = read_feather(GTACR_PATH)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    gtacr_nicks = [p["gtacr_nickname"] for p in GENOTYPE_PAIRS]
    df = df[(df["Genotype"].isin(gtacr_nicks)) & (df["BlueLight"] == "on")].copy()
    print(f"  GtACR rows (BlueLight=on): {len(df)}")
    if test_mode:
        df = df.sample(n=min(len(df), 50), random_state=42).reset_index(drop=True)
    return df


def load_tnt(test_mode: bool = False) -> pd.DataFrame:
    print(f"Loading TNT dataset from: {TNT_PATH}")
    df = read_feather(TNT_PATH)
    df = Config.cleanup_data(df)
    # Normalise pause-count column name: TNT calls it number_of_pauses, GtACR uses nb_pauses
    if "number_of_pauses" in df.columns and "nb_pauses" not in df.columns:
        df = df.rename(columns={"number_of_pauses": "nb_pauses"})
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    tnt_nicks = [p["tnt_nickname"] for p in GENOTYPE_PAIRS]
    df = df[df["Nickname"].isin(tnt_nicks)].copy()
    print(f"  TNT rows (selected genotypes): {len(df)}")
    if test_mode:
        df = df.sample(n=min(len(df), 50), random_state=42).reset_index(drop=True)
    return df


# ── Analysis ───────────────────────────────────────────────────────────────────


def run_comparison(
    gtacr_vals: np.ndarray,
    tnt_vals: np.ndarray,
    metric: str,
    rng: np.random.RandomState,
    n_permutations: int,
) -> dict:
    result = permutation_test(
        gtacr_vals,
        tnt_vals,
        statistic="median",
        n_permutations=n_permutations,
        rng=rng,
    )
    d = cohens_d(gtacr_vals, tnt_vals)
    return {
        "metric": metric,
        "n_gtacr": len(gtacr_vals),
        "n_tnt": len(tnt_vals),
        "median_gtacr": float(np.median(gtacr_vals)),
        "median_tnt": float(np.median(tnt_vals)),
        "cohens_d": d,
        "p_raw": result.p_value,
    }


def analyse_pair(
    pair: dict,
    gtacr_df: pd.DataFrame,
    tnt_df: pd.DataFrame,
    common_metrics: list[str],
    n_permutations: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    gtacr_group = gtacr_df[gtacr_df["Genotype"] == pair["gtacr_nickname"]]
    tnt_group = tnt_df[tnt_df["Nickname"] == pair["tnt_nickname"]]
    print(f"  {pair['display']}: GtACR n={len(gtacr_group)}, TNT n={len(tnt_group)}")

    rows = []
    for metric in common_metrics:
        g = (
            gtacr_group[metric].dropna().to_numpy()
            if metric in gtacr_group.columns
            else np.array([])
        )
        t = (
            tnt_group[metric].dropna().to_numpy()
            if metric in tnt_group.columns
            else np.array([])
        )
        if len(g) < 2 or len(t) < 2:
            continue
        rows.append(run_comparison(g, t, metric, rng, n_permutations))

    if not rows:
        return pd.DataFrame()

    stats = pd.DataFrame(rows)
    _, p_fdr, _, _ = multipletests(stats["p_raw"].values, method="fdr_bh")
    stats["p_fdr"] = p_fdr
    return stats


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_comparison(
    pair: dict,
    stats: pd.DataFrame,
    gtacr_df: pd.DataFrame,
    tnt_df: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = stats["metric"].tolist()
    n = len(metrics)
    if n == 0:
        print(f"  [skip] no common metrics for {pair['display']}")
        return

    ncols = min(NCOLS, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(PANEL_W_IN * ncols, PANEL_H_IN * nrows)
    )
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    gtacr_group = gtacr_df[gtacr_df["Genotype"] == pair["gtacr_nickname"]]
    tnt_group = tnt_df[tnt_df["Nickname"] == pair["tnt_nickname"]]

    for idx, row in enumerate(stats.itertuples()):
        metric = row.metric
        ax = axes_flat[idx]

        g = (
            gtacr_group[metric].dropna().to_numpy()
            if metric in gtacr_group.columns
            else np.array([])
        )
        t = (
            tnt_group[metric].dropna().to_numpy()
            if metric in tnt_group.columns
            else np.array([])
        )

        paired_boxplot(ax, [g, t], colors=[pair["gtacr_color"], pair["tnt_color"]])
        draw_significance_bar(ax, 0, 1, p_value=row.p_fdr)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"GtACR\n(n={row.n_gtacr})", f"TNT\n(n={row.n_tnt})"],
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

    fig.suptitle(
        f"{pair['display']}:  GtACR (OGS32, light on)  vs  TNT screen",
        fontsize=7,
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test", action="store_true", help="Subsample data for quick testing"
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_metrics = load_pca_metrics()

    gtacr_df = load_gtacr(test_mode=args.test)
    tnt_df = load_tnt(test_mode=args.test)

    all_stats: list[pd.DataFrame] = []

    for pair in GENOTYPE_PAIRS:
        print(f"\n{'='*60}")
        print(f"Comparing  {pair['display']}:  GtACR (on)  vs  TNT")
        print(f"{'='*60}")

        # Metrics available in both datasets for this specific pair
        gtacr_group = gtacr_df[gtacr_df["Genotype"] == pair["gtacr_nickname"]]
        tnt_group = tnt_df[tnt_df["Nickname"] == pair["tnt_nickname"]]

        if gtacr_group.empty or tnt_group.empty:
            print(f"  [skip] missing data for {pair['display']}")
            continue

        # Resolve metrics available in both groups
        gtacr_metrics = resolve_metrics_for_columns(
            gtacr_group.columns.tolist(), pca_metrics
        )
        tnt_metrics = resolve_metrics_for_columns(
            tnt_group.columns.tolist(), pca_metrics
        )
        common_metrics = [m for m in gtacr_metrics if m in set(tnt_metrics)]
        # Identify genuinely skipped metrics (not resolvable in at least one dataset)
        covered = set(gtacr_metrics) & set(tnt_metrics)
        for m in pca_metrics:
            resolved = METRIC_REMAP.get(m, m)
            if resolved not in covered:
                print(f"  [skip] {m!r} (resolved: {resolved!r})")
        print(f"  Common metrics: {len(common_metrics)}")

        stats = analyse_pair(
            pair,
            gtacr_df,
            tnt_df,
            common_metrics,
            n_permutations=args.n_permutations,
        )
        if stats.empty:
            continue

        stats["genotype"] = pair["display"]
        all_stats.append(stats)

        out_stem = f"gtacr_vs_tnt_{pair['display'].lower()}"
        plot_comparison(pair, stats, gtacr_df, tnt_df, output_dir / f"{out_stem}.pdf")

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        stats_path = output_dir / "gtacr_vs_tnt_statistics.csv"
        combined.to_csv(stats_path, index=False)
        print(f"\nStatistics saved to: {stats_path}")

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
