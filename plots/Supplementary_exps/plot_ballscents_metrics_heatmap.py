#!/usr/bin/env python3
"""
Generate metrics heatmaps for BallScents experiments.

Styling and behavior mirror `plot_gtacr_or67d_metrics_heatmap.py` but without
ATR filtering (single condition). Shows Cohen's d per ball type vs control
(`Scented` by default), uses PCA canonical metric ordering if available,
clips colorbar to CLIP_EFFECTS and annotates sample sizes under genotype labels.
"""
from pathlib import Path
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

# add src to path so we can reuse PCA helpers if present
sys.path.append(str(Path(__file__).parent.parent))

try:
    from PCA.plot_detailed_metric_statistics import METRIC_DISPLAY_NAMES as PCA_METRIC_DISPLAY_NAMES
    from PCA.plot_detailed_metric_statistics import METRICS_PATH as PCA_METRICS_PATH
except Exception:
    PCA_METRIC_DISPLAY_NAMES = {}
    PCA_METRICS_PATH = None

# Color clipping to match PCA
CLIP_EFFECTS = 1.5


def get_display_name(metric_name):
    return PCA_METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


def normalize_ball_scent_labels(df, group_col="BallScent"):
    """Normalize/alias BallScent values to canonical factorial labels.

    Uses substring and fuzzy matching to map existing values to one of:
    Ctrl, CtrlScent, Washed, Scented, New, NewScent
    """
    if group_col not in df.columns:
        return df

    design_keys = ["Ctrl", "CtrlScent", "Washed", "Scented", "New", "NewScent"]
    available = pd.Series(df[group_col].dropna().unique()).astype(str).tolist()
    from difflib import get_close_matches

    mapping = {}
    for val in available:
        if val in design_keys:
            mapping[val] = val
            continue
        vs = str(val).strip().lower()
        found = None
        # exact or substring
        for k in design_keys:
            ks = k.lower()
            if vs == ks or ks in vs or vs in ks:
                found = k
                break
        # fuzzy
        if not found:
            candidates = get_close_matches(vs, [k.lower() for k in design_keys], n=1, cutoff=0.6)
            if candidates:
                for k in design_keys:
                    if k.lower() == candidates[0]:
                        found = k
                        break

        mapping[val] = found if found is not None else val

    # apply mapping and report
    # For any entries that were not mapped, try simple substring heuristics
    for val in list(mapping.keys()):
        if mapping[val] == val:
            vs = str(val).strip().lower()
            if "new" in vs and "scent" in vs:
                mapping[val] = "NewScent"
            elif "new" in vs:
                mapping[val] = "New"
            elif "wash" in vs or "washed" in vs:
                # If it also mentions scent, prefer Scented
                if "scent" in vs:
                    mapping[val] = "Scented"
                else:
                    mapping[val] = "Washed"
            elif "ctrl" in vs and "scent" in vs:
                mapping[val] = "CtrlScent"
            elif "scent" in vs:
                # Prefer mapping to Scented/NewScent when 'scent' appears alone
                if "new" in vs:
                    mapping[val] = "NewScent"
                elif "wash" in vs or "washed" in vs or vs == "scented":
                    mapping[val] = "Scented"
                else:
                    # If it mentions ctrl explicitly, map to CtrlScent, otherwise assume Scented
                    mapping[val] = "CtrlScent" if "ctrl" in vs else "Scented"
            elif "pre" in vs or "exposed" in vs:
                mapping[val] = "CtrlScent"
            else:
                mapping[val] = val

    remapped = {k: v for k, v in mapping.items() if k != v}
    print(f"Normalizing {group_col} labels (total variants: {len(mapping)}):")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")

    df = df.copy()
    df[group_col] = df[group_col].map(lambda x: mapping.get(str(x), x))
    # Preserve canonical mapping in a separate column for downstream matching
    canonical_col = f"{group_col}_canonical"
    df[canonical_col] = df[group_col]

    # Map canonical keys to descriptive factorial labels for plotting
    display_map = {
        "Ctrl": "Ctrl",
        "CtrlScent": "Pre-exposed",
        "Washed": "Washed",
        "Scented": "Washed + Pre-exposed",
        "New": "New",
        "NewScent": "New + Pre-exposed",
    }

    df[group_col] = df[group_col].map(lambda x: display_map.get(x, x))

    return df


def load_canonical_metric_order():
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
                        return lines
        except Exception:
            continue
    return []


def calculate_cohens_d(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


def build_significance_matrix(stats_df, control_label, raw_data, group_col="BallScent"):
    print("   Calculating Cohen's d matrix for BallScents...")

    # Filter to only canonical PCA metrics first
    canonical = load_canonical_metric_order()
    if canonical:
        all_metrics_in_stats = stats_df["Metric"].unique()
        metrics_before = len(all_metrics_in_stats)
        # Keep only metrics that are in the canonical list
        metrics_to_keep = [m for m in all_metrics_in_stats if m in canonical]
        stats_df = stats_df[stats_df["Metric"].isin(metrics_to_keep)].copy()
        print(f"   Filtered to PCA canonical metrics: {metrics_before} → {len(metrics_to_keep)} metrics")
        if len(metrics_to_keep) < metrics_before:
            excluded = set(all_metrics_in_stats) - set(metrics_to_keep)
            print(f"   Excluded {len(excluded)} non-canonical metrics: {sorted(excluded)}")
    else:
        print("   Warning: No canonical metric list found; using all metrics from stats")

    # Determine which column in stats_df refers to the tested group
    test_col = "Test" if "Test" in stats_df.columns else ("BallScent" if "BallScent" in stats_df.columns else None)
    if test_col is None:
        raise KeyError("Could not find a column indicating test group in stats_df (expected 'Test' or 'BallScent')")

    # Use canonical column if present for matching against stats names
    canonical_col = f"{group_col}_canonical"
    if canonical_col in raw_data.columns:
        available_groups = sorted(raw_data[canonical_col].dropna().unique())
        use_canonical = True
    else:
        available_groups = sorted(raw_data[group_col].dropna().unique())
        use_canonical = False

    print(f"   Available groups in data (for matching): {available_groups}")

    # If the requested control is missing from the raw data, allow the special
    # stats-only control 'Ctrl' to be used: don't raise an error. In this case
    # Cohen's d values will be taken from the stats file (if present) rather
    # than computed from raw control rows.
    ctrl_missing = False
    if control_label not in available_groups:
        if str(control_label).strip() == "Ctrl":
            ctrl_missing = True
            print(
                "   Info: requested control 'Ctrl' not present in raw data; "
                "using stats-only control handling (stars from stats_df; effect sizes from stats_df if available)."
            )
        else:
            # Otherwise try to map requested control label to an available group using substring/fuzzy matching
            cl = str(control_label).strip().lower()
            # substring candidates
            substr_cands = [g for g in available_groups if cl in str(g).strip().lower() or str(g).strip().lower() in cl]
            from difflib import get_close_matches

            fuzzy_cands = get_close_matches(cl, [str(g).strip().lower() for g in available_groups], n=1, cutoff=0.6)

            if substr_cands:
                mapped_control = substr_cands[0]
                print(f"  Info: remapping requested control '{control_label}' -> '{mapped_control}' (substring match)")
                control_label = mapped_control
            elif fuzzy_cands:
                # find original group matching the fuzzy candidate
                cand = fuzzy_cands[0]
                for g in available_groups:
                    if str(g).strip().lower() == cand:
                        mapped_control = g
                        break
                print(f"  Info: remapping requested control '{control_label}' -> '{mapped_control}' (fuzzy match)")
                control_label = mapped_control
            else:
                raise ValueError(
                    f"Control label '{control_label}' not found in data. "
                    f"Available groups: {available_groups}. "
                    f"Use --control to specify the correct control group."
                )

    # Map stats_df test names to raw data group names to handle naming mismatches
    stats_names = list(stats_df[test_col].unique())
    stats_to_raw = {}
    from difflib import get_close_matches

    def map_name_to_raw(name):
        if name in available_groups:
            return name
        name_s = str(name).strip().lower()
        # exact substring match
        for g in available_groups:
            if g is None:
                continue
            if name_s == str(g).strip().lower():
                return g
            if str(g).strip().lower() in name_s or name_s in str(g).strip().lower():
                return g
        # fuzzy match
        candidates = get_close_matches(name_s, [str(g).strip().lower() for g in available_groups], n=1, cutoff=0.6)
        if candidates:
            for g in available_groups:
                if str(g).strip().lower() == candidates[0]:
                    return g
        return None

    reverse_map = {}
    unmapped = []
    for s in stats_names:
        mapped = map_name_to_raw(s)
        if mapped:
            stats_to_raw[s] = mapped
            reverse_map.setdefault(mapped, []).append(s)
        else:
            unmapped.append(s)

    if unmapped:
        print(f"   Warning: could not map these stats test names to raw data groups (they will be skipped): {unmapped}")

    # Build genotypes list = mapped raw groups excluding the control
    # As a recovery step, include any raw groups that appear as substrings in stats names
    added = []
    for g in available_groups:
        if g == control_label:
            continue
        if g in set(stats_to_raw.values()):
            continue
        g_s = str(g).strip().lower()
        for s in stats_names:
            if g_s in str(s).strip().lower() or str(s).strip().lower() in g_s:
                # map this stats name to the raw group
                stats_to_raw[s] = g
                reverse_map.setdefault(g, []).append(s)
                added.append(g)
                break
    if added:
        print(f"   Info: added raw groups matched by substring to stats names: {sorted(set(added))}")

    # When control is stats-only (ctrl_missing=True) the raw data lacks the
    # control group; still build genotypes from mapped stats names but do not
    # require a raw control row to compute significance stars.
    if ctrl_missing:
        genotypes = sorted([g for g in sorted(set(stats_to_raw.values())) if g is not None])
    else:
        genotypes = sorted([g for g in sorted(set(stats_to_raw.values())) if g != control_label])
    print(f"   Control: {control_label}, Test groups used for Cohen's d (canonical): {genotypes}")
    metrics = sorted(stats_df["Metric"].unique())

    # Build DataFrame indexed by display labels (if available) but matched by canonical keys
    row_index = []
    for g in genotypes:
        if use_canonical:
            disp_vals = raw_data.loc[raw_data[canonical_col] == g, group_col].dropna().unique()
            display_label = disp_vals[0] if len(disp_vals) > 0 else g
        else:
            display_label = g
        row_index.append(display_label)

    matrix = pd.DataFrame(0.0, index=row_index, columns=metrics)

    for genotype, disp_label in zip(genotypes, matrix.index):
        for metric in metrics:
            if not ctrl_missing:
                if use_canonical:
                    test_vals = raw_data[raw_data[canonical_col] == genotype][metric]
                    ctrl_vals = raw_data[raw_data[canonical_col] == control_label][metric]
                else:
                    test_vals = raw_data[raw_data[group_col] == genotype][metric]
                    ctrl_vals = raw_data[raw_data[group_col] == control_label][metric]
                if len(test_vals.dropna()) > 1 and len(ctrl_vals.dropna()) > 1:
                    matrix.loc[disp_label, metric] = calculate_cohens_d(test_vals, ctrl_vals)
                else:
                    matrix.loc[disp_label, metric] = 0.0
            else:
                # Stats-only control path: try to extract an effect-size from stats_df
                # for the test vs 'Ctrl' comparison. Prefer 'effect_size' column,
                # fall back to (test_median - control_median) if available.
                stats_names_for_geno = reverse_map.get(genotype, [])
                val = 0.0
                found = False
                for sname in stats_names_for_geno:
                    if "Control" in stats_df.columns:
                        mask = (
                            (stats_df[test_col] == sname)
                            & (stats_df["Metric"] == metric)
                            & (stats_df["Control"] == control_label)
                        )
                    else:
                        mask = (stats_df[test_col] == sname) & (stats_df["Metric"] == metric)
                    if mask.any():
                        row = stats_df.loc[mask].iloc[0]
                        if "effect_size" in stats_df.columns:
                            val = row.get("effect_size", 0.0)
                        elif "test_median" in stats_df.columns and "control_median" in stats_df.columns:
                            val = float(row.get("test_median", 0.0)) - float(row.get("control_median", 0.0))
                        else:
                            val = 0.0
                        found = True
                        break
                matrix.loc[disp_label, metric] = float(val) if found else 0.0

    # Apply canonical ordering if available
    canonical = load_canonical_metric_order()
    if canonical:
        ordered = [m for m in canonical if m in matrix.columns]
        remaining = [m for m in matrix.columns if m not in ordered]
        if ordered:
            matrix = matrix.loc[:, ordered + remaining]
            print(f"   Applied canonical metric order: {len(ordered)} metrics reordered")

    # Return matrix and reverse mapping from raw group -> list of stats_df names
    # If we used canonical keys, convert reverse_map to use display labels as keys
    if use_canonical:
        reverse_map_display = {}
        for canon_key, names in reverse_map.items():
            disp_vals = raw_data.loc[raw_data[canonical_col] == canon_key, group_col].dropna().unique()
            display_label = disp_vals[0] if len(disp_vals) > 0 else canon_key
            reverse_map_display[display_label] = names
        return matrix, reverse_map_display

    return matrix, reverse_map


def plot_two_way_dendrogram(
    matrix,
    reverse_map,
    stats_df,
    output_dir,
    control_label,
    raw_data,
    group_col="BallScent",
    nickname_to_region=None,
    color_dict_regions=None,
):
    if matrix.empty:
        print("Empty matrix; skipping")
        return

    print("Creating two-way dendrogram heatmap (BallScents)...")
    # Determine stats DataFrame column that identifies the tested group
    test_col = "Test" if "Test" in stats_df.columns else ("BallScent" if "BallScent" in stats_df.columns else None)
    if test_col is None:
        raise KeyError("Could not find a column indicating test group in stats_df (expected 'Test' or 'BallScent')")
    # rows clustering
    if matrix.shape[0] > 2:
        row_linkage = linkage(pdist(matrix.values, metric="euclidean"), method="ward")
        row_order = dendrogram(row_linkage, no_plot=True)["leaves"]
    else:
        row_order = list(range(matrix.shape[0]))

    # columns clustering or canonical
    canonical = load_canonical_metric_order()
    if canonical:
        col_order = [matrix.columns.get_loc(m) for m in canonical if m in matrix.columns]
        col_linkage = None
    elif matrix.shape[1] > 1:
        corr = matrix.corr().fillna(0.0)
        dist = 1 - np.abs(corr.values)
        np.fill_diagonal(dist, 0.0)
        n = dist.shape[0]
        col_linkage = linkage(dist[np.triu_indices(n, k=1)], method="average")
        col_order = dendrogram(col_linkage, no_plot=True)["leaves"]
    else:
        col_order = [0]
        col_linkage = None

    matrix_ordered = matrix.iloc[row_order, col_order]

    # plot (simplified but matching styles)
    n_metrics = matrix_ordered.shape[1]
    n_rows = matrix_ordered.shape[0]
    fig_w = max(12, n_metrics * 0.6)
    fig_h = max(6, n_rows * 0.8)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = plt.GridSpec(3, 3, width_ratios=[1.5, 8, 0.4], height_ratios=[1.2, 0.6, 5], hspace=0.08, wspace=0.03)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_labels = fig.add_subplot(gs[1, 1])
    ax_ylabels = fig.add_subplot(gs[2, 0])
    ax_hm = fig.add_subplot(gs[2, 1])
    ax_cbar = fig.add_subplot(gs[2, 2])

    # heatmap
    max_abs = max(abs(matrix_ordered.values.min()), abs(matrix_ordered.values.max()))
    if CLIP_EFFECTS is not None:
        vmin, vmax = -CLIP_EFFECTS, CLIP_EFFECTS
        clipped = max_abs > CLIP_EFFECTS
    else:
        vmin, vmax = -max_abs, max_abs
        clipped = False

    im = ax_hm.imshow(
        matrix_ordered.values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        extent=[-0.5, n_metrics - 0.5, n_rows - 0.5, -0.5],
    )
    # grid
    for i in range(n_rows + 1):
        ax_hm.axhline(i - 0.5, color="gray", linewidth=0.5)
    for j in range(n_metrics + 1):
        ax_hm.axvline(j - 0.5, color="gray", linewidth=0.5)

    # stars
    for i in range(n_rows):
        for j in range(n_metrics):
            geno = matrix_ordered.index[i]
            metric = matrix_ordered.columns[j]
            stats_names_for_geno = reverse_map.get(geno, [])
            if not stats_names_for_geno:
                continue
            mask = (stats_df[test_col].isin(stats_names_for_geno)) & (stats_df["Metric"] == metric)
            if mask.any():
                p = stats_df.loc[mask, "pval_fdr"].values[0]
                stars = "" if p >= 0.05 else ("***" if p < 0.001 else "**" if p < 0.01 else "*")
                if stars:
                    bg = matrix_ordered.values[i, j]
                    color = "white" if abs(bg) >= 0.5 else "black"
                    ax_hm.text(
                        j + 0.5, i + 0.5, stars, ha="center", va="center", color=color, fontsize=10, fontweight="bold"
                    )

    ax_hm.set_xticks([])
    ax_hm.set_yticks([])

    ax_top.axis("off")
    ax_labels.axis("off")

    # metric labels
    ax_labels.set_xlim(ax_hm.get_xlim())
    ax_labels.set_ylim(0, 1)
    x_pos = np.arange(n_metrics)
    for xp, lab in zip(x_pos, [get_display_name(m) for m in matrix_ordered.columns]):
        ax_labels.text(xp, 0.2, lab, ha="right", va="top", fontsize=7, rotation=45)

    # y labels with sample sizes
    ax_ylabels.axis("off")
    ax_ylabels.set_ylim(ax_hm.get_ylim())
    y_pos = np.arange(n_rows)
    for yp, geno in zip(y_pos, matrix_ordered.index):
        n = int(raw_data[raw_data[group_col] == geno].shape[0]) if raw_data is not None else 0
        label = f"{geno}\n(n = {n})"
        color = (
            color_dict_regions.get(nickname_to_region.get(geno, "Unknown"), "black")
            if (nickname_to_region and color_dict_regions)
            else "black"
        )
        ax_ylabels.text(0.95, yp, label, ha="right", va="center", fontsize=10, color=color)

    # colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")), cax=ax_cbar)
    cbar.set_label("Cohen's d\n(Blue: Lower, Red: Higher)")
    ticks = np.linspace(vmin, vmax, 5)
    labels = [f"{t:.2f}" for t in ticks]
    if clipped:
        labels[0] = f"< {vmin:.2f}"
        labels[-1] = f"> {vmax:.2f}"
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    fig.suptitle(
        f"BallScents - Cohen's d Effect Sizes (vs {control_label})\n({len(matrix_ordered)} comparisons × {len(matrix_ordered.columns)} metrics)"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "ballscents_metrics_dendrogram.png"
    pdf = output_dir / "ballscents_metrics_dendrogram.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png}, {pdf}")


def plot_simple_heatmap(
    matrix,
    reverse_map,
    stats_df,
    output_dir,
    control_label,
    raw_data,
    group_col="BallScent",
    nickname_to_region=None,
    color_dict_regions=None,
):
    if matrix.empty:
        return
    print("Creating simple heatmap (BallScents)...")
    # Determine stats DataFrame column that identifies the tested group
    test_col = "Test" if "Test" in stats_df.columns else ("BallScent" if "BallScent" in stats_df.columns else None)
    if test_col is None:
        raise KeyError("Could not find a column indicating test group in stats_df (expected 'Test' or 'BallScent')")
    # sort rows by name
    row_order = sorted(matrix.index)
    # columns canonical/clustering
    canonical = load_canonical_metric_order()
    if canonical and any(m in matrix.columns for m in canonical):
        col_order = [m for m in canonical if m in matrix.columns] + [m for m in matrix.columns if m not in canonical]
    elif matrix.shape[1] > 1:
        corr = matrix.corr().fillna(0.0)
        dist = 1 - np.abs(corr.values)
        np.fill_diagonal(dist, 0.0)
        n = dist.shape[0]
        col_order = [
            matrix.columns[i]
            for i in dendrogram(linkage(dist[np.triu_indices(n, k=1)], method="average"), no_plot=True)["leaves"]
        ]
    else:
        col_order = list(matrix.columns)

    matrix_ordered = matrix.loc[row_order, col_order]

    fig, (ax_main, ax_cbar) = plt.subplots(
        1, 2, figsize=(16, max(6, len(matrix_ordered) * 0.6)), gridspec_kw={"width_ratios": [20, 1], "wspace": 0.05}
    )
    max_abs = max(abs(matrix_ordered.values.min()), abs(matrix_ordered.values.max()))
    if CLIP_EFFECTS is not None:
        vmin, vmax = -CLIP_EFFECTS, CLIP_EFFECTS
        clipped = max_abs > CLIP_EFFECTS
    else:
        vmin, vmax = -max_abs, max_abs
        clipped = False

    sns.heatmap(
        matrix_ordered,
        ax=ax_main,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
    )

    # stars and sample-size y labels
    for i in range(matrix_ordered.shape[0]):
        for j in range(matrix_ordered.shape[1]):
            geno = matrix_ordered.index[i]
            metric = matrix_ordered.columns[j]
            stats_names_for_geno = reverse_map.get(geno, [])
            if not stats_names_for_geno:
                continue
            mask = (stats_df[test_col].isin(stats_names_for_geno)) & (stats_df["Metric"] == metric)
            if mask.any():
                p = stats_df.loc[mask, "pval_fdr"].values[0]
                stars = "" if p >= 0.05 else ("***" if p < 0.001 else "**" if p < 0.01 else "*")
                if stars:
                    bg = matrix_ordered.values[i, j]
                    color = "white" if abs(bg) >= 0.5 else "black"
                    ax_main.text(
                        j + 0.5, i + 0.5, stars, ha="center", va="center", color=color, fontsize=10, fontweight="bold"
                    )

    # xticklabels
    ax_main.set_xticklabels(
        [get_display_name(l.get_text()) for l in ax_main.get_xticklabels()], rotation=45, ha="right", fontsize=9
    )

    # yticklabels with sample sizes
    ylabels = []
    for tick in ax_main.get_yticklabels():
        geno = tick.get_text()
        n = int(raw_data[raw_data[group_col] == geno].shape[0]) if raw_data is not None else 0
        ylabels.append(f"{geno}\n(n = {n})")
    ax_main.set_yticklabels(ylabels, rotation=0, fontsize=9)

    # colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r")), cax=ax_cbar)
    cbar.set_label("Cohen's d\n(Blue: Lower, Red: Higher)")
    ticks = np.linspace(vmin, vmax, 5)
    labels = [f"{t:.2f}" for t in ticks]
    if clipped:
        labels[0] = f"< {vmin:.2f}"
        labels[-1] = f"> {vmax:.2f}"
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "ballscents_metrics_simple_heatmap.png"
    pdf = output_dir / "ballscents_metrics_simple_heatmap.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png}, {pdf}")


def parse_args():
    p = argparse.ArgumentParser(description="BallScents metrics heatmap generator")
    p.add_argument("--stats-file", type=str, default=None, help="Path to Mann-Whitney statistics CSV")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory to save plots")
    p.add_argument("--control", type=str, default="Ctrl", help="Control ball type (default: Ctrl)")
    return p.parse_args()


def find_stats_file_default():
    # Use canonical path produced by the Mann-Whitney BallScents analysis.
    stats_path = Path(
        "/mnt/upramdya_data/MD/Ball_scents/Plots/summaries/Genotype_Mannwhitney/genotype_mannwhitney_statistics.csv"
    )
    if stats_path.exists():
        return stats_path
    # Fall back to a local file if present
    local = Path("genotype_mannwhitney_statistics.csv")
    if local.exists():
        return local
    raise FileNotFoundError(
        f"Could not find stats file at {stats_path}; please run run_mannwhitney_ballscents.py to generate it or provide --stats-file"
    )


def load_raw_data():
    data_path = Path(
        "/mnt/upramdya_data/MD/Ball_scents/Datasets/251103_10_summary_ballscents_Data/summary/pooled_summary.feather"
    )
    print(f"Loading raw data from: {data_path}")
    df = pd.read_feather(data_path)
    return df


def main():
    args = parse_args()
    stats_file = Path(args.stats_file) if args.stats_file else find_stats_file_default()
    output_dir = Path(args.output_dir) if args.output_dir else stats_file.parent / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_df = pd.read_csv(stats_file)
    raw_data = load_raw_data()

    # Normalize BallScent labels to canonical factorial names
    raw_data = normalize_ball_scent_labels(raw_data, group_col="BallScent")

    matrix, reverse_map = build_significance_matrix(stats_df, args.control, raw_data, group_col="BallScent")
    print(f"Matrix shape: {matrix.shape}")
    plot_two_way_dendrogram(matrix, reverse_map, stats_df, output_dir, args.control, raw_data, group_col="BallScent")
    plot_simple_heatmap(matrix, reverse_map, stats_df, output_dir, args.control, raw_data, group_col="BallScent")


if __name__ == "__main__":
    main()
