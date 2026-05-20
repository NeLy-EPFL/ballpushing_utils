#!/usr/bin/env python3
"""Interactive explorer for the TNT silencing-screen ballpushing metrics.

Provides two views:
- **Boxplots**: one box-and-strip plot per selected metric, grouped by genotype,
  with a bootstrapped 95 % CI band for the control group as a reference
- **Scatter / Correlation**: two-metric scatter coloured by brain region or genotype;
  controls always rendered as diamonds with a 95 % data-confidence ellipse

Usage
-----
    panel serve apps/screen_explorer.py --show
    panel serve apps/screen_explorer.py --show --autoreload   # dev mode

If the screen feather is not yet present, the app shows download instructions.
Run ``ballpushing-fetch --archive screen`` to fetch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import panel as pn
    import holoviews as hv
    from holoviews import opts
except ImportError as _exc:
    sys.exit(
        f"This app requires the '[interactive]' extras (panel, holoviews, bokeh).\n"
        f"Install them with:\n\n"
        f"    pip install -e '.[interactive]'\n\n"
        f"Missing package: {_exc.name}"
    )

pn.extension("bokeh")
hv.extension("bokeh")

# Allow running from a fresh clone without pip install
try:
    import ballpushing_utils  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ballpushing_utils import read_feather
from ballpushing_utils.paths import find_feather


# ── Constants ─────────────────────────────────────────────────────────────────

# Server-relative path; find_feather resolves this to the Dataverse basename
# (ballpushing_metrics_silencing_screen.feather) for external users.
_SCREEN_SERVER_PATH = (
    "Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data"
    "/summary/pooled_summary.feather"
)

REGION_COLORS: dict[str, str] = {
    "MB": "#1f77b4",
    "Vision": "#ff7f0e",
    "LH": "#2ca02c",
    "Neuropeptide": "#d62728",
    "Olfaction": "#9467bd",
    "MB extrinsic neurons": "#8c564b",
    "CX": "#e377c2",
    "Control": "#7f7f7f",
    "DN": "#bcbd22",
    "fchON": "#17becf",
    "JON": "#ffbb78",
}

# Covers both old velocity* and new speed* column names (read_feather compat shim
# normalises them before the data reaches this app, but the dict handles edge cases)
METRIC_LABELS: dict[str, str] = {
    "first_major_event": "First major event rank",
    "major_event": "First major event rank",
    "first_major_event_time": "First major event time (s)",
    "major_event_time": "First major event time (s)",
    "max_event": "Max ball displacement event rank",
    "max_event_time": "Max ball displacement time (s)",
    "nb_events": "Number of contact events",
    "normalized_speed": "Normalized walking speed",
    "normalized_velocity": "Normalized walking speed",
    "speed_trend": "Speed trend (slope)",
    "velocity_trend": "Speed trend (slope)",
    "speed_during_interactions": "Speed during interactions",
    "velocity_during_interactions": "Speed during interactions",
    "head_pushing_ratio": "Head-pushing ratio",
    "pulling_ratio": "Pulling ratio",
    "pulled": "Significant pulling events (#)",
    "significant_ratio": "Fraction of significant events",
    "distance_moved": "Total distance moved (mm)",
    "max_distance": "Max ball displacement (mm)",
    "interaction_persistence": "Avg. interaction duration (s)",
    "chamber_exit_time": "Chamber exit time (s)",
    "chamber_ratio": "Chamber ratio",
    "distance_ratio": "Distance ratio",
    "nb_freeze": "Number of freezes",
    "flailing": "Flailing fraction",
    "fraction_not_facing_ball": "Fraction not facing ball",
    "number_of_pauses": "Number of pauses",
    "persistence_at_end": "Persistence at end",
    "time_chamber_beginning": "Time to first chamber contact (s)",
}

_META_COLS: frozenset[str] = frozenset({
    "has_finished", "has_major", "has_significant", "nb_significant_events",
    "fly", "experiment", "Genotype", "Nickname", "Brain region", "Split",
    "Simplified Nickname", "Simplified region", "corridor", "arena",
    "fly_id", "is_dead",
})

_CONTROL_NICKNAMES: frozenset[str] = frozenset({
    "Empty-gal4", "Empty-Split", "TNTxPR", "TNTxCS",
})

# One metric by default keeps the initial view lightweight
DEFAULT_METRICS: list[str] = ["first_major_event"]
DEFAULT_REGIONS: list[str] = ["Control", "MB", "Vision"]

# Fixed seed for reproducible bootstrap draws
_RNG = np.random.default_rng(42)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_screen() -> pd.DataFrame | None:
    result = find_feather(_SCREEN_SERVER_PATH)
    if result is None:
        return None
    paths = result if isinstance(result, list) else [result]
    parts = [read_feather(p) for p in paths]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    for old, new in [
        ("major_event", "first_major_event"),
        ("major_event_time", "first_major_event_time"),
    ]:
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    return df


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if c not in _META_COLS]


def _label(col: str) -> str:
    return METRIC_LABELS.get(col, col.replace("_", " ").title())


def _nick_color_map(df_full: pd.DataFrame, sub: pd.DataFrame) -> dict[str, str]:
    """Map each Nickname to its brain-region colour."""
    nick_to_region = (
        df_full.drop_duplicates("Nickname")
        .set_index("Nickname")["Brain region"]
        .to_dict()
    )
    return {
        nick: REGION_COLORS.get(nick_to_region.get(nick, ""), "#888888")
        for nick in sub["Nickname"].unique()
    }


# ── Statistical helpers ───────────────────────────────────────────────────────

def _bootstrap_ci_median(
    arr: np.ndarray, *, n_boot: int = 1000, ci: float = 95
) -> tuple[float, float]:
    """Vectorised bootstrap CI of the median (reproducible via module-level RNG)."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan, np.nan
    boots = _RNG.choice(arr, size=(n_boot, len(arr)), replace=True)
    medians = np.median(boots, axis=1)
    lo = float(np.percentile(medians, (100 - ci) / 2))
    hi = float(np.percentile(medians, 100 - (100 - ci) / 2))
    return lo, hi


def _confidence_ellipse_xy(
    x: np.ndarray, y: np.ndarray, *, n_std: float = 2.45, n_pts: int = 120
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Path tracing a bivariate data-confidence ellipse.

    n_std=2.45 ≈ sqrt(chi2(2, 0.95)) — encloses ~95 % of bivariate-normal data.
    """
    xy = np.column_stack([x, y])
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) < 4:
        return None, None
    mean = xy.mean(axis=0)
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.arctan2(*vecs[:, 0][::-1])
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    t = np.linspace(0, 2 * np.pi, n_pts + 1)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ex, ey = w / 2 * np.cos(t), h / 2 * np.sin(t)
    return cos_a * ex - sin_a * ey + mean[0], sin_a * ex + cos_a * ey + mean[1]


# ── Bokeh hook for enforcing categorical axis order ───────────────────────────

def _make_order_hook(nick_order: list[str]):
    """Return a finalise hook that sets FactorRange.factors to nick_order.

    hv.Dimension(values=...) and pd.Categorical are both silently ignored by
    BoxWhisker's internal grouping.  The only reliable fix is to reach into the
    underlying bokeh figure after rendering and overwrite the FactorRange.
    """
    def _hook(plot, element) -> None:
        try:
            from bokeh.models import FactorRange
            fig = plot.handles["plot"]
            for rng in (fig.x_range, fig.y_range):
                if isinstance(rng, FactorRange):
                    rng.factors = list(nick_order)
        except Exception:
            pass
    return _hook


# ── Missing-data screen ───────────────────────────────────────────────────────

def _missing_app() -> pn.template.BootstrapTemplate:
    alert = pn.pane.Alert(
        """
**Screen data not found.**

Download it with:

```
ballpushing-fetch --archive screen
```

or, if `ballpushing_utils` is not installed yet:

```
python -m ballpushing_utils.dataverse_download --archive screen
```

Files land in `Datasets/` by default (or `$BALLPUSHING_DATA_ROOT` if set).
Restart this app once the download completes.
""",
        alert_type="warning",
    )
    return pn.template.BootstrapTemplate(
        title="Screen Explorer — data not available",
        main=[alert],
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def _build_app(df: pd.DataFrame) -> pn.template.BootstrapTemplate:
    all_regions = sorted(r for r in df["Brain region"].unique() if pd.notna(r) and r != "None")
    all_metrics = _metric_cols(df)
    default_regions = [r for r in DEFAULT_REGIONS if r in all_regions]
    default_metrics = [c for c in DEFAULT_METRICS if c in all_metrics]

    # Labels displayed in dropdowns; reverse mapping used to recover column names
    metric_label_to_col: dict[str, str] = {_label(c): c for c in all_metrics}
    metric_labels_ordered: list[str] = list(metric_label_to_col)

    # ── Shared sidebar widgets ─────────────────────────────────────────────────

    region_w = pn.widgets.MultiChoice(
        name="Brain regions",
        options=all_regions,
        value=default_regions,
    )
    genotype_w = pn.widgets.MultiChoice(
        name="Genotypes",
        options=[],
        value=[],
        placeholder="All in selected regions",
    )
    controls_w = pn.widgets.Checkbox(name="Include controls", value=True)

    @pn.depends(region_w.param.value, watch=True)
    def _sync_genotypes(regions: list[str]) -> None:
        mask = df["Brain region"].isin(regions) if regions else pd.Series(True, index=df.index)
        candidates = sorted(df.loc[mask, "Nickname"].unique())
        genotype_w.options = candidates
        genotype_w.value = [v for v in genotype_w.value if v in candidates]

    _sync_genotypes(default_regions)

    def _filtered_df(genotypes: list[str], regions: list[str], include_controls: bool) -> pd.DataFrame:
        if genotypes:
            mask = df["Nickname"].isin(genotypes)
            # Controls are always appended when the toggle is on so they provide
            # a visible reference regardless of which experimental subset is chosen.
            if include_controls:
                mask = mask | df["Nickname"].isin(_CONTROL_NICKNAMES)
        elif regions:
            mask = df["Brain region"].isin(regions)
            if not include_controls:
                mask = mask & ~df["Nickname"].isin(_CONTROL_NICKNAMES)
        else:
            mask = pd.Series(True, index=df.index)
            if not include_controls:
                mask = mask & ~df["Nickname"].isin(_CONTROL_NICKNAMES)
        return df[mask].copy()

    @pn.depends(genotype_w.param.value, region_w.param.value, controls_w.param.value)
    def _fly_count(genotypes, regions, include_controls):
        n = len(_filtered_df(genotypes, regions, include_controls))
        return pn.pane.Markdown(f"**{n:,} flies** in selection")

    # ── Boxplot tab ────────────────────────────────────────────────────────────

    boxplot_metric_w = pn.widgets.MultiChoice(
        name="Metrics",
        options=metric_labels_ordered,
        value=[_label(c) for c in default_metrics],
    )
    n_cols_w = pn.widgets.IntSlider(name="Columns", start=1, end=4, value=1, width=160)
    invert_axes_w = pn.widgets.Checkbox(name="Horizontal layout", value=True)
    sort_desc_w = pn.widgets.Checkbox(name="High to low", value=False)

    @pn.depends(
        genotype_w.param.value,
        region_w.param.value,
        controls_w.param.value,
        boxplot_metric_w.param.value,
        n_cols_w.param.value,
        invert_axes_w.param.value,
        sort_desc_w.param.value,
    )
    def _boxplot_view(genotypes, regions, include_controls, metric_labels, n_cols, invert_axes, sort_desc):
        sub = _filtered_df(genotypes, regions, include_controls)
        metrics = [metric_label_to_col[lbl] for lbl in metric_labels if lbl in metric_label_to_col]

        if sub.empty or not metrics:
            return pn.pane.Markdown("_No data for the current selection._")

        cmap = _nick_color_map(df, sub)
        ctrl_mask = sub["Nickname"].isin(_CONTROL_NICKNAMES)
        n_nicks = sub["Nickname"].nunique()
        # Width scales with genotype count; height is fixed for readability
        box_w = max(340, 75 * n_nicks)

        plots = []
        for metric in metrics:
            if metric not in sub.columns:
                continue
            data = sub[["Nickname", metric]].dropna()
            if data.empty:
                continue

            # Sort genotypes by per-nickname median.
            # hv.Dimension(values=...) and pd.Categorical are both ignored by
            # BoxWhisker's internal grouping.  We use a bokeh finalise hook that
            # sets FactorRange.factors directly after the figure is built.
            nick_order = (
                data.groupby("Nickname")[metric]
                .median()
                .sort_values(ascending=sort_desc)
                .index.tolist()
            )

            bw = hv.BoxWhisker(data, kdims=["Nickname"], vdims=[metric]).opts(
                opts.BoxWhisker(
                    width=400 if invert_axes else box_w,
                    height=max(300, 22 * n_nicks) if invert_axes else 380,
                    xrotation=0 if invert_axes else 45,
                    box_fill_color="white",
                    box_fill_alpha=0,
                    box_line_color="black",
                    box_line_width=1.2,
                    whisker_color="black",
                    show_grid=True,
                    toolbar="above",
                    title=_label(metric),
                    ylabel="" if invert_axes else _label(metric),
                    xlabel=_label(metric) if invert_axes else "",
                    invert_axes=invert_axes,
                    hooks=[_make_order_hook(nick_order)],
                )
            )
            sc = hv.Scatter(data, kdims=["Nickname"], vdims=[metric]).opts(
                opts.Scatter(
                    color=hv.dim("Nickname").categorize(cmap, default="#888888"),
                    alpha=0.45,
                    size=5,
                    jitter=0.4,
                    tools=["hover"],
                    toolbar=None,
                    invert_axes=invert_axes,
                )
            )
            layers: list = [bw, sc]

            # Bootstrapped 95 % CI band from all controls combined.
            # Rendered behind box/scatter; disappears automatically when controls
            # are hidden, so no separate toggle is needed.
            if include_controls:
                ctrl_vals = sub.loc[ctrl_mask, metric].dropna().values
                if len(ctrl_vals) >= 2:
                    ci_lo, ci_hi = _bootstrap_ci_median(ctrl_vals)
                    if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                        # VSpan for horizontal layout (value axis is now x),
                        # HSpan for vertical (value axis is y)
                        Span = hv.VSpan if invert_axes else hv.HSpan
                        band = Span(ci_lo, ci_hi).opts(
                            fill_color="#888888",
                            fill_alpha=0.18,
                            line_color="#555555",
                            line_alpha=0.45,
                        )
                        layers.insert(0, band)

            plots.append(hv.Overlay(layers))

        if not plots:
            return pn.pane.Markdown("_No valid metrics for the current selection._")

        layout = hv.Layout(plots).cols(n_cols)
        return pn.pane.HoloViews(layout, sizing_mode="stretch_width")

    boxplot_tab = pn.Column(
        pn.Row(boxplot_metric_w, n_cols_w, invert_axes_w, sort_desc_w),
        _boxplot_view,
        sizing_mode="stretch_width",
    )

    # ── Scatter tab ────────────────────────────────────────────────────────────

    scatter_x_w = pn.widgets.Select(
        name="X axis",
        options=metric_label_to_col,
        value=all_metrics[0] if all_metrics else None,
    )
    scatter_y_w = pn.widgets.Select(
        name="Y axis",
        options=metric_label_to_col,
        value=all_metrics[1] if len(all_metrics) > 1 else None,
    )
    color_by_w = pn.widgets.Select(
        name="Colour by",
        options={"Brain region": "Brain region", "Genotype": "Nickname"},
        value="Brain region",
    )

    @pn.depends(
        genotype_w.param.value,
        region_w.param.value,
        controls_w.param.value,
        scatter_x_w.param.value,
        scatter_y_w.param.value,
        color_by_w.param.value,
    )
    def _scatter_view(genotypes, regions, include_controls, x_col, y_col, color_col):
        sub = _filtered_df(genotypes, regions, include_controls)

        if sub.empty or x_col is None or y_col is None:
            return pn.pane.Markdown("_No data for the current selection._")

        # Build column list without duplicates (order matters for vdims)
        seen: set[str] = set()
        all_cols: list[str] = []
        for c in [x_col, y_col, color_col, "Nickname", "Brain region"]:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)
        data = sub[all_cols].dropna(subset=[x_col, y_col])

        if data.empty:
            return pn.pane.Markdown("_No data after dropping NaN values._")

        color_vals = data[color_col].unique()
        if color_col == "Brain region":
            cmap = {v: REGION_COLORS.get(v, "#888888") for v in color_vals}
        else:
            from bokeh.palettes import Category20  # type: ignore[import]
            palette = list(Category20[20]) * (len(color_vals) // 20 + 1)
            cmap = {v: palette[i] for i, v in enumerate(sorted(color_vals))}

        vdims = [c for c in all_cols if c not in (x_col, y_col)]
        ctrl_mask = data["Nickname"].isin(_CONTROL_NICKNAMES)
        ctrl_data = data[ctrl_mask]
        main_data = data[~ctrl_mask]

        # Base display opts shared by both scatter layers
        base = dict(
            width=750, height=520,
            xlabel=_label(x_col), ylabel=_label(y_col),
            title=f"{_label(x_col)}  vs  {_label(y_col)}",
            show_grid=True, toolbar="above",
        )

        layers: list = []

        # ── Per-control confidence ellipses (rendered first, behind points) ──
        # One ellipse per control nickname so their individual spreads are visible.
        if include_controls:
            for ctrl_nick in sorted(ctrl_data["Nickname"].unique()):
                cd = ctrl_data[ctrl_data["Nickname"] == ctrl_nick]
                if len(cd) < 4:
                    continue
                xs, ys = _confidence_ellipse_xy(cd[x_col].values, cd[y_col].values)
                if xs is None:
                    continue
                # Use the control's own colour from the active cmap
                ctrl_color = cmap.get(
                    cd[color_col].iloc[0] if color_col == "Brain region" else ctrl_nick,
                    REGION_COLORS["Control"],
                )
                layers.append(
                    hv.Polygons([{"x": xs, "y": ys}]).opts(
                        opts.Polygons(
                            fill_color=ctrl_color,
                            fill_alpha=0.12,
                            line_color=ctrl_color,
                            line_alpha=0.8,
                            line_width=1.5,
                            line_dash="dashed",
                            **base,
                        )
                    )
                )

        # ── Non-control scatter (circles) ──────────────────────────────────────
        if not main_data.empty:
            layers.append(
                hv.Scatter(main_data, kdims=[x_col, y_col], vdims=vdims).opts(
                    opts.Scatter(
                        color=hv.dim(color_col).categorize(cmap, default="#888888"),
                        alpha=0.65,
                        size=7,
                        marker="circle",
                        tools=["hover"],
                        show_legend=True,
                        legend_position="right",
                        **base,
                    )
                )
            )

        # ── Control scatter (diamonds, on top, with black outline) ─────────────
        if include_controls and not ctrl_data.empty:
            layers.append(
                hv.Scatter(ctrl_data, kdims=[x_col, y_col], vdims=vdims).opts(
                    opts.Scatter(
                        color=hv.dim(color_col).categorize(cmap, default="#888888"),
                        alpha=0.9,
                        size=11,
                        marker="diamond",
                        line_color="black",
                        line_width=1.2,
                        tools=["hover"],
                        # Legend handled by the main scatter above; showing it
                        # here would duplicate entries for the same colour groups.
                        show_legend=main_data.empty,
                        legend_position="right",
                        **base,
                    )
                )
            )

        if not layers:
            return pn.pane.Markdown("_No data to display._")

        return pn.pane.HoloViews(hv.Overlay(layers), sizing_mode="stretch_width")

    scatter_tab = pn.Column(
        pn.Row(scatter_x_w, scatter_y_w, color_by_w),
        _scatter_view,
        sizing_mode="stretch_width",
    )

    # ── Assemble ───────────────────────────────────────────────────────────────

    sidebar = pn.Column(
        pn.pane.Markdown("### Filters"),
        region_w,
        genotype_w,
        controls_w,
        pn.layout.Divider(),
        _fly_count,
        width=280,
    )

    tabs = pn.Tabs(
        ("Boxplots", boxplot_tab),
        ("Scatter / Correlation", scatter_tab),
        sizing_mode="stretch_width",
        dynamic=True,
    )

    return pn.template.BootstrapTemplate(
        title="Screen Explorer — TNT Silencing Screen",
        sidebar=[sidebar],
        main=[tabs],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

_df = _load_screen()
app = _build_app(_df) if _df is not None else _missing_app()
app.servable()
