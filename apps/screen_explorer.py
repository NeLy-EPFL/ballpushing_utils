#!/usr/bin/env python3
"""Interactive explorer for the TNT silencing-screen ballpushing metrics.

Provides two views:
- **Boxplots**: one box-and-strip plot per selected metric, grouped by genotype
- **Scatter / Correlation**: one scatter plot comparing two metrics, coloured by
  brain region or genotype

Usage
-----
    panel serve apps/screen_explorer.py --show
    # with auto-reload during development:
    panel serve apps/screen_explorer.py --show --autoreload

If the screen feather is not yet present, the app shows download instructions.
Run ``ballpushing-fetch --archive screen`` to fetch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import opts

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

# Server-relative path used by the figure scripts; find_feather resolves this
# to the Dataverse basename (ballpushing_metrics_silencing_screen.feather) when
# the flat Datasets/ layout is used by external users.
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

# Human-readable labels for metric columns (covers both old velocity* and new
# speed* names, since read_feather runs the compat shim before we see the data)
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

# Columns that should never appear as selectable metrics
_META_COLS: frozenset[str] = frozenset({
    "has_finished", "has_major", "has_significant", "nb_significant_events",
    "fly", "experiment", "Genotype", "Nickname", "Brain region", "Split",
    "Simplified Nickname", "Simplified region", "corridor", "arena",
    "fly_id", "is_dead",
})

_CONTROL_NICKNAMES: frozenset[str] = frozenset({
    "Empty-gal4", "Empty-Split", "TNTxPR", "TNTxCS",
})

DEFAULT_METRICS: list[str] = [
    "first_major_event", "max_event", "nb_events", "normalized_speed",
]

DEFAULT_REGIONS: list[str] = ["Control", "MB", "Vision"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_screen() -> pd.DataFrame | None:
    result = find_feather(_SCREEN_SERVER_PATH)
    if result is None:
        return None
    paths = result if isinstance(result, list) else [result]
    parts = [read_feather(p) for p in paths]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    # Rename columns that changed between dataset-builder versions
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
    """Map each Nickname to its brain region colour."""
    nick_to_region = (
        df_full.drop_duplicates("Nickname")
        .set_index("Nickname")["Brain region"]
        .to_dict()
    )
    return {
        nick: REGION_COLORS.get(nick_to_region.get(nick, ""), "#888888")
        for nick in sub["Nickname"].unique()
    }


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

    # Metric dropdown options: displayed as human-readable labels, values are
    # column names. We use a list of labels for MultiChoice (simpler) and a
    # separate dict to convert labels→column names in callbacks.
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
        elif regions:
            mask = df["Brain region"].isin(regions)
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
    n_cols_w = pn.widgets.IntSlider(name="Columns", start=1, end=4, value=2, width=160)

    @pn.depends(
        genotype_w.param.value,
        region_w.param.value,
        controls_w.param.value,
        boxplot_metric_w.param.value,
        n_cols_w.param.value,
    )
    def _boxplot_view(genotypes, regions, include_controls, metric_labels, n_cols):
        sub = _filtered_df(genotypes, regions, include_controls)
        metrics = [metric_label_to_col[lbl] for lbl in metric_labels if lbl in metric_label_to_col]

        if sub.empty or not metrics:
            return pn.pane.Markdown("_No data for the current selection._")

        cmap = _nick_color_map(df, sub)
        box_w = max(320, 75 * sub["Nickname"].nunique())

        plots = []
        for metric in metrics:
            if metric not in sub.columns:
                continue
            data = sub[["Nickname", metric]].dropna()
            if data.empty:
                continue

            bw = hv.BoxWhisker(data, kdims=["Nickname"], vdims=[metric]).opts(
                opts.BoxWhisker(
                    width=box_w,
                    height=370,
                    xrotation=45,
                    box_color=hv.dim("Nickname").categorize(cmap, default="#888888"),
                    whisker_color="black",
                    show_grid=True,
                    toolbar="above",
                    title=_label(metric),
                    ylabel=_label(metric),
                    xlabel="",
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
                )
            )
            plots.append(bw * sc)

        if not plots:
            return pn.pane.Markdown("_No valid metrics for the current selection._")

        layout = hv.Layout(plots).cols(n_cols)
        return pn.pane.HoloViews(layout, sizing_mode="stretch_width")

    boxplot_tab = pn.Column(
        pn.Row(boxplot_metric_w, n_cols_w),
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

        # Keep colour column + hover extras in vdims without duplication
        hover_extra = [c for c in ("Nickname", "Brain region") if c not in (x_col, y_col, color_col)]
        data = sub[[x_col, y_col, color_col] + hover_extra].dropna(subset=[x_col, y_col])

        if data.empty:
            return pn.pane.Markdown("_No data after dropping NaN values._")

        color_vals = data[color_col].unique()
        if color_col == "Brain region":
            cmap = {v: REGION_COLORS.get(v, "#888888") for v in color_vals}
        else:
            # Category20 gives 20 distinguishable colours; cycle for larger sets
            from bokeh.palettes import Category20  # type: ignore[import]
            palette = list(Category20[20]) * (len(color_vals) // 20 + 1)
            cmap = {v: palette[i] for i, v in enumerate(sorted(color_vals))}

        vdims = [color_col] + hover_extra
        scatter = hv.Scatter(data, kdims=[x_col, y_col], vdims=vdims).opts(
            opts.Scatter(
                color=hv.dim(color_col).categorize(cmap, default="#888888"),
                alpha=0.65,
                size=7,
                tools=["hover"],
                width=750,
                height=520,
                xlabel=_label(x_col),
                ylabel=_label(y_col),
                title=f"{_label(x_col)}  vs  {_label(y_col)}",
                show_grid=True,
                legend_position="right",
                show_legend=True,
                toolbar="above",
            )
        )
        return pn.pane.HoloViews(scatter, sizing_mode="stretch_width")

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
