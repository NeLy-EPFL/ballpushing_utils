"""Per-fly metric summary diagnostic.

Walks ``fly.event_summaries`` (the dict that
:class:`~ballpushing_utils.BallPushingMetrics` produces) and emits a flat
table::

    fly_idx | ball_idx | metric | value | dtype | verdict | note

with one row per (fly_idx, ball_idx, metric) triple. ``verdict`` is one of
``ok``, ``low``, ``high``, ``nan`` according to the corresponding
:class:`~ballpushing_utils.diagnostics.RangeCheck` in
:data:`DEFAULT_METRIC_RANGES`. Metrics for which no range is declared get
``verdict="ok"`` but are still listed in the table — never silently
skipped.

The accompanying summary section reports counts of NaNs and
out-of-range values per metric, which is what you usually want to
eyeball first ("the median pause duration is suddenly NaN for half the
flies — what changed?").
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from ..fly import Fly
from .report import DiagnosticReport, DiagnosticSection, RangeCheck

__all__ = ["DEFAULT_METRIC_RANGES", "build_metric_report"]


def _r(metric: str, lo=None, hi=None, note: str = "") -> RangeCheck:
    return RangeCheck(metric=metric, lo=lo, hi=hi, note=note)


#: Plausible ranges for the metrics produced by :class:`BallPushingMetrics`.
#: Bounds are deliberately generous — a "high" verdict means *probably worth
#: a look*, not "definitely wrong". Add or tighten entries as confidence in
#: the pipeline grows.
DEFAULT_METRIC_RANGES: Mapping[str, RangeCheck] = {
    rc.metric: rc
    for rc in [
        _r("nb_events", lo=0, hi=2000, note="number of detected interaction events"),
        _r("nb_significant_events", lo=0, hi=2000),
        _r("significant_ratio", lo=0.0, hi=1.0),
        _r("max_distance", lo=0.0, hi=2000.0, note="px"),
        _r("raw_max_distance", lo=0.0, hi=2000.0, note="px"),
        _r("distance_moved", lo=0.0, hi=20000.0, note="px"),
        _r("raw_distance_moved", lo=0.0, hi=20000.0, note="px"),
        _r("distance_ratio", lo=0.0),
        _r("first_significant_event", lo=0),
        _r("first_significant_event_time", lo=0.0, note="s"),
        _r("first_major_event", lo=0),
        _r("first_major_event_time", lo=0.0, note="s"),
        _r("final_event", lo=0),
        _r("final_event_time", lo=0.0, note="s"),
        _r("max_event", lo=0),
        _r("max_event_time", lo=0.0),
        _r("chamber_time", lo=0.0, note="s"),
        _r("chamber_ratio", lo=0.0, hi=1.0),
        _r("chamber_exit_time", lo=0.0),
        _r("time_chamber_beginning", lo=0.0),
        _r("time_to_first_interaction", lo=0.0),
        _r("cumulated_breaks_duration", lo=0.0),
        _r("interaction_persistence", lo=0.0),
        _r("interaction_proportion", lo=0.0, hi=1.0),
        _r("persistence_at_end", lo=0.0, hi=1.0),
        _r("fraction_not_facing_ball", lo=0.0, hi=1.0),
        _r("pulling_ratio", lo=0.0, hi=1.0),
        _r("head_pushing_ratio", lo=0.0, hi=1.0),
        _r("leg_visibility_ratio", lo=0.0, hi=1.0),
        _r("nb_stops", lo=0),
        _r("nb_pauses", lo=0),
        _r("nb_long_pauses", lo=0),
        _r("total_stop_duration", lo=0.0),
        _r("total_pause_duration", lo=0.0),
        _r("total_long_pause_duration", lo=0.0),
        _r("median_stop_duration", lo=0.0),
        _r("median_pause_duration", lo=0.0),
        _r("median_long_pause_duration", lo=0.0),
        _r("has_finished", lo=0, hi=1),
        _r("has_significant", lo=0, hi=1),
        _r("has_major", lo=0, hi=1),
        _r("has_long_pauses", lo=0, hi=1),
        _r("overall_interaction_rate", lo=0.0),
        _r("speed_during_interactions", lo=0.0),
        _r("normalized_speed", lo=0.0),
        _r("fly_distance_moved", lo=0.0, note="mm"),
    ]
}


def _is_scalar(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)) or value is None


_LONG_TABLE_COLUMNS = ["fly_idx", "ball_idx", "key", "metric", "value", "dtype", "verdict", "note"]


def _flatten_summaries(summaries: dict) -> pd.DataFrame:
    """Walk ``{key: metrics_dict}`` and return a long DataFrame.

    The schema is fixed (see :data:`_LONG_TABLE_COLUMNS`) even when no
    rows are emitted — downstream code (``build_metric_report``,
    ``write_report``) relies on the columns existing.
    """
    rows: list[dict[str, object]] = []
    for key, metrics in summaries.items():
        if not isinstance(metrics, dict):
            continue
        # key is "fly_<i>_ball_<j>" or similar — extract for the table.
        fly_idx = metrics.get("fly_idx", "")
        ball_idx = metrics.get("ball_idx", "")
        for metric_name, value in metrics.items():
            if metric_name in ("fly_idx", "ball_idx", "ball_identity"):
                continue
            if not _is_scalar(value):
                continue
            v = float(value) if value is not None and not isinstance(value, bool) else value
            check = DEFAULT_METRIC_RANGES.get(metric_name)
            if check is None:
                verdict = "ok"
                note = ""
            else:
                verdict = check.verdict(v if v is not None else float("nan"))
                note = check.note
            rows.append({
                "fly_idx": fly_idx,
                "ball_idx": ball_idx,
                "key": key,
                "metric": metric_name,
                "value": v,
                "dtype": type(value).__name__,
                "verdict": verdict,
                "note": note,
            })
    if not rows:
        return pd.DataFrame(columns=_LONG_TABLE_COLUMNS)
    return pd.DataFrame(rows, columns=_LONG_TABLE_COLUMNS)


def _summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the long table into one row per metric."""
    if table.empty:
        return pd.DataFrame(
            columns=["metric", "n_observed", "n_nan", "n_low", "n_high", "min", "max"]
        )

    def _agg(group: pd.DataFrame) -> pd.Series:
        numeric = pd.to_numeric(group["value"], errors="coerce")
        return pd.Series({
            "n_observed": len(group),
            "n_nan": int((group["verdict"] == "nan").sum()),
            "n_low": int((group["verdict"] == "low").sum()),
            "n_high": int((group["verdict"] == "high").sum()),
            "min": numeric.min(skipna=True),
            "max": numeric.max(skipna=True),
        })

    # ``include_groups=False`` silences a pandas FutureWarning and ensures
    # the grouping column isn't passed into ``_agg`` itself.
    try:
        return table.groupby("metric", sort=True).apply(_agg, include_groups=False).reset_index()
    except TypeError:
        # Older pandas (<2.1) doesn't accept ``include_groups``.
        return table.groupby("metric", sort=True).apply(_agg).reset_index()


def build_metric_report(fly: Fly) -> DiagnosticReport:
    """Diagnostic report for ``fly.event_summaries``.

    Sections returned (in order):

    1. ``summary`` — one row per metric with NaN / low / high counts and
       observed range.
    2. ``flagged`` — only the rows whose verdict is not ``ok`` (typically
       what you want to look at first).
    3. ``all_metrics`` — the full long table (one row per
       ``(fly_idx, ball_idx, metric)``).
    """
    name = fly.metadata.name if hasattr(fly, "metadata") else str(fly.directory)
    summaries = fly.event_summaries or {}
    long_table = _flatten_summaries(summaries)
    summary_table = _summarise(long_table)
    flagged_table = long_table[long_table["verdict"].isin(("nan", "low", "high"))].reset_index(drop=True)

    report = DiagnosticReport(
        title="Ball-pushing metric report",
        subject=name,
        metadata={
            "n_metric_rows": len(long_table),
            "n_flagged": len(flagged_table),
            "n_metrics_with_range": int((long_table["verdict"] != "ok").any()) if not long_table.empty else 0,
        },
    )
    report.add(DiagnosticSection(
        name="summary",
        description="One row per metric: counts of NaN / out-of-range values and observed min/max.",
        table=summary_table,
    ))
    report.add(DiagnosticSection(
        name="flagged",
        description="Only rows whose verdict is `nan`, `low`, or `high`. This is usually where to start.",
        table=flagged_table,
        notes=(
            ["Nothing flagged — every metric falls inside the declared plausible range."]
            if flagged_table.empty else []
        ),
    ))
    report.add(DiagnosticSection(
        name="all_metrics",
        description="Full long table: one row per (fly_idx, ball_idx, metric).",
        table=long_table,
    ))
    return report
