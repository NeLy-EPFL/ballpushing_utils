"""Per-event timeline diagnostic.

Given a :class:`~ballpushing_utils.Fly`, produce a flat table with one row
per detected interaction event::

    fly_idx | ball_idx | event_idx | start_frame | end_frame | start_time_s
        | end_time_s | start_mm_ss | end_mm_ss | duration_s
        | ball_displacement_px | is_significant | is_major

Two reasons this matters:

1. Each row has ``start_mm_ss`` / ``end_mm_ss`` so you can scrub the video
   at the listed timestamps and confirm the detected event corresponds to
   real fly-ball interaction (the original motivation from M.D.).
2. Per-event ``is_significant`` and ``is_major`` flags let you spot
   threshold-tuning issues at a glance — e.g. "everything is flagged
   major" or "the first major event is way too late vs. what I see in
   the video".

Companion plot: a Gantt-like timeline of every event drawn on the same
time axis as the experiment, colour-coded by significance class. Useful
for eyeballing event density and gap distributions.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..fly import Fly
from .report import DiagnosticReport, DiagnosticSection

__all__ = ["build_event_timeline"]


def _format_mm_ss(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return ""
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def _classify(displacement_px: float, *, signif_threshold: float, major_threshold: float) -> str:
    if displacement_px is None or (isinstance(displacement_px, float) and np.isnan(displacement_px)):
        return "unknown"
    if displacement_px >= major_threshold:
        return "major"
    if displacement_px > signif_threshold:
        return "significant"
    return "minor"


_CLASS_COLOR = {"major": "#d62728", "significant": "#1f77b4", "minor": "#7f7f7f", "unknown": "#bbbbbb"}


def _events_table(
    fly: Fly,
    *,
    signif_threshold: float = 5.0,
    major_threshold: float = 20.0,
) -> pd.DataFrame:
    """Flatten ``fly.tracking_data.interaction_events`` into a DataFrame."""
    fps = float(fly.experiment.fps)
    rows: list[dict[str, object]] = []
    td = fly.tracking_data
    if td is None or td.interaction_events is None:
        return pd.DataFrame(columns=[
            "fly_idx", "ball_idx", "event_idx", "start_frame", "end_frame",
            "start_time_s", "end_time_s", "start_mm_ss", "end_mm_ss",
            "duration_s", "ball_displacement_px", "classification",
            "is_significant", "is_major",
        ])

    for fly_idx, ball_dict in td.interaction_events.items():
        for ball_idx, events in ball_dict.items():
            for event_idx, event in enumerate(events):
                start_frame, end_frame, displacement = event[0], event[1], event[2] if len(event) > 2 else np.nan
                start_s = start_frame / fps if fps else float("nan")
                end_s = end_frame / fps if fps else float("nan")
                cls = _classify(displacement, signif_threshold=signif_threshold, major_threshold=major_threshold)
                rows.append({
                    "fly_idx": fly_idx,
                    "ball_idx": ball_idx,
                    "event_idx": event_idx,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time_s": round(start_s, 3),
                    "end_time_s": round(end_s, 3),
                    "start_mm_ss": _format_mm_ss(start_s),
                    "end_mm_ss": _format_mm_ss(end_s),
                    "duration_s": round(end_s - start_s, 3),
                    "ball_displacement_px": float(displacement) if displacement is not None else float("nan"),
                    "classification": cls,
                    "is_significant": cls in ("significant", "major"),
                    "is_major": cls == "major",
                })
    return pd.DataFrame(rows)


def _make_timeline_figure(events: pd.DataFrame, fly_name: str) -> plt.Figure:
    """Gantt-style plot of event start/end times, coloured by classification."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    if events.empty:
        ax.text(0.5, 0.5, "No interaction events detected.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # One row per (fly_idx, ball_idx) combo.
    combos = events[["fly_idx", "ball_idx"]].drop_duplicates().reset_index(drop=True)
    combo_to_row = {(r.fly_idx, r.ball_idx): i for i, r in combos.iterrows()}
    for _, ev in events.iterrows():
        y = combo_to_row[(ev["fly_idx"], ev["ball_idx"])]
        ax.hlines(
            y,
            ev["start_time_s"],
            ev["end_time_s"],
            colors=_CLASS_COLOR.get(ev["classification"], "#bbbbbb"),
            linewidth=4,
        )
    ax.set_yticks(list(combo_to_row.values()))
    ax.set_yticklabels([f"fly {f}, ball {b}" for (f, b) in combo_to_row])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Interaction events — {fly_name}")
    # Manual legend
    handles = [plt.Line2D([], [], color=c, linewidth=4, label=k) for k, c in _CLASS_COLOR.items() if k != "unknown"]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def build_event_timeline(
    fly: Fly,
    *,
    signif_threshold: float = 5.0,
    major_threshold: float = 20.0,
) -> DiagnosticReport:
    """Assemble an event-timeline diagnostic for a single :class:`Fly`.

    Parameters
    ----------
    fly:
        The fly to inspect.
    signif_threshold, major_threshold:
        Pixel-displacement thresholds used to bucket each event into
        ``minor`` / ``significant`` / ``major``. Defaults match
        ``Config.signif_event_threshold`` and ``Config.major_event_threshold``.

    Returns
    -------
    DiagnosticReport
        Two sections: the per-event table and the Gantt-style timeline
        figure. Pass to :func:`~ballpushing_utils.diagnostics.write_report`
        to materialise on disk.
    """
    name = fly.metadata.name if hasattr(fly, "metadata") else str(fly.directory)
    table = _events_table(fly, signif_threshold=signif_threshold, major_threshold=major_threshold)

    report = DiagnosticReport(
        title="Interaction event timeline",
        subject=name,
        metadata={
            "fps": float(fly.experiment.fps),
            "signif_threshold_px": signif_threshold,
            "major_threshold_px": major_threshold,
            "n_events": len(table),
        },
    )
    report.add(DiagnosticSection(
        name="events",
        description=(
            "One row per detected interaction event. Use ``start_mm_ss`` / "
            "``end_mm_ss`` to scrub the video and confirm each event."
        ),
        table=table,
    ))
    report.add(DiagnosticSection(
        name="timeline",
        description="Gantt-style view of every event on the experiment time axis.",
        figure=_make_timeline_figure(table, name),
    ))
    if table.empty:
        report.sections[0].notes.append(
            "No interaction events detected. If the fly was active in the video, "
            "investigate `interaction_threshold` and `events_min_length` in `Config`."
        )
    return report


def build_event_timelines_for_experiment(experiment) -> Iterable[DiagnosticReport]:
    """Convenience: yield one :class:`DiagnosticReport` per fly in the experiment."""
    for fly in experiment.flies:
        yield build_event_timeline(fly)
