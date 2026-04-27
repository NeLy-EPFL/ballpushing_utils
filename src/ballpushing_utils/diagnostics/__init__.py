"""Human-inspectable diagnostics for ball-pushing pipelines.

The diagnostics layer is a **separate concern** from the pytest suite:

* :mod:`ballpushing_utils.diagnostics` assembles human-friendly artifacts
  (event tables with ``mm:ss`` timestamps, metric summaries with NaN tallies
  and out-of-range flags, plots) so that *you* can cross-reference a
  pipeline run against the underlying video or expected ranges.
* The pytest suite under ``tests/`` asserts hermetic invariants — fast,
  no SLEAP data required, suitable for CI.

Each diagnostic is a *builder* function that returns plain pandas
DataFrames + a :class:`DiagnosticReport` dataclass. A separate writer
(:func:`write_report`) turns those data structures into a per-run folder
on disk::

    diagnostics_output/<fly-name>/<timestamp>/
        summary.md           # human-readable overview with embedded plots
        events.csv           # per-event table
        metrics.csv          # per-metric summary + flags
        plots/               # supporting PNG plots

Because the builders return ordinary data (not side-effectful prints), the
same functions are reusable from a Jupyter notebook or a panel dashboard —
see ``diagnostics/`` for CLI entry points and (future) notebook /
dashboard wrappers.
"""

from __future__ import annotations

from .report import (
    DiagnosticReport,
    DiagnosticSection,
    RangeCheck,
    write_report,
)
from .event_timeline import build_event_timeline
from .metric_report import DEFAULT_METRIC_RANGES, build_metric_report

__all__ = [
    "DEFAULT_METRIC_RANGES",
    "DiagnosticReport",
    "DiagnosticSection",
    "RangeCheck",
    "build_event_timeline",
    "build_metric_report",
    "write_report",
]
