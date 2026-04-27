#!/usr/bin/env python3
"""Interactive Panel dashboard over the ball-pushing diagnostics.

Usage
-----

Install the interactive extras (``panel``, ``bokeh``, ...) then run::

    pip install -e ".[dev,interactive]"
    python tools/diagnostics_dashboard.py /mnt/…/arena2/corridor5

or, to serve it as a browser app on ``http://localhost:5006``::

    panel serve tools/diagnostics_dashboard.py --args /mnt/…/arena2/corridor5

The dashboard exposes:

- The per-event table with ``start_mm_ss`` / ``end_mm_ss`` so you can
  scrub the video at the timestamps of flagged events.
- The Gantt-style event timeline, redrawn live as you move the
  significant / major pixel-displacement sliders.
- The per-metric report with verdict flags (``ok`` / ``low`` /
  ``high`` / ``nan``) against ``DEFAULT_METRIC_RANGES``.

The builders in :mod:`ballpushing_utils.diagnostics` do all the heavy
lifting; this module is purely UI glue. Every widget callback just
re-invokes a builder and hands the resulting DataFrame / Figure to
Panel. Keep this file thin — any analysis logic belongs in the
diagnostics subpackage where the hermetic unit tests can lock it
down.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ballpushing_utils import Fly
from ballpushing_utils.diagnostics import (
    build_event_timeline,
    build_metric_report,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "fly_path",
        type=Path,
        help="Absolute path to a fly directory "
        "(the one that contains SLEAP HDF5s and metadata).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port for ``panel serve`` (default: 5006). Ignored when the "
        "dashboard is already being served via ``panel serve``.",
    )
    return parser.parse_args(argv)


def _load_fly(fly_path: Path) -> Fly:
    if not fly_path.exists():
        raise SystemExit(
            f"Fly path not found: {fly_path}\n"
            "Check that BALLPUSHING_DATA_ROOT is set and the path is correct."
        )
    return Fly(fly_path, as_individual=True)


def build_app(fly: Fly):
    """Return a Panel ``Column`` wrapping the diagnostics for *fly*.

    Imported lazily so the module is importable even in environments
    without the interactive extras installed (e.g. CI). If ``panel`` is
    missing, raise a friendly error with the install hint.
    """
    try:
        import panel as pn
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "The diagnostics dashboard needs the `interactive` extras "
            "(panel, bokeh, ...). Install them with "
            "`pip install -e \".[interactive]\"` and try again."
        ) from exc

    pn.extension("tabulator")

    signif_slider = pn.widgets.FloatSlider(
        name="Significant threshold (px)",
        value=5.0,
        start=0.5,
        end=50.0,
        step=0.5,
    )
    major_slider = pn.widgets.FloatSlider(
        name="Major threshold (px)",
        value=20.0,
        start=5.0,
        end=200.0,
        step=1.0,
    )

    @pn.depends(signif_slider, major_slider)
    def event_table(signif: float, major: float):
        report = build_event_timeline(
            fly, signif_threshold=signif, major_threshold=max(major, signif)
        )
        events = next(s for s in report.sections if s.name == "events").table
        if events.empty:
            return pn.pane.Markdown("_No interaction events detected._")
        return pn.widgets.Tabulator(
            events,
            pagination="remote",
            page_size=15,
            sizing_mode="stretch_width",
        )

    @pn.depends(signif_slider, major_slider)
    def event_figure(signif: float, major: float):
        report = build_event_timeline(
            fly, signif_threshold=signif, major_threshold=max(major, signif)
        )
        fig = next(s for s in report.sections if s.name == "timeline").figure
        return pn.pane.Matplotlib(fig, tight=True, dpi=100, sizing_mode="stretch_width")

    def metric_table():
        report = build_metric_report(fly)
        metrics = next((s for s in report.sections if s.name == "metrics"), None)
        if metrics is None or metrics.table is None or metrics.table.empty:
            return pn.pane.Markdown("_`fly.event_summaries` is empty._")
        return pn.widgets.Tabulator(
            metrics.table,
            pagination="remote",
            page_size=20,
            sizing_mode="stretch_width",
            # Colour verdicts so out-of-range rows pop visually.
            formatters={
                "verdict": {
                    "type": "lookup",
                    "params": {
                        "ok": "ok",
                        "low": "⚠ low",
                        "high": "⚠ high",
                        "nan": "⚠ nan",
                    },
                },
            },
        )

    def metric_summary():
        report = build_metric_report(fly)
        summary = next((s for s in report.sections if s.name == "summary"), None)
        if summary is None or summary.table is None or summary.table.empty:
            return pn.pane.Markdown("_No per-metric summary available._")
        return pn.widgets.Tabulator(
            summary.table, sizing_mode="stretch_width", disabled=True
        )

    name = getattr(fly.metadata, "name", str(fly.directory))
    header = pn.pane.Markdown(
        f"# Diagnostics — {name}\n"
        f"_Path:_ `{fly.directory}`\n\n"
        "Adjust the thresholds below to re-classify events on the fly "
        "(they do not touch the underlying data)."
    )

    return pn.Column(
        header,
        pn.Row(signif_slider, major_slider),
        pn.Card(event_figure, title="Event timeline (Gantt)", collapsed=False),
        pn.Card(event_table, title="Events", collapsed=False),
        pn.Card(metric_table, title="Metrics", collapsed=True),
        pn.Card(metric_summary, title="Metric summary", collapsed=True),
        sizing_mode="stretch_width",
    )


def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    fly = _load_fly(args.fly_path)
    app = build_app(fly)

    # When this script is executed by `panel serve`, ``pn.state.served``
    # is True and we mark the app servable; otherwise we launch a
    # standalone Bokeh server on the given port.
    import panel as pn

    app.servable(title=f"ballpushing diagnostics — {fly.metadata.name}")
    if pn.state.curdoc is None:  # standalone run, not `panel serve`
        pn.serve(app, port=args.port, show=True)


# When loaded by `panel serve`, the module is executed as `__main__`.
if __name__ == "__main__" or (
    # panel serve sets __name__ to 'bokeh_app_...' in some versions.
    __name__.startswith("bokeh_app")
):
    _main(sys.argv[1:] if __name__ == "__main__" else None)
