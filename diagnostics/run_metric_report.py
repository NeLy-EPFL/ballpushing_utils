#!/usr/bin/env python3
"""CLI entry point: per-fly metric summary diagnostic.

Usage:
    python diagnostics/run_metric_report.py --fly /abs/path/to/fly
    python diagnostics/run_metric_report.py --fly rel/path/under/data-root
    python diagnostics/run_metric_report.py --experiment /abs/path/to/experiment

Writes one report folder per fly, containing:

    summary.md      # human-readable overview (NaN / flagged counts per metric)
    summary.csv     # one row per metric with aggregate NaN / low / high counts
    flagged.csv     # only rows whose verdict is not ``ok``
    all_metrics.csv # full long table (one row per metric per ball)

Use this when you've changed the metric code and want to spot-check that
nothing regressed into NaN territory or blew past the declared plausible
range in :data:`ballpushing_utils.diagnostics.DEFAULT_METRIC_RANGES`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ballpushing_utils import Experiment, Fly, dataset, figures_root, load_dotenv
from ballpushing_utils.diagnostics import build_metric_report, write_report


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else dataset(path)


def _timestamped_dir(root: Path, subject: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in subject)
    return root / "diagnostics" / safe / stamp


def _run_one(fly: Fly, out_root: Path) -> Path:
    report = build_metric_report(fly)
    out_dir = _timestamped_dir(out_root, report.subject)
    return write_report(report, out_dir)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Per-fly metric summary diagnostic.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fly", help="Path to a fly directory (absolute, or relative to BALLPUSHING_DATA_ROOT).")
    group.add_argument("--experiment", help="Path to an experiment directory; one report per fly is emitted.")
    parser.add_argument("--out", help="Output root (default: $BALLPUSHING_FIGURES_ROOT).")
    args = parser.parse_args()

    out_root = Path(args.out) if args.out else figures_root()

    if args.fly:
        fly = Fly(_resolve(args.fly), as_individual=True)
        summary = _run_one(fly, out_root)
        print(f"Saved: {summary}")
    else:
        exp = Experiment(_resolve(args.experiment))
        for fly in exp.flies:
            summary = _run_one(fly, out_root)
            print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
