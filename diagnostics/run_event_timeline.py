#!/usr/bin/env python3
"""CLI entry point: per-fly interaction-event timeline diagnostic.

Usage:
    python diagnostics/run_event_timeline.py --fly /abs/path/to/fly
    python diagnostics/run_event_timeline.py --fly rel/path/under/data-root
    python diagnostics/run_event_timeline.py --experiment /abs/path/to/experiment
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ballpushing_utils import Experiment, Fly, dataset, figures_root, load_dotenv
from ballpushing_utils.diagnostics import build_event_timeline, write_report
from ballpushing_utils.diagnostics.event_timeline import build_event_timelines_for_experiment


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else dataset(path)


def _timestamped_dir(root: Path, subject: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in subject)
    return root / "diagnostics" / safe / stamp


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Per-fly interaction-event timeline diagnostic.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fly", help="Path to a fly directory (absolute, or relative to BALLPUSHING_DATA_ROOT).")
    group.add_argument("--experiment", help="Path to an experiment directory; one report per fly is emitted.")
    parser.add_argument("--out", help="Output root (default: $BALLPUSHING_FIGURES_ROOT).")
    parser.add_argument("--signif-threshold", type=float, default=5.0, help="px threshold for 'significant' events.")
    parser.add_argument("--major-threshold", type=float, default=20.0, help="px threshold for 'major' events.")
    args = parser.parse_args()

    out_root = Path(args.out) if args.out else figures_root()

    if args.fly:
        fly_path = _resolve(args.fly)
        fly = Fly(fly_path, as_individual=True)
        report = build_event_timeline(
            fly,
            signif_threshold=args.signif_threshold,
            major_threshold=args.major_threshold,
        )
        out_dir = _timestamped_dir(out_root, report.subject)
        summary = write_report(report, out_dir)
        print(f"Saved: {summary}")
    else:
        exp = Experiment(_resolve(args.experiment))
        for report in build_event_timelines_for_experiment(exp):
            out_dir = _timestamped_dir(out_root, report.subject)
            summary = write_report(report, out_dir)
            print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
