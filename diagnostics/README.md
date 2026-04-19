# diagnostics/

Human-inspection harness for the ball-pushing pipeline. The scripts here
load a real :class:`~ballpushing_utils.Fly` (or
:class:`~ballpushing_utils.Experiment`) and emit per-run report folders
containing CSVs, PNG plots, and a Markdown summary that you can open
side-by-side with the underlying video to validate the pipeline.

This is a **separate concern** from the pytest suite under `tests/`. The
pytest suite asserts hermetic invariants (fast, no SLEAP data needed,
runs in CI). The diagnostics here need real recordings and are designed
for human review, not pass/fail.

## Layout

```
diagnostics/
├── README.md                  # this file
├── run_event_timeline.py      # per-fly interaction-event table + Gantt plot
└── run_metric_report.py       # per-fly metric summary with NaN / outlier flags

src/ballpushing_utils/diagnostics/
├── __init__.py
├── report.py                  # DiagnosticReport / DiagnosticSection / write_report
├── event_timeline.py          # build_event_timeline
└── metric_report.py           # build_metric_report + DEFAULT_METRIC_RANGES

tests/unit/diagnostics/        # hermetic unit tests for the builders
```

The Python API lives at `src/ballpushing_utils/diagnostics/` so the same
builders can be re-used from notebooks or a panel dashboard (planned
follow-up).

## Quickstart

```bash
# Inspect events for one fly (lab share)
python diagnostics/run_event_timeline.py \
    --fly /mnt/upramdya_data/MD/MultiMazeRecorder/Videos/.../arena2/corridor5

# Same fly — but spot-check the metric values instead.
python diagnostics/run_metric_report.py \
    --fly /mnt/upramdya_data/MD/MultiMazeRecorder/Videos/.../arena2/corridor5

# External users: relative to BALLPUSHING_DATA_ROOT
export BALLPUSHING_DATA_ROOT=/path/to/dataverse/sample
python diagnostics/run_event_timeline.py \
    --fly MultiMazeRecorder/Videos/.../arena2/corridor5
python diagnostics/run_metric_report.py \
    --fly MultiMazeRecorder/Videos/.../arena2/corridor5

# Run every fly in an experiment folder (one report per fly).
python diagnostics/run_event_timeline.py --experiment /path/to/experiment
python diagnostics/run_metric_report.py  --experiment /path/to/experiment
```

Each invocation writes an output folder under
`$BALLPUSHING_DATA_ROOT/diagnostics/<fly-name>/<timestamp>/` (override
with `--out`). Inside you'll find:

```
summary.md       # human-readable overview with embedded plots + table previews
events.csv       # full per-event table (run_event_timeline)
metrics.csv      # full long table (run_metric_report)
plots/           # supporting PNG plots
```

The Markdown summary is the recommended entry point: open it in any
viewer and click through to the CSVs / PNGs that interest you.

## Programmatic use

```python
from ballpushing_utils import Fly
from ballpushing_utils.diagnostics import (
    build_event_timeline, build_metric_report, write_report,
)

fly = Fly("/path/to/fly")

events_report  = build_event_timeline(fly)
metrics_report = build_metric_report(fly)

write_report(events_report,  "out/events")
write_report(metrics_report, "out/metrics")
```

Both builders return a `DiagnosticReport` (data structure only — no disk
I/O), so they're trivial to wrap in a notebook or dashboard.
