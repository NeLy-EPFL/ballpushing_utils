# `tests/integration/` review

Status as of the reorg branch (`tests/reorg-and-diagnostics`). This
directory contains 32 files (~11K LOC) that accumulated during
development. The table below classifies each file and proposes an
action so the integration suite can ship with the paper.

**Nothing in this review has been deleted or moved yet** — each action
requires your sign-off because the majority touch workstation-only data
paths and several are ad-hoc `print`-based scripts that happen to start
with `def test_*`, which pytest will silently collect.

## Headlines

- **16 files** are ad-hoc / debug / plotting scripts with no real
  assertions. They pass pytest trivially because the "test" functions
  print and return without raising. Safe to delete after a glance.
- **12 files** are real pytest tests but hard-code
  `/mnt/upramdya_data/...` paths. Candidates for a hermetic rewrite
  with stubbed `FlyTrackingData` / `Experiment` — same pattern we used
  for `tests/unit/diagnostics/`.
- **4 files** are already (or nearly) hermetic and can move to
  `tests/unit/` with minimal touch-ups. Three more are ~hermetic but
  are print-only "smoke" scripts; promoting them properly means
  replacing the prints with real `assert` statements.
- **Duplicated coverage**: `test_distance_metrics.py` vs
  `test_distance_metrics_f1.py` (both plotting, not tests);
  `time_range_exp.py` vs `time_ranges_test.py` (same Experiment
  parameter check from two angles);
  `test_dataset_*_integration.py` (three files, overlapping coverage).

## Classification

Columns: **kind** / **pytest?** / **data?** / **action**.

| File | kind | pytest? | data? | action | note |
|---|---|---|---|---|---|
| `ballpushing_metrics.py` | pytest+\_\_main\_\_ | yes | no | keep | 2.3K LOC comprehensive metrics benchmark; runs end-to-end |
| `behavior_umap.py` | ad-hoc script | no | no | delete | Passthrough print script, no asserts |
| `contact_annotated_interaction_events.py` | CLI | no | no | move to `tools/` | Generates MP4 via argparse |
| `contacts_grid.py` | CLI | no | no | move to `tools/` | MP4 grid generator |
| `debug_ball_coordinates.py` | debug | mixed | no | delete | `debug_` prefix; print-heavy diagnostics |
| `debug_missing_flies.py` | debug | mixed | yes | delete | `debug_` prefix; hard-coded F1 paths |
| `experiment_initialisation.py` | pytest fixtures | no | no | keep | Module-level experiment setup fixture |
| `f1_metrics_test.py` | pytest+\_\_main\_\_ | yes | no | hermetic rewrite | 1.6K LOC F1 metrics; stub tracking data |
| `fly_trackingdata_test.py` | pytest+\_\_main\_\_ | yes | yes | hermetic rewrite | Ball-identity assignment; stub `FlyTrackingData` |
| `freezing_detection.py` | script | no | yes | delete | Prints only, hard-coded data path |
| `interaction_based_metrics.py` | script | no | no | delete | JSON print helper, no asserts |
| `test_contact_standardized_events.py` | pytest+\_\_main\_\_ | yes | yes | hermetic rewrite | SkeletonMetrics; stub `Experiment` |
| `test_dataset_builder_contacts.py` | pytest+\_\_main\_\_ | yes | yes | hermetic rewrite | Dataset-with-contacts surface |
| `test_dataset_missing_flies.py` | pytest+\_\_main\_\_ | yes | yes | delete | Debug workflow, superseded by other tests |
| `test_dataset_summary_integration.py` | pytest+\_\_main\_\_ | yes | no | hermetic rewrite | Introspection checks on summary mode |
| `test_distance_metrics.py` | plotting | no | yes | delete / move | Produces PNGs, not a test |
| `test_distance_metrics_f1.py` | plotting | no | yes | delete / move | Duplicate of the above for F1 |
| `test_f1_adjusted_time_metrics.py` | pytest+\_\_main\_\_ | yes | yes | delete | 1.1K LOC but debug-flavoured and non-reproducible |
| `test_f1_adjusted_times.py` | pytest+\_\_main\_\_ | yes | no | move to unit | Small, hermetic F1 mock test |
| `test_f1_premature_exit_logic.py` | pytest+\_\_main\_\_ | yes | yes | delete | Manual test with hard-coded paths |
| `test_fly_centered_raw_ball.py` | pytest+\_\_main\_\_ | yes | yes | hermetic rewrite | Coordinate transform; stub `Experiment` |
| `test_mannwhitney_plots.py` | plotting | no | yes | delete / move | PNG generator, not a test |
| `test_nan_handling.py` | print-smoke | no | no | promote to unit | Hermetic but needs real `assert`s |
| `test_raw_ball_coordinates.py` | pytest+\_\_main\_\_ | yes | yes | hermetic rewrite | Raw ball coords in standardized contacts |
| `test_spontaneous_ball_movement.py` | pytest+\_\_main\_\_ | yes | yes | hermetic rewrite | Spontaneous-movement detection |
| `test_standardized_events_comparison.py` | script | no | yes | delete | Print-only mode comparison |
| `test_summary_integration.py` | print-smoke | yes | no | promote to unit | `def test_*` with no asserts; add `assert` & move |
| `test_transformation_logic.py` | print-smoke | yes | no | promote to unit | `def test_*` with no asserts; add `assert` & move |
| `time_range_exp.py` | pytest+\_\_main\_\_ | yes | no | move to unit | Experiment time-range parameter |
| `time_ranges_test.py` | pytest+\_\_main\_\_ | yes | no | move to unit | Fly time-range parameter; partial duplicate |
| `transformed_dataset.py` | script | no | no | delete | Pass-through, no asserts |
| `transposed_data.py` | script | no | no | delete | Pass-through, no asserts |

## Proposed plan

In order of increasing risk / effort:

1. **Delete the 16 ad-hoc / debug / plotting-only files.** None have
   real assertions, and several claim to be tests but are actually
   print diagnostics. A single commit titled *"Prune ad-hoc scripts
   from tests/integration/"* is enough.
2. **Promote the 3 print-smoke pseudo-tests** (`test_nan_handling.py`,
   `test_summary_integration.py`, `test_transformation_logic.py`) to
   `tests/unit/` after replacing their `print("✓ ...")` lines with
   `assert` statements. They already don't touch data.
3. **Move the 2 CLI video generators** (`contact_annotated_interaction_events.py`,
   `contacts_grid.py`) to `tools/` so they're still runnable but no
   longer collected by pytest.
4. **Move the 3 small hermetic pytest files** (`test_f1_adjusted_times.py`,
   `time_range_exp.py`, `time_ranges_test.py`) into `tests/unit/` as-is.
5. **Hermetic rewrite** of the 9 remaining `pytest+data` files, using
   the stub-`Fly` / stub-`Experiment` pattern already established in
   `tests/unit/diagnostics/`.
6. **Keep** `ballpushing_metrics.py` and `experiment_initialisation.py`
   under `tests/integration/` as data-dependent end-to-end smoke
   tests, explicitly gated (skipped when `BALLPUSHING_DATA_ROOT` is
   unset or paths don't exist).

Steps 1–4 together take out ~70% of the directory and require no data
access; steps 5–6 need workstation-SSH work later.
