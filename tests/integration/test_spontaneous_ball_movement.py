"""Spontaneous ball movement detection against the bundled fixtures.

The detector flags frames where the ball moves between interaction
events (i.e. not during fly-induced pushing). The summary for each
ball must expose the canonical key set; values depend on the recording.

These tests do not assert a specific outcome — they just exercise the
pipeline end-to-end and check the returned summary is well-formed so
regressions like the ``too many values to unpack`` bug (fixed in the
preceding commit) can't sneak back in.
"""

from __future__ import annotations

import pytest

from ballpushing_utils import Experiment, Fly
from ballpushing_utils.config import Config


_REQUIRED_SUMMARY_KEYS = {
    "num_spontaneous_frames",
    "total_displacement",
    "max_displacement",
    "max_consecutive_frames",
    "spontaneous_movement_times",
}


def _spontaneous_config(experiment_type: str | None = None) -> Config:
    config = Config()
    if experiment_type is not None:
        config.experiment_type = experiment_type
    config.check_spontaneous_ball_movement = True
    config.spontaneous_movement_threshold = 10.0
    config.debugging = False
    return config


def _assert_summary_shape(summary, label: str) -> None:
    assert summary is not None, f"{label}: spontaneous movement summary is None"
    missing = _REQUIRED_SUMMARY_KEYS - summary.keys()
    assert not missing, f"{label}: summary missing keys {sorted(missing)!r}"


@pytest.mark.integration
def test_spontaneous_movement_on_example_fly(example_fly_path):
    """Regular (non-F1) paradigm — canonical fly."""
    fly = Fly(example_fly_path, as_individual=True, custom_config=_spontaneous_config())
    assert fly.tracking_data is not None and fly.tracking_data.valid_data, (
        "canonical example fly should load with valid tracking data"
    )

    num_balls = len(fly.tracking_data.balltrack.objects) if fly.tracking_data.balltrack else 0
    assert num_balls >= 1, "example fly should have at least one tracked ball"

    for ball_idx in range(num_balls):
        summary = fly.tracking_data.get_spontaneous_movement_summary(ball_idx)
        _assert_summary_shape(summary, f"example_fly ball {ball_idx}")


@pytest.mark.integration
def test_spontaneous_movement_on_example_experiment(example_experiment_path):
    """Experiment-level: loads once, walks every fly that loaded cleanly."""
    experiment = Experiment(
        example_experiment_path, custom_config=_spontaneous_config()
    )
    assert experiment.flies, (
        "example_experiment should produce at least one fly when loaded with "
        "spontaneous-movement config"
    )

    for fly in experiment.flies:
        if fly.tracking_data is None or not fly.tracking_data.valid_data:
            continue
        num_balls = (
            len(fly.tracking_data.balltrack.objects) if fly.tracking_data.balltrack else 0
        )
        for ball_idx in range(num_balls):
            summary = fly.tracking_data.get_spontaneous_movement_summary(ball_idx)
            _assert_summary_shape(summary, f"{fly.metadata.name} ball {ball_idx}")


@pytest.mark.integration
def test_spontaneous_movement_on_f1_fly(example_f1_fly_path):
    """F1 paradigm smoke check — same contract, F1 experiment config."""
    fly = Fly(
        example_f1_fly_path,
        as_individual=True,
        custom_config=_spontaneous_config(experiment_type="F1"),
    )
    assert fly.tracking_data is not None and fly.tracking_data.valid_data, (
        "canonical F1 fly should load with valid tracking data"
    )

    num_balls = len(fly.tracking_data.balltrack.objects) if fly.tracking_data.balltrack else 0
    assert num_balls >= 1, "F1 fixture fly should expose at least one tracked ball"

    for ball_idx in range(num_balls):
        summary = fly.tracking_data.get_spontaneous_movement_summary(ball_idx)
        _assert_summary_shape(summary, f"f1_fly ball {ball_idx}")
