"""Pure-logic tests for the ball-assignment algorithm.

Extracted from the former ``tests/integration/fly_trackingdata_test.py``
CLI dev script. These helpers mirror the classification logic inside
:class:`ballpushing_utils.FlyTrackingData`, but operate on raw track-name
lists so the test has no data dependency and runs offline in the unit
suite.

If the real assignment logic in ``FlyTrackingData`` ever diverges from
the simulators below, the fix is to update the simulators (and this
test) so they stay in lockstep with the source. The parametrised cases
are the intended behavioural spec.
"""

from __future__ import annotations

import pytest


def simulate_ball_assignment(track_names: list[str]):
    """Reproduce the ball-role assignment that :class:`FlyTrackingData` uses."""
    training_ball_idx: int | None = None
    test_ball_idx: int | None = None
    ball_identities: dict[int, str] = {}
    identity_to_idx: dict[str, int] = {}

    num_balls = len(track_names)

    for ball_idx, track_name in enumerate(track_names):
        track_name_lower = track_name.lower()

        if track_name_lower in ["training", "train", "training_ball"]:
            identity = "training"
            training_ball_idx = ball_idx
            identity_to_idx["training"] = ball_idx
        elif track_name_lower in ["test", "testing", "test_ball"]:
            identity = "test"
            test_ball_idx = ball_idx
            identity_to_idx["test"] = ball_idx
        else:
            identity = track_name_lower

            # F1-style experiments with two generic track names: map
            # first → training, second → test.
            if num_balls == 2:
                if ball_idx == 0 and training_ball_idx is None:
                    training_ball_idx = ball_idx
                    identity_to_idx["training"] = ball_idx
                elif ball_idx == 1 and test_ball_idx is None:
                    test_ball_idx = ball_idx
                    identity_to_idx["test"] = ball_idx
            else:
                if training_ball_idx is None and ball_idx == 0:
                    training_ball_idx = ball_idx
                    identity_to_idx["training"] = ball_idx

        ball_identities[ball_idx] = identity

    has_explicit_test_naming = any(
        name.lower() in ["test", "testing", "test_ball"] for name in track_names
    )
    has_explicit_training_naming = any(
        name.lower() in ["training", "train", "training_ball"] for name in track_names
    )

    # Fallback for regular experiments: if nothing was explicitly named,
    # promote the first ball to "training" so downstream code can always
    # find a reference ball.
    if (
        training_ball_idx is None
        and num_balls > 0
        and not has_explicit_test_naming
        and not has_explicit_training_naming
    ):
        training_ball_idx = 0
        if "training" not in identity_to_idx:
            identity_to_idx["training"] = 0

    return training_ball_idx, test_ball_idx, ball_identities, identity_to_idx


def simulate_success_cutoff_logic(training_ball_idx, test_ball_idx):
    """Pick the ball used for the success cutoff (test ball wins)."""
    if test_ball_idx is not None:
        return test_ball_idx, "test"
    if training_ball_idx is not None:
        return training_ball_idx, "training"
    return 0, "fallback"


# Cases kept identical to the original dev script so we don't silently
# regress the behavioural spec. ``expected_cutoff`` is new — it locks in
# the cutoff-selection contract alongside the role assignment.
_ASSIGNMENT_CASES = [
    pytest.param(
        ["training_ball", "test_ball"], 0, 1, (1, "test"),
        id="f1_explicit_names",
    ),
    pytest.param(
        ["test_ball"], None, 0, (0, "test"),
        id="control_test_only",
    ),
    pytest.param(
        ["training", "test"], 0, 1, (1, "test"),
        id="legacy_training_test_names",
    ),
    pytest.param(
        ["track_1", "track_2"], 0, 1, (1, "test"),
        id="f1_generic_names",
    ),
    pytest.param(
        ["ball"], 0, None, (0, "training"),
        id="single_ball_regular",
    ),
]


@pytest.mark.parametrize(
    ("track_names", "expected_training", "expected_test", "expected_cutoff"),
    _ASSIGNMENT_CASES,
)
def test_ball_assignment_logic(track_names, expected_training, expected_test, expected_cutoff):
    training_idx, test_idx, _identities, _identity_to_idx = simulate_ball_assignment(track_names)
    cutoff = simulate_success_cutoff_logic(training_idx, test_idx)

    assert training_idx == expected_training, (
        f"training_ball_idx for {track_names!r}: expected {expected_training}, got {training_idx}"
    )
    assert test_idx == expected_test, (
        f"test_ball_idx for {track_names!r}: expected {expected_test}, got {test_idx}"
    )
    assert cutoff == expected_cutoff, (
        f"success cutoff for {track_names!r}: expected {expected_cutoff}, got {cutoff}"
    )
