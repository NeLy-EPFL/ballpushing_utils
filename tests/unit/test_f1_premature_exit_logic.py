"""Pure-logic tests for the F1 premature-exit threshold.

Extracted from the former ``tests/integration/test_f1_premature_exit_logic.py``
CLI dev script (now at ``tools/dev/test_f1_premature_exit_logic.py``).
The data-backed parts of that script need the lab share, but the
55-minute threshold decision itself is a self-contained rule that belongs
in the unit suite.

Contract: a fly is flagged as a premature exit iff ``exit_time_seconds``
is not ``None`` **and** strictly less than 55 minutes. A fly that never
exits (``exit_time_seconds is None``) is kept.
"""

from __future__ import annotations

import pytest


PREMATURE_EXIT_THRESHOLD_SECONDS = 55 * 60  # 55 minutes


def _should_discard(exit_time_minutes: float | None) -> bool:
    """Replicate the threshold decision from the F1 exit-detection code."""
    if exit_time_minutes is None:
        return False
    return exit_time_minutes * 60 < PREMATURE_EXIT_THRESHOLD_SECONDS


@pytest.mark.parametrize(
    "exit_time_minutes, expected_discard",
    [
        pytest.param(None, False, id="fly_never_exits"),
        pytest.param(30, True, id="well_below_threshold"),
        pytest.param(45, True, id="below_threshold"),
        pytest.param(54.9, True, id="just_below_threshold"),
        pytest.param(55, False, id="exactly_at_threshold"),
        pytest.param(60, False, id="above_threshold"),
        pytest.param(120, False, id="well_above_threshold"),
    ],
)
def test_f1_premature_exit_threshold(exit_time_minutes, expected_discard):
    assert _should_discard(exit_time_minutes) is expected_discard
