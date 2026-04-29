"""Raw ball coordinates are included/excluded based on ``fly_only`` config.

With ``fly_only=False`` the standardized-contacts events DataFrame must
include raw ball coordinate columns (``*centre_raw*``). With
``fly_only=True`` it must not. Skips cleanly on fixtures without
skeleton data.
"""

from __future__ import annotations

import pytest

import ballpushing_utils
from ballpushing_utils.config import Config


def _make_config(fly_only: bool) -> Config:
    config = Config()
    config.standardized_events_mode = "contact_events"
    config.fly_only = fly_only
    config.frames_before_onset = 30
    config.frames_after_onset = 30
    config.generate_random = False
    return config


@pytest.mark.integration
def test_raw_ball_coordinates_included_when_fly_only_false(example_fly):
    fly = example_fly
    if fly.tracking_data is None or fly.tracking_data.skeletontrack is None:
        pytest.skip("example_fly has no skeleton track; SkeletonMetrics can't run.")

    fly.config = _make_config(fly_only=False)
    events_df = ballpushing_utils.SkeletonMetrics(fly).events_based_contacts

    raw_cols = [c for c in events_df.columns if "centre_raw" in c]
    assert raw_cols, (
        f"expected raw ball coordinate columns (*centre_raw*) with fly_only=False, "
        f"got {list(events_df.columns)!r}"
    )


@pytest.mark.integration
def test_raw_ball_coordinates_excluded_when_fly_only_true(example_fly):
    fly = example_fly
    if fly.tracking_data is None or fly.tracking_data.skeletontrack is None:
        pytest.skip("example_fly has no skeleton track; SkeletonMetrics can't run.")

    fly.config = _make_config(fly_only=True)
    events_df = ballpushing_utils.SkeletonMetrics(fly).events_based_contacts

    raw_cols = [c for c in events_df.columns if "centre_raw" in c]
    assert not raw_cols, (
        f"did not expect raw ball coordinate columns with fly_only=True, "
        f"got {raw_cols!r}"
    )
