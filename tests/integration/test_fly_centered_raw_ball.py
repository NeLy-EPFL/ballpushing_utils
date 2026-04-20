"""Raw ball coordinates are transformed into the fly-centered frame.

Verifies that ``SkeletonMetrics.fly_centered_tracks`` contains the
fly-centered variants of the raw ball coordinates
(``*centre_raw_fly*``) when ``fly_only=False``, and that they also show
up in the standardized events DataFrame. Skips cleanly on fixtures
without skeleton data.
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
def test_fly_centered_raw_ball_coordinates_present(example_fly):
    fly = example_fly
    if fly.tracking_data is None or fly.tracking_data.skeletontrack is None:
        pytest.skip("example_fly has no skeleton track; SkeletonMetrics can't run.")

    fly.config = _make_config(fly_only=False)
    skeleton_metrics = ballpushing_utils.SkeletonMetrics(fly)
    fly_centered = skeleton_metrics.fly_centered_tracks

    centered_raw_cols = [c for c in fly_centered.columns if "centre_raw_fly" in c]
    assert centered_raw_cols, (
        f"expected fly-centered raw ball columns (*centre_raw_fly*) in "
        f"fly_centered_tracks, got {list(fly_centered.columns)!r}"
    )

    # The transformed columns should also show up on the events DataFrame
    # so downstream consumers see consistent fields.
    events_df = skeleton_metrics.events_based_contacts
    event_centered_raw = [c for c in events_df.columns if "centre_raw_fly" in c]
    assert event_centered_raw, (
        f"expected fly-centered raw ball columns on events_based_contacts, "
        f"got {list(events_df.columns)!r}"
    )


@pytest.mark.integration
def test_fly_centered_tracks_excludes_ball_when_fly_only_true(example_fly):
    fly = example_fly
    if fly.tracking_data is None or fly.tracking_data.skeletontrack is None:
        pytest.skip("example_fly has no skeleton track; SkeletonMetrics can't run.")

    fly.config = _make_config(fly_only=True)
    fly_centered = ballpushing_utils.SkeletonMetrics(fly).fly_centered_tracks

    # Ball coordinate columns show up as *centre_raw* / *centre_preprocessed*;
    # a plain "centre" substring also matches fly-body parts (e.g. thorax
    # centre), so filter on the ball-specific patterns only.
    ball_cols = [
        c
        for c in fly_centered.columns
        if "centre_raw" in c or "centre_preprocessed" in c
    ]
    assert not ball_cols, (
        f"did not expect any ball-coordinate columns in fly_centered_tracks with "
        f"fly_only=True, got {ball_cols!r}"
    )
