"""Contact-based standardized events smoke test.

Exercises the SkeletonMetrics contact pipeline on the canonical example
fly: contact-annotated dataset, contact period detection, and the
standardized contact events DataFrame. Skips cleanly if the fixture
doesn't have skeleton tracking (the only way to compute contacts).
"""

from __future__ import annotations

import pytest

import ballpushing_utils
from ballpushing_utils.config import Config


@pytest.mark.integration
def test_contact_standardized_events(example_fly):
    fly = example_fly
    if fly.tracking_data is None or fly.tracking_data.skeletontrack is None:
        pytest.skip("example_fly has no skeleton track; SkeletonMetrics can't run.")

    # Tighter window than the default so this stays fast on the fixture.
    config = Config()
    config.frames_before_onset = 30
    config.frames_after_onset = 30
    config.contact_min_length = 2
    fly.config = config

    skeleton_metrics = ballpushing_utils.SkeletonMetrics(fly)

    annotated_df = skeleton_metrics.get_contact_annotated_dataset()
    assert annotated_df is not None
    assert not annotated_df.empty, "contact-annotated dataset should not be empty"
    assert "is_contact" in annotated_df.columns, (
        "contact-annotated dataset is missing the is_contact column; the "
        "contact detector output shape has drifted."
    )

    # ``_find_contact_periods`` should return a list (possibly empty) of
    # (start, end) indices into the annotated frame.
    contact_periods = skeleton_metrics._find_contact_periods(annotated_df)
    assert isinstance(contact_periods, list)

    # The events-based contacts DataFrame should exist; it may be empty
    # on the fixture if no valid contacts are detected, but it must be a
    # DataFrame with the ``event_type`` column when non-empty.
    events_df = skeleton_metrics.events_based_contacts
    assert events_df is not None
    if not events_df.empty:
        assert "event_type" in events_df.columns
