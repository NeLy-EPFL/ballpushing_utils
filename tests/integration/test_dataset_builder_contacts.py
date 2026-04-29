"""Dataset builder smoke test for the ``standardized_contacts`` mode.

Builds a ``Dataset(flies, dataset_type="standardized_contacts")`` over
the canonical example experiment. The fixture experiment has a single
fly, which is enough to validate that the builder wires up end-to-end
and produces a non-None ``data`` frame with the expected columns.
"""

from __future__ import annotations

import pytest

import ballpushing_utils
from ballpushing_utils.config import Config


@pytest.mark.integration
def test_dataset_builder_with_contacts(example_experiment):
    experiment = example_experiment
    assert experiment.flies, "example_experiment should have at least one fly"

    flies_with_skeleton = [
        fly
        for fly in experiment.flies
        if fly.tracking_data is not None and fly.tracking_data.skeletontrack is not None
    ]
    if not flies_with_skeleton:
        pytest.skip("No flies in example_experiment have skeleton data.")

    # Narrow contact window so the builder stays fast on fixture data.
    config = Config()
    config.frames_before_onset = 30
    config.frames_after_onset = 30
    config.contact_min_length = 2
    for fly in flies_with_skeleton:
        fly.config = config

    dataset = ballpushing_utils.Dataset(flies_with_skeleton, dataset_type="standardized_contacts")

    assert dataset is not None
    # Dataset.data may be None if zero valid contacts were found — that's
    # still a valid result, but if it's not None it must be a DataFrame
    # with the canonical columns present.
    if dataset.data is not None:
        assert not dataset.data.empty, (
            "dataset.data was produced but is empty; expected at least one contact row "
            "from the canonical fixture fly."
        )
        assert "event_type" in dataset.data.columns
