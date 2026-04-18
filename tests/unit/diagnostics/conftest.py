"""Stub Fly fixtures for the diagnostics unit tests.

The diagnostic builders only touch a small surface of :class:`Fly` —
``experiment.fps``, ``tracking_data.interaction_events``,
``event_summaries``, plus a name attribute. We mimic that surface with
plain dataclasses / SimpleNamespace so the tests run hermetically (no
SLEAP, no video, no disk I/O) and can be wired into CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest


@dataclass
class _StubExperiment:
    fps: float = 30.0


@dataclass
class _StubTrackingData:
    # Nested dict matching ``Fly.tracking_data.interaction_events``:
    # {fly_idx: {ball_idx: [[start_frame, end_frame, displacement_px], ...]}}
    interaction_events: dict[int, dict[int, list[list[float]]]] | None = None


@dataclass
class _StubFly:
    experiment: _StubExperiment = field(default_factory=_StubExperiment)
    tracking_data: _StubTrackingData = field(default_factory=_StubTrackingData)
    event_summaries: dict[str, dict[str, Any]] | None = None
    metadata: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(name="stub_fly"))
    directory: str = "/tmp/stub_fly"


@pytest.fixture
def stub_events_fly() -> _StubFly:
    """Fly stub with a small but realistic interaction-events table."""
    fly = _StubFly()
    fly.tracking_data.interaction_events = {
        0: {
            0: [
                # [start_frame, end_frame, displacement_px]
                [30, 90, 2.5],     # minor   (< signif_threshold=5.0)
                [300, 360, 12.0],  # significant
                [900, 1020, 35.0], # major   (>= major_threshold=20.0)
            ],
            1: [
                [600, 660, 8.0],  # significant
            ],
        }
    }
    return fly


@pytest.fixture
def stub_empty_events_fly() -> _StubFly:
    """Fly stub with no interaction events."""
    fly = _StubFly()
    fly.tracking_data.interaction_events = {}
    return fly


@pytest.fixture
def stub_metrics_fly() -> _StubFly:
    """Fly stub with a small but realistic ``event_summaries`` dict.

    Mixes in-range metrics, an out-of-range one, a NaN, and a metric not
    declared in :data:`DEFAULT_METRIC_RANGES` (which should still appear
    with verdict ``ok``).
    """
    fly = _StubFly()
    fly.event_summaries = {
        "fly_0_ball_0": {
            "fly_idx": 0,
            "ball_idx": 0,
            "nb_events": 12,                  # ok
            "significant_ratio": 0.4,         # ok
            "max_distance": 250.0,            # ok
            "chamber_ratio": 1.5,             # high (>1.0)
            "median_pause_duration": float("nan"),  # nan
            "made_up_metric": 7.0,            # not in DEFAULT_METRIC_RANGES → ok
        },
        "fly_0_ball_1": {
            "fly_idx": 0,
            "ball_idx": 1,
            "nb_events": -3,                  # low (<0)
            "significant_ratio": 0.0,         # ok
        },
    }
    return fly


@pytest.fixture
def stub_empty_metrics_fly() -> _StubFly:
    """Fly stub with no event_summaries dict."""
    fly = _StubFly()
    fly.event_summaries = {}
    return fly
