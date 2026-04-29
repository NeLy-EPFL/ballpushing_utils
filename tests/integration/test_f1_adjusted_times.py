#!/usr/bin/env python3
"""
Test script to verify that F1 test ball metrics use adjusted times properly.
This script creates a mock F1 experiment setup and validates that time-based metrics
are correctly adjusted by subtracting the corridor exit time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ballpushing_utils import Fly, Experiment, BallPushingMetrics


def create_mock_f1_tracking_data():
    """Create mock tracking data for F1 experiment testing."""

    class MockConfig:
        def __init__(self):
            self.experiment_type = "F1"
            self.debugging = True
            self.final_event_threshold = 300
            self.final_event_F1_threshold = 150
            self.major_event_threshold = 30
            self.max_event_threshold = 10
            self.significant_threshold = 5
            self.chamber_radius = 50
            self.dead_threshold = 30
            self.enabled_metrics = None

    class MockExperiment:
        def __init__(self):
            self.fps = 30

    class MockMetadata:
        def __init__(self):
            self.name = "test_f1_fly"
            self.F1_condition = "test"

    class MockFly:
        def __init__(self):
            self.config = MockConfig()
            self.experiment = MockExperiment()
            self.metadata = MockMetadata()

    class MockTrackingObject:
        def __init__(self, data):
            self.dataset = data

    class MockSleapTracks:
        def __init__(self, objects, track_names=None):
            self.objects = objects
            self.track_names = track_names if track_names else []

    class MockTrackingData:
        def __init__(self):
            self.fly = MockFly()

            # Create mock fly data - fly starts at (100, 100) and moves to (200, 100) at frame 900 (30s)
            n_frames = 3000  # 100 seconds at 30 fps
            fly_data = pd.DataFrame(
                {
                    "time": np.arange(n_frames),
                    "x_thorax": np.concatenate(
                        [
                            np.full(900, 100),  # Stay at x=100 for 30 seconds
                            np.linspace(100, 200, n_frames - 900),  # Move to x=200 after 30s
                        ]
                    ),
                    "y_thorax": np.full(n_frames, 100),
                }
            )

            # Create mock ball data for training ball (first corridor)
            training_ball_data = pd.DataFrame(
                {
                    "time": np.arange(n_frames),
                    "x_centre": np.full(n_frames, 120),  # Training ball near start
                    "y_centre": np.concatenate(
                        [
                            np.full(600, 100),  # No movement first 20s
                            np.linspace(100, 200, 300),  # Move 20-30s
                            np.full(n_frames - 900, 200),  # Stay moved after 30s
                        ]
                    ),
                }
            )

            # Create mock ball data for test ball (second corridor)
            test_ball_data = pd.DataFrame(
                {
                    "time": np.arange(n_frames),
                    "x_centre": np.full(n_frames, 300),  # Test ball far from start
                    "y_centre": np.concatenate(
                        [
                            np.full(900, 100),  # No movement for first 30s (before corridor exit)
                            np.linspace(100, 150, 600),  # Move after corridor exit (30-50s)
                            np.full(n_frames - 1500, 150),  # Stay moved after 50s
                        ]
                    ),
                }
            )

            self.flytrack = MockSleapTracks([MockTrackingObject(fly_data)])
            self.balltrack = MockSleapTracks(
                [MockTrackingObject(training_ball_data), MockTrackingObject(test_ball_data)],
                track_names=["training_ball", "test_ball"],
            )
            self.skeletontrack = None

            # Set up F1-specific attributes
            self.start_x = 100
            self.start_y = 100
            self.f1_exit_time = 30.0  # Fly exits corridor to second arena at 30 seconds
            self.duration = n_frames / 30  # Total duration in seconds
            self.chamber_exit_times = {0: 3.0}  # Fly exits initial chamber at 3 seconds

            # Real TrackingData sets ``chamber_exit_time`` (singular) in
            # __init__ — see fly_trackingdata.py. BallPushingMetrics' debug
            # branch reads it directly, so the mock needs it too. Keep the
            # plural dict above since other code paths still read it.
            self.chamber_exit_time = self.chamber_exit_times[0]

            # Ball identities for F1 experiment
            self.ball_identities = {0: "training", 1: "test"}
            self.identity_to_idx = {"training": 0, "test": 1}
            self.training_ball_idx = 0
            self.test_ball_idx = 1

            # Create mock interaction events
            # Training ball events (0-30s, before corridor exit)
            training_events = [
                (600, 750, 5.0),  # Event at 20-25s
                (750, 900, 5.0),  # Event at 25-30s
            ]

            # Test ball events (after 30s corridor exit)
            test_events = [
                (1200, 1350, 5.0),  # Event at 40-45s (10-15s adjusted time)
                (1500, 1650, 5.0),  # Event at 50-55s (20-25s adjusted time)
            ]

            self.interaction_events = {0: {0: training_events, 1: test_events}}

    return MockTrackingData()


def test_f1_adjusted_times():
    """Test that F1 test ball metrics use adjusted times correctly."""

    tracking_data = create_mock_f1_tracking_data()
    metrics = BallPushingMetrics(tracking_data, compute_metrics_on_init=False)

    # Which balls the system decides to adjust.
    should_adjust_training = metrics._should_use_adjusted_time(0, 0)
    should_adjust_test = metrics._should_use_adjusted_time(0, 1)

    assert should_adjust_test, "test ball must use F1-adjusted time"
    assert not should_adjust_training, "training ball must NOT use F1-adjusted time"

    # Test ball times are raw_time - f1_exit_time; training ball times pass through.
    raw_time = 45.0
    adjusted_training = metrics._adjust_time_for_f1_test_ball(raw_time, 0, 0)
    adjusted_test = metrics._adjust_time_for_f1_test_ball(raw_time, 0, 1)

    assert adjusted_training == raw_time, (
        f"training-ball time should pass through unchanged, got {adjusted_training}s"
    )
    expected_test = raw_time - tracking_data.f1_exit_time  # 45 - 30 == 15
    assert abs(adjusted_test - expected_test) < 0.1, (
        f"test-ball time should be raw - f1_exit_time = {expected_test}s, got {adjusted_test}s"
    )

    # Smoke-check the time-based metric helpers still execute end-to-end
    # with both ball identities — regressions here are usually a KeyError or
    # a wrong ball_idx lookup, not a numeric drift, so the smoke is what we
    # want to catch.
    assert metrics.get_max_event(0, 0) is not None
    assert metrics.get_max_event(0, 1) is not None
    assert metrics.get_final_event(0, 0) is not None
    assert metrics.get_final_event(0, 1) is not None
    assert metrics.get_major_event(0, 0) is not None
    assert metrics.get_major_event(0, 1) is not None
    assert metrics.get_first_significant_event(0, 0) is not None
    assert metrics.get_first_significant_event(0, 1) is not None
