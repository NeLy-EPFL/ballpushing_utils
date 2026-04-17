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

from Ballpushing_utils import Fly, Experiment, BallPushingMetrics


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

    print("🧪 Testing F1 Adjusted Times Implementation")
    print("=" * 60)

    # Create mock tracking data
    tracking_data = create_mock_f1_tracking_data()

    # Create metrics object
    metrics = BallPushingMetrics(tracking_data, compute_metrics_on_init=False)

    print(f"F1 exit time: {tracking_data.f1_exit_time}s")
    print(f"Chamber exit time: {tracking_data.chamber_exit_times[0]}s")
    print(f"Ball identities: {tracking_data.ball_identities}")

    # Test helper methods
    print("\n🔍 Testing Helper Methods:")
    print("-" * 40)

    # Test _should_use_adjusted_time
    should_adjust_training = metrics._should_use_adjusted_time(0, 0)  # Training ball
    should_adjust_test = metrics._should_use_adjusted_time(0, 1)  # Test ball

    print(f"Should adjust training ball: {should_adjust_training}")
    print(f"Should adjust test ball: {should_adjust_test}")

    # Test _adjust_time_for_f1_test_ball
    raw_time = 45.0  # 45 seconds from experiment start
    adjusted_training = metrics._adjust_time_for_f1_test_ball(raw_time, 0, 0)  # Training ball
    adjusted_test = metrics._adjust_time_for_f1_test_ball(raw_time, 0, 1)  # Test ball

    print(f"Raw time: {raw_time}s")
    print(f"Adjusted training ball time: {adjusted_training}s")
    print(f"Adjusted test ball time: {adjusted_test}s (should be {raw_time - tracking_data.f1_exit_time}s)")

    # Test individual time-based metrics
    print("\n📊 Testing Time-Based Metrics:")
    print("-" * 40)

    # Test get_max_event
    max_event_training = metrics.get_max_event(0, 0)
    max_event_test = metrics.get_max_event(0, 1)

    print(f"Max event training ball: idx={max_event_training[0]}, time={max_event_training[1]}s")
    print(f"Max event test ball: idx={max_event_test[0]}, time={max_event_test[1]}s")

    # Test get_final_event
    final_event_training = metrics.get_final_event(0, 0)
    final_event_test = metrics.get_final_event(0, 1)

    print(f"Final event training ball: {final_event_training}")
    print(f"Final event test ball: {final_event_test}")

    # Test get_major_event
    major_event_training = metrics.get_major_event(0, 0)
    major_event_test = metrics.get_major_event(0, 1)

    print(f"Major event training ball: idx={major_event_training[0]}, time={major_event_training[1]}s")
    print(f"Major event test ball: idx={major_event_test[0]}, time={major_event_test[1]}s")

    # Test get_first_significant_event
    first_sig_training = metrics.get_first_significant_event(0, 0)
    first_sig_test = metrics.get_first_significant_event(0, 1)

    print(f"First significant training ball: idx={first_sig_training[0]}, time={first_sig_training[1]}s")
    print(f"First significant test ball: idx={first_sig_test[0]}, time={first_sig_test[1]}s")

    print("\n✅ F1 Adjusted Times Test Completed!")
    print(f"Expected behavior: Test ball times should be {tracking_data.f1_exit_time}s less than raw times")

    return {
        "should_adjust_training": should_adjust_training,
        "should_adjust_test": should_adjust_test,
        "adjusted_test_time": adjusted_test,
        "max_event_test_time": max_event_test[1],
        "major_event_test_time": major_event_test[1],
    }


if __name__ == "__main__":
    results = test_f1_adjusted_times()

    # Validate results
    expected_f1_exit_time = 30.0  # F1 corridor exit time
    expected_chamber_exit_time = 3.0  # Initial chamber exit time
    success = True

    if not results["should_adjust_test"]:
        print("❌ ERROR: Test ball should use adjusted time but doesn't")
        success = False

    if results["should_adjust_training"]:
        print("❌ ERROR: Training ball should NOT use adjusted time but does")
        success = False

    if abs(results["adjusted_test_time"] - 15.0) > 0.1:  # 45 - 30 = 15
        print(f"❌ ERROR: Test ball time adjustment incorrect. Expected 15.0, got {results['adjusted_test_time']}")
        success = False

    if success:
        print("\n🎉 All tests passed! F1 adjusted time system working correctly.")
    else:
        print("\n❌ Some tests failed. Check implementation.")
        sys.exit(1)
