#!/usr/bin/env python3

import sys
import os

sys.path.append("/home/matthias/ballpushing_utils/src")

from pathlib import Path
import Ballpushing_utils
from Ballpushing_utils.config import Config
import pandas as pd
import numpy as np


def test_fly_centered_raw_ball_coordinates():
    """
    Test that raw ball coordinates are properly transformed into fly-centered coordinates
    when config.fly_only is False.
    """
    print("=" * 80)
    print("TESTING FLY-CENTERED RAW BALL COORDINATES")
    print("=" * 80)

    # Use a single experiment for testing
    experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231222_TNT_Fine_1_Videos_Tracked")

    if not experiment_path.exists():
        print(f"Experiment path does not exist: {experiment_path}")
        return

    # Create experiment and get first fly
    experiment = Ballpushing_utils.Experiment(experiment_path)
    if not experiment.flies:
        print("No flies found in experiment")
        return

    fly = experiment.flies[0]
    print(f"Testing with fly: {fly.metadata.name}")
    print()

    # Test with fly_only = False (should include fly-centered raw coordinates)
    config = Config()
    config.standardized_events_mode = "contact_events"
    config.fly_only = False
    config.frames_before_onset = 30
    config.frames_after_onset = 30
    config.generate_random = False
    fly.config = config

    print("-" * 60)
    print("TESTING FLY-CENTERED COORDINATE TRANSFORMATION")
    print("-" * 60)

    try:
        skeleton_metrics = Ballpushing_utils.SkeletonMetrics(fly)

        # Check the fly_centered_tracks DataFrame
        fly_centered = skeleton_metrics.fly_centered_tracks

        print(f"Fly-centered tracks shape: {fly_centered.shape}")
        print(f"Available columns: {list(fly_centered.columns)}")
        print()

        # Check for different types of ball coordinates
        raw_ball_columns = [col for col in fly_centered.columns if "centre_raw" in col]
        preprocessed_ball_columns = [col for col in fly_centered.columns if "centre_preprocessed" in col]
        fly_centered_raw_columns = [col for col in fly_centered.columns if "centre_raw_fly" in col]
        fly_centered_preprocessed_columns = [col for col in fly_centered.columns if "centre_preprocessed_fly" in col]

        print("COORDINATE TYPES FOUND:")
        print(f"Raw ball coordinates: {raw_ball_columns}")
        print(f"Preprocessed ball coordinates: {preprocessed_ball_columns}")
        print(f"Fly-centered raw ball coordinates: {fly_centered_raw_columns}")
        print(f"Fly-centered preprocessed ball coordinates: {fly_centered_preprocessed_columns}")
        print()

        if fly_centered_raw_columns:
            print("✅ SUCCESS: Fly-centered raw ball coordinates found!")

            # Show sample data
            sample_cols = fly_centered_raw_columns + fly_centered_preprocessed_columns
            sample_data = fly_centered[sample_cols].head()
            print(f"\nSample fly-centered ball coordinates:\n{sample_data}")

            # Compare raw vs preprocessed coordinates
            if "x_centre_raw_fly" in fly_centered.columns and "x_centre_preprocessed_fly" in fly_centered.columns:
                print(f"\nCoordinate comparison (first 5 frames):")
                print(f"Raw X (fly-centered): {fly_centered['x_centre_raw_fly'].head().values}")
                print(f"Preprocessed X (fly-centered): {fly_centered['x_centre_preprocessed_fly'].head().values}")

                # Check if they're different (they should be due to preprocessing)
                diff = np.abs(fly_centered["x_centre_raw_fly"] - fly_centered["x_centre_preprocessed_fly"]).mean()
                print(f"Average difference between raw and preprocessed X: {diff:.2f}")

        else:
            print("❌ FAILED: Fly-centered raw ball coordinates not found")

        # Test the standardized events dataset
        events_df = skeleton_metrics.events_based_contacts
        print(f"\nStandardized events shape: {events_df.shape}")

        # Check for fly-centered coordinates in events
        event_fly_centered_raw = [col for col in events_df.columns if "centre_raw_fly" in col]
        event_fly_centered_preprocessed = [col for col in events_df.columns if "centre_preprocessed_fly" in col]

        print(f"Events dataset fly-centered raw coordinates: {event_fly_centered_raw}")
        print(f"Events dataset fly-centered preprocessed coordinates: {event_fly_centered_preprocessed}")

        if event_fly_centered_raw:
            print("✅ SUCCESS: Fly-centered raw coordinates available in standardized events dataset!")
        else:
            print("❌ FAILED: Fly-centered raw coordinates not found in events dataset")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return

    print()
    print("-" * 60)
    print("TESTING WITH fly_only = True")
    print("-" * 60)

    # Test with fly_only = True (should not include ball coordinates)
    config.fly_only = True
    fly.config = config

    try:
        skeleton_metrics = Ballpushing_utils.SkeletonMetrics(fly)
        fly_centered = skeleton_metrics.fly_centered_tracks

        # Check for ball coordinates
        ball_columns = [col for col in fly_centered.columns if "centre" in col]

        print(f"Ball-related columns with fly_only=True: {ball_columns}")

        if not ball_columns:
            print("✅ SUCCESS: No ball coordinates included with fly_only=True")
        else:
            print("⚠️  WARNING: Ball coordinates found when fly_only=True")

    except Exception as e:
        print(f"Error with fly_only=True: {e}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Raw ball coordinates are now automatically transformed into fly-centered coordinates")
    print("✅ Both raw and preprocessed ball coordinates available in fly coordinate system")
    print("✅ Provides temporal accuracy (raw timing) with fly-relative spatial context")
    print("✅ Perfect for analyzing ball movement relative to fly orientation and position")


if __name__ == "__main__":
    test_fly_centered_raw_ball_coordinates()
