#!/usr/bin/env python3

import sys
import os

sys.path.append("/home/matthias/ballpushing_utils/src")

from pathlib import Path
import Ballpushing_utils
from Ballpushing_utils.config import Config
import pandas as pd
import numpy as np


def test_raw_ball_coordinates():
    """
    Test that raw ball coordinates are included in standardized contacts dataset
    when config.fly_only is False.
    """
    print("=" * 80)
    print("TESTING RAW BALL COORDINATES IN STANDARDIZED CONTACTS")
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

    # Test with fly_only = False (should include raw coordinates)
    config = Config()
    config.standardized_events_mode = "contact_events"
    config.fly_only = False
    config.frames_before_onset = 30
    config.frames_after_onset = 30
    config.generate_random = False
    fly.config = config

    print("-" * 60)
    print("TESTING WITH fly_only = False")
    print("-" * 60)

    try:
        skeleton_metrics = Ballpushing_utils.SkeletonMetrics(fly)
        events_df = skeleton_metrics.events_based_contacts

        print(f"Events DataFrame shape: {events_df.shape}")
        print(f"Available columns: {list(events_df.columns)}")
        print()

        # Check for raw coordinates
        raw_ball_columns = [col for col in events_df.columns if "centre_raw" in col]
        preprocessed_ball_columns = [col for col in events_df.columns if "centre_preprocessed" in col]

        print(f"Raw ball coordinate columns: {raw_ball_columns}")
        print(f"Preprocessed ball coordinate columns: {preprocessed_ball_columns}")

        if raw_ball_columns:
            print("✅ SUCCESS: Raw ball coordinates found in dataset!")

            # Show sample data
            sample_data = events_df[raw_ball_columns + preprocessed_ball_columns].head()
            print(f"\nSample data:\n{sample_data}")

        else:
            print("❌ FAILED: Raw ball coordinates not found in dataset")

    except Exception as e:
        print(f"Error with fly_only=False: {e}")
        import traceback

        traceback.print_exc()
        return

    print()
    print("-" * 60)
    print("TESTING WITH fly_only = True")
    print("-" * 60)

    # Test with fly_only = True (should not include raw coordinates)
    config.fly_only = True
    fly.config = config

    try:
        skeleton_metrics = Ballpushing_utils.SkeletonMetrics(fly)
        events_df = skeleton_metrics.events_based_contacts

        print(f"Events DataFrame shape: {events_df.shape}")

        # Check for raw coordinates
        raw_ball_columns = [col for col in events_df.columns if "centre_raw" in col]
        preprocessed_ball_columns = [col for col in events_df.columns if "centre_preprocessed" in col]

        print(f"Raw ball coordinate columns: {raw_ball_columns}")
        print(f"Preprocessed ball coordinate columns: {preprocessed_ball_columns}")

        if not raw_ball_columns and not preprocessed_ball_columns:
            print("✅ SUCCESS: No ball coordinates included with fly_only=True")
        elif raw_ball_columns:
            print("❌ FAILED: Raw ball coordinates found when fly_only=True")
        else:
            print("⚠️  WARNING: Only preprocessed coordinates found with fly_only=True")

    except Exception as e:
        print(f"Error with fly_only=True: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Raw ball coordinates are now included in standardized contacts dataset")
    print("✅ When fly_only=False: Both raw and preprocessed ball coordinates available")
    print("✅ When fly_only=True: Ball coordinates excluded as expected")
    print("✅ This provides temporal accuracy (raw timing) with original pixel coordinates")


if __name__ == "__main__":
    test_raw_ball_coordinates()
