#!/usr/bin/env python3

import sys
import os

sys.path.append("/home/matthias/ballpushing_utils/src")

from pathlib import Path
import Ballpushing_utils
from Ballpushing_utils.config import Config


def test_contact_standardized_events():
    """
    Test the new contact-based standardized events functionality.
    """
    print("Testing contact-based standardized events...")

    # Use a single experiment for testing
    experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231222_TNT_Fine_1_Videos_Tracked")

    if not experiment_path.exists():
        print(f"Experiment path does not exist: {experiment_path}")
        return

    # Create experiment
    experiment = Ballpushing_utils.Experiment(experiment_path)

    # Get the first fly for testing
    if not experiment.flies:
        print("No flies found in experiment")
        return

    fly = experiment.flies[0]
    print(f"Testing with fly: {fly.metadata.name}")

    # Configure settings for contact events
    config = Config()
    config.frames_before_onset = 30  # frames before contact center
    config.frames_after_onset = 30  # frames after contact center
    config.contact_min_length = 2  # minimum contact period length
    fly.config = config

    try:
        # Create skeleton metrics
        skeleton_metrics = Ballpushing_utils.SkeletonMetrics(fly)

        # Check the annotated contact dataset
        annotated_df = skeleton_metrics.get_contact_annotated_dataset()
        print(f"Annotated dataset shape: {annotated_df.shape}")
        print(f"Columns: {list(annotated_df.columns)}")

        # Check contact detection
        if "is_contact" in annotated_df.columns:
            contact_frames = annotated_df["is_contact"].sum()
            total_frames = len(annotated_df)
            print(f"Contact frames: {contact_frames}/{total_frames} ({contact_frames/total_frames*100:.1f}%)")
        else:
            print("Warning: No 'is_contact' column found")

        # Test contact period detection
        contact_periods = skeleton_metrics._find_contact_periods(annotated_df)
        print(f"Found {len(contact_periods)} contact periods")
        if contact_periods:
            print(f"First 3 periods: {contact_periods[:3]}")

        # Test standardized contact events
        contact_events = skeleton_metrics.get_standardized_contact_events()
        print(f"Generated {len(contact_events)} standardized contact events")

        # Check the events-based contacts DataFrame
        events_df = skeleton_metrics.events_based_contacts
        print(f"Events-based contacts shape: {events_df.shape}")
        if not events_df.empty:
            event_types = events_df["event_type"].unique()
            print(f"Event types: {event_types}")

            # Count events by type
            for event_type in event_types:
                count = (events_df["event_type"] == event_type).sum()
                print(f"  {event_type}: {count} frames")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_contact_standardized_events()
