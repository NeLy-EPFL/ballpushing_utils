#!/usr/bin/env python3

import sys
import os

sys.path.append("/home/matthias/ballpushing_utils/src")

from pathlib import Path
import Ballpushing_utils
from Ballpushing_utils.config import Config


def test_dataset_builder_with_contacts():
    """
    Test the dataset builder with the new contact-based standardized events.
    """
    print("Testing dataset builder with contact-based standardized events...")

    # Use a single experiment for testing
    experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231222_TNT_Fine_1_Videos_Tracked")

    if not experiment_path.exists():
        print(f"Experiment path does not exist: {experiment_path}")
        return

    # Create experiment
    experiment = Ballpushing_utils.Experiment(experiment_path)

    # Get the first few flies for testing (limit to avoid long processing)
    test_flies = experiment.flies[:3]  # Test with 3 flies
    print(f"Testing with {len(test_flies)} flies")

    # Configure settings for contact events
    config = Config()
    config.frames_before_onset = 30  # frames before contact center
    config.frames_after_onset = 30  # frames after contact center
    config.contact_min_length = 2  # minimum contact period length

    # Apply config to all test flies
    for fly in test_flies:
        fly.config = config

    try:
        # Create a dataset with standardized_contacts metrics
        dataset = Ballpushing_utils.Dataset(test_flies, dataset_type="standardized_contacts")

        print(f"Dataset created successfully!")
        print(f"Dataset data type: {type(dataset.data)}")

        if dataset.data is not None:
            print(f"Dataset shape: {dataset.data.shape}")
            print(f"Dataset columns: {list(dataset.data.columns)}")

            # Check event types if available
            if "event_type" in dataset.data.columns:
                event_types = dataset.data["event_type"].unique()
                print(f"Event types in dataset: {event_types}")

                # Count by event type
                for event_type in event_types:
                    count = (dataset.data["event_type"] == event_type).sum()
                    print(f"  {event_type}: {count} rows")

            # Check some statistics
            if "fly_name" in dataset.data.columns:
                fly_counts = dataset.data["fly_name"].value_counts()
                print(f"Rows per fly: {fly_counts}")
        else:
            print("Dataset data is None")

    except Exception as e:
        print(f"Error during dataset building: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dataset_builder_with_contacts()
