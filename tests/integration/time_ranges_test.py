import argparse
from pathlib import Path
from Ballpushing_utils import Fly


def test_time_range(fly_path, time_ranges):
    """
    Test different values of time_range for a Fly object.

    Args:
        fly_path (str): Path to the fly data.
        time_ranges (list): List of time_range tuples to test.
    """
    print(f"Testing Fly object at path: {fly_path}")

    for time_range in time_ranges:
        print(f"\nTesting with time_range: {time_range}")

        # Skip invalid time ranges
        if time_range and time_range[0] is not None and time_range[1] is not None and time_range[0] >= time_range[1]:
            print(f"Skipping invalid time_range: {time_range}")
            continue

        # Initialize the Fly object with the given time_range
        try:
            fly = Fly(fly_path, as_individual=True, custom_config={"time_range": time_range})

            # Access interaction events to trigger filtering
            interaction_events = fly.tracking_data.interaction_events

            if interaction_events:
                print(f"Interaction events found: {len(interaction_events)}")
            else:
                print("No interaction events found.")

            # Print cutoff reference if available
            cutoff_reference = fly.tracking_data.cutoff_reference
            print(f"Cutoff reference: {cutoff_reference}")

        except Exception as e:
            print(f"Error while testing time_range {time_range}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different time_range values for a Fly object.")
    parser.add_argument("--path", required=True, help="Path to the fly data.")
    args = parser.parse_args()

    # Define a list of time_range values to test
    time_ranges_to_test = [
        None,  # No time range
        (0, 2400),  # Valid range
        (2400, 3000),  # Overlapping range
        (2400, None),  # Open-ended range
        (1000, None),
        (None, 2400),  # Open start range
        # (10, 5),  # Invalid range (end < start)
        # (0, 0),  # Empty range
    ]

    test_time_range(args.path, time_ranges_to_test)
