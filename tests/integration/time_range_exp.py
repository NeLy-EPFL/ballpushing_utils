import argparse
from pathlib import Path
from Ballpushing_utils import Experiment


def test_time_range_experiment(exp_path, time_ranges):
    """
    Test different values of time_range for all Fly objects in an Experiment.

    Args:
        exp_path (str): Path to the experiment directory.
        time_ranges (list): List of time_range tuples to test.
    """
    print(f"Testing Experiment at path: {exp_path}")

    for time_range in time_ranges:
        print(f"\nTesting with time_range: {time_range}")

        # Skip invalid time ranges
        if time_range and time_range[0] is not None and time_range[1] is not None and time_range[0] >= time_range[1]:
            print(f"Skipping invalid time_range: {time_range}")
            continue

        try:
            # Load the experiment with the custom time_range for all flies
            experiment = Experiment(exp_path, custom_config={"time_range": time_range})

            for fly in experiment.flies:
                print(f"  Fly: {fly.metadata.name}")
                try:
                    interaction_events = fly.tracking_data.interaction_events
                    if interaction_events:
                        print(f"    Interaction events found: {len(interaction_events)}")
                    else:
                        print("    No interaction events found.")

                    cutoff_reference = fly.tracking_data.cutoff_reference
                    print(f"    Cutoff reference: {cutoff_reference}")
                except Exception as e:
                    print(f"    Error for fly {fly.metadata.name}: {e}")

        except Exception as e:
            print(f"Error while testing time_range {time_range}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different time_range values for an Experiment.")
    parser.add_argument("--path", required=True, help="Path to the experiment directory.")
    args = parser.parse_args()

    time_ranges_to_test = [
        None,
        (0, 2400),
        (2400, 3000),
        (2400, None),
        (1000, None),
        (None, 2400),
        (10, 5),
        (0, 0),
    ]

    test_time_range_experiment(args.path, time_ranges_to_test)
