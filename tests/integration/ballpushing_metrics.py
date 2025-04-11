import argparse
from pathlib import Path
import json
from Ballpushing_utils import Fly, BallPushingMetrics
from utils_behavior import Utils


def process_fly(fly_path, metrics_to_test):
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    # Initialize the BallPushingMetrics object
    metrics = BallPushingMetrics(ExampleFly.tracking_data)

    # Store results in a dictionary
    results = {}

    # Test the selected metrics
    if "pauses" in metrics_to_test:
        pauses = metrics.detect_pauses(0)
        results["pauses"] = pauses if pauses is not None else "No pauses detected."

    if "max_event" in metrics_to_test:
        max_event = metrics.get_max_event(0, 0)
        results["max_event"] = max_event if max_event is not None else "Max Event is None."

    if "final_event" in metrics_to_test:
        final_event = metrics.get_final_event(0, 0)
        results["final_event"] = final_event if final_event is not None else "Final Event is None."

    # Add more metrics as needed
    # Example:
    # if "some_other_metric" in metrics_to_test:
    #     result = metrics.some_other_function()
    #     results["some_other_metric"] = result

    # Print results in a structured format
    print(json.dumps(results, indent=4))
    output_dir = Path(__file__).parent / "outputs"
    # Optionally, write results to a file
    with open(output_dir / "metrics_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_dir / 'metrics_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test individual metrics for a fly.")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the fly data.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="List of metrics to test (e.g., pauses max_event final_event).",
    )

    args = parser.parse_args()

    # Convert the path to a Path object
    fly_path = Path(args.path)

    # Process the fly and test the selected metrics
    process_fly(fly_path, args.metrics)
