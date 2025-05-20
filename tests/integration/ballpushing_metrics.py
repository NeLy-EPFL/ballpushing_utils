import argparse
from pathlib import Path
import json
from Ballpushing_utils import Fly, BallPushingMetrics
from utils_behavior import Utils
import inspect


def process_fly(fly_path, metrics_to_test):
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    # Initialize the BallPushingMetrics object
    metrics = BallPushingMetrics(ExampleFly.tracking_data)

    # If "all" is in metrics_to_test, collect all public methods that are metrics
    if "all" in metrics_to_test:
        # Exclude private methods and __init__
        all_methods = [
            name
            for name, func in inspect.getmembers(metrics, predicate=inspect.ismethod)
            if not name.startswith("_") and name != "compute_metrics"
        ]
        metrics_to_test = all_methods

    results = {}

    for metric_name in metrics_to_test:
        # Handle known metrics with required arguments
        if metric_name == "pauses":
            try:
                result = metrics.detect_pauses(0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "max_event":
            try:
                result = metrics.get_max_event(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "final_event":
            try:
                result = metrics.get_final_event(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        else:
            # Try to call the metric with no arguments
            try:
                func = getattr(metrics, metric_name)
                result = func()
            except Exception as e:
                result = f"Error: {e}"
        results[metric_name] = result if result is not None else "No result."

    print(json.dumps(results, indent=4))
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
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
