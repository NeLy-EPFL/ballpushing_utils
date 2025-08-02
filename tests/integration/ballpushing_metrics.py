import argparse
from pathlib import Path
import json
import numpy as np
from Ballpushing_utils import Fly, BallPushingMetrics
from utils_behavior import Utils
import inspect


def convert_to_serializable(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


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
        if metric_name == "pauses" or metric_name == "detect_pauses":
            try:
                result = metrics.detect_pauses(0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "max_event" or metric_name == "get_max_event":
            try:
                result = metrics.get_max_event(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "final_event" or metric_name == "get_final_event":
            try:
                result = metrics.get_final_event(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "has_finished" or metric_name == "get_has_finished":
            try:
                result = metrics.get_has_finished(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        # Methods that need only fly_idx
        elif metric_name in [
            "persistence_at_end",
            "compute_persistence_at_end",
            "fly_distance_moved",
            "compute_fly_distance_moved",
            "time_chamber_beginning",
            "get_time_chamber_beginning",
            "median_freeze_duration",
            "compute_median_freeze_duration",
            "chamber_ratio",
            "get_chamber_time",
            "compute_pause_metrics",
            "compute_velocity_trend",
        ]:
            try:
                func = getattr(metrics, metric_name)
                result = func(0)
            except Exception as e:
                result = f"Error: {e}"
        # Methods that need fly_idx and ball_idx
        elif metric_name in [
            "compute_auc",
            "compute_binned_auc",
            "compute_binned_slope",
            "compute_event_influence",
            "compute_interaction_persistence",
            "compute_interaction_rate_by_bin",
            "compute_learning_slope",
            "compute_logistic_features",
            "compute_normalized_velocity",
            "compute_overall_interaction_rate",
            "compute_overall_slope",
            "compute_velocity_during_interactions",
            "find_breaks",
            "find_events_direction",
            "get_adjusted_nb_events",
            "get_chamber_exit_time",
            "get_cumulated_breaks_duration",
            "get_distance_moved",
            "get_distance_ratio",
            "get_first_significant_event",
            "get_insight_effect",
            "get_major_event",
            "get_max_distance",
            "get_significant_events",
            "get_success_direction",
        ]:
            try:
                func = getattr(metrics, metric_name)
                result = func(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        # Methods with special arguments
        elif metric_name == "find_event_by_distance":
            try:
                result = metrics.find_event_by_distance(0, 0, 100)  # threshold=100
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "check_yball_variation":
            try:
                # This needs an event tuple and ball_data, skip for now
                result = "Skipped: Requires specific event and ball_data"
            except Exception as e:
                result = f"Error: {e}"
        else:
            # Try to call the metric with no arguments as fallback
            try:
                func = getattr(metrics, metric_name)
                result = func()
            except Exception as e:
                result = f"Error: {e}"
        results[metric_name] = result if result is not None else "No result."

    # Convert numpy types to JSON serializable types
    serializable_results = convert_to_serializable(results)

    print(json.dumps(serializable_results, indent=4))
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "metrics_results.json", "w") as f:
        json.dump(serializable_results, f, indent=4)
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
