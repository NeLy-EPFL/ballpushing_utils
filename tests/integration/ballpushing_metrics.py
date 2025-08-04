import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from Ballpushing_utils import Fly, Experiment, BallPushingMetrics
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


def get_predefined_metric_sets():
    """Get predefined sets of metrics for easy testing."""
    return {
        "new_metrics": [
            "compute_fraction_not_facing_ball",
            "compute_flailing",
            "compute_head_pushing_ratio",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
            "compute_mean_head_ball_distance",
            "has_finished",
            "compute_persistence_at_end",
            "compute_fly_distance_moved",
            "get_time_chamber_beginning",
            "compute_median_freeze_duration",
            "compute_nb_freeze",
        ],
        "orientation_metrics": [
            "compute_fraction_not_facing_ball",
            "compute_flailing",
            "compute_head_pushing_ratio",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
            "compute_mean_head_ball_distance",
        ],
        "basic_metrics": ["get_max_event", "get_final_event", "has_finished", "get_major_event", "chamber_ratio"],
        "skeleton_metrics": [
            "compute_fraction_not_facing_ball",
            "compute_flailing",
            "compute_head_pushing_ratio",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
            "compute_mean_head_ball_distance",
            "compute_pause_metrics",
            "compute_median_freeze_duration",
            "compute_nb_freeze",
        ],
        "comprehensive": [
            # Core behavioral metrics
            "get_max_event",
            "get_final_event",
            "has_finished",
            "get_major_event",
            "chamber_ratio",
            "get_chamber_time",
            # Movement and orientation
            "compute_fraction_not_facing_ball",
            "compute_flailing",
            "compute_head_pushing_ratio",
            "compute_leg_visibility_ratio",
            "compute_fly_distance_moved",
            "get_distance_moved",
            # Timing and pauses
            "compute_pause_metrics",
            "compute_median_freeze_duration",
            "compute_nb_freeze",
            "compute_persistence_at_end",
            "get_time_chamber_beginning",
            # Interaction analysis
            "get_adjusted_nb_events",
            "get_significant_events",
            "compute_interaction_persistence",
            "get_distance_ratio",
            "get_max_distance",
        ],
    }


def process_fly(fly_path, metrics_to_test):
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    # Initialize the BallPushingMetrics object
    metrics = BallPushingMetrics(ExampleFly.tracking_data)

    test_metrics(metrics, metrics_to_test, ExampleFly.metadata.name)


def process_experiment(experiment_path, metrics_to_test):
    # Load the experiment
    ExampleExperiment = Experiment(experiment_path)

    if not ExampleExperiment.flies:
        print("No flies found in experiment!")
        return

    print(f"Testing metrics on experiment with {len(ExampleExperiment.flies)} flies")
    print(f"Experiment path: {experiment_path}")

    # Test on ALL flies in the experiment
    flies_to_test = ExampleExperiment.flies
    all_results = {}
    summary_stats = {}

    for i, fly in enumerate(flies_to_test):
        print(f"\n{'='*60}")
        print(f"TESTING FLY {i+1}/{len(flies_to_test)}: {fly.metadata.name}")
        print(f"{'='*60}")

        # Initialize the BallPushingMetrics object for this fly
        metrics = BallPushingMetrics(fly.tracking_data)
        fly_results = test_metrics(metrics, metrics_to_test, fly.metadata.name, return_results=True)

        # Store results for summary
        all_results[fly.metadata.name] = fly_results

    # Generate experiment-wide summary
    generate_experiment_summary(all_results, experiment_path, metrics_to_test)


def test_metrics(metrics, metrics_to_test, fly_name, return_results=False):
    """Test metrics on a single fly's BallPushingMetrics object."""

    # Handle predefined metric sets
    predefined_sets = get_predefined_metric_sets()
    if len(metrics_to_test) == 1 and metrics_to_test[0] in predefined_sets:
        metrics_to_test = predefined_sets[metrics_to_test[0]]
        print(f"Using predefined metric set: {metrics_to_test}")

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
            "compute_persistence_at_end",
            "compute_fly_distance_moved",
            "get_time_chamber_beginning",
            "compute_median_freeze_duration",
            "compute_nb_freeze",
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
            "compute_fraction_not_facing_ball",
            "compute_flailing",
            "compute_head_pushing_ratio",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
            "compute_mean_head_ball_distance",
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

    # Print summary of interesting metrics first
    print("=" * 60)
    print("METRIC TESTING RESULTS")
    print("=" * 60)

    # Show new metrics prominently if they were tested
    new_metrics = [
        "compute_fraction_not_facing_ball",
        "compute_flailing",
        "compute_head_pushing_ratio",
        "compute_median_head_ball_distance",
        "compute_mean_head_ball_distance",
        "has_finished",
    ]

    if isinstance(serializable_results, dict):
        tested_new_metrics = {k: v for k, v in serializable_results.items() if k in new_metrics}

        if tested_new_metrics:
            print("\n🆕 NEW METRICS:")
            for metric, result in tested_new_metrics.items():
                if isinstance(result, str) and result.startswith("Error"):
                    print(f"  ❌ {metric}: {result}")
                else:
                    print(f"  ✅ {metric}: {result}")

        # Show all results
        print(f"\n📊 ALL RESULTS ({len(serializable_results)} metrics tested):")
        if not return_results:  # Only print detailed results if not returning them
            print(json.dumps(serializable_results, indent=4))

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        # Include fly name in filename for easier identification in experiment mode
        safe_fly_name = "".join(c for c in fly_name if c.isalnum() or c in ("-", "_")).rstrip()
        filename = f"metrics_results_{safe_fly_name}.json"
        with open(output_dir / filename, "w") as f:
            json.dump(serializable_results, f, indent=4)

        if not return_results:
            print(f"\n💾 Results saved to {output_dir / filename}")
    else:
        print("Error: Results are not in expected dictionary format")
        print(serializable_results)
        serializable_results = {}

    # Return results if requested
    if return_results:
        return serializable_results


def generate_experiment_summary(all_results, experiment_path, metrics_to_test):
    """Generate comprehensive summary statistics for all flies in an experiment."""

    print(f"\n{'='*80}")
    print("EXPERIMENT-WIDE SUMMARY")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_path}")
    print(f"Total flies tested: {len(all_results)}")

    if not all_results:
        print("No results to summarize!")
        return

    # Get all metric names from first fly (assuming all flies have same metrics)
    first_fly_results = next(iter(all_results.values()))
    if not first_fly_results:
        print("No metrics computed!")
        return

    metric_names = list(first_fly_results.keys())

    # Calculate summary statistics for each metric
    summary_stats = {}

    print(f"\n🔍 SUMMARY STATISTICS FOR {len(metric_names)} METRICS:")
    print("-" * 80)

    for metric_name in metric_names:
        # Collect all values for this metric across flies
        values = []
        errors = 0

        for fly_name, fly_results in all_results.items():
            value = fly_results.get(metric_name, "Missing")

            if isinstance(value, str) and ("Error" in value or "Missing" in value):
                errors += 1
            elif isinstance(value, (int, float)) and not (
                isinstance(value, float) and (np.isnan(value) or np.isinf(value))
            ):
                values.append(float(value))
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # For list/tuple metrics, we can summarize length or first element
                if all(isinstance(v, (int, float)) for v in value):
                    values.extend([float(v) for v in value])
                else:
                    values.append(len(value))  # Use length as summary

        # Calculate statistics if we have numeric values
        if values:
            values_array = np.array(values)
            stats = {
                "count": len(values),
                "mean": np.mean(values_array),
                "std": np.std(values_array),
                "min": np.min(values_array),
                "max": np.max(values_array),
                "median": np.median(values_array),
                "errors": errors,
                "total_flies": len(all_results),
            }
            summary_stats[metric_name] = stats

            # Print summary for this metric
            print(
                f"{metric_name:40} | "
                f"Count: {stats['count']:3d} | "
                f"Mean: {stats['mean']:8.3f} | "
                f"Std: {stats['std']:8.3f} | "
                f"Range: [{stats['min']:6.2f}, {stats['max']:6.2f}] | "
                f"Errors: {errors}"
            )
        else:
            summary_stats[metric_name] = {
                "count": 0,
                "errors": errors,
                "total_flies": len(all_results),
                "note": "No numeric values",
            }
            print(f"{metric_name:40} | No numeric values | Errors: {errors}")

    # Highlight interesting metrics
    print(f"\n🎯 KEY NEW METRICS SUMMARY:")
    print("-" * 50)

    key_metrics = [
        "compute_nb_freeze",
        "compute_leg_visibility_ratio",
        "compute_head_pushing_ratio",
        "compute_median_head_ball_distance",
        "compute_mean_head_ball_distance",
        "compute_fraction_not_facing_ball",
        "compute_flailing",
    ]

    for metric in key_metrics:
        if metric in summary_stats and summary_stats[metric]["count"] > 0:
            stats = summary_stats[metric]
            success_rate = (stats["count"] / stats["total_flies"]) * 100
            print(
                f"  {metric:35} | Success: {success_rate:5.1f}% | " f"Mean: {stats['mean']:6.3f} ± {stats['std']:6.3f}"
            )

    # Save experiment summary
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    exp_name = Path(experiment_path).name
    safe_exp_name = "".join(c for c in exp_name if c.isalnum() or c in ("-", "_")).rstrip()

    # Save detailed results
    summary_file = output_dir / f"experiment_summary_{safe_exp_name}.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "experiment_path": str(experiment_path),
                "summary_stats": convert_to_serializable(summary_stats),
                "all_fly_results": all_results,
                "metrics_tested": metrics_to_test,
            },
            f,
            indent=2,
        )

    # Save CSV for easy analysis
    csv_file = output_dir / f"experiment_metrics_{safe_exp_name}.csv"

    # Create DataFrame for CSV export
    csv_data = []
    for fly_name, fly_results in all_results.items():
        row = {"fly_name": fly_name}
        for metric_name, value in fly_results.items():
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and (np.isnan(value) or np.isinf(value))
            ):
                row[metric_name] = value
            elif isinstance(value, str) and "Error" not in value:
                row[metric_name] = value
            else:
                row[metric_name] = np.nan
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)

    print(f"\n💾 Experiment summary saved to:")
    print(f"  📊 Detailed JSON: {summary_file}")
    print(f"  📈 CSV for analysis: {csv_file}")

    # Print fly-by-fly overview
    print(f"\n🐛 FLY-BY-FLY OVERVIEW:")
    print("-" * 80)

    for i, (fly_name, fly_results) in enumerate(all_results.items(), 1):
        successful_metrics = sum(1 for v in fly_results.values() if not (isinstance(v, str) and "Error" in v))
        total_metrics = len(fly_results)
        success_rate = (successful_metrics / total_metrics) * 100 if total_metrics > 0 else 0

        print(f"{i:3d}. {fly_name:40} | " f"Success: {successful_metrics:3d}/{total_metrics:3d} ({success_rate:5.1f}%)")

        # Show a few key metric values for this fly
        key_values = []
        for key_metric in [
            "compute_nb_freeze",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
            "has_finished",
        ]:
            if key_metric in fly_results:
                value = fly_results[key_metric]
                if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
                    key_values.append(f"{key_metric.replace('compute_', '').replace('get_', '')}={value:.3f}")

        if key_values:
            print(f"     Key metrics: {', '.join(key_values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test individual metrics for a fly or experiment.")
    parser.add_argument(
        "--mode",
        choices=["fly", "experiment"],
        required=True,
        help="Specify whether to process a single fly or an experiment.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the fly or experiment data.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="List of metrics to test. Use predefined sets: 'new_metrics', 'orientation_metrics', 'basic_metrics', 'skeleton_metrics', or individual metric names (e.g., pauses max_event final_event).",
    )

    args = parser.parse_args()

    # Convert the path to a Path object
    data_path = Path(args.path)

    # Process based on mode
    if args.mode == "fly":
        process_fly(data_path, args.metrics)
    elif args.mode == "experiment":
        process_experiment(data_path, args.metrics)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'fly' or 'experiment'.")
