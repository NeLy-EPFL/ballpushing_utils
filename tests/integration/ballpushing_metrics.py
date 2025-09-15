import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import time
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
        # Test conditional processing functionality
        "basic_fast": [
            "nb_events",
            "max_event",
            "max_event_time",
            "final_event",
            "final_event_time",
            "chamber_time",
            "chamber_ratio",
            "nb_long_pauses",
            "median_long_pause_duration",
        ],
        "core_analysis": [
            "nb_events",
            "max_event",
            "max_event_time",
            "max_distance",
            "final_event",
            "final_event_time",
            "nb_significant_events",
            "significant_ratio",
            "first_significant_event",
            "first_significant_event_time",
            "distance_moved",
            "distance_ratio",
            "chamber_time",
            "chamber_ratio",
            "pushed",
            "pulled",
            "pulling_ratio",
        ],
        "expensive_only": [
            "learning_slope",
            "learning_slope_r2",
            "logistic_L",
            "logistic_k",
            "logistic_t0",
            "logistic_r2",
            "binned_slope_0",
            "binned_slope_1",
            "binned_slope_2",
            "interaction_rate_bin_0",
            "interaction_rate_bin_1",
            "binned_auc_0",
            "binned_auc_1",
        ],
        "new_metrics": [
            "compute_fraction_not_facing_ball",
            "compute_flailing",
            "compute_head_pushing_ratio",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
            "compute_mean_head_ball_distance",
            "has_finished",
            "has_major",
            "has_significant",
            "compute_persistence_at_end",
            "compute_fly_distance_moved",
            "get_time_chamber_beginning",
            "compute_median_freeze_duration",
            "compute_nb_freeze",
            "compute_long_pause_metrics",
        ],
        "orientation_metrics": [
            "compute_fraction_not_facing_ball",
            "compute_flailing",
            "compute_head_pushing_ratio",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
            "compute_mean_head_ball_distance",
        ],
        "basic_metrics": [
            "get_max_event",
            "get_final_event",
            "has_finished",
            "has_major",
            "has_significant",
            "get_major_event",
            "chamber_ratio",
        ],
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
            "compute_long_pause_metrics",
        ],
        "comprehensive": [
            # Core behavioral metrics
            "get_max_event",
            "get_final_event",
            "has_finished",
            "has_major",
            "has_significant",
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


def test_conditional_processing_performance(fly_path):
    """Test the performance difference between conditional and full processing."""
    print(f"\n{'='*80}")
    print("🚀 CONDITIONAL PROCESSING PERFORMANCE TEST")
    print(f"{'='*80}")

    # Test different metric configurations
    test_configs = [
        {
            "name": "All Metrics (None)",
            "enabled_metrics": None,
            "description": "Default behavior - compute all metrics",
        },
        {
            "name": "Basic Fast",
            "enabled_metrics": get_predefined_metric_sets()["basic_fast"],
            "description": "Only basic, fast metrics",
        },
        {
            "name": "Core Analysis",
            "enabled_metrics": get_predefined_metric_sets()["core_analysis"],
            "description": "Core metrics without expensive computations",
        },
        {
            "name": "Expensive Only",
            "enabled_metrics": get_predefined_metric_sets()["expensive_only"],
            "description": "Only expensive learning/binned metrics",
        },
        {"name": "Empty List", "enabled_metrics": [], "description": "No metrics enabled (testing edge case)"},
    ]

    performance_results = {}

    for config in test_configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"   Description: {config['description']}")

        if config["enabled_metrics"] is not None:
            print(f"   Metrics count: {len(config['enabled_metrics'])}")
            if len(config["enabled_metrics"]) <= 10:
                print(f"   Metrics: {config['enabled_metrics']}")
            else:
                print(f"   Metrics: {config['enabled_metrics'][:3]}... (+{len(config['enabled_metrics'])-3} more)")
        else:
            print(f"   Metrics count: ALL")

        try:
            # Load fly with specific configuration
            test_fly = Fly(fly_path, as_individual=True)

            # Set the enabled_metrics configuration
            test_fly.config.enabled_metrics = config["enabled_metrics"]

            # Time the metrics computation
            start_time = time.time()
            metrics = BallPushingMetrics(test_fly.tracking_data, compute_metrics_on_init=True)
            end_time = time.time()

            computation_time = end_time - start_time
            metrics_computed = sum(len(metrics_dict) for metrics_dict in metrics.metrics.values())

            performance_results[config["name"]] = {
                "time": computation_time,
                "metrics_computed": metrics_computed,
                "config_size": len(config["enabled_metrics"]) if config["enabled_metrics"] is not None else "ALL",
                "success": True,
            }

            print(f"   ⏱️  Time: {computation_time:.4f}s")
            print(f"   📈 Metrics computed: {metrics_computed}")
            print(f"   ✅ Success: True")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            performance_results[config["name"]] = {
                "time": float("inf"),
                "metrics_computed": 0,
                "config_size": len(config["enabled_metrics"]) if config["enabled_metrics"] is not None else "ALL",
                "success": False,
                "error": str(e),
            }

    # Analyze and report performance gains
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    print("-" * 60)

    if "All Metrics (None)" in performance_results and performance_results["All Metrics (None)"]["success"]:
        baseline_time = performance_results["All Metrics (None)"]["time"]
        baseline_metrics = performance_results["All Metrics (None)"]["metrics_computed"]

        print(f"Baseline (All Metrics): {baseline_time:.4f}s, {baseline_metrics} metrics")
        print()

        for name, result in performance_results.items():
            if name != "All Metrics (None)" and result["success"]:
                speedup = baseline_time / result["time"] if result["time"] > 0 else float("inf")
                metrics_reduction = (
                    (baseline_metrics - result["metrics_computed"]) / baseline_metrics * 100
                    if baseline_metrics > 0
                    else 0
                )

                print(
                    f"{name:20} | "
                    f"Time: {result['time']:6.4f}s | "
                    f"Speedup: {speedup:5.1f}x | "
                    f"Metrics: {result['metrics_computed']:3d} | "
                    f"Reduction: {metrics_reduction:5.1f}%"
                )

    return performance_results


def test_enabled_metrics_functionality(fly_path):
    """Test the is_metric_enabled functionality with different configurations."""
    print(f"\n{'='*80}")
    print("🧪 ENABLED METRICS FUNCTIONALITY TEST")
    print(f"{'='*80}")

    # Load a test fly
    test_fly = Fly(fly_path, as_individual=True)

    # Test different configurations
    test_cases = [
        {
            "name": "All Enabled (None)",
            "config": None,
            "test_metrics": ["nb_events", "learning_slope", "binned_slope_0", "nonexistent_metric"],
            "expected": [True, True, True, True],
        },
        {
            "name": "Basic Only",
            "config": ["nb_events", "max_event", "chamber_time"],
            "test_metrics": ["nb_events", "max_event", "chamber_time", "learning_slope", "binned_slope_0"],
            "expected": [True, True, True, False, False],
        },
        {
            "name": "Empty List",
            "config": [],
            "test_metrics": ["nb_events", "max_event", "learning_slope"],
            "expected": [False, False, False],
        },
        {
            "name": "Expensive Only",
            "config": ["learning_slope", "logistic_L", "binned_slope_0"],
            "test_metrics": ["nb_events", "learning_slope", "logistic_L", "binned_slope_0"],
            "expected": [False, True, True, True],
        },
    ]

    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        print(f"   Config: {test_case['config']}")

        # Set the configuration
        test_fly.config.enabled_metrics = test_case["config"]

        # Create metrics object (without computing metrics)
        metrics = BallPushingMetrics(test_fly.tracking_data, compute_metrics_on_init=False)

        # Test is_metric_enabled for each metric
        all_passed = True
        for metric, expected in zip(test_case["test_metrics"], test_case["expected"]):
            actual = metrics.is_metric_enabled(metric)
            passed = actual == expected
            all_passed = all_passed and passed

            status = "✅" if passed else "❌"
            print(f"     {status} {metric:25} | Expected: {expected:5} | Actual: {actual:5}")

        overall_status = "✅ PASSED" if all_passed else "❌ FAILED"
        print(f"   {overall_status}")

    return True


def test_metrics_computation_selectivity(fly_path):
    """Test that only enabled metrics are actually computed and stored."""
    print(f"\n{'='*80}")
    print("🎯 METRICS COMPUTATION SELECTIVITY TEST")
    print(f"{'='*80}")

    # Test with a small set of enabled metrics
    test_fly = Fly(fly_path, as_individual=True)
    test_fly.config.enabled_metrics = ["nb_events", "max_event", "chamber_time"]

    print(f"Enabled metrics: {test_fly.config.enabled_metrics}")

    # Compute metrics
    metrics = BallPushingMetrics(test_fly.tracking_data, compute_metrics_on_init=True)

    # Check what metrics were actually computed
    all_computed_metrics = set()
    for key, metrics_dict in metrics.metrics.items():
        all_computed_metrics.update(metrics_dict.keys())

    print(f"\nActually computed metrics: {sorted(all_computed_metrics)}")
    print(f"Total computed: {len(all_computed_metrics)}")

    # Check if expensive metrics were skipped
    expensive_metrics = {
        "learning_slope",
        "learning_slope_r2",
        "logistic_L",
        "logistic_k",
        "logistic_t0",
        "logistic_r2",
        "binned_slope_0",
        "binned_slope_1",
        "interaction_rate_bin_0",
        "binned_auc_0",
    }

    skipped_expensive = expensive_metrics - all_computed_metrics
    computed_expensive = expensive_metrics & all_computed_metrics

    print(f"\n💰 Expensive metrics analysis:")
    print(f"   Skipped expensive metrics: {sorted(skipped_expensive)} ({len(skipped_expensive)})")
    print(f"   Computed expensive metrics: {sorted(computed_expensive)} ({len(computed_expensive)})")

    # Test with expensive metrics enabled
    print(f"\n🔄 Testing with expensive metrics enabled...")
    test_fly.config.enabled_metrics = ["learning_slope", "logistic_L", "binned_slope_0"]

    metrics_expensive = BallPushingMetrics(test_fly.tracking_data, compute_metrics_on_init=True)

    all_computed_expensive = set()
    for key, metrics_dict in metrics_expensive.metrics.items():
        all_computed_expensive.update(metrics_dict.keys())

    print(f"Computed with expensive config: {sorted(all_computed_expensive)}")

    # Verify that expensive metrics are now computed
    expensive_now_computed = expensive_metrics & all_computed_expensive
    basic_now_skipped = {"nb_events", "max_event", "chamber_time"} - all_computed_expensive

    print(f"   Expensive metrics now computed: {sorted(expensive_now_computed)} ({len(expensive_now_computed)})")
    print(f"   Basic metrics now skipped: {sorted(basic_now_skipped)} ({len(basic_now_skipped)})")

    return {
        "basic_config_computed": len(all_computed_metrics),
        "expensive_config_computed": len(all_computed_expensive),
        "expensive_skipped_in_basic": len(skipped_expensive),
        "expensive_computed_in_expensive": len(expensive_now_computed),
    }


def process_fly(fly_path, metrics_to_test):
    """Process a single fly with comprehensive conditional processing tests."""

    # First run the conditional processing tests if requested
    if "conditional_test" in metrics_to_test or "performance_test" in metrics_to_test:
        print(f"\n{'='*80}")
        print("🧪 COMPREHENSIVE CONDITIONAL PROCESSING TESTS")
        print(f"{'='*80}")
        print(f"Fly path: {fly_path}")

        # Run all conditional processing tests
        print("\n1️⃣ Testing enabled metrics functionality...")
        test_enabled_metrics_functionality(fly_path)

        print("\n2️⃣ Testing performance differences...")
        performance_results = test_conditional_processing_performance(fly_path)

        print("\n3️⃣ Testing metrics computation selectivity...")
        selectivity_results = test_metrics_computation_selectivity(fly_path)

        # Save comprehensive test results
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        fly_name = Path(fly_path).stem
        test_results = {
            "fly_path": str(fly_path),
            "performance_results": performance_results,
            "selectivity_results": selectivity_results,
            "test_timestamp": time.time(),
        }

        results_file = output_dir / f"conditional_processing_test_{fly_name}.json"
        with open(results_file, "w") as f:
            json.dump(convert_to_serializable(test_results), f, indent=2)

        print(f"\n💾 Conditional processing test results saved to: {results_file}")

        # If only testing conditional processing, return here
        if metrics_to_test == ["conditional_test"] or metrics_to_test == ["performance_test"]:
            return

    # Continue with standard metric testing if other metrics are specified
    standard_metrics = [m for m in metrics_to_test if m not in ["conditional_test", "performance_test"]]
    if standard_metrics:
        print(f"\n{'='*60}")
        print("📊 STANDARD METRICS TESTING")
        print(f"{'='*60}")

        # Load an example fly
        ExampleFly = Fly(fly_path, as_individual=True)

        # Initialize the BallPushingMetrics object
        metrics = BallPushingMetrics(ExampleFly.tracking_data)

        test_metrics(metrics, standard_metrics, ExampleFly.metadata.name)


def process_experiment(experiment_path, metrics_to_test):
    """Process an experiment with optional conditional processing tests."""

    # Load the experiment
    ExampleExperiment = Experiment(experiment_path)

    if not ExampleExperiment.flies:
        print("No flies found in experiment!")
        return

    print(f"Testing metrics on experiment with {len(ExampleExperiment.flies)} flies")
    print(f"Experiment path: {experiment_path}")

    # Handle conditional processing tests for experiments
    if "conditional_test" in metrics_to_test or "performance_test" in metrics_to_test:
        print(f"\n{'='*80}")
        print("🧪 EXPERIMENT-WIDE CONDITIONAL PROCESSING TESTS")
        print(f"{'='*80}")

        # Test conditional processing on a subset of flies (first 3 or all if less than 3)
        test_flies = ExampleExperiment.flies[: min(3, len(ExampleExperiment.flies))]

        experiment_performance_results = {}

        for i, fly in enumerate(test_flies):
            print(f"\n🐛 Testing conditional processing on fly {i+1}/{len(test_flies)}: {fly.metadata.name}")

            # Test performance with different configurations
            configs_to_test = [
                ("All Metrics", None),
                ("Basic Fast", get_predefined_metric_sets()["basic_fast"]),
                ("Core Analysis", get_predefined_metric_sets()["core_analysis"]),
            ]

            fly_results = {}
            for config_name, enabled_metrics in configs_to_test:
                try:
                    fly.config.enabled_metrics = enabled_metrics

                    start_time = time.time()
                    metrics = BallPushingMetrics(fly.tracking_data, compute_metrics_on_init=True)
                    end_time = time.time()

                    computation_time = end_time - start_time
                    metrics_computed = sum(len(metrics_dict) for metrics_dict in metrics.metrics.values())

                    fly_results[config_name] = {
                        "time": computation_time,
                        "metrics_computed": metrics_computed,
                        "success": True,
                    }

                    print(f"     {config_name:15} | Time: {computation_time:.3f}s | Metrics: {metrics_computed}")

                except Exception as e:
                    fly_results[config_name] = {"success": False, "error": str(e)}
                    print(f"     {config_name:15} | Error: {str(e)}")

            experiment_performance_results[fly.metadata.name] = fly_results

        # Save experiment performance results
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        exp_name = Path(experiment_path).name
        safe_exp_name = "".join(c for c in exp_name if c.isalnum() or c in ("-", "_")).rstrip()

        perf_file = output_dir / f"experiment_conditional_performance_{safe_exp_name}.json"
        with open(perf_file, "w") as f:
            json.dump(convert_to_serializable(experiment_performance_results), f, indent=2)

        print(f"\n💾 Experiment conditional processing results saved to: {perf_file}")

        # If only testing conditional processing, return here
        if metrics_to_test == ["conditional_test"] or metrics_to_test == ["performance_test"]:
            return

    # Continue with standard processing
    standard_metrics = [m for m in metrics_to_test if m not in ["conditional_test", "performance_test"]]
    if standard_metrics:
        print(f"\n{'='*60}")
        print("📊 STANDARD EXPERIMENT METRICS TESTING")
        print(f"{'='*60}")

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
            fly_results = test_metrics(metrics, standard_metrics, fly.metadata.name, return_results=True)

            # Store results for summary
            all_results[fly.metadata.name] = fly_results

        # Generate experiment-wide summary
        generate_experiment_summary(all_results, experiment_path, standard_metrics)


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

        elif metric_name == "has_major" or metric_name == "get_has_major":
            try:
                result = metrics.get_has_major(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "has_significant" or metric_name == "get_has_significant":
            try:
                result = metrics.get_has_significant(0, 0)
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
            "compute_long_pause_metrics",
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
            "has_major",
            "has_significant",
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
        "has_major",
        "has_significant",
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
        "compute_long_pause_metrics",
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
            "compute_long_pause_metrics",
            "has_finished",
            "has_major",
            "has_significant",
        ]:
            if key_metric in fly_results:
                value = fly_results[key_metric]
                if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value)):
                    key_values.append(f"{key_metric.replace('compute_', '').replace('get_', '')}={value:.3f}")

        if key_values:
            print(f"     Key metrics: {', '.join(key_values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test individual metrics and conditional processing for a fly or experiment."
    )
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
        help="""List of metrics to test. Options include:

        CONDITIONAL PROCESSING TESTS:
        - 'conditional_test': Run comprehensive conditional processing tests
        - 'performance_test': Run performance comparison tests

        PREDEFINED METRIC SETS:
        - 'basic_fast': Fast basic metrics only
        - 'core_analysis': Core metrics without expensive computations
        - 'expensive_only': Only expensive learning/binned metrics
        - 'new_metrics': New skeleton and behavior metrics
        - 'orientation_metrics': Orientation-based metrics
        - 'basic_metrics': Basic behavioral metrics
        - 'skeleton_metrics': Skeleton-based metrics
        - 'comprehensive': All available metrics

        INDIVIDUAL METRICS:
        - Individual metric names (e.g., nb_events max_event learning_slope)
        """,
    )

    args = parser.parse_args()

    # Convert the path to a Path object
    data_path = Path(args.path)

    print(f"🧪 BallPushing Metrics Testing Suite")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Path: {data_path}")
    print(f"Metrics/Tests: {args.metrics}")

    # Show available predefined sets
    if any(m in ["help", "--help", "-h"] for m in args.metrics):
        print(f"\n📋 Available predefined metric sets:")
        for name, metrics in get_predefined_metric_sets().items():
            print(f"  {name:20} | {len(metrics):3d} metrics | {metrics[:3]}...")
        exit(0)

    # Process based on mode
    if args.mode == "fly":
        process_fly(data_path, args.metrics)
    elif args.mode == "experiment":
        process_experiment(data_path, args.metrics)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'fly' or 'experiment'.")

    print(f"\n✅ Testing completed successfully!")
    print(f"\n💡 Tips for conditional processing:")
    print(f"   - Use 'conditional_test' to test the enabled_metrics functionality")
    print(f"   - Use predefined sets like 'basic_fast' for quick analysis")
    print(f"   - Use 'expensive_only' to test only computationally intensive metrics")
    print(f"   - Set fly.config.enabled_metrics in your code to control which metrics are computed")
