import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import time
from Ballpushing_utils import Fly, Experiment, BallPushingMetrics
from utils_behavior import Utils
import inspect
import warnings

warnings.filterwarnings("ignore")


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


def format_metric_value(value, max_width=15, verbose_dicts=True):
    """Format a metric value for display in tables."""
    if value is None:
        return "None"
    elif isinstance(value, str):
        if value.startswith("Error"):
            return "❌ ERROR"
        elif "Skipped" in value:
            return "⏩ SKIP"
        elif len(value) > max_width:
            return value[: max_width - 3] + "..."
        else:
            return value
    elif isinstance(value, bool):
        return "✅ Yes" if value else "❌ No"
    elif isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    elif isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return "NaN"
        elif abs(value) < 0.001:
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        elif abs(value) < 100:
            return f"{value:.3f}"
        else:
            return f"{value:.1f}"
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        elif len(value) <= 3:
            formatted_items = [format_metric_value(item, max_width // len(value), verbose_dicts) for item in value]
            return f"[{', '.join(formatted_items)}]"
        else:
            return f"[{len(value)} items]"
    elif isinstance(value, dict):
        # Special handling for ball identity results (training/test)
        if set(value.keys()).issubset({"training", "test"}):
            parts = []
            for ball_type in ["training", "test"]:
                if ball_type in value:
                    ball_value = value[ball_type]
                    if isinstance(ball_value, (int, float)):
                        if isinstance(ball_value, float) and (np.isnan(ball_value) or np.isinf(ball_value)):
                            formatted = "NaN"
                        elif abs(ball_value) < 0.001 and ball_value != 0:
                            formatted = f"{ball_value:.2e}"
                        elif abs(ball_value) < 1 and ball_value != 0:
                            formatted = f"{ball_value:.3f}"
                        else:
                            formatted = f"{ball_value}"
                    elif isinstance(ball_value, bool):
                        formatted = "✅" if ball_value else "❌"
                    elif isinstance(ball_value, str) and ball_value.startswith("Error"):
                        formatted = "ERR"
                    else:
                        formatted = str(ball_value)[:8]
                    parts.append(f"{ball_type}: {formatted}")
            return " | ".join(parts)
        elif not verbose_dicts:
            return f"{{dict: {len(value)} keys}}"
        elif len(value) == 0:
            return "{}"
        elif len(value) <= 3:
            # Show small dictionaries in full
            items = []
            for k, v in list(value.items())[:3]:
                formatted_v = format_metric_value(v, max_width=8, verbose_dicts=verbose_dicts)
                items.append(f"{k}: {formatted_v}")
            return "{" + ", ".join(items) + "}"
        else:
            # Show first few items plus count
            items = []
            for k, v in list(value.items())[:2]:
                formatted_v = format_metric_value(v, max_width=6, verbose_dicts=verbose_dicts)
                items.append(f"{k}: {formatted_v}")
            return "{" + ", ".join(items) + f", +{len(value)-2} more" + "}"
    else:
        str_val = str(value)
        if len(str_val) > max_width:
            return str_val[: max_width - 3] + "..."
        return str_val


def format_dict_detailed(value, indent="  "):
    """Format a dictionary with detailed key-value pairs on separate lines."""
    if not isinstance(value, dict) or len(value) == 0:
        return str(value)

    lines = ["{"]
    for k, v in value.items():
        if isinstance(v, dict) and len(v) > 0:
            nested_dict = format_dict_detailed(v, indent + "  ")
            lines.append(f"{indent}{k}: {nested_dict}")
        else:
            formatted_v = format_metric_value(v, max_width=30, verbose_dicts=False)
            lines.append(f"{indent}{k}: {formatted_v}")
    lines.append("}")
    return "\n".join(lines)


def create_metrics_table(results, fly_name, table_style="summary"):
    """
    Create a formatted table for metrics results.

    Parameters
    ----------
    results : dict
        Dictionary of metric names and their values
    fly_name : str
        Name of the fly being tested
    table_style : str
        Style of table to create: "summary", "detailed", "categorical"

    Returns
    -------
    str
        Formatted table string
    """
    if not results:
        return "No metrics results to display."

    # Convert results to DataFrame
    data = []
    for metric_name, value in results.items():
        formatted_value = format_metric_value(value)

        # Determine metric category
        category = categorize_metric(metric_name)

        # Determine status
        if isinstance(value, str) and "Error" in value:
            status = "❌ ERROR"
        elif isinstance(value, str) and "Skipped" in value:
            status = "⏩ SKIP"
        elif value is None:
            status = "❓ NULL"
        else:
            status = "✅ OK"

        data.append(
            {
                "Metric": metric_name,
                "Category": category,
                "Value": formatted_value,
                "Status": status,
                "Raw_Value": value,
            }
        )

    df = pd.DataFrame(data)

    if table_style == "summary":
        # Summary table with key metrics highlighted
        summary_df = df[["Metric", "Value", "Status"]].copy()

        # Prioritize key metrics
        key_metrics = [
            "nb_events",
            "max_event",
            "final_event",
            "chamber_time",
            "chamber_ratio",
            "nb_stops",
            "nb_pauses",
            "nb_long_pauses",
            "has_long_pauses",
            "median_stop_duration",
            "median_pause_duration",
            "median_long_pause_duration",
            "has_finished",
            "has_major",
            "has_significant",
        ]

        # Sort: key metrics first, then alphabetically
        def metric_priority(metric):
            if metric in key_metrics:
                return (0, key_metrics.index(metric))
            else:
                return (1, metric)

        summary_df["sort_key"] = summary_df["Metric"].apply(metric_priority)
        summary_df = summary_df.sort_values("sort_key").drop("sort_key", axis=1)

        output = f"""
{'='*80}
📊 METRICS SUMMARY TABLE - {fly_name}
{'='*80}
{summary_df.to_string(index=False, max_colwidth=25)}

📈 Success Rate: {len(df[df['Status'] == '✅ OK'])}/{len(df)} ({len(df[df['Status'] == '✅ OK'])/len(df)*100:.1f}%)
"""

        # Add detailed dictionary section if there are any
        dict_metrics = {k: v for k, v in results.items() if isinstance(v, dict) and len(v) > 0}
        if dict_metrics:
            output += f"""

🗂️  DETAILED DICTIONARY METRICS:
{'='*60}
"""
            for metric_name, dict_value in dict_metrics.items():
                output += f"""
📋 {metric_name}:
{format_dict_detailed(dict_value)}
"""

        return output

    elif table_style == "categorical":
        # Group by category
        category_summary = (
            df.groupby("Category")
            .agg({"Status": lambda x: f"{len(x[x == '✅ OK'])}/{len(x)} OK", "Metric": "count"})
            .rename(columns={"Metric": "Count"})
        )

        output = f"""
{'='*80}
📊 METRICS BY CATEGORY - {fly_name}
{'='*80}
{category_summary.to_string()}

"""

        # Show details for each category
        for category in df["Category"].unique():
            cat_df = df[df["Category"] == category][["Metric", "Value", "Status"]]
            output += f"""
🏷️  {category.upper()} METRICS:
{'-'*60}
{cat_df.to_string(index=False, max_colwidth=20)}

"""
        return output

    elif table_style == "detailed":
        # Full detailed table
        detailed_df = df[["Metric", "Category", "Value", "Status"]].copy()
        detailed_df = detailed_df.sort_values(["Category", "Metric"])

        output = f"""
{'='*80}
📊 DETAILED METRICS TABLE - {fly_name}
{'='*80}
{detailed_df.to_string(index=False, max_colwidth=30)}

📊 SUMMARY BY STATUS:
{df['Status'].value_counts().to_string()}
"""

        # Add detailed dictionary section
        dict_metrics = {k: v for k, v in results.items() if isinstance(v, dict) and len(v) > 0}
        if dict_metrics:
            output += f"""

🗂️  DETAILED DICTIONARY METRICS:
{'='*60}
"""
            for metric_name, dict_value in dict_metrics.items():
                category = categorize_metric(metric_name)
                output += f"""
📋 {metric_name} ({category}):
{format_dict_detailed(dict_value)}
"""

        return output

    else:
        return df.to_string(index=False)


def categorize_metric(metric_name):
    """Categorize metrics into logical groups."""
    behavioral_metrics = [
        "nb_events",
        "max_event",
        "final_event",
        "has_finished",
        "has_major",
        "has_significant",
        "get_max_event",
        "get_final_event",
        "get_major_event",
    ]

    movement_metrics = [
        "distance_moved",
        "distance_ratio",
        "max_distance",
        "compute_fly_distance_moved",
        "get_distance_moved",
        "get_max_distance",
    ]

    timing_metrics = [
        "chamber_time",
        "chamber_ratio",
        "chamber_exit_time",
        "max_event_time",
        "final_event_time",
        "first_significant_event_time",
        "get_chamber_time",
        "get_time_chamber_beginning",
    ]

    pause_metrics = [
        "compute_pause_metrics",
        "compute_stop_metrics",
        "compute_long_pause_metrics",
        "nb_pauses",
        "median_pause_duration",
        "total_pause_duration",
        "nb_stops",
        "median_stop_duration",
        "total_stop_duration",
        "nb_long_pauses",
        "median_long_pause_duration",
        "total_long_pause_duration",
        "has_long_pauses",
        "compute_nb_stops",
        "compute_median_freeze_duration",
        "compute_nb_freeze",
        "pauses_persistence",
        "compute_pauses_persistence",
    ]

    orientation_metrics = [
        "compute_fraction_not_facing_ball",
        "compute_flailing",
        "compute_head_pushing_ratio",
        "compute_leg_visibility_ratio",
        "compute_median_head_ball_distance",
        "compute_mean_head_ball_distance",
    ]

    learning_metrics = [
        "learning_slope",
        "learning_slope_r2",
        "logistic_L",
        "logistic_k",
        "logistic_t0",
        "logistic_r2",
        "compute_learning_slope",
        "compute_logistic_features",
    ]

    advanced_metrics = [
        "binned_slope_0",
        "binned_slope_1",
        "binned_slope_2",
        "interaction_rate_bin_0",
        "interaction_rate_bin_1",
        "binned_auc_0",
        "binned_auc_1",
        "compute_binned_slope",
        "compute_binned_auc",
        "compute_interaction_rate_by_bin",
    ]

    if metric_name in behavioral_metrics:
        return "Behavioral"
    elif metric_name in movement_metrics:
        return "Movement"
    elif metric_name in timing_metrics:
        return "Timing"
    elif metric_name in pause_metrics:
        return "Pauses"
    elif metric_name in orientation_metrics:
        return "Orientation"
    elif metric_name in learning_metrics:
        return "Learning"
    elif metric_name in advanced_metrics:
        return "Advanced"
    else:
        return "Other"


def create_experiment_metrics_table(all_results, metrics_to_summarize=None, max_flies=None):
    """
    Create a comprehensive table showing metrics across all flies in an experiment.

    Parameters
    ----------
    all_results : dict
        Dictionary with fly names as keys and their metric results as values
    metrics_to_summarize : list, optional
        List of specific metrics to include in the summary. If None, uses key metrics.
    max_flies : int, optional
        Maximum number of flies to show in the table. If None, shows all.

    Returns
    -------
    str
        Formatted experiment table
    """
    if not all_results:
        return "No experiment results to display."

    # Default key metrics if not specified
    if metrics_to_summarize is None:
        metrics_to_summarize = [
            "nb_events",
            "max_event",
            "final_event",
            "chamber_time",
            "chamber_ratio",
            "nb_long_pauses",
            "pauses_persistence",
            "compute_nb_freeze",
            "has_finished",
            "has_major",
            "has_significant",
            "compute_median_freeze_duration",
            "compute_head_pushing_ratio",
        ]

    # Filter metrics that actually exist in the data
    first_fly_results = next(iter(all_results.values()))
    available_metrics = [m for m in metrics_to_summarize if m in first_fly_results]

    if not available_metrics:
        # Fall back to first 10 metrics if none of the preferred ones exist
        available_metrics = list(first_fly_results.keys())[:10]

    # Build the data matrix
    data = []
    fly_names = list(all_results.keys())
    if max_flies:
        fly_names = fly_names[:max_flies]

    for fly_name in fly_names:
        row = {"Fly": fly_name}
        fly_results = all_results[fly_name]

        for metric in available_metrics:
            value = fly_results.get(metric, "Missing")
            formatted_value = format_metric_value(value, max_width=10)
            row[metric] = formatted_value

        data.append(row)

    df = pd.DataFrame(data)

    # Calculate summary statistics
    numeric_data = []
    for fly_name in fly_names:
        fly_results = all_results[fly_name]
        row = {"Fly": fly_name}
        for metric in available_metrics:
            value = fly_results.get(metric, np.nan)
            # Convert to numeric if possible
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and (np.isnan(value) or np.isinf(value))
            ):
                row[metric] = float(value)
            else:
                row[metric] = np.nan
        numeric_data.append(row)

    numeric_df = pd.DataFrame(numeric_data).set_index("Fly")

    # Calculate summary statistics
    summary_stats = pd.DataFrame(
        {
            "Mean": numeric_df.mean(),
            "Std": numeric_df.std(),
            "Min": numeric_df.min(),
            "Max": numeric_df.max(),
            "Count": numeric_df.count(),
        }
    ).round(3)

    output = f"""
{'='*100}
📊 EXPERIMENT METRICS TABLE
{'='*100}
Flies: {len(fly_names)} {'(showing first {})'.format(max_flies) if max_flies and len(all_results) > max_flies else ''}
Metrics: {len(available_metrics)}

📋 FLY-BY-FLY VALUES:
{'-'*100}
{df.to_string(index=False, max_colwidth=12)}

📈 SUMMARY STATISTICS:
{'-'*60}
{summary_stats.to_string()}
"""

    # Add success rate information
    success_info = []
    for metric in available_metrics:
        success_count = sum(
            1
            for fly_name in fly_names
            if not (
                isinstance(all_results[fly_name].get(metric), str) and "Error" in str(all_results[fly_name].get(metric))
            )
        )
        success_rate = (success_count / len(fly_names)) * 100
        success_info.append(f"{metric}: {success_rate:.0f}%")

    output += f"""

✅ SUCCESS RATES BY METRIC:
{'-'*40}
{', '.join(success_info)}
"""

    return output


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
            "nb_stops",
            "nb_pauses",
            "nb_long_pauses",
            "has_long_pauses",
            "median_stop_duration",
            "median_pause_duration",
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
            "compute_stop_metrics",
            "compute_pause_metrics",
            "compute_long_pause_metrics",
            "has_long_pauses",
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
            "compute_stop_metrics",
            "compute_pause_metrics",
            "compute_long_pause_metrics",
            "has_long_pauses",
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
        "pause_metrics_focus": [
            # Updated pause metrics with non-overlapping ranges
            "compute_stop_metrics",
            "compute_pause_metrics",
            "compute_long_pause_metrics",
            "has_long_pauses",
            "nb_stops",
            "median_stop_duration",
            "total_stop_duration",
            "nb_pauses",
            "median_pause_duration",
            "total_pause_duration",
            "nb_long_pauses",
            "median_long_pause_duration",
            "total_long_pause_duration",
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


def test_ball_identity_functionality(fly_path):
    """Test ball identity features for F1 experiments."""
    print(f"\n{'='*80}")
    print("🎯 BALL IDENTITY FUNCTIONALITY TEST")
    print(f"{'='*80}")

    # Load the fly
    test_fly = Fly(fly_path, as_individual=True)

    print(f"Fly path: {fly_path}")
    print(f"Fly name: {test_fly.metadata.name}")

    # Test ball identity assignment
    tracking_data = test_fly.tracking_data

    print(f"\n📊 Ball Identity Analysis:")

    # Check if tracking data exists
    if tracking_data is None:
        print(f"   ❌ No tracking data available")
        return False

    if hasattr(tracking_data, "balltrack") and tracking_data.balltrack is not None:
        # Check if it's a SLEAP tracks object or regular tracking data
        if hasattr(tracking_data.balltrack, "objects"):
            print(f"   Number of balls: {len(tracking_data.balltrack.objects)}")
        elif hasattr(tracking_data.balltrack, "nb_objects"):
            print(f"   Number of balls: {tracking_data.balltrack.nb_objects}")
        else:
            print(f"   Number of balls: Unknown (no objects attribute)")

        # Get SLEAP track names if available
        if hasattr(tracking_data.balltrack, "track_names"):
            track_names = tracking_data.balltrack.track_names
            print(f"   SLEAP track names: {track_names}")
        elif hasattr(tracking_data.balltrack, "sleap_tracks") and tracking_data.balltrack.sleap_tracks is not None:
            sleap_tracks = tracking_data.balltrack.sleap_tracks
            if hasattr(sleap_tracks, "track_names"):
                track_names = sleap_tracks.track_names
                print(f"   SLEAP track names: {track_names}")
            else:
                print(f"   No track names available in SLEAP tracks")
                track_names = None
        else:
            print(f"   No SLEAP track names available")
            track_names = None
    else:
        print(f"   ❌ No ball tracking data available")
        return False

    # Test ball identity methods
    try:
        tracking_data.assign_ball_identities()

        print(f"\n🔧 Ball Identity Methods:")
        print(f"   Training ball index: {tracking_data.training_ball_idx}")
        print(f"   Test ball index: {tracking_data.test_ball_idx}")
        print(f"   Ball identities: {tracking_data.ball_identities}")
        print(f"   Identity mappings: {tracking_data.identity_to_idx}")

        # Test helper methods
        print(f"\n🧪 Helper Method Tests:")
        print(f"   has_training_ball(): {tracking_data.has_training_ball()}")
        print(f"   has_test_ball(): {tracking_data.has_test_ball()}")
        print(f"   get_all_ball_identities(): {tracking_data.get_all_ball_identities()}")

        if tracking_data.has_training_ball():
            training_data = tracking_data.get_training_ball_data()
            print(f"   Training ball data available: {training_data is not None}")

        if tracking_data.has_test_ball():
            test_data = tracking_data.get_test_ball_data()
            print(f"   Test ball data available: {test_data is not None}")

        # Test pattern matching
        training_balls = tracking_data.get_balls_by_identity_pattern("training")
        test_balls = tracking_data.get_balls_by_identity_pattern("test")
        print(f"   Balls matching 'training': {training_balls}")
        print(f"   Balls matching 'test': {test_balls}")

    except Exception as e:
        print(f"   ❌ Error in ball identity assignment: {e}")
        return False

    # Test metrics with ball identity
    print(f"\n📈 Ball Identity Metrics Test:")
    try:
        metrics = BallPushingMetrics(tracking_data, compute_metrics_on_init=True)

        # Check if metrics include ball identity information
        print(f"   Metrics computed: {len(metrics.metrics)} entries")

        # Show metrics structure for each ball
        for key, metric_dict in metrics.metrics.items():
            print(f"   Key: {key}")
            if "ball_identity" in metric_dict:
                print(f"     Ball identity: {metric_dict['ball_identity']}")
            if "fly_idx" in metric_dict:
                print(f"     Fly idx: {metric_dict['fly_idx']}")
            if "ball_idx" in metric_dict:
                print(f"     Ball idx: {metric_dict['ball_idx']}")

        # Test ball identity retrieval methods
        print(f"\n🎯 Ball Identity Retrieval:")

        try:
            get_by_identity = getattr(metrics, "get_metrics_by_ball_identity", None)
            if get_by_identity is not None:
                if tracking_data.has_training_ball():
                    training_metrics = get_by_identity("training")
                    if training_metrics:
                        print(f"   Training ball metrics: {len(training_metrics)} entries")
                        sample_keys = list(training_metrics.keys())[:3]
                        print(f"     Sample keys: {sample_keys}")
                    else:
                        print(f"   No training ball metrics found")

                if tracking_data.has_test_ball():
                    test_metrics = get_by_identity("test")
                    if test_metrics:
                        print(f"   Test ball metrics: {len(test_metrics)} entries")
                        sample_keys = list(test_metrics.keys())[:3]
                        print(f"     Sample keys: {sample_keys}")
                    else:
                        print(f"   No test ball metrics found")
            else:
                print(f"   ⚠️  get_metrics_by_ball_identity method not available")
        except Exception as e:
            print(f"   ⚠️  Error testing ball identity retrieval: {e}")

            # Test get_all_metrics_with_identity
            try:
                all_with_identity = getattr(metrics, "get_all_metrics_with_identity", None)
                if all_with_identity is not None:
                    result = all_with_identity()
                    if result:
                        print(f"   All metrics with identity: {len(result)} entries")
                    else:
                        print(f"   No metrics with identity information found")
                else:
                    print(f"   ⚠️  get_all_metrics_with_identity method not available")
            except Exception as e:
                print(f"   ⚠️  Error calling get_all_metrics_with_identity: {e}")
        else:
            print(f"   ⚠️  Ball identity retrieval methods not yet implemented")
            print(f"   📋 Available methods: {[m for m in dir(metrics) if not m.startswith('_')]}")

    except Exception as e:
        print(f"   ❌ Error in ball identity metrics: {e}")
        return False

    print(f"\n✅ Ball identity functionality test completed!")
    return True


def process_fly(
    fly_path, metrics_to_test, table_style="summary", show_json=False, test_f1=False, test_ball_identity=False
):
    """Process a single fly with comprehensive conditional processing tests."""

    # Test ball identity functionality if requested
    if test_f1 or test_ball_identity:
        test_ball_identity_functionality(fly_path)

        # If only testing ball identity, return here
        if (test_f1 or test_ball_identity) and not metrics_to_test:
            return

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

    # Test harmonized pause metrics if requested
    if "harmonized_test" in metrics_to_test:
        harmonized_results = test_harmonized_pause_metrics(fly_path)

        # If only testing harmonized metrics, return here
        if metrics_to_test == ["harmonized_test"]:
            return

    # Test updated pause metrics if requested
    if "updated_pause_test" in metrics_to_test:
        updated_pause_results = test_updated_pause_metrics(fly_path)

        # If only testing updated pause metrics, return here
        if metrics_to_test == ["updated_pause_test"]:
            return

    # Continue with standard metric testing if other metrics are specified
    standard_metrics = (
        [
            m
            for m in metrics_to_test
            if m not in ["conditional_test", "performance_test", "harmonized_test", "updated_pause_test"]
        ]
        if metrics_to_test
        else []
    )
    if standard_metrics:
        print(f"\n{'='*60}")
        print("📊 STANDARD METRICS TESTING")
        print(f"{'='*60}")

        # Load an example fly
        ExampleFly = Fly(fly_path, as_individual=True)

        # Initialize the BallPushingMetrics object
        metrics = BallPushingMetrics(ExampleFly.tracking_data)

        test_metrics(metrics, standard_metrics, ExampleFly.metadata.name, table_style=table_style, show_json=show_json)


def process_experiment(
    experiment_path,
    metrics_to_test,
    table_style="summary",
    show_json=False,
    max_flies=20,
    test_f1=False,
    test_ball_identity=False,
):
    """Process an experiment with optional conditional processing tests."""

    # Load the experiment
    ExampleExperiment = Experiment(experiment_path)

    if not ExampleExperiment.flies:
        print("No flies found in experiment!")
        return

    print(f"Testing metrics on experiment with {len(ExampleExperiment.flies)} flies")
    print(f"Experiment path: {experiment_path}")

    # Test ball identity functionality if requested
    if test_f1 or test_ball_identity:
        print(f"\n{'='*80}")
        print("🧪 EXPERIMENT-WIDE BALL IDENTITY TESTS")
        print(f"{'='*80}")

        # Test on first few flies to see ball identity patterns
        test_flies = ExampleExperiment.flies[: min(3, len(ExampleExperiment.flies))]

        for i, fly in enumerate(test_flies):
            print(f"\n🐛 Testing ball identity on fly {i+1}/{len(test_flies)}: {fly.metadata.name}")
            test_ball_identity_functionality(fly.directory)

        # If only testing ball identity, return here
        if (test_f1 or test_ball_identity) and not metrics_to_test:
            return

    # Handle harmonized metrics tests for experiments
    if "harmonized_test" in metrics_to_test:
        print(f"\n{'='*80}")
        print("🧪 EXPERIMENT-WIDE HARMONIZED METRICS TESTS")
        print(f"{'='*80}")

        # Test harmonized metrics on a subset of flies (first 3 or all if less than 3)
        test_flies = ExampleExperiment.flies[: min(3, len(ExampleExperiment.flies))]

        for i, fly in enumerate(test_flies):
            print(f"\n🐛 Testing harmonized metrics on fly {i+1}/{len(test_flies)}: {fly.metadata.name}")
            test_harmonized_pause_metrics(fly.directory)

        # If only testing harmonized metrics, return here
        if metrics_to_test == ["harmonized_test"]:
            return

    # Handle updated pause metrics tests for experiments
    if "updated_pause_test" in metrics_to_test:
        print(f"\n{'='*80}")
        print("🧪 EXPERIMENT-WIDE UPDATED PAUSE METRICS TESTS")
        print(f"{'='*80}")

        # Test updated pause metrics on a subset of flies (first 3 or all if less than 3)
        test_flies = ExampleExperiment.flies[: min(3, len(ExampleExperiment.flies))]

        for i, fly in enumerate(test_flies):
            print(f"\n🐛 Testing updated pause metrics on fly {i+1}/{len(test_flies)}: {fly.metadata.name}")
            test_updated_pause_metrics(fly.directory)

        # If only testing updated pause metrics, return here
        if metrics_to_test == ["updated_pause_test"]:
            return

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
    standard_metrics = [
        m
        for m in metrics_to_test
        if m not in ["conditional_test", "performance_test", "harmonized_test", "updated_pause_test"]
    ]
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
        generate_experiment_summary(all_results, experiment_path, standard_metrics, max_flies)


def test_metrics(metrics, metrics_to_test, fly_name, return_results=False, table_style="summary", show_json=False):
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
                # Check for F1 experiment with ball identities
                if hasattr(metrics.tracking_data, "has_training_ball") and hasattr(
                    metrics.tracking_data, "has_test_ball"
                ):
                    training_ball = metrics.tracking_data.has_training_ball()
                    test_ball = metrics.tracking_data.has_test_ball()

                    if training_ball or test_ball:
                        ball_results = {}
                        if training_ball:
                            training_idx = metrics.tracking_data.training_ball_idx
                            try:
                                ball_results["training"] = metrics.get_max_event(0, training_idx)
                            except Exception as e:
                                ball_results["training"] = f"Error: {e}"
                        if test_ball:
                            test_idx = metrics.tracking_data.test_ball_idx
                            try:
                                ball_results["test"] = metrics.get_max_event(0, test_idx)
                            except Exception as e:
                                ball_results["test"] = f"Error: {e}"
                        result = ball_results
                    else:
                        result = metrics.get_max_event(0, 0)
                else:
                    result = metrics.get_max_event(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "final_event" or metric_name == "get_final_event":
            try:
                # Check for F1 experiment with ball identities
                if hasattr(metrics.tracking_data, "has_training_ball") and hasattr(
                    metrics.tracking_data, "has_test_ball"
                ):
                    training_ball = metrics.tracking_data.has_training_ball()
                    test_ball = metrics.tracking_data.has_test_ball()

                    if training_ball or test_ball:
                        ball_results = {}
                        if training_ball:
                            training_idx = metrics.tracking_data.training_ball_idx
                            try:
                                ball_results["training"] = metrics.get_final_event(0, training_idx)
                            except Exception as e:
                                ball_results["training"] = f"Error: {e}"
                        if test_ball:
                            test_idx = metrics.tracking_data.test_ball_idx
                            try:
                                ball_results["test"] = metrics.get_final_event(0, test_idx)
                            except Exception as e:
                                ball_results["test"] = f"Error: {e}"
                        result = ball_results
                    else:
                        result = metrics.get_final_event(0, 0)
                else:
                    result = metrics.get_final_event(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "has_finished" or metric_name == "get_has_finished":
            try:
                # Check for F1 experiment with ball identities
                if hasattr(metrics.tracking_data, "has_training_ball") and hasattr(
                    metrics.tracking_data, "has_test_ball"
                ):
                    training_ball = metrics.tracking_data.has_training_ball()
                    test_ball = metrics.tracking_data.has_test_ball()

                    if training_ball or test_ball:
                        ball_results = {}
                        if training_ball:
                            training_idx = metrics.tracking_data.training_ball_idx
                            try:
                                ball_results["training"] = metrics.get_has_finished(0, training_idx)
                            except Exception as e:
                                ball_results["training"] = f"Error: {e}"
                        if test_ball:
                            test_idx = metrics.tracking_data.test_ball_idx
                            try:
                                ball_results["test"] = metrics.get_has_finished(0, test_idx)
                            except Exception as e:
                                ball_results["test"] = f"Error: {e}"
                        result = ball_results
                    else:
                        result = metrics.get_has_finished(0, 0)
                else:
                    result = metrics.get_has_finished(0, 0)
            except Exception as e:
                result = f"Error: {e}"

        elif metric_name == "has_major" or metric_name == "get_has_major":
            try:
                # Check for F1 experiment with ball identities
                if hasattr(metrics.tracking_data, "has_training_ball") and hasattr(
                    metrics.tracking_data, "has_test_ball"
                ):
                    training_ball = metrics.tracking_data.has_training_ball()
                    test_ball = metrics.tracking_data.has_test_ball()

                    if training_ball or test_ball:
                        ball_results = {}
                        if training_ball:
                            training_idx = metrics.tracking_data.training_ball_idx
                            try:
                                ball_results["training"] = metrics.get_has_major(0, training_idx)
                            except Exception as e:
                                ball_results["training"] = f"Error: {e}"
                        if test_ball:
                            test_idx = metrics.tracking_data.test_ball_idx
                            try:
                                ball_results["test"] = metrics.get_has_major(0, test_idx)
                            except Exception as e:
                                ball_results["test"] = f"Error: {e}"
                        result = ball_results
                    else:
                        result = metrics.get_has_major(0, 0)
                else:
                    result = metrics.get_has_major(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "has_significant" or metric_name == "get_has_significant":
            try:
                # Check for F1 experiment with ball identities
                if hasattr(metrics.tracking_data, "has_training_ball") and hasattr(
                    metrics.tracking_data, "has_test_ball"
                ):
                    training_ball = metrics.tracking_data.has_training_ball()
                    test_ball = metrics.tracking_data.has_test_ball()

                    if training_ball or test_ball:
                        ball_results = {}
                        if training_ball:
                            training_idx = metrics.tracking_data.training_ball_idx
                            try:
                                ball_results["training"] = metrics.get_has_significant(0, training_idx)
                            except Exception as e:
                                ball_results["training"] = f"Error: {e}"
                        if test_ball:
                            test_idx = metrics.tracking_data.test_ball_idx
                            try:
                                ball_results["test"] = metrics.get_has_significant(0, test_idx)
                            except Exception as e:
                                ball_results["test"] = f"Error: {e}"
                        result = ball_results
                    else:
                        result = metrics.get_has_significant(0, 0)
                else:
                    result = metrics.get_has_significant(0, 0)
            except Exception as e:
                result = f"Error: {e}"
        elif metric_name == "has_long_pauses":
            try:
                result = metrics.has_long_pauses(0)
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
            "compute_stop_metrics",
            "compute_long_pause_metrics",
            "compute_pauses_persistence",
            "compute_velocity_trend",
            "compute_median_stop_duration",
            "compute_nb_stops",
        ]:
            try:
                func = getattr(metrics, metric_name)
                result = func(0)
            except Exception as e:
                result = f"Error: {e}"

        # Handle metrics that are derived from other computation methods (not standalone methods)
        elif metric_name in [
            "nb_stops",
            "median_stop_duration",
            "total_stop_duration",
            "nb_pauses",
            "median_pause_duration",
            "total_pause_duration",
            "nb_long_pauses",
            "median_long_pause_duration",
            "total_long_pause_duration",
        ]:
            try:
                # These are not methods but results from compute_*_metrics methods
                if metric_name in ["nb_stops", "median_stop_duration", "total_stop_duration"]:
                    parent_result = metrics.compute_stop_metrics(0)
                    result = parent_result.get(metric_name, "Not available")
                elif metric_name in ["nb_pauses", "median_pause_duration", "total_pause_duration"]:
                    parent_result = metrics.compute_pause_metrics(0)
                    result = parent_result.get(metric_name, "Not available")
                elif metric_name in ["nb_long_pauses", "median_long_pause_duration", "total_long_pause_duration"]:
                    parent_result = metrics.compute_long_pause_metrics(0)
                    result = parent_result.get(metric_name, "Not available")
                else:
                    result = "Unknown derived metric"
            except Exception as e:
                result = f"Error: {e}"

        # Methods that need special handling (ball identity, multiple args, etc.)
        elif metric_name in [
            "get_all_ball_identities",
            "get_metrics_by_identity",
            "get_test_ball_metrics",
            "get_training_ball_metrics",
            "has_ball_identity",
            "get_raw_pauses",
            "is_metric_enabled",
        ]:
            try:
                if metric_name == "get_all_ball_identities":
                    result = "Skipped: Ball identity method, requires F1 experiment"
                elif metric_name == "get_metrics_by_identity":
                    result = "Skipped: Ball identity method, requires fly_idx and ball_identity args"
                elif metric_name == "get_test_ball_metrics":
                    result = "Skipped: Ball identity method, requires fly_idx"
                elif metric_name == "get_training_ball_metrics":
                    result = "Skipped: Ball identity method, requires fly_idx"
                elif metric_name == "has_ball_identity":
                    result = "Skipped: Ball identity method, requires fly_idx and ball_identity args"
                elif metric_name == "get_raw_pauses":
                    result = "Skipped: Raw pause method, requires fly_idx"
                elif metric_name == "is_metric_enabled":
                    result = "Skipped: Utility method, requires metric_name arg"
                else:
                    result = "Skipped: Special handling required"
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

                # Check if we have ball identities (F1 experiment)
                if hasattr(metrics.tracking_data, "has_training_ball") and hasattr(
                    metrics.tracking_data, "has_test_ball"
                ):
                    training_ball = metrics.tracking_data.has_training_ball()
                    test_ball = metrics.tracking_data.has_test_ball()

                    if training_ball or test_ball:
                        # F1 experiment with identified balls - get results for each ball
                        ball_results = {}

                        if training_ball:
                            training_idx = metrics.tracking_data.training_ball_idx
                            try:
                                training_result = func(0, training_idx)
                                ball_results["training"] = training_result
                            except Exception as e:
                                ball_results["training"] = f"Error: {e}"

                        if test_ball:
                            test_idx = metrics.tracking_data.test_ball_idx
                            try:
                                test_result = func(0, test_idx)
                                ball_results["test"] = test_result
                            except Exception as e:
                                ball_results["test"] = f"Error: {e}"

                        result = ball_results
                    else:
                        # Regular experiment - use ball index 0
                        result = func(0, 0)
                else:
                    # No ball identity information - use ball index 0
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

    if isinstance(serializable_results, dict):
        # 🎯 NEW TABLE-BASED OUTPUT
        if not return_results:
            # Show table in requested style
            print(create_metrics_table(serializable_results, fly_name, table_style))

            # Always show categorical breakdown if there are many metrics and style is summary
            if len(serializable_results) > 15 and table_style == "summary":
                print(create_metrics_table(serializable_results, fly_name, "categorical"))

            # Highlight new/important metrics (unless detailed view already shows everything)
            if table_style != "detailed":
                new_metrics = [
                    "compute_fraction_not_facing_ball",
                    "compute_flailing",
                    "compute_head_pushing_ratio",
                    "compute_median_head_ball_distance",
                    "compute_mean_head_ball_distance",
                    "compute_long_pause_metrics",
                    "compute_pauses_persistence",
                    "nb_long_pauses",
                    "pauses_persistence",
                    "has_finished",
                    "has_major",
                    "has_significant",
                ]

                tested_new_metrics = {k: v for k, v in serializable_results.items() if k in new_metrics}
                if tested_new_metrics:
                    print(f"\n🆕 HIGHLIGHTED NEW/IMPORTANT METRICS:")
                    print("-" * 60)
                    for metric, result in tested_new_metrics.items():
                        formatted_value = format_metric_value(result)
                        status = "✅" if not (isinstance(result, str) and "Error" in result) else "❌"
                        print(f"  {status} {metric:35} | {formatted_value}")

            # Optional: Show raw JSON for debugging (only if specifically requested)
            if show_json:
                print(f"\n� RAW JSON (debug view):")
                print("-" * 40)
                print(json.dumps(serializable_results, indent=2))

        # Save results
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        safe_fly_name = "".join(c for c in fly_name if c.isalnum() or c in ("-", "_")).rstrip()

        # Save JSON results
        filename = f"metrics_results_{safe_fly_name}.json"
        with open(output_dir / filename, "w") as f:
            json.dump(serializable_results, f, indent=4)

        # Save CSV table for easy analysis
        df_results = pd.DataFrame(
            [
                {
                    "Metric": k,
                    "Value": format_metric_value(v, max_width=50),
                    "Raw_Value": v,
                    "Category": categorize_metric(k),
                }
                for k, v in serializable_results.items()
            ]
        )

        csv_filename = f"metrics_table_{safe_fly_name}.csv"
        df_results.to_csv(output_dir / csv_filename, index=False)

        if not return_results:
            print(f"\n💾 Results saved:")
            print(f"  📊 JSON: {output_dir / filename}")
            print(f"  📈 CSV:  {output_dir / csv_filename}")
    else:
        print("Error: Results are not in expected dictionary format")
        print(serializable_results)
        serializable_results = {}

    # Return results if requested
    if return_results:
        return serializable_results


def generate_experiment_summary(all_results, experiment_path, metrics_to_test, max_flies=20):
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

    # 🎯 NEW: Show experiment-wide table
    print(create_experiment_metrics_table(all_results, max_flies=max_flies))

    # Calculate detailed summary statistics for each metric
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
        "compute_stop_metrics",
        "compute_pause_metrics",
        "compute_long_pause_metrics",
        "has_long_pauses",
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
            "compute_stop_metrics",
            "compute_pause_metrics",
            "compute_long_pause_metrics",
            "has_long_pauses",
            "compute_leg_visibility_ratio",
            "compute_median_head_ball_distance",
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


def test_harmonized_pause_metrics(fly_path):
    """Test the harmonized pause metrics naming convention and consistency."""
    print(f"\n{'='*80}")
    print("🎯 HARMONIZED PAUSE METRICS TEST")
    print(f"{'='*80}")

    # Load a test fly
    test_fly = Fly(fly_path, as_individual=True)
    print(f"Testing fly: {test_fly.metadata.name}")

    # Test the three harmonized pause metrics
    metrics = BallPushingMetrics(test_fly.tracking_data, compute_metrics_on_init=True)

    print(f"\n📊 Testing Harmonized Metrics Structure:")
    print("-" * 60)

    # Expected harmonized naming convention
    expected_metrics = {
        "stops": ["nb_stops", "median_stop_duration", "total_stop_duration"],
        "pauses": ["nb_pauses", "median_pause_duration", "total_pause_duration"],
        "long_pauses": ["nb_long_pauses", "median_long_pause_duration", "total_long_pause_duration"],
    }

    # Check if metrics exist in the computed results
    all_computed_metrics = set()
    for key, metrics_dict in metrics.metrics.items():
        if isinstance(metrics_dict, dict):
            all_computed_metrics.update(metrics_dict.keys())

    print(f"Total computed metrics: {len(all_computed_metrics)}")

    # Test each harmonized metric category
    results = {}
    for category, metric_names in expected_metrics.items():
        print(f"\n{category.upper()} (≥{2 if category != 'long_pauses' else 10}s threshold):")
        category_results = {}

        for metric_name in metric_names:
            if metric_name in all_computed_metrics:
                # Find the metric value in the results
                metric_value = None
                for key, metrics_dict in metrics.metrics.items():
                    if isinstance(metrics_dict, dict) and metric_name in metrics_dict:
                        metric_value = metrics_dict[metric_name]
                        break

                category_results[metric_name] = metric_value
                status = "✅ FOUND" if metric_value is not None else "❌ NULL"
                print(f"  {metric_name:30} | {format_metric_value(metric_value, max_width=12)} | {status}")
            else:
                category_results[metric_name] = None
                print(f"  {metric_name:30} | NOT COMPUTED | ❌ MISSING")

        results[category] = category_results

    print(f"\n🔍 CONSISTENCY CHECKS:")
    print("-" * 40)

    # Check if stops and pauses are consistent (both should use ≥2s threshold)
    stops_count = results["stops"].get("nb_stops")
    pauses_count = results["pauses"].get("nb_pauses")

    if stops_count is not None and pauses_count is not None:
        if abs(stops_count - pauses_count) < 0.1:  # Allow for small floating point differences
            print(f"✅ Stops vs Pauses consistency: PASSED ({stops_count} ≈ {pauses_count})")
        else:
            print(f"❌ Stops vs Pauses consistency: FAILED ({stops_count} != {pauses_count})")
    else:
        print(f"⚠️  Cannot check stops vs pauses consistency (missing data)")

    # Check naming convention compliance
    naming_check = True
    for category, metric_names in expected_metrics.items():
        for metric_name in metric_names:
            if results[category].get(metric_name) is None:
                naming_check = False
                break

    if naming_check:
        print(f"✅ Naming convention: PASSED - All harmonized metrics found")
    else:
        print(f"❌ Naming convention: FAILED - Some metrics missing")

    print(f"\n📈 SUMMARY:")
    print(f"  - Stops (≥2s): Short-term movement cessation")
    print(f"  - Pauses (≥2s): General movement cessation (should equal stops)")
    print(f"  - Long pauses (≥10s): Extended freezing behavior")
    print(f"  - All use consistent naming: nb_metric, median_metric_duration, total_metric_duration")

    # Test that old inconsistent metrics are removed
    old_metrics_to_check = ["pauses_persistence", "compute_nb_freeze"]
    print(f"\n🧹 CLEANUP CHECK:")
    print("-" * 25)

    for old_metric in old_metrics_to_check:
        if old_metric in all_computed_metrics:
            if old_metric == "compute_nb_freeze":
                print(f"⚠️  {old_metric}: Still present (should be renamed to nb_stops)")
            else:
                print(f"⚠️  {old_metric}: Still present (should be removed)")
        else:
            print(f"✅ {old_metric}: Correctly removed/renamed")

    print(f"\n✅ Harmonized pause metrics test completed!")
    return results


def test_updated_pause_metrics(fly_path):
    """Test the updated non-overlapping pause metrics with has_long_pauses."""
    print(f"\n{'='*80}")
    print("🎯 UPDATED PAUSE METRICS TEST")
    print(f"{'='*80}")

    # Load a test fly
    test_fly = Fly(fly_path, as_individual=True)
    print(f"Testing fly: {test_fly.metadata.name}")

    # Test the updated pause metrics with non-overlapping ranges
    metrics = BallPushingMetrics(test_fly.tracking_data, compute_metrics_on_init=True)

    print(f"\n📊 Testing Updated Pause Metrics Structure:")
    print("-" * 60)

    # Expected updated pause metrics
    expected_metrics = {
        "stops": {
            "range": "2s ≤ duration < 5s",
            "metrics": ["nb_stops", "median_stop_duration", "total_stop_duration"],
        },
        "pauses": {
            "range": "5s ≤ duration < 10s",
            "metrics": ["nb_pauses", "median_pause_duration", "total_pause_duration"],
        },
        "long_pauses": {
            "range": "duration ≥ 10s",
            "metrics": ["nb_long_pauses", "median_long_pause_duration", "total_long_pause_duration"],
        },
    }

    # Check if metrics exist in the computed results
    all_computed_metrics = set()
    for key, metrics_dict in metrics.metrics.items():
        if isinstance(metrics_dict, dict):
            all_computed_metrics.update(metrics_dict.keys())

    print(f"Total computed metrics: {len(all_computed_metrics)}")

    # Test each updated metric category
    results = {}
    for category, config in expected_metrics.items():
        print(f"\n{category.upper()} ({config['range']}):")
        category_results = {}

        for metric_name in config["metrics"]:
            if metric_name in all_computed_metrics:
                # Get the metric value from first available fly-ball pair
                for metrics_dict in metrics.metrics.values():
                    if isinstance(metrics_dict, dict) and metric_name in metrics_dict:
                        value = metrics_dict[metric_name]
                        category_results[metric_name] = value
                        status = "✅" if value is not None else "❌"
                        formatted_value = format_metric_value(value)
                        print(f"  {status} {metric_name}: {formatted_value}")
                        break
            else:
                category_results[metric_name] = "Missing"
                print(f"  ❌ {metric_name}: Missing from computed metrics")

        results[category] = category_results

    # Test the new has_long_pauses metric
    print(f"\n🔍 NEW BOOLEAN METRIC:")
    print("-" * 30)

    if "has_long_pauses" in all_computed_metrics:
        # Get the metric value from first available fly
        for metrics_dict in metrics.metrics.values():
            if isinstance(metrics_dict, dict) and "has_long_pauses" in metrics_dict:
                has_long_pauses_value = metrics_dict["has_long_pauses"]
                results["has_long_pauses"] = has_long_pauses_value
                status = "✅" if has_long_pauses_value is not None else "❌"
                formatted_value = (
                    "Yes"
                    if has_long_pauses_value == 1
                    else "No" if has_long_pauses_value == 0 else str(has_long_pauses_value)
                )
                print(f"  {status} has_long_pauses: {formatted_value}")
                break
    else:
        results["has_long_pauses"] = "Missing"
        print(f"  ❌ has_long_pauses: Missing from computed metrics")

    print(f"\n🔍 NON-OVERLAPPING RANGE VALIDATION:")
    print("-" * 40)

    # Check that the ranges are truly non-overlapping by comparing counts
    stops_count = results["stops"].get("nb_stops")
    pauses_count = results["pauses"].get("nb_pauses")
    long_pauses_count = results["long_pauses"].get("nb_long_pauses")

    if all(isinstance(x, (int, float)) and not np.isnan(x) for x in [stops_count, pauses_count, long_pauses_count]):
        total_categorized = stops_count + pauses_count + long_pauses_count
        print(f"✅ Range validation:")
        print(f"   Stops (2-5s): {stops_count}")
        print(f"   Pauses (5-10s): {pauses_count}")
        print(f"   Long pauses (≥10s): {long_pauses_count}")
        print(f"   Total categorized events: {total_categorized}")

        # Check if has_long_pauses is consistent with long_pauses_count
        has_long_pauses_val = results.get("has_long_pauses")
        if isinstance(has_long_pauses_val, (int, float)):
            expected_has_long = 1 if long_pauses_count > 0 else 0
            if has_long_pauses_val == expected_has_long:
                print(f"✅ has_long_pauses consistency: {has_long_pauses_val} matches count > 0: {expected_has_long}")
            else:
                print(f"❌ has_long_pauses inconsistency: {has_long_pauses_val} != expected {expected_has_long}")

    else:
        print(f"⚠️  Cannot validate ranges (some counts are missing or NaN)")

    print(f"\n📈 SUMMARY:")
    print(f"  - Stops (2-5s): Short-term movement cessation")
    print(f"  - Pauses (5-10s): Medium-term movement cessation")
    print(f"  - Long pauses (≥10s): Extended freezing behavior")
    print(f"  - has_long_pauses: Boolean indicator for presence of long pauses")
    print(f"  - All ranges are now NON-OVERLAPPING")

    print(f"\n✅ Updated pause metrics test completed!")
    return results


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
        - 'harmonized_test': Test harmonized pause metrics naming convention
        - 'updated_pause_test': Test updated non-overlapping pause metrics with has_long_pauses

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
    parser.add_argument(
        "--table-style",
        choices=["summary", "detailed", "categorical"],
        default="summary",
        help="Style of table output for single fly results (default: summary)",
    )
    parser.add_argument(
        "--max-experiment-flies",
        type=int,
        default=20,
        help="Maximum number of flies to show in experiment table (default: 20)",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Also show raw JSON output for debugging",
    )
    parser.add_argument(
        "--test-f1",
        action="store_true",
        help="Test F1 experiment ball identity functionality",
    )
    parser.add_argument(
        "--test-ball-identity",
        action="store_true",
        help="Test ball identity features (training/test ball metrics)",
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
        process_fly(data_path, args.metrics, args.table_style, args.show_json, args.test_f1, args.test_ball_identity)
    elif args.mode == "experiment":
        process_experiment(
            data_path,
            args.metrics,
            args.table_style,
            args.show_json,
            args.max_experiment_flies,
            args.test_f1,
            args.test_ball_identity,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'fly' or 'experiment'.")

    print(f"\n✅ Testing completed successfully!")
    print(f"\n💡 Tips for conditional processing:")
    print(f"   - Use 'conditional_test' to test the enabled_metrics functionality")
    print(f"   - Use predefined sets like 'basic_fast' for quick analysis")
    print(f"   - Use 'expensive_only' to test only computationally intensive metrics")
    print(f"   - Set fly.config.enabled_metrics in your code to control which metrics are computed")
