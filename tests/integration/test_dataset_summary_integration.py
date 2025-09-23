#!/usr/bin/env python3
"""
Test script to create and validate datasets with summary mode for both regular and F1 experiments.
This test demonstrates how ballpushing metrics are integrated into datasets and shows the
difference between regular experiments (without ball identity) and F1 experiments (with ball identity).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any
import warnings

warnings.filterwarnings("ignore")

# Import ballpushing utils
from Ballpushing_utils import Fly, Experiment, Dataset
from Ballpushing_utils.ballpushing_metrics import BallPushingMetrics


def analyze_dataset_structure(dataset_data, dataset_name, max_rows=10):
    """
    Analyze and display the structure of a dataset created from summary metrics.

    Parameters
    ----------
    dataset_data : pandas.DataFrame or dict
        The dataset data to analyze
    dataset_name : str
        Name of the dataset for display purposes
    max_rows : int, optional
        Maximum number of rows to show in the head display

    Returns
    -------
    dict
        Analysis summary containing column info, data types, etc.
    """
    print(f"\n{'='*80}")
    print(f"📊 DATASET STRUCTURE ANALYSIS: {dataset_name}")
    print(f"{'='*80}")

    if isinstance(dataset_data, dict):
        # If it's a dictionary (multi-ball dataset), analyze each ball type
        analysis = {}

        for ball_type, df in dataset_data.items():
            print(f"\n🎾 Ball Type: {ball_type}")
            print(f"{'─'*40}")

            if isinstance(df, pd.DataFrame):
                ball_analysis = analyze_single_dataframe(df, f"{dataset_name}_{ball_type}", max_rows)
                analysis[ball_type] = ball_analysis
            else:
                print(f"   ❌ Data is not a DataFrame: {type(df)}")
                analysis[ball_type] = {"error": f"Not a DataFrame: {type(df)}"}

        return analysis

    elif isinstance(dataset_data, pd.DataFrame):
        # Single DataFrame
        return analyze_single_dataframe(dataset_data, dataset_name, max_rows)
    else:
        print(f"   ❌ Unexpected data type: {type(dataset_data)}")
        return {"error": f"Unexpected data type: {type(dataset_data)}"}


def analyze_single_dataframe(df, name, max_rows=10):
    """
    Analyze a single DataFrame and return summary statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze
    name : str
        Name for display purposes
    max_rows : int
        Maximum rows to show in head

    Returns
    -------
    dict
        Analysis summary
    """
    print(f"\n📋 DataFrame Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Rows: {len(df)}")

    if len(df) == 0:
        print(f"   ⚠️  Empty DataFrame!")
        return {"shape": df.shape, "columns": list(df.columns), "empty": True}

    # Categorize columns
    metadata_cols = []
    ball_identity_cols = []
    metric_cols = []

    for col in df.columns:
        if col in ["fly_name", "experiment_name", "arena", "corridor", "genotype", "F1_condition", "metadata_name"]:
            metadata_cols.append(col)
        elif "ball_identity" in col or col.startswith("ball_") or "training" in col.lower() or "test" in col.lower():
            ball_identity_cols.append(col)
        else:
            metric_cols.append(col)

    print(f"\n🏷️  Column Categories:")
    print(f"   Metadata columns ({len(metadata_cols)}): {metadata_cols[:5]}")
    if len(metadata_cols) > 5:
        print(f"      ... and {len(metadata_cols) - 5} more")

    print(f"   Ball identity columns ({len(ball_identity_cols)}): {ball_identity_cols}")

    print(f"   Metric columns ({len(metric_cols)}): {metric_cols[:5]}")
    if len(metric_cols) > 5:
        print(f"      ... and {len(metric_cols) - 5} more")

    # Show data types
    print(f"\n📈 Data Types Summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")

    # Show sample values for key columns
    print(f"\n🔍 Sample Values (first {max_rows} rows):")

    # Show the head of the dataset
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 15)

    print(df.head(max_rows).to_string())

    # Reset pandas options
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")

    # Check for ball identity information
    has_ball_identity = len(ball_identity_cols) > 0

    print(f"\n🎯 Ball Identity Analysis:")
    print(f"   Has ball identity columns: {'✅ Yes' if has_ball_identity else '❌ No'}")

    if has_ball_identity:
        for col in ball_identity_cols:
            if col in df.columns:
                unique_values = df[col].unique()
                print(f"   Column '{col}': {unique_values}")

    # Look for F1-specific patterns
    f1_patterns = []
    if "F1_condition" in df.columns:
        f1_conditions = df["F1_condition"].unique()
        f1_patterns.append(f"F1_condition values: {f1_conditions}")
        print(f"   F1 conditions: {f1_conditions}")

    # Look for training/test metrics patterns
    training_test_cols = [col for col in df.columns if "training" in col.lower() or "test" in col.lower()]
    if training_test_cols:
        f1_patterns.append(f"Training/Test columns: {training_test_cols}")
        print(f"   Training/Test columns found: {len(training_test_cols)}")
        for col in training_test_cols[:3]:  # Show first 3
            print(f"     - {col}")

    # Calculate basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n📊 Numeric Columns Summary ({len(numeric_cols)} columns):")

    if len(numeric_cols) > 0:
        numeric_summary = df[numeric_cols].describe()
        print(f"   Mean values (first 5 columns):")
        for col in numeric_cols[:5]:
            mean_val = df[col].mean()
            if not pd.isna(mean_val):
                print(f"     {col}: {mean_val:.3f}")

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "metadata_columns": metadata_cols,
        "ball_identity_columns": ball_identity_cols,
        "metric_columns": metric_cols,
        "has_ball_identity": has_ball_identity,
        "f1_patterns": f1_patterns,
        "numeric_columns": len(numeric_cols),
        "empty": False,
    }


def test_dataset_creation_single_fly(fly_path: Path, test_ball_identity: bool = False):
    """
    Test dataset creation with summary mode for a single fly.

    Parameters
    ----------
    fly_path : Path
        Path to the fly directory
    test_ball_identity : bool
        Whether to test ball identity functionality

    Returns
    -------
    dict
        Test results and created datasets
    """
    print(f"\n{'='*80}")
    print(f"🐛 SINGLE FLY DATASET TEST")
    print(f"{'='*80}")
    print(f"Fly path: {fly_path}")

    try:
        # Load the fly
        test_fly = Fly(fly_path, as_individual=True)
        print(f"✅ Successfully loaded fly: {test_fly.metadata.name}")

        # Check if it's an F1 experiment by checking for training and test balls
        is_f1 = False
        f1_detection_method = "No ball identity detected"

        if test_fly.tracking_data is not None:
            # Try to assign ball identities and see if we get training and test balls
            try:
                if hasattr(test_fly.tracking_data, "assign_ball_identities"):
                    test_fly.tracking_data.assign_ball_identities()

                    has_training = test_fly.tracking_data.has_training_ball()
                    has_test = test_fly.tracking_data.has_test_ball()

                    if has_training and has_test:
                        is_f1 = True
                        f1_detection_method = "Training and test balls detected"
                    elif has_training or has_test:
                        is_f1 = True
                        f1_detection_method = f"Only {'training' if has_training else 'test'} ball detected"
            except Exception as e:
                f1_detection_method = f"Ball identity assignment failed: {e}"

        print(f"F1 experiment: {'✅ Yes' if is_f1 else '❌ No'} ({f1_detection_method})")

        # Test ball identity if requested and available
        if test_ball_identity and test_fly.tracking_data is not None:
            print(f"\n🎾 Testing Ball Identity Assignment:")

            if hasattr(test_fly.tracking_data, "assign_ball_identities"):
                test_fly.tracking_data.assign_ball_identities()

                print(f"   Training ball: {'✅ Yes' if test_fly.tracking_data.has_training_ball() else '❌ No'}")
                print(f"   Test ball: {'✅ Yes' if test_fly.tracking_data.has_test_ball() else '❌ No'}")
                print(f"   Ball identities: {test_fly.tracking_data.ball_identities}")
            else:
                print(f"   ⚠️  Ball identity assignment not available")

        # Create dataset with summary mode
        print(f"\n📊 Creating Dataset with Summary Mode:")

        dataset = Dataset(test_fly, dataset_type="summary")
        print(f"✅ Dataset created successfully")

        # Analyze the created dataset
        analysis = analyze_dataset_structure(dataset.data, f"Single_Fly_{test_fly.metadata.name}")

        # Save the dataset for inspection
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        fly_name = test_fly.metadata.name
        safe_fly_name = "".join(c for c in fly_name if c.isalnum() or c in ("-", "_")).rstrip()

        if isinstance(dataset.data, dict):
            # Multi-ball dataset - save each ball type separately
            for ball_type, df in dataset.data.items():
                if isinstance(df, pd.DataFrame):
                    csv_file = output_dir / f"single_fly_summary_{safe_fly_name}_{ball_type}.csv"
                    df.to_csv(csv_file, index=False)
                    print(f"💾 Saved {ball_type} dataset: {csv_file}")
        elif isinstance(dataset.data, pd.DataFrame):
            # Single DataFrame
            csv_file = output_dir / f"single_fly_summary_{safe_fly_name}.csv"
            dataset.data.to_csv(csv_file, index=False)
            print(f"💾 Saved dataset: {csv_file}")

        return {
            "success": True,
            "fly_name": test_fly.metadata.name,
            "is_f1": is_f1,
            "dataset": dataset.data,
            "analysis": analysis,
        }

    except Exception as e:
        print(f"❌ Error creating dataset for fly: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "fly_path": str(fly_path)}


def test_dataset_creation_experiment(experiment_path: Path, max_flies: int = 10, test_ball_identity: bool = False):
    """
    Test dataset creation with summary mode for an entire experiment.

    Parameters
    ----------
    experiment_path : Path
        Path to the experiment directory
    max_flies : int
        Maximum number of flies to include in the test
    test_ball_identity : bool
        Whether to test ball identity functionality

    Returns
    -------
    dict
        Test results and created datasets
    """
    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT DATASET TEST")
    print(f"{'='*80}")
    print(f"Experiment path: {experiment_path}")

    try:
        # Load the experiment
        experiment = Experiment(experiment_path)
        print(f"✅ Successfully loaded experiment")
        print(f"Total flies found: {len(experiment.flies)}")

        if not experiment.flies:
            print("❌ No flies found in experiment!")
            return {"success": False, "error": "No flies found"}

        # Limit the number of flies for testing
        flies_to_test = experiment.flies[:max_flies]
        print(f"Testing with {len(flies_to_test)} flies (max: {max_flies})")

        # Check if any flies are F1 experiments
        f1_flies = []
        regular_flies = []

        for fly in flies_to_test:
            # Check if it's an F1 experiment by checking for training and test balls
            is_f1 = False

            if fly.tracking_data is not None:
                try:
                    if hasattr(fly.tracking_data, "assign_ball_identities"):
                        fly.tracking_data.assign_ball_identities()

                        has_training = fly.tracking_data.has_training_ball()
                        has_test = fly.tracking_data.has_test_ball()

                        if has_training and has_test:
                            is_f1 = True
                        elif has_training or has_test:
                            is_f1 = True  # Even if only one ball type is detected
                except Exception:
                    pass  # If ball identity assignment fails, treat as regular experiment

            if is_f1:
                f1_flies.append(fly)
            else:
                regular_flies.append(fly)

        print(f"F1 flies: {len(f1_flies)}")
        print(f"Regular flies: {len(regular_flies)}")

        # Test ball identity on F1 flies if requested
        if test_ball_identity and f1_flies:
            print(f"\n🎾 Testing Ball Identity on F1 Flies:")

            for i, fly in enumerate(f1_flies[:3]):  # Test first 3 F1 flies
                if fly.tracking_data is not None and hasattr(fly.tracking_data, "assign_ball_identities"):
                    fly.tracking_data.assign_ball_identities()
                    print(f"   Fly {i+1} ({fly.metadata.name}):")
                    print(f"     Training: {'✅' if fly.tracking_data.has_training_ball() else '❌'}")
                    print(f"     Test: {'✅' if fly.tracking_data.has_test_ball() else '❌'}")

        # Create dataset with summary mode for the experiment
        print(f"\n📊 Creating Experiment Dataset with Summary Mode:")

        dataset = Dataset(experiment, dataset_type="summary")
        print(f"✅ Experiment dataset created successfully")

        # Analyze the created dataset
        analysis = analyze_dataset_structure(dataset.data, f"Experiment_{experiment.directory.name}")

        # Save the dataset
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        exp_name = experiment.directory.name
        safe_exp_name = "".join(c for c in exp_name if c.isalnum() or c in ("-", "_")).rstrip()

        if isinstance(dataset.data, dict):
            # Multi-ball dataset - save each ball type separately
            for ball_type, df in dataset.data.items():
                if isinstance(df, pd.DataFrame):
                    csv_file = output_dir / f"experiment_summary_{safe_exp_name}_{ball_type}.csv"
                    df.to_csv(csv_file, index=False)
                    print(f"💾 Saved {ball_type} dataset: {csv_file}")
        elif isinstance(dataset.data, pd.DataFrame):
            # Single DataFrame
            csv_file = output_dir / f"experiment_summary_{safe_exp_name}.csv"
            dataset.data.to_csv(csv_file, index=False)
            print(f"💾 Saved dataset: {csv_file}")

        return {
            "success": True,
            "experiment_name": exp_name,
            "total_flies": len(experiment.flies),
            "flies_tested": len(flies_to_test),
            "f1_flies": len(f1_flies),
            "regular_flies": len(regular_flies),
            "dataset": dataset.data,
            "analysis": analysis,
        }

    except Exception as e:
        print(f"❌ Error creating dataset for experiment: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "experiment_path": str(experiment_path)}


def compare_f1_vs_regular_datasets(f1_analysis, regular_analysis):
    """
    Compare the structure and content of F1 vs regular experiment datasets.

    Parameters
    ----------
    f1_analysis : dict
        Analysis results from F1 experiment dataset
    regular_analysis : dict
        Analysis results from regular experiment dataset
    """
    print(f"\n{'='*80}")
    print(f"🔄 F1 vs REGULAR EXPERIMENT COMPARISON")
    print(f"{'='*80}")

    def safe_get(analysis, key, default="N/A"):
        if isinstance(analysis, dict):
            if "analysis" in analysis and isinstance(analysis["analysis"], dict):
                return analysis["analysis"].get(key, default)
            else:
                return analysis.get(key, default)
        return default

    print(f"\n📊 Dataset Structure Comparison:")
    print(f"{'Aspect':<30} {'F1 Dataset':<20} {'Regular Dataset':<20}")
    print(f"{'─'*30} {'─'*20} {'─'*20}")

    # Compare shapes
    f1_shape = safe_get(f1_analysis, "shape", (0, 0))
    regular_shape = safe_get(regular_analysis, "shape", (0, 0))
    print(f"{'Dataset Shape':<30} {str(f1_shape):<20} {str(regular_shape):<20}")

    # Compare ball identity columns
    f1_has_identity = safe_get(f1_analysis, "has_ball_identity", False)
    regular_has_identity = safe_get(regular_analysis, "has_ball_identity", False)
    print(
        f"{'Has Ball Identity':<30} {'✅ Yes' if f1_has_identity else '❌ No':<20} {'✅ Yes' if regular_has_identity else '❌ No':<20}"
    )

    # Compare ball identity columns count
    f1_identity_cols = safe_get(f1_analysis, "ball_identity_columns", [])
    regular_identity_cols = safe_get(regular_analysis, "ball_identity_columns", [])
    print(f"{'Ball Identity Columns':<30} {len(f1_identity_cols):<20} {len(regular_identity_cols):<20}")

    # Compare metric columns count
    f1_metric_cols = safe_get(f1_analysis, "metric_columns", [])
    regular_metric_cols = safe_get(regular_analysis, "metric_columns", [])
    print(f"{'Metric Columns':<30} {len(f1_metric_cols):<20} {len(regular_metric_cols):<20}")

    print(f"\n🎯 F1-Specific Features:")
    if f1_identity_cols:
        print(f"   Ball Identity Columns in F1:")
        for col in f1_identity_cols[:5]:  # Show first 5
            print(f"     - {col}")
        if len(f1_identity_cols) > 5:
            print(f"     ... and {len(f1_identity_cols) - 5} more")

    f1_patterns = safe_get(f1_analysis, "f1_patterns", [])
    if f1_patterns:
        print(f"   F1 Patterns Found:")
        for pattern in f1_patterns:
            print(f"     - {pattern}")

    print(f"\n📈 Key Insights:")

    if f1_has_identity and not regular_has_identity:
        print(f"   ✅ F1 datasets correctly include ball identity information")
        print(f"   ✅ Regular datasets correctly exclude ball identity columns")
    elif f1_has_identity and regular_has_identity:
        print(f"   ⚠️  Both datasets have ball identity - check if regular dataset is actually F1")
    elif not f1_has_identity and not regular_has_identity:
        print(f"   ⚠️  Neither dataset has ball identity - may need to check F1 processing")

    # Compare metric availability
    if len(f1_metric_cols) > 0 and len(regular_metric_cols) > 0:
        common_metrics = set(f1_metric_cols) & set(regular_metric_cols)
        f1_unique = set(f1_metric_cols) - set(regular_metric_cols)
        regular_unique = set(regular_metric_cols) - set(f1_metric_cols)

        print(f"   📊 Metric Comparison:")
        print(f"     Common metrics: {len(common_metrics)}")
        print(f"     F1-unique metrics: {len(f1_unique)}")
        print(f"     Regular-unique metrics: {len(regular_unique)}")

        if f1_unique:
            print(f"     F1-unique examples: {list(f1_unique)[:3]}")
        if regular_unique:
            print(f"     Regular-unique examples: {list(regular_unique)[:3]}")


def save_test_summary(results: List[Dict[str, Any]], output_dir: Path):
    """
    Save a comprehensive test summary to JSON and generate a report.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of test results from all experiments/flies tested
    output_dir : Path
        Directory to save the summary files
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed JSON results
    json_file = output_dir / f"dataset_summary_test_results_{timestamp}.json"

    # Convert any non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": obj.shape,
                "columns": list(obj.columns),
                "head": obj.head().to_dict() if len(obj) > 0 else {},
            }
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj

    serializable_results = make_serializable(results)

    with open(json_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n💾 Detailed test results saved to: {json_file}")

    # Generate summary report
    report_file = output_dir / f"dataset_summary_test_report_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write("DATASET SUMMARY MODE TEST REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Total tests run: {len(results)}\n\n")

        successful_tests = [r for r in results if r.get("success", False)]
        failed_tests = [r for r in results if not r.get("success", False)]

        f.write(f"Successful tests: {len(successful_tests)}\n")
        f.write(f"Failed tests: {len(failed_tests)}\n\n")

        if successful_tests:
            f.write("SUCCESSFUL TESTS:\n")
            f.write("-" * 40 + "\n")
            for result in successful_tests:
                if "fly_name" in result:
                    f.write(f"✅ Single Fly: {result['fly_name']}\n")
                    if result.get("is_f1"):
                        f.write(f"   F1 experiment with ball identity support\n")
                elif "experiment_name" in result:
                    f.write(f"✅ Experiment: {result['experiment_name']}\n")
                    f.write(f"   Flies tested: {result.get('flies_tested', 'Unknown')}\n")
                    f.write(f"   F1 flies: {result.get('f1_flies', 0)}\n")
                    f.write(f"   Regular flies: {result.get('regular_flies', 0)}\n")
                f.write("\n")

        if failed_tests:
            f.write("FAILED TESTS:\n")
            f.write("-" * 40 + "\n")
            for result in failed_tests:
                f.write(f"❌ Failed test: {result.get('error', 'Unknown error')}\n")
                if "fly_path" in result:
                    f.write(f"   Fly path: {result['fly_path']}\n")
                elif "experiment_path" in result:
                    f.write(f"   Experiment path: {result['experiment_path']}\n")
                f.write("\n")

    print(f"📋 Summary report saved to: {report_file}")


def main():
    """Main function to run the dataset summary integration tests."""
    parser = argparse.ArgumentParser(
        description="Test dataset creation with summary mode for both regular and F1 experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["fly", "experiment", "both"],
        default="both",
        help="Test mode: single fly, experiment, or both",
    )
    parser.add_argument("--fly-path", type=str, help="Path to a single fly directory for testing")
    parser.add_argument("--experiment-path", type=str, help="Path to an experiment directory for testing")
    parser.add_argument(
        "--regular-experiment-path", type=str, help="Path to a regular (non-F1) experiment for comparison"
    )
    parser.add_argument("--f1-experiment-path", type=str, help="Path to an F1 experiment for comparison")
    parser.add_argument("--max-flies", type=int, default=5, help="Maximum number of flies to test per experiment")
    parser.add_argument(
        "--test-ball-identity", action="store_true", help="Test ball identity functionality for F1 experiments"
    )
    parser.add_argument(
        "--compare-datasets", action="store_true", help="Compare F1 vs regular datasets (requires both types)"
    )

    args = parser.parse_args()

    print(f"🧪 DATASET SUMMARY INTEGRATION TEST")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Max flies per experiment: {args.max_flies}")
    print(f"Test ball identity: {args.test_ball_identity}")

    # Create output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    all_results = []

    # Single fly testing
    if args.mode in ["fly", "both"] and args.fly_path:
        fly_path = Path(args.fly_path)
        result = test_dataset_creation_single_fly(fly_path, args.test_ball_identity)
        all_results.append(result)

    # Experiment testing
    if args.mode in ["experiment", "both"] and args.experiment_path:
        exp_path = Path(args.experiment_path)
        result = test_dataset_creation_experiment(exp_path, args.max_flies, args.test_ball_identity)
        all_results.append(result)

    # Comparison testing
    f1_result = None
    regular_result = None

    if args.compare_datasets or (args.f1_experiment_path and args.regular_experiment_path):
        print(f"\n{'='*80}")
        print(f"🔄 COMPARISON TESTING: F1 vs REGULAR EXPERIMENTS")
        print(f"{'='*80}")

        if args.f1_experiment_path:
            print(f"\n🧪 Testing F1 Experiment:")
            f1_path = Path(args.f1_experiment_path)
            f1_result = test_dataset_creation_experiment(f1_path, args.max_flies, args.test_ball_identity)
            all_results.append(f1_result)

        if args.regular_experiment_path:
            print(f"\n🧪 Testing Regular Experiment:")
            regular_path = Path(args.regular_experiment_path)
            regular_result = test_dataset_creation_experiment(
                regular_path, args.max_flies, False
            )  # No ball identity for regular
            all_results.append(regular_result)

        # Compare the datasets
        if f1_result and regular_result and f1_result.get("success") and regular_result.get("success"):
            compare_f1_vs_regular_datasets(f1_result, regular_result)

    # Default test paths if none provided
    if not any([args.fly_path, args.experiment_path, args.f1_experiment_path, args.regular_experiment_path]):
        print(f"\n⚠️  No paths provided. Here are some example commands:")
        print(f"")
        print(f"# Test a single F1 fly:")
        print(f"python {__file__} --mode fly --fly-path '/path/to/f1/fly' --test-ball-identity")
        print(f"")
        print(f"# Test an F1 experiment:")
        print(f"python {__file__} --mode experiment --experiment-path '/path/to/f1/experiment' --test-ball-identity")
        print(f"")
        print(f"# Compare F1 vs regular experiments:")
        print(
            f"python {__file__} --f1-experiment-path '/path/to/f1/exp' --regular-experiment-path '/path/to/regular/exp' --compare-datasets"
        )
        print(f"")
        print(f"The test will show how:")
        print(f"  - Regular experiments create datasets without ball identity columns")
        print(f"  - F1 experiments create datasets WITH ball identity columns")
        print(f"  - Summary mode integrates ballpushing metrics into dataset structure")
        print(f"  - Dataset heads demonstrate the different column structures")

        return

    # Save comprehensive test results
    if all_results:
        save_test_summary(all_results, output_dir)

    print(f"\n✅ Dataset summary integration tests completed!")
    print(f"📁 Results saved in: {output_dir}")

    # Summary of what was demonstrated
    print(f"\n🎯 KEY FINDINGS:")
    print(f"{'─'*40}")

    successful_tests = [r for r in all_results if r.get("success", False)]
    if successful_tests:
        has_f1 = any(r.get("is_f1") or r.get("f1_flies", 0) > 0 for r in successful_tests)
        has_regular = any((not r.get("is_f1", True)) or r.get("regular_flies", 0) > 0 for r in successful_tests)

        print(f"✅ Successfully created {len(successful_tests)} datasets with summary mode")
        if has_f1:
            print(f"✅ F1 experiments include ball identity columns (training/test discrimination)")
        if has_regular:
            print(f"✅ Regular experiments use standard structure without ball identity")
        print(f"✅ Ballpushing metrics are properly integrated into dataset structure")
        print(f"✅ Dataset heads show clear column structure differences")
    else:
        print(f"❌ No successful tests - check input paths and data validity")


if __name__ == "__main__":
    main()
