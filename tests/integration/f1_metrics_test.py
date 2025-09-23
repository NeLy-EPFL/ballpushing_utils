#!/usr/bin/env python3
"""
F1 Experiment Data Processing Test

Test script for F1 experiments with dual-ball learning paradigm.
Based on the ballpushing_metrics.py test framework but adapted for F1-specific metrics.

Usage:
    python f1_metrics_test.py --mode fly --path /path/to/f1/fly --test comprehensive
    python f1_metrics_test.py --mode experiment --path /path/to/f1/experiment --test comprehensive
    python f1_metrics_test.py --mode experiment --path /path/to/f1/experiment --test metrics --max-flies 10
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import time
import sys
import traceback
from datetime import datetime

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from Ballpushing_utils import Fly, Experiment, BallPushingMetrics, F1Metrics


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


def get_f1_metric_sets():
    """Get predefined sets of F1-specific metrics for testing."""
    return {
        "f1_basic": [
            "adjusted_time",
            "direction_match",
        ],
        "f1_checkpoints": [
            "F1_checkpoints",
        ],
        "f1_ball_distances": [
            "training_ball_distances",
            "test_ball_distances",
        ],
        "f1_comprehensive": [
            "adjusted_time",
            "training_ball_distances",
            "test_ball_distances",
            "F1_checkpoints",
            "direction_match",
        ],
        "ballpushing_basic": [
            "nb_events",
            "max_event",
            "max_event_time",
            "final_event",
            "final_event_time",
            "chamber_time",
            "chamber_ratio",
        ],
        "ballpushing_comprehensive": [
            "nb_events",
            "max_event",
            "final_event",
            "has_finished",
            "has_major",
            "has_significant",
            "chamber_ratio",
            "distance_moved",
            "pulling_ratio",
        ],
        "combined_analysis": [
            # F1-specific
            "adjusted_time",
            "F1_checkpoints",
            "direction_match",
            # Standard ballpushing
            "nb_events",
            "max_event",
            "final_event",
            "chamber_ratio",
            "has_finished",
        ],
    }


def validate_f1_fly_directory(fly_path):
    """
    Validate that the F1 fly directory contains the expected files.

    Returns:
        dict: Validation results with success status and details
    """
    fly_path = Path(fly_path)
    validation = {
        "valid": True,
        "issues": [],
        "files_found": {},
        "expected_files": ["*.slp", "*ball*.slp", "*fly*.slp", "*.mp4"],
    }

    print(f"🔍 Validating F1 fly directory: {fly_path}")

    # Check if directory exists
    if not fly_path.exists():
        validation["valid"] = False
        validation["issues"].append(f"Directory does not exist: {fly_path}")
        return validation

    if not fly_path.is_dir():
        validation["valid"] = False
        validation["issues"].append(f"Path is not a directory: {fly_path}")
        return validation

    # Check for required files
    all_files = list(fly_path.glob("*"))
    validation["files_found"]["all_files"] = [f.name for f in all_files]

    # Check for SLEAP tracking files
    ball_files = list(fly_path.glob("*ball*.slp"))
    fly_files = list(fly_path.glob("*fly*.slp"))
    video_files = list(fly_path.glob("*.mp4"))

    validation["files_found"]["ball_tracking"] = [f.name for f in ball_files]
    validation["files_found"]["fly_tracking"] = [f.name for f in fly_files]
    validation["files_found"]["videos"] = [f.name for f in video_files]

    # Validate required files
    if not ball_files:
        validation["valid"] = False
        validation["issues"].append("No ball tracking files (*ball*.slp) found")

    if not fly_files:
        validation["valid"] = False
        validation["issues"].append("No fly tracking files (*fly*.slp) found")

    if not video_files:
        validation["issues"].append("No video files (*.mp4) found - may not be critical")

    # Check for expected F1 structure (single or dual balls are both valid)
    if len(ball_files) == 0:
        validation["valid"] = False
        validation["issues"].append("No ball tracking files found - F1 experiments require at least one ball")
    elif len(ball_files) > 2:
        validation["issues"].append(f"Multiple ball tracking files found ({len(ball_files)}) - may need investigation")
    # Note: Both single-ball and dual-ball F1 experiments are valid

    # Print validation results
    if validation["valid"]:
        print(f"✅ F1 directory validation passed")
        print(f"   Ball tracking files: {len(ball_files)}")
        print(f"   Fly tracking files: {len(fly_files)}")
        print(f"   Video files: {len(video_files)}")
    else:
        print(f"❌ F1 directory validation failed:")
        for issue in validation["issues"]:
            print(f"   - {issue}")

    if validation["issues"]:
        print(f"⚠️  Validation warnings:")
        for issue in validation["issues"]:
            print(f"   - {issue}")

    return validation


def test_f1_fly_initialization(fly_path):
    """
    Test F1 fly initialization and basic properties.

    Returns:
        dict: Test results
    """
    print(f"\n🧪 Testing F1 Fly Initialization")
    print(f"{'='*60}")

    results = {
        "initialization": {"success": False, "error": None},
        "tracking_data": {"success": False, "details": {}},
        "f1_compatibility": {"success": False, "details": {}},
    }

    try:
        # Initialize F1 fly
        print(f"📁 Loading F1 fly from: {fly_path}")
        start_time = time.time()

        f1_fly = Fly(fly_path, as_individual=True)

        load_time = time.time() - start_time
        print(f"✅ Fly loaded successfully in {load_time:.3f}s")

        results["initialization"]["success"] = True
        results["initialization"]["load_time"] = load_time

        # Test tracking data
        tracking_data = f1_fly.tracking_data

        if tracking_data is None:
            print(f"❌ Tracking data is None - file loading failed")
            results["tracking_data"]["success"] = False
            results["tracking_data"]["error"] = "tracking_data is None"

            # Try to diagnose the issue
            print(f"🔍 Diagnosing file loading issue...")
            print(f"   Fly directory: {fly_path}")
            print(f"   Expected fly name: {getattr(f1_fly, 'name', 'UNKNOWN')}")

            # List actual files in directory
            actual_files = list(Path(fly_path).glob("*.slp"))
            print(f"   Actual .slp files found: {[f.name for f in actual_files]}")

            # Check what files the Fly class expected
            fly_name = getattr(f1_fly, "name", "UNKNOWN")
            expected_ball_file = f"{fly_name}_tracked_ball.slp"
            expected_fly_file = f"{fly_name}_tracked_fly.slp"
            print(f"   Expected ball file: {expected_ball_file}")
            print(f"   Expected fly file: {expected_fly_file}")

            # Check if files exist with correct names
            ball_files = list(Path(fly_path).glob("*ball*.slp"))
            fly_files = list(Path(fly_path).glob("*fly*.slp"))
            print(f"   Ball files found: {[f.name for f in ball_files]}")
            print(f"   Fly files found: {[f.name for f in fly_files]}")

        elif not hasattr(tracking_data, "valid_data"):
            print(f"❌ Tracking data object has no 'valid_data' attribute")
            results["tracking_data"]["success"] = False
            results["tracking_data"]["error"] = "tracking_data missing valid_data attribute"

        elif not tracking_data.valid_data:
            print(f"❌ Tracking data is invalid")
            results["tracking_data"]["success"] = False
            results["tracking_data"]["error"] = "tracking_data.valid_data is False"

        else:
            print(f"✅ Tracking data is valid")
            results["tracking_data"]["success"] = True

            # Analyze tracking data structure
            details = results["tracking_data"]["details"]
            details["duration"] = tracking_data.duration if hasattr(tracking_data, "duration") else None
            details["start_position"] = (tracking_data.start_x, tracking_data.start_y)
            details["exit_time"] = tracking_data.exit_time

            # Ball tracking analysis
            if tracking_data.balltrack and tracking_data.balltrack.objects:
                details["num_balls"] = len(tracking_data.balltrack.objects)
                details["ball_info"] = []

                for i, ball_obj in enumerate(tracking_data.balltrack.objects):
                    ball_data = ball_obj.dataset
                    ball_info = {
                        "ball_index": i,
                        "data_length": len(ball_data),
                        "initial_position": (ball_data["x_centre"].iloc[0], ball_data["y_centre"].iloc[0]),
                        "has_euclidean_distance": "euclidean_distance" in ball_data.columns,
                    }
                    details["ball_info"].append(ball_info)

                print(f"🏀 Ball tracking: {details['num_balls']} balls detected")
                for ball_info in details["ball_info"]:
                    print(
                        f"   Ball {ball_info['ball_index']}: {ball_info['data_length']} frames, "
                        f"pos=({ball_info['initial_position'][0]:.1f}, {ball_info['initial_position'][1]:.1f})"
                    )
            else:
                details["num_balls"] = 0
                print(f"❌ No ball tracking data found")

            # Fly tracking analysis
            if tracking_data.flytrack and tracking_data.flytrack.objects:
                details["num_flies"] = len(tracking_data.flytrack.objects)
                fly_data = tracking_data.flytrack.objects[0].dataset
                details["fly_data_length"] = len(fly_data)
                print(f"🐛 Fly tracking: {details['num_flies']} flies, {details['fly_data_length']} frames")
            else:
                details["num_flies"] = 0
                print(f"❌ No fly tracking data found")

        # Test F1 compatibility
        if results["tracking_data"]["success"]:
            f1_details = results["f1_compatibility"]["details"]

            # Check ball setup for F1 (both single and dual ball are valid)
            num_balls = results["tracking_data"]["details"]["num_balls"]
            f1_details["num_balls"] = num_balls

            if num_balls >= 2:
                f1_details["ball_setup_type"] = "dual_ball"
                print(f"✅ F1 dual-ball setup detected ({num_balls} balls)")

                # Check ball positioning for F1 (training vs test ball)
                ball_positions = [info["initial_position"] for info in results["tracking_data"]["details"]["ball_info"]]
                start_x = results["tracking_data"]["details"]["start_position"][0]

                distances_from_fly = [abs(pos[0] - start_x) for pos in ball_positions]
                f1_details["ball_distances_from_start"] = distances_from_fly

                # Classify balls as training vs test based on distance
                close_balls = sum(1 for d in distances_from_fly if d < 100)
                far_balls = sum(1 for d in distances_from_fly if d >= 100)

                f1_details["potential_training_balls"] = close_balls
                f1_details["potential_test_balls"] = far_balls

                print(f"📊 Ball analysis: {close_balls} training candidates, {far_balls} test candidates")

                if close_balls >= 1 and far_balls >= 1:
                    f1_details["f1_layout_detected"] = "dual_ball_training_test"
                    results["f1_compatibility"]["success"] = True
                    print(f"✅ F1 dual-ball training/test layout detected")
                else:
                    f1_details["f1_layout_detected"] = "dual_ball_unclear"
                    results["f1_compatibility"]["success"] = True  # Still valid F1
                    print(f"✅ F1 dual-ball experiment (layout unclear but valid)")

            elif num_balls == 1:
                f1_details["ball_setup_type"] = "single_ball"
                f1_details["f1_layout_detected"] = "single_ball"
                results["f1_compatibility"]["success"] = True
                print(f"✅ F1 single-ball setup detected - valid F1 experiment")

                # For single ball, just record its position
                if results["tracking_data"]["details"]["ball_info"]:
                    ball_pos = results["tracking_data"]["details"]["ball_info"][0]["initial_position"]
                    start_x = results["tracking_data"]["details"]["start_position"][0]
                    distance_from_fly = abs(ball_pos[0] - start_x)
                    f1_details["ball_distance_from_start"] = distance_from_fly
                    print(f"📊 Ball distance from fly start: {distance_from_fly:.1f} pixels")
            else:
                f1_details["ball_setup_type"] = "no_balls"
                f1_details["f1_layout_detected"] = "invalid"
                print(f"❌ No balls detected - invalid F1 experiment")

        return results

    except Exception as e:
        print(f"❌ Error during F1 fly initialization: {e}")
        results["initialization"]["error"] = str(e)
        results["initialization"]["traceback"] = traceback.format_exc()
        return results


def test_f1_metrics_computation(fly_path):
    """
    Test F1-specific metrics computation.

    Returns:
        dict: F1 metrics test results
    """
    print(f"\n🧪 Testing F1 Metrics Computation")
    print(f"{'='*60}")

    results = {
        "f1_metrics": {"success": False, "metrics": {}, "error": None},
        "ballpushing_metrics": {"success": False, "metrics": {}, "error": None},
        "compatibility": {"success": False, "details": {}},
    }

    try:
        # Load fly
        f1_fly = Fly(fly_path, as_individual=True)

        if (
            f1_fly.tracking_data is None
            or not hasattr(f1_fly.tracking_data, "valid_data")
            or not f1_fly.tracking_data.valid_data
        ):
            results["f1_metrics"]["error"] = "Invalid tracking data"
            return results

        print(f"🔬 Computing F1-specific metrics...")

        # Check flyball_positions columns first
        if f1_fly.flyball_positions is not None:
            print(f"   Flyball columns: {list(f1_fly.flyball_positions.columns)}")
            print(f"   Flyball shape: {f1_fly.flyball_positions.shape}")
        else:
            print(f"   Flyball positions is None")

        # Test F1 metrics
        try:
            start_time = time.time()
            f1_metrics = F1Metrics(f1_fly.tracking_data)
            f1_compute_time = time.time() - start_time

            print(f"✅ F1 metrics computed in {f1_compute_time:.3f}s")
            results["f1_metrics"]["success"] = True
            results["f1_metrics"]["compute_time"] = f1_compute_time
            results["f1_metrics"]["metrics"] = convert_to_serializable(f1_metrics.metrics)

            # Analyze F1 metrics
            print(f"📊 F1 Metrics Analysis:")
            for metric_name, value in f1_metrics.metrics.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    print(f"   {metric_name}: DataFrame/Series with shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"   {metric_name}: Dict with {len(value)} keys")
                elif isinstance(value, (list, tuple)):
                    print(f"   {metric_name}: List/Tuple with {len(value)} items")
                else:
                    print(f"   {metric_name}: {type(value).__name__} = {value}")

        except Exception as e:
            print(f"❌ F1 metrics computation failed: {e}")
            results["f1_metrics"]["error"] = str(e)
            results["f1_metrics"]["traceback"] = traceback.format_exc()

        # Test standard ballpushing metrics for comparison
        try:
            print(f"\n🔬 Computing standard ballpushing metrics for comparison...")
            start_time = time.time()
            bp_metrics = BallPushingMetrics(f1_fly.tracking_data)
            bp_compute_time = time.time() - start_time

            print(f"✅ Ballpushing metrics computed in {bp_compute_time:.3f}s")
            results["ballpushing_metrics"]["success"] = True
            results["ballpushing_metrics"]["compute_time"] = bp_compute_time

            # Extract sample metrics for comparison
            sample_metrics = {}
            for key, metrics_dict in bp_metrics.metrics.items():
                if isinstance(metrics_dict, dict):
                    # Take a few key metrics
                    for metric in ["nb_events", "max_event", "chamber_ratio", "has_finished"]:
                        if metric in metrics_dict:
                            sample_key = f"{key}_{metric}"
                            sample_metrics[sample_key] = convert_to_serializable(metrics_dict[metric])

            results["ballpushing_metrics"]["sample_metrics"] = sample_metrics

            print(f"📊 Sample Ballpushing Metrics:")
            for key, value in sample_metrics.items():
                print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Ballpushing metrics computation failed: {e}")
            results["ballpushing_metrics"]["error"] = str(e)

        # Test compatibility between F1 and standard metrics
        if results["f1_metrics"]["success"] and results["ballpushing_metrics"]["success"]:
            print(f"\n🔗 Testing F1/Ballpushing compatibility...")
            compatibility = results["compatibility"]["details"]

            # Compare computation times
            f1_time = results["f1_metrics"]["compute_time"]
            bp_time = results["ballpushing_metrics"]["compute_time"]
            compatibility["relative_compute_time"] = f1_time / bp_time if bp_time > 0 else None

            print(f"⏱️  Relative computation time: F1={f1_time:.3f}s vs BP={bp_time:.3f}s")
            print(
                f"   Ratio: {compatibility['relative_compute_time']:.2f}x"
                if compatibility["relative_compute_time"]
                else ""
            )

            # Check for overlapping data requirements
            f1_metrics_dict = results["f1_metrics"]["metrics"]
            f1_requires_dual_balls = isinstance(f1_metrics_dict, dict) and "training_ball_distances" in f1_metrics_dict
            bp_works_with_dual = results["ballpushing_metrics"]["success"]

            compatibility["f1_dual_ball_support"] = f1_requires_dual_balls
            compatibility["bp_dual_ball_tolerance"] = bp_works_with_dual

            if bp_works_with_dual:  # F1 works regardless of single/dual ball setup
                results["compatibility"]["success"] = True
                print(f"✅ F1 and standard ballpushing metrics are compatible")
            else:
                print(f"⚠️  Potential compatibility issues detected")

        return results

    except Exception as e:
        print(f"❌ Error during metrics testing: {e}")
        results["f1_metrics"]["error"] = str(e)
        return results


def test_f1_specific_features(fly_path):
    """
    Test F1-specific features and functionality.

    Returns:
        dict: F1 feature test results
    """
    print(f"\n🧪 Testing F1-Specific Features")
    print(f"{'='*60}")

    results = {
        "checkpoint_analysis": {"success": False, "checkpoints": {}},
        "ball_classification": {"success": False, "classification": {}},
        "direction_matching": {"success": False, "direction_data": {}},
        "adjusted_timing": {"success": False, "timing_data": {}},
    }

    try:
        # Load fly and compute F1 metrics
        f1_fly = Fly(fly_path, as_individual=True)
        f1_metrics = F1Metrics(f1_fly.tracking_data)

        # Test checkpoint analysis
        print(f"🎯 Testing checkpoint analysis...")
        checkpoints = f1_metrics.F1_checkpoints
        if checkpoints:
            results["checkpoint_analysis"]["success"] = True
            results["checkpoint_analysis"]["checkpoints"] = convert_to_serializable(checkpoints)

            print(f"✅ F1 checkpoints computed:")
            for distance, time_val in checkpoints.items():
                status = f"{time_val:.3f}s" if time_val is not None else "Not reached"
                print(f"   {distance}px: {status}")
        else:
            print(f"⚠️  No checkpoints computed")

        # Test ball classification (training vs test)
        print(f"\n🏀 Testing ball classification...")
        training_data = f1_metrics.training_ball_distances
        test_data = f1_metrics.test_ball_distances

        classification = results["ball_classification"]["classification"]

        if training_data is not None:
            classification["training_ball_detected"] = True
            classification["training_data_length"] = len(training_data)
            classification["training_initial_pos"] = (
                float(training_data["x_centre"].iloc[0]),
                float(training_data["y_centre"].iloc[0]),
            )
            print(f"✅ Training ball detected: {len(training_data)} frames")
        else:
            classification["training_ball_detected"] = False
            print(f"⚠️  No training ball detected")

        if test_data is not None:
            classification["test_ball_detected"] = True
            classification["test_data_length"] = len(test_data)
            classification["test_initial_pos"] = (
                float(test_data["x_centre"].iloc[0]),
                float(test_data["y_centre"].iloc[0]),
            )
            print(f"✅ Test ball detected: {len(test_data)} frames")
        else:
            classification["test_ball_detected"] = False
            print(f"⚠️  No test ball detected")

        if classification["training_ball_detected"] and classification["test_ball_detected"]:
            results["ball_classification"]["success"] = True

            # Calculate separation between balls
            train_pos = classification["training_initial_pos"]
            test_pos = classification["test_initial_pos"]
            separation = np.sqrt((train_pos[0] - test_pos[0]) ** 2 + (train_pos[1] - test_pos[1]) ** 2)
            classification["ball_separation_px"] = float(separation)

            print(f"📏 Ball separation: {separation:.1f} pixels")

        # Test direction matching
        print(f"\n🧭 Testing direction matching...")
        direction_match = f1_metrics.direction_match

        if direction_match is not None:
            results["direction_matching"]["success"] = True
            results["direction_matching"]["direction_data"]["match_result"] = direction_match
            print(f"✅ Direction matching result: {direction_match}")
        else:
            print(f"⚠️  Direction matching not available")

        # Test adjusted timing
        print(f"\n⏰ Testing adjusted timing...")
        adjusted_time = f1_metrics.metrics.get("adjusted_time")

        if adjusted_time is not None:
            results["adjusted_timing"]["success"] = True
            timing_data = results["adjusted_timing"]["timing_data"]

            if hasattr(adjusted_time, "shape"):
                timing_data["data_type"] = "Series/Array"
                timing_data["length"] = len(adjusted_time)
                timing_data["range"] = (float(adjusted_time.min()), float(adjusted_time.max()))
            else:
                timing_data["data_type"] = type(adjusted_time).__name__
                timing_data["value"] = convert_to_serializable(adjusted_time)

            print(f"✅ Adjusted timing computed: {timing_data}")
        else:
            print(f"⚠️  Adjusted timing not available")

        return results

    except Exception as e:
        print(f"❌ Error during F1 features testing: {e}")
        for key in results:
            results[key]["error"] = str(e)
        return results


def run_comprehensive_f1_test(fly_path):
    """
    Run a comprehensive test of F1 experiment processing.

    Returns:
        dict: Complete test results
    """
    print(f"\n{'='*80}")
    print(f"🧪 COMPREHENSIVE F1 EXPERIMENT TEST")
    print(f"{'='*80}")
    print(f"📁 Test fly: {fly_path}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize comprehensive results
    comprehensive_results = {
        "test_info": {
            "fly_path": str(fly_path),
            "test_start_time": datetime.now().isoformat(),
            "test_duration": None,
        },
        "validation": {},
        "initialization": {},
        "metrics_computation": {},
        "f1_features": {},
        "overall_success": False,
        "summary": {},
    }

    test_start_time = time.time()

    try:
        # Step 1: Directory validation
        print(f"\n" + "=" * 60)
        print(f"STEP 1: F1 Directory Validation")
        print(f"=" * 60)

        validation_results = validate_f1_fly_directory(fly_path)
        comprehensive_results["validation"] = validation_results

        if not validation_results["valid"]:
            print(f"❌ Directory validation failed - stopping test")
            return comprehensive_results

        # Step 2: Fly initialization
        print(f"\n" + "=" * 60)
        print(f"STEP 2: F1 Fly Initialization")
        print(f"=" * 60)

        init_results = test_f1_fly_initialization(fly_path)
        comprehensive_results["initialization"] = init_results

        if not init_results["initialization"]["success"]:
            print(f"❌ Fly initialization failed - stopping test")
            return comprehensive_results

        # Step 3: Metrics computation
        print(f"\n" + "=" * 60)
        print(f"STEP 3: F1 Metrics Computation")
        print(f"=" * 60)

        metrics_results = test_f1_metrics_computation(fly_path)
        comprehensive_results["metrics_computation"] = metrics_results

        # Step 4: F1-specific features (continue even if metrics fail)
        print(f"\n" + "=" * 60)
        print(f"STEP 4: F1-Specific Features")
        print(f"=" * 60)

        if metrics_results["f1_metrics"]["success"]:
            features_results = test_f1_specific_features(fly_path)
            comprehensive_results["f1_features"] = features_results
        else:
            print(f"⚠️  Skipping F1 features test due to metrics computation failure")
            comprehensive_results["f1_features"] = {"skipped": True, "reason": "metrics_failed"}

        # Calculate test duration
        test_duration = time.time() - test_start_time
        comprehensive_results["test_info"]["test_duration"] = test_duration
        comprehensive_results["test_info"]["test_end_time"] = datetime.now().isoformat()

        # Determine overall success
        overall_success = (
            validation_results["valid"]
            and init_results["initialization"]["success"]
            and init_results["tracking_data"]["success"]
            and metrics_results["f1_metrics"]["success"]
        )

        comprehensive_results["overall_success"] = overall_success

        # Generate summary
        summary = comprehensive_results["summary"]
        summary["validation_passed"] = validation_results["valid"]
        summary["initialization_passed"] = init_results["initialization"]["success"]
        summary["f1_metrics_passed"] = metrics_results["f1_metrics"]["success"]
        summary["ballpushing_metrics_passed"] = metrics_results["ballpushing_metrics"]["success"]
        summary["f1_compatibility"] = metrics_results.get("compatibility", {}).get("success", False)

        if "f1_features" in comprehensive_results and not comprehensive_results["f1_features"].get("skipped"):
            features = comprehensive_results["f1_features"]
            summary["checkpoint_analysis_passed"] = features.get("checkpoint_analysis", {}).get("success", False)
            summary["ball_classification_passed"] = features.get("ball_classification", {}).get("success", False)

        summary["test_duration_seconds"] = test_duration
        summary["overall_success"] = overall_success

        # Print final summary
        print(f"\n" + "=" * 80)
        print(f"📊 COMPREHENSIVE TEST SUMMARY")
        print(f"=" * 80)
        print(f"🕐 Total duration: {test_duration:.3f} seconds")
        print(f"📁 Test fly: {Path(fly_path).name}")

        print(f"\n📋 Test Results:")
        for test_name, passed in summary.items():
            if test_name.endswith("_passed"):
                emoji = "✅" if passed else "❌"
                display_name = test_name.replace("_passed", "").replace("_", " ").title()
                print(f"   {emoji} {display_name}")

        overall_emoji = "✅" if overall_success else "❌"
        print(f"\n{overall_emoji} OVERALL RESULT: {'PASSED' if overall_success else 'FAILED'}")

        if overall_success:
            print(f"\n🎉 F1 experiment processing is working correctly!")
        else:
            print(f"\n⚠️  F1 experiment processing needs attention.")
            print(f"   Check individual test results for details.")

        return comprehensive_results

    except Exception as e:
        print(f"❌ Unexpected error during comprehensive test: {e}")
        comprehensive_results["test_info"]["unexpected_error"] = str(e)
        comprehensive_results["test_info"]["traceback"] = traceback.format_exc()
        return comprehensive_results


def save_test_results(results, output_dir=None):
    """Save test results to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fly_name = Path(results["test_info"]["fly_path"]).name
    filename = f"f1_test_results_{fly_name}_{timestamp}.json"

    output_file = output_dir / filename

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Test results saved to: {output_file}")
    return output_file


def process_f1_experiment(experiment_path, test_type="comprehensive", max_flies=None, save_results=True):
    """
    Process an entire F1 experiment with multiple flies.

    Parameters:
    -----------
    experiment_path : Path
        Path to the F1 experiment directory
    test_type : str
        Type of test to run on each fly
    max_flies : int, optional
        Maximum number of flies to process
    save_results : bool
        Whether to save individual fly results

    Returns:
    --------
    dict: Experiment-wide results
    """
    print(f"\n{'='*80}")
    print(f"🧪 F1 EXPERIMENT PROCESSING")
    print(f"{'='*80}")
    print(f"📁 Experiment path: {experiment_path}")
    print(f"🔬 Test type: {test_type}")

    experiment_results = {
        "experiment_info": {
            "experiment_path": str(experiment_path),
            "test_type": test_type,
            "start_time": datetime.now().isoformat(),
        },
        "fly_results": {},
        "experiment_summary": {},
        "overall_success": False,
    }

    # Find all fly directories in the experiment
    fly_directories = []

    # F1 experiments can have two structures:
    # 1. Direct fly directories: experiment_path/fly_name/
    # 2. Arena-based structure: experiment_path/arena1/Left/, experiment_path/arena1/Right/

    for item in experiment_path.iterdir():
        if item.is_dir():
            # Check if this is a direct fly directory (has tracking files)
            if any(item.glob("*.slp")) or any(item.glob("*.mp4")):
                fly_directories.append(item)
            else:
                # Check if this is an arena directory containing fly directories
                for sub_item in item.iterdir():
                    if sub_item.is_dir() and (any(sub_item.glob("*.slp")) or any(sub_item.glob("*.mp4"))):
                        fly_directories.append(sub_item)

    if not fly_directories:
        print(f"❌ No fly directories found in experiment path")
        experiment_results["error"] = "No fly directories found"
        return experiment_results

    # Limit number of flies if specified
    if max_flies:
        fly_directories = fly_directories[:max_flies]

    print(f"🐛 Found {len(fly_directories)} flies to process")
    if max_flies and len(fly_directories) == max_flies:
        print(f"   (Limited to first {max_flies} flies)")

    # Process each fly
    successful_flies = 0
    failed_flies = 0

    for i, fly_dir in enumerate(fly_directories, 1):
        fly_name = fly_dir.name
        print(f"\n" + "-" * 60)
        print(f"🐛 Processing fly {i}/{len(fly_directories)}: {fly_name}")
        print(f"-" * 60)

        try:
            # Run the specified test on this fly
            if test_type == "validation":
                fly_results = {"validation": validate_f1_fly_directory(fly_dir)}
            elif test_type == "initialization":
                fly_results = {"initialization": test_f1_fly_initialization(fly_dir)}
            elif test_type == "metrics":
                fly_results = {"metrics": test_f1_metrics_computation(fly_dir)}
            elif test_type == "f1_features":
                fly_results = {"f1_features": test_f1_specific_features(fly_dir)}
            else:  # comprehensive
                fly_results = run_comprehensive_f1_test(fly_dir)

            # Check if this fly was successful (depends on test type)
            if test_type == "validation":
                fly_success = fly_results.get("validation", {}).get("valid", False)
            elif test_type == "initialization":
                fly_success = fly_results.get("initialization", {}).get("initialization", {}).get("success", False)
            elif test_type == "metrics":
                fly_success = fly_results.get("metrics", {}).get("f1_metrics", {}).get("success", False)
            elif test_type == "f1_features":
                fly_success = fly_results.get("f1_features", {}).get("checkpoint_analysis", {}).get("success", False)
            else:  # comprehensive
                fly_success = fly_results.get("overall_success", False)

            if fly_success:
                successful_flies += 1
                print(f"✅ Fly {fly_name}: SUCCESS")
            else:
                failed_flies += 1
                print(f"❌ Fly {fly_name}: FAILED")

            experiment_results["fly_results"][fly_name] = fly_results

            # Save individual fly results if requested
            if save_results and test_type == "comprehensive":
                # Only save individual results for comprehensive tests (they have test_info)
                save_test_results(fly_results)

        except Exception as e:
            print(f"❌ Error processing fly {fly_name}: {e}")
            failed_flies += 1
            experiment_results["fly_results"][fly_name] = {"error": str(e), "overall_success": False}

    # Generate experiment summary
    print(f"\n{'='*80}")
    print(f"📊 F1 EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    experiment_summary = experiment_results["experiment_summary"]
    experiment_summary["total_flies"] = len(fly_directories)
    experiment_summary["successful_flies"] = successful_flies
    experiment_summary["failed_flies"] = failed_flies
    experiment_summary["success_rate"] = successful_flies / len(fly_directories) if fly_directories else 0

    print(f"🐛 Total flies processed: {len(fly_directories)}")
    print(f"✅ Successful flies: {successful_flies}")
    print(f"❌ Failed flies: {failed_flies}")
    print(f"📈 Success rate: {experiment_summary['success_rate']:.1%}")

    # Overall experiment success (>80% flies successful)
    experiment_results["overall_success"] = experiment_summary["success_rate"] >= 0.8

    # Analyze common patterns in successful/failed flies
    if experiment_results["fly_results"]:
        print(f"\n🔍 DETAILED ANALYSIS:")
        print(f"-" * 40)

        # Count specific test successes across all flies
        if test_type == "comprehensive":
            test_categories = ["validation", "initialization", "f1_metrics", "ballpushing_metrics"]
            for category in test_categories:
                success_count = 0
                total_count = 0
                for fly_name, fly_result in experiment_results["fly_results"].items():
                    if not fly_result.get("error"):
                        total_count += 1
                        # Check if this specific test passed
                        if category == "f1_metrics":
                            success = (
                                fly_result.get("metrics_computation", {}).get("f1_metrics", {}).get("success", False)
                            )
                        elif category == "ballpushing_metrics":
                            success = (
                                fly_result.get("metrics_computation", {})
                                .get("ballpushing_metrics", {})
                                .get("success", False)
                            )
                        else:
                            success = fly_result.get(category, {}).get(
                                "valid" if category == "validation" else "success", False
                            )

                        if success:
                            success_count += 1

                if total_count > 0:
                    print(
                        f"   {category.replace('_', ' ').title()}: {success_count}/{total_count} ({success_count/total_count:.1%})"
                    )

        # Identify flies with common issues
        common_errors = {}
        for fly_name, fly_result in experiment_results["fly_results"].items():
            if fly_result.get("error"):
                error_msg = fly_result["error"]
                if error_msg not in common_errors:
                    common_errors[error_msg] = []
                common_errors[error_msg].append(fly_name)

        if common_errors:
            print(f"\n⚠️  Common Issues:")
            for error, flies in common_errors.items():
                print(f"   '{error}': {len(flies)} flies")
                if len(flies) <= 3:
                    print(f"     Flies: {', '.join(flies)}")

    # Set experiment end time
    experiment_results["experiment_info"]["end_time"] = datetime.now().isoformat()
    experiment_results["experiment_info"]["duration_seconds"] = (
        datetime.fromisoformat(experiment_results["experiment_info"]["end_time"])
        - datetime.fromisoformat(experiment_results["experiment_info"]["start_time"])
    ).total_seconds()

    return experiment_results


def main():
    """Main function for F1 metrics testing."""
    parser = argparse.ArgumentParser(
        description="Test F1 experiment data processing and metrics computation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test a single F1 fly with comprehensive analysis
    python f1_metrics_test.py --mode fly --path /path/to/f1/fly --test comprehensive

    # Test entire F1 experiment with all flies
    python f1_metrics_test.py --mode experiment --path /path/to/f1/experiment --test comprehensive

    # Test F1-specific metrics only on experiment (first 10 flies)
    python f1_metrics_test.py --mode experiment --path /path/to/f1/experiment --test metrics --max-flies 10

    # Quick validation test on single fly
    python f1_metrics_test.py --mode fly --path /path/to/f1/fly --test validation
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["fly", "experiment"],
        required=True,
        help="Specify whether to process a single fly or an experiment.",
    )

    parser.add_argument("--path", required=True, help="Path to the F1 fly or experiment data.")

    parser.add_argument(
        "--test",
        choices=["validation", "initialization", "metrics", "f1_features", "comprehensive"],
        default="comprehensive",
        help="Type of test to run (default: comprehensive).",
    )

    parser.add_argument(
        "--max-flies", type=int, help="Maximum number of flies to process in experiment mode (default: all flies)."
    )

    parser.add_argument("--output-dir", help="Directory to save test results (default: tests/integration/outputs).")

    parser.add_argument("--save-results", action="store_true", help="Save detailed test results to JSON file.")

    args = parser.parse_args()

    # Convert path to Path object
    data_path = Path(args.path)

    print(f"🧪 F1 EXPERIMENT TESTING SUITE")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Path: {data_path}")
    print(f"Test type: {args.test}")
    if args.mode == "experiment" and args.max_flies:
        print(f"Max flies: {args.max_flies}")
    print(f"{'='*80}")

    # Initialize results variable
    results = None

    # Run tests based on mode and test type
    if args.mode == "fly":
        if args.test == "validation":
            results = {"validation": validate_f1_fly_directory(data_path)}
        elif args.test == "initialization":
            results = {"initialization": test_f1_fly_initialization(data_path)}
        elif args.test == "metrics":
            results = {"metrics": test_f1_metrics_computation(data_path)}
        elif args.test == "f1_features":
            results = {"f1_features": test_f1_specific_features(data_path)}
        else:  # comprehensive
            results = run_comprehensive_f1_test(data_path)

    elif args.mode == "experiment":
        results = process_f1_experiment(
            data_path, test_type=args.test, max_flies=args.max_flies, save_results=args.save_results
        )

    # Save results if requested
    if args.save_results and results:
        if args.mode == "experiment":
            # Save experiment-wide results
            output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs"
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = data_path.name
            filename = f"f1_experiment_results_{exp_name}_{timestamp}.json"

            output_file = output_dir / filename
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\n💾 Experiment results saved to: {output_file}")
        else:
            save_test_results(results, args.output_dir)

    # Return appropriate exit code
    if results and results.get("overall_success", False):
        print(f"\n✅ All tests completed successfully!")
        return 0
    else:
        print(f"\n❌ Some tests failed - check results for details")
        return 1


if __name__ == "__main__":
    exit(main())
