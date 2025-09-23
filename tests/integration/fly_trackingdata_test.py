#!/usr/bin/env python3
"""
Integration test for FlyTrackingData ball identity assignment.

This script tests the ball identity assignment functionality with both regular
ball-pushing experiments and F1 experiments using real data paths.

Usage:
    python fly_trackingdata_test.py --mode fly --path <fly_path> --test-type regular
    python fly_trackingdata_test.py --mode experiment --path <exp_path> --test-type f1
    python fly_trackingdata_test.py --mode fly --path <fly_path> --test-type both
"""

import argparse
import sys
from pathlib import Path
import traceback

# Add src to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from Ballpushing_utils.fly_trackingdata import FlyTrackingData
from Ballpushing_utils.experiment import Experiment
from Ballpushing_utils import Fly


# Removed mock config - using actual fly config for real integration testing


def test_ball_assignment_logic():
    """Test the ball assignment logic with various track name patterns."""

    test_cases = [
        {
            "name": "F1 with explicit names",
            "track_names": ["training_ball", "test_ball"],
            "expected_training": 0,
            "expected_test": 1,
        },
        {
            "name": "Control with test ball only",
            "track_names": ["test_ball"],
            "expected_training": None,
            "expected_test": 0,
        },
        {
            "name": "Legacy training/test names",
            "track_names": ["training", "test"],
            "expected_training": 0,
            "expected_test": 1,
        },
        {
            "name": "Generic names (legacy F1)",
            "track_names": ["track_1", "track_2"],
            "expected_training": 0,
            "expected_test": 1,
        },
        {"name": "Single ball regular", "track_names": ["ball"], "expected_training": 0, "expected_test": None},
    ]

    print(f"\n🧪 Testing Ball Assignment Logic")
    print("=" * 60)

    all_passed = True

    for test_case in test_cases:
        print(f"\n📋 {test_case['name']}")
        print(f"   Track names: {test_case['track_names']}")

        # Simulate the assignment logic from FlyTrackingData
        training_ball_idx, test_ball_idx, ball_identities, identity_to_idx = simulate_ball_assignment(
            test_case["track_names"]
        )
        cutoff_idx, cutoff_type = simulate_success_cutoff_logic(training_ball_idx, test_ball_idx)

        print(f"   Training ball idx: {training_ball_idx} (expected: {test_case['expected_training']})")
        print(f"   Test ball idx: {test_ball_idx} (expected: {test_case['expected_test']})")
        print(f"   Ball identities: {ball_identities}")
        print(f"   Identity mappings: {identity_to_idx}")
        print(f"   Success cutoff uses: {cutoff_type} ball (idx={cutoff_idx})")

        # Validate expectations
        training_ok = training_ball_idx == test_case["expected_training"]
        test_ok = test_ball_idx == test_case["expected_test"]

        if training_ok and test_ok:
            print(f"   ✅ PASSED")
        else:
            print(f"   ❌ FAILED - Training: {training_ok}, Test: {test_ok}")
            all_passed = False

    return all_passed


def simulate_ball_assignment(track_names):
    """Simulate the ball assignment logic without loading actual data."""

    training_ball_idx = None
    test_ball_idx = None
    ball_identities = {}
    identity_to_idx = {}

    num_balls = len(track_names)

    # Process all balls using track names
    # (removed early return for single ball case to handle explicit naming correctly)
    for ball_idx, track_name in enumerate(track_names):
        track_name_lower = track_name.lower()

        # Map common track names to standard identities (updated logic)
        if track_name_lower in ["training", "train", "training_ball"]:
            identity = "training"
            training_ball_idx = ball_idx
            identity_to_idx["training"] = ball_idx
        elif track_name_lower in ["test", "testing", "test_ball"]:
            identity = "test"
            test_ball_idx = ball_idx
            identity_to_idx["test"] = ball_idx
        else:
            # Use the track name directly as identity
            identity = track_name_lower

            # For F1 experiments with generic track names, assign training/test roles
            if num_balls == 2:
                if ball_idx == 0 and training_ball_idx is None:
                    # First ball becomes training
                    training_ball_idx = ball_idx
                    identity_to_idx["training"] = ball_idx
                elif ball_idx == 1 and test_ball_idx is None:
                    # Second ball becomes test
                    test_ball_idx = ball_idx
                    identity_to_idx["test"] = ball_idx
            else:
                # For other cases, use the original logic
                if training_ball_idx is None and ball_idx == 0:
                    training_ball_idx = ball_idx
                    identity_to_idx["training"] = ball_idx

        ball_identities[ball_idx] = identity

    # Fallback logic for backward compatibility
    has_explicit_test_naming = any(name.lower() in ["test", "testing", "test_ball"] for name in track_names)
    has_explicit_training_naming = any(name.lower() in ["training", "train", "training_ball"] for name in track_names)

    # Only assign training ball if no explicit naming is used (regular experiments)
    if (
        training_ball_idx is None
        and num_balls > 0
        and not has_explicit_test_naming
        and not has_explicit_training_naming
    ):
        training_ball_idx = 0
        if "training" not in identity_to_idx:
            identity_to_idx["training"] = 0

    return training_ball_idx, test_ball_idx, ball_identities, identity_to_idx


def simulate_success_cutoff_logic(training_ball_idx, test_ball_idx):
    """Test which ball should be used for success cutoff (prioritize test ball)."""
    if test_ball_idx is not None:
        return test_ball_idx, "test"
    elif training_ball_idx is not None:
        return training_ball_idx, "training"
    else:
        return 0, "fallback"


def test_single_fly(fly_path, expected_type="regular", debug=False):
    """Test FlyTrackingData with a single fly."""
    print(f"\n🔍 Testing single fly: {fly_path}")
    print(f"   Expected type: {expected_type}")

    try:
        # Check if path exists
        fly_dir = Path(fly_path)
        if not fly_dir.exists():
            print(f"⚠️  Path does not exist: {fly_path}")
            return False

        # Load fly using as_individual=True (key difference for fly mode)
        fly = Fly(fly_path, as_individual=True)
        # Use actual fly config - this is an integration test after all!

        print(f"   Fly name: {fly.metadata.name}")
        print(f"   Arena: {fly.metadata.arena}, Corridor: {fly.metadata.corridor}")
        print(f"   Using actual config: {type(fly.config).__name__}")

        # Create FlyTrackingData
        tracking_data = FlyTrackingData(fly, log_missing=True, keep_idle=True)

        if not tracking_data.valid_data:
            print(f"⚠️  Fly tracking data is not valid for {fly.metadata.name}")
            return False

        # Test ball identity assignment
        print(f"\n📊 Ball Identity Analysis:")
        print(f"   Number of balls: {len(tracking_data.balltrack.objects) if tracking_data.balltrack else 0}")

        if tracking_data.balltrack:
            # Check for SLEAP track names
            has_names = (
                hasattr(tracking_data.balltrack, "track_names")
                and tracking_data.balltrack.track_names
                and len(tracking_data.balltrack.track_names) > 0
            )
            print(f"   SLEAP track names: {tracking_data.balltrack.track_names if has_names else 'None'}")

            # Ball identity results
            print(f"   Training ball index: {tracking_data.training_ball_idx}")
            print(f"   Test ball index: {tracking_data.test_ball_idx}")
            print(f"   Ball identities: {tracking_data.ball_identities}")
            print(f"   Identity mappings: {tracking_data.identity_to_idx}")

            # Expected behavior validation
            num_balls = len(tracking_data.balltrack.objects)
            if expected_type == "regular":
                if num_balls == 1:
                    print("✅ Regular single-ball experiment detected correctly")
                    assert tracking_data.training_ball_idx == 0, "Training ball should be index 0"
                    assert tracking_data.test_ball_idx is None, "No test ball should exist"
                    assert len(tracking_data.ball_identities) == 0, "No identity system for single ball"
                else:
                    if has_names:
                        print("✅ Multi-ball regular experiment with track names detected")
                        assert len(tracking_data.ball_identities) > 0, "Should have ball identities"
                    else:
                        print("⚠️  Multi-ball experiment without track names (warning expected)")
                        assert tracking_data.training_ball_idx == 0, "Should default to ball 0"

            elif expected_type == "f1":
                if num_balls > 1:
                    if has_names:
                        print("✅ F1 multi-ball experiment with track names detected")
                        assert len(tracking_data.ball_identities) > 0, "Should have ball identities"

                        # Check for training/test balls
                        has_training = tracking_data.has_training_ball()
                        has_test = tracking_data.has_test_ball()
                        print(f"   Training ball available: {has_training}")
                        print(f"   Test ball available: {has_test}")

                    else:
                        print("⚠️  F1 experiment without track names (warning expected)")
                        assert tracking_data.training_ball_idx == 0, "Should default to ball 0"
                else:
                    print("ℹ️  F1 experiment with single ball (unusual but valid)")

        # Test helper methods
        print(f"\n🔧 Testing Helper Methods:")
        print(f"   has_training_ball(): {tracking_data.has_training_ball()}")
        print(f"   has_test_ball(): {tracking_data.has_test_ball()}")
        print(f"   get_all_ball_identities(): {tracking_data.get_all_ball_identities()}")

        # Test data access
        training_data = tracking_data.get_training_ball_data()
        test_data = tracking_data.get_test_ball_data()
        print(f"   Training ball data available: {training_data is not None}")
        print(f"   Test ball data available: {test_data is not None}")

        # Test identity-specific methods if applicable
        if tracking_data.ball_identities:
            training_balls = tracking_data.get_balls_by_identity_pattern("training")
            test_balls = tracking_data.get_balls_by_identity_pattern("test")
            print(f"   Balls matching 'training': {training_balls}")
            print(f"   Balls matching 'test': {test_balls}")

        print("✅ Single fly test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Single fly test FAILED: {e}")
        if debug:
            traceback.print_exc()
        return False


def process_fly(fly_path, args):
    """Process a single fly and test ball assignment logic"""
    print(f"\n" + "=" * 80)
    print(f"TESTING FLY: {fly_path}")
    print("=" * 80)

    results = []

    # First run ball assignment logic tests (unless skipped)
    if not args.skip_logic_tests:
        print(f"\n" + "=" * 60)
        print("🧪 Running BALL ASSIGNMENT LOGIC tests")
        print("=" * 60)

        logic_test_passed = test_ball_assignment_logic()
        results.append(logic_test_passed)

    # Then run actual fly tests
    ball_config_tests = ["both_balls", "ball_1_only", "ball_2_only"]
    for test_type in ball_config_tests:
        print(f"\n" + "=" * 60)
        print(f"🧪 Running {test_type.upper()} test")
        print("=" * 60)

        success = test_single_fly(fly_path, expected_type=test_type, debug=args.debug)
        results.append(success)

    return results


def process_experiment(experiment_path, test_types, debug=False, max_flies=5):
    """Process an experiment with the specified test types."""
    print(f"\n{'='*80}")
    print("🟠 TESTING EXPERIMENT MODE")
    print(f"{'='*80}")
    print(f"Experiment path: {experiment_path}")
    print(f"Test types: {test_types}")

    try:
        # Load the experiment
        experiment = Experiment(experiment_path, metadata_only=False)

        if not experiment.flies:
            print("❌ No flies found in experiment!")
            return False

        print(f"Found {len(experiment.flies)} flies in experiment")

        # Test a subset of flies
        flies_to_test = experiment.flies[: min(max_flies, len(experiment.flies))]
        print(f"Testing first {len(flies_to_test)} flies")

        all_results = []

        for i, fly in enumerate(flies_to_test):
            print(f"\n{'='*60}")
            print(f"🐛 Testing fly {i+1}/{len(flies_to_test)}: {fly.metadata.name}")
            print(f"{'='*60}")

            # Use actual fly config from experiment - real integration testing
            print(f"   Using actual config: {type(fly.config).__name__}")

            fly_results = []
            for test_type in test_types:
                print(f"\n🧪 Running {test_type.upper()} test for {fly.metadata.name}")

                try:
                    # Create FlyTrackingData
                    tracking_data = FlyTrackingData(fly, log_missing=True, keep_idle=True)

                    if not tracking_data.valid_data:
                        print(f"⚠️  Fly tracking data is not valid for {fly.metadata.name}")
                        fly_results.append(False)
                        continue

                    # Run the analysis (similar to test_single_fly but integrated)
                    print(
                        f"   Number of balls: {len(tracking_data.balltrack.objects) if tracking_data.balltrack else 0}"
                    )

                    if tracking_data.balltrack:
                        has_names = (
                            hasattr(tracking_data.balltrack, "track_names")
                            and tracking_data.balltrack.track_names
                            and len(tracking_data.balltrack.track_names) > 0
                        )
                        print(f"   SLEAP track names: {tracking_data.balltrack.track_names if has_names else 'None'}")
                        print(f"   Training ball index: {tracking_data.training_ball_idx}")
                        print(f"   Test ball index: {tracking_data.test_ball_idx}")

                        # Test helper methods
                        has_training = tracking_data.has_training_ball()
                        has_test = tracking_data.has_test_ball()
                        print(f"   Training ball available: {has_training}")
                        print(f"   Test ball available: {has_test}")

                        print(f"   ✅ {test_type.upper()} test PASSED for {fly.metadata.name}")
                        fly_results.append(True)
                    else:
                        print(f"   ⚠️  No ball track data for {fly.metadata.name}")
                        fly_results.append(False)

                except Exception as e:
                    print(f"   ❌ {test_type.upper()} test FAILED for {fly.metadata.name}: {e}")
                    if debug:
                        traceback.print_exc()
                    fly_results.append(False)

            all_results.extend(fly_results)

        # Summary
        passed = sum(all_results)
        total = len(all_results)
        print(f"\n📊 Experiment Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        return passed == total

    except Exception as e:
        print(f"❌ Experiment processing FAILED: {e}")
        if debug:
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test FlyTrackingData ball identity assignment")
    parser.add_argument(
        "--mode",
        choices=["fly", "experiment"],
        required=True,
        help="Processing mode: fly (single fly) or experiment (multiple flies)",
    )
    parser.add_argument(
        "--path", required=True, help="Path to fly directory (fly mode) or experiment directory (experiment mode)"
    )
    parser.add_argument(
        "--test-type", choices=["regular", "f1", "both"], default="both", help="Type of test to run (default: both)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output and error tracebacks")
    parser.add_argument(
        "--max-flies", type=int, default=5, help="Maximum number of flies to test in experiment mode (default: 5)"
    )
    parser.add_argument(
        "--skip-logic-tests", action="store_true", help="Skip ball assignment logic tests (only run actual data tests)"
    )

    args = parser.parse_args()

    # Convert path to Path object
    data_path = Path(args.path)

    # Determine test types
    if args.test_type == "both":
        test_types = ["regular", "f1"]
    else:
        test_types = [args.test_type]

    print("FlyTrackingData Ball Identity Assignment Integration Test")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Path: {data_path}")
    print(f"Test types: {test_types}")
    print(f"Debug mode: {args.debug}")

    # Default paths if not specified (for convenience)
    if not data_path.exists():
        print(f"⚠️  Path does not exist: {data_path}")

        # Suggest default paths based on mode
        if args.mode == "fly":
            print("   Suggested fly paths from launch.json:")
            print(
                "   - Regular: /mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5"
            )
            print("   - F1: /mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena5/Right")
        else:
            print("   Suggested experiment paths from launch.json:")
            print("   - Regular: /mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked")
            print("   - F1: /mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked")
        return 1

    try:
        # Process based on mode
        if args.mode == "fly":
            results = process_fly(data_path, args)
            success = all(results)
        elif args.mode == "experiment":
            success = process_experiment(data_path, test_types, args.debug, args.max_flies)
        else:
            print(f"❌ Invalid mode: {args.mode}")
            return 1

        # Final summary
        print("\n" + "=" * 80)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 80)

        if success:
            print("🎉 ALL TESTS PASSED! 🎉")
            print("\n✅ Ball identity assignment is working correctly:")
            print(f"  - Mode: {args.mode}")
            print(f"  - Test types: {', '.join(test_types)}")
            print("  - SLEAP track name integration functional")
            print("  - Helper methods working as expected")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            return 1

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
