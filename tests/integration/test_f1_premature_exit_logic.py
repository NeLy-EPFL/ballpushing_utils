#!/usr/bin/env python3
"""
F1 Premature Exit Detection Test

Specific test script to validate the F1 premature exit detection logic using
manually verified examples of flies that did and did not exit prematurely.

This test uses real data to verify that:
1. Manually detected premature exits are correctly flagged by the algorithm
2. Normal flies (non-premature exits) are correctly kept
3. The 55-minute threshold is properly applied
4. The exit detection logic (100px x-movement) works correctly

Usage:
    python test_f1_premature_exit_logic.py
    python test_f1_premature_exit_logic.py --verbose
    python test_f1_premature_exit_logic.py --debug
"""

import sys
from pathlib import Path
import argparse
import time
from datetime import datetime
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from Ballpushing_utils import Fly


class F1PrematureExitTester:
    """Test class for F1 premature exit detection."""

    def __init__(self, verbose=False, debug=False):
        self.verbose = verbose
        self.debug = debug

        # Test cases with manual verification - focused on newer experiments (2509/2510)
        self.test_cases = {
            "premature_exits": {
                # Known premature exits (manually verified)
                "arena3_right_250904": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena3/Right/",
                "arena7_right_250905": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250905_F1_New_Videos_Checked/arena7/Right/",
            },
            "normal_flies": {
                # Normal flies from the same newer experiments - should NOT be flagged as premature exit
                "arena1_left_250904": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena1/Left/",
                "arena2_right_250904": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena2/Right/",
                "arena4_left_250904": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena4/Left/",
                "arena5_right_250904": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena5/Right/",
                "arena6_left_250904": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena6/Left/",
                "arena1_right_250905": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250905_F1_New_Videos_Checked/arena1/Right/",
                "arena2_left_250905": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250905_F1_New_Videos_Checked/arena2/Left/",
                "arena3_left_250905": "/mnt/upramdya_data/MD/F1_Tracks/Videos/250905_F1_New_Videos_Checked/arena3/Left/",
            },
        }

        self.results = {
            "premature_exits": {},
            "normal_flies": {},
            "summary": {
                "total_tested": 0,
                "correct_detections": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "errors": 0,
            },
        }

    def log(self, message, force=False):
        """Log message if verbose mode is enabled."""
        if self.verbose or force:
            print(message)

    def test_single_fly(self, fly_path, expected_premature, fly_name):
        """
        Test a single fly for premature exit detection.

        Args:
            fly_path (str): Path to the fly directory
            expected_premature (bool): Whether this fly is expected to be flagged as premature exit
            fly_name (str): Name identifier for this fly

        Returns:
            dict: Test results for this fly
        """
        fly_path = Path(fly_path)

        self.log(f"\n{'='*60}")
        self.log(f"🧪 Testing Fly: {fly_name}")
        self.log(f"📁 Path: {fly_path}")
        self.log(f"🎯 Expected: {'PREMATURE EXIT' if expected_premature else 'NORMAL FLY'}")
        self.log(f"{'='*60}")

        test_result = {
            "fly_name": fly_name,
            "fly_path": str(fly_path),
            "expected_premature": expected_premature,
            "actual_premature": None,
            "exit_time": None,
            "exit_time_minutes": None,
            "valid_data": None,
            "success": False,
            "correct_detection": False,
            "error": None,
            "test_details": {},
        }

        try:
            # Check if fly directory exists
            if not fly_path.exists():
                raise FileNotFoundError(f"Fly directory does not exist: {fly_path}")

            # Load the fly with F1 configuration
            self.log("📥 Loading fly data...")
            start_time = time.time()

            try:
                # Create fly with F1 experiment type
                fly = Fly(fly_path, as_individual=True)
            except Exception as fly_creation_error:
                raise ValueError(f"Failed to create Fly object: {fly_creation_error}")

            # Ensure F1 experiment type is set
            if hasattr(fly, "config"):
                fly.config.experiment_type = "F1"
                if self.debug:
                    fly.config.debugging = True

            load_time = time.time() - start_time
            self.log(f"✅ Fly loaded in {load_time:.3f}s")

            # Get tracking data - handle case where tracking_data is None
            try:
                tracking_data = fly.tracking_data
            except Exception as tracking_error:
                raise ValueError(f"Failed to access tracking_data property: {tracking_error}")

            if tracking_data is None:
                raise ValueError("Tracking data is None - fly.tracking_data returned None")

            # Check if tracking data is valid
            if not hasattr(tracking_data, "valid_data"):
                raise ValueError("Tracking data missing valid_data attribute")

            test_result["valid_data"] = tracking_data.valid_data

            if not tracking_data.valid_data:
                # Provide detailed information about why data is invalid
                error_details = []

                # Check various potential issues
                if tracking_data.flytrack is None:
                    error_details.append("flytrack is None")
                if tracking_data.balltrack is None:
                    error_details.append("balltrack is None")
                if tracking_data.skeletontrack is None:
                    error_details.append("skeletontrack is None")

                error_msg = f"Invalid tracking data - {'; '.join(error_details) if error_details else 'unknown reason'}"
                self.log(f"⚠️  {error_msg}")

                # Still try to get exit time if possible (for debugging)
                try:
                    exit_time = tracking_data.f1_exit_time if hasattr(tracking_data, "f1_exit_time") else None
                    test_result["exit_time"] = exit_time
                    if exit_time:
                        test_result["exit_time_minutes"] = exit_time / 60
                        self.log(f"🚪 Exit time (despite invalid data): {exit_time:.2f}s ({exit_time/60:.1f} minutes)")
                except Exception as e:
                    self.log(f"   Could not get exit time: {e}")

                # Don't test premature exit logic if data is invalid
                test_result["actual_premature"] = None
                test_result["correct_detection"] = False
                raise ValueError(error_msg)

            # Get exit time
            exit_time = tracking_data.f1_exit_time
            test_result["exit_time"] = exit_time

            if exit_time is not None:
                test_result["exit_time_minutes"] = exit_time / 60
                self.log(f"🚪 Exit time: {exit_time:.2f}s ({exit_time/60:.1f} minutes)")
            else:
                self.log(f"🚪 No exit detected (fly remained in first corridor)")

            # Test premature exit detection
            if hasattr(tracking_data, "check_f1_premature_exit"):
                self.log("🔍 Running premature exit check...")

                is_premature = tracking_data.check_f1_premature_exit()
                test_result["actual_premature"] = is_premature

                if is_premature:
                    self.log(f"⚠️  ALGORITHM RESULT: PREMATURE EXIT detected")
                else:
                    self.log(f"✅ ALGORITHM RESULT: Normal timing (no premature exit)")

                # Check if detection is correct
                test_result["correct_detection"] = is_premature == expected_premature

                if test_result["correct_detection"]:
                    self.log(f"🎯 DETECTION: CORRECT ✅")
                    self.results["summary"]["correct_detections"] += 1
                else:
                    if expected_premature and not is_premature:
                        self.log(f"❌ DETECTION: FALSE NEGATIVE (missed premature exit)")
                        self.results["summary"]["false_negatives"] += 1
                    elif not expected_premature and is_premature:
                        self.log(f"❌ DETECTION: FALSE POSITIVE (flagged normal fly)")
                        self.results["summary"]["false_positives"] += 1

                test_result["success"] = True

            else:
                raise ValueError("check_f1_premature_exit method not found in tracking_data")

            # Additional test details
            test_details = test_result["test_details"]
            test_details["load_time_seconds"] = load_time
            test_details["tracking_duration"] = getattr(tracking_data, "duration", None)
            test_details["start_position"] = (
                getattr(tracking_data, "start_x", None),
                getattr(tracking_data, "start_y", None),
            )

            # Ball information
            if tracking_data.balltrack and tracking_data.balltrack.objects:
                test_details["num_balls"] = len(tracking_data.balltrack.objects)
                test_details["ball_identities"] = getattr(tracking_data, "ball_identities", {})

            self.log(f"📊 Test completed successfully")

        except Exception as e:
            error_type = type(e).__name__
            self.log(f"❌ Error testing fly {fly_name}: {error_type}: {e}", force=True)

            # Provide additional debugging information
            if "string index out of range" in str(e):
                self.log(f"   🔍 This is likely an arena/corridor parsing issue", force=True)
                self.log(f"   🔍 Check if fly path matches expected F1 format: arena#/Left or arena#/Right", force=True)
            elif "Tracking data is None" in str(e):
                self.log(f"   🔍 Fly object created but tracking_data property returned None", force=True)
                self.log(f"   🔍 This could be due to missing .h5 files or invalid data quality", force=True)
            elif "Invalid tracking data" in str(e):
                self.log(f"   🔍 Tracking data was loaded but marked as invalid", force=True)
                self.log(f"   🔍 Check if fly movement meets quality thresholds", force=True)

            test_result["error"] = f"{error_type}: {e}"
            test_result["success"] = False
            self.results["summary"]["errors"] += 1

        self.results["summary"]["total_tested"] += 1
        return test_result

    def run_all_tests(self):
        """Run all premature exit detection tests."""
        print(f"🧪 F1 PREMATURE EXIT DETECTION TEST")
        print(f"{'='*80}")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Testing manually verified examples...")

        total_start_time = time.time()

        # Test premature exit cases (should be flagged)
        print(f"\n🔴 TESTING KNOWN PREMATURE EXITS")
        print(f"{'='*50}")

        for fly_name, fly_path in self.test_cases["premature_exits"].items():
            result = self.test_single_fly(fly_path, expected_premature=True, fly_name=fly_name)
            self.results["premature_exits"][fly_name] = result

        # Test normal fly cases (should NOT be flagged)
        print(f"\n🟢 TESTING NORMAL FLIES")
        print(f"{'='*50}")

        for fly_name, fly_path in self.test_cases["normal_flies"].items():
            result = self.test_single_fly(fly_path, expected_premature=False, fly_name=fly_name)
            self.results["normal_flies"][fly_name] = result

        # Calculate total test time
        total_time = time.time() - total_start_time

        # Generate summary
        self.print_summary(total_time)

        return self.results

    def print_summary(self, total_time):
        """Print test summary results."""
        print(f"\n{'='*80}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*80}")

        summary = self.results["summary"]

        print(f"⏱️  Total test time: {total_time:.3f} seconds")
        print(f"🧪 Total flies tested: {summary['total_tested']}")
        print(f"✅ Correct detections: {summary['correct_detections']}")
        print(f"❌ False positives: {summary['false_positives']}")
        print(f"❌ False negatives: {summary['false_negatives']}")
        print(f"💥 Errors: {summary['errors']}")

        if summary["total_tested"] > 0:
            accuracy = summary["correct_detections"] / summary["total_tested"]
            print(f"🎯 Accuracy: {accuracy:.1%}")

        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'-'*60}")

        # Premature exit results
        print(f"\n🔴 PREMATURE EXITS (should be flagged):")
        for fly_name, result in self.results["premature_exits"].items():
            status = "✅ CORRECT" if result["correct_detection"] else "❌ MISSED"
            exit_info = f"({result['exit_time_minutes']:.1f} min)" if result["exit_time"] else "(no exit)"
            error_info = f" - ERROR: {result['error']}" if result.get("error") else ""
            print(f"   {fly_name}: {status} {exit_info}{error_info}")

        # Normal fly results
        print(f"\n🟢 NORMAL FLIES (should NOT be flagged):")
        for fly_name, result in self.results["normal_flies"].items():
            status = "✅ CORRECT" if result["correct_detection"] else "❌ FALSE ALARM"
            exit_info = f"({result['exit_time_minutes']:.1f} min)" if result["exit_time"] else "(no exit)"
            error_info = f" - ERROR: {result['error']}" if result.get("error") else ""
            print(f"   {fly_name}: {status} {exit_info}{error_info}")

        # Overall assessment
        print(f"\n{'='*60}")
        if summary["errors"] == 0 and summary["false_positives"] == 0 and summary["false_negatives"] == 0:
            print(f"🎉 PERFECT PERFORMANCE: All tests passed!")
        elif summary["correct_detections"] == summary["total_tested"] - summary["errors"]:
            print(f"✅ GOOD PERFORMANCE: All non-error cases correct")
        else:
            print(f"⚠️  NEEDS ATTENTION: Check false positives/negatives")

        print(f"{'='*60}")

        return summary

    def find_additional_test_flies(self, base_dir="/mnt/upramdya_data/MD/F1_Tracks/Videos/"):
        """
        Helper method to find additional test flies from newer F1 experiment directories (2509/2510).
        This can be used to expand the test dataset with flies from recent experiments.
        """
        base_path = Path(base_dir)
        potential_flies = []

        if not base_path.exists():
            self.log(f"Base directory not found: {base_dir}")
            return potential_flies

        # Look for newer F1 experiment directories (containing 2509 or 2510)
        for exp_dir in base_path.glob("*"):
            if exp_dir.is_dir() and any(pattern in exp_dir.name for pattern in ["2509", "2510"]):
                # Look for arena directories
                for arena_dir in exp_dir.glob("arena*"):
                    if arena_dir.is_dir():
                        # Look for side directories (Left/Right)
                        for side_dir in arena_dir.glob("*"):
                            if side_dir.is_dir() and side_dir.name in ["Left", "Right"]:
                                # Check if it has tracking files
                                if any(side_dir.glob("*.slp")) or any(side_dir.glob("*.h5")):
                                    potential_flies.append(str(side_dir))

        return potential_flies


def test_basic_logic():
    """Test the basic logic for F1 premature exit detection (unit test)."""

    print("🧪 Testing F1 premature exit detection LOGIC:")
    print("=" * 60)

    # Test cases: (exit_time_minutes, expected_discard, description)
    test_cases = [
        (None, False, "Fly never exits first corridor"),
        (30, True, "Fly exits at 30 minutes (before 55 min threshold)"),
        (45, True, "Fly exits at 45 minutes (before 55 min threshold)"),
        (54.9, True, "Fly exits at 54.9 minutes (just before threshold)"),
        (55, False, "Fly exits exactly at 55 minutes (at threshold)"),
        (60, False, "Fly exits at 60 minutes (after threshold)"),
        (120, False, "Fly exits at 120 minutes (well after threshold)"),
    ]

    premature_exit_threshold = 55 * 60  # 55 minutes in seconds

    all_passed = True

    for exit_time_minutes, expected_discard, description in test_cases:
        # Convert to seconds if not None
        exit_time_seconds = exit_time_minutes * 60 if exit_time_minutes is not None else None

        # Apply the logic
        if exit_time_seconds is None:
            should_discard = False
        else:
            should_discard = exit_time_seconds < premature_exit_threshold

        # Check result
        correct = should_discard == expected_discard
        status = "✅" if correct else "❌"
        discard_text = "DISCARD" if should_discard else "KEEP"

        print(f"{status} {description}")
        print(f"    Exit time: {exit_time_minutes} min → Decision: {discard_text}")

        if not correct:
            print(f"    ERROR: Expected {expected_discard}, got {should_discard}")
            all_passed = False
        print()

    if all_passed:
        print("🎉 All logic tests passed!")
    else:
        print("❌ Some logic tests failed!")

    return all_passed


def main():
    """Main function for F1 premature exit testing."""
    parser = argparse.ArgumentParser(
        description="Test F1 premature exit detection using manually verified examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This test validates the F1 premature exit detection logic using real data:

Known premature exits (should be flagged):
  - /mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena3/Right/
  - /mnt/upramdya_data/MD/F1_Tracks/Videos/250905_F1_New_Videos_Checked/arena7/Right/

Normal flies (should NOT be flagged):
  - /mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena1/Left/
  - /mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena2/Right/

The test verifies:
1. Exit time detection (100px x-movement threshold)
2. Premature exit flagging (55-minute threshold)
3. Correct classification of known cases
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output for detailed test information"
    )

    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode for maximum detail")

    parser.add_argument("--logic-only", action="store_true", help="Only run basic logic tests (no real data loading)")

    parser.add_argument(
        "--find-flies", action="store_true", help="Search for additional potential test flies in the data directory"
    )

    args = parser.parse_args()

    # Run basic logic test first
    print("STEP 1: Testing basic logic...")
    logic_passed = test_basic_logic()

    if args.logic_only:
        return 0 if logic_passed else 1

    if not logic_passed:
        print("⚠️  Basic logic tests failed - real data tests may also fail")

    # Create tester instance
    tester = F1PrematureExitTester(verbose=args.verbose, debug=args.debug)

    # Option to find additional flies
    if args.find_flies:
        print("\nSTEP 2: Searching for additional test flies...")
        potential_flies = tester.find_additional_test_flies()
        print(f"Found {len(potential_flies)} potential test flies:")
        for fly_path in potential_flies[:10]:  # Show first 10
            print(f"  {fly_path}")
        if len(potential_flies) > 10:
            print(f"  ... and {len(potential_flies) - 10} more")
        return 0

    # Run the main test
    print(f"\nSTEP 2: Testing with real data...")
    try:
        results = tester.run_all_tests()

        # Determine exit code based on results
        summary = results["summary"]
        if summary["errors"] > 0:
            print(f"\n❌ Test completed with errors")
            return 1
        elif summary["false_positives"] > 0 or summary["false_negatives"] > 0:
            print(f"\n⚠️  Test completed with detection issues")
            return 2
        else:
            print(f"\n✅ All tests passed successfully!")
            return 0

    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
