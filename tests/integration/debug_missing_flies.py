#!/usr/bin/env python3
"""
Debug script for testing missing flies in standardized contacts pipeline.
This script is inspired by the existing integration tests and provides detailed debugging
for flies that are present in past datasets but missing from current ones.
"""

import sys
import os
import argparse
import traceback
from pathlib import Path

# Add the source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import pandas as pd
import numpy as np
import Ballpushing_utils
from Ballpushing_utils.config import Config


def test_missing_fly_pipeline(fly_path, detailed_debug=True):
    """
    Test the standardized contacts pipeline for a specific missing fly.

    Args:
        fly_path (str): Path to the fly directory
        detailed_debug (bool): Whether to run detailed debugging
    """
    print(f"🧪 Testing Missing Fly Pipeline")
    print(f"📁 Fly path: {fly_path}")
    print("=" * 80)

    try:
        # Step 1: Load the fly
        print("Step 1: Loading fly...")
        fly = Ballpushing_utils.Fly(fly_path, as_individual=True)
        print(f"✅ Fly loaded: {fly.metadata.name}")

        # Step 2: Check basic data availability
        print("\nStep 2: Checking data availability...")

        print(f"  🔍 Checking tracking data...")
        if fly.tracking_data is None:
            print(f"  ❌ No tracking data")
            return False

        if not hasattr(fly.tracking_data, "raw_balltrack") or fly.tracking_data.raw_balltrack is None:
            print(f"  ❌ No raw_balltrack data")
            return False
        print(f"  ✅ raw_balltrack available")

        if not hasattr(fly.tracking_data, "skeletontrack") or fly.tracking_data.skeletontrack is None:
            print(f"  ❌ No skeletontrack data")
            return False
        print(f"  ✅ skeletontrack available")

        # Check data shapes
        raw_ball_data = fly.tracking_data.raw_balltrack.objects[0].dataset
        skeleton_data = fly.tracking_data.skeletontrack.objects[0].dataset

        print(f"  📊 Raw ball data shape: {raw_ball_data.shape}")
        print(f"  📊 Skeleton data shape: {skeleton_data.shape}")

        # Check for required columns
        ball_cols = [col for col in raw_ball_data.columns if "centre" in col]
        print(f"  📋 Ball columns: {ball_cols}")

        # Step 3: Test SkeletonMetrics initialization
        print(f"\nStep 3: Testing SkeletonMetrics initialization...")

        try:
            skeleton_metrics = Ballpushing_utils.SkeletonMetrics(fly)
            print(f"  ✅ SkeletonMetrics initialized successfully")
        except Exception as e:
            print(f"  ❌ SkeletonMetrics initialization failed: {e}")
            if detailed_debug:
                print(f"  📋 Full traceback:\n{traceback.format_exc()}")
            return False

        # Step 4: Test ball preprocessing
        print(f"\nStep 4: Testing ball preprocessing...")

        try:
            ball_data = skeleton_metrics.ball.objects[0].dataset

            if "x_centre_preprocessed" not in ball_data.columns:
                print(f"  ❌ Preprocessed coordinates not created")
                return False

            print(f"  ✅ Ball preprocessing successful")

            # Detailed analysis of preprocessed data
            x_preprocessed = ball_data["x_centre_preprocessed"].dropna()
            y_preprocessed = ball_data["y_centre_preprocessed"].dropna()

            print(f"  📊 Preprocessed data points: {len(x_preprocessed)}/{len(ball_data)}")

            if len(x_preprocessed) > 0:
                print(f"  📊 X range: {x_preprocessed.min():.1f} to {x_preprocessed.max():.1f}")
                print(f"  📊 Y range: {y_preprocessed.min():.1f} to {y_preprocessed.max():.1f}")
            else:
                print(f"  ❌ No valid preprocessed coordinates")
                return False

        except Exception as e:
            print(f"  ❌ Ball preprocessing failed: {e}")
            if detailed_debug:
                print(f"  📋 Full traceback:\n{traceback.format_exc()}")
            return False

        # Step 5: Test contact detection
        print(f"\nStep 5: Testing contact detection...")

        try:
            contacts = skeleton_metrics.all_contacts
            print(f"  ✅ Contact detection successful: {len(contacts)} contacts found")

            if len(contacts) > 0:
                print(f"  📊 First 5 contacts: {contacts[:5]}")

                # Analyze contact durations
                durations = [contact[1] - contact[0] for contact in contacts if len(contact) >= 2]
                if durations:
                    print(
                        f"  📊 Contact duration stats: min={min(durations)}, max={max(durations)}, mean={np.mean(durations):.1f}"
                    )
            else:
                print(f"  ⚠️  No contacts detected - this might be the issue!")

                # Debug why no contacts were found
                if detailed_debug:
                    print(f"  🔍 Debugging contact detection...")

                    # Check contact threshold
                    threshold = (
                        fly.experiment.config.contact_threshold
                        if hasattr(fly.experiment.config, "contact_threshold")
                        else "unknown"
                    )
                    print(f"    Contact threshold: {threshold}")

                    # Check contact nodes
                    nodes = (
                        fly.experiment.config.contact_nodes
                        if hasattr(fly.experiment.config, "contact_nodes")
                        else "unknown"
                    )
                    print(f"    Contact nodes: {nodes}")

                    # Check if skeleton data has the required nodes
                    skeleton_cols = [
                        col
                        for col in skeleton_data.columns
                        if any(node in col for node in (nodes if isinstance(nodes, list) else []))
                    ]
                    print(f"    Available skeleton columns for contact nodes: {skeleton_cols}")

        except Exception as e:
            print(f"  ❌ Contact detection failed: {e}")
            if detailed_debug:
                print(f"  📋 Full traceback:\n{traceback.format_exc()}")
            return False

        # Step 6: Test euclidean distance calculation
        print(f"\nStep 6: Testing euclidean distance calculation...")

        try:
            ball_data = skeleton_metrics.ball.objects[0].dataset

            if "euclidean_distance" not in ball_data.columns:
                print(f"  ⚠️  Euclidean distance not automatically calculated, computing manually...")
                ball_data["euclidean_distance"] = np.sqrt(
                    (ball_data["x_centre_preprocessed"] - ball_data["x_centre_preprocessed"].iloc[0]) ** 2
                    + (ball_data["y_centre_preprocessed"] - ball_data["y_centre_preprocessed"].iloc[0]) ** 2
                )

            euclidean_dist = ball_data["euclidean_distance"].dropna()
            if len(euclidean_dist) > 0:
                print(f"  ✅ Euclidean distance available")
                print(f"  📊 Distance range: {euclidean_dist.min():.1f} to {euclidean_dist.max():.1f}")
                print(f"  📊 Distance mean: {euclidean_dist.mean():.1f}")
            else:
                print(f"  ❌ No valid euclidean distance values")

        except Exception as e:
            print(f"  ❌ Euclidean distance calculation failed: {e}")
            if detailed_debug:
                print(f"  📋 Full traceback:\n{traceback.format_exc()}")

        # Step 7: Test annotated dataset creation
        print(f"\nStep 7: Testing annotated dataset creation...")

        try:
            annotated_df = skeleton_metrics.get_contact_annotated_dataset()

            if annotated_df is not None and not annotated_df.empty:
                print(f"  ✅ Annotated dataset created: {annotated_df.shape}")

                if "is_contact" in annotated_df.columns:
                    contact_frames = annotated_df["is_contact"].sum()
                    total_frames = len(annotated_df)
                    contact_percentage = (contact_frames / total_frames) * 100
                    print(f"  📊 Contact frames: {contact_frames}/{total_frames} ({contact_percentage:.1f}%)")
                else:
                    print(f"  ❌ 'is_contact' column missing from annotated dataset")

            else:
                print(f"  ❌ Annotated dataset is empty or None")

        except Exception as e:
            print(f"  ❌ Annotated dataset creation failed: {e}")
            if detailed_debug:
                print(f"  📋 Full traceback:\n{traceback.format_exc()}")

        # Step 8: Test standardized events generation
        print(f"\nStep 8: Testing standardized events generation...")

        try:
            events_df = skeleton_metrics.events_based_contacts

            print(f"  ✅ Standardized events generation completed: {len(events_df)} rows")

            if len(events_df) > 0:
                # Analyze event types
                if "event_type" in events_df.columns:
                    event_types = events_df["event_type"].value_counts()
                    print(f"  📊 Event types: {event_types.to_dict()}")
                else:
                    print(f"  ⚠️  No 'event_type' column in events")

                # Check for required columns
                required_cols = ["event_id", "fly_idx", "ball_idx", "adjusted_frame"]
                missing_cols = [col for col in required_cols if col not in events_df.columns]
                if missing_cols:
                    print(f"  ⚠️  Missing required columns: {missing_cols}")
                else:
                    print(f"  ✅ All required columns present")

            else:
                print(f"  ❌ No standardized events generated - THIS IS LIKELY THE MAIN ISSUE!")

                # Debug why no events were generated
                if detailed_debug:
                    print(f"  🔍 Debugging event generation...")

                    # Check standardized interactions
                    if fly.tracking_data and hasattr(fly.tracking_data, "standardized_interactions"):
                        std_interactions = fly.tracking_data.standardized_interactions
                        print(f"    Standardized interactions available: {len(std_interactions)}")
                        for key, events in std_interactions.items():
                            print(f"      {key}: {len(events)} events")
                    else:
                        print(f"    No standardized_interactions attribute found")

                    # Check config settings
                    config_attrs = [
                        "standardized_events_mode",
                        "generate_random",
                        "frames_before_onset",
                        "frames_after_onset",
                    ]
                    for attr in config_attrs:
                        if hasattr(fly.config, attr):
                            value = getattr(fly.config, attr)
                            print(f"    {attr}: {value}")
                        else:
                            print(f"    {attr}: NOT SET")

        except Exception as e:
            print(f"  ❌ Standardized events generation failed: {e}")
            if detailed_debug:
                print(f"  📋 Full traceback:\n{traceback.format_exc()}")
            return False

        print(f"\n✅ Pipeline test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        if detailed_debug:
            print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False


def find_missing_fly_paths():
    """Find paths to some of the missing flies for testing."""

    print("🔍 Finding Missing Fly Paths...")

    # Known missing flies from our earlier analysis
    missing_fly_names = [
        "231115_TNT_Fine_1_Videos_Tracked_arena1_corridor1",
        "231115_TNT_Fine_1_Videos_Tracked_arena1_corridor2",
        "231115_TNT_Fine_1_Videos_Tracked_arena1_corridor3",
        "231213_TNT_Fine_1_Videos_Tracked_arena2_corridor5",  # This matches the pattern you mentioned
    ]

    # Common base paths to search
    base_paths = [
        Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos"),
    ]

    found_flies = []

    for fly_name in missing_fly_names:
        print(f"\n🔍 Searching for: {fly_name}")

        # Extract experiment info from fly name
        parts = fly_name.split("_")
        date_part = parts[0] if parts else ""
        arena_part = parts[-2] if len(parts) >= 2 else ""
        corridor_part = parts[-1] if parts else ""

        # Build potential paths
        potential_paths = []
        for base_path in base_paths:
            if not base_path.exists():
                continue

            # Pattern 1: date_TNT_Fine_1_Videos_Tracked/arena/corridor
            exp_name = f"{date_part}_TNT_Fine_1_Videos_Tracked"
            potential_paths.extend(
                [
                    base_path / exp_name / arena_part / corridor_part,
                    base_path / "Experiments" / exp_name / arena_part / corridor_part,
                    base_path / "Videos" / exp_name / arena_part / corridor_part,
                ]
            )

        # Check each potential path
        for path in potential_paths:
            if path.exists():
                print(f"  ✅ Found: {path}")
                found_flies.append((fly_name, str(path)))
                break
        else:
            print(f"  ❌ Not found: {fly_name}")

    return found_flies


def main():
    """Main function with command line argument support."""

    parser = argparse.ArgumentParser(description="Debug missing flies in standardized contacts pipeline")
    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="auto",
        help="Auto: find missing flies automatically, Manual: use provided path",
    )
    parser.add_argument("--path", type=str, help="Path to specific fly directory for manual testing")
    parser.add_argument("--detailed", action="store_true", help="Enable detailed debugging output")

    args = parser.parse_args()

    print("🔍 Missing Fly Pipeline Debugger")
    print("=" * 50)

    if args.mode == "manual" and args.path:
        # Test specific fly
        fly_path = Path(args.path)
        if not fly_path.exists():
            print(f"❌ Path does not exist: {fly_path}")
            return

        success = test_missing_fly_pipeline(str(fly_path), detailed_debug=args.detailed)
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Testing {fly_path.name}")

    else:
        # Auto mode: find and test missing flies
        found_flies = find_missing_fly_paths()

        if not found_flies:
            print("❌ No missing flies found for testing")
            return

        print(f"\n🧪 Testing {len(found_flies)} found missing flies...")

        results = []
        for fly_name, fly_path in found_flies:
            print(f"\n{'='*80}")
            success = test_missing_fly_pipeline(fly_path, detailed_debug=args.detailed)
            results.append((fly_name, success))

        # Summary
        print(f"\n{'='*80}")
        print("📊 SUMMARY")
        print("=" * 50)

        successful = sum(1 for _, success in results if success)
        total = len(results)

        print(f"Total tested: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")

        for fly_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status}: {fly_name}")


if __name__ == "__main__":
    main()
