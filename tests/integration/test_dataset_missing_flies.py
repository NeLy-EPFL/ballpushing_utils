#!/usr/bin/env python3
"""
Test dataset generation with missing flies to identify where they get filtered out.
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


def test_dataset_generation_with_missing_flies(detailed_debug=True):
    """
    Test dataset generation using the standardized_contacts method with known missing flies.
    """
    print(f"🧪 Testing Dataset Generation with Missing Flies")
    print("=" * 80)

    # Known working fly paths from our previous test
    missing_fly_paths = [
        "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_1_Videos_Tracked/arena1/corridor1",
        "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_1_Videos_Tracked/arena1/corridor2",
        "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_1_Videos_Tracked/arena1/corridor3",
        "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231213_TNT_Fine_1_Videos_Tracked/arena2/corridor5",
    ]

    print(f"📁 Testing with {len(missing_fly_paths)} missing flies")

    try:
        # Step 1: Load flies individually first
        print("\nStep 1: Loading flies individually...")
        flies = []

        for i, fly_path in enumerate(missing_fly_paths):
            print(f"  Loading fly {i+1}: {Path(fly_path).name}")
            try:
                fly = Ballpushing_utils.Fly(fly_path, as_individual=True)
                flies.append(fly)
                print(f"  ✅ Loaded: {fly.metadata.name}")
            except Exception as e:
                print(f"  ❌ Failed to load fly: {e}")
                if detailed_debug:
                    print(f"  📋 Traceback:\n{traceback.format_exc()}")

        print(f"\n✅ Successfully loaded {len(flies)} flies individually")

        # Step 2: Test dataset creation with individual flies
        print(f"\nStep 2: Creating dataset with individual flies...")

        try:
            dataset = Ballpushing_utils.Dataset(flies, dataset_type="standardized_contacts")
            print(f"✅ Dataset created successfully")

            if dataset.data is not None:
                print(f"📊 Dataset shape: {dataset.data.shape}")

                if not dataset.data.empty:
                    print(
                        f"📊 Unique flies in dataset: {dataset.data['fly'].nunique() if 'fly' in dataset.data.columns else 'unknown'}"
                    )

                    # Check which flies made it into the dataset
                    if "fly" in dataset.data.columns:
                        flies_in_dataset = set(dataset.data["fly"].unique())
                        print(f"📋 Flies in dataset: {sorted(flies_in_dataset)}")

                        # Check which flies are missing
                        expected_fly_names = {fly.metadata.name for fly in flies}
                        missing_from_dataset = expected_fly_names - flies_in_dataset

                        if missing_from_dataset:
                            print(f"❌ Missing from dataset: {sorted(missing_from_dataset)}")
                        else:
                            print(f"✅ All flies present in dataset")

                else:
                    print(f"❌ Dataset data is empty!")
            else:
                print(f"❌ Dataset.data is None!")

        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            if detailed_debug:
                print(f"📋 Full traceback:\n{traceback.format_exc()}")
            return False

        # Step 3: Test with experiment-based approach
        print(f"\nStep 3: Testing experiment-based dataset creation...")

        # Group flies by experiment
        experiment_paths = {}
        for fly_path in missing_fly_paths:
            exp_path = Path(fly_path).parent.parent  # Go up two levels to get experiment dir
            if exp_path not in experiment_paths:
                experiment_paths[exp_path] = []
            experiment_paths[exp_path].append(fly_path)

        for exp_path, fly_paths in experiment_paths.items():
            print(f"\n  📁 Testing experiment: {exp_path.name}")
            print(f"    With {len(fly_paths)} flies")

            try:
                experiment = Ballpushing_utils.Experiment(exp_path)
                print(f"    ✅ Experiment loaded with {len(experiment.flies)} total flies")

                # Filter to only the specific flies we want
                # Convert full paths to expected fly names (experiment_arena_corridor format)
                target_fly_names = set()
                for fp in fly_paths:
                    path_parts = Path(fp).parts
                    # Extract experiment, arena, corridor from path
                    exp_name = path_parts[-3]  # experiment folder name
                    arena = path_parts[-2]  # arena folder name
                    corridor = path_parts[-1]  # corridor folder name
                    expected_name = f"{exp_name}_{arena}_{corridor}"
                    target_fly_names.add(expected_name)

                print(f"    🎯 Looking for fly names: {sorted(target_fly_names)}")
                print(
                    f"    📋 Available fly names: {sorted([fly.metadata.name for fly in experiment.flies[:5]])}...({len(experiment.flies)} total)"
                )

                target_flies = [fly for fly in experiment.flies if fly.metadata.name in target_fly_names]

                print(f"    🎯 Found {len(target_flies)} target flies in experiment")

                if target_flies:
                    # Test dataset creation with these specific flies
                    dataset_exp = Ballpushing_utils.Dataset(target_flies, dataset_type="standardized_contacts")

                    if dataset_exp.data is not None:
                        print(f"    ✅ Experiment-based dataset created: {dataset_exp.data.shape}")

                        if not dataset_exp.data.empty:
                            flies_count = dataset_exp.data["fly"].nunique() if "fly" in dataset_exp.data.columns else 0
                            print(f"    📊 Flies in experiment dataset: {flies_count}")
                        else:
                            print(f"    ❌ Experiment dataset data is empty!")
                    else:
                        print(f"    ❌ Experiment dataset.data is None!")
                else:
                    print(f"    ⚠️  No target flies found in experiment")

            except Exception as e:
                print(f"    ❌ Experiment-based dataset creation failed: {e}")
                if detailed_debug:
                    print(f"    📋 Traceback:\n{traceback.format_exc()}")

        # Step 4: Debug the dataset generation process step by step
        print(f"\nStep 4: Debugging dataset generation process...")

        if flies:
            print(f"  🔍 Analyzing first fly: {flies[0].metadata.name}")
            fly = flies[0]

            # Check if fly passes the valid_data filter
            print(
                f"    Valid tracking data: {fly.tracking_data.valid_data if hasattr(fly.tracking_data, 'valid_data') else 'unknown'}"
            )

            # Check skeleton metrics
            if hasattr(fly, "skeleton_metrics") and fly.skeleton_metrics:
                print(f"    Skeleton metrics available: ✅")
                events_df = fly.skeleton_metrics.events_based_contacts
                print(f"    Events-based contacts shape: {events_df.shape}")
            else:
                print(f"    Skeleton metrics available: ❌")

            # Test the _prepare_dataset_standardized_contacts method directly
            try:
                print(f"  🧪 Testing _prepare_dataset_standardized_contacts directly...")
                dataset_obj = Ballpushing_utils.Dataset(
                    [fly], dataset_type="coordinates"
                )  # Create with dummy type first
                result = dataset_obj._prepare_dataset_standardized_contacts(fly)
                print(f"    ✅ Direct method call successful: {result.shape}")

                if not result.empty:
                    print(f"    📊 Columns: {list(result.columns)}")
                    print(f"    📊 Sample data:\n{result.head(2)}")
                else:
                    print(f"    ❌ Direct method returned empty DataFrame")

            except Exception as e:
                print(f"    ❌ Direct method call failed: {e}")
                if detailed_debug:
                    print(f"    📋 Traceback:\n{traceback.format_exc()}")

        print(f"\n✅ Dataset generation analysis completed")
        return True

    except Exception as e:
        print(f"❌ Dataset generation test failed: {e}")
        if detailed_debug:
            print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False


def test_dataset_filtering_logic():
    """
    Test the filtering logic in Dataset.__init__ to see where flies get dropped.
    """
    print(f"\n🔍 Testing Dataset Filtering Logic")
    print("=" * 50)

    # Use one of our known working flies
    fly_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_1_Videos_Tracked/arena1/corridor1"

    try:
        print(f"📁 Loading fly: {Path(fly_path).name}")
        fly = Ballpushing_utils.Fly(fly_path, as_individual=True)
        print(f"✅ Fly loaded: {fly.metadata.name}")

        # Check the filtering condition in Dataset.__init__
        print(f"\n🔍 Checking Dataset filtering conditions...")

        # Check valid_data attribute
        if hasattr(fly, "_tracking_data") and fly._tracking_data is not None:
            tracking_data = fly._tracking_data
            print(f"  _tracking_data exists: ✅")

            if hasattr(tracking_data, "valid_data"):
                valid_data = tracking_data.valid_data
                print(f"  valid_data: {valid_data}")

                if not valid_data:
                    print(f"  ❌ Fly would be filtered out due to invalid tracking data!")
                else:
                    print(f"  ✅ Fly passes valid_data filter")
            else:
                print(f"  ❌ No valid_data attribute found")
        else:
            print(f"  ❌ No _tracking_data attribute found or it's None")

        # Check tracking_data (without underscore)
        if hasattr(fly, "tracking_data") and fly.tracking_data is not None:
            tracking_data = fly.tracking_data
            print(f"  tracking_data exists: ✅")

            if hasattr(tracking_data, "valid_data"):
                valid_data = tracking_data.valid_data
                print(f"  tracking_data.valid_data: {valid_data}")
            else:
                print(f"  No valid_data in tracking_data")
        else:
            print(f"  ❌ No tracking_data attribute found or it's None")

        # Test the actual filtering line from Dataset.__init__
        print(f"\n🧪 Testing actual filtering logic...")
        flies_list = [fly]

        # Original filtering line: self.flies = [fly for fly in self.flies if fly._tracking_data.valid_data]
        try:
            filtered_flies = [f for f in flies_list if f._tracking_data is not None and f._tracking_data.valid_data]
            print(f"  ✅ Original filter passes: {len(filtered_flies)}/{len(flies_list)} flies retained")
        except AttributeError as e:
            print(f"  ❌ Original filter fails: {e}")

            # Try alternative attribute
            try:
                filtered_flies = [f for f in flies_list if f.tracking_data is not None and f.tracking_data.valid_data]
                print(f"  ✅ Alternative filter passes: {len(filtered_flies)}/{len(flies_list)} flies retained")
            except AttributeError as e2:
                print(f"  ❌ Alternative filter also fails: {e2}")

        return True

    except Exception as e:
        print(f"❌ Filtering logic test failed: {e}")
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False


def main():
    """Main function with command line argument support."""

    parser = argparse.ArgumentParser(description="Test dataset generation with missing flies")
    parser.add_argument("--detailed", action="store_true", help="Enable detailed debugging output")

    args = parser.parse_args()

    print("🔍 Dataset Generation with Missing Flies Test")
    print("=" * 60)

    # Test dataset generation
    success1 = test_dataset_generation_with_missing_flies(detailed_debug=args.detailed)

    # Test filtering logic
    success2 = test_dataset_filtering_logic()

    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print("=" * 30)

    if success1 and success2:
        print("✅ All tests completed successfully")
    else:
        print("❌ Some tests failed")
        print(f"  Dataset generation: {'✅' if success1 else '❌'}")
        print(f"  Filtering logic: {'✅' if success2 else '❌'}")


if __name__ == "__main__":
    main()
