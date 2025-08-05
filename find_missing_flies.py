#!/usr/bin/env python3
"""
Find flies present in past but missing in current dataset, then test loading them directly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the source directory to the path
sys.path.append("/home/matthias/ballpushing_utils/src")


def find_missing_flies():
    """Find flies present in past but missing in current dataset."""

    print("🔍 Finding Missing Flies in Current Dataset")
    print("=" * 50)

    # File paths
    past_file = Path(
        "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250418_summary_TNT_screen_Data/standardized_contacts/pooled_standardized_contacts.feather"
    )
    current_file = Path(
        "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250802_standardized_contacts_TNT_screen_Data/standardized_contacts/pooled_standardized_contacts.feather"
    )

    try:
        # Try to load with pyarrow first, then fallback to other methods
        try:
            past_df = pd.read_feather(past_file)
            current_df = pd.read_feather(current_file)
            print("✅ Loaded datasets using feather format")
        except ImportError:
            print("⚠️  pyarrow not available, trying alternative loading methods...")
            # Try loading as parquet if feather fails
            try:
                past_df = pd.read_parquet(past_file)
                current_df = pd.read_parquet(current_file)
                print("✅ Loaded datasets using parquet format")
            except:
                print("❌ Cannot load feather files without pyarrow. Trying to install...")
                import subprocess

                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
                past_df = pd.read_feather(past_file)
                current_df = pd.read_feather(current_file)
                print("✅ Installed pyarrow and loaded datasets")

        print(f"📊 Past dataset: {len(past_df):,} rows")
        print(f"📊 Current dataset: {len(current_df):,} rows")

        # Look for fly identifier columns
        print("\n🔍 Analyzing fly identifier columns...")

        # Check available columns that might contain fly names
        potential_fly_cols = [
            col
            for col in past_df.columns
            if any(keyword in col.lower() for keyword in ["fly", "name", "arena", "experiment"])
        ]

        print(f"Potential fly identifier columns: {potential_fly_cols}")

        # Try different column names to find the fly identifier
        fly_col = None
        for col in ["fly_name", "name", "fly", "experiment_name"]:
            if col in past_df.columns:
                fly_col = col
                break

        if not fly_col:
            # Try the first column that might contain fly info
            if potential_fly_cols:
                fly_col = potential_fly_cols[0]
                print(f"⚠️  Using best guess column: {fly_col}")
            else:
                print("❌ Cannot find fly identifier column")
                return None

        print(f"✅ Using fly identifier column: {fly_col}")

        # Get unique flies from each dataset
        past_flies = set(past_df[fly_col].dropna().unique())
        current_flies = set(current_df[fly_col].dropna().unique())

        print(f"\n📊 Past dataset: {len(past_flies)} unique flies")
        print(f"📊 Current dataset: {len(current_flies)} unique flies")

        # Find missing flies
        missing_flies = past_flies - current_flies
        new_flies = current_flies - past_flies

        print(f"\n❌ Missing in current: {len(missing_flies)} flies")
        print(f"✅ New in current: {len(new_flies)} flies")

        if missing_flies:
            print(f"\n🔍 Missing flies (showing first 20):")
            missing_list = sorted(list(missing_flies))
            for i, fly in enumerate(missing_list[:20]):
                print(f"  {i+1:2d}. {fly}")

            if len(missing_list) > 20:
                print(f"     ... and {len(missing_list) - 20} more")

            # Look for flies matching the pattern you mentioned
            pattern_flies = [fly for fly in missing_list if "TNT" in fly and "Videos_Tracked" in fly and "arena" in fly]

            if pattern_flies:
                print(f"\n🎯 Missing flies matching pattern (TNT_*_Videos_Tracked_arena*): {len(pattern_flies)}")
                for i, fly in enumerate(pattern_flies[:10]):
                    print(f"  {i+1:2d}. {fly}")

                return pattern_flies
            else:
                print(f"\n⚠️  No missing flies match the expected pattern")
                return missing_list
        else:
            print("\n✅ No missing flies found")
            return []

    except Exception as e:
        print(f"❌ Error analyzing datasets: {e}")
        import traceback

        print(f"Full traceback:\n{traceback.format_exc()}")
        return None


def test_direct_fly_loading(fly_name):
    """Test loading a specific fly directly to see what happens."""

    print(f"\n🧪 Testing Direct Loading of Fly: {fly_name}")
    print("=" * 60)

    try:
        from Ballpushing_utils.experiment import Experiment
        from Ballpushing_utils.skeleton_metrics import SkeletonMetrics

        # Try to find the experiment directory from the fly name
        # Pattern: 231213_TNT_Fine_1_Videos_Tracked_arena2_corridor5
        # This suggests experiment directory might be like: *231213*TNT*Fine*1*Videos*Tracked*

        base_path = Path("/mnt/upramdya_data")

        # Extract parts from fly name to find experiment directory
        name_parts = fly_name.split("_")
        date_part = name_parts[0] if name_parts else ""

        print(f"🔍 Searching for experiment containing fly: {fly_name}")
        print(f"   Date part: {date_part}")

        # Search for experiment directories
        search_patterns = [
            f"*{date_part}*TNT*Videos*Tracked*",
            f"*{date_part}*TNT*",
            f"*TNT*{date_part}*",
            "*TNT*Videos*Tracked*",
        ]

        experiment_dirs = []
        for pattern in search_patterns:
            found_dirs = list(base_path.rglob(pattern))
            experiment_dirs.extend([d for d in found_dirs if d.is_dir()])

        # Remove duplicates
        experiment_dirs = list(set(experiment_dirs))

        print(f"🔍 Found {len(experiment_dirs)} potential experiment directories:")
        for exp_dir in experiment_dirs[:10]:  # Show first 10
            print(f"   {exp_dir}")

        if not experiment_dirs:
            print("❌ No experiment directories found")
            return

        # Try each experiment directory to find the fly
        for exp_dir in experiment_dirs:
            print(f"\n📁 Checking experiment: {exp_dir.name}")

            try:
                experiment = Experiment(exp_dir)
                print(f"   ✅ Loaded experiment with {len(experiment.flies)} flies")

                # Look for the specific fly
                target_fly = None
                for fly in experiment.flies:
                    if fly.metadata.name == fly_name:
                        target_fly = fly
                        break

                if target_fly:
                    print(f"   ✅ Found target fly: {fly_name}")

                    # Test the standardized contacts pipeline
                    print(f"   🧪 Testing standardized contacts pipeline...")

                    # Check required data
                    if (
                        not hasattr(target_fly.tracking_data, "raw_balltrack")
                        or target_fly.tracking_data.raw_balltrack is None
                    ):
                        print(f"   ❌ No raw_balltrack data")
                        continue

                    if (
                        not hasattr(target_fly.tracking_data, "skeletontrack")
                        or target_fly.tracking_data.skeletontrack is None
                    ):
                        print(f"   ❌ No skeletontrack data")
                        continue

                    print(f"   ✅ Required tracking data available")

                    # Try to create SkeletonMetrics
                    try:
                        skeleton_metrics = SkeletonMetrics(target_fly)
                        print(f"   ✅ SkeletonMetrics created successfully")

                        # Check contacts
                        print(f"   📊 Found {len(skeleton_metrics.all_contacts)} contact events")

                        # Check standardized events
                        events_df = skeleton_metrics.events_based_contacts
                        print(f"   📊 Generated {len(events_df)} standardized event rows")

                        if len(events_df) > 0:
                            print(f"   ✅ Successfully generated standardized contacts")
                            print(
                                f"   📋 Event types: {events_df.get('event_type', pd.Series()).value_counts().to_dict()}"
                            )
                        else:
                            print(f"   ⚠️  No standardized events generated")

                        return True

                    except Exception as e:
                        print(f"   ❌ SkeletonMetrics failed: {e}")
                        import traceback

                        print(f"   📋 Traceback:\n{traceback.format_exc()}")
                        return False

                else:
                    print(f"   ⚠️  Fly {fly_name} not found in this experiment")
                    # Show available flies for reference
                    available_flies = [f.metadata.name for f in experiment.flies[:5]]
                    print(f"   Available flies: {available_flies}")

            except Exception as e:
                print(f"   ❌ Failed to load experiment: {e}")

        print(f"\n❌ Could not find fly {fly_name} in any experiment directory")
        return False

    except Exception as e:
        print(f"❌ Error in direct fly loading: {e}")
        import traceback

        print(f"Full traceback:\n{traceback.format_exc()}")
        return False


def main():
    """Main function to find missing flies and test loading one."""

    # Find missing flies
    missing_flies = find_missing_flies()

    if missing_flies and len(missing_flies) > 0:
        # Test loading the first missing fly
        test_fly = missing_flies[0]
        print(f"\n🎯 Testing the first missing fly: {test_fly}")

        success = test_direct_fly_loading(test_fly)

        if not success and len(missing_flies) > 1:
            print(f"\n🔄 First fly failed, trying second fly: {missing_flies[1]}")
            test_direct_fly_loading(missing_flies[1])

    else:
        print("\n⚠️  No missing flies found to test")


if __name__ == "__main__":
    main()
