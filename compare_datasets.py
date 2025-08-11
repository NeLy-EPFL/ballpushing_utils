#!/usr/bin/env python3
"""
Dataset Comparison Script
Compare two ballpushing datasets to analyze differences in flies, metrics, and data quality.
"""

import pandas as pd
import numpy as np
import sys
import os


def load_dataset(path, name):
    """Load dataset with error handling"""
    try:
        df = pd.read_feather(path)
        print(f"✅ Successfully loaded {name}")
        print(f"   Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None


def analyze_unique_flies(df, dataset_name):
    """Analyze unique flies in the dataset"""
    print(f"\n🔍 UNIQUE FLIES ANALYSIS - {dataset_name}")
    print("=" * 60)

    # Check if 'fly_id' column exists, if not use other identifier columns
    id_columns = ["fly", "Nickname", "exp_id", "experiment_id"]
    available_id_cols = [col for col in id_columns if col in df.columns]

    print(f"Available ID columns: {available_id_cols}")

    for col in available_id_cols:
        unique_count = df[col].nunique()
        total_rows = len(df)
        print(f"  • {col}: {unique_count} unique values (out of {total_rows} total rows)")

        # Show some examples
        if unique_count <= 10:
            print(f"    Values: {sorted(df[col].unique())}")
        else:
            print(f"    Sample values: {sorted(df[col].unique())[:5]}... (showing first 5)")

    # If fly_id exists, do more detailed analysis
    if "fly_id" in df.columns:
        print(f"\n📊 FLY_ID DETAILED ANALYSIS:")

        # Count observations per fly
        fly_counts = df["fly_id"].value_counts()
        print(f"  • Flies with 1 observation: {(fly_counts == 1).sum()}")
        print(f"  • Flies with multiple observations: {(fly_counts > 1).sum()}")
        print(f"  • Max observations per fly: {fly_counts.max()}")
        print(f"  • Mean observations per fly: {fly_counts.mean():.2f}")

        # Show flies with most observations
        if fly_counts.max() > 1:
            print(f"  • Top 5 flies by observation count:")
            for fly_id, count in fly_counts.head().items():
                print(f"    - {fly_id}: {count} observations")

    return df


def compare_columns(df1, df2, name1, name2):
    """Compare columns between datasets"""
    print(f"\n📋 COLUMN COMPARISON")
    print("=" * 60)

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    common_cols = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1

    print(f"Common columns: {len(common_cols)}")
    print(f"Only in {name1}: {len(only_in_1)}")
    print(f"Only in {name2}: {len(only_in_2)}")

    if only_in_1:
        print(f"\nColumns only in {name1}:")
        for col in sorted(only_in_1):
            print(f"  • {col}")

    if only_in_2:
        print(f"\nColumns only in {name2}:")
        for col in sorted(only_in_2):
            print(f"  • {col}")

    return common_cols, only_in_1, only_in_2


def compare_data_quality(df1, df2, name1, name2, common_cols):
    """Compare data quality metrics"""
    print(f"\n📊 DATA QUALITY COMPARISON")
    print("=" * 60)

    # Compare missing values for common columns
    missing_comparison = []

    for col in sorted(common_cols):
        missing1 = df1[col].isna().sum()
        missing2 = df2[col].isna().sum()
        pct1 = (missing1 / len(df1)) * 100
        pct2 = (missing2 / len(df2)) * 100

        if missing1 > 0 or missing2 > 0:
            missing_comparison.append(
                {
                    "Column": col,
                    f"{name1}_missing": missing1,
                    f"{name1}_pct": pct1,
                    f"{name2}_missing": missing2,
                    f"{name2}_pct": pct2,
                    "Difference": missing2 - missing1,
                }
            )

    if missing_comparison:
        missing_df = pd.DataFrame(missing_comparison)
        missing_df = missing_df.sort_values("Difference", key=abs, ascending=False)

        print("All columns with missing value differences:")
        print(missing_df.to_string(index=False))

        # Summary statistics
        print(f"\nMissing values summary:")
        print(f"  • Columns with missing values in {name1}: {(missing_df[f'{name1}_missing'] > 0).sum()}")
        print(f"  • Columns with missing values in {name2}: {(missing_df[f'{name2}_missing'] > 0).sum()}")
        print(f"  • Largest increase in missing values: {missing_df['Difference'].max()}")
        print(f"  • Largest decrease in missing values: {missing_df['Difference'].min()}")

        # Show only columns with differences
        differences = missing_df[missing_df["Difference"] != 0]
        if len(differences) > 0:
            print(f"\n🔍 COLUMNS WITH ACTUAL DIFFERENCES ({len(differences)} total):")
            print(differences.to_string(index=False))

            # Analyze which flies have missing values in the newer dataset
            analyze_flies_with_missing_values(df1, df2, differences, name1, name2)
        else:
            print("\n✅ No differences in missing values between datasets!")
    else:
        print("No missing values found in common columns.")


def analyze_flies_with_missing_values(df1, df2, differences_df, name1, name2):
    """Analyze which specific flies have missing values in the newer dataset"""
    print(f"\n🔬 ANALYZING FLIES WITH MISSING VALUES")
    print("=" * 60)

    # Look for path-related columns
    path_columns = [
        col
        for col in df1.columns
        if any(keyword in col.lower() for keyword in ["path", "folder", "file", "video", "track"])
    ]
    print(f"Found potential path columns: {path_columns}")

    for _, row in differences_df.iterrows():
        column = row["Column"]
        print(f"\n📁 COLUMN: {column}")
        print(f"   Missing increase: {row['Difference']} flies")

        # Find flies that are missing in newer dataset but not in older
        missing_in_old = df1[df1[column].isna()]
        missing_in_new = df2[df2[column].isna()]

        # Get fly IDs that became missing (missing in new but not in old)
        flies_old_missing = set(missing_in_old["fly"].values) if "fly" in df1.columns else set()
        flies_new_missing = set(missing_in_new["fly"].values) if "fly" in df2.columns else set()

        newly_missing_flies = flies_new_missing - flies_old_missing

        print(f"   Flies that became missing: {len(newly_missing_flies)}")

        if len(newly_missing_flies) > 0:
            # Show some examples
            sample_flies = list(newly_missing_flies)[:5]
            print(f"   Sample flies that became missing:")

            for fly_id in sample_flies:
                print(f"     • {fly_id}")

                # Show path information if available
                if path_columns:
                    fly_row = df1[df1["fly"] == fly_id] if "fly" in df1.columns else None
                    if fly_row is not None and len(fly_row) > 0:
                        for path_col in path_columns:
                            if path_col in fly_row.columns:
                                path_value = fly_row[path_col].iloc[0]
                                if pd.notna(path_value):
                                    print(f"       {path_col}: {path_value}")

            # Save the full list for further analysis
            print(f"\n   💾 Saving full list of newly missing flies for {column}...")
            newly_missing_df = df1[df1["fly"].isin(newly_missing_flies)] if "fly" in df1.columns else pd.DataFrame()

            if len(newly_missing_df) > 0:
                output_file = f"newly_missing_flies_{column.replace('/', '_')}.csv"
                columns_to_save = ["fly", "Nickname"] + path_columns
                available_cols = [col for col in columns_to_save if col in newly_missing_df.columns]
                newly_missing_df[available_cols].to_csv(output_file, index=False)
                print(f"   Saved to: {output_file}")


def compare_fly_overlap(df1, df2, name1, name2):
    """Compare fly overlap between datasets"""
    print(f"\n🔄 FLY OVERLAP ANALYSIS")
    print("=" * 60)

    # Use fly_id if available, otherwise use other identifier
    id_col = None
    for col in ["fly_id", "Nickname", "exp_id"]:
        if col in df1.columns and col in df2.columns:
            id_col = col
            break

    if not id_col:
        print("❌ No common identifier column found for fly overlap analysis")
        return

    print(f"Using '{id_col}' for overlap analysis")

    flies1 = set(df1[id_col].unique())
    flies2 = set(df2[id_col].unique())

    common_flies = flies1 & flies2
    only_in_1 = flies1 - flies2
    only_in_2 = flies2 - flies1

    print(f"\nFly overlap statistics:")
    print(f"  • {name1} unique {id_col}s: {len(flies1)}")
    print(f"  • {name2} unique {id_col}s: {len(flies2)}")
    print(f"  • Common {id_col}s: {len(common_flies)}")
    print(f"  • Only in {name1}: {len(only_in_1)}")
    print(f"  • Only in {name2}: {len(only_in_2)}")
    print(f"  • Overlap percentage: {len(common_flies) / len(flies1 | flies2) * 100:.1f}%")

    # Show some examples
    if only_in_1 and len(only_in_1) <= 10:
        print(f"\n{id_col}s only in {name1}: {sorted(only_in_1)}")
    elif only_in_1:
        print(f"\nFirst 5 {id_col}s only in {name1}: {sorted(list(only_in_1))[:5]}")

    if only_in_2 and len(only_in_2) <= 10:
        print(f"\n{id_col}s only in {name2}: {sorted(only_in_2)}")
    elif only_in_2:
        print(f"\nFirst 5 {id_col}s only in {name2}: {sorted(list(only_in_2))[:5]}")


def main():
    # Dataset paths
    dataset1_path = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250804_17_summary_TNT_screen_Data/summary/pooled_summary.feather"
    dataset2_path = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250809_02_standardized_contacts_TNT_screen_Data/summary/pooled_summary.feather"

    name1 = "Dataset_250804 (older)"
    name2 = "Dataset_250809 (newer)"

    print("🔬 BALLPUSHING DATASET COMPARISON")
    print("=" * 80)
    print(f"Dataset 1: {dataset1_path}")
    print(f"Dataset 2: {dataset2_path}")
    print("=" * 80)

    # Load datasets
    df1 = load_dataset(dataset1_path, name1)
    df2 = load_dataset(dataset2_path, name2)

    if df1 is None or df2 is None:
        print("❌ Failed to load one or both datasets. Exiting.")
        return

    # Analyze unique flies in each dataset
    analyze_unique_flies(df1, name1)
    analyze_unique_flies(df2, name2)

    # Compare columns
    common_cols, only_in_1, only_in_2 = compare_columns(df1, df2, name1, name2)

    # Compare data quality
    compare_data_quality(df1, df2, name1, name2, common_cols)

    # Compare fly overlap
    compare_fly_overlap(df1, df2, name1, name2)

    # Summary
    print(f"\n📈 SUMMARY")
    print("=" * 60)
    print(f"  • {name1}: {df1.shape[0]} rows, {df1.shape[1]} columns")
    print(f"  • {name2}: {df2.shape[0]} rows, {df2.shape[1]} columns")
    print(f"  • Row difference: {df2.shape[0] - df1.shape[0]:+d}")
    print(f"  • Column difference: {df2.shape[1] - df1.shape[1]:+d}")
    print(f"  • Common columns: {len(common_cols)}")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
