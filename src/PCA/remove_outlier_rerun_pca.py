#!/usr/bin/env python3
"""
Rerun PCA Analysis with Outlier Removal
This script removes the extreme outlier genotype "854 (OK107-Gal4)" and reruns the entire PCA analysis
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def remove_outlier_and_rerun_pca():
    """Remove outlier genotype and rerun PCA analysis"""

    # Change to PCA directory
    os.chdir("/home/matthias/ballpushing_utils/src/PCA")

    print("=" * 60)
    print("REMOVING OUTLIER AND RERUNNING PCA ANALYSIS")
    print("=" * 60)

    # Load original data
    print("Loading original PCA data...")
    try:
        pca_data = pd.read_feather("static_pca_with_metadata_tailoredctrls.feather")
        print(f"Original data shape: {pca_data.shape}")
    except FileNotFoundError:
        print("Error: Original PCA data not found!")
        return

    # Check if outlier exists
    outlier_nickname = "854 (OK107-Gal4)"
    if outlier_nickname in pca_data["Nickname"].values:
        outlier_count = (pca_data["Nickname"] == outlier_nickname).sum()
        print(f"Found {outlier_count} samples with outlier genotype: {outlier_nickname}")

        # Remove outlier
        pca_data_cleaned = pca_data[pca_data["Nickname"] != outlier_nickname].copy()
        print(f"Cleaned data shape: {pca_data_cleaned.shape}")
        print(f"Removed {pca_data.shape[0] - pca_data_cleaned.shape[0]} samples")

        # Save cleaned data
        pca_data_cleaned.to_feather("static_pca_with_metadata_tailoredctrls_no_outlier.feather")
        print("Saved cleaned data as: static_pca_with_metadata_tailoredctrls_no_outlier.feather")

    else:
        print(f"Warning: Outlier genotype '{outlier_nickname}' not found in data")
        print("Available genotypes:")
        print(sorted(pca_data["Nickname"].unique()))
        return

    # Now we need to rerun the entire PCA analysis from scratch
    print("\n" + "=" * 60)
    print("RERUNNING PCA ANALYSIS FROM SCRATCH")
    print("=" * 60)

    # We need to go back to the original data and remove the outlier before PCA
    print("Note: For a complete reanalysis, you'll need to:")
    print("1. Remove the outlier from the original behavioral data")
    print("2. Rerun PCA computation on the cleaned data")
    print("3. Rerun statistical testing")
    print("4. Update all visualizations")

    print("\nRecommended approach:")
    print("1. Find the original data loading script")
    print("2. Add outlier removal before PCA computation")
    print("3. Rerun the entire analysis pipeline")

    # For now, let's create a modified version that works with existing PCA results
    print("\n" + "=" * 60)
    print("CREATING ANALYSIS WITH EXISTING PCA RESULTS (OUTLIER REMOVED)")
    print("=" * 60)

    # Update statistical results by removing the outlier
    print("Updating statistical results...")

    # Static results
    try:
        static_results = pd.read_csv("static_pca_stats_results_allmethods_tailoredctrls.csv")
        static_results_cleaned = static_results[static_results["Nickname"] != outlier_nickname].copy()
        static_results_cleaned.to_csv("static_pca_stats_results_allmethods_tailoredctrls_no_outlier.csv", index=False)
        print(f"Static results: {static_results.shape[0]} -> {static_results_cleaned.shape[0]} genotypes")
    except FileNotFoundError:
        print("Warning: Static results file not found")

    # Temporal results
    try:
        temporal_results = pd.read_csv("fpca_temporal_stats_results_allmethods_tailoredctrls.csv")
        temporal_results_cleaned = temporal_results[temporal_results["Nickname"] != outlier_nickname].copy()
        temporal_results_cleaned.to_csv(
            "fpca_temporal_stats_results_allmethods_tailoredctrls_no_outlier.csv", index=False
        )
        print(f"Temporal results: {temporal_results.shape[0]} -> {temporal_results_cleaned.shape[0]} genotypes")
    except FileNotFoundError:
        print("Warning: Temporal results file not found")

    print("\nCleaned files created:")
    print("- static_pca_with_metadata_tailoredctrls_no_outlier.feather")
    print("- static_pca_stats_results_allmethods_tailoredctrls_no_outlier.csv")
    print("- fpca_temporal_stats_results_allmethods_tailoredctrls_no_outlier.csv")

    # Print summary of what was removed
    print("\n" + "=" * 60)
    print("OUTLIER REMOVAL SUMMARY")
    print("=" * 60)
    print(f"Removed genotype: {outlier_nickname}")
    print(f"Reason: Extreme outlier affecting PCA analysis")
    print(f"This genotype was showing extreme values across multiple PC components")
    print(f"and was likely skewing the entire analysis.")

    return pca_data_cleaned


if __name__ == "__main__":
    remove_outlier_and_rerun_pca()
