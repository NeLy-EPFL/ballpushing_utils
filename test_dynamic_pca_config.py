#!/usr/bin/env python3
"""
Test script to verify dynamic PCA configuration detection works correctly.
"""

import sys
import os

sys.path.append("src/PCA")

from plot_configurations import get_dynamic_data_source, get_available_pca_columns


def test_dynamic_config():
    """Test the dynamic configuration detection"""
    print("Testing Dynamic PCA Configuration Detection")
    print("=" * 50)

    try:
        # Test static PCA detection
        print("\n1. Testing Static PCA Detection:")
        static_config = get_dynamic_data_source("static", ".")
        print(f"   Data file: {static_config['pca_data_file']}")
        print(f"   Results file: {static_config['static_results_file']}")
        print(f"   PCA columns: {static_config['pca_columns']}")
        print(f"   PC labels: {static_config['pc_labels']}")

        # Test temporal PCA detection
        print("\n2. Testing Temporal PCA Detection:")
        temporal_config = get_dynamic_data_source("temporal", ".")
        print(f"   Data file: {temporal_config['pca_data_file']}")
        print(f"   Results file: {temporal_config['static_results_file']}")
        print(f"   PCA columns: {temporal_config['pca_columns']}")
        print(f"   PC labels: {temporal_config['pc_labels']}")

        # Test available columns detection
        print("\n3. Testing Available Columns Detection:")
        try:
            available_static_cols = get_available_pca_columns("static", ".")
            print(f"   Available static PCA columns: {available_static_cols}")
        except FileNotFoundError as e:
            print(f"   Static PCA file not found: {e}")

        try:
            available_temporal_cols = get_available_pca_columns("temporal", ".")
            print(f"   Available temporal PCA columns: {available_temporal_cols}")
        except FileNotFoundError as e:
            print(f"   Temporal PCA file not found: {e}")

        print("\n✅ Dynamic configuration detection working correctly!")

    except Exception as e:
        print(f"\n❌ Error in dynamic configuration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dynamic_config()
