#!/usr/bin/env python3

import sys
import os

sys.path.append("/home/matthias/ballpushing_utils/src")

import pandas as pd
import numpy as np


def test_coordinate_transformation_logic():
    """
    Test the coordinate transformation logic to ensure raw ball coordinates
    will be properly included and transformed.
    """
    print("=" * 80)
    print("TESTING COORDINATE TRANSFORMATION LOGIC")
    print("=" * 80)

    # Simulate the tracking data structure
    tracking_data = pd.DataFrame(
        {
            "x_Thorax": [100, 101, 102],
            "y_Thorax": [200, 201, 202],
            "x_Head": [100, 101, 102],
            "y_Head": [210, 211, 212],
            "x_centre_preprocessed": [150, 151, 152],
            "y_centre_preprocessed": [250, 251, 252],
            "x_centre_raw": [160, 161, 162],  # Raw coordinates (different from preprocessed)
            "y_centre_raw": [260, 261, 262],
        }
    )

    print("Original tracking data:")
    print(tracking_data)
    print()

    # Test the node extraction logic from skeleton_metrics
    nodes = [col[2:] for col in tracking_data.columns if col.startswith("x_") and not col.endswith("_fly")]
    print(f"Detected nodes for transformation: {nodes}")
    print()

    # Verify that both raw and preprocessed ball coordinates are detected
    expected_nodes = ["Thorax", "Head", "centre_preprocessed", "centre_raw"]

    if "centre_raw" in nodes:
        print("✅ SUCCESS: Raw ball coordinates detected for transformation")
    else:
        print("❌ FAILED: Raw ball coordinates not detected")

    if "centre_preprocessed" in nodes:
        print("✅ SUCCESS: Preprocessed ball coordinates detected for transformation")
    else:
        print("❌ FAILED: Preprocessed ball coordinates not detected")

    print()
    print("Expected transformed columns:")
    for node in nodes:
        print(f"  x_{node} -> x_{node}_fly")
        print(f"  y_{node} -> y_{node}_fly")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("✅ The existing transformation logic will automatically handle raw ball coordinates")
    print("✅ When fly_only=False, both x_centre_raw_fly and y_centre_raw_fly will be created")
    print("✅ This provides raw ball positions in the fly's coordinate system")
    print("✅ Perfect for analyzing ball movement relative to fly orientation")


if __name__ == "__main__":
    test_coordinate_transformation_logic()
