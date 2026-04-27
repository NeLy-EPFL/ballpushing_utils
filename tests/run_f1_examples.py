#!/usr/bin/env python3
"""
Example script to run F1 Mann-Whitney analysis with different configurations.

This script demonstrates various ways to run the F1 Mann-Whitney analysis:
1. Test ball analysis (most common for F1 experiments)
2. Training ball analysis
3. Both balls combined analysis
4. Test mode for debugging
5. Specific metrics analysis
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 60)

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Run different F1 analysis configurations"""

    script_path = Path(__file__).parent / "run_mannwhitney_f1_metrics.py"

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return

    print("🎾 F1 Mann-Whitney Analysis Examples")
    print(f"Script location: {script_path}")

    # Example 1: Test mode for quick debugging
    print("\n1️⃣ Running in TEST MODE (fast debugging)...")
    if not run_command(
        [sys.executable, str(script_path), "--test", "--ball-identity", "test"], "Test mode with test ball"
    ):
        print("Test mode failed, stopping...")
        return

    # Example 2: Specific metrics analysis
    print("\n2️⃣ Running specific metrics analysis...")
    run_command(
        [
            sys.executable,
            str(script_path),
            "--metrics",
            "nb_events",
            "has_finished",
            "max_distance",
            "chamber_time",
            "--ball-identity",
            "test",
        ],
        "Specific metrics for test ball",
    )

    # Example 3: Training ball analysis
    print("\n3️⃣ Running training ball analysis...")
    run_command(
        [
            sys.executable,
            str(script_path),
            "--ball-identity",
            "training",
            "--test",  # Use test mode for faster execution
        ],
        "Training ball analysis (test mode)",
    )

    # Example 4: Both balls combined analysis
    print("\n4️⃣ Running both balls combined analysis...")
    run_command(
        [sys.executable, str(script_path), "--ball-identity", "both", "--test"],  # Use test mode for faster execution
        "Both balls combined (test mode)",
    )

    print("\n✅ F1 analysis examples completed!")
    print("Check the output directories for results:")
    print("- /mnt/upramdya_data/MD/F1_Tracks/Plots/Summary_metrics/F1_Metrics_byPretraining_Mannwhitney/ball_test/")
    print("- /mnt/upramdya_data/MD/F1_Tracks/Plots/Summary_metrics/F1_Metrics_byPretraining_Mannwhitney/ball_training/")
    print("- /mnt/upramdya_data/MD/F1_Tracks/Plots/Summary_metrics/F1_Metrics_byPretraining_Mannwhitney/ball_both/")


if __name__ == "__main__":
    main()
