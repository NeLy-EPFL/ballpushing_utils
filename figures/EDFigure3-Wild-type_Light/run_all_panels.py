#!/usr/bin/env python3
"""
Master script to generate all panels of Extended Data Figure 3.

This script runs all analysis scripts for Extended Data Figure 3:
- Panel (a): Ball trajectory over time (light on vs off)
- Panel (b): Summary metrics comparison (first major event time, chamber time, AUC)

Usage:
    python run_all_panels.py [--test]

Arguments:
    --test: Test mode - use limited data for quick verification
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path, test_mode=False):
    """
    Run a Python script and handle errors.

    Parameters:
    -----------
    script_path : Path
        Path to the script to run
    test_mode : bool
        Whether to run in test mode
    """
    cmd = [sys.executable, str(script_path)]
    if test_mode:
        cmd.append("--test")

    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, cwd=script_path.parent)
        print(f"✅ {script_path.name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_path.name} failed with exit code {e.returncode}")
        return False


def main():
    """Main function to run all panel scripts"""
    parser = argparse.ArgumentParser(
        description="Generate all panels of Extended Data Figure 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--test", action="store_true", help="Test mode: use limited data for quick verification")

    args = parser.parse_args()

    # Get directory containing this script
    script_dir = Path(__file__).parent

    # Define all panel scripts in order
    panel_scripts = [
        script_dir / "edfigure3_a_trajectories.py",
        script_dir / "edfigure3_b_summary_metrics.py",
    ]

    # Check that all scripts exist
    missing_scripts = [s for s in panel_scripts if not s.exists()]
    if missing_scripts:
        print("❌ Missing scripts:")
        for s in missing_scripts:
            print(f"  - {s}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Extended Data Figure 3: Wild-type Light Condition Analysis")
    print(f"{'='*60}")
    print(f"Running {len(panel_scripts)} panel scripts...")
    if args.test:
        print("⚠️  TEST MODE: Using limited data")

    # Run all scripts
    results = []
    for script in panel_scripts:
        success = run_script(script, test_mode=args.test)
        results.append((script.name, success))

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_success = True
    for script_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {script_name}")
        if not success:
            all_success = False

    if all_success:
        print(f"\n✅ All panels generated successfully!")
        print(f"\nOutputs are on the server under:")
        print(f"  <BALLPUSHING_FIGURES_ROOT>/EDFigure3/edfigure3_a_trajectories/")
        print(f"  <BALLPUSHING_FIGURES_ROOT>/EDFigure3/edfigure3_b_summary_metrics/")
        print(f"(default: /mnt/upramdya_data/MD/Affordance_Figures/)")
        sys.exit(0)
    else:
        print(f"\n❌ Some panels failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
