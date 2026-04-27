#!/usr/bin/env python3
"""
Master script to generate all panels of Extended Data Figure 7.

Extended Data Figure 7: Impact of ball surface treatment on pushing.

Panels:
- Panel (a): Photos of balls — no script needed (static image in paper)
- Panel (b): Time to first major push (>1.2 mm) by surface treatment
              Boxplots with permutation tests and FDR correction (n=24 per group)
              Expected: rusty p=0.036 (*), sandpaper p=0.036 (*)
- Panel (c): Ball distance trajectories over time
              Control vs rusty (top) and control vs sandpaper (bottom)
              5-min time bins, permutation tests with FDR correction
              Expected: no significant time bins (all FDR-corrected p ≥ 0.08)

Usage:
    python run_all_panels.py [--test] [--n-bins N] [--n-permutations N]

Arguments:
    --test: Test mode — use limited data for quick verification of all panels
    --n-bins: Number of time bins for panel (c) (default: 12)
    --n-permutations: Number of permutations for tests (default: 10000)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path, extra_args=None):
    """
    Run a Python script and handle errors.

    Parameters:
    -----------
    script_path : Path
        Path to the script to run
    extra_args : list, optional
        Additional command-line arguments to pass to the script

    Returns:
    --------
    bool
        True if the script completed successfully, False otherwise
    """
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    if extra_args:
        print(f"Args: {' '.join(extra_args)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, cwd=script_path.parent)
        print(f"✅ {script_path.name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_path.name} failed with exit code {e.returncode}")
        return False


def main():
    """Main function to run all panel scripts for Extended Data Figure 7."""
    parser = argparse.ArgumentParser(
        description="Generate all panels of Extended Data Figure 7 (ball surface treatments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_panels.py                         # Full analysis
  python run_all_panels.py --test                  # Quick test with limited data
  python run_all_panels.py --n-permutations 1000   # Fewer permutations (faster, less precise)
        """,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: use limited data for quick verification")
    parser.add_argument(
        "--n-bins", type=int, default=12, help="Number of time bins for panel (c) (default: 12 = 5-min segments)"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical tests (default: 10000)",
    )
    args = parser.parse_args()

    # Get directory containing this script
    script_dir = Path(__file__).parent

    # Build shared arguments
    shared_args = []
    if args.test:
        shared_args.append("--test")
    shared_args += ["--n-permutations", str(args.n_permutations)]

    # Panel-specific scripts and their arguments
    panel_scripts = [
        (
            script_dir / "edfigure7_b_first_push_time.py",
            shared_args,
        ),
        (
            script_dir / "edfigure7_c_trajectories.py",
            shared_args + ["--n-bins", str(args.n_bins)],
        ),
    ]

    # Check that all scripts exist
    missing = [s for s, _ in panel_scripts if not s.exists()]
    if missing:
        print("❌ Missing scripts:")
        for s in missing:
            print(f"  - {s}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Extended Data Figure 7: Impact of Ball Surface Treatment on Pushing")
    print(f"{'='*60}")
    print(f"Running {len(panel_scripts)} panel scripts...")
    if args.test:
        print("⚠️  TEST MODE: Using limited data")
    print(f"Permutations: {args.n_permutations}")
    print(f"Time bins (panel c): {args.n_bins} (= {60 // args.n_bins}-min segments for 60-min experiment)")

    # Run all scripts
    results = []
    for script, extra_args in panel_scripts:
        success = run_script(script, extra_args=extra_args)
        results.append((script.name, success))

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_success = True
    for script_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {script_name}")
        if not success:
            all_success = False

    output_dir = script_dir / "outputs"
    print(f"\n{'='*60}")
    if all_success:
        print("✅ All panels generated successfully!")
    else:
        print("⚠️  Some panels failed — check output above for details.")
    print(f"{'='*60}")
    print(f"\nOutputs are on the server under:")
    print(f"  <BALLPUSHING_FIGURES_ROOT>/EDFigure7/edfigure7_b_first_push_time/")
    print(f"  <BALLPUSHING_FIGURES_ROOT>/EDFigure7/edfigure7_c_trajectories/")
    print(f"(default: /mnt/upramdya_data/MD/Affordance_Figures/)")
    print(f"\nPanel descriptions:")
    print(f"  (b) edfigure7b_first_push_time.pdf            — Time to first major push boxplot")
    print(f"      edfigure7b_statistics.csv                 — Permutation test statistics")
    print(f"  (c) edfigure7c_trajectory_rusty_vs_control.pdf     — Rusty vs control trajectory")
    print(f"      edfigure7c_trajectory_sandpaper_vs_control.pdf — Sandpaper vs control trajectory")
    print(f"      edfigure7c_trajectory_statistics.csv           — Per-bin permutation test statistics")

    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
