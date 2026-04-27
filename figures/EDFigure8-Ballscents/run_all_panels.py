#!/usr/bin/env python3
"""
Master script to generate all panels of Extended Data Figure 8.

Extended Data Figure 8: Effect of ball scent treatment on pushing.

Panels:
  (a) Interaction rate by ball scent treatment (boxplot)
  (b) Pulling ratio by ball scent treatment (boxplot)
  (c) Time to first major push (>1.2 mm) by ball scent treatment (boxplot)
      All treatments vs control ('New'), permutation tests FDR-corrected (α = 0.05)
      n = 36 for control, n = 17–18 per treatment
      Expected: all FDR-corrected p ≥ 0.13 (no significant differences)

  (d) Ball distance trajectory: New vs New + Pre-exposed (fly odors)
  (e) Ball distance trajectory: New vs Washed (ethanol)
  (f) Ball distance trajectory: New vs Washed + Pre-exposed
      Binned into 10-min segments, permutation tests with FDR correction
      n = 36 for control, n = 18 per treatment
      Expected: no significant bins (all FDR-corrected p ≥ 0.11)

Usage:
    python run_all_panels.py [--test] [--n-bins N] [--n-permutations N]

Arguments:
    --test: Test mode — use limited data for quick verification of all panels
    --n-bins: Number of time bins for panels (d-f) (default: 6 = 10-min segments)
    --n-permutations: Number of permutations for tests (default: 10000)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path, extra_args=None):
    """
    Run a Python script and handle errors.

    Parameters
    ----------
    script_path : Path
    extra_args : list, optional

    Returns
    -------
    bool   True if successful
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
    """Main function to run all panel scripts for Extended Data Figure 8."""
    parser = argparse.ArgumentParser(
        description="Generate all panels of Extended Data Figure 8 (ball scent treatments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_panels.py                         # Full analysis
  python run_all_panels.py --test                  # Quick test with limited data
  python run_all_panels.py --n-permutations 1000   # Fewer permutations (faster, less precise)
  python run_all_panels.py --n-bins 6              # 10-min segments (default)
        """,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: use limited data for quick verification")
    parser.add_argument(
        "--n-bins",
        type=int,
        default=6,
        help="Number of time bins for panels (d-f) (default: 6 = 10-min segments for 60-min experiment)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for statistical tests (default: 10000)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    shared_args = []
    if args.test:
        shared_args.append("--test")
    shared_args += ["--n-permutations", str(args.n_permutations)]

    panel_scripts = [
        (
            script_dir / "edfigure8_abc_metrics.py",
            shared_args,
        ),
        (
            script_dir / "edfigure8_def_trajectories.py",
            shared_args + ["--n-bins", str(args.n_bins)],
        ),
    ]

    # Verify all scripts exist
    missing = [s for s, _ in panel_scripts if not s.exists()]
    if missing:
        print("❌ Missing scripts:")
        for s in missing:
            print(f"  - {s}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Extended Data Figure 8: Effect of Ball Scent Treatment on Pushing")
    print(f"{'='*60}")
    print(f"Running {len(panel_scripts)} panel scripts...")
    if args.test:
        print("⚠️  TEST MODE: Using limited data")
    print(f"Permutations: {args.n_permutations}")
    print(f"Time bins (panels d-f): {args.n_bins} = {60 // args.n_bins}-min segments for 60-min experiment")

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

    print(f"\n{'='*60}")
    if all_success:
        print("✅ All panels generated successfully!")
    else:
        print("⚠️  Some panels failed — check output above for details.")
    print(f"{'='*60}")
    print(f"\nOutputs are on the server under:")
    print(f"  <BALLPUSHING_FIGURES_ROOT>/EDFigure8/edfigure8_abc_metrics/")
    print(f"  <BALLPUSHING_FIGURES_ROOT>/EDFigure8/edfigure8_def_trajectories/")
    print(f"(default: /mnt/upramdya_data/MD/Affordance_Figures/)")
    print(f"\nPanel descriptions:")
    print(f"  Panels (a-c):")
    print(f"    edfigure8_abc_metrics_combined.pdf      — all three metric boxplots")
    print(f"    edfigure8a_overall_interaction_rate.pdf — (a) interaction rate")
    print(f"    edfigure8b_pulling_ratio.pdf            — (b) pulling ratio")
    print(f"    edfigure8c_first_major_event_time.pdf   — (c) time to first major push")
    print(f"    edfigure8_abc_statistics.csv            — permutation test statistics")
    print(f"  Panels (d-f):")
    print(f"    edfigure8_def_trajectories_combined.pdf    — all three trajectory panels")
    print(f"    edfigure8d_trajectory_Pre-exposed_vs_New.pdf")
    print(f"    edfigure8e_trajectory_Washed_vs_New.pdf")
    print(f"    edfigure8f_trajectory_Washed_plus_Pre-exposed_vs_New.pdf")
    print(f"    edfigure8_def_trajectory_statistics.csv    — per-bin statistics")
    print(f"\n  Conditions: New (ctrl, gray), Pre-exposed (blue), Washed (green), Washed+Pre-exposed (orange)")

    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
