#!/usr/bin/env python3
"""
Master script to generate all panels of Extended Data Figure 10.

Extended Data Figure 10: Effect of nutritional state on ball pushing behaviour.

Panels:
  (a) Ball distance trajectory grouped by nutritional state (fed / starved / starved+water-deprived)
      12 × 5-min bins, bootstrapped 95% CI, pairwise permutation tests with FDR correction
      n = 84 / 119 / 119 flies
  (b) Distribution of final ball positions by nutritional state
      Overlaid histograms + KDE, median lines
  (c) Behavioral metrics with significant differences across nutritional states:
      - Ratio of significant interaction events
      - Interaction persistence
  (d) Behavioral metrics showing no significant differences:
      - Number of events
      - Pulling ratio
      - Flailing
      - Time to first major push (min)

Usage:
    python run_all_panels.py [--test] [--n-bins N] [--n-permutations N]

Arguments:
    --test:            Test mode — limited data for quick verification of all panels
    --n-bins:          Number of time bins for panel (a) (default: 12)
    --n-permutations:  Number of permutations for all tests (default: 10000)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path, extra_args=None):
    """Run a Python script and return True if successful."""
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    if extra_args:
        print(f"Args: {' '.join(extra_args)}")
    print(f"{'='*60}")

    try:
        subprocess.run(cmd, check=True, cwd=script_path.parent)
        print(f"✅ {script_path.name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_path.name} failed with exit code {e.returncode}")
        return False


def main():
    """Main function to run all panel scripts for Extended Data Figure 10."""
    parser = argparse.ArgumentParser(
        description="Generate all panels of Extended Data Figure 10 (nutritional state)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_panels.py                         # Full analysis
  python run_all_panels.py --test                  # Quick test with limited data
  python run_all_panels.py --n-permutations 1000   # Fewer permutations (faster)
  python run_all_panels.py --n-bins 12             # 5-min segments (default)
        """,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: limited data for quick verification")
    parser.add_argument(
        "--n-bins",
        type=int,
        default=12,
        help="Number of time bins for panel (a) (default: 12 = 5-min segments for 60-min experiment)",
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
    shared_args.extend(["--n-permutations", str(args.n_permutations)])

    results = {}

    # Panel (a): trajectories
    results["a"] = run_script(
        script_dir / "edfigure10_a_trajectories.py",
        shared_args + ["--n-bins", str(args.n_bins)],
    )

    # Panel (b): final position distribution
    results["b"] = run_script(
        script_dir / "edfigure10_b_final_position.py",
    )

    # Panel (c): significant metrics
    results["c"] = run_script(
        script_dir / "edfigure10_c_metrics_significant.py",
        shared_args,
    )

    # Panel (d): non-significant metrics
    results["d"] = run_script(
        script_dir / "edfigure10_d_metrics_nonsignificant.py",
        shared_args,
    )

    # Summary
    print(f"\n{'='*60}")
    print("Extended Data Figure 10 — Summary")
    print(f"{'='*60}")
    for panel, success in results.items():
        status = "✅" if success else "❌"
        print(f"  Panel ({panel}): {status}")

    n_success = sum(results.values())
    n_total = len(results)
    print(f"\n{n_success}/{n_total} panels completed successfully.")
    if n_success < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
