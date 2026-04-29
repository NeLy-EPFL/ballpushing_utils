#!/usr/bin/env python3
"""
Master script to generate all panels of Extended Data Figure 9.

Extended Data Figure 9: Effect of illumination on ball pushing in IR8a flies.

Panels:
  (a) Ball distance trajectory for IR8a, Light ON (orange) vs Light OFF (black)
      24 × 5-min bins, bootstrapped 95% CI, permutation tests with FDR correction
      n = 30 flies per condition
  (b) Pulling ratio for IR8a, Light ON (orange) vs Light OFF (black)
      Single permutation test (no FDR needed)

Usage:
    python run_all_panels.py [--test] [--n-bins N] [--n-permutations N]

Arguments:
    --test:            Test mode — limited data for quick verification of all panels
    --n-bins:          Number of time bins for panel (a) (default: 24)
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
    """Main function to run all panel scripts for Extended Data Figure 9."""
    parser = argparse.ArgumentParser(
        description="Generate all panels of Extended Data Figure 9 (IR8a light effect)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_panels.py                         # Full analysis
  python run_all_panels.py --test                  # Quick test with limited data
  python run_all_panels.py --n-permutations 1000   # Fewer permutations (faster)
  python run_all_panels.py --n-bins 24             # 5-min segments (default)
        """,
    )
    parser.add_argument("--test", action="store_true", help="Test mode: limited data for quick verification")
    parser.add_argument(
        "--n-bins",
        type=int,
        default=24,
        help="Number of time bins for panel (a) (default: 24 = 5-min segments for 2-h experiment)",
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

    # Panel (a): IR8a trajectories
    results["a"] = run_script(
        script_dir / "edfigure9_a_trajectories.py",
        shared_args + ["--n-bins", str(args.n_bins)],
    )

    # Panel (b): pulling ratio boxplot
    results["b"] = run_script(
        script_dir / "edfigure9_b_pulling_ratio.py",
        shared_args,
    )

    # Summary
    print(f"\n{'='*60}")
    print("Extended Data Figure 9 — Summary")
    print(f"{'='*60}")
    for panel, success in results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"  Panel ({panel}): {status}")

    n_failed = sum(1 for s in results.values() if not s)
    if n_failed == 0:
        print(f"\n✅ All {len(results)} panels generated successfully.")
    else:
        print(f"\n⚠️  {n_failed}/{len(results)} panel(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
