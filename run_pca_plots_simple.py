#!/usr/bin/env python3
"""
Quick PCA Plotting Script Runner
Simplified version to run interactive or static scripts with minimal verbosity.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_script_simple(script_path):
    """Run a script and return success status"""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path) if os.path.dirname(script_path) else os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out"
    except Exception as e:
        return False, "", str(e)

def main():
    parser = argparse.ArgumentParser(description="Run PCA plotting scripts")
    parser.add_argument("--type", choices=["interactive", "static", "all"],
                       default="all", help="Type of scripts to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")

    args = parser.parse_args()

    # Script definitions
    interactive_scripts = [
        "interactive_scatterplot_matrix_individual.py",
        "interactive_scatterplot_matrix_pca.py",
        "interactive_scatterplot_matrix_temporal_individual.py",
        "interactive_scatterplot_matrix_temporal_pca.py"
    ]

    static_scripts = [
        "static_scatterplot_matrix_individual.py",
        "static_scatterplot_matrix_pca.py",
        "static_scatterplot_matrix_temporal_individual.py",
        "static_scatterplot_matrix_temporal_pca.py"
    ]

    # Select scripts to run
    scripts_to_run = []
    if args.type in ["interactive", "all"]:
        scripts_to_run.extend(interactive_scripts)
    if args.type in ["static", "all"]:
        scripts_to_run.extend(static_scripts)

    # Change to PCA directory
    script_dir = Path.cwd() / "src" / "PCA"
    if not script_dir.exists():
        print("❌ PCA directory not found. Run from project root.")
        return 1

    print(f"📁 Script directory: {script_dir}")
    os.chdir(script_dir)
    print(f"📁 Changed to: {os.getcwd()}")
    print(f"📁 Files in directory: {list(Path('.').glob('static_*.py'))}")

    print(f"🚀 Running {len(scripts_to_run)} {args.type} PCA scripts...")

    successful = 0
    failed = 0

    for script in scripts_to_run:
        script_path = script_dir / script
        if not script_path.exists():
            print(f"❌ {script} - File not found")
            failed += 1
            continue

        print(f"⏳ Running {script}...", end="", flush=True)

        success, stdout, stderr = run_script_simple(str(script_path))

        if success:
            print(" ✅")
            successful += 1
            if args.verbose and stdout:
                print(f"   Output: {stdout.strip()[-100:]}")  # Last 100 chars
        else:
            print(" ❌")
            failed += 1
            if args.verbose and stderr:
                print(f"   Error: {stderr.strip()[:200]}")  # First 200 chars

    print(f"\n📊 Results: {successful} successful, {failed} failed")

    if args.type == "interactive":
        print("📁 Interactive plots saved as .html files")
    elif args.type == "static":
        print("📁 Static plots saved as .png/.svg/.pdf files")
    else:
        print("📁 All plots generated in multiple formats")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
