#!/usr/bin/env python3
"""
Example usage script for TrajectoryMetricsAnalysis.py

This script demonstrates how to use the trajectory metrics analysis tool
with different command-line options.
"""

import subprocess
import sys
from pathlib import Path

def run_analysis_example():
    """Run example analysis commands."""

    script_path = Path(__file__).parent / "TrajectoryMetricsAnalysis.py"

    print("=== Trajectory Metrics Analysis Examples ===\n")

    # Example 1: Single YAML file
    print("Example 1: Analyze flies from a single YAML file")
    print("Command:")
    cmd1 = f"python {script_path} --yaml experiments_yaml/control_folders.yaml --output_dir outputs/single_yaml_analysis"
    print(cmd1)
    print()

    # Example 2: Multiple YAML files
    print("Example 2: Analyze flies from multiple YAML files")
    print("Command:")
    cmd2 = f"python {script_path} --yaml experiments_yaml/control_folders.yaml experiments_yaml/test_experiment.yaml --output_dir outputs/multi_yaml_analysis --save_data"
    print(cmd2)
    print()

    # Example 3: With custom minimum sample size
    print("Example 3: Analyze with custom minimum sample size")
    print("Command:")
    cmd3 = f"python {script_path} --yaml experiments_yaml/control_folders.yaml --min_samples 5 --output_dir outputs/min_samples_analysis --save_data"
    print(cmd3)
    print()

    # Ask user if they want to run any example
    print("Would you like to run any of these examples? (1/2/3/n for none): ", end="")
    choice = input().strip()

    commands = {
        "1": cmd1,
        "2": cmd2,
        "3": cmd3
    }

    if choice in commands:
        print(f"\nRunning example {choice}...")
        try:
            result = subprocess.run(commands[choice], shell=True, check=True)
            print("Analysis completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error running analysis: {e}")
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user.")
    elif choice.lower() == 'n':
        print("No analysis run.")
    else:
        print("Invalid choice.")

def show_help():
    """Show detailed help for the analysis script."""
    script_path = Path(__file__).parent / "TrajectoryMetricsAnalysis.py"

    print("=== Trajectory Metrics Analysis Help ===\n")
    subprocess.run([sys.executable, str(script_path), "--help"])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        run_analysis_example()
