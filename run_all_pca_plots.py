#!/usr/bin/env python3
"""
Master Script to Run All PCA Plotting Scripts
This script executes all interactive and static PCA plotting scripts with progress tracking and error handling.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a header with formatting"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")

def print_step(step_num, total_steps, script_name):
    """Print current step information"""
    print(f"{Colors.BOLD}{Colors.BLUE}[{step_num}/{total_steps}]{Colors.END} Running: {Colors.WHITE}{script_name}{Colors.END}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")

def run_script(script_path, script_type="python", timeout=600):
    """
    Run a script and capture output with timeout

    Args:
        script_path (str): Path to the script
        script_type (str): Type of script (python, etc.)
        timeout (int): Timeout in seconds

    Returns:
        tuple: (success, stdout, stderr, execution_time)
    """
    start_time = time.time()

    try:
        if script_type == "python":
            cmd = [sys.executable, script_path]
        else:
            cmd = [script_path]

        print_info(f"Command: {' '.join(cmd)}")
        print_info(f"Working directory: {os.getcwd()}")

        # Run the script
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(script_path) if os.path.dirname(script_path) else os.getcwd()
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            execution_time = time.time() - start_time

            success = process.returncode == 0
            return success, stdout, stderr, execution_time

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            return False, stdout, f"Script timed out after {timeout} seconds\n{stderr}", execution_time

    except Exception as e:
        execution_time = time.time() - start_time
        return False, "", str(e), execution_time

def check_prerequisites():
    """Check if required data files exist"""
    print_info("Checking prerequisites...")

    required_files = [
        "static_pca_with_metadata_tailoredctrls.feather",
        "fpca_temporal_with_metadata_tailoredctrls.feather",
    ]

    optional_files = [
        "static_pca_stats_results_allmethods_tailoredctrls.csv",
        "fpca_temporal_stats_results_allmethods_tailoredctrls.csv",
    ]

    missing_required = []
    missing_optional = []

    # Check required files
    for file in required_files:
        if not os.path.exists(file):
            missing_required.append(file)
        else:
            print_success(f"Found required file: {file}")

    # Check optional files
    for file in optional_files:
        if not os.path.exists(file):
            missing_optional.append(file)
        else:
            print_success(f"Found optional file: {file}")

    if missing_required:
        print_error("Missing required files:")
        for file in missing_required:
            print(f"  • {file}")
        return False

    if missing_optional:
        print_warning("Missing optional files (plots will work but without significance annotations):")
        for file in missing_optional:
            print(f"  • {file}")

    return True

def main():
    """Main function to run all scripts"""
    print_header("PCA Plotting Scripts Runner")
    print(f"{Colors.BOLD}Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}\n")

    # Check if we're in the right directory
    current_dir = Path.cwd()
    print_info(f"Current directory: {current_dir}")

    # Expected script paths
    script_dir = current_dir / "src" / "PCA"

    if not script_dir.exists():
        print_error(f"PCA script directory not found: {script_dir}")
        print_info("Please run this script from the project root directory")
        return 1

    # Change to the PCA directory for execution
    os.chdir(script_dir)
    print_info(f"Changed working directory to: {script_dir}")

    # Check prerequisites
    if not check_prerequisites():
        print_error("Prerequisites check failed. Please ensure required data files are present.")
        return 1

    # Define all scripts to run
    scripts = [
        # Interactive scripts (HTML output)
        {
            "name": "Interactive Individual PCA",
            "path": "interactive_scatterplot_matrix_individual.py",
            "type": "python",
            "category": "interactive"
        },
        {
            "name": "Interactive PCA Averages",
            "path": "interactive_scatterplot_matrix_pca.py",
            "type": "python",
            "category": "interactive"
        },
        {
            "name": "Interactive Temporal Individual PCA",
            "path": "interactive_scatterplot_matrix_temporal_individual.py",
            "type": "python",
            "category": "interactive"
        },
        {
            "name": "Interactive Temporal PCA Averages",
            "path": "interactive_scatterplot_matrix_temporal_pca.py",
            "type": "python",
            "category": "interactive"
        },
        # Static scripts (PNG/SVG/PDF output)
        {
            "name": "Static Individual PCA",
            "path": "static_scatterplot_matrix_individual.py",
            "type": "python",
            "category": "static"
        },
        {
            "name": "Static PCA Averages",
            "path": "static_scatterplot_matrix_pca.py",
            "type": "python",
            "category": "static"
        },
        {
            "name": "Static Temporal Individual PCA",
            "path": "static_scatterplot_matrix_temporal_individual.py",
            "type": "python",
            "category": "static"
        },
        {
            "name": "Static Temporal PCA Averages",
            "path": "static_scatterplot_matrix_temporal_pca.py",
            "type": "python",
            "category": "static"
        },
    ]

    total_scripts = len(scripts)
    results = []

    print_header(f"Executing {total_scripts} Scripts")

    for i, script in enumerate(scripts, 1):
        print_step(i, total_scripts, script["name"])

        # Check if script exists
        script_path = script_dir / script["path"]
        if not script_path.exists():
            print_error(f"Script not found: {script_path}")
            results.append({
                "name": script["name"],
                "success": False,
                "error": "Script file not found",
                "time": 0
            })
            continue

        # Run the script
        success, stdout, stderr, exec_time = run_script(str(script_path), script["type"])

        if success:
            print_success(f"Completed in {exec_time:.1f}s")
            if stdout:
                # Show last few lines of output
                output_lines = stdout.strip().split('\n')
                if len(output_lines) > 5:
                    print_info("Last 5 lines of output:")
                    for line in output_lines[-5:]:
                        print(f"  {line}")
                else:
                    print_info("Output:")
                    for line in output_lines:
                        print(f"  {line}")
        else:
            print_error(f"Failed after {exec_time:.1f}s")
            if stderr:
                print(f"{Colors.RED}Error output:{Colors.END}")
                error_lines = stderr.strip().split('\n')
                # Show first 10 lines of error
                for line in error_lines[:10]:
                    print(f"  {line}")
                if len(error_lines) > 10:
                    print(f"  ... ({len(error_lines) - 10} more lines)")

        results.append({
            "name": script["name"],
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "time": exec_time,
            "category": script["category"]
        })

        print()  # Add spacing between scripts

    # Print summary
    print_header("Execution Summary")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"{Colors.BOLD}Total scripts: {total_scripts}{Colors.END}")
    print_success(f"Successful: {len(successful)}")
    if failed:
        print_error(f"Failed: {len(failed)}")

    # Categorize results
    interactive_results = [r for r in results if r["category"] == "interactive"]
    static_results = [r for r in results if r["category"] == "static"]

    print(f"\n{Colors.BOLD}Interactive Scripts (HTML output):{Colors.END}")
    for result in interactive_results:
        status = "✓" if result["success"] else "✗"
        color = Colors.GREEN if result["success"] else Colors.RED
        print(f"  {color}{status} {result['name']} ({result['time']:.1f}s){Colors.END}")

    print(f"\n{Colors.BOLD}Static Scripts (PNG/SVG/PDF output):{Colors.END}")
    for result in static_results:
        status = "✓" if result["success"] else "✗"
        color = Colors.GREEN if result["success"] else Colors.RED
        print(f"  {color}{status} {result['name']} ({result['time']:.1f}s){Colors.END}")

    # Show failed scripts details
    if failed:
        print(f"\n{Colors.BOLD}Failed Scripts Details:{Colors.END}")
        for result in failed:
            print_error(f"• {result['name']}")
            if result.get('stderr'):
                print(f"  Error: {result['stderr'][:200]}...")

    total_time = sum(r["time"] for r in results)
    print(f"\n{Colors.BOLD}Total execution time: {total_time:.1f}s{Colors.END}")
    print(f"{Colors.BOLD}Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")

    # Return appropriate exit code
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
