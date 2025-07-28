#!/usr/bin/env python3
"""
Master Script for Generating All PCA Scatterplot Matrices
Generates both interactive and static versions of all plot types:
- Individual flies vs Genotype averages
- Static PCA vs Temporal PCA (fPCA)

This script replaces the need for 4 separate scripts by using a unified approach.
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import plot configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from plot_configurations import list_available_configs

def run_script(script_name, plot_type, n_components=6, verbose=False, additional_args=None):
    """Run a unified script with the specified plot type"""
    # Get the full path to the script since we're running from workspace root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_script_path = os.path.join(script_dir, script_name)

    cmd = [sys.executable, full_script_path, plot_type]

    if n_components != 6:
        cmd.extend(["--n-components", str(n_components)])

    if verbose:
        cmd.append("--verbose")

    if additional_args:
        cmd.extend(additional_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        # Run from the workspace root: src/PCA -> src -> ballpushing_utils
        workspace_root = os.path.dirname(os.path.dirname(script_dir))
        result = subprocess.run(cmd, check=True, cwd=workspace_root)
        print(f"✓ Successfully generated {plot_type}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate {plot_type}: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_name}")
        return False

def generate_all_plots(plot_types=None, static_only=False, interactive_only=False, n_components=6, verbose=False):
    """Generate all specified plot types"""

    # Default to all plot types if none specified
    if plot_types is None:
        plot_types = ["individual_static", "genotype_static", "individual_temporal", "genotype_temporal"]

    # Determine which script types to run
    run_static = not interactive_only
    run_interactive = not static_only

    results = {"static": {}, "interactive": {}}
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for plot_type in plot_types:
        print(f"\n🎯 Generating {plot_type}...")

        # Run static version
        if run_static:
            static_script = os.path.join(script_dir, "unified_static_scatterplot_matrix.py")
            results["static"][plot_type] = run_script(
                static_script, plot_type, n_components, verbose
            )

        # Run interactive version
        if run_interactive:
            interactive_script = os.path.join(script_dir, "unified_interactive_scatterplot_matrix.py")
            results["interactive"][plot_type] = run_script(
                interactive_script, plot_type, n_components, verbose
            )

    return results

def print_summary(results):
    """Print a summary of all generated plots"""
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")

    total_success = 0
    total_attempted = 0

    for script_type, plot_results in results.items():
        if not plot_results:  # Skip if no plots were attempted for this script type
            continue

        print(f"\n{script_type.upper()} PLOTS:")
        for plot_type, success in plot_results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {plot_type:<25} {status}")
            total_attempted += 1
            if success:
                total_success += 1

    print(f"\nOVERALL: {total_success}/{total_attempted} plots generated successfully")

    if total_success == total_attempted:
        print("🎉 All plots generated successfully!")
    elif total_success > 0:
        print("⚠️  Some plots failed to generate. Check the output above for details.")
    else:
        print("❌ No plots were generated successfully.")

def generate_legend():
    """Generate the combined PCA legend"""
    print(f"\n{'='*60}")
    print("Generating PCA Legend...")
    print(f"{'='*60}")

    try:
        # Run the legend creation script
        legend_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_pca_legends.py")
        result = subprocess.run([sys.executable, legend_script],
                              check=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True)
        print("✓ Successfully generated PCA legend")
        # Print the script's output
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate legend: {e}")
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print("✗ Legend script not found: create_pca_legends.py")
        return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Master script for generating all PCA scatterplot matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots (both static and interactive)
  python master_scatterplot_generator.py

  # Generate only static plots
  python master_scatterplot_generator.py --static-only

  # Generate only individual fly plots
  python master_scatterplot_generator.py --plot-types individual_static individual_temporal

  # Generate with verbose output and 4 components
  python master_scatterplot_generator.py --verbose --n-components 4
        """
    )

    parser.add_argument("--plot-types", nargs="+",
                       choices=["individual_static", "genotype_static", "individual_temporal", "genotype_temporal"],
                       help="Specific plot types to generate (default: all)")
    parser.add_argument("--static-only", action="store_true",
                       help="Generate only static plots")
    parser.add_argument("--interactive-only", action="store_true",
                       help="Generate only interactive plots")
    parser.add_argument("--n-components", type=int, default=6,
                       help="Number of PCA components to include (default: 6)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--list-configs", action="store_true",
                       help="List all available plot configurations and exit")

    args = parser.parse_args()

    # Handle list configs
    if args.list_configs:
        list_available_configs()
        return

    # Validate arguments
    if args.static_only and args.interactive_only:
        parser.error("Cannot specify both --static-only and --interactive-only")

    # Print header
    print("🚀 PCA Scatterplot Matrix Master Generator")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.verbose:
        print(f"Plot types: {args.plot_types or 'all'}")
        print(f"Static only: {args.static_only}")
        print(f"Interactive only: {args.interactive_only}")
        print(f"Components: {args.n_components}")

    # Generate plots
    results = generate_all_plots(
        plot_types=args.plot_types,
        static_only=args.static_only,
        interactive_only=args.interactive_only,
        n_components=args.n_components,
        verbose=args.verbose
    )

    # Generate the combined legend
    legend_success = generate_legend()

    # Print summary
    print_summary(results)

    # Add legend to summary
    if legend_success:
        print("\n📊 LEGEND:")
        print("  pca_matrices/pca_legend_combined.png      ✓ SUCCESS")
    else:
        print("\n📊 LEGEND:")
        print("  pca_matrices/pca_legend_combined.png      ✗ FAILED")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
