#!/usr/bin/env python3
"""
Complete PCA Analysis Pipeline
Runs the full static PCA analysis workflow in sequence:
1. PCA_Static.py - Main PCA analysis
2. plot_pc_loadings.py - Generate loadings heatmap
3. plot_detailed_PC_statistics.py - Detailed PC statistics
4. plot_hits_heatmap.py - Generate hits heatmap

All outputs are organized in a timestamped results directory for easy review.
"""

import subprocess
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path


def create_results_directory():
    """Create a timestamped results directory for this analysis run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"pca_analysis_results_{timestamp}"

    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)

    # Create subdirectories for organization
    subdirs = ["data_files", "plots", "statistics", "logs"]
    for subdir in subdirs:
        os.makedirs(f"{results_dir}/{subdir}", exist_ok=True)

    print(f"📁 Created results directory: {results_dir}/")
    return results_dir


def run_script_with_logging(script_name, description, results_dir):
    """Run a script and capture its output"""
    print(f"\n{'='*80}")
    print(f"🚀 STEP: {description}")
    print(f"   Script: {script_name}")
    print(f"{'='*80}")

    # Log file for this step
    log_file = f"{results_dir}/logs/{script_name.replace('.py', '.log')}"

    try:
        # Run the script and capture output
        # Set working directory to src/ so PCA module imports work correctly
        src_dir = os.path.join(os.path.dirname(os.getcwd()), ".") if "PCA" in os.getcwd() else os.getcwd()
        if "PCA" in os.getcwd():
            src_dir = os.path.dirname(os.getcwd())
            script_path = os.path.join("PCA", script_name)
        else:
            src_dir = os.getcwd()
            script_path = script_name

        result = subprocess.run(
            [sys.executable, script_path, results_dir], check=True, capture_output=True, text=True, cwd=src_dir
        )

        # Save output to log file
        with open(log_file, "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\n")
            f.write(f"STDERR:\n{result.stderr}\n")

        # Print key output to console
        if result.stdout:
            print("📋 Script output:")
            print(result.stdout)

        print(f"✅ {description} completed successfully!")
        print(f"📄 Full log saved to: {log_file}")
        return True

    except subprocess.CalledProcessError as e:
        error_msg = f"❌ {description} failed with exit code {e.returncode}"
        print(error_msg)

        # Save error to log file
        with open(log_file, "w") as f:
            f.write(f"ERROR: Exit code {e.returncode}\n\n")
            f.write(f"STDOUT:\n{e.stdout}\n\n")
            f.write(f"STDERR:\n{e.stderr}\n")

        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)

        print(f"📄 Error log saved to: {log_file}")
        return False

    except FileNotFoundError:
        error_msg = f"❌ Script not found: {script_name}"
        print(error_msg)

        with open(log_file, "w") as f:
            f.write(f"ERROR: Script not found - {script_name}\n")

        return False


def organize_outputs(results_dir):
    """Organize all generated files that are already in the results directory"""
    print(f"\n{'='*80}")
    print("📦 ORGANIZING OUTPUTS")
    print(f"{'='*80}")

    # Define file patterns and their destinations within the results directory
    file_patterns = {
        "data_files": [
            "static_*pca_scores.csv",
            "static_*pca_loadings.csv",
            "static_*pca_with_metadata_*.feather",
            "static_*pca_stats_*_tailoredctrls.csv",
            "pooled_summary_harmonized.feather",
        ],
        "plots": [
            "static_*pca_loadings_*.png",
            "static_*pca_loadings_*.pdf",
            "static_*pca_hits_detailed_heatmap.png",
            "static_*pca_hits_detailed_heatmap.pdf",
            "mannwhitney_static_tailored_split.png",
            "mannwhitney_static_tailored_split.pdf",
            "mannwhitney_static_tailored_split_log.png",
            "mannwhitney_static_tailored_split_log.pdf",
        ],
    }

    moved_files = {"data_files": [], "plots": [], "statistics": []}

    # Move files from results_dir root to organized subdirectories
    for category, patterns in file_patterns.items():
        category_dir = f"{results_dir}/{category}"
        for pattern in patterns:
            # Look for files in the results directory
            matching_files = list(Path(results_dir).glob(pattern))
            for file_path in matching_files:
                if file_path.exists() and file_path.parent == Path(results_dir):
                    dest_path = f"{category_dir}/{file_path.name}"
                    try:
                        shutil.move(str(file_path), dest_path)
                        moved_files[category].append(file_path.name)
                        print(f"   📄 {file_path.name} → {category}/")
                    except Exception as e:
                        print(f"   ⚠️  Could not move {file_path.name}: {e}")
                elif file_path.exists():
                    # File is already in the right subdirectory, just count it
                    moved_files[category].append(file_path.name)
                    print(f"   ✓ {file_path.name} (already organized)")

    # Generate summary
    print(f"\n📊 FILES ORGANIZED:")
    for category, files in moved_files.items():
        if files:
            print(f"   {category.upper()}: {len(files)} files")
            for file in files:
                print(f"      • {file}")
        else:
            print(f"   {category.upper()}: No files found")

    return moved_files


def generate_analysis_summary(results_dir, step_results, moved_files):
    """Generate a comprehensive summary of the analysis run"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_content = f"""
# PCA Analysis Pipeline Summary
Generated: {timestamp}
Results Directory: {results_dir}

## Pipeline Steps Executed

"""

    # Add step results
    steps = [
        ("PCA_Static.py", "Main PCA Analysis"),
        ("plot_pc_loadings.py", "Loadings Heatmap Generation"),
        ("plot_detailed_PC_statistics.py", "Detailed PC Statistics"),
        ("plot_hits_heatmap.py", "Hits Heatmap Generation"),
    ]

    for script, description in steps:
        status = "✅ SUCCESS" if step_results.get(script, False) else "❌ FAILED"
        summary_content += f"- {description}: {status}\n"

    summary_content += f"""

## Generated Files

### Data Files ({len(moved_files['data_files'])} files)
"""
    for file in moved_files["data_files"]:
        summary_content += f"- {file}\n"

    summary_content += f"""
### Plots ({len(moved_files['plots'])} files)
"""
    for file in moved_files["plots"]:
        summary_content += f"- {file}\n"

    summary_content += f"""

## Key Outputs to Review

1. **PCA Loadings Heatmap**: `plots/static_*pca_loadings_simplified_heatmap.png`
   - Shows which metrics contribute most to each PC

2. **Detailed PC Statistics**: `plots/static_*pca_hits_detailed_heatmap.png`
   - Shows which PCs are significant for each genotype
   - Red = upregulated, Blue = downregulated vs control

3. **Hits Heatmap**: `plots/mannwhitney_static_tailored_split.png`
   - Overview of significant genotypes (ball pushing hits)

4. **Data Files**: `data_files/`
   - PCA scores, loadings, and statistical results
   - Harmonized dataset for reproducibility

## Logs
- Individual step logs available in `logs/` directory
- Check logs if any step failed

## Usage
To view all plots quickly:
```bash
cd {results_dir}/plots
ls -la *.png
```

To load PCA results in Python:
```python
import pandas as pd
# Load PCA scores with metadata
scores = pd.read_feather("data_files/static_sparsepca_with_metadata_tailoredctrls.feather")
# Load detailed statistics
stats = pd.read_csv("data_files/static_sparsepca_stats_simplified_tailoredctrls.csv")
```
"""

    # Save summary
    summary_file = f"{results_dir}/ANALYSIS_SUMMARY.md"
    with open(summary_file, "w") as f:
        f.write(summary_content)

    print(f"📋 Analysis summary saved to: {summary_file}")
    return summary_file


def main():
    """Run the complete PCA analysis pipeline"""
    print("🧬 PCA ANALYSIS PIPELINE")
    print("=" * 80)
    print("This script will run the complete static PCA analysis workflow:")
    print("1. Main PCA analysis (PCA_Static.py)")
    print("2. Loadings heatmap (plot_pc_loadings.py)")
    print("3. Detailed PC statistics (plot_detailed_PC_statistics.py)")
    print("4. Hits heatmap (plot_hits_heatmap.py)")
    print("=" * 80)

    # Create results directory
    results_dir = create_results_directory()
    print(f"📂 All outputs will be saved to: {results_dir}")
    print("=" * 80)

    # Define the pipeline steps
    pipeline_steps = [
        ("PCA_Static.py", "Main PCA Analysis"),
        ("plot_pc_loadings.py", "Loadings Heatmap Generation"),
        ("plot_detailed_PC_statistics.py", "Detailed PC Statistics"),
        ("plot_hits_heatmap.py", "Hits Heatmap Generation"),
    ]

    # Track step results
    step_results = {}

    # Execute each step
    for script_name, description in pipeline_steps:
        success = run_script_with_logging(script_name, description, results_dir)
        step_results[script_name] = success

        # If a critical step fails, ask user if they want to continue
        if not success and script_name in ["PCA_Static.py"]:
            response = input(f"\n⚠️  Critical step failed: {description}\nContinue anyway? (y/n): ")
            if response.lower() != "y":
                print("❌ Pipeline aborted by user")
                return

    # Organize outputs
    moved_files = organize_outputs(results_dir)

    # Generate summary
    summary_file = generate_analysis_summary(results_dir, step_results, moved_files)

    # Final summary
    print(f"\n{'='*80}")
    print("🎉 PIPELINE COMPLETED!")
    print(f"{'='*80}")

    successful_steps = sum(step_results.values())
    total_steps = len(step_results)

    print(f"📊 RESULTS: {successful_steps}/{total_steps} steps completed successfully")
    print(f"📁 All outputs saved to: {results_dir}/")
    print(f"📋 Summary: {summary_file}")

    if successful_steps == total_steps:
        print("✅ All steps completed successfully!")
        print("\n🔍 QUICK REVIEW:")
        print(f"   • View plots: ls {results_dir}/plots/")
        print(f"   • Check summary: cat {summary_file}")
    else:
        print("⚠️  Some steps failed - check individual logs for details")
        failed_steps = [step for step, success in step_results.items() if not success]
        print(f"   Failed steps: {', '.join(failed_steps)}")

    print(f"\n📂 Navigate to results: cd {results_dir}")


if __name__ == "__main__":
    main()
