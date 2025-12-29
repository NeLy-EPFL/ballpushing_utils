#!/usr/bin/env python3
"""
Complete PCA Analysis Pipeline
Runs the full static PCA analysis workflow in sequence:
1. PCA_Static.py - Main PCA analysis
2. plot_pc_loadings.py - Generate loadings heatmap
3. plot_detailed_PC_statistics.py - Detailed PC statistics
4. plot_hits_heatmap.py - Generate hits heatmap
5. plot_detailed_metric_statistics.py - Detailed metric statistics

Supports running with different control modes:
- tailored: Use split-based tailored controls (default)
- emptysplit: Use Empty-Split as universal control
- both: Run both modes and compare results

All outputs are organized in a timestamped results directory for easy review.
"""
# TODO: Rework this with new PCA pipeline structure
import subprocess
import sys
import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Global variable for statistical testing mode
STAT_MODE = "permutation"  # Default value (matches argparse default)


def compare_control_modes(tailored_dir, emptysplit_dir):
    """
    Compare results between tailored controls and Empty-Split control modes.
    Identifies hits that are unique to each mode and hits common to both.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    print("📊 Loading results from both control modes...")

    # Create comparison output directory
    comparison_dir = "comparison_tailored_vs_emptysplit"
    os.makedirs(comparison_dir, exist_ok=True)

    # Find the consistency scores files
    tailored_stats = None
    emptysplit_stats = None

    for file in Path(tailored_dir).rglob("enhanced_consistency_scores.csv"):
        tailored_stats = file
        break

    for file in Path(emptysplit_dir).rglob("enhanced_consistency_scores.csv"):
        emptysplit_stats = file
        break

    if not tailored_stats or not emptysplit_stats:
        print("⚠️  Could not find enhanced_consistency_scores.csv files for comparison")
        print(f"  Looking in:")
        print(f"    Tailored: {tailored_dir}")
        print(f"    EmptySplit: {emptysplit_dir}")
        return

    print(f"  • Tailored: {tailored_stats}")
    print(f"  • Empty-Split: {emptysplit_stats}")

    # Load data
    df_tailored = pd.read_csv(tailored_stats)
    df_emptysplit = pd.read_csv(emptysplit_stats)

    # Identify significant hits (genotypes with >0 hits across configurations)
    # A genotype is considered a hit if it was significant in at least one configuration
    hits_tailored = set(df_tailored[df_tailored["Total_Hit_Count"] > 0]["Genotype"])
    hits_emptysplit = set(df_emptysplit[df_emptysplit["Total_Hit_Count"] > 0]["Genotype"])

    # Calculate overlaps
    hits_both = hits_tailored & hits_emptysplit
    hits_tailored_only = hits_tailored - hits_emptysplit
    hits_emptysplit_only = hits_emptysplit - hits_tailored

    print(f"\n📈 COMPARISON RESULTS:")
    print(f"{'='*60}")
    print(f"Hits with tailored controls:        {len(hits_tailored):3d}")
    print(f"Hits with Empty-Split control:      {len(hits_emptysplit):3d}")
    print(f"{'='*60}")
    print(f"Hits in BOTH modes:                 {len(hits_both):3d}")
    print(f"Hits ONLY with tailored controls:   {len(hits_tailored_only):3d}")
    print(f"Hits ONLY with Empty-Split control: {len(hits_emptysplit_only):3d}")
    print(f"{'='*60}")

    # Create detailed comparison CSV
    all_genotypes = hits_tailored | hits_emptysplit
    comparison_data = []

    for genotype in sorted(all_genotypes):
        in_tailored = genotype in hits_tailored
        in_emptysplit = genotype in hits_emptysplit

        # Get consistency scores
        consistency_tailored = df_tailored[df_tailored["Genotype"] == genotype]["Overall_Consistency"].values
        consistency_emptysplit = df_emptysplit[df_emptysplit["Genotype"] == genotype]["Overall_Consistency"].values

        cons_t = consistency_tailored[0] if len(consistency_tailored) > 0 else 0.0
        cons_e = consistency_emptysplit[0] if len(consistency_emptysplit) > 0 else 0.0

        # Get hit counts
        hitcount_t = df_tailored[df_tailored["Genotype"] == genotype]["Total_Hit_Count"].values
        hitcount_e = df_emptysplit[df_emptysplit["Genotype"] == genotype]["Total_Hit_Count"].values

        hits_t = hitcount_t[0] if len(hitcount_t) > 0 else 0
        hits_e = hitcount_e[0] if len(hitcount_e) > 0 else 0

        category = "Both"
        if in_tailored and not in_emptysplit:
            category = "Tailored only"
        elif in_emptysplit and not in_tailored:
            category = "Empty-Split only"

        comparison_data.append(
            {
                "Genotype": genotype,
                "Category": category,
                "Hit_Tailored": in_tailored,
                "Hit_EmptySplit": in_emptysplit,
                "HitCount_Tailored": hits_t,
                "HitCount_EmptySplit": hits_e,
                "Consistency_Tailored": cons_t,
                "Consistency_EmptySplit": cons_e,
                "Consistency_Difference": abs(cons_t - cons_e),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(["Category", "Consistency_Tailored"], ascending=[True, False])

    # Save comparison CSV
    csv_path = os.path.join(comparison_dir, "control_mode_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed comparison saved to: {csv_path}")

    # Create summary text file
    summary_path = os.path.join(comparison_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CONTROL MODE COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Tailored Controls:        {len(hits_tailored):3d} hits\n")
        f.write(f"Empty-Split Control:      {len(hits_emptysplit):3d} hits\n\n")
        f.write(f"Hits in BOTH modes:       {len(hits_both):3d}\n")
        f.write(f"Tailored ONLY:            {len(hits_tailored_only):3d}\n")
        f.write(f"Empty-Split ONLY:         {len(hits_emptysplit_only):3d}\n\n")

        f.write("\nHITS IN BOTH MODES (most robust):\n")
        f.write("-" * 60 + "\n")
        for genotype in sorted(hits_both):
            cons_t = comparison_df[comparison_df["Genotype"] == genotype]["Consistency_Tailored"].values[0]
            cons_e = comparison_df[comparison_df["Genotype"] == genotype]["Consistency_EmptySplit"].values[0]
            hits_t = comparison_df[comparison_df["Genotype"] == genotype]["HitCount_Tailored"].values[0]
            hits_e = comparison_df[comparison_df["Genotype"] == genotype]["HitCount_EmptySplit"].values[0]
            f.write(
                f"{genotype:30s}  consistency_t={cons_t:.2%} ({hits_t} hits)  consistency_e={cons_e:.2%} ({hits_e} hits)\n"
            )

        f.write("\n\nHITS ONLY WITH TAILORED CONTROLS:\n")
        f.write("-" * 60 + "\n")
        for genotype in sorted(hits_tailored_only):
            cons_t = comparison_df[comparison_df["Genotype"] == genotype]["Consistency_Tailored"].values[0]
            cons_e = comparison_df[comparison_df["Genotype"] == genotype]["Consistency_EmptySplit"].values[0]
            hits_t = comparison_df[comparison_df["Genotype"] == genotype]["HitCount_Tailored"].values[0]
            hits_e = comparison_df[comparison_df["Genotype"] == genotype]["HitCount_EmptySplit"].values[0]
            f.write(
                f"{genotype:30s}  consistency_t={cons_t:.2%} ({hits_t} hits)  consistency_e={cons_e:.2%} ({hits_e} hits)\n"
            )

        f.write("\n\nHITS ONLY WITH EMPTY-SPLIT CONTROL:\n")
        f.write("-" * 60 + "\n")
        for genotype in sorted(hits_emptysplit_only):
            cons_t = comparison_df[comparison_df["Genotype"] == genotype]["Consistency_Tailored"].values[0]
            cons_e = comparison_df[comparison_df["Genotype"] == genotype]["Consistency_EmptySplit"].values[0]
            hits_t = comparison_df[comparison_df["Genotype"] == genotype]["HitCount_Tailored"].values[0]
            hits_e = comparison_df[comparison_df["Genotype"] == genotype]["HitCount_EmptySplit"].values[0]
            f.write(
                f"{genotype:30s}  consistency_t={cons_t:.2%} ({hits_t} hits)  consistency_e={cons_e:.2%} ({hits_e} hits)\n"
            )

    print(f"💾 Summary saved to: {summary_path}")

    # Create Venn diagram visualization
    try:
        from matplotlib_venn import venn2

        plt.figure(figsize=(10, 8))
        venn = venn2(
            [hits_tailored, hits_emptysplit],
            set_labels=("Tailored Controls", "Empty-Split Control"),
        )
        plt.title(
            f"Control Mode Comparison: Significant Hits\n(Total unique hits: {len(all_genotypes)})",
            fontsize=14,
            fontweight="bold",
        )

        venn_path = os.path.join(comparison_dir, "control_mode_venn.png")
        plt.savefig(venn_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"💾 Venn diagram saved to: {venn_path}")
    except ImportError:
        print("⚠️  matplotlib_venn not available, skipping Venn diagram")

    # Create comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Both modes", "Tailored only", "Empty-Split only"]
    counts = [len(hits_both), len(hits_tailored_only), len(hits_emptysplit_only)]
    colors = ["green", "blue", "orange"]

    ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel("Number of Genotypes", fontsize=12)
    ax.set_title("Comparison of Hits by Control Mode", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2)

    # Add count labels on bars
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, count + max(counts) * 0.02, str(count), ha="center", fontsize=12, fontweight="bold")

    bar_path = os.path.join(comparison_dir, "control_mode_comparison.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Comparison bar plot saved to: {bar_path}")

    # Print detailed lists
    print(f"\n🏆 HITS IN BOTH MODES ({len(hits_both)}):")
    for genotype in sorted(hits_both)[:20]:  # Show first 20
        print(f"   • {genotype}")
    if len(hits_both) > 20:
        print(f"   ... and {len(hits_both) - 20} more")

    if hits_tailored_only:
        print(f"\n🔵 HITS ONLY WITH TAILORED CONTROLS ({len(hits_tailored_only)}):")
        for genotype in sorted(hits_tailored_only)[:20]:
            print(f"   • {genotype}")
        if len(hits_tailored_only) > 20:
            print(f"   ... and {len(hits_tailored_only) - 20} more")

    if hits_emptysplit_only:
        print(f"\n🟠 HITS ONLY WITH EMPTY-SPLIT CONTROL ({len(hits_emptysplit_only)}):")
        for genotype in sorted(hits_emptysplit_only)[:20]:
            print(f"   • {genotype}")
        if len(hits_emptysplit_only) > 20:
            print(f"   ... and {len(hits_emptysplit_only) - 20} more")

    print(f"\n✅ Comparison complete! All results in: {comparison_dir}/")


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


def run_script_with_logging(script_name, description, results_dir, control_mode="tailored", dataset_path=None):
    """Run a script and capture its output"""
    print(f"\n{'='*80}")
    print(f"🚀 STEP: {description}")
    print(f"   Script: {script_name}")
    print(f"   Control mode: {control_mode}")
    if dataset_path:
        print(f"   Dataset: {dataset_path}")
    print(f"{'='*80}")

    # Log file for this step
    log_file = f"{results_dir}/logs/{script_name.replace('.py', '')}_{control_mode}.log"

    try:
        # Run the script and capture output
        # Get the absolute path to the PCA directory
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_script_dir, script_name)

        # Set working directory to where we run from (typically src/)
        # This keeps relative paths in output working as expected
        src_dir = os.getcwd()

        # Build command with control mode argument for scripts that support it
        if script_name == "plot_detailed_metric_statistics.py":
            # This script uses --output-dir instead of positional results_dir
            # Pass consistency-dir pointing to data_files subdirectory where PCA_Static.py saves consistency CSVs
            consistency_dir = os.path.join(results_dir, "data_files")
            cmd = [
                sys.executable,
                script_path,
                "--output-dir",
                results_dir,
                "--consistency-dir",
                consistency_dir,
                "--control-mode",
                control_mode,
            ]
            if dataset_path:
                cmd.extend(["--data-path", dataset_path])
        elif script_name == "simple_consistency_plot.py":
            # Simple consistency plot takes analysis_dir as positional arg and output dir as -o
            # Point it to data_files subdirectory where consistency CSVs are located
            consistency_dir = os.path.join(results_dir, "data_files")
            plots_dir = os.path.join(results_dir, "plots")
            cmd = [
                sys.executable,
                script_path,
                consistency_dir,
                "-o",
                plots_dir,
                "--with-hits-only",
                "--control-mode",
                control_mode,
            ]
        elif script_name == "PCA_Static.py":
            # Add statistical testing flag based on STAT_MODE
            cmd = [sys.executable, script_path, results_dir, "--control-mode", control_mode]
            if dataset_path:
                cmd.extend(["--dataset", dataset_path])
            if STAT_MODE == "multivariate":
                cmd.append("--multivariate-only")
            elif STAT_MODE == "triple":
                cmd.append("--triple-test")
            elif STAT_MODE == "permutation":
                cmd.append("--permutation-only")
        else:
            cmd = [sys.executable, script_path, results_dir]
            if script_name in ["PCA_Static.py", "plot_detailed_PC_statistics.py", "plot_hits_heatmap.py"]:
                cmd.extend(["--control-mode", control_mode])
            # Add dataset path for scripts that support it
            if script_name == "plot_detailed_PC_statistics.py" and dataset_path:
                cmd.extend(["--dataset", dataset_path])

        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=src_dir)

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
            "static_*pca_stats_*_emptysplit.csv",
            "static_*sparsepca*_allmethods*_tailored*.csv",
            "static_*sparsepca*_allmethods*_emptysplit*.csv",
            "enhanced_consistency_scores.csv",
            "optimized_only_consistency_ranking.csv",
            "combined_consistency_ranking.csv",
            "enhanced_configuration_summary.csv",
            "pooled_summary_harmonized.feather",
            "high_consistency_overlap_analysis_*.txt",
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
            "mannwhitney_static_emptysplit*.png",
            "mannwhitney_static_emptysplit*.pdf",
            "enhanced_consistency_plots.png",
            "simple_consistency_scores*.png",
            "simple_consistency_scores*.pdf",
            "threshold_comparison.png",
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

    # Ensure root-level access to key CSVs (symlink or copy fallback)
    key_csvs = [
        "enhanced_consistency_scores.csv",
        "optimized_only_consistency_ranking.csv",
        "combined_consistency_ranking.csv",
    ]
    for fname in key_csvs:
        src = Path(results_dir) / "data_files" / fname
        dst = Path(results_dir) / fname
        if src.exists() and not dst.exists():
            try:
                # Create a relative symlink for portability
                dst.symlink_to(Path("data_files") / fname)
                print(f"   🔗 Created symlink: {dst.name} → data_files/{fname}")
            except Exception:
                try:
                    shutil.copy2(src, dst)
                    print(f"   📎 Copied file to root: {dst.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not expose {fname} at root: {e}")

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


def generate_analysis_summary(results_dir, step_results, moved_files, control_mode="tailored"):
    """Generate a comprehensive summary of the analysis run"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get dataset information
    from pathlib import Path

    # Import DATA_PATH from PCA_Static.py by reading it
    data_path = None
    data_info = ""
    pca_static_path = Path(__file__).parent / "PCA_Static.py"
    if pca_static_path.exists():
        with open(pca_static_path) as f:
            for line in f:
                if line.startswith("DATA_PATH ="):
                    # Extract the path from the line
                    data_path = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if data_path and os.path.exists(data_path):
        stat_info = os.stat(data_path)
        mod_time = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = stat_info.st_size / (1024 * 1024)
        data_info = f"""
## Dataset Information
- **Path**: `{data_path}`
- **Size**: {size_mb:.1f} MB
- **Last Modified**: {mod_time}
"""
    elif data_path:
        data_info = f"""
## Dataset Information
- **Path**: `{data_path}` ⚠️ (file not found)
"""

    summary_content = f"""
# PCA Analysis Pipeline Summary
Generated: {timestamp}
Results Directory: {results_dir}
Control Mode: {control_mode.upper()}
{data_info}
## Pipeline Steps Executed

"""

    # Add step results
    steps = [
        ("PCA_Static.py", "Main PCA Analysis"),
        ("plot_pc_loadings.py", "Loadings Heatmap Generation"),
        ("plot_detailed_PC_statistics.py", "Detailed PC Statistics"),
        ("plot_hits_heatmap.py", "Hits Heatmap Generation"),
        ("plot_detailed_metric_statistics.py", "Detailed Metric Statistics"),
        ("simple_consistency_plot.py", "Simple Consistency Visualization"),
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

4. **Simple Consistency Plots**: `plots/simple_consistency_scores_*.png`
   - Visual representation of genotype consistency scores
   - Shows which genotypes pass the 80% threshold
   - Includes both all-genotypes and hits-only views

5. **Data Files**: `data_files/`
   - PCA scores, loadings, and statistical results
   - Harmonized dataset for reproducibility
   - Consistency analysis results

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run complete PCA analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Control Modes:
  tailored (default): Use split-based tailored controls (n=Empty-Gal4, y=Empty-Split, m=TNTxPR)
  emptysplit:         Use Empty-Split for GAL4/split lines, TNTxPR for mutants (n/y=Empty-Split, m=TNTxPR)
  both:               Run both modes and generate comparison

Statistical Modes:
  triple (default):      Triple test (MW + Permutation + Mahalanobis, publication mode)
  permutation:           Permutation test only (most sensitive single-test)
  multivariate:          Permutation + Mahalanobis (sensitive, uses full PCA space)

Dataset Configuration:
  --dataset: Specify dataset path. Default is August 2024 dataset (250811_18) used in reproduced analysis.
             All scripts in pipeline will use the same dataset for consistency.

Examples:
  python run_complete_pca_pipeline.py                           # Both control modes, triple test (publication settings), August dataset
  python run_complete_pca_pipeline.py --control-mode tailored   # Tailored controls only, triple test
  python run_complete_pca_pipeline.py --stat-mode permutation   # Permutation test only (most sensitive)
  python run_complete_pca_pipeline.py --dataset /path/to/data   # Use custom dataset
        """,
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=["tailored", "emptysplit", "both"],
        default="both",
        help="Control selection mode (default: both for full rerun)",
    )
    parser.add_argument(
        "--stat-mode",
        type=str,
        choices=["multivariate", "triple", "permutation"],
        default="triple",
        help="Statistical testing mode: 'triple' (Mann-Whitney + Permutation + Mahalanobis, default, publication mode), 'permutation' (Permutation only), or 'multivariate' (Permutation + Mahalanobis)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather",
        help="Path to pooled_summary.feather dataset (default: August 2024 dataset matching reproduced analysis)",
    )
    args = parser.parse_args()

    print("🧬 PCA ANALYSIS PIPELINE")
    print("=" * 80)
    print("This script will run the complete static PCA analysis workflow:")
    print("1. Main PCA analysis (PCA_Static.py)")
    print("2. Loadings heatmap (plot_pc_loadings.py)")
    print("3. Detailed PC statistics (plot_detailed_PC_statistics.py)")
    print("4. Hits heatmap (plot_hits_heatmap.py)")
    print("5. Detailed metric statistics (plot_detailed_metric_statistics.py)")
    print("=" * 80)
    print(f"🎛️  Control mode: {args.control_mode}")
    print(f"⚖️  Statistical mode: {args.stat_mode}")
    print(f"📊 Dataset: {args.dataset}")
    print("=" * 80)

    # Determine which control modes to run
    if args.control_mode == "both":
        control_modes = ["tailored", "emptysplit"]
        run_comparison = True
    else:
        control_modes = [args.control_mode]
        run_comparison = False

    # Expose STAT_MODE to run_script_with_logging
    global STAT_MODE
    STAT_MODE = args.stat_mode

    all_results = {}

    for control_mode in control_modes:
        print(f"\n\n{'='*80}")
        print(f"📦 RUNNING PIPELINE WITH {control_mode.upper()} CONTROLS")
        print(f"{'='*80}\n")

        # Create results directory with control mode suffix - always include for clarity
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"pca_analysis_results_{control_mode}_{timestamp}"

        # Create main results directory
        os.makedirs(results_dir, exist_ok=True)

        # Create subdirectories for organization
        subdirs = ["data_files", "plots", "statistics", "logs"]
        for subdir in subdirs:
            os.makedirs(f"{results_dir}/{subdir}", exist_ok=True)

        print(f"📁 Created results directory: {results_dir}/")
        print(f"🎛️  Control mode: {control_mode}")
        print("=" * 80)

        # Define the pipeline steps
        pipeline_steps = [
            ("PCA_Static.py", "Main PCA Analysis"),
            ("plot_pc_loadings.py", "Loadings Heatmap Generation"),
            ("plot_detailed_PC_statistics.py", "Detailed PC Statistics"),
            ("plot_hits_heatmap.py", "Hits Heatmap Generation"),
            ("plot_detailed_metric_statistics.py", "Detailed Metric Statistics"),
            ("simple_consistency_plot.py", "Simple Consistency Visualization"),
        ]

        # Track step results
        step_results = {}

        # Execute each step
        for script_name, description in pipeline_steps:
            success = run_script_with_logging(script_name, description, results_dir, control_mode, args.dataset)
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
        summary_file = generate_analysis_summary(results_dir, step_results, moved_files, control_mode)

        # Store results for comparison
        all_results[control_mode] = {
            "results_dir": results_dir,
            "step_results": step_results,
            "summary_file": summary_file,
        }

        # Final summary for this control mode
        print(f"\n{'='*80}")
        print(f"🎉 PIPELINE COMPLETED FOR {control_mode.upper()} CONTROLS!")
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

    # Run comparison if both modes were executed
    if run_comparison and len(all_results) == 2:
        print(f"\n\n{'='*80}")
        print("🔍 GENERATING COMPARISON BETWEEN CONTROL MODES")
        print(f"{'='*80}\n")

        compare_control_modes(
            all_results["tailored"]["results_dir"],
            all_results["emptysplit"]["results_dir"],
        )

        # Also run the visual comparison script with brain-region colors and robust highlights
        try:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            vis_script = os.path.join(current_script_dir, "compare_control_modes_visual.py")
            vis_out_dir = "comparison_tailored_vs_emptysplit"
            cmd = [
                sys.executable,
                vis_script,
                "--tailored-dir",
                all_results["tailored"]["results_dir"],
                "--emptysplit-dir",
                all_results["emptysplit"]["results_dir"],
                "--output-dir",
                vis_out_dir,
                "--metric",
                "combined",  # Use combined consistency (includes edge cases)
            ]
            subprocess.run(cmd, check=True, text=True)
        except Exception as e:
            print(f"⚠️  Could not run visual comparison script: {e}")

        print("\n✅ Comparison analysis complete!")
        print(f"\nResults available in:")
        print(f"  • Tailored controls: {all_results['tailored']['results_dir']}")
        print(f"  • Empty-Split control: {all_results['emptysplit']['results_dir']}")
        print(f"  • Comparison: comparison_tailored_vs_emptysplit/")


if __name__ == "__main__":
    main()
