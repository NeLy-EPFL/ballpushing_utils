from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from Ballpushing_utils.fly import Fly
from Ballpushing_utils.trajectory_metrics import TrajectoryMetrics
import argparse
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Using matplotlib-only plotting.")


def compute_fly_metrics(fly_dir):
    """
    Compute trajectory metrics for a single fly.

    Args:
        fly_dir (str): Path to the fly directory

    Returns:
        dict: Dictionary containing fly metadata and computed metrics
    """
    try:
        fly = Fly(Path(fly_dir), as_individual=True)
        metrics = TrajectoryMetrics(fly)

        # Get fly metadata
        fly_name = getattr(fly.metadata, "name", Path(fly_dir).stem)
        nickname = getattr(fly.metadata, "nickname", "Unknown")
        genotype = getattr(fly.metadata, "Genotype", "Unknown")
        brain_region = getattr(fly.metadata, "brain_region", "Unknown")

        # Compute metrics
        contact_prop = metrics.contact_proportion()
        event_length = metrics.major_event_length()
        ball_displacement = metrics.ball_displacement_to_major_event()
        ball_displacement_during = metrics.ball_displacement_during_major_event()

        return {
            "fly_name": fly_name,
            "nickname": nickname,
            "genotype": genotype,
            "brain_region": brain_region,
            "contact_proportion": contact_prop,
            "major_event_length": event_length,
            "ball_displacement_to_major_event": ball_displacement,
            "ball_displacement_during_major_event": ball_displacement_during,
            "fly_dir": fly_dir
        }

    except Exception as e:
        print(f"Failed to process {fly_dir}: {e}")
        return None


def load_flies_from_yamls(yaml_files):
    """
    Load fly directories from one or more YAML files.

    Args:
        yaml_files (list): List of paths to YAML files

    Returns:
        list: List of fly directory paths
    """
    all_fly_dirs = []

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
                fly_dirs = data.get("directories", [])
                all_fly_dirs.extend(fly_dirs)
                print(f"Loaded {len(fly_dirs)} flies from {yaml_file}")
        except Exception as e:
            print(f"Failed to load {yaml_file}: {e}")

    print(f"Total flies loaded: {len(all_fly_dirs)}")
    return all_fly_dirs


def create_boxplot_with_scatter(df, metric_name, output_path=None, title=None):
    """
    Create a boxplot with superimposed scatterplot for a given metric.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        metric_name (str): Name of the metric column to plot
        output_path (str, optional): Path to save the plot
        title (str, optional): Custom title for the plot
    """
    # Filter out rows where the metric is None/NaN
    df_filtered = df.dropna(subset=[metric_name])

    if df_filtered.empty:
        print(f"No valid data for {metric_name}")
        return

    # Create figure
    plt.figure(figsize=(12, 8))

    if HAS_SEABORN:
        # Create boxplot with seaborn
        import seaborn as sns
        sns.boxplot(data=df_filtered, x="nickname", y=metric_name,
                    color="lightblue", width=0.6, showfliers=False)

        # Superimpose scatterplot with jitter
        sns.stripplot(data=df_filtered, x="nickname", y=metric_name,
                      color="red", size=6, alpha=0.7, jitter=True)
    else:
        # Use matplotlib only
        nicknames = df_filtered["nickname"].unique()
        positions = range(len(nicknames))

        # Create boxplot data
        box_data = [df_filtered[df_filtered["nickname"] == nickname][metric_name].values
                   for nickname in nicknames]

        # Create boxplot
        bp = plt.boxplot(box_data, positions=positions, patch_artist=True,
                        showfliers=False, widths=0.6)

        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        # Add scatter points with jitter
        for i, nickname in enumerate(nicknames):
            y_data = df_filtered[df_filtered["nickname"] == nickname][metric_name].values
            x_data = np.random.normal(i, 0.04, size=len(y_data))  # Add jitter
            plt.scatter(x_data, y_data, color="red", alpha=0.7, s=36)

        plt.xticks(positions, nicknames)

    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Genotype Nickname")

    if metric_name == "contact_proportion":
        plt.ylabel("Contact Proportion During Major Event")
        if title is None:
            title = "Contact Proportion During Major Events by Genotype"
    elif metric_name == "major_event_length":
        plt.ylabel("Major Event Length (seconds)")
        if title is None:
            title = "Major Event Duration by Genotype"
    elif metric_name == "ball_displacement_to_major_event":
        plt.ylabel("Ball Displacement to Major Event (pixels)")
        if title is None:
            title = "Ball Displacement from Start to Major Event by Genotype"
    elif metric_name == "ball_displacement_during_major_event":
        plt.ylabel("Ball Displacement During Major Event (pixels)")
        if title is None:
            title = "Ball Displacement During Major Event by Genotype"
    else:
        plt.ylabel(metric_name.replace("_", " ").title())
        if title is None:
            title = f"{metric_name.replace('_', ' ').title()} by Genotype"

    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add sample size annotations
    for i, nickname in enumerate(df_filtered["nickname"].unique()):
        subset = df_filtered[df_filtered["nickname"] == nickname]
        n_samples = len(subset)
        plt.text(i, plt.ylim()[0], f'n={n_samples}',
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def create_summary_statistics(df, output_path=None):
    """
    Create summary statistics table for the metrics.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_path (str, optional): Path to save the summary table
    """
    metrics = ["contact_proportion", "major_event_length", "ball_displacement_to_major_event", "ball_displacement_during_major_event"]

    summary_stats = []

    for metric in metrics:
        df_metric = df.dropna(subset=[metric])

        if df_metric.empty:
            continue

        grouped = df_metric.groupby("nickname")[metric]

        stats = grouped.agg([
            'count', 'mean', 'std', 'median',
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            'min', 'max'
        ]).round(4)

        stats['metric'] = metric
        stats = stats.reset_index()
        summary_stats.append(stats)

    if summary_stats:
        combined_stats = pd.concat(summary_stats, ignore_index=True)

        if output_path:
            combined_stats.to_csv(output_path, index=False)
            print(f"Summary statistics saved to {output_path}")
        else:
            print("\nSummary Statistics:")
            print(combined_stats.to_string(index=False))

        return combined_stats
    else:
        print("No valid data for summary statistics")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory metrics for flies from YAML file(s) and create boxplots."
    )
    parser.add_argument(
        "--yaml",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to YAML file(s) listing experiment directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/trajectory_analysis",
        help="Base output directory for plots and data."
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Save the computed metrics data to CSV."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=3,
        help="Minimum number of samples per genotype to include in analysis."
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load fly directories from YAML files
    fly_dirs = load_flies_from_yamls(args.yaml)

    if not fly_dirs:
        print("No fly directories found in YAML files.")
        return

    # Compute metrics for all flies
    print(f"\nComputing metrics for {len(fly_dirs)} flies...")
    all_metrics = []

    for i, fly_dir in enumerate(fly_dirs):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(fly_dirs)} flies...")

        metrics_data = compute_fly_metrics(fly_dir)
        if metrics_data:
            all_metrics.append(metrics_data)

    if not all_metrics:
        print("No valid metrics computed.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    print(f"\nSuccessfully computed metrics for {len(df)} flies")

    # Filter by minimum sample size
    nickname_counts = df["nickname"].value_counts()
    valid_nicknames = nickname_counts[nickname_counts >= args.min_samples].index
    df_filtered = df[df["nickname"].isin(valid_nicknames)]

    print(f"Genotypes with at least {args.min_samples} samples: {len(valid_nicknames)}")
    print(f"Flies included in analysis: {len(df_filtered)}")

    if df_filtered.empty:
        print("No genotypes meet the minimum sample size requirement.")
        return

    # Save data if requested
    if args.save_data:
        data_path = os.path.join(args.output_dir, "trajectory_metrics_data.csv")
        df_filtered.to_csv(data_path, index=False)
        print(f"Data saved to {data_path}")

    # Create plots
    metrics_to_plot = ["contact_proportion", "major_event_length", "ball_displacement_to_major_event", "ball_displacement_during_major_event"]

    for metric in metrics_to_plot:
        plot_path = os.path.join(args.output_dir, f"{metric}_boxplot.png")
        create_boxplot_with_scatter(df_filtered, metric, plot_path)

    # Create summary statistics
    summary_path = os.path.join(args.output_dir, "summary_statistics.csv")
    create_summary_statistics(df_filtered, summary_path)

    # Print final summary
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Genotypes analyzed: {', '.join(sorted(df_filtered['nickname'].unique()))}")


if __name__ == "__main__":
    main()
