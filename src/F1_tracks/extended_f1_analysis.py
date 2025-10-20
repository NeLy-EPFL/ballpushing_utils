#!/usr/bin/env python3
"""
Example of how to extend the unified F1 analysis framework for boxplot analysis.
This demonstrates how to add new analysis types to the framework.
"""

from unified_f1_analysis import F1AnalysisFramework
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats


class ExtendedF1Analysis(F1AnalysisFramework):
    """Extended analysis framework with additional analysis types."""

    def analyze_boxplots(self, mode, metric_type="interaction_rate"):
        """Analyze continuous metrics using boxplots with scatter overlay."""
        # Load and prepare data
        df = self.load_dataset(mode)
        if df is None:
            return

        detected_cols = self.detect_columns(df, mode)
        if not all(key in detected_cols for key in ["primary", "secondary", "ball_condition"]):
            print("Could not detect all required columns")
            return

        # Filter data
        df_clean = self.filter_for_test_ball(df, detected_cols["ball_condition"])

        # Get metric column
        continuous_metrics = self.config["analysis_parameters"]["continuous_metrics"]

        if metric_type in continuous_metrics:
            if isinstance(continuous_metrics[metric_type], str):
                metric_col = continuous_metrics[metric_type]
            else:
                metric_col = continuous_metrics[metric_type][0]  # Take first if list
        else:
            metric_col = metric_type  # Use as-is if not in config

        if metric_col not in df_clean.columns:
            print(f"Metric column '{metric_col}' not found in dataset")
            return

        # Prepare analysis data
        analysis_cols = [
            detected_cols["primary"],
            detected_cols["secondary"],
            detected_cols["ball_condition"],
            metric_col,
        ]
        df_analysis = df_clean[analysis_cols].dropna(
            subset=[detected_cols["primary"], detected_cols["secondary"], detected_cols["ball_condition"]]
        )

        # Remove infinite values
        mask = pd.isna(df_analysis[metric_col]) | np.isinf(df_analysis[metric_col])
        df_analysis = df_analysis[~mask]

        print(f"Analysis data shape: {df_analysis.shape}")

        if len(df_analysis) < 2:
            print("Insufficient data for analysis")
            return

        # Create plots based on mode
        if mode == "control":
            self._create_control_boxplots(df_analysis, detected_cols, metric_col, mode)
        else:  # TNT modes
            self._create_tnt_boxplots(df_analysis, detected_cols, metric_col, mode)

    def _create_control_boxplots(self, data, cols, metric_col, mode):
        """Create boxplots for control mode."""
        plot_config = self.config["plot_styling"][mode]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot by primary grouping
        self._create_boxplot_with_scatter(
            data,
            cols["primary"],
            metric_col,
            f"{metric_col.replace('_', ' ').title()} by {cols['primary'].title()}",
            ax1,
            mode,
            "pretraining",
        )

        # Plot by secondary grouping
        self._create_boxplot_with_scatter(
            data,
            cols["secondary"],
            metric_col,
            f"{metric_col.replace('_', ' ').title()} by {cols['secondary'].replace('_', ' ').title()}",
            ax2,
            mode,
            "f1_condition",
        )

        plt.tight_layout()
        self._save_plot(fig, f"{metric_col}_boxplots", mode)

    def _create_tnt_boxplots(self, data, cols, metric_col, mode):
        """Create boxplots for TNT modes."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Combined plot
        self._create_combined_boxplot_with_scatter(
            data,
            cols["primary"],
            cols["secondary"],
            metric_col,
            f"{metric_col.replace('_', ' ').title()} by {cols['primary'].title()} + {cols['secondary'].title()}",
            ax,
            mode,
        )

        plt.tight_layout()
        self._save_plot(fig, f"{metric_col}_boxplots", mode)

    def _create_boxplot_with_scatter(self, data, x_col, y_col, title, ax, mode, var_type):
        """Create boxplot with scatter overlay."""
        # Get ordering and colors
        plot_config = self.config["plot_styling"][mode]
        unique_values = sorted(data[x_col].unique())
        color_mapping = self.create_color_mapping(mode, unique_values, var_type)

        # Create boxplot
        box_plot = sns.boxplot(
            data=data,
            x=x_col,
            y=y_col,
            ax=ax,
            order=unique_values,
            palette=[color_mapping.get(v, "gray") for v in unique_values],
        )

        # Style boxplot
        for patch in box_plot.patches:
            patch.set_facecolor("none")
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)

        for line in ax.lines:
            line.set_color("black")
            line.set_linewidth(1.5)

        # Add scatter
        for i, condition in enumerate(unique_values):
            condition_data = data[data[x_col] == condition]
            color = color_mapping.get(condition, "black")

            y_values = condition_data[y_col].values
            x_positions = np.random.normal(i, 0.1, size=len(y_values))

            ax.scatter(x_positions, y_values, c=color, s=30, alpha=0.7, edgecolors="black", linewidth=0.5)

        # Formatting
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

        # Add sample sizes
        for i, condition in enumerate(unique_values):
            n = len(data[data[x_col] == condition])
            ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", va="top", fontsize=10, fontweight="bold")

        # Statistical test
        groups = [data[data[x_col] == condition][y_col].values for condition in unique_values]
        if len(groups) == 2:
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            test_name = "Mann-Whitney U"
        elif len(groups) > 2:
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        else:
            return ax

        ax.text(
            0.02,
            0.98,
            f"{test_name}: p = {p_value:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        return ax

    def _create_combined_boxplot_with_scatter(self, data, primary_col, secondary_col, y_col, title, ax, mode):
        """Create combined boxplot for TNT modes."""
        # Create combined grouping
        data_copy = data.copy()
        data_copy["combined_group"] = data_copy[secondary_col].astype(str) + " + " + data_copy[primary_col].astype(str)

        unique_groups = sorted(data_copy["combined_group"].unique())

        # Get colors based on genotype brain regions
        genotype_vals = [group.split(" + ")[0] for group in unique_groups]
        unique_genotypes = list(set(genotype_vals))
        genotype_color_map = self.create_color_mapping(mode, unique_genotypes, "brain_region")

        # Create boxplot
        colors = []
        for group in unique_groups:
            genotype_val = group.split(" + ")[0]
            colors.append(genotype_color_map[genotype_val])

        box_plot = sns.boxplot(data=data_copy, x="combined_group", y=y_col, ax=ax, order=unique_groups, palette=colors)

        # Style based on pretraining
        for i, (patch, group) in enumerate(zip(box_plot.patches, unique_groups)):
            genotype_val, pretraining_val = group.split(" + ")
            brain_region_color = genotype_color_map[genotype_val]

            if pretraining_val == "n":
                patch.set_facecolor("white")
                patch.set_edgecolor(brain_region_color)
                patch.set_linewidth(2)
            else:
                patch.set_facecolor(brain_region_color)
                patch.set_edgecolor(brain_region_color)
                patch.set_alpha(0.7)

        # Style whiskers and medians
        for i, group in enumerate(unique_groups):
            genotype_val, pretraining_val = group.split(" + ")
            brain_region_color = genotype_color_map[genotype_val]

            # Style lines
            start_idx = 6 * i
            whiskers = box_plot.lines[start_idx : start_idx + 2]
            caps = box_plot.lines[start_idx + 2 : start_idx + 4]
            median = box_plot.lines[start_idx + 4 : start_idx + 5]

            for whisker in whiskers:
                whisker.set_color(brain_region_color)
                whisker.set_linewidth(1.5)
            for cap in caps:
                cap.set_color(brain_region_color)
                cap.set_linewidth(1.5)
            for med in median:
                med.set_color(brain_region_color)
                med.set_linewidth(2)

        # Add scatter
        sns.stripplot(
            data=data_copy,
            x="combined_group",
            y=y_col,
            ax=ax,
            size=6,
            alpha=0.7,
            jitter=True,
            dodge=False,
            color="black",
            order=unique_groups,
        )

        # Formatting
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Genotype + Pretraining", fontsize=12)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
        ax.tick_params(axis="x", rotation=45)

        # Add sample sizes
        for i, group in enumerate(unique_groups):
            n = len(data_copy[data_copy["combined_group"] == group])
            ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", va="top", fontsize=10, fontweight="bold")

        # Statistical test
        groups = [data_copy[data_copy["combined_group"] == group][y_col].values for group in unique_groups]
        if len(groups) == 2:
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            test_name = "Mann-Whitney U"
        elif len(groups) > 2:
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        else:
            return ax

        ax.text(
            0.02,
            0.98,
            f"{test_name}: p = {p_value:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        return ax


if __name__ == "__main__":
    """Example usage of the extended framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Extended F1 tracks analysis framework")
    parser.add_argument("--mode", required=True, choices=["control", "tnt_mb247", "tnt_lc10_2"], help="Analysis mode")
    parser.add_argument(
        "--analysis", required=True, choices=["binary_metrics", "boxplots"], help="Type of analysis to perform"
    )
    parser.add_argument("--metric", default="interaction_rate", help="Metric for boxplot analysis")
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    # Initialize extended framework
    framework = ExtendedF1Analysis(args.config)

    # Run analysis
    if args.analysis == "binary_metrics":
        framework.analyze_binary_metrics(args.mode)
    elif args.analysis == "boxplots":
        framework.analyze_boxplots(args.mode, args.metric)

    print(f"Analysis completed for mode: {args.mode}, analysis: {args.analysis}")
