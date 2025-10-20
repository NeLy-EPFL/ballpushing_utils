#!/usr/bin/env python3
"""
Unified F1 tracks analysis framework.
This script consolidates the functionality of multiple redundant analysis scripts
using a configuration-driven approach.

Usage:
    python unified_f1_analysis.py --mode control --analysis binary_metrics
    python unified_f1_analysis.py --mode tnt_mb247 --analysis boxplots --metric interaction_rate
    python unified_f1_analysis.py --mode tnt_lc10_2 --analysis coordinates --plot_type percentage
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import argparse
import sys
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from matplotlib.patches import Rectangle


class F1AnalysisFramework:
    """Unified framework for F1 tracks analysis."""

    def __init__(self, config_path=None):
        """Initialize the framework with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "analysis_config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize brain region mappings if available
        self.setup_brain_region_mappings()

    def setup_brain_region_mappings(self):
        """Setup brain region mappings from Config if available."""
        try:
            brain_config = self.config["brain_region_config"]
            if brain_config["use_config_file"]:
                config_path = brain_config["config_path"]
                sys.path.append(str(Path(config_path).parent))
                from PCA import Config

                self.nickname_to_brainregion = dict(
                    zip(Config.SplitRegistry["Nickname"], Config.SplitRegistry["Simplified region"])
                )
                self.brain_region_color_dict = Config.color_dict
                self.has_brain_regions = True
                print(f"✅ Loaded brain region mappings: {len(self.nickname_to_brainregion)} genotypes")
        except Exception as e:
            print(f"⚠️  Could not load brain region mappings: {e}")
            self.nickname_to_brainregion = {}
            self.brain_region_color_dict = self.config["brain_region_config"]["fallback_colors"]
            self.has_brain_regions = False

    def load_dataset(self, mode):
        """Load dataset for specified mode."""
        mode_config = self.config["analysis_modes"][mode]
        dataset_path = mode_config["dataset_path"]

        try:
            df = pd.read_feather(dataset_path)
            print(f"Dataset loaded successfully! Shape: {df.shape}")

            # Filter out excluded dates
            excluded_dates = mode_config.get("excluded_dates", [])
            if excluded_dates and "Date" in df.columns:
                initial_shape = df.shape
                df = df[~df["Date"].isin(excluded_dates)]
                print(f"Removed excluded dates {excluded_dates}. Shape: {initial_shape} -> {df.shape}")

            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def detect_columns(self, df, mode):
        """Auto-detect column names based on patterns."""
        mode_config = self.config["analysis_modes"][mode]
        grouping_vars = mode_config["grouping_variables"]
        detection_config = self.config["column_detection"]

        detected_cols = {}

        # Detect primary grouping variable (usually pretraining)
        primary_key = grouping_vars["primary"]
        if primary_key in detection_config:
            patterns = detection_config[primary_key]["patterns"]
            for col in df.columns:
                if any(pattern.lower() in col.lower() for pattern in patterns):
                    detected_cols["primary"] = col
                    break

        # Detect secondary grouping variable (F1_condition or Genotype)
        secondary_key = grouping_vars["secondary"]
        if secondary_key in detection_config:
            patterns = detection_config[secondary_key]["patterns"]
            for col in df.columns:
                if any(pattern.lower() in col.lower() for pattern in patterns):
                    detected_cols["secondary"] = col
                    break

        # Detect ball condition column
        ball_config = detection_config["ball_condition"]
        for priority_col in ball_config["priority"]:
            if priority_col in df.columns:
                detected_cols["ball_condition"] = priority_col
                break

        # If not found, use pattern matching
        if "ball_condition" not in detected_cols:
            for col in df.columns:
                if any(pattern.lower() in col.lower() for pattern in ball_config["patterns"]):
                    detected_cols["ball_condition"] = col
                    break

        print(f"Detected columns: {detected_cols}")
        return detected_cols

    def filter_for_test_ball(self, df, ball_condition_col):
        """Filter dataset for test ball condition."""
        ball_config = self.config["analysis_parameters"]["ball_condition"]

        if not ball_config["filter_for_test_ball"]:
            return df

        test_ball_values = ball_config["test_ball_values"]
        test_ball_data = pd.DataFrame()

        for test_val in test_ball_values:
            test_subset = df[df[ball_condition_col] == test_val]
            if not test_subset.empty:
                test_ball_data = test_subset
                print(f"Found test ball data using value: '{test_val}'")
                break

        if test_ball_data.empty:
            print("No specific 'test' ball data found. Using all data.")
            return df

        return test_ball_data

    def get_brain_region_for_genotype(self, genotype, mode):
        """Get brain region for a genotype."""
        mode_config = self.config["analysis_modes"][mode]

        # Check manual mapping first
        if "brain_region_mapping" in mode_config:
            manual_mapping = mode_config["brain_region_mapping"]
            if genotype in manual_mapping:
                return manual_mapping[genotype]

        # Check Config mappings
        if self.has_brain_regions and genotype in self.nickname_to_brainregion:
            return self.nickname_to_brainregion[genotype]

        # Try variations
        if self.has_brain_regions:
            variations = [
                genotype.replace("TNTx", ""),
                genotype.replace("x", ""),
                genotype.replace("-", ""),
                genotype.split("x")[-1] if "x" in genotype else genotype,
            ]

            for variation in variations:
                if variation in self.nickname_to_brainregion:
                    return self.nickname_to_brainregion[variation]

        return "Unknown"

    def create_color_mapping(self, mode, values, variable_type):
        """Create color mapping for values based on mode configuration."""
        plot_config = self.config["plot_styling"][mode]

        if variable_type == "pretraining" and "pretraining_colors" in plot_config:
            return plot_config["pretraining_colors"]
        elif variable_type == "f1_condition" and "f1_condition_colors" in plot_config:
            return plot_config["f1_condition_colors"]
        elif variable_type == "brain_region" and "brain_region_colors" in plot_config:
            # For genotype-based analysis, map genotypes to brain regions then to colors
            color_mapping = {}
            for genotype in values:
                brain_region = self.get_brain_region_for_genotype(genotype, mode)
                color_mapping[genotype] = plot_config["brain_region_colors"].get(brain_region, "#808080")
            return color_mapping
        else:
            # Fallback to seaborn palette
            colors = sns.color_palette("Set2", n_colors=len(values))
            return dict(zip(values, colors))

    def calculate_proportions_and_stats(self, data, group_col, binary_col):
        """Calculate proportions and statistical tests for binary data."""
        # Calculate proportions
        prop_data = data.groupby(group_col)[binary_col].agg(["count", "sum", "mean"]).reset_index()
        prop_data["proportion"] = prop_data["mean"]
        prop_data["n_positive"] = prop_data["sum"]
        prop_data["n_total"] = prop_data["count"]

        # Perform statistical test
        groups = data[group_col].unique()
        if len(groups) >= 2:
            contingency = pd.crosstab(data[group_col], data[binary_col])

            if len(groups) == 2 and contingency.shape == (2, 2):
                try:
                    odds_ratio, p_value = fisher_exact(contingency)
                    test_name = "Fisher's exact"
                except:
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    test_name = "Chi-square"
            else:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                test_name = "Chi-square"
        else:
            p_value = np.nan
            test_name = "No test"

        return prop_data, p_value, test_name

    def analyze_binary_metrics(self, mode):
        """Analyze binary metrics for the specified mode."""
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

        # Get binary metrics
        binary_metrics = self.config["analysis_parameters"]["binary_metrics"]
        available_metrics = [m for m in binary_metrics if m in df_clean.columns]

        if not available_metrics:
            print("No binary metrics found in dataset!")
            return

        # Prepare analysis columns
        analysis_cols = [
            detected_cols["primary"],
            detected_cols["secondary"],
            detected_cols["ball_condition"],
        ] + available_metrics
        df_analysis = df_clean[analysis_cols].dropna()

        print(f"Analysis data shape: {df_analysis.shape}")

        # Create plots based on mode
        if mode == "control":
            self._create_control_binary_plots(df_analysis, detected_cols, available_metrics, mode)
        else:  # TNT modes
            self._create_tnt_binary_plots(df_analysis, detected_cols, available_metrics, mode)

    def _create_control_binary_plots(self, data, cols, metrics, mode):
        """Create binary metric plots for control mode."""
        plot_config = self.config["plot_styling"][mode]
        n_metrics = len(metrics)

        fig = plt.figure(figsize=plot_config["figure_size"])
        gs = fig.add_gridspec(3, max(n_metrics, 2), height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

        # Row 1: By primary grouping (pretraining)
        ax_primary = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
        self._create_binary_barplot(
            data,
            cols["primary"],
            metrics,
            f"Binary Metrics by {cols['primary'].title()}",
            ax_primary,
            mode,
            "pretraining",
        )

        # Row 2: By secondary grouping (F1_condition)
        ax_secondary = [fig.add_subplot(gs[1, i]) for i in range(n_metrics)]
        self._create_binary_barplot(
            data,
            cols["secondary"],
            metrics,
            f"Binary Metrics by {cols['secondary'].replace('_', ' ').title()}",
            ax_secondary,
            mode,
            "f1_condition",
        )

        # Row 3: Heatmaps
        ax_heatmap1 = fig.add_subplot(gs[2, 0])
        self._create_heatmap(data, cols["primary"], metrics, f"Binary Metrics - {cols['primary'].title()}", ax_heatmap1)

        ax_heatmap2 = fig.add_subplot(gs[2, 1])
        self._create_heatmap(
            data,
            cols["secondary"],
            metrics,
            f"Binary Metrics - {cols['secondary'].replace('_', ' ').title()}",
            ax_heatmap2,
        )

        fig.suptitle(
            f"{mode.replace('_', ' ').title()}: Binary Metrics Analysis (Test Ball Only)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        self._save_plot(fig, "binary_metrics_analysis", mode)

    def _create_tnt_binary_plots(self, data, cols, metrics, mode):
        """Create binary metric plots for TNT modes."""
        plot_config = self.config["plot_styling"][mode]
        n_metrics = len(metrics)

        fig = plt.figure(figsize=plot_config["figure_size"])
        gs = fig.add_gridspec(2, max(n_metrics, 1), height_ratios=[1, 0.6], hspace=0.3, wspace=0.3)

        # Row 1: Combined pretraining + genotype
        ax_combined = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
        self._create_combined_binary_barplot(
            data,
            cols["primary"],
            cols["secondary"],
            metrics,
            f"Binary Metrics by {cols['primary'].title()} + {cols['secondary'].title()}",
            ax_combined,
            mode,
        )

        # Row 2: Heatmap
        ax_heatmap = fig.add_subplot(gs[1, :])
        self._create_combined_heatmap(
            data,
            cols["primary"],
            cols["secondary"],
            metrics,
            f"Binary Metrics - {cols['primary'].title()} + {cols['secondary'].title()}",
            ax_heatmap,
        )

        fig.suptitle(
            f"{mode.replace('_', ' ').title()}: Binary Metrics Analysis (Test Ball Only)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        self._save_plot(fig, "binary_metrics_analysis", mode)

    def _create_binary_barplot(self, data, group_col, metrics, title_prefix, ax_row, mode, var_type):
        """Create bar plots for binary metrics."""
        unique_values = data[group_col].unique()
        color_mapping = self.create_color_mapping(mode, unique_values, var_type)

        for i, metric in enumerate(metrics):
            ax = ax_row[i]

            prop_data, p_value, test_name = self.calculate_proportions_and_stats(data, group_col, metric)

            # Create bars
            bars = ax.bar(prop_data[group_col], prop_data["proportion"], alpha=0.7, edgecolor="black", linewidth=1)

            # Color bars
            for bar, group_val in zip(bars, prop_data[group_col]):
                color = color_mapping.get(group_val, "gray")
                bar.set_facecolor(color)

            # Add labels
            for j, (idx, row) in enumerate(prop_data.iterrows()):
                height = row["proportion"]
                ax.text(
                    j,
                    height + 0.01,
                    f"{row['n_positive']}/{row['n_total']}\\n({height:.2%})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            # Formatting
            ax.set_title(f"{title_prefix}\\n{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel("Proportion", fontsize=10)
            ax.set_ylim(0, 1.1)

            # Add statistical annotation
            if not np.isnan(p_value):
                ax.text(
                    0.02,
                    0.98,
                    f"{test_name}\\np = {p_value:.4f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )

    def _create_combined_binary_barplot(self, data, primary_col, secondary_col, metrics, title_prefix, ax_row, mode):
        """Create combined binary bar plots for TNT modes."""
        for i, metric in enumerate(metrics):
            ax = ax_row[i]

            # Create combined grouping
            data_copy = data.copy()
            data_copy["combined_group"] = (
                data_copy[primary_col].astype(str) + " + " + data_copy[secondary_col].astype(str)
            )

            prop_data, p_value, test_name = self.calculate_proportions_and_stats(data_copy, "combined_group", metric)

            # Create bars
            bars = ax.bar(range(len(prop_data)), prop_data["proportion"], alpha=1.0, linewidth=2)

            # Style bars based on brain regions and pretraining
            genotype_vals = [group.split(" + ")[1] for group in prop_data["combined_group"]]
            pretraining_vals = [group.split(" + ")[0] for group in prop_data["combined_group"]]

            unique_genotypes = list(set(genotype_vals))
            genotype_color_map = self.create_color_mapping(mode, unique_genotypes, "brain_region")

            for bar, group in zip(bars, prop_data["combined_group"]):
                pretraining_val, genotype_val = group.split(" + ")
                brain_region_color = genotype_color_map[genotype_val]

                if pretraining_val == "n":
                    bar.set_facecolor("white")
                    bar.set_edgecolor(brain_region_color)
                else:
                    bar.set_facecolor(brain_region_color)
                    bar.set_edgecolor(brain_region_color)

            # Add labels and formatting
            for j, (idx, row) in enumerate(prop_data.iterrows()):
                height = row["proportion"]
                ax.text(
                    j,
                    height + 0.01,
                    f"{row['n_positive']}/{row['n_total']}\\n({height:.2%})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.set_title(f"{title_prefix}\\n{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Pretraining + Genotype", fontsize=10)
            ax.set_ylabel("Proportion", fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.set_xticks(range(len(prop_data)))
            ax.set_xticklabels(prop_data["combined_group"], rotation=45, ha="right")

            # Add statistical annotation
            if not np.isnan(p_value):
                ax.text(
                    0.02,
                    0.98,
                    f"{test_name}\\np = {p_value:.4f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )

    def _create_heatmap(self, data, group_col, metrics, title, ax):
        """Create heatmap for binary metrics."""
        heatmap_data = []
        for metric in metrics:
            prop_data, _, _ = self.calculate_proportions_and_stats(data, group_col, metric)
            proportions = dict(zip(prop_data[group_col], prop_data["proportion"]))
            heatmap_data.append(proportions)

        heatmap_df = pd.DataFrame(heatmap_data, index=[m.replace("_", " ").title() for m in metrics])

        sns.heatmap(heatmap_df, annot=True, fmt=".2%", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Proportion"})

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Binary Metrics", fontsize=12)

    def _create_combined_heatmap(self, data, primary_col, secondary_col, metrics, title, ax):
        """Create combined heatmap for TNT modes."""
        data_copy = data.copy()
        data_copy["combined_group"] = data_copy[primary_col].astype(str) + " + " + data_copy[secondary_col].astype(str)

        heatmap_data = []
        for metric in metrics:
            prop_data, _, _ = self.calculate_proportions_and_stats(data_copy, "combined_group", metric)
            proportions = dict(zip(prop_data["combined_group"], prop_data["proportion"]))
            heatmap_data.append(proportions)

        heatmap_df = pd.DataFrame(heatmap_data, index=[m.replace("_", " ").title() for m in metrics])

        sns.heatmap(heatmap_df, annot=True, fmt=".2%", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Proportion"})

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Pretraining + Genotype", fontsize=12)
        ax.set_ylabel("Binary Metrics", fontsize=12)
        ax.tick_params(axis="x", rotation=45)

    def _save_plot(self, fig, plot_name, mode):
        """Save plot to file."""
        output_config = self.config["output"]
        if output_config["save_plots"]:
            # Save in mode-specific subdirectory if it's a TNT mode
            if mode.startswith("tnt_"):
                output_dir = Path(__file__).parent / mode.replace("tnt_", "").upper()
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{plot_name}.{output_config['plot_format']}"
            else:
                output_path = Path(__file__).parent / f"{plot_name}.{output_config['plot_format']}"

            plt.savefig(output_path, dpi=output_config["dpi"], bbox_inches=output_config["bbox_inches"])
            print(f"Plot saved to: {output_path}")

        if output_config["show_plots"]:
            plt.show()
        else:
            plt.close()

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

    def analyze_coordinates(self, mode):
        """Analyze coordinate data and create trajectory plots."""
        mode_config = self.config['analysis_modes'][mode]
        coordinates_config = mode_config.get('coordinates', {})
        
        if not coordinates_config:
            print(f"No coordinates configuration found for mode: {mode}")
            return
            
        # Get dataset paths from config
        coordinates_path = coordinates_config['coordinates_path']
        summary_path = coordinates_config.get('summary_path')
        alt_coord_paths = coordinates_config.get('alternative_coordinates_paths', [])
        alt_summary_paths = coordinates_config.get('alternative_summary_paths', [])
        plot_params = coordinates_config.get('coordinates_plot_params', {})
        
        # Load coordinates dataset
        df_coords = None
        used_coords_path = None
        
        for path in [coordinates_path] + alt_coord_paths:
            try:
                if Path(path).exists():
                    df_coords = pd.read_feather(path)
                    used_coords_path = path
                    print(f"✅ Coordinates dataset loaded: {path}")
                    print(f"   Shape: {df_coords.shape}")
                    break
            except Exception as e:
                print(f"⚠️  Could not load coordinates from {path}: {e}")
                continue
                
        if df_coords is None:
            print("❌ Could not load coordinates dataset from any path")
            return
            
        # Load summary dataset for filtering (optional)
        df_summary = None
        if summary_path:
            for path in [summary_path] + alt_summary_paths:
                try:
                    if Path(path).exists():
                        df_summary = pd.read_feather(path)
                        print(f"✅ Summary dataset loaded: {path}")
                        print(f"   Shape: {df_summary.shape}")
                        break
                except Exception as e:
                    print(f"⚠️  Could not load summary from {path}: {e}")
                    continue
                    
        # Filter coordinates data based on summary dataset
        if df_summary is not None and 'fly' in df_coords.columns and 'fly' in df_summary.columns:
            summary_flies = set(df_summary['fly'].unique())
            coords_flies = set(df_coords['fly'].unique())
            common_flies = summary_flies.intersection(coords_flies)
            
            print(f"📊 Dataset filtering:")
            print(f"   Flies in coordinates: {len(coords_flies)}")
            print(f"   Flies in summary: {len(summary_flies)}")
            print(f"   Common flies: {len(common_flies)}")
            
            # Filter to common flies
            df_coords = df_coords[df_coords['fly'].isin(common_flies)]
            print(f"   Filtered coordinates shape: {df_coords.shape}")
            
        # Detect grouping columns
        detected_cols = self.detect_columns(df_coords, mode)
        if 'primary' not in detected_cols or 'secondary' not in detected_cols:
            print("❌ Could not detect required grouping columns for coordinates")
            return
            
        primary_col = detected_cols['primary']
        secondary_col = detected_cols['secondary']
        
        print(f"📋 Using grouping columns: {primary_col}, {secondary_col}")
        
        # Check for required coordinate columns
        coord_cols = ['training_ball_euclidean_distance', 'test_ball_euclidean_distance']
        missing_cols = [col for col in coord_cols if col not in df_coords.columns]
        if missing_cols:
            # Try alternative column names
            alt_coord_cols = ['training_ball', 'test_ball']
            missing_alt = [col for col in alt_coord_cols if col not in df_coords.columns]
            if not missing_alt:
                coord_cols = alt_coord_cols
                print(f"📋 Using alternative coordinate columns: {coord_cols}")
            else:
                print(f"❌ Missing coordinate columns: {missing_cols}")
                print(f"   Available columns: {list(df_coords.columns)}")
                return
        
        # Get plotting parameters
        normalize_methods = plot_params.get('normalize_methods', ['percentage', 'trimmed'])
        
        # Create plots for each normalization method
        for method in normalize_methods:
            print(f"\n📈 Creating {method} plots...")
            
            if method == 'percentage':
                self._create_percentage_coordinate_plots(df_coords, primary_col, secondary_col, coord_cols, mode)
            elif method == 'trimmed':
                self._create_trimmed_coordinate_plots(df_coords, primary_col, secondary_col, coord_cols, mode)
                
        print(f"✅ Coordinate analysis completed for mode: {mode}")

    def _normalize_time_to_percentage(self, df):
        """Normalize adjusted_time to 0-100% for each fly."""
        if 'adjusted_time' not in df.columns or 'fly' not in df.columns:
            print("⚠️  Cannot normalize - missing 'adjusted_time' or 'fly' columns")
            return df
            
        print(f"🔄 Normalizing time to percentage...")
        standard_time_grid = np.arange(0, 100.1, 0.1)
        normalized_data = []
        
        for fly_id in df['fly'].unique():
            fly_data = df[df['fly'] == fly_id].copy()
            
            # Skip flies with no valid adjusted_time data
            if fly_data['adjusted_time'].isna().all():
                continue
                
            valid_mask = ~fly_data['adjusted_time'].isna()
            if not valid_mask.any():
                continue
                
            valid_times = fly_data.loc[valid_mask, 'adjusted_time']
            min_time = valid_times.min()
            max_time = valid_times.max()
            
            if max_time <= min_time or valid_mask.sum() < 2:
                continue
                
            # Normalize to 0-100%
            normalized_time = ((fly_data['adjusted_time'] - min_time) / (max_time - min_time)) * 100
            
            # Create DataFrame with standard time grid
            fly_normalized = pd.DataFrame({'adjusted_time': standard_time_grid})
            
            # Add metadata columns
            for col in ['fly', 'Pretraining', 'Genotype', 'F1_condition']:
                if col in fly_data.columns:
                    fly_normalized[col] = fly_data[col].iloc[0]
                    
            # Interpolate ball distances
            for ball_col in ['training_ball_euclidean_distance', 'test_ball_euclidean_distance', 'training_ball', 'test_ball']:
                if ball_col in fly_data.columns:
                    if valid_mask.sum() > 1:
                        fly_normalized[ball_col] = np.interp(
                            standard_time_grid, 
                            normalized_time[valid_mask], 
                            fly_data.loc[valid_mask, ball_col]
                        )
                    else:
                        fly_normalized[ball_col] = np.nan
                        
            normalized_data.append(fly_normalized)
            
        if not normalized_data:
            print("⚠️  No flies could be normalized")
            return df
            
        result_df = pd.concat(normalized_data, ignore_index=True)
        print(f"✅ Normalized {len(df['fly'].unique())} flies to 0-100% time grid")
        return result_df

    def _find_maximum_shared_time(self, df):
        """Find maximum time that all flies share."""
        if 'fly' not in df.columns or 'adjusted_time' not in df.columns:
            return df['adjusted_time'].max() if 'adjusted_time' in df.columns else 30
            
        max_times_per_fly = df.groupby('fly')['adjusted_time'].max()
        max_shared_time = max_times_per_fly.min()
        
        print(f"📊 Maximum shared time: {max_shared_time:.2f}s")
        print(f"   Flies with longer data: {(max_times_per_fly > max_shared_time).sum()}")
        
        return max_shared_time

    def _create_percentage_coordinate_plots(self, df, primary_col, secondary_col, coord_cols, mode):
        """Create percentage-normalized coordinate plots."""
        # Normalize time to percentage
        df_norm = self._normalize_time_to_percentage(df)
        
        if df_norm.empty:
            print("⚠️  No data after percentage normalization")
            return
            
        # Create plots grouped by primary variable (pretraining)
        self._create_coordinate_line_plots(
            df_norm, 'adjusted_time', coord_cols, primary_col,
            f"F1 Coordinates by {primary_col.title()} (Percentage Time)",
            "f1_coordinates_percentage_time_pretraining.png", mode,
            xlabel="Adjusted Time (%)", time_type="percentage"
        )
        
        # Create plots grouped by secondary variable (F1_condition or Genotype)
        self._create_coordinate_line_plots(
            df_norm, 'adjusted_time', coord_cols, secondary_col,
            f"F1 Coordinates by {secondary_col.replace('_', ' ').title()} (Percentage Time)",
            "f1_coordinates_percentage_time_f1condition.png", mode,
            xlabel="Adjusted Time (%)", time_type="percentage"
        )

    def _create_trimmed_coordinate_plots(self, df, primary_col, secondary_col, coord_cols, mode):
        """Create trimmed-time coordinate plots."""
        # Find maximum shared time and trim data
        max_shared_time = self._find_maximum_shared_time(df)
        df_trimmed = df[df['adjusted_time'] <= max_shared_time].copy()
        
        print(f"📊 Trimmed data: {df_trimmed.shape[0]} points from {df_trimmed['fly'].nunique()} flies")
        
        # Create plots grouped by primary variable
        self._create_coordinate_line_plots(
            df_trimmed, 'adjusted_time', coord_cols, primary_col,
            f"F1 Coordinates by {primary_col.title()} (Trimmed Time)",
            "f1_coordinates_trimmed_time_pretraining.png", mode,
            xlabel="Adjusted Time (seconds)", time_type="trimmed"
        )
        
        # Create plots grouped by secondary variable
        self._create_coordinate_line_plots(
            df_trimmed, 'adjusted_time', coord_cols, secondary_col,
            f"F1 Coordinates by {secondary_col.replace('_', ' ').title()} (Trimmed Time)",
            "f1_coordinates_trimmed_time_f1condition.png", mode,
            xlabel="Adjusted Time (seconds)", time_type="trimmed"
        )

    def _create_coordinate_line_plots(self, data, x_col, y_cols, hue_col, title, filename, mode, xlabel="Time", time_type="percentage"):
        """Create line plots for coordinate data with confidence intervals."""
        if data.empty:
            print(f"⚠️  No data for {filename}")
            return
            
        # Create figure
        fig, axes = plt.subplots(len(y_cols), 1, figsize=(12, 6 * len(y_cols)))
        if len(y_cols) == 1:
            axes = [axes]
            
        # Get color mapping
        unique_values = data[hue_col].unique()
        if mode.startswith('tnt_'):
            # Use brain region colors for TNT modes
            color_mapping = self.create_color_mapping(mode, unique_values, 'brain_region')
        else:
            # Use F1_condition colors for control mode
            color_mapping = self.create_color_mapping(mode, unique_values, 'f1_condition')
            
        for i, y_col in enumerate(y_cols):
            ax = axes[i]
            
            if y_col not in data.columns:
                ax.text(0.5, 0.5, f"Column '{y_col}' not found", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
                
            # Plot each group
            for hue_val in unique_values:
                subset = data[data[hue_col] == hue_val]
                if subset.empty or subset[y_col].isna().all():
                    continue
                    
                # Skip training_ball for control groups (usually all NaN)
                if 'training' in y_col and subset[y_col].isna().all():
                    continue
                    
                # Bin data and calculate statistics
                subset_clean = subset.dropna(subset=[y_col])
                if subset_clean.empty:
                    continue
                    
                # Create time bins
                bin_size = 0.1 if time_type == "percentage" else 0.1
                subset_clean = subset_clean.copy()
                subset_clean['time_bin'] = np.floor(subset_clean[x_col] / bin_size) * bin_size
                
                # Get median per fly per time bin
                fly_medians = subset_clean.groupby(['fly', 'time_bin'])[y_col].median().reset_index()
                
                # Calculate statistics across flies for each time bin
                time_stats = fly_medians.groupby('time_bin')[y_col].agg(['mean', 'std', 'count', 'sem']).reset_index()
                
                # Calculate confidence intervals
                confidence_level = 0.95
                alpha = 1 - confidence_level
                time_stats['ci'] = time_stats.apply(
                    lambda row: stats.t.ppf(1 - alpha/2, row['count'] - 1) * row['sem'] if row['count'] > 1 else 0,
                    axis=1
                )
                
                # Plot mean trajectory
                color = color_mapping.get(hue_val, 'black')
                label = str(hue_val)
                
                ax.plot(time_stats['time_bin'], time_stats['mean'], 
                       color=color, linewidth=2, alpha=0.8, label=label)
                
                # Add confidence intervals
                ax.fill_between(time_stats['time_bin'], 
                               time_stats['mean'] - time_stats['ci'],
                               time_stats['mean'] + time_stats['ci'],
                               color=color, alpha=0.2)
                               
            # Formatting
            ax.set_title(f"{y_col.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Distance (pixels)", fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add vertical line at time 0
            ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, label='Exit time')
            
            # Add sample size
            n_flies = data['fly'].nunique() if 'fly' in data.columns else len(data)
            ax.text(0.02, 0.98, f"N flies = {n_flies}", transform=ax.transAxes,
                   fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                   
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        self._save_plot(fig, filename.replace('.png', ''), mode)

    def run_all_analyses(self, mode, metric=None):
        """Run all available analyses for the specified mode."""
        print(f"\n{'='*60}")
        print(f"Running all analyses for mode: {mode}")
        print(f"{'='*60}")

        # 1. Binary metrics analysis
        print(f"\n🔬 Running binary metrics analysis...")
        try:
            self.analyze_binary_metrics(mode)
            print("✅ Binary metrics analysis completed")
        except Exception as e:
            print(f"❌ Binary metrics analysis failed: {e}")

        # 2. Boxplot analyses for multiple metrics
        continuous_metrics = self.config["analysis_parameters"]["continuous_metrics"]

        # Get list of metrics to analyze
        metrics_to_analyze = []
        if metric:
            # If specific metric provided, use it
            metrics_to_analyze = [metric]
        else:
            # Otherwise, analyze key metrics
            metrics_to_analyze = ["interaction_rate"]

            # Add distance metrics if available
            if "distance_metrics" in continuous_metrics:
                dist_metrics = continuous_metrics["distance_metrics"]
                if isinstance(dist_metrics, list):
                    metrics_to_analyze.extend(dist_metrics[:2])  # Take first 2 to avoid too many plots

            # Add pulling metrics if available
            if "pulling_metrics" in continuous_metrics:
                pull_metrics = continuous_metrics["pulling_metrics"]
                if isinstance(pull_metrics, list):
                    metrics_to_analyze.extend(pull_metrics[:2])  # Take first 2

        # Run boxplot analysis for each metric
        for metric_name in metrics_to_analyze:
            print(f"\n📊 Running boxplot analysis for {metric_name}...")
            try:
                self.analyze_boxplots(mode, metric_name)
                print(f"✅ Boxplot analysis for {metric_name} completed")
            except Exception as e:
                print(f"❌ Boxplot analysis for {metric_name} failed: {e}")

        # 3. Coordinate analysis
        print(f"\n📈 Running coordinate analysis...")
        try:
            self.analyze_coordinates(mode)
            print(f"✅ Coordinate analysis completed")
        except Exception as e:
            print(f"❌ Coordinate analysis failed: {e}")

        print(f"\n{'='*60}")
        print(f"🎉 All available analyses completed for mode: {mode}")
        print(f"{'='*60}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Unified F1 tracks analysis framework")
    parser.add_argument("--mode", required=True, choices=["control", "tnt_mb247", "tnt_lc10_2"], help="Analysis mode")
    parser.add_argument(
        "--analysis",
        choices=["binary_metrics", "boxplots", "coordinates", "all"],
        help="Type of analysis to perform. If not specified, runs all available analyses.",
    )
    parser.add_argument("--metric", help="Specific metric for boxplot analysis")
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    # Initialize framework
    framework = F1AnalysisFramework(args.config)

    # If no analysis specified, run all available analyses
    if args.analysis is None or args.analysis == "all":
        print(f"Running all available analyses for mode: {args.mode}")
        framework.run_all_analyses(args.mode, args.metric)
    else:
        # Run specific analysis
        if args.analysis == "binary_metrics":
            framework.analyze_binary_metrics(args.mode)
        elif args.analysis == "boxplots":
            framework.analyze_boxplots(args.mode, args.metric or "interaction_rate")
        elif args.analysis == "coordinates":
            framework.analyze_coordinates(args.mode)

        print(f"Analysis completed for mode: {args.mode}, analysis: {args.analysis}")


if __name__ == "__main__":
    main()
