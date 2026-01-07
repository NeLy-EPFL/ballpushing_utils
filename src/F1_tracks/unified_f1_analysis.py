#!/usr/bin/env python3
"""
Unified F1 tracks analysis framework.
This script consolidates the functionality of multiple redundant analysis scripts
using a configuration-driven approach.

Usage:
    python unified_f1_analysis.py --mode control --analysis binary_metrics
    python unified_f1_analysis.py --mode tnt_mb247 --analysis boxplots --metric interaction_rate
    python unified_f1_analysis.py --mode tnt_lc10_2 --analysis coordinates --plot_type percentage
    python unified_f1_analysis.py --mode tnt_ddc --analysis binary --show
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
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle
from itertools import combinations
from tqdm import tqdm

# Pixel to mm conversion factor (500 pixels = 30 mm)
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


def is_distance_metric(metric_name):
    """Check if a metric represents distance (in pixels)"""
    distance_keywords = [
        "distance",
        "dist",
        "head_ball",
        "fly_distance_moved",
        "max_distance",
        "distance_moved",
        "distance_ratio",
    ]
    return any(keyword in metric_name.lower() for keyword in distance_keywords)


def is_time_metric(metric_name):
    """Check if a metric represents time (in seconds)"""
    time_keywords = ["time", "duration", "pause", "stop", "freeze", "chamber_exit_time", "time_chamber_beginning"]
    # Exclude ratio metrics
    if "ratio" in metric_name.lower():
        return False
    return any(keyword in metric_name.lower() for keyword in time_keywords)


def convert_metric_data(data, metric_name):
    """Convert metric data from pixels to mm or seconds to minutes"""
    if is_distance_metric(metric_name):
        return data / PIXELS_PER_MM
    elif is_time_metric(metric_name):
        return data / 60  # Convert seconds to minutes
    return data


def get_metric_unit(metric_name):
    """Get the display unit for a metric"""
    if is_distance_metric(metric_name):
        return "(mm)"
    elif is_time_metric(metric_name):
        return "(min)"
    return ""


def get_elegant_metric_name(metric_name):
    """Convert metric code names to elegant display names"""
    metric_map = {
        # Event metrics
        "nb_events": "Number of events",
        "has_major": "Has major event",
        "first_major_event": "First major event",
        "first_major_event_time": "First major event time",
        "max_event": "Max event",
        "max_event_time": "Max event time",
        "final_event": "Final event",
        "final_event_time": "Final event time",
        "has_significant": "Has significant event",
        "nb_significant_events": "Number of significant events",
        "significant_ratio": "Significant event ratio",
        "first_significant_event": "First significant event",
        "first_significant_event_time": "First significant event time",
        # Distance metrics
        "max_distance": "Max ball distance",
        "distance_moved": "Total ball distance moved",
        "distance_ratio": "Distance ratio",
        "fly_distance_moved": "Fly distance moved",
        # Interaction metrics
        "overall_interaction_rate": "Interaction rate",
        "interaction_persistence": "Interaction persistence",
        "interaction_proportion": "Interaction proportion",
        "time_to_first_interaction": "Time to first interaction",
        # Push/pull metrics
        "pushed": "Number of pushes",
        "pulled": "Number of pulls",
        "pulling_ratio": "Pulling ratio",
        "head_pushing_ratio": "Head pushing ratio",
        # Velocity metrics
        "normalized_velocity": "Normalized velocity",
        "velocity_during_interactions": "Velocity during interactions",
        "velocity_trend": "Velocity trend",
        # Pause/stop metrics
        "has_long_pauses": "Has long pauses",
        "nb_stops": "Number of stops",
        "total_stop_duration": "Total stop duration",
        "median_stop_duration": "Median stop duration",
        "nb_pauses": "Number of pauses",
        "total_pause_duration": "Total pause duration",
        "median_pause_duration": "Median pause duration",
        "nb_long_pauses": "Number of long pauses",
        "total_long_pause_duration": "Total long pause duration",
        "median_long_pause_duration": "Median long pause duration",
        "pauses_persistence": "Pauses persistence",
        "cumulated_breaks_duration": "Total break duration",
        # Freeze metrics
        "nb_freeze": "Number of freezes",
        "median_freeze_duration": "Median freeze duration",
        # Chamber metrics
        "chamber_time": "Chamber time",
        "chamber_ratio": "Chamber ratio",
        "chamber_exit_time": "Chamber exit time",
        "time_chamber_beginning": "Time in chamber (early)",
        # Task completion
        "has_finished": "Task completed",
        "persistence_at_end": "Persistence at end",
        # Other behavioral metrics
        "fraction_not_facing_ball": "Fraction not facing ball",
        "leg_visibility_ratio": "Leg visibility ratio",
        "flailing": "Leg flailing",
        "median_head_ball_distance": "Median head-ball distance",
        "mean_head_ball_distance": "Mean head-ball distance",
        # Ball proximity metrics
        "ball_proximity_proportion_70px": "Ball proximity (<70px)",
        "ball_proximity_proportion_140px": "Ball proximity (<140px)",
        "ball_proximity_proportion_200px": "Ball proximity (<200px)",
    }
    return metric_map.get(metric_name, metric_name)


def format_metric_label(metric_name):
    """Format metric name with elegant display name and appropriate unit"""
    elegant_name = get_elegant_metric_name(metric_name)
    unit = get_metric_unit(metric_name)
    if unit:
        return f"{elegant_name} {unit}"
    return elegant_name


def format_pretraining_label(value):
    """Convert pretraining code to intuitive label"""
    if str(value).lower() == "n":
        return "Control"
    elif str(value).lower() == "y":
        return "Pretrained"
    return str(value)


class F1AnalysisFramework:
    """Unified framework for F1 tracks analysis."""

    def __init__(self, config_path=None, output_dir=None, control_filter=None):
        """Initialize the framework with configuration.

        Args:
            config_path: Path to YAML configuration file
            output_dir: Base output directory for plots. If None, uses defaults based on mode:
                       - control: /mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New
                       - tnt_mb247: /mnt/upramdya_data/MD/F1_Tracks/MB247
                       - tnt_lc10_2: /mnt/upramdya_data/MD/F1_Tracks/LC10-2
                       - tnt_ddc: /mnt/upramdya_data/MD/F1_Tracks/DDC
            control_filter: Which control genotypes to include. Options:
                           - None: Include all genotypes in brain_region_mapping (default)
                           - 'EmptyGal4': Only TNTxEmptyGal4
                           - 'EmptySplit': Only TNTxEmptySplit
                           - 'both': Both EmptyGal4 and EmptySplit
                           - 'none': No controls (only experimental genotypes)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "analysis_config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Store custom output directory if provided
        self.custom_output_dir = Path(output_dir) if output_dir else None

        # Store control filter preference
        self.control_filter = control_filter

        # Storage for global axis ranges (will be populated when loading all data)
        self.global_axis_ranges = {}

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
        """Load dataset for specified mode, with support for pooled modes."""
        mode_config = self.config["analysis_modes"][mode]

        # Check if this is a pooled mode
        if "primary_dataset" in mode_config:
            return self._load_pooled_dataset(mode, mode_config)

        # Regular mode - load single dataset
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

            # Filter genotypes based on brain_region_mapping
            df = self._filter_allowed_genotypes(df, mode)

            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def _load_pooled_dataset(self, mode, mode_config):
        """Load and combine datasets for pooled analysis."""
        print(f"\n🔄 Loading pooled dataset for mode: {mode}")

        # Get primary dataset mode
        primary_mode = mode_config["primary_dataset"]
        primary_config = self.config["analysis_modes"][primary_mode]

        # Load primary dataset
        print(f"  📁 Loading primary dataset from: {primary_mode}")
        try:
            df_primary = pd.read_feather(primary_config["dataset_path"])
            print(f"  ✓ Primary dataset loaded: {df_primary.shape}")
        except Exception as e:
            print(f"  ❌ Error loading primary dataset: {e}")
            return None

        # Get genotypes to pool
        shared_genotypes = mode_config.get("shared_genotypes", [])
        pool_from_modes = mode_config.get("pool_from_modes", [])

        if not shared_genotypes or not pool_from_modes:
            print("  ⚠️  No shared genotypes or pool_from_modes specified, using primary dataset only")
            return df_primary

        # Find genotype column in primary dataset
        genotype_col = None
        for col in df_primary.columns:
            if "genotype" in col.lower():
                genotype_col = col
                break

        if not genotype_col:
            print("  ⚠️  Could not find genotype column, using primary dataset only")
            return df_primary

        # Collect pooled data for shared genotypes
        pooled_dfs = [df_primary]

        for pool_mode in pool_from_modes:
            if pool_mode not in self.config["analysis_modes"]:
                print(f"  ⚠️  Mode '{pool_mode}' not found in config, skipping")
                continue

            pool_config = self.config["analysis_modes"][pool_mode]

            # Skip if this is also a pooled mode (avoid recursion)
            if "primary_dataset" in pool_config:
                print(f"  ⚠️  Skipping pooled mode '{pool_mode}' to avoid recursion")
                continue

            try:
                print(f"  📁 Loading additional data from: {pool_mode}")
                df_pool = pd.read_feather(pool_config["dataset_path"])

                # Filter for shared genotypes only
                if genotype_col in df_pool.columns:
                    df_pool_filtered = df_pool[df_pool[genotype_col].isin(shared_genotypes)]

                    if len(df_pool_filtered) > 0:
                        print(
                            f"  ✓ Added {len(df_pool_filtered)} rows from {pool_mode} ({', '.join(shared_genotypes)})"
                        )
                        pooled_dfs.append(df_pool_filtered)
                    else:
                        print(f"  ⚠️  No matching genotypes found in {pool_mode}")
                else:
                    print(f"  ⚠️  Genotype column not found in {pool_mode}")

            except Exception as e:
                print(f"  ⚠️  Error loading from {pool_mode}: {e}")

        # Combine all datasets
        if len(pooled_dfs) > 1:
            df_combined = pd.concat(pooled_dfs, ignore_index=True)
            print(f"\n  ✅ Pooled dataset created: {df_combined.shape} (from {len(pooled_dfs)} sources)")

            # Filter out excluded dates
            excluded_dates = mode_config.get("excluded_dates", [])
            if excluded_dates and "Date" in df_combined.columns:
                initial_shape = df_combined.shape
                df_combined = df_combined[~df_combined["Date"].isin(excluded_dates)]
                print(f"  Removed excluded dates {excluded_dates}. Shape: {initial_shape} -> {df_combined.shape}")

            # Filter genotypes based on brain_region_mapping
            df_combined = self._filter_allowed_genotypes(df_combined, mode)

            # Print genotype distribution
            if genotype_col in df_combined.columns:
                print(f"\n  📊 Genotype distribution in pooled dataset:")
                genotype_counts = df_combined[genotype_col].value_counts()
                for genotype, count in genotype_counts.items():
                    print(f"     {genotype}: {count}")

            return df_combined
        else:
            print(f"\n  ℹ️  No additional data pooled, using primary dataset only")
            return df_primary

    def _filter_allowed_genotypes(self, df, mode):
        """Filter dataframe to only include genotypes specified in brain_region_mapping,
        with optional control genotype filtering.

        Args:
            df: DataFrame to filter
            mode: Analysis mode

        Returns:
            Filtered DataFrame
        """
        mode_config = self.config["analysis_modes"][mode]
        brain_region_mapping = mode_config.get("brain_region_mapping")

        if not brain_region_mapping:
            return df

        # Find genotype column
        genotype_col = None
        for col in df.columns:
            if "genotype" in col.lower():
                genotype_col = col
                break

        if not genotype_col:
            print("  ⚠️  Could not find genotype column for filtering")
            return df

        # Get allowed genotypes from brain_region_mapping
        allowed_genotypes = list(brain_region_mapping.keys())

        # Apply control filter if specified
        if self.control_filter:
            filtered_genotypes = self._apply_control_filter(allowed_genotypes, mode)
            if filtered_genotypes != allowed_genotypes:
                print(f"  🎯 Control filter '{self.control_filter}' applied")
                allowed_genotypes = filtered_genotypes

        # Filter dataframe
        initial_shape = df.shape
        df_filtered = df[df[genotype_col].isin(allowed_genotypes)]

        if df_filtered.shape[0] < initial_shape[0]:
            removed_count = initial_shape[0] - df_filtered.shape[0]
            removed_genotypes = set(df[~df[genotype_col].isin(allowed_genotypes)][genotype_col].unique())
            print(f"  🔍 Filtered genotypes: removed {removed_count} rows")
            print(f"     Removed genotypes: {', '.join(sorted(removed_genotypes))}")
            print(f"     Kept genotypes: {', '.join(sorted(allowed_genotypes))}")
            print(f"     Shape: {initial_shape} -> {df_filtered.shape}")

        return df_filtered

    def _apply_control_filter(self, genotypes, mode):
        """Apply control filter to genotype list.

        Args:
            genotypes: List of genotype names
            mode: Analysis mode

        Returns:
            Filtered list of genotypes
        """
        if not self.control_filter or self.control_filter.lower() == "none":
            # 'none' means no controls at all - only experimental genotypes
            if self.control_filter and self.control_filter.lower() == "none":
                return [g for g in genotypes if "Empty" not in g]
            return genotypes

        filter_lower = self.control_filter.lower()

        # Start with all genotypes
        result = []

        for genotype in genotypes:
            # Always keep non-control genotypes (experimental lines)
            if "Empty" not in genotype:
                result.append(genotype)
                continue

            # Filter control genotypes based on control_filter
            if filter_lower == "both":
                # Keep all controls
                result.append(genotype)
            elif filter_lower == "emptygal4":
                # Keep only EmptyGal4
                if "EmptyGal4" in genotype:
                    result.append(genotype)
            elif filter_lower == "emptysplit":
                # Keep only EmptySplit
                if "EmptySplit" in genotype:
                    result.append(genotype)

        return result

    def _load_fly_positions_data(self, mode):
        """Load fly positions data for the specified mode, with support for pooled modes."""
        mode_config = self.config["analysis_modes"][mode]

        # Check if this is a pooled mode
        if "primary_dataset" in mode_config:
            return self._load_pooled_fly_positions(mode, mode_config)

        # Regular mode - load single fly positions dataset
        fly_positions_path = mode_config.get("fly_positions_path")

        if not fly_positions_path:
            print("ℹ️  No fly_positions_path specified for this mode")
            return None

        try:
            df_positions = pd.read_feather(fly_positions_path)
            print(f"✓ Loaded fly positions data: {df_positions.shape}")
            return df_positions
        except Exception as e:
            print(f"⚠️  Could not load fly positions data: {e}")
            return None

    def _load_pooled_fly_positions(self, mode, mode_config):
        """Load and combine fly positions data for pooled analysis."""
        print(f"\n🔄 Loading pooled fly positions data for mode: {mode}")

        # Get primary dataset mode
        primary_mode = mode_config["primary_dataset"]
        primary_config = self.config["analysis_modes"][primary_mode]

        # Load primary fly positions dataset
        print(f"  📁 Loading primary fly positions from: {primary_mode}")
        primary_positions_path = primary_config.get("fly_positions_path")

        if not primary_positions_path:
            print(f"  ⚠️  No fly_positions_path in primary mode '{primary_mode}'")
            return None

        try:
            df_positions_primary = pd.read_feather(primary_positions_path)
            print(f"  ✓ Primary fly positions loaded: {df_positions_primary.shape}")
        except Exception as e:
            print(f"  ❌ Error loading primary fly positions: {e}")
            return None

        # Get genotypes to pool
        shared_genotypes = mode_config.get("shared_genotypes", [])
        pool_from_modes = mode_config.get("pool_from_modes", [])

        if not shared_genotypes or not pool_from_modes:
            print("  ℹ️  No shared genotypes or pool_from_modes specified, using primary fly positions only")
            return df_positions_primary

        # Find genotype column in primary dataset
        genotype_col = None
        for col in df_positions_primary.columns:
            if "genotype" in col.lower():
                genotype_col = col
                break

        if not genotype_col:
            print("  ⚠️  Could not find genotype column in fly positions, using primary data only")
            return df_positions_primary

        # Collect pooled fly positions data for shared genotypes
        pooled_positions_dfs = [df_positions_primary]

        for pool_mode in pool_from_modes:
            if pool_mode not in self.config["analysis_modes"]:
                print(f"  ⚠️  Mode '{pool_mode}' not found in config, skipping")
                continue

            pool_config = self.config["analysis_modes"][pool_mode]

            # Skip if this is also a pooled mode (avoid recursion)
            if "primary_dataset" in pool_config:
                print(f"  ⚠️  Skipping pooled mode '{pool_mode}' to avoid recursion")
                continue

            # Get fly positions path for this pool mode
            pool_positions_path = pool_config.get("fly_positions_path")
            if not pool_positions_path:
                print(f"  ⚠️  No fly_positions_path in mode '{pool_mode}', skipping")
                continue

            try:
                print(f"  📁 Loading fly positions from: {pool_mode}")
                df_pool_positions = pd.read_feather(pool_positions_path)

                # Filter for shared genotypes only
                if genotype_col in df_pool_positions.columns:
                    df_pool_positions_filtered = df_pool_positions[
                        df_pool_positions[genotype_col].isin(shared_genotypes)
                    ]

                    if len(df_pool_positions_filtered) > 0:
                        # Count unique flies
                        n_flies = (
                            df_pool_positions_filtered["fly"].nunique()
                            if "fly" in df_pool_positions_filtered.columns
                            else "?"
                        )
                        print(
                            f"  ✓ Added {len(df_pool_positions_filtered)} position frames from {pool_mode} "
                            f"({n_flies} flies, {', '.join(shared_genotypes)})"
                        )
                        pooled_positions_dfs.append(df_pool_positions_filtered)
                    else:
                        print(f"  ⚠️  No matching genotypes found in {pool_mode} fly positions")
                else:
                    print(f"  ⚠️  Genotype column not found in {pool_mode} fly positions")

            except Exception as e:
                print(f"  ⚠️  Error loading fly positions from {pool_mode}: {e}")

        # Combine all fly positions datasets
        if len(pooled_positions_dfs) > 1:
            df_combined_positions = pd.concat(pooled_positions_dfs, ignore_index=True)
            n_flies = df_combined_positions["fly"].nunique() if "fly" in df_combined_positions.columns else "?"
            print(
                f"\n  ✅ Pooled fly positions created: {df_combined_positions.shape} "
                f"({n_flies} flies, from {len(pooled_positions_dfs)} sources)"
            )

            # Print genotype distribution
            if genotype_col in df_combined_positions.columns:
                print(f"\n  📊 Genotype distribution in pooled fly positions:")
                genotype_fly_counts = df_combined_positions.groupby(genotype_col)["fly"].nunique()
                for genotype, count in genotype_fly_counts.items():
                    print(f"     {genotype}: {count} flies")

            return df_combined_positions
        else:
            print(f"\n  ℹ️  No additional fly positions pooled, using primary data only")
            return df_positions_primary

    def detect_columns(self, df, mode):
        """Auto-detect column names based on patterns."""
        mode_config = self.config["analysis_modes"][mode]
        grouping_vars = mode_config["grouping_variables"]
        detection_config = self.config["column_detection"]

        detected_cols = {}

        # Columns to explicitly exclude from detection (never use as grouping variables)
        excluded_cols = {"fly", "time_bin", "time_bin_center", "adjusted_time"}

        # Detect primary grouping variable (usually pretraining)
        primary_key = grouping_vars["primary"]
        if primary_key in detection_config:
            patterns = detection_config[primary_key]["patterns"]
            for col in df.columns:
                # Skip excluded columns
                if col.lower() in excluded_cols or col in excluded_cols:
                    continue

                # Skip columns with "distance" or "ball" in the name (these are measurements, not groups)
                if "distance" in col.lower() or "euclidean" in col.lower():
                    continue

                if any(pattern.lower() in col.lower() for pattern in patterns):
                    # prefer non-numeric columns for grouping variables (e.g., 'Pretraining')
                    try:
                        col_dtype = df[col].dtype
                    except Exception:
                        col_dtype = None

                    if col_dtype is not None and np.issubdtype(col_dtype, np.number):
                        # skip numeric columns like training distances
                        continue

                    detected_cols["primary"] = col
                    break

        # Detect secondary grouping variable (F1_condition or Genotype)
        secondary_key = grouping_vars["secondary"]
        if secondary_key in detection_config:
            patterns = detection_config[secondary_key]["patterns"]
            for col in df.columns:
                # Skip excluded columns
                if col.lower() in excluded_cols or col in excluded_cols:
                    continue

                # Skip columns with "distance" or "ball" in the name
                if "distance" in col.lower() or "euclidean" in col.lower():
                    continue

                if any(pattern.lower() in col.lower() for pattern in patterns):
                    # prefer non-numeric columns for grouping variables
                    try:
                        col_dtype = df[col].dtype
                    except Exception:
                        col_dtype = None

                    if col_dtype is not None and np.issubdtype(col_dtype, np.number):
                        continue

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

    def calculate_global_axis_ranges(self, df, metrics_to_analyze):
        """Calculate global min/max ranges for all metrics after conversion.

        Args:
            df: DataFrame with all data
            metrics_to_analyze: List of metric column names

        Returns:
            Dictionary mapping metric names to (min, max) tuples
        """
        print(f"\n📊 Calculating global axis ranges for {len(metrics_to_analyze)} metrics...")

        ranges = {}
        for metric in metrics_to_analyze:
            if metric not in df.columns:
                continue

            # Get raw data
            raw_data = df[metric].dropna()

            if len(raw_data) == 0:
                continue

            # Convert data
            converted_data = convert_metric_data(raw_data, metric)

            # Calculate range with some padding (5% on each side)
            data_min = converted_data.min()
            data_max = converted_data.max()
            data_range = data_max - data_min
            padding = data_range * 0.05

            ranges[metric] = (data_min - padding, data_max + padding)

            print(f"  {metric}: [{data_min:.2f}, {data_max:.2f}] -> [{ranges[metric][0]:.2f}, {ranges[metric][1]:.2f}]")

        return ranges

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
        elif variable_type in ("brain_region", "genotype") and "brain_region_colors" in plot_config:
            # Map either brain-region names or genotypes to brain-region colors.
            # If the provided value is already a brain-region name, use it directly;
            # otherwise try to resolve genotype -> brain_region -> color.
            color_mapping = {}
            for val in values:
                # if the value directly matches a brain region entry, use it
                if val in plot_config["brain_region_colors"]:
                    color_mapping[val] = plot_config["brain_region_colors"].get(val)
                    continue

                # otherwise, attempt to map genotype -> brain region -> color
                brain_region = self.get_brain_region_for_genotype(val, mode)
                color_mapping[val] = plot_config["brain_region_colors"].get(brain_region, "#808080")
            return color_mapping
        else:
            # Fallback to seaborn palette
            colors = sns.color_palette("Set2", n_colors=len(values))
            return dict(zip(values, colors))

    def create_combined_style_mapping(self, mode, genotypes, pretraining_values):
        """
        Create combined style mapping for genotype + pretraining.
        Returns dict mapping (genotype, pretraining) tuples to {'color': ..., 'linestyle': ...}.

        Args:
            mode: Analysis mode
            genotypes: List of unique genotype values
            pretraining_values: List of unique pretraining values

        Returns:
            Dictionary with keys (genotype, pretraining) and values {'color': str, 'linestyle': str}
        """
        plot_config = self.config["plot_styling"][mode]
        style_mapping = {}

        # Map genotypes to brain region colors
        genotype_colors = {}
        if "brain_region_colors" in plot_config:
            for genotype in genotypes:
                brain_region = self.get_brain_region_for_genotype(genotype, mode)
                genotype_colors[genotype] = plot_config["brain_region_colors"].get(brain_region, "#808080")
        else:
            # Fallback to default colors
            colors = sns.color_palette("Set2", n_colors=len(genotypes))
            genotype_colors = dict(zip(genotypes, colors))

        # Linestyle mapping: dotted for 'n', solid for 'y'
        linestyle_map = {"n": ":", "y": "-"}

        # Create combined mapping
        for genotype in genotypes:
            for pretraining in pretraining_values:
                style_mapping[(genotype, pretraining)] = {
                    "color": genotype_colors[genotype],
                    "linestyle": linestyle_map.get(pretraining, "-"),
                }

        return style_mapping

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
            if p_value is not None and isinstance(p_value, (int, float, np.floating)) and not np.isnan(p_value):
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
            if p_value is not None and isinstance(p_value, (int, float, np.floating)) and not np.isnan(p_value):
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
        """Save plot to file in both PNG and PDF formats."""
        # Set PDF font to Type 42 (TrueType) for editability
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        output_config = self.config["output"]
        if output_config["save_plots"]:
            # Determine output directory
            if self.custom_output_dir:
                # Use custom output directory if provided
                output_dir = self.custom_output_dir
            else:
                # Use default paths based on mode (handle pooled modes)
                base_mode = mode.replace("_pooled", "")
                if base_mode == "control":
                    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New")
                elif base_mode == "tnt_mb247":
                    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/MB247")
                elif base_mode == "tnt_lc10_2":
                    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/LC10-2")
                elif base_mode == "tnt_ddc":
                    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/DDC")
                else:
                    # Fallback to current directory for unknown modes
                    output_dir = Path(__file__).parent

            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Add pooled suffix to filename if in pooled mode
            if "_pooled" in mode:
                plot_filename = f"{plot_name}_pooled"
            else:
                plot_filename = plot_name

            # Add control filter suffix to filename if specified
            if self.control_filter:
                control_suffix = self.control_filter.replace(" ", "")
                plot_filename = f"{plot_filename}_{control_suffix}"

            # Save in both PNG and PDF formats
            formats_to_save = ["png", "pdf"]

            for fmt in formats_to_save:
                output_path = output_dir / f"{plot_filename}.{fmt}"

                # Attempt save, but catch very large-image errors and retry with safer settings
                try:
                    plt.savefig(output_path, dpi=output_config["dpi"], bbox_inches=output_config["bbox_inches"])
                    print(f"Plot saved to: {output_path}")
                except Exception as e:
                    print(f"⚠️  Save failed ({fmt}): {e}")
                    # Try a safer save: lower DPI and no bbox trimming (for PNG only)
                    if fmt == "png":
                        try:
                            safe_dpi = max(80, int(output_config.get("dpi", 300) / 2))
                            print(f"ℹ️  Retrying PNG save with dpi={safe_dpi} and bbox_inches=None")
                            plt.savefig(output_path, dpi=safe_dpi, bbox_inches=None)
                            print(f"Plot saved to (safe): {output_path}")
                        except Exception as e2:
                            print(f"⚠️  Safe PNG save also failed: {e2}")
                            # As a last resort, save a small thumbnail
                            try:
                                thumb_path = output_dir / f"{plot_name}_thumbnail.png"
                                print(f"ℹ️  Saving thumbnail to {thumb_path} (low dpi)")
                                plt.savefig(thumb_path, dpi=72, bbox_inches=None)
                                print(f"Thumbnail saved to: {thumb_path}")
                            except Exception as e3:
                                print(f"❌ Failed to save PNG plot or thumbnail: {e3}")
                    else:
                        # For PDF, just try with no bbox trimming
                        try:
                            print(f"ℹ️  Retrying PDF save with bbox_inches=None")
                            plt.savefig(output_path, bbox_inches=None)
                            print(f"PDF saved to: {output_path}")
                        except Exception as e2:
                            print(f"❌ Failed to save PDF: {e2}")

        if output_config["show_plots"]:
            try:
                plt.show()
            except Exception:
                # If display fails (headless env) just close
                plt.close()
        else:
            plt.close()

    def _permutation_test(self, groups, stat_func, n_permutations=10000):
        """
        Perform permutation test for difference between groups.

        Args:
            groups: List of arrays, one per group
            stat_func: Function to calculate test statistic (e.g., lambda x: [np.mean(g) for g in x])
            n_permutations: Number of permutations

        Returns:
            p_value: Two-tailed p-value
            observed_stat: Observed test statistic
        """
        # Calculate observed statistic
        observed_stat = stat_func(groups)

        # For mean difference: use difference between groups
        # For median difference: use difference between groups
        if len(groups) == 2:
            observed_diff = observed_stat[0] - observed_stat[1]
        else:
            # For >2 groups, use variance of means/medians as test statistic
            observed_diff = np.var(observed_stat)

        # Combine all data
        all_data = np.concatenate(groups)
        group_sizes = [len(g) for g in groups]

        # Perform permutations
        perm_stats = []
        for _ in range(n_permutations):
            # Shuffle data
            shuffled = np.random.permutation(all_data)

            # Split into groups
            perm_groups = []
            start_idx = 0
            for size in group_sizes:
                perm_groups.append(shuffled[start_idx : start_idx + size])
                start_idx += size

            # Calculate statistic
            perm_stat = stat_func(perm_groups)

            if len(groups) == 2:
                perm_diff = perm_stat[0] - perm_stat[1]
            else:
                perm_diff = np.var(perm_stat)

            perm_stats.append(perm_diff)

        # Calculate p-value (two-tailed)
        perm_stats = np.array(perm_stats)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_diff))

        return p_value, observed_diff

    def _bootstrap_ci(self, data, stat_func=np.mean, n_bootstrap=10000, confidence=0.95):
        """
        Calculate bootstrap confidence interval for a statistic.

        Args:
            data: Array of data
            stat_func: Function to calculate statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            ci_low, ci_high: Confidence interval bounds
            point_estimate: Point estimate of the statistic
        """
        bootstrap_stats = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(stat_func(sample))

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        ci_low = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_high = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        point_estimate = stat_func(data)

        return ci_low, ci_high, point_estimate

    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def _fdr_correction(self, p_values, alpha=0.05):
        """
        Benjamini-Hochberg FDR correction for multiple comparisons.

        Args:
            p_values: List or array of p-values
            alpha: FDR threshold

        Returns:
            rejected: Boolean array indicating which hypotheses are rejected
            corrected_p: FDR-corrected p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)

        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Calculate critical values
        critical_values = (np.arange(1, n + 1) / n) * alpha

        # Find largest i where p(i) <= (i/n)*alpha
        comparisons = sorted_p <= critical_values
        if np.any(comparisons):
            max_idx = np.where(comparisons)[0][-1]
            rejected_sorted = np.zeros(n, dtype=bool)
            rejected_sorted[: max_idx + 1] = True
        else:
            rejected_sorted = np.zeros(n, dtype=bool)

        # Restore original order
        rejected = np.zeros(n, dtype=bool)
        rejected[sorted_indices] = rejected_sorted

        # Calculate corrected p-values (q-values)
        corrected_p = np.minimum(1, p_values * n / np.arange(1, n + 1)[np.argsort(np.argsort(p_values))])

        return rejected, corrected_p

    # TODO: Cleanup this part

    def calculate_ball_proximity_time(self, df, proximity_thresholds=[70, 140, 200], movement_threshold=100, n_avg=100):
        """
        DEPRECATED: This method is no longer needed and should not be used.

        Ball proximity time is now calculated during dataset generation as part of
        ballpushing_metrics.compute_ball_proximity_proportions(). The metrics are available
        in summary datasets as 'ball_proximity_proportion_XXpx' columns.

        To enable these metrics in your dataset:
        1. Add them to 'enabled_metrics' in your config, OR
        2. Regenerate your dataset (they will be computed automatically)

        This method is kept for backward compatibility only.

        Calculate the proportion of time each fly spends near the ball at multiple distance thresholds.

        This metric is calculated from fly position data by:
        1. Filtering data to keep only points after fly first moves >movement_threshold from starting position
        2. Calculating Euclidean distance between fly thorax and ball center
        3. Computing proportion of time where distance < proximity_threshold for each threshold

        Args:
            df: DataFrame with fly and ball position data (requires fly_positions data)
            proximity_thresholds: List of distance thresholds for proximity in pixels (default: [70, 140, 200])
            movement_threshold: Distance fly must move from start before counting (default: 100)
            n_avg: Number of initial points to average for starting position (default: 100)

        Returns:
            DataFrame with fly, condition, and ball_proximity_proportion_XXpx columns for each threshold
        """
        print("\n⚠️  WARNING: calculate_ball_proximity_time() is DEPRECATED")
        print("   Ball proximity is now computed during dataset generation.")
        print("   Please use summary datasets with these metrics enabled instead.\n")
        if not isinstance(proximity_thresholds, list):
            proximity_thresholds = [proximity_thresholds]

        print(f"\n📏 Calculating ball proximity time (thresholds={proximity_thresholds}px)...")

        # Check for required columns
        required_cols = ["fly", "x_thorax_fly_0", "y_thorax_fly_0"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ⚠️  Missing required columns for ball proximity calculation: {missing_cols}")
            print("  This metric requires fly_positions data, not summary data")
            return None

        # Filter data from first movement
        filtered_dfs = []

        for fly_id in df["fly"].unique():
            fly_data = df[df["fly"] == fly_id].copy()

            # Sort by frame or index
            if "frame" in fly_data.columns:
                fly_data = fly_data.sort_values("frame")
            else:
                fly_data = fly_data.sort_index()

            # Calculate starting position
            n_use = min(n_avg, len(fly_data))
            if n_use == 0:
                continue

            x_start = fly_data["x_thorax_fly_0"].iloc[:n_use].mean()

            # Calculate distance from start
            fly_data["distance_from_start"] = np.abs(fly_data["x_thorax_fly_0"] - x_start)

            # Find first time distance exceeds threshold
            threshold_mask = fly_data["distance_from_start"] > movement_threshold

            if threshold_mask.any():
                first_threshold_idx = threshold_mask.idxmax()
                fly_data_filtered = fly_data.loc[first_threshold_idx:]
                filtered_dfs.append(fly_data_filtered)

        if not filtered_dfs:
            print("  ⚠️  No flies met movement threshold criteria")
            return None

        df_filtered = pd.concat(filtered_dfs, ignore_index=True)
        print(f"  Filtered to {len(df_filtered)} frames from {df_filtered['fly'].nunique()} flies")

        # Calculate proximity proportions
        results = []

        for fly_id in df_filtered["fly"].unique():
            fly_data = df_filtered[df_filtered["fly"] == fly_id].copy()

            if len(fly_data) == 0:
                continue

            # Get condition information
            f1_condition = fly_data["F1_condition"].iloc[0] if "F1_condition" in fly_data.columns else None
            pretraining = fly_data["Pretraining"].iloc[0] if "Pretraining" in fly_data.columns else None
            genotype = fly_data["Genotype"].iloc[0] if "Genotype" in fly_data.columns else None

            # Determine which ball to use based on condition
            if f1_condition == "control" or pretraining == "n":
                ball_x_col = "x_centre_ball_0"
                ball_y_col = "y_centre_ball_0"
            else:
                ball_x_col = "x_centre_ball_1"
                ball_y_col = "y_centre_ball_1"

            # Check if ball columns exist
            if ball_x_col not in fly_data.columns or ball_y_col not in fly_data.columns:
                continue

            # Get fly and ball positions
            fly_x = fly_data["x_thorax_fly_0"].values
            fly_y = fly_data["y_thorax_fly_0"].values
            ball_x = fly_data[ball_x_col].values
            ball_y = fly_data[ball_y_col].values

            # Calculate distance to ball
            distance_to_ball = np.sqrt((fly_x - ball_x) ** 2 + (fly_y - ball_y) ** 2)

            # Calculate proportion near ball for each threshold
            result_dict = {
                "fly": fly_id,
                "F1_condition": f1_condition,
                "Pretraining": pretraining,
                "Genotype": genotype,
                "n_timepoints": len(fly_data),
            }

            for threshold in proximity_thresholds:
                near_ball = distance_to_ball < threshold
                proportion_near = near_ball.sum() / len(near_ball) if len(near_ball) > 0 else 0
                result_dict[f"ball_proximity_proportion_{threshold}px"] = proportion_near
                result_dict[f"n_near_ball_{threshold}px"] = near_ball.sum()

            results.append(result_dict)

        result_df = pd.DataFrame(results)

        # Print summary for each threshold
        for threshold in proximity_thresholds:
            col_name = f"ball_proximity_proportion_{threshold}px"
            if col_name in result_df.columns:
                mean_proximity = result_df[col_name].mean()
                print(f"  ✓ {threshold}px: mean proximity = {mean_proximity:.3f} ({len(result_df)} flies)")

        return result_df

    def add_calculated_metrics(self, df, mode, include_proximity=False):
        """
        Add calculated metrics to the dataset.

        Args:
            df: DataFrame to add metrics to
            mode: Analysis mode
            include_proximity: DEPRECATED - ball proximity is now calculated during dataset generation
                              as part of ballpushing_metrics. The metrics should already be present
                              in summary datasets as 'ball_proximity_proportion_XXpx' columns.

        Returns:
            DataFrame with added metrics
        """
        df_enriched = df.copy()

        # Ensure time_to_first_interaction exists
        if "time_to_first_interaction" not in df_enriched.columns:
            # Check for alternative column names
            alt_names = ["first_interaction_time", "first_event_time", "first_significant_event_time"]
            found = False

            for alt_name in alt_names:
                if alt_name in df_enriched.columns:
                    print(f"  ℹ️  Using '{alt_name}' as 'time_to_first_interaction'")
                    df_enriched["time_to_first_interaction"] = df_enriched[alt_name]
                    found = True
                    break

            if not found:
                print("  ⚠️  'time_to_first_interaction' not found in dataset")
                print("     This metric should be calculated during data preprocessing")

        # Ball proximity metrics should already be in the summary dataset
        # Check if they exist and warn if not
        proximity_metrics = [
            "ball_proximity_proportion_70px",
            "ball_proximity_proportion_140px",
            "ball_proximity_proportion_200px",
        ]
        missing_proximity = [m for m in proximity_metrics if m not in df_enriched.columns]

        if missing_proximity and include_proximity:
            print(f"  ⚠️  Ball proximity metrics not found in dataset: {missing_proximity}")
            print("     These should be calculated during dataset generation by enabling them in ballpushing_metrics.")
            print("     Add them to 'enabled_metrics' in your config or regenerate the dataset.")
        elif not missing_proximity:
            print(f"  ✓ Ball proximity metrics present: {proximity_metrics}")

        return df_enriched

    def analyze_boxplots(self, mode, metric_type=None, test_type="both"):
        """Analyze continuous metrics using boxplots with scatter overlay and pairwise comparisons.

        If metric_type is None, analyzes all available metrics from the config.

        Args:
            mode: Analysis mode (control, tnt_mb247, tnt_lc10_2, tnt_ddc, etc.)
            metric_type: Specific metric to analyze, or None for all metrics
            test_type: Statistical test to use ('mannwhitney', 'permutation', or 'both')
        """
        # Load and prepare data
        df = self.load_dataset(mode)
        if df is None:
            return

        # Try to load fly positions data for ball proximity calculation
        df_positions = self._load_fly_positions_data(mode)

        # Calculate ball proximity if fly positions data is available
        if df_positions is not None:
            try:
                # Calculate ball proximity metric with multiple thresholds
                proximity_data = self.calculate_ball_proximity_time(df_positions, proximity_thresholds=[70, 140, 200])

                if proximity_data is not None:
                    # Merge proximity metrics into main dataframe
                    proximity_cols = [
                        col for col in proximity_data.columns if col.startswith("ball_proximity_proportion_")
                    ]
                    merge_cols = ["fly"] + proximity_cols
                    df = df.merge(
                        proximity_data[merge_cols],
                        on="fly",
                        how="left",
                        suffixes=("", "_proximity"),
                    )
                    print(f"✓ Added {len(proximity_cols)} ball proximity metrics to dataset: {proximity_cols}")
            except Exception as e:
                print(f"⚠️  Error calculating ball proximity: {e}")
                print("   Ball proximity metrics will not be available")

        detected_cols = self.detect_columns(df, mode)
        if not all(key in detected_cols for key in ["primary", "secondary", "ball_condition"]):
            print("Could not detect all required columns")
            return

        # Filter data
        df_clean = self.filter_for_test_ball(df, detected_cols["ball_condition"])

        # Discover available metrics
        continuous_metrics_config = self.config["analysis_parameters"]["continuous_metrics"]

        # Collect all metrics to analyze
        metrics_to_analyze = []

        if metric_type is None:
            # Discover all available metrics
            print("\n🔍 Discovering available metrics...")

            # Get base metrics
            base_metrics = continuous_metrics_config.get("base_metrics", [])
            for metric in base_metrics:
                if metric in df_clean.columns:
                    metrics_to_analyze.append(metric)
                    print(f"  ✓ {metric}")

            # Get pattern-based metrics
            additional_patterns = continuous_metrics_config.get("additional_patterns", [])
            excluded_patterns = continuous_metrics_config.get("excluded_patterns", [])

            for pattern in additional_patterns:
                pattern_cols = [col for col in df_clean.columns if pattern in col.lower()]
                for col in pattern_cols:
                    if col not in metrics_to_analyze and col not in [
                        detected_cols["primary"],
                        detected_cols["secondary"],
                        detected_cols["ball_condition"],
                    ]:
                        # Check if should be excluded
                        should_exclude = any(excl_pattern in col.lower() for excl_pattern in excluded_patterns)
                        if should_exclude:
                            continue
                        # Check if numeric
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            metrics_to_analyze.append(col)
                            print(f"  ✓ {col} (pattern: '{pattern}')")

            print(f"\n📊 Found {len(metrics_to_analyze)} metrics to analyze")

        else:
            # Single metric specified
            if metric_type in continuous_metrics_config:
                if isinstance(continuous_metrics_config[metric_type], str):
                    metrics_to_analyze = [continuous_metrics_config[metric_type]]
                else:
                    metrics_to_analyze = continuous_metrics_config[metric_type]
            else:
                metrics_to_analyze = [metric_type]

        if not metrics_to_analyze:
            print("❌ No metrics found to analyze")
            return

        # Calculate global axis ranges from all data
        print(f"\n🔍 Calculating global axis ranges across all {len(metrics_to_analyze)} metrics...")
        self.global_axis_ranges = self.calculate_global_axis_ranges(df_clean, metrics_to_analyze)

        # Storage for all results (for markdown summary)
        all_results = []

        # Analyze each metric
        for metric_idx, metric_col in enumerate(metrics_to_analyze):
            print(f"\n{'='*60}")
            print(f"Analyzing metric {metric_idx + 1}/{len(metrics_to_analyze)}: {metric_col}")
            print(f"{'='*60}")

            if metric_col not in df_clean.columns:
                print(f"  ⚠️  Metric '{metric_col}' not found in dataset, skipping")
                continue

            # Prepare analysis data for this metric
            analysis_cols = [
                detected_cols["primary"],
                detected_cols["secondary"],
                detected_cols["ball_condition"],
                metric_col,
            ]
            df_metric = df_clean[analysis_cols].dropna(
                subset=[detected_cols["primary"], detected_cols["secondary"], detected_cols["ball_condition"]]
            )

            # Remove infinite values
            mask = pd.isna(df_metric[metric_col]) | np.isinf(df_metric[metric_col])
            df_metric = df_metric[~mask]

            if len(df_metric) < 2:
                print(f"  ⚠️  Insufficient data for '{metric_col}', skipping")
                continue

            print(f"  Data shape: {df_metric.shape}")

            # Perform pairwise comparisons and create plots
            # Create separate plots for each test type
            test_types = []
            if test_type == "both":
                test_types = ["mannwhitney", "permutation"]
            else:
                test_types = [test_type]

            for current_test in test_types:
                print(f"  Creating plots with {current_test} test...")

                if mode == "control":
                    metric_results = self._create_control_boxplots_with_pairwise(
                        df_metric, detected_cols, metric_col, mode, test_type=current_test
                    )
                else:  # TNT modes
                    metric_results = self._create_tnt_boxplots_with_pairwise(
                        df_metric, detected_cols, metric_col, mode, test_type=current_test
                    )

                # Store results
                if metric_results:
                    all_results.append({"metric": metric_col, "results": metric_results, "test_type": current_test})

        # Generate markdown summary
        if all_results:
            self._generate_boxplot_summary_markdown(all_results, mode, detected_cols)

        print(f"\n{'='*60}")
        print(f"✅ Boxplot analysis complete! Analyzed {len(all_results)} metrics")
        print(f"{'='*60}\n")

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

        # Prepare data for statistical tests (RAW, unconverted)
        groups_raw = [data[data[x_col] == condition][y_col].dropna().values for condition in unique_values]

        # Convert data for display
        data_display = data.copy()
        data_display[y_col] = convert_metric_data(data_display[y_col], y_col)

        # Prepare converted data for plotting
        groups_converted = [
            data_display[data_display[x_col] == condition][y_col].dropna().values for condition in unique_values
        ]

        # Create boxplot with converted data FIRST (drawn below)
        box_plot = sns.boxplot(
            data=data_display,
            x=x_col,
            y=y_col,
            ax=ax,
            order=unique_values,
            palette=[color_mapping.get(v, "gray") for v in unique_values],
            width=0.6,
            zorder=1,  # Lower z-order so scatter appears on top
        )

        # Add scatter with converted data on top
        for i, condition in enumerate(unique_values):
            condition_data = groups_converted[i]
            x_pos = np.random.normal(i, 0.04, size=len(condition_data))
            ax.scatter(
                x_pos,
                condition_data,
                alpha=0.6,
                s=50,
                color=color_mapping.get(condition, "gray"),
                edgecolors="black",
                linewidths=0.5,
                zorder=2,  # Higher z-order so scatter appears on top
            )

        # Style boxplot (F1 style: transparent boxes, colored edges)
        for patch in box_plot.patches:
            patch.set_facecolor("white")
            patch.set_alpha(0.7)
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_linewidth(2)

        for line in ax.lines:
            line.set_color("black")
            line.set_linewidth(1.5)

        # Formatting with elegant metric name
        ax.set_title(title, fontsize=14, fontweight="bold", fontname="Arial")
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12, fontname="Arial")
        ax.set_ylabel(format_metric_label(y_col), fontsize=12, fontname="Arial")

        # Apply global axis range if available
        if y_col in self.global_axis_ranges:
            ax.set_ylim(self.global_axis_ranges[y_col])

        # Add sample sizes to x-axis labels
        x_labels = []
        for i, condition in enumerate(unique_values):
            n = len(groups_raw[i])
            # Format pretraining labels if this is pretraining column
            if var_type == "pretraining":
                display_name = format_pretraining_label(condition)
            else:
                display_name = str(condition)
            x_labels.append(f"{display_name}\n(n={n})")
        ax.set_xticklabels(x_labels, fontname="Arial")

        # Statistical testing with permutation tests on RAW data
        if len(groups_raw) < 2:
            return ax

        # Perform permutation tests for mean and median
        p_mean, diff_mean = self._permutation_test(groups_raw, lambda x: [np.mean(g) for g in x])
        p_median, diff_median = self._permutation_test(groups_raw, lambda x: [np.median(g) for g in x])

        # For multiple groups, apply FDR correction
        if len(groups_raw) > 2:
            # Pairwise comparisons
            p_values_mean = []
            p_values_median = []
            comparisons = []

            for i in range(len(groups_raw)):
                for j in range(i + 1, len(groups_raw)):
                    p_m, _ = self._permutation_test([groups_raw[i], groups_raw[j]], lambda x: [np.mean(g) for g in x])
                    p_md, _ = self._permutation_test(
                        [groups_raw[i], groups_raw[j]], lambda x: [np.median(g) for g in x]
                    )
                    p_values_mean.append(p_m)
                    p_values_median.append(p_md)
                    comparisons.append(f"{unique_values[i]} vs {unique_values[j]}")

            # FDR correction
            _, p_mean_corrected = self._fdr_correction(p_values_mean)
            _, p_median_corrected = self._fdr_correction(p_values_median)

            # Use the minimum corrected p-value for display
            p_mean_display = np.min(p_mean_corrected)
            p_median_display = np.min(p_median_corrected)
            test_label = "Permutation (FDR corrected)"
        else:
            p_mean_display = p_mean
            p_median_display = p_median
            test_label = "Permutation test"

        # Bootstrap confidence intervals for mean and median (on converted data for display)
        ci_text_lines = []
        for i, condition in enumerate(unique_values):
            group_data_converted = convert_metric_data(groups_raw[i], y_col)
            # Mean CI
            ci_low_mean, ci_high_mean, mean_est = self._bootstrap_ci(group_data_converted, np.mean)
            # Median CI
            ci_low_median, ci_high_median, median_est = self._bootstrap_ci(group_data_converted, np.median)
            ci_text_lines.append(f"{condition}:")
            ci_text_lines.append(f"  μ={mean_est:.3f} [{ci_low_mean:.3f}, {ci_high_mean:.3f}]")
            ci_text_lines.append(f"  M={median_est:.3f} [{ci_low_median:.3f}, {ci_high_median:.3f}]")

        # Calculate effect size for 2-group comparison
        if len(groups_raw) == 2:
            cohens_d = self._cohens_d(groups_raw[0], groups_raw[1])
            effect_text = f"Cohen's d = {cohens_d:.3f}"
        else:
            effect_text = ""

        # Create annotation text
        stat_text = f"{test_label}\n"
        stat_text += f"Mean: p = {p_mean_display:.4f}\n"
        stat_text += f"Median: p = {p_median_display:.4f}"
        if effect_text:
            stat_text += f"\n{effect_text}"

        # Add statistical annotation box
        ax.text(
            0.02,
            0.98,
            stat_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        # Add bootstrap CI information
        ci_full_text = "Bootstrap 95% CI:\n" + "\n".join(ci_text_lines)
        ax.text(
            0.98,
            0.98,
            ci_full_text,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
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

        # Prepare data for statistical tests (RAW, unconverted)
        groups_raw = [data_copy[data_copy["combined_group"] == group][y_col].dropna().values for group in unique_groups]

        # Convert data for display
        data_display = data_copy.copy()
        data_display[y_col] = convert_metric_data(data_display[y_col], y_col)

        # Prepare converted data for plotting
        groups_converted = [
            data_display[data_display["combined_group"] == group][y_col].dropna().values for group in unique_groups
        ]

        # Prepare colors
        colors = []
        for group in unique_groups:
            genotype_val = group.split(" + ")[0]
            colors.append(genotype_color_map[genotype_val])

        # Create boxplot with converted data FIRST (drawn below)
        box_plot = sns.boxplot(
            data=data_display,
            x="combined_group",
            y=y_col,
            ax=ax,
            order=unique_groups,
            palette=colors,
            width=0.6,
            zorder=1,  # Lower z-order so scatter appears on top
        )

        # Add scatter with converted data on top
        for i, group in enumerate(unique_groups):
            group_data = groups_converted[i]
            x_pos = np.random.normal(i, 0.04, size=len(group_data))
            ax.scatter(
                x_pos, group_data, alpha=0.6, s=50, color=colors[i], edgecolors="black", linewidths=0.5, zorder=2
            )

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

        # Formatting with elegant metric name
        ax.set_title(title, fontsize=14, fontweight="bold", fontname="Arial")
        ax.set_xlabel("Genotype + Pretraining", fontsize=12, fontname="Arial")
        ax.set_ylabel(format_metric_label(y_col), fontsize=12, fontname="Arial")
        ax.tick_params(axis="x", rotation=45)

        # Apply global axis range if available
        if y_col in self.global_axis_ranges:
            ax.set_ylim(self.global_axis_ranges[y_col])

        # Add sample sizes to x-axis labels
        x_labels = []
        for i, group in enumerate(unique_groups):
            n = len(groups_raw[i])
            genotype, pretraining = group.split(" + ")
            pretraining_label = format_pretraining_label(pretraining)
            display_label = f"{genotype} + {pretraining_label}"
            x_labels.append(f"{display_label}\n(n={n})")
        ax.set_xticklabels(x_labels, fontname="Arial", rotation=45, ha="right")

        # Statistical testing with permutation tests on RAW data
        if len(groups_raw) < 2:
            return ax

        # Perform permutation tests for mean and median
        p_mean, diff_mean = self._permutation_test(groups_raw, lambda x: [np.mean(g) for g in x])
        p_median, diff_median = self._permutation_test(groups_raw, lambda x: [np.median(g) for g in x])

        # For multiple groups, apply FDR correction
        if len(groups_raw) > 2:
            # Pairwise comparisons
            p_values_mean = []
            p_values_median = []

            for i in range(len(groups_raw)):
                for j in range(i + 1, len(groups_raw)):
                    p_m, _ = self._permutation_test([groups_raw[i], groups_raw[j]], lambda x: [np.mean(g) for g in x])
                    p_md, _ = self._permutation_test(
                        [groups_raw[i], groups_raw[j]], lambda x: [np.median(g) for g in x]
                    )
                    p_values_mean.append(p_m)
                    p_values_median.append(p_md)

            # FDR correction
            _, p_mean_corrected = self._fdr_correction(p_values_mean)
            _, p_median_corrected = self._fdr_correction(p_values_median)

            # Use the minimum corrected p-value for display
            p_mean_display = np.min(p_mean_corrected)
            p_median_display = np.min(p_median_corrected)
            test_label = "Permutation (FDR corrected)"
        else:
            p_mean_display = p_mean
            p_median_display = p_median
            test_label = "Permutation test"

        # Bootstrap confidence intervals for mean and median (for 2 groups only, to keep annotation manageable)
        if len(groups_raw) == 2:
            ci_text_lines = []
            for i, group in enumerate(unique_groups):
                group_data_converted = convert_metric_data(groups_raw[i], y_col)
                # Mean CI
                ci_low_mean, ci_high_mean, mean_est = self._bootstrap_ci(group_data_converted, np.mean)
                # Median CI
                ci_low_median, ci_high_median, median_est = self._bootstrap_ci(group_data_converted, np.median)
                ci_text_lines.append(f"{group}:")
                ci_text_lines.append(f"  μ={mean_est:.3f} [{ci_low_mean:.3f}, {ci_high_mean:.3f}]")
                ci_text_lines.append(f"  M={median_est:.3f} [{ci_low_median:.3f}, {ci_high_median:.3f}]")

            # Add bootstrap CI box
            ci_full_text = "Bootstrap 95% CI:\n" + "\n".join(ci_text_lines)
            ax.text(
                0.98,
                0.98,
                ci_full_text,
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )

        # Calculate effect size for 2-group comparison
        if len(groups_raw) == 2:
            cohens_d = self._cohens_d(groups_raw[0], groups_raw[1])
            effect_text = f"Cohen's d = {cohens_d:.3f}"
        else:
            effect_text = ""

        # Create annotation text
        stat_text = f"{test_label}\n"
        stat_text += f"Mean: p = {p_mean_display:.4f}\n"
        stat_text += f"Median: p = {p_median_display:.4f}"
        if effect_text:
            stat_text += f"\n{effect_text}"

        # Add statistical annotation
        ax.text(
            0.02,
            0.98,
            stat_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        return ax

    def _create_control_boxplots_with_pairwise(self, data, cols, metric_col, mode, test_type="mannwhitney"):
        """Create boxplots for control mode with pairwise comparisons."""
        plot_config = self.config["plot_styling"][mode]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot by primary grouping with pairwise stats
        results_primary = self._create_boxplot_with_pairwise(
            data,
            cols["primary"],
            metric_col,
            f"{metric_col.replace('_', ' ').title()} by {cols['primary'].title()}",
            ax1,
            mode,
            "pretraining",
            test_type=test_type,
        )

        # Plot by secondary grouping with pairwise stats
        results_secondary = self._create_boxplot_with_pairwise(
            data,
            cols["secondary"],
            metric_col,
            f"{metric_col.replace('_', ' ').title()} by {cols['secondary'].replace('_', ' ').title()}",
            ax2,
            mode,
            "f1_condition",
            test_type=test_type,
        )

        plt.tight_layout()
        # Add test type to filename
        test_suffix = "_permutation" if test_type == "permutation" else "_mannwhitney"
        self._save_plot(fig, f"{metric_col}_boxplots{test_suffix}", mode)

        return {
            "primary": results_primary,
            "secondary": results_secondary,
        }

    def _create_tnt_boxplots_with_pairwise(self, data, cols, metric_col, mode, test_type="mannwhitney"):
        """Create boxplots for TNT modes with pairwise comparisons."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Combined plot with pairwise stats
        results_combined = self._create_combined_boxplot_with_pairwise(
            data,
            cols["primary"],
            cols["secondary"],
            metric_col,
            f"{metric_col.replace('_', ' ').title()} by {cols['primary'].title()} + {cols['secondary'].title()}",
            ax,
            mode,
            test_type=test_type,
        )

        plt.tight_layout()
        # Add test type to filename
        test_suffix = "_permutation" if test_type == "permutation" else "_mannwhitney"
        self._save_plot(fig, f"{metric_col}_boxplots{test_suffix}", mode)

        return {
            "combined": results_combined,
        }

    def _create_boxplot_with_pairwise(self, data, x_col, y_col, title, ax, mode, var_type, test_type="mannwhitney"):
        """Create boxplot with scatter overlay and pairwise comparisons.

        Args:
            test_type: 'mannwhitney' or 'permutation' - determines which statistical test to use
        """
        # Get ordering and colors
        unique_values = sorted(data[x_col].unique())
        color_mapping = self.create_color_mapping(mode, unique_values, var_type)

        # Prepare data for statistical tests (RAW, unconverted)
        groups_raw = [data[data[x_col] == condition][y_col].dropna().values for condition in unique_values]

        # Convert data for display
        data_display = data.copy()
        data_display[y_col] = convert_metric_data(data_display[y_col], y_col)

        # Prepare converted data for plotting
        groups_converted = [
            data_display[data_display[x_col] == condition][y_col].dropna().values for condition in unique_values
        ]

        # Create boxplot with converted data FIRST (drawn below)
        box_plot = sns.boxplot(
            data=data_display,
            x=x_col,
            y=y_col,
            ax=ax,
            order=unique_values,
            palette=[color_mapping.get(v, "gray") for v in unique_values],
            width=0.6,
            zorder=1,  # Lower z-order so scatter appears on top
        )

        # Add scatter with converted data on top
        for i, condition in enumerate(unique_values):
            condition_data = groups_converted[i]
            x_pos = np.random.normal(i, 0.04, size=len(condition_data))
            ax.scatter(
                x_pos,
                condition_data,
                alpha=0.6,
                s=50,
                color=color_mapping.get(condition, "gray"),
                edgecolors="black",
                linewidths=0.5,
                zorder=2,  # Higher z-order so scatter appears on top
            )

        # Style boxplot (F1 style: transparent boxes, colored edges)
        for patch in box_plot.patches:
            patch.set_facecolor("white")
            patch.set_alpha(0.7)
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_linewidth(2)

        # Style whiskers and medians
        for i, line in enumerate(ax.lines):
            line.set_color("black")
            line.set_linewidth(1.5)

        # Formatting with elegant metric name
        ax.set_title(title, fontsize=14, fontweight="bold", fontname="Arial")
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12, fontname="Arial")
        ax.set_ylabel(format_metric_label(y_col), fontsize=12, fontname="Arial")

        # Apply global axis range if available
        if y_col in self.global_axis_ranges:
            ax.set_ylim(self.global_axis_ranges[y_col])

        # Add sample sizes to x-axis labels
        x_labels = []
        for i, condition in enumerate(unique_values):
            n = len(groups_raw[i])
            # Format pretraining labels if this is pretraining column
            if var_type == "pretraining":
                display_name = format_pretraining_label(condition)
            else:
                display_name = str(condition)
            x_labels.append(f"{display_name}\n(n={n})")
        ax.set_xticklabels(x_labels, fontname="Arial")

        # Statistical testing with pairwise comparisons on RAW data
        group_names = unique_values

        pairwise_results = []

        if len(groups_raw) >= 2:
            # Perform all pairwise comparisons
            p_values = []
            comparisons = []

            for i in range(len(groups_raw)):
                for j in range(i + 1, len(groups_raw)):
                    # Choose statistical test based on test_type
                    if test_type == "permutation":
                        # Permutation test for mean difference
                        p_value, diff_value = self._permutation_test(
                            [groups_raw[i], groups_raw[j]], lambda x: [np.mean(g) for g in x]
                        )
                    else:  # mannwhitney
                        # Mann-Whitney U test
                        from scipy.stats import mannwhitneyu

                        statistic, p_value = mannwhitneyu(groups_raw[i], groups_raw[j], alternative="two-sided")
                        diff_value = np.median(groups_raw[i]) - np.median(groups_raw[j])

                    p_values.append(p_value)
                    comparisons.append((group_names[i], group_names[j]))

                    # Store detailed results (converted for display)
                    pairwise_results.append(
                        {
                            "group1": group_names[i],
                            "group2": group_names[j],
                            "n1": len(groups_raw[i]),
                            "n2": len(groups_raw[j]),
                            "mean1": np.mean(convert_metric_data(groups_raw[i], y_col)),
                            "mean2": np.mean(convert_metric_data(groups_raw[j], y_col)),
                            "median1": np.median(convert_metric_data(groups_raw[i], y_col)),
                            "median2": np.median(convert_metric_data(groups_raw[j], y_col)),
                            "p_value_raw": p_value,
                            "diff_value": diff_value,
                            "test_type": test_type,
                        }
                    )

            # Apply FDR correction
            if len(p_values) > 0:
                _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

                # Update pairwise results with corrected p-values
                for idx, result in enumerate(pairwise_results):
                    result["p_value_fdr"] = p_corrected[idx]
                    result["significant"] = p_corrected[idx] < 0.05

                # Draw significance bars and p-values for ALL comparisons
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                y_max = ax.get_ylim()[1]
                bar_height = y_range * 0.02

                # Track which vertical positions we've used for bars
                bar_y_positions = []

                for idx, (comp_i, comp_j) in enumerate(comparisons):
                    result = pairwise_results[idx]

                    # Find group indices
                    i = group_names.index(comp_i)
                    j = group_names.index(comp_j)

                    # Find a good y position for this bar
                    max_overlap_level = 0
                    for existing_i, existing_j, level in bar_y_positions:
                        # Check if bars would overlap
                        if not (j < existing_i or i > existing_j):
                            max_overlap_level = max(max_overlap_level, level + 1)

                    bar_level = max_overlap_level
                    y_bar = y_max + y_range * (0.05 + bar_level * 0.08)
                    bar_y_positions.append((i, j, bar_level))

                    # Draw bar for all comparisons
                    ax.plot([i, j], [y_bar, y_bar], "k-", linewidth=1.5)
                    ax.plot([i, i], [y_bar - bar_height, y_bar], "k-", linewidth=1.5)
                    ax.plot([j, j], [y_bar - bar_height, y_bar], "k-", linewidth=1.5)

                    # Add p-value text (always show, regardless of significance)
                    p_val = result["p_value_fdr"]
                    if p_val < 0.001:
                        p_text = "p < 0.001"
                    else:
                        p_text = f"p = {p_val:.3f}"

                    # Add stars for significant results
                    if result["significant"]:
                        if p_val < 0.001:
                            stars = "***"
                        elif p_val < 0.01:
                            stars = "**"
                        else:
                            stars = "*"
                        p_text = f"{stars} {p_text}"

                    ax.text(
                        (i + j) / 2,
                        y_bar + bar_height,
                        p_text,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        fontname="Arial",
                    )

                # Adjust y-axis limits to accommodate all bars and labels
                if pairwise_results:
                    max_bar_level = max((level for _, _, level in bar_y_positions), default=0)
                    ax.set_ylim(ax.get_ylim()[0], y_max + y_range * (0.15 + max_bar_level * 0.08))

        return {
            "groups": group_names,
            "n_samples": [len(g) for g in groups_raw],
            "means": [np.mean(convert_metric_data(g, y_col)) for g in groups_raw],
            "medians": [np.median(convert_metric_data(g, y_col)) for g in groups_raw],
            "pairwise": pairwise_results,
        }

    def _create_combined_boxplot_with_pairwise(
        self, data, primary_col, secondary_col, y_col, title, ax, mode, test_type="mannwhitney"
    ):
        """Create combined boxplot for TNT modes with pairwise comparisons."""
        # Create combined grouping
        data_copy = data.copy()
        data_copy["combined_group"] = data_copy[secondary_col].astype(str) + " + " + data_copy[primary_col].astype(str)

        unique_groups = sorted(data_copy["combined_group"].unique())

        # Get colors based on genotype brain regions
        genotype_vals = [group.split(" + ")[0] for group in unique_groups]
        unique_genotypes = list(set(genotype_vals))
        genotype_color_map = self.create_color_mapping(mode, unique_genotypes, "brain_region")

        # Prepare data for statistical tests (RAW, unconverted)
        groups_raw = [data_copy[data_copy["combined_group"] == group][y_col].dropna().values for group in unique_groups]

        # Convert data for display
        data_display = data_copy.copy()
        data_display[y_col] = convert_metric_data(data_display[y_col], y_col)

        # Prepare converted data for plotting
        groups_converted = [
            data_display[data_display["combined_group"] == group][y_col].dropna().values for group in unique_groups
        ]

        # Prepare colors
        colors = []
        for group in unique_groups:
            genotype = group.split(" + ")[0]
            colors.append(genotype_color_map.get(genotype, "gray"))

        # Create boxplot with converted data FIRST (drawn below)
        box_plot = sns.boxplot(
            data=data_display,
            x="combined_group",
            y=y_col,
            ax=ax,
            order=unique_groups,
            palette=colors,
            width=0.6,
            zorder=1,  # Lower z-order so scatter appears on top
        )

        # Add scatter with converted data on top
        for i, group in enumerate(unique_groups):
            group_data = groups_converted[i]
            x_pos = np.random.normal(i, 0.04, size=len(group_data))
            ax.scatter(
                x_pos, group_data, alpha=0.6, s=50, color=colors[i], edgecolors="black", linewidths=0.5, zorder=2
            )

        # Style based on pretraining
        for i, (patch, group) in enumerate(zip(box_plot.patches, unique_groups)):
            pretraining = group.split(" + ")[1]
            if pretraining == "n":
                patch.set_facecolor("white")
                patch.set_edgecolor(colors[i])
                patch.set_linewidth(2)
            else:
                patch.set_facecolor(colors[i])
                patch.set_edgecolor("black")
                patch.set_linewidth(1.5)

        # Style whiskers and medians
        for i, group in enumerate(unique_groups):
            pretraining = group.split(" + ")[1]
            color = colors[i]
            for j in range(i * 6, (i + 1) * 6):
                if j < len(ax.lines):
                    line = ax.lines[j]
                    if pretraining == "n":
                        line.set_color(color)
                        line.set_linewidth(2)
                    else:
                        line.set_color("black")
                        line.set_linewidth(1.5)

        # Formatting with elegant metric name
        ax.set_title(title, fontsize=14, fontweight="bold", fontname="Arial")
        ax.set_xlabel("Genotype + Pretraining", fontsize=12, fontname="Arial")
        ax.set_ylabel(format_metric_label(y_col), fontsize=12, fontname="Arial")
        ax.tick_params(axis="x", rotation=45)

        # Apply global axis range if available
        if y_col in self.global_axis_ranges:
            ax.set_ylim(self.global_axis_ranges[y_col])

        # Add sample sizes to x-axis labels
        x_labels = []
        for i, group in enumerate(unique_groups):
            n = len(groups_raw[i])
            genotype, pretraining = group.split(" + ")
            pretraining_label = format_pretraining_label(pretraining)
            display_label = f"{genotype} + {pretraining_label}"
            x_labels.append(f"{display_label}\n(n={n})")
        ax.set_xticklabels(x_labels, fontname="Arial", rotation=45, ha="right")

        # Statistical testing with SIMPLIFIED pairwise comparisons
        # NEW STRATEGY: Only compare pretraining within each genotype (n vs y)
        # This reduces comparisons from all-pairwise to just 2: ctrl n vs ctrl y, and tnt n vs tnt y
        group_names = unique_groups

        pairwise_results = []

        if len(groups_raw) >= 2:
            # Build a mapping of group indices
            group_info = {}
            for i, group_name in enumerate(group_names):
                genotype = group_name.split(" + ")[0]
                pretraining = group_name.split(" + ")[1]
                group_info[group_name] = {"idx": i, "genotype": genotype, "pretraining": pretraining}

            # Perform SIMPLIFIED pairwise comparisons
            # Only compare n vs y within the SAME genotype
            p_values = []
            comparisons = []

            for i in range(len(groups_raw)):
                for j in range(i + 1, len(groups_raw)):
                    group1_name = group_names[i]
                    group2_name = group_names[j]

                    g1_info = group_info[group1_name]
                    g2_info = group_info[group2_name]

                    # ONLY compare if: Same genotype, different pretraining (n vs y)
                    same_genotype = g1_info["genotype"] == g2_info["genotype"]
                    diff_pretraining = g1_info["pretraining"] != g2_info["pretraining"]

                    if same_genotype and diff_pretraining:
                        # Choose statistical test based on test_type
                        if test_type == "permutation":
                            # Permutation test for mean difference
                            p_value, diff_value = self._permutation_test(
                                [groups_raw[i], groups_raw[j]], lambda x: [np.mean(g) for g in x]
                            )
                        else:  # mannwhitney
                            # Mann-Whitney U test
                            from scipy.stats import mannwhitneyu

                            statistic, p_value = mannwhitneyu(groups_raw[i], groups_raw[j], alternative="two-sided")
                            diff_value = np.median(groups_raw[i]) - np.median(groups_raw[j])

                        p_values.append(p_value)
                        comparisons.append((group_names[i], group_names[j], i, j))

                        # Store detailed results (converted for display)
                        pairwise_results.append(
                            {
                                "group1": group_names[i],
                                "group2": group_names[j],
                                "group1_idx": i,
                                "group2_idx": j,
                                "n1": len(groups_raw[i]),
                                "n2": len(groups_raw[j]),
                                "mean1": np.mean(convert_metric_data(groups_raw[i], y_col)),
                                "mean2": np.mean(convert_metric_data(groups_raw[j], y_col)),
                                "median1": np.median(convert_metric_data(groups_raw[i], y_col)),
                                "median2": np.median(convert_metric_data(groups_raw[j], y_col)),
                                "p_value_raw": p_value,
                                "diff_value": diff_value,
                                "comparison_type": "within_genotype",
                                "test_type": test_type,
                            }
                        )

            # Apply FDR correction (less harsh now with only 2 comparisons!)
            if len(p_values) > 0:
                _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

                # Update pairwise results with corrected p-values
                for idx, result in enumerate(pairwise_results):
                    result["p_value_fdr"] = p_corrected[idx]
                    result["significant"] = p_corrected[idx] < 0.05

                # Draw bars and p-values for ALL comparisons
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                y_max = ax.get_ylim()[1]
                bar_height = y_range * 0.02

                # Track which vertical positions we've used for bars
                bar_y_positions = []

                for idx, result in enumerate(pairwise_results):
                    # Always draw bars and p-values, not just for significant ones
                    i = result["group1_idx"]
                    j = result["group2_idx"]

                    # Find a good y position for this bar
                    max_overlap_level = 0
                    for existing_i, existing_j, level in bar_y_positions:
                        # Check if bars would overlap
                        if not (j < existing_i or i > existing_j):
                            max_overlap_level = max(max_overlap_level, level + 1)

                    bar_level = max_overlap_level
                    y_bar = y_max + y_range * (0.05 + bar_level * 0.08)
                    bar_y_positions.append((i, j, bar_level))

                    # Draw bar for all comparisons
                    ax.plot([i, j], [y_bar, y_bar], "k-", linewidth=1.5)
                    ax.plot([i, i], [y_bar - bar_height, y_bar], "k-", linewidth=1.5)
                    ax.plot([j, j], [y_bar - bar_height, y_bar], "k-", linewidth=1.5)

                    # Add p-value text (always show)
                    p_val = result["p_value_fdr"]
                    if p_val < 0.001:
                        p_text = "p < 0.001"
                    else:
                        p_text = f"p = {p_val:.3f}"
                        fontweight = ("bold",)

                    # Add stars for significant results
                    if result["significant"]:
                        if p_val < 0.001:
                            stars = "***"
                        elif p_val < 0.01:
                            stars = "**"
                        else:
                            stars = "*"
                        p_text = f"{stars} {p_text}"

                    ax.text(
                        (i + j) / 2,
                        y_bar + bar_height,
                        p_text,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        fontname="Arial",
                    )

                # Adjust y-axis limits to accommodate all bars and labels
                if pairwise_results:
                    max_bar_level = max((level for _, _, level in bar_y_positions), default=0)
                    ax.set_ylim(ax.get_ylim()[0], y_max + y_range * (0.15 + max_bar_level * 0.08))

        return {
            "groups": group_names,
            "n_samples": [len(g) for g in groups_raw],
            "means": [np.mean(convert_metric_data(g, y_col)) for g in groups_raw],
            "medians": [np.median(convert_metric_data(g, y_col)) for g in groups_raw],
            "pairwise": pairwise_results,
        }

    def _generate_boxplot_summary_markdown(self, all_results, mode, detected_cols):
        """Generate markdown summary of boxplot pairwise comparisons."""
        # Determine output directory (handle pooled modes)
        base_mode = mode.replace("_pooled", "")
        if self.custom_output_dir:
            output_dir = self.custom_output_dir
        elif base_mode == "control":
            output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New")
        elif base_mode == "tnt_mb247":
            output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/MB247")
        elif base_mode == "tnt_lc10_2":
            output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/LC10-2")
        elif base_mode == "tnt_ddc":
            output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/DDC")
        else:
            output_dir = Path(".")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Add pooled suffix to filename if in pooled mode
        if "_pooled" in mode:
            summary_file = output_dir / f"boxplot_pairwise_summary_{mode}.md"
        else:
            summary_file = output_dir / f"boxplot_pairwise_summary_{mode}.md"

        # Check if this is an append operation
        is_appending = summary_file.exists()

        lines = []

        if not is_appending:
            # Full header for new file
            lines.append(f"# Boxplot Pairwise Comparison Summary - {mode.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"**Analysis Mode:** {mode}")
            lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(
                f"**Note:** This file contains cumulative results. New metrics are appended as they are analyzed."
            )
            lines.append("")
            lines.append("## Overview")
            lines.append("")
            lines.append(f"- **Total metrics analyzed:** {len(all_results)}")
            lines.append(f"- **Statistical method:** Permutation tests with FDR correction (Benjamini-Hochberg)")
            lines.append(f"- **Significance threshold:** α = 0.05 (FDR-corrected)")
            lines.append("")
        else:
            # Just add a separator and timestamp for appended content
            lines.append("")
            lines.append(f"<!-- Metrics added: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} -->")
            lines.append("")

        # Count total significant results
        total_sig_mean = 0
        total_sig_median = 0
        total_comparisons = 0

        for metric_result in all_results:
            for key, results in metric_result["results"].items():
                if "pairwise" in results:
                    for pw in results["pairwise"]:
                        total_comparisons += 1
                        if pw.get("significant_mean", False):
                            total_sig_mean += 1
                        if pw.get("significant_median", False):
                            total_sig_median += 1

        if not is_appending:
            # Add overview statistics for new file
            lines.append(f"- **Total pairwise comparisons:** {total_comparisons}")
            lines.append(
                f"- **Significant (mean):** {total_sig_mean} ({100*total_sig_mean/max(1, total_comparisons):.1f}%)"
            )
            lines.append(
                f"- **Significant (median):** {total_sig_median} ({100*total_sig_median/max(1, total_comparisons):.1f}%)"
            )
            lines.append("")
            lines.append("---")
            lines.append("")

        # Per-metric detailed results
        for metric_result in all_results:
            metric = metric_result["metric"]
            results = metric_result["results"]

            lines.append(f"## {metric.replace('_', ' ').title()}")
            lines.append("")

            for key, result_data in results.items():
                if key == "primary":
                    section_title = f"By {detected_cols['primary'].title()}"
                elif key == "secondary":
                    section_title = f"By {detected_cols['secondary'].replace('_', ' ').title()}"
                else:
                    section_title = "Combined Analysis"

                lines.append(f"### {section_title}")
                lines.append("")

                # Summary statistics
                lines.append("**Group Statistics:**")
                lines.append("")
                lines.append("| Group | N | Mean | Median |")
                lines.append("|-------|---|------|--------|")
                for i, group in enumerate(result_data["groups"]):
                    n = result_data["n_samples"][i]
                    mean = result_data["means"][i]
                    median = result_data["medians"][i]
                    lines.append(f"| {group} | {n} | {mean:.4f} | {median:.4f} |")
                lines.append("")

                # Pairwise comparisons
                if "pairwise" in result_data and len(result_data["pairwise"]) > 0:
                    lines.append("**Pairwise Comparisons (FDR-corrected):**")
                    lines.append("")
                    lines.append("| Comparison | N1 vs N2 | Mean Diff | Median Diff | p-value (FDR) | Significant |")
                    lines.append("|------------|----------|-----------|-------------|---------------|-------------|")

                    for pw in result_data["pairwise"]:
                        group1 = pw["group1"]
                        group2 = pw["group2"]
                        n_comp = f"{pw['n1']} vs {pw['n2']}"
                        mean_diff = pw["mean1"] - pw["mean2"]
                        median_diff = pw["median1"] - pw["median2"]
                        p_val = pw.get("p_value_fdr", pw.get("p_value_raw", 0.0))
                        significant = "✓" if pw.get("significant", False) else ""

                        lines.append(
                            f"| {group1} vs {group2} | {n_comp} | {mean_diff:.4f} | {median_diff:.4f} | {p_val:.4f} | {significant} |"
                        )
                    lines.append("")

                    # Highlight significant comparisons
                    sig_comps = [pw for pw in result_data["pairwise"] if pw.get("significant", False)]

                    if sig_comps:
                        lines.append("**Significant Differences:**")
                        for pw in sig_comps:
                            mean_diff = pw["mean1"] - pw["mean2"]
                            median_diff = pw["median1"] - pw["median2"]
                            direction_mean = "↑" if mean_diff > 0 else "↓"
                            direction_median = "↑" if median_diff > 0 else "↓"
                            p_val = pw.get("p_value_fdr", pw.get("p_value_raw", 0.0))
                            lines.append(
                                f"- {pw['group1']} vs {pw['group2']}: Mean {direction_mean} {abs(mean_diff):.4f}, Median {direction_median} {abs(median_diff):.4f} (p = {p_val:.4f})"
                            )
                        lines.append("")

                lines.append("---")
                lines.append("")

        # Write to file - append if exists, create new otherwise
        if is_appending:
            # Append to existing file
            with open(summary_file, "a") as f:
                f.write("\n".join(lines))
            print(f"\n📄 Markdown summary appended to: {summary_file} ({len(all_results)} new metrics)")
        else:
            # Create new file
            with open(summary_file, "w") as f:
                f.write("\n".join(lines))
            print(f"\n📄 Markdown summary created: {summary_file} ({len(all_results)} metrics)")

    def analyze_coordinates(self, mode):
        """Analyze coordinate data and create trajectory plots."""
        mode_config = self.config["analysis_modes"][mode]
        coordinates_config = mode_config.get("coordinates", {})

        if not coordinates_config:
            print(f"No coordinates configuration found for mode: {mode}")
            return

        # Get dataset paths from config
        coordinates_path = coordinates_config["coordinates_path"]
        summary_path = coordinates_config.get("summary_path")
        alt_coord_paths = coordinates_config.get("alternative_coordinates_paths", [])
        alt_summary_paths = coordinates_config.get("alternative_summary_paths", [])
        plot_params = coordinates_config.get("coordinates_plot_params", {})

        # Check if this is a pooled mode
        is_pooled = "primary_dataset" in mode_config

        # Load coordinates dataset (with pooling support)
        df_coords = None
        used_coords_path = None

        if is_pooled:
            print(f"\n🔄 Loading pooled coordinates for mode: {mode}")

            # Get primary dataset coordinates
            primary_mode = mode_config["primary_dataset"]
            primary_config = self.config["analysis_modes"][primary_mode]
            primary_coords_config = primary_config.get("coordinates", {})

            if not primary_coords_config:
                print(f"  ⚠️  No coordinates config found for primary mode: {primary_mode}")
                return

            primary_coords_path = primary_coords_config["coordinates_path"]

            try:
                df_coords = pd.read_feather(primary_coords_path)
                print(f"  📁 Primary coordinates loaded from {primary_mode}: {df_coords.shape}")
            except Exception as e:
                print(f"  ❌ Error loading primary coordinates: {e}")
                return

            # Get genotypes to pool
            shared_genotypes = mode_config.get("shared_genotypes", [])
            pool_from_modes = mode_config.get("pool_from_modes", [])

            if shared_genotypes and pool_from_modes:
                # Find genotype column
                genotype_col = None
                for col in df_coords.columns:
                    if "genotype" in col.lower():
                        genotype_col = col
                        break

                if genotype_col:
                    pooled_coords = [df_coords]

                    # Load coordinates from other modes
                    for pool_mode in pool_from_modes:
                        if pool_mode not in self.config["analysis_modes"]:
                            continue

                        pool_config = self.config["analysis_modes"][pool_mode]

                        # Skip if also pooled (avoid recursion)
                        if "primary_dataset" in pool_config:
                            continue

                        pool_coords_config = pool_config.get("coordinates", {})
                        if not pool_coords_config:
                            continue

                        pool_coords_path = pool_coords_config.get("coordinates_path")
                        if not pool_coords_path:
                            continue

                        try:
                            df_pool = pd.read_feather(pool_coords_path)

                            # Filter for shared genotypes
                            if genotype_col in df_pool.columns:
                                df_pool_filtered = df_pool[df_pool[genotype_col].isin(shared_genotypes)]

                                if len(df_pool_filtered) > 0:
                                    print(f"  ✓ Added {len(df_pool_filtered)} frames from {pool_mode}")
                                    pooled_coords.append(df_pool_filtered)
                        except Exception as e:
                            print(f"  ⚠️  Could not load coordinates from {pool_mode}: {e}")

                    # Combine all coordinate datasets
                    if len(pooled_coords) > 1:
                        df_coords = pd.concat(pooled_coords, ignore_index=True)
                        print(f"  ✅ Pooled coordinates created: {df_coords.shape} (from {len(pooled_coords)} sources)")

                        # Print genotype distribution
                        if genotype_col in df_coords.columns:
                            print(f"\n  📊 Genotype distribution in pooled coordinates:")
                            genotype_counts = df_coords[genotype_col].value_counts()
                            for genotype, count in genotype_counts.items():
                                print(f"     {genotype}: {count} frames")

            used_coords_path = f"{primary_mode} (pooled)"

        else:
            # Non-pooled mode: load coordinates normally
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
        # Skip filtering for pooled modes since coordinates already contain pooled data
        if df_summary is not None and "fly" in df_coords.columns and "fly" in df_summary.columns and not is_pooled:
            summary_flies = set(df_summary["fly"].unique())
            coords_flies = set(df_coords["fly"].unique())
            common_flies = summary_flies.intersection(coords_flies)

            print(f"📊 Dataset filtering:")
            print(f"   Flies in coordinates: {len(coords_flies)}")
            print(f"   Flies in summary: {len(summary_flies)}")
            print(f"   Common flies: {len(common_flies)}")

            # Filter to common flies
            df_coords = df_coords[df_coords["fly"].isin(common_flies)]
            print(f"   Filtered coordinates shape: {df_coords.shape}")
        elif is_pooled:
            print(f"📊 Pooled mode: using all pooled coordinates without summary filtering")
            print(f"   Total flies in pooled coordinates: {df_coords['fly'].nunique()}")
            print(f"   Total frames: {len(df_coords)}")

        # Detect grouping columns from summary dataset (or use coordinates if no summary)
        detection_df = df_summary if df_summary is not None else df_coords
        detected_cols = self.detect_columns(detection_df, mode)

        # For coordinates analysis, we need Pretraining and Genotype columns
        # Try exact column names first
        if "Pretraining" in df_coords.columns:
            primary_col = "Pretraining"
        elif "primary" in detected_cols:
            primary_col = detected_cols["primary"]
        else:
            print("❌ Could not find Pretraining column in coordinates dataset")
            return

        if "Genotype" in df_coords.columns:
            secondary_col = "Genotype"
        elif "secondary" in detected_cols:
            secondary_col = detected_cols["secondary"]
        else:
            print("❌ Could not find Genotype column in coordinates dataset")
            return

        print(f"📋 Using grouping columns: {primary_col}, {secondary_col}")

        # Check for required coordinate columns
        coord_cols = ["test_ball_euclidean_distance"]  # Only test ball - training ball not needed

        # Try alternative column names
        if "test_ball_euclidean_distance" not in df_coords.columns:
            if "test_ball" in df_coords.columns:
                coord_cols = ["test_ball"]
            else:
                print(f"❌ Could not find test ball distance column")
                print(f"   Available columns: {list(df_coords.columns)}")
                return

        # Get plotting parameters
        normalize_methods = plot_params.get("normalize_methods", ["percentage", "trimmed", "adaptive_trimmed"])
        run_permutation_tests = plot_params.get("run_permutation_tests", True)

        # Create plots for each normalization method
        for method in normalize_methods:
            print(f"\n📈 Creating {method} plots...")

            if method == "percentage":
                self._create_percentage_coordinate_plots(df_coords, primary_col, secondary_col, coord_cols, mode)
            elif method == "trimmed":
                self._create_trimmed_coordinate_plots(df_coords, primary_col, secondary_col, coord_cols, mode)
            elif method == "adaptive_trimmed":
                self._create_adaptive_trimmed_coordinate_plots(df_coords, primary_col, secondary_col, coord_cols, mode)

        # Run permutation tests on percentage-normalized data
        if run_permutation_tests and "percentage" in normalize_methods:
            print(f"\n🔬 Running permutation tests on trajectory data...")

            # Normalize to percentage time
            df_norm = self._normalize_time_to_percentage(df_coords)

            if not df_norm.empty:
                # Run tests for each ball distance metric
                for coord_col in coord_cols:
                    if coord_col not in df_norm.columns:
                        continue

                    # Skip if all NaN
                    if df_norm[coord_col].isna().all():
                        continue

                    print(f"\n📊 Testing: {coord_col}")

                    # Compute permutation tests
                    perm_results = self.compute_trajectory_permutation_tests(
                        data=df_norm,
                        primary_col=primary_col,
                        secondary_col=secondary_col,
                        metric_col=coord_col,
                        n_permutations=10000,
                        alpha=0.05,
                        bin_size=10.0,  # 10% bins (0-10, 10-20, ..., 90-100)
                        progress=True,
                    )

                    # Plot results with significant bins highlighted
                    self.plot_permutation_test_results(
                        data=df_norm,
                        perm_results=perm_results,
                        primary_col=primary_col,
                        secondary_col=secondary_col,
                        metric_col=coord_col,
                        mode=mode,
                        bin_size=10.0,  # Match permutation test bin size
                    )

                    # Save detailed results to CSV
                    self._save_permutation_results_to_csv(perm_results, coord_col, mode, suffix="percentage")

        # Run permutation tests on adaptive_trimmed data
        if run_permutation_tests and "adaptive_trimmed" in normalize_methods:
            print(f"\n🔬 Running permutation tests on adaptive trimmed trajectory data...")

            # Find adaptive maximum time and trim data
            adaptive_max_time = self._find_adaptive_maximum_time(df_coords, min_fly_fraction=0.5)
            df_adaptive = df_coords[df_coords["adjusted_time"] <= adaptive_max_time].copy()

            if not df_adaptive.empty:
                # Determine appropriate bin size based on the time range
                time_range = adaptive_max_time
                # Use approximately 12 bins similar to the percentage plots
                adaptive_bin_size = time_range / 12.0

                print(f"  Time range: 0 to {adaptive_max_time:.2f}s")
                print(f"  Bin size: {adaptive_bin_size:.2f}s (~12 bins)")

                # Run tests for each ball distance metric
                for coord_col in coord_cols:
                    if coord_col not in df_adaptive.columns:
                        continue

                    # Skip if all NaN
                    if df_adaptive[coord_col].isna().all():
                        continue

                    print(f"\n📊 Testing (adaptive trimmed): {coord_col}")

                    # Compute permutation tests with time-based bins
                    perm_results = self.compute_trajectory_permutation_tests_time_bins(
                        data=df_adaptive,
                        primary_col=primary_col,
                        secondary_col=secondary_col,
                        metric_col=coord_col,
                        n_permutations=10000,
                        alpha=0.05,
                        bin_size=adaptive_bin_size,
                        max_time=adaptive_max_time,
                        progress=True,
                    )

                    # Plot results with significant bins highlighted
                    self.plot_permutation_test_results_time_bins(
                        data=df_adaptive,
                        perm_results=perm_results,
                        primary_col=primary_col,
                        secondary_col=secondary_col,
                        metric_col=coord_col,
                        mode=mode,
                        bin_size=adaptive_bin_size,
                        max_time=adaptive_max_time,
                    )

                    # Save detailed results to CSV
                    self._save_permutation_results_to_csv(perm_results, coord_col, mode, suffix="adaptive_trimmed")

        print(f"✅ Coordinate analysis completed for mode: {mode}")

    def _normalize_time_to_percentage(self, df):
        """Normalize adjusted_time to 0-100% for each fly."""
        if "adjusted_time" not in df.columns or "fly" not in df.columns:
            print("⚠️  Cannot normalize - missing 'adjusted_time' or 'fly' columns")
            return df

        print(f"🔄 Normalizing time to percentage...")

        # More efficient approach: normalize in-place without interpolation
        normalized_data = []

        for fly_id in df["fly"].unique():
            fly_data = df[df["fly"] == fly_id].copy()

            # Skip flies with no valid adjusted_time data
            if fly_data["adjusted_time"].isna().all():
                continue

            valid_mask = ~fly_data["adjusted_time"].isna()
            if not valid_mask.any():
                continue

            valid_times = fly_data.loc[valid_mask, "adjusted_time"]
            min_time = valid_times.min()
            max_time = valid_times.max()

            if max_time <= min_time:
                continue

            # Normalize to 0-100% in-place (no interpolation to grid)
            fly_data.loc[:, "adjusted_time"] = ((fly_data["adjusted_time"] - min_time) / (max_time - min_time)) * 100

            normalized_data.append(fly_data)

        if not normalized_data:
            print("⚠️  No flies could be normalized")
            return df

        result_df = pd.concat(normalized_data, ignore_index=True)
        print(f"✅ Normalized {len(df['fly'].unique())} flies to 0-100% (original data points preserved)")
        return result_df

    def _find_maximum_shared_time(self, df):
        """Find maximum time that all flies share."""
        if "fly" not in df.columns or "adjusted_time" not in df.columns:
            return df["adjusted_time"].max() if "adjusted_time" in df.columns else 30

        max_times_per_fly = df.groupby("fly")["adjusted_time"].max()
        max_shared_time = max_times_per_fly.min()

        print(f"📊 Maximum shared time: {max_shared_time:.2f}s")
        print(f"   Flies with longer data: {(max_times_per_fly > max_shared_time).sum()}")

        return max_shared_time

    def _find_adaptive_maximum_time(self, df, min_fly_fraction=0.5):
        """Find maximum time where at least min_fly_fraction of flies still have data.

        This is less restrictive than _find_maximum_shared_time which requires ALL flies.
        It finds the largest time point where you still have enough flies for proper statistics.

        Args:
            df: DataFrame with fly and adjusted_time columns
            min_fly_fraction: Minimum fraction of total flies required (default 0.5 = 50%)

        Returns:
            Maximum time where at least min_fly_fraction flies have data
        """
        if "fly" not in df.columns or "adjusted_time" not in df.columns:
            return df["adjusted_time"].max() if "adjusted_time" in df.columns else 30

        total_flies = df["fly"].nunique()
        min_flies_required = max(3, int(total_flies * min_fly_fraction))  # At least 3 flies

        # Get max time for each fly
        max_times_per_fly = df.groupby("fly")["adjusted_time"].max().sort_values()

        # Find the time at which we still have min_flies_required
        # This is the max_time of the (total_flies - min_flies_required)th fly
        # (counting from the end, so we keep the min_flies_required flies with longest data)
        if len(max_times_per_fly) >= min_flies_required:
            # Take the time of the fly at position where we still have min_flies_required flies after it
            adaptive_max_time = max_times_per_fly.iloc[-(min_flies_required)]
        else:
            # If we don't have enough flies total, just use the minimum
            adaptive_max_time = max_times_per_fly.min()

        n_flies_retained = (max_times_per_fly >= adaptive_max_time).sum()
        percent_retained = (n_flies_retained / total_flies) * 100

        print(f"📊 Adaptive maximum time: {adaptive_max_time:.2f}s")
        print(f"   Retains {n_flies_retained}/{total_flies} flies ({percent_retained:.1f}%)")
        print(f"   (minimum required: {min_flies_required} = {min_fly_fraction*100:.0f}% of total)")

        return adaptive_max_time

    def _create_percentage_coordinate_plots(self, df, primary_col, secondary_col, coord_cols, mode):
        """Create percentage-normalized coordinate plots."""
        # Normalize time to percentage
        df_norm = self._normalize_time_to_percentage(df)

        if df_norm.empty:
            print("⚠️  No data after percentage normalization")
            return

        # Get plot configuration
        mode_config = self.config["analysis_modes"][mode]
        coordinates_config = mode_config.get("coordinates", {})
        plot_params = coordinates_config.get("coordinates_plot_params", {})
        create_individual_plots = plot_params.get("create_individual_fly_plots", False)

        # Create averaged plots grouped by primary variable (pretraining)
        self._create_coordinate_line_plots(
            df_norm,
            "adjusted_time",
            coord_cols,
            primary_col,
            f"F1 Coordinates by {primary_col.title()} (Percentage Time)",
            "f1_coordinates_percentage_time_pretraining.png",
            mode,
            xlabel="Adjusted Time (%)",
            time_type="percentage",
            var_type="pretraining",  # Pass the variable type for correct colors
        )

        # Create averaged plots grouped by secondary variable (F1_condition or Genotype)
        var_type_secondary = "genotype" if mode.startswith("tnt_") else "f1_condition"
        self._create_coordinate_line_plots(
            df_norm,
            "adjusted_time",
            coord_cols,
            secondary_col,
            f"F1 Coordinates by {secondary_col.replace('_', ' ').title()} (Percentage Time)",
            "f1_coordinates_percentage_time_f1condition.png",
            mode,
            xlabel="Adjusted Time (%)",
            time_type="percentage",
            var_type=var_type_secondary,  # Pass the correct variable type
        )

        # For TNT modes, create combined pretraining + genotype plots
        if mode.startswith("tnt_"):
            print("📊 Creating combined pretraining + genotype plots (percentage)...")
            self._create_combined_coordinate_plots(
                df_norm,
                "adjusted_time",
                coord_cols,
                primary_col,
                secondary_col,
                f"F1 Coordinates by Pretraining + Genotype (Percentage Time)",
                "f1_coordinates_percentage_time_combined.png",
                mode,
                xlabel="Adjusted Time (%)",
                time_type="percentage",
            )

        # Create individual fly plots only if enabled in config
        if create_individual_plots:
            print("📊 Creating individual fly plots (percentage)...")
            # Create individual fly plots grouped by primary variable
            self._create_individual_fly_plots(
                df_norm,
                "adjusted_time",
                coord_cols,
                primary_col,
                f"F1 Coordinates by {primary_col.title()} - Individual Flies (Percentage Time)",
                "f1_coordinates_percentage_time_pretraining_individual.png",
                mode,
                xlabel="Adjusted Time (%)",
                time_type="percentage",
                var_type="pretraining",
            )

            # Create individual fly plots grouped by secondary variable
            self._create_individual_fly_plots(
                df_norm,
                "adjusted_time",
                coord_cols,
                secondary_col,
                f"F1 Coordinates by {secondary_col.replace('_', ' ').title()} - Individual Flies (Percentage Time)",
                "f1_coordinates_percentage_time_f1condition_individual.png",
                mode,
                xlabel="Adjusted Time (%)",
                time_type="percentage",
                var_type=var_type_secondary,
            )
        else:
            print("ℹ️  Skipping individual fly plots (disabled in config)")

    def _create_trimmed_coordinate_plots(self, df, primary_col, secondary_col, coord_cols, mode):
        """Create trimmed-time coordinate plots."""
        # Find maximum shared time and trim data
        max_shared_time = self._find_maximum_shared_time(df)
        df_trimmed = df[df["adjusted_time"] <= max_shared_time].copy()

        print(f"📊 Trimmed data: {df_trimmed.shape[0]} points from {df_trimmed['fly'].nunique()} flies")

        # Get plot configuration
        mode_config = self.config["analysis_modes"][mode]
        coordinates_config = mode_config.get("coordinates", {})
        plot_params = coordinates_config.get("coordinates_plot_params", {})
        create_individual_plots = plot_params.get("create_individual_fly_plots", False)

        # Create averaged plots grouped by primary variable
        self._create_coordinate_line_plots(
            df_trimmed,
            "adjusted_time",
            coord_cols,
            primary_col,
            f"F1 Coordinates by {primary_col.title()} (Trimmed Time)",
            "f1_coordinates_trimmed_time_pretraining.png",
            mode,
            xlabel="Adjusted Time (seconds)",
            time_type="trimmed",
            var_type="pretraining",
        )

        # Create averaged plots grouped by secondary variable
        var_type_secondary = "genotype" if mode.startswith("tnt_") else "f1_condition"
        self._create_coordinate_line_plots(
            df_trimmed,
            "adjusted_time",
            coord_cols,
            secondary_col,
            f"F1 Coordinates by {secondary_col.replace('_', ' ').title()} (Trimmed Time)",
            "f1_coordinates_trimmed_time_f1condition.png",
            mode,
            xlabel="Adjusted Time (seconds)",
            time_type="trimmed",
            var_type=var_type_secondary,
        )

        # For TNT modes, create combined pretraining + genotype plots
        if mode.startswith("tnt_"):
            print("📊 Creating combined pretraining + genotype plots (trimmed)...")
            self._create_combined_coordinate_plots(
                df_trimmed,
                "adjusted_time",
                coord_cols,
                primary_col,
                secondary_col,
                f"F1 Coordinates by Pretraining + Genotype (Trimmed Time)",
                "f1_coordinates_trimmed_time_combined.png",
                mode,
                xlabel="Adjusted Time (seconds)",
                time_type="trimmed",
            )

        # Create individual fly plots only if enabled in config
        if create_individual_plots:
            print("📊 Creating individual fly plots (trimmed)...")
            # Create individual fly plots grouped by primary variable
            self._create_individual_fly_plots(
                df_trimmed,
                "adjusted_time",
                coord_cols,
                primary_col,
                f"F1 Coordinates by {primary_col.title()} - Individual Flies (Trimmed Time)",
                "f1_coordinates_trimmed_time_pretraining_individual.png",
                mode,
                xlabel="Adjusted Time (seconds)",
                time_type="trimmed",
                var_type="pretraining",
            )

            # Create individual fly plots grouped by secondary variable
            self._create_individual_fly_plots(
                df_trimmed,
                "adjusted_time",
                coord_cols,
                secondary_col,
                f"F1 Coordinates by {secondary_col.replace('_', ' ').title()} - Individual Flies (Trimmed Time)",
                "f1_coordinates_trimmed_time_f1condition_individual.png",
                mode,
                xlabel="Adjusted Time (seconds)",
                time_type="trimmed",
                var_type=var_type_secondary,
            )
        else:
            print("ℹ️  Skipping individual fly plots (disabled in config)")

    def _create_adaptive_trimmed_coordinate_plots(self, df, primary_col, secondary_col, coord_cols, mode):
        """Create adaptive-trimmed coordinate plots (keeps 50%+ flies with sufficient time coverage)."""
        # Find adaptive maximum time (keeps at least 50% of flies)
        adaptive_max_time = self._find_adaptive_maximum_time(df, min_fly_fraction=0.5)
        df_adaptive = df[df["adjusted_time"] <= adaptive_max_time].copy()

        print(f"📊 Adaptive trimmed data: {df_adaptive.shape[0]} points from {df_adaptive['fly'].nunique()} flies")

        # Get plot configuration
        mode_config = self.config["analysis_modes"][mode]
        coordinates_config = mode_config.get("coordinates", {})
        plot_params = coordinates_config.get("coordinates_plot_params", {})
        create_individual_plots = plot_params.get("create_individual_fly_plots", False)

        # Create averaged plots grouped by primary variable
        self._create_coordinate_line_plots(
            df_adaptive,
            "adjusted_time",
            coord_cols,
            primary_col,
            f"F1 Coordinates by {primary_col.title()} (Adaptive Trimmed - 50%+ Flies)",
            "f1_coordinates_adaptive_trimmed_pretraining.png",
            mode,
            xlabel="Adjusted Time (seconds)",
            time_type="adaptive_trimmed",
            var_type="pretraining",
        )

        # Create averaged plots grouped by secondary variable
        var_type_secondary = "genotype" if mode.startswith("tnt_") else "f1_condition"
        self._create_coordinate_line_plots(
            df_adaptive,
            "adjusted_time",
            coord_cols,
            secondary_col,
            f"F1 Coordinates by {secondary_col.replace('_', ' ').title()} (Adaptive Trimmed - 50%+ Flies)",
            "f1_coordinates_adaptive_trimmed_f1condition.png",
            mode,
            xlabel="Adjusted Time (seconds)",
            time_type="adaptive_trimmed",
            var_type=var_type_secondary,
        )

        # For TNT modes, create combined pretraining + genotype plots
        if mode.startswith("tnt_"):
            print("📊 Creating combined pretraining + genotype plots (adaptive trimmed)...")
            self._create_combined_coordinate_plots(
                df_adaptive,
                "adjusted_time",
                coord_cols,
                primary_col,
                secondary_col,
                f"F1 Coordinates by Pretraining + Genotype (Adaptive Trimmed - 50%+ Flies)",
                "f1_coordinates_adaptive_trimmed_combined.png",
                mode,
                xlabel="Adjusted Time (seconds)",
                time_type="adaptive_trimmed",
            )

        # Create individual fly plots only if enabled in config
        if create_individual_plots:
            print("📊 Creating individual fly plots (adaptive trimmed)...")
            self._create_individual_fly_plots(
                df_adaptive,
                "adjusted_time",
                coord_cols,
                primary_col,
                f"F1 Individual Fly Coordinates by {primary_col.title()} (Adaptive Trimmed)",
                "f1_coordinates_adaptive_trimmed_individual_flies_pretraining.png",
                mode,
                xlabel="Adjusted Time (seconds)",
                time_type="adaptive_trimmed",
                var_type="pretraining",
            )

            self._create_individual_fly_plots(
                df_adaptive,
                "adjusted_time",
                coord_cols,
                secondary_col,
                f"F1 Individual Fly Coordinates by {secondary_col.replace('_', ' ').title()} (Adaptive Trimmed)",
                "f1_coordinates_adaptive_trimmed_individual_flies_f1condition.png",
                mode,
                xlabel="Adjusted Time (seconds)",
                time_type="adaptive_trimmed",
                var_type=var_type_secondary,
            )
        else:
            print("ℹ️  Skipping individual fly plots (disabled in config)")

    def compute_trajectory_permutation_tests(
        self,
        data,
        primary_col,
        secondary_col,
        metric_col,
        n_permutations=10000,
        alpha=0.05,
        bin_size=10.0,  # Changed default to 10% bins
        progress=True,
    ):
        """
        Compute pairwise permutation tests for all condition combinations across time bins.

        This tests differences between all 4 conditions (2 pretraining × 2 genotypes) at each
        time bin, with FDR correction for multiple comparisons.

        Parameters
        ----------
        data : pd.DataFrame
            Normalized percentage time data with fly-level information
        primary_col : str
            Primary grouping column (e.g., 'Pretraining')
        secondary_col : str
            Secondary grouping column (e.g., 'Genotype')
        metric_col : str
            Column containing the metric to test (e.g., 'test_ball_euclidean_distance')
        n_permutations : int
            Number of permutations for each test
        alpha : float
            Significance level for FDR correction
        bin_size : float
            Size of time bins in percentage (default 10.0 gives 10 bins across 0-100%)
        progress : bool
            Show progress bars

        Returns
        -------
        dict
            Dictionary with test results for each pairwise comparison
        """
        print(f"\n{'='*80}")
        print(f"PERMUTATION TESTS: {metric_col}")
        print(f"{'='*80}")
        print(f"Settings:")
        print(f"  • Permutations: {n_permutations}")
        print(f"  • FDR alpha: {alpha}")
        print(f"  • Time bin size: {bin_size}%")

        # Create combined grouping variable
        data_copy = data.copy()
        data_copy["combined_group"] = data_copy[primary_col].astype(str) + "_" + data_copy[secondary_col].astype(str)

        # Get all unique groups
        groups = sorted(data_copy["combined_group"].unique())
        print(f"\nGroups: {groups}")

        # Create bins from adjusted_time (0-100%)
        # Use bin edges: 0-10, 10-20, 20-30, ..., 90-100
        bin_edges = np.arange(0, 100 + bin_size, bin_size)
        data_copy["time_bin"] = pd.cut(
            data_copy["adjusted_time"],
            bins=bin_edges,
            labels=bin_edges[:-1],  # Use left edge as label
            include_lowest=True,
        )

        # Convert categorical to numeric for easier handling
        data_copy["time_bin"] = pd.to_numeric(data_copy["time_bin"])

        # Get fly-level medians for each time bin (to handle repeated measures)
        fly_medians = data_copy.groupby(["fly", "combined_group", "time_bin"])[metric_col].median().reset_index()

        time_bins = sorted(fly_medians["time_bin"].unique())
        n_bins = len(time_bins)
        print(f"Time bins: {n_bins} bins from {time_bins[0]:.1f}% to {time_bins[-1]:.1f}%")

        # Get all pairwise combinations
        comparisons = list(combinations(groups, 2))
        print(f"\nPairwise comparisons: {len(comparisons)}")

        results = {}

        # For each pairwise comparison
        for group1, group2 in comparisons:
            comparison_name = f"{group1} vs {group2}"
            print(f"\n{'─'*80}")
            print(f"Testing: {comparison_name}")

            # Filter data for this comparison
            comparison_data = fly_medians[fly_medians["combined_group"].isin([group1, group2])]

            # Store results for each bin
            observed_diffs = []
            p_values_raw = []
            n_group1_per_bin = []
            n_group2_per_bin = []

            iterator = tqdm(time_bins, desc=f"  {comparison_name}") if progress else time_bins

            for time_bin in iterator:
                bin_data = comparison_data[comparison_data["time_bin"] == time_bin]

                # Get values for each group
                group1_vals = bin_data[bin_data["combined_group"] == group1][metric_col].values
                group2_vals = bin_data[bin_data["combined_group"] == group2][metric_col].values

                n_group1_per_bin.append(len(group1_vals))
                n_group2_per_bin.append(len(group2_vals))

                if len(group1_vals) == 0 or len(group2_vals) == 0:
                    observed_diffs.append(np.nan)
                    p_values_raw.append(1.0)
                    continue

                # Observed difference (group2 - group1)
                obs_diff = np.mean(group2_vals) - np.mean(group1_vals)
                observed_diffs.append(obs_diff)

                # Permutation test
                combined = np.concatenate([group1_vals, group2_vals])
                n1 = len(group1_vals)
                n2 = len(group2_vals)

                perm_diffs = []
                for _ in range(n_permutations):
                    # Shuffle and split
                    shuffled = np.random.permutation(combined)
                    perm_g1 = shuffled[:n1]
                    perm_g2 = shuffled[n1:]
                    perm_diff = np.mean(perm_g2) - np.mean(perm_g1)
                    perm_diffs.append(perm_diff)

                perm_diffs = np.array(perm_diffs)

                # Two-tailed p-value
                p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
                p_values_raw.append(p_value)

            # Apply FDR correction across all time bins for this comparison
            p_values_raw = np.array(p_values_raw)
            valid_mask = ~np.isnan(p_values_raw)

            p_values_corrected = np.ones_like(p_values_raw)
            if valid_mask.sum() > 0:
                rejected, p_corrected, _, _ = multipletests(p_values_raw[valid_mask], alpha=alpha, method="fdr_bh")
                p_values_corrected[valid_mask] = p_corrected
                significant_mask = np.zeros(len(p_values_raw), dtype=bool)
                significant_mask[valid_mask] = rejected
            else:
                significant_mask = np.zeros(len(p_values_raw), dtype=bool)

            # Store results
            results[comparison_name] = {
                "group1": group1,
                "group2": group2,
                "time_bins": time_bins,
                "observed_diffs": observed_diffs,
                "p_values_raw": p_values_raw,
                "p_values_corrected": p_values_corrected,
                "significant_timepoints": np.where(significant_mask)[0],
                "n_significant": np.sum(significant_mask),
                "n_significant_raw": np.sum(p_values_raw < alpha),
                "n_group1_per_bin": n_group1_per_bin,
                "n_group2_per_bin": n_group2_per_bin,
            }

            print(f"  Raw significant bins: {results[comparison_name]['n_significant_raw']}/{n_bins}")
            print(f"  FDR significant bins: {results[comparison_name]['n_significant']}/{n_bins} (α={alpha})")

        print(f"\n{'='*80}")
        print(f"✅ Permutation tests completed")
        print(f"{'='*80}\n")

        return results

    def compute_trajectory_permutation_tests_time_bins(
        self,
        data,
        primary_col,
        secondary_col,
        metric_col,
        n_permutations=10000,
        alpha=0.05,
        bin_size=2.5,
        max_time=30.0,
        progress=True,
    ):
        """
        Compute pairwise permutation tests using time-based bins (for adaptive_trimmed data).

        Similar to compute_trajectory_permutation_tests but uses actual time bins instead of percentage bins.

        Parameters
        ----------
        data : pd.DataFrame
            Time-trimmed data with fly-level information
        primary_col : str
            Primary grouping column (e.g., 'Pretraining')
        secondary_col : str
            Secondary grouping column (e.g., 'Genotype')
        metric_col : str
            Column containing the metric to test
        n_permutations : int
            Number of permutations for each test
        alpha : float
            Significance level for FDR correction
        bin_size : float
            Size of time bins in seconds (e.g., 2.5s)
        max_time : float
            Maximum time to consider
        progress : bool
            Show progress bars

        Returns
        -------
        dict
            Dictionary with test results for each pairwise comparison
        """
        print(f"\n{'='*80}")
        print(f"PERMUTATION TESTS (Time-based): {metric_col}")
        print(f"{'='*80}")
        print(f"Settings:")
        print(f"  • Permutations: {n_permutations}")
        print(f"  • FDR alpha: {alpha}")
        print(f"  • Time bin size: {bin_size:.2f}s")
        print(f"  • Max time: {max_time:.2f}s")

        # Create combined grouping variable
        data_copy = data.copy()
        data_copy["combined_group"] = data_copy[primary_col].astype(str) + "_" + data_copy[secondary_col].astype(str)

        # Get all unique groups
        groups = sorted(data_copy["combined_group"].unique())
        print(f"\nGroups: {groups}")

        # Create bins from adjusted_time
        bin_edges = np.arange(0, max_time + bin_size, bin_size)
        data_copy["time_bin"] = pd.cut(
            data_copy["adjusted_time"],
            bins=bin_edges,
            labels=bin_edges[:-1],
            include_lowest=True,
        )

        # Convert categorical to numeric for easier handling
        data_copy["time_bin"] = pd.to_numeric(data_copy["time_bin"])

        # Get fly-level medians for each time bin (to handle repeated measures)
        fly_medians = data_copy.groupby(["fly", "combined_group", "time_bin"])[metric_col].median().reset_index()

        time_bins = sorted(fly_medians["time_bin"].unique())
        n_bins = len(time_bins)
        print(f"Time bins: {n_bins} bins from {time_bins[0]:.1f}s to {time_bins[-1]:.1f}s")

        # Get all pairwise combinations
        comparisons = list(combinations(groups, 2))
        print(f"\nPairwise comparisons: {len(comparisons)}")

        results = {}

        # For each pairwise comparison
        for group1, group2 in comparisons:
            comparison_name = f"{group1} vs {group2}"
            print(f"\n{'─'*80}")
            print(f"Testing: {comparison_name}")

            # Filter data for this comparison
            comparison_data = fly_medians[fly_medians["combined_group"].isin([group1, group2])]

            # Store results for each bin
            observed_diffs = []
            p_values_raw = []
            n_group1_per_bin = []
            n_group2_per_bin = []

            iterator = tqdm(time_bins, desc=f"  {comparison_name}") if progress else time_bins

            for time_bin in iterator:
                bin_data = comparison_data[comparison_data["time_bin"] == time_bin]

                # Get values for each group
                group1_vals = bin_data[bin_data["combined_group"] == group1][metric_col].values
                group2_vals = bin_data[bin_data["combined_group"] == group2][metric_col].values

                n_group1_per_bin.append(len(group1_vals))
                n_group2_per_bin.append(len(group2_vals))

                if len(group1_vals) == 0 or len(group2_vals) == 0:
                    observed_diffs.append(np.nan)
                    p_values_raw.append(1.0)
                    continue

                # Observed difference (group2 - group1)
                obs_diff = np.mean(group2_vals) - np.mean(group1_vals)
                observed_diffs.append(obs_diff)

                # Permutation test
                combined = np.concatenate([group1_vals, group2_vals])
                n1 = len(group1_vals)
                n2 = len(group2_vals)

                perm_diffs = []
                for _ in range(n_permutations):
                    # Shuffle and split
                    shuffled = np.random.permutation(combined)
                    perm_g1 = shuffled[:n1]
                    perm_g2 = shuffled[n1:]
                    perm_diff = np.mean(perm_g2) - np.mean(perm_g1)
                    perm_diffs.append(perm_diff)

                perm_diffs = np.array(perm_diffs)

                # Two-tailed p-value
                p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
                p_values_raw.append(p_value)

            # Apply FDR correction across all time bins for this comparison
            p_values_raw = np.array(p_values_raw)
            valid_mask = ~np.isnan(p_values_raw)

            p_values_corrected = np.ones_like(p_values_raw)
            if valid_mask.sum() > 0:
                rejected, p_corrected, _, _ = multipletests(p_values_raw[valid_mask], alpha=alpha, method="fdr_bh")
                p_values_corrected[valid_mask] = p_corrected
                significant_mask = np.zeros(len(p_values_raw), dtype=bool)
                significant_mask[valid_mask] = rejected
            else:
                significant_mask = np.zeros(len(p_values_raw), dtype=bool)

            # Store results
            results[comparison_name] = {
                "group1": group1,
                "group2": group2,
                "time_bins": time_bins,
                "observed_diffs": observed_diffs,
                "p_values_raw": p_values_raw,
                "p_values_corrected": p_values_corrected,
                "significant_timepoints": np.where(significant_mask)[0],
                "n_significant": np.sum(significant_mask),
                "n_significant_raw": np.sum(p_values_raw < alpha),
                "n_group1_per_bin": n_group1_per_bin,
                "n_group2_per_bin": n_group2_per_bin,
            }

            print(f"  Raw significant bins: {results[comparison_name]['n_significant_raw']}/{n_bins}")
            print(f"  FDR significant bins: {results[comparison_name]['n_significant']}/{n_bins} (α={alpha})")

        print(f"\n{'='*80}")
        print(f"✅ Permutation tests completed")
        print(f"{'='*80}\n")

        return results

    def plot_permutation_test_results(
        self,
        data,
        perm_results,
        primary_col,
        secondary_col,
        metric_col,
        mode,
        bin_size=10.0,  # Changed default to match compute function
    ):
        """
        Plot trajectory with permutation test results highlighting significant time bins.

        Parameters
        ----------
        data : pd.DataFrame
            Normalized percentage time data
        perm_results : dict
            Results from compute_trajectory_permutation_tests
        primary_col : str
            Primary grouping column
        secondary_col : str
            Secondary grouping column
        metric_col : str
            Metric column name
        mode : str
            Analysis mode
        bin_size : float
            Time bin size (should match what was used in permutation tests)
        """
        # Create combined grouping
        data_copy = data.copy()
        data_copy["combined_group"] = data_copy[primary_col].astype(str) + "_" + data_copy[secondary_col].astype(str)

        groups = sorted(data_copy["combined_group"].unique())

        # Get color mapping
        genotype_vals = [g.split("_")[1] for g in groups]
        unique_genotypes = list(set(genotype_vals))
        genotype_color_map = self.create_color_mapping(mode, unique_genotypes, "brain_region")

        # Create figure with subplots for each pairwise comparison
        n_comparisons = len(perm_results)
        n_cols = 2
        n_rows = int(np.ceil(n_comparisons / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
        if n_comparisons == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (comparison_name, result) in enumerate(perm_results.items()):
            ax = axes[idx]

            group1 = result["group1"]
            group2 = result["group2"]

            # Plot trajectories for these two groups
            for group in [group1, group2]:
                subset = data_copy[data_copy["combined_group"] == group]

                # Bin the data to match the combined plots (8.33% bins for ~12 bins across 0-100%)
                subset = subset.copy()
                subset["time_bin"] = np.floor(subset["adjusted_time"] / 8.33) * 8.33

                # Get fly-level medians per time bin
                fly_medians = subset.groupby(["fly", "time_bin"])[metric_col].median().reset_index()

                # Calculate statistics across flies at each time bin
                time_stats = fly_medians.groupby("time_bin")[metric_col].agg(["mean", "sem", "count"]).reset_index()

                # Get color
                pretraining_val, genotype_val = group.split("_")
                color = genotype_color_map.get(genotype_val, "black")
                linestyle = "-" if pretraining_val == "y" else "--"

                # Plot
                ax.plot(
                    time_stats["time_bin"],
                    time_stats["mean"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    alpha=0.9,
                    label=group.replace("_", " + "),
                )

                # Add CI
                ci = stats.t.ppf(0.975, time_stats["count"] - 1) * time_stats["sem"]
                ax.fill_between(
                    time_stats["time_bin"],
                    time_stats["mean"] - ci,
                    time_stats["mean"] + ci,
                    color=color,
                    alpha=0.15,
                )

            # Highlight significant time bins
            time_bins = result["time_bins"]
            significant_bins = result["significant_timepoints"]

            if len(significant_bins) > 0:
                y_min, y_max = ax.get_ylim()
                for sig_idx in significant_bins:
                    t_bin = time_bins[sig_idx]
                    # Shade the entire bin width
                    ax.axvspan(t_bin, t_bin + bin_size, alpha=0.2, color="red", zorder=0)

            # Formatting
            ax.set_title(
                f"{comparison_name}\n{result['n_significant']} significant bins (FDR α=0.05)",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Time (%)", fontsize=10)
            ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=10)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.7)

        # Hide unused subplots
        for idx in range(n_comparisons, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f'Trajectory Permutation Tests: {metric_col.replace("_", " ").title()}\n'
            f"(Red shading = FDR-corrected significant difference)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save
        self._save_plot(fig, f"{metric_col}_permutation_tests", mode)

    def plot_permutation_test_results_time_bins(
        self,
        data,
        perm_results,
        primary_col,
        secondary_col,
        metric_col,
        mode,
        bin_size=2.5,
        max_time=30.0,
    ):
        """
        Plot trajectory with permutation test results for time-based bins (adaptive_trimmed).

        Parameters
        ----------
        data : pd.DataFrame
            Time-trimmed data
        perm_results : dict
            Results from compute_trajectory_permutation_tests_time_bins
        primary_col : str
            Primary grouping column
        secondary_col : str
            Secondary grouping column
        metric_col : str
            Metric column name
        mode : str
            Analysis mode
        bin_size : float
            Time bin size in seconds
        max_time : float
            Maximum time value
        """
        # Create combined grouping
        data_copy = data.copy()
        data_copy["combined_group"] = data_copy[primary_col].astype(str) + "_" + data_copy[secondary_col].astype(str)

        groups = sorted(data_copy["combined_group"].unique())

        # Get color mapping
        genotype_vals = [g.split("_")[1] for g in groups]
        unique_genotypes = list(set(genotype_vals))
        genotype_color_map = self.create_color_mapping(mode, unique_genotypes, "brain_region")

        # Create figure with subplots for each pairwise comparison
        n_comparisons = len(perm_results)
        n_cols = 2
        n_rows = int(np.ceil(n_comparisons / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
        if n_comparisons == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (comparison_name, result) in enumerate(perm_results.items()):
            ax = axes[idx]

            group1 = result["group1"]
            group2 = result["group2"]

            # Plot trajectories for these two groups
            for group in [group1, group2]:
                subset = data_copy[data_copy["combined_group"] == group]

                # Bin the data (use same bin_size as permutation tests)
                subset = subset.copy()
                subset["time_bin"] = np.floor(subset["adjusted_time"] / bin_size) * bin_size

                # Get fly-level medians per time bin
                fly_medians = subset.groupby(["fly", "time_bin"])[metric_col].median().reset_index()

                # Calculate statistics across flies at each time bin
                time_stats = fly_medians.groupby("time_bin")[metric_col].agg(["mean", "sem", "count"]).reset_index()

                # Get color
                pretraining_val, genotype_val = group.split("_")
                color = genotype_color_map.get(genotype_val, "black")
                linestyle = "-" if pretraining_val == "y" else "--"

                # Plot
                ax.plot(
                    time_stats["time_bin"],
                    time_stats["mean"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    alpha=0.9,
                    label=group.replace("_", " + "),
                )

                # Add CI
                ci = stats.t.ppf(0.975, time_stats["count"] - 1) * time_stats["sem"]
                ax.fill_between(
                    time_stats["time_bin"],
                    time_stats["mean"] - ci,
                    time_stats["mean"] + ci,
                    color=color,
                    alpha=0.15,
                )

            # Highlight significant time bins
            time_bins = result["time_bins"]
            significant_bins = result["significant_timepoints"]

            if len(significant_bins) > 0:
                y_min, y_max = ax.get_ylim()
                for sig_idx in significant_bins:
                    t_bin = time_bins[sig_idx]
                    # Shade the entire bin width
                    ax.axvspan(t_bin, t_bin + bin_size, alpha=0.2, color="red", zorder=0)

            # Formatting
            ax.set_title(
                f"{comparison_name}\n{result['n_significant']} significant bins (FDR α=0.05)",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=10)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.7)

        # Hide unused subplots
        for idx in range(n_comparisons, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f'Trajectory Permutation Tests (Adaptive Trimmed): {metric_col.replace("_", " ").title()}\n'
            f"(Red shading = FDR-corrected significant difference)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save
        self._save_plot(fig, f"{metric_col}_permutation_tests_adaptive_trimmed", mode)

    def _save_permutation_results_to_csv(self, perm_results, metric_col, mode, suffix=""):
        """Save permutation test results to CSV file.

        Args:
            perm_results: Results dictionary from permutation tests
            metric_col: Name of the metric column
            mode: Analysis mode
            suffix: Optional suffix for filename (e.g., 'percentage', 'adaptive_trimmed')
        """
        output_config = self.config["output"]

        # Prepare data for export
        export_data = []

        for comparison_name, result in perm_results.items():
            for i, time_bin in enumerate(result["time_bins"]):
                export_data.append(
                    {
                        "comparison": comparison_name,
                        "group1": result["group1"],
                        "group2": result["group2"],
                        "time_bin": time_bin,
                        "observed_diff": result["observed_diffs"][i],
                        "p_value_raw": result["p_values_raw"][i],
                        "p_value_fdr": result["p_values_corrected"][i],
                        "significant_fdr": result["p_values_corrected"][i] < 0.05,
                        "n_group1": result["n_group1_per_bin"][i],
                        "n_group2": result["n_group2_per_bin"][i],
                    }
                )

        df_export = pd.DataFrame(export_data)

        # Construct filename with optional suffix
        filename_base = f"{metric_col}_permutation_results"
        if suffix:
            filename_base = f"{metric_col}_permutation_results_{suffix}"

        # Save to CSV
        if mode.startswith("tnt_"):
            output_dir = Path(__file__).parent / mode.replace("tnt_", "").upper()
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{filename_base}.csv"
        else:
            output_path = Path(__file__).parent / f"{filename_base}.csv"

        df_export.to_csv(output_path, index=False)
        print(f"📄 Permutation results saved to: {output_path}")

    def _create_coordinate_line_plots(
        self, data, x_col, y_cols, hue_col, title, filename, mode, xlabel="Time", time_type="percentage", var_type=None
    ):
        """Create line plots for coordinate data with confidence intervals."""
        if data.empty:
            print(f"⚠️  No data for {filename}")
            return

        # Create figure
        fig, axes = plt.subplots(len(y_cols), 1, figsize=(12, 6 * len(y_cols)))
        if len(y_cols) == 1:
            axes = [axes]

        # Get color mapping - use var_type if provided, otherwise infer from mode
        unique_values = data[hue_col].unique()
        if var_type:
            # Use the explicitly provided variable type
            color_mapping = self.create_color_mapping(mode, unique_values, var_type)
        elif mode.startswith("tnt_"):
            # Use brain region colors for TNT modes
            color_mapping = self.create_color_mapping(mode, unique_values, "brain_region")
        else:
            # Use F1_condition colors for control mode
            color_mapping = self.create_color_mapping(mode, unique_values, "f1_condition")

        for i, y_col in enumerate(y_cols):
            ax = axes[i]

            if y_col not in data.columns:
                ax.text(0.5, 0.5, f"Column '{y_col}' not found", ha="center", va="center", transform=ax.transAxes)
                continue

            # Plot each group
            for hue_val in unique_values:
                subset = data[data[hue_col] == hue_val]
                if subset.empty or subset[y_col].isna().all():
                    continue

                # Skip training_ball for control groups (usually all NaN)
                if "training" in y_col and subset[y_col].isna().all():
                    continue

                # Bin data and calculate statistics
                subset_clean = subset.dropna(subset=[y_col])
                if subset_clean.empty:
                    continue

                # Create time bins (match permutation test bin size: 8.33 = ~12 bins across 0-100%)
                bin_size = 8.33 if time_type == "percentage" else 8.33
                subset_clean = subset_clean.copy()
                subset_clean["time_bin"] = np.floor(subset_clean[x_col] / bin_size) * bin_size

                # Get median per fly per time bin
                fly_medians = subset_clean.groupby(["fly", "time_bin"])[y_col].median().reset_index()

                # Calculate statistics across flies for each time bin
                time_stats = fly_medians.groupby("time_bin")[y_col].agg(["mean", "std", "count", "sem"]).reset_index()

                # Filter out bins with too few flies (prevents artifacts at the end)
                # Use 50% threshold for trimmed plots (more aggressive), 30% for percentage
                threshold = 0.5 if time_type == "trimmed" else 0.3
                min_flies_per_bin = max(3, int(subset["fly"].nunique() * threshold))
                time_stats = time_stats[time_stats["count"] >= min_flies_per_bin].copy()

                if time_stats.empty:
                    continue

                # Additional safeguard: Remove last bin if it shows suspicious drop
                # (mean drops by >15% from second-to-last bin)
                if len(time_stats) >= 2:
                    last_mean = time_stats.iloc[-1]["mean"]
                    second_last_mean = time_stats.iloc[-2]["mean"]
                    if last_mean < second_last_mean * 0.85:  # 15% drop threshold
                        print(
                            f"  ⚠️  Removing last bin for {hue_val} (suspicious {((second_last_mean - last_mean) / second_last_mean * 100):.1f}% drop)"
                        )
                        time_stats = time_stats.iloc[:-1].copy()

                if time_stats.empty:
                    continue

                # Calculate confidence intervals
                confidence_level = 0.95
                alpha = 1 - confidence_level
                time_stats["ci"] = time_stats.apply(
                    lambda row: stats.t.ppf(1 - alpha / 2, row["count"] - 1) * row["sem"] if row["count"] > 1 else 0,
                    axis=1,
                )

                # Plot mean trajectory
                color = color_mapping.get(hue_val, "black")
                label = str(hue_val)

                ax.plot(time_stats["time_bin"], time_stats["mean"], color=color, linewidth=2, alpha=0.8, label=label)

                # Add confidence intervals
                ax.fill_between(
                    time_stats["time_bin"],
                    time_stats["mean"] - time_stats["ci"],
                    time_stats["mean"] + time_stats["ci"],
                    color=color,
                    alpha=0.2,
                )

            # Formatting
            ax.set_title(f"{y_col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Distance (pixels)", fontsize=10)

            # Avoid very large legends which can blow up figure layout and slow rendering.
            max_legend_entries = 25
            try:
                n_entries = len(unique_values)
            except Exception:
                n_entries = 0

            if n_entries and n_entries <= max_legend_entries:
                ax.legend(loc="upper right", framealpha=0.9)
            elif n_entries and n_entries > max_legend_entries:
                ax.text(
                    0.02,
                    0.85,
                    f"{n_entries} groups — legend omitted for performance",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                )

            ax.grid(True, alpha=0.3)

            # Add vertical line at time 0
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Exit time")

            # Add sample sizes per condition
            if "fly" in data.columns and hue_col in data.columns:
                # Calculate sample size for each condition
                condition_counts = data.groupby(hue_col)["fly"].nunique().to_dict()
                # Format as "condition1: n1, condition2: n2, ..."
                sample_text = ", ".join([f"{cond}: n={count}" for cond, count in sorted(condition_counts.items())])
                ax.text(
                    0.02,
                    0.98,
                    sample_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )
            else:
                # Fallback to total if fly column not available
                n_flies = data["fly"].nunique() if "fly" in data.columns else len(data)
                ax.text(
                    0.02,
                    0.98,
                    f"N flies = {n_flies}",
                    transform=ax.transAxes,
                    fontsize=10,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save plot
        self._save_plot(fig, filename.replace(".png", ""), mode)

    def _create_individual_fly_plots(
        self, data, x_col, y_cols, hue_col, title, filename, mode, xlabel="Time", time_type="percentage", var_type=None
    ):
        """Create line plots showing individual fly trajectories with no averaging."""
        if data.empty:
            print(f"⚠️  No data for {filename}")
            return

        if "fly" not in data.columns:
            print(f"⚠️  No 'fly' column found - cannot create individual fly plots")
            return

        # Create figure
        fig, axes = plt.subplots(len(y_cols), 1, figsize=(12, 6 * len(y_cols)))
        if len(y_cols) == 1:
            axes = [axes]

        # Get color mapping based on var_type (explicit takes priority)
        unique_values = data[hue_col].unique()
        if var_type:
            # Use explicit var_type (pretraining, f1_condition, genotype, etc.)
            color_mapping = self.create_color_mapping(mode, unique_values, var_type)
        elif mode.startswith("tnt_"):
            # Use brain region colors for TNT modes
            color_mapping = self.create_color_mapping(mode, unique_values, "brain_region")
        else:
            # Use F1_condition colors for control mode
            color_mapping = self.create_color_mapping(mode, unique_values, "f1_condition")

        for i, y_col in enumerate(y_cols):
            ax = axes[i]

            if y_col not in data.columns:
                ax.text(0.5, 0.5, f"Column '{y_col}' not found", ha="center", va="center", transform=ax.transAxes)
                continue

            # Simple direct plotting: iterate through flies and plot raw coordinates
            fly_count_per_group = {}
            legend_handles = {}

            # Get max flies per group from config
            mode_config = self.config["analysis_modes"].get(mode, {})
            plot_params = mode_config.get("coordinates", {}).get("coordinates_plot_params", {})
            per_group_cap = int(plot_params.get("max_individual_lines_per_group", 50))

            # Group flies by condition for organized plotting
            condition_groups = data.groupby(hue_col)

            for hue_val, group_data in condition_groups:
                # Get color for this condition
                color = color_mapping.get(hue_val, "black")

                # Get all flies in this condition
                fly_ids = group_data["fly"].unique()
                n_flies = len(fly_ids)

                # Cap number of flies if needed (sample uniformly)
                if n_flies > per_group_cap:
                    indices = np.linspace(0, n_flies - 1, per_group_cap, dtype=int)
                    fly_ids = fly_ids[indices]

                # Plot each fly
                for fly_id in fly_ids:
                    fly_data = group_data[group_data["fly"] == fly_id]

                    # Clean data (remove NaNs)
                    fly_clean = fly_data.dropna(subset=[y_col, x_col])
                    if fly_clean.empty:
                        continue

                    # Plot this fly's trajectory
                    line = ax.plot(
                        fly_clean[x_col],
                        fly_clean[y_col],
                        color=color,
                        linewidth=1.5,
                        alpha=0.15,
                    )

                    # Track for legend (one entry per condition)
                    if hue_val not in legend_handles:
                        legend_handles[hue_val] = line[0]

                    fly_count_per_group[hue_val] = fly_count_per_group.get(hue_val, 0) + 1

            # Create legend with one entry per condition
            if legend_handles:
                handles = [legend_handles[hue_val] for hue_val in sorted(legend_handles.keys())]
                labels = [f"{hue_val} (n={fly_count_per_group[hue_val]})" for hue_val in sorted(legend_handles.keys())]
                ax.legend(handles, labels, loc="upper right", framealpha=0.9)

            # Formatting
            ax.set_title(f"{y_col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Distance (pixels)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add vertical line at time 0
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, linewidth=1.5)

            # Add sample sizes per condition
            condition_counts = data.groupby(hue_col)["fly"].nunique().to_dict()
            # Format as "condition1: n1, condition2: n2, ..."
            sample_text = ", ".join([f"{cond}: n={count}" for cond, count in sorted(condition_counts.items())])
            ax.text(
                0.02,
                0.98,
                sample_text,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save plot
        self._save_plot(fig, filename.replace(".png", ""), mode)

    def _create_combined_coordinate_plots(
        self,
        data,
        x_col,
        y_cols,
        primary_col,
        secondary_col,
        title,
        filename,
        mode,
        xlabel="Time",
        time_type="percentage",
    ):
        """
        Create combined coordinate plots showing genotype (color) + pretraining (linestyle).
        For TNT modes: genotype determines color (via brain region), pretraining determines linestyle.
        """
        if data.empty:
            print(f"⚠️  No data for {filename}")
            return

        if "fly" not in data.columns:
            print(f"⚠️  No 'fly' column found - cannot create combined plots")
            return

        # Create combined grouping variable
        data = data.copy()
        data["combined_group"] = data[secondary_col].astype(str) + " + " + data[primary_col].astype(str)

        # Get unique genotypes and pretraining values
        unique_genotypes = data[secondary_col].unique()
        unique_pretraining = data[primary_col].unique()

        # Create combined style mapping
        style_mapping = self.create_combined_style_mapping(mode, unique_genotypes, unique_pretraining)

        # Create figure
        fig, axes = plt.subplots(len(y_cols), 1, figsize=(12, 6 * len(y_cols)))
        if len(y_cols) == 1:
            axes = [axes]

        for i, y_col in enumerate(y_cols):
            ax = axes[i]

            if y_col not in data.columns:
                ax.text(0.5, 0.5, f"Column '{y_col}' not found", ha="center", va="center", transform=ax.transAxes)
                continue

            # Plot each genotype + pretraining combination
            for genotype in unique_genotypes:
                for pretraining in unique_pretraining:
                    subset = data[(data[secondary_col] == genotype) & (data[primary_col] == pretraining)]
                    if subset.empty or subset[y_col].isna().all():
                        continue

                    # Get style for this combination
                    style_info = style_mapping.get((genotype, pretraining), {"color": "gray", "linestyle": "-"})
                    color = style_info["color"]
                    linestyle = style_info["linestyle"]

                    # Bin data and calculate statistics
                    subset_clean = subset.dropna(subset=[y_col])
                    if subset_clean.empty:
                        continue

                    # Create time bins (use 8.33 for ~12 bins across 0-100%)
                    bin_size = 8.33 if time_type == "percentage" else 8.33
                    subset_clean = subset_clean.copy()
                    subset_clean["time_bin"] = np.floor(subset_clean[x_col] / bin_size) * bin_size

                    # Get median per fly per time bin
                    fly_medians = subset_clean.groupby(["fly", "time_bin"])[y_col].median().reset_index()

                    # Calculate statistics across flies for each time bin
                    time_stats = (
                        fly_medians.groupby("time_bin")[y_col].agg(["mean", "std", "count", "sem"]).reset_index()
                    )

                    # Filter out bins with too few flies
                    threshold = 0.5 if time_type == "trimmed" else 0.3
                    min_flies_per_bin = max(3, int(subset["fly"].nunique() * threshold))
                    time_stats = time_stats[time_stats["count"] >= min_flies_per_bin].copy()

                    if time_stats.empty:
                        continue

                    # Remove last bin if suspicious drop
                    if len(time_stats) >= 2:
                        last_mean = time_stats.iloc[-1]["mean"]
                        second_last_mean = time_stats.iloc[-2]["mean"]
                        if last_mean < second_last_mean * 0.85:
                            time_stats = time_stats.iloc[:-1].copy()

                    if time_stats.empty:
                        continue

                    # Calculate confidence intervals
                    confidence_level = 0.95
                    alpha = 1 - confidence_level
                    time_stats["ci"] = time_stats.apply(
                        lambda row: (
                            stats.t.ppf(1 - alpha / 2, row["count"] - 1) * row["sem"] if row["count"] > 1 else 0
                        ),
                        axis=1,
                    )

                    # Create label
                    label = f"{genotype} + {pretraining}"

                    # Plot mean trajectory with specific linestyle
                    ax.plot(
                        time_stats["time_bin"],
                        time_stats["mean"],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2,
                        alpha=0.8,
                        label=label,
                    )

                    # Add confidence intervals
                    ax.fill_between(
                        time_stats["time_bin"],
                        time_stats["mean"] - time_stats["ci"],
                        time_stats["mean"] + time_stats["ci"],
                        color=color,
                        alpha=0.15,
                    )

            # Formatting
            ax.set_title(f"{y_col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Distance (pixels)", fontsize=10)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add vertical line at time 0
            ax.axvline(x=0, color="black", linestyle=":", alpha=0.7, label="Exit time")

            # Add sample sizes per combined group
            if "combined_group" in data.columns:
                condition_counts = data.groupby("combined_group")["fly"].nunique().to_dict()
                sample_text = ", ".join([f"{cond}: n={count}" for cond, count in sorted(condition_counts.items())])
                ax.text(
                    0.02,
                    0.98,
                    sample_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save plot
        self._save_plot(fig, filename.replace(".png", ""), mode)

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

        # 2. Boxplot analyses - analyze ALL discovered metrics
        print(f"\n📊 Running comprehensive boxplot analysis...")
        try:
            # Pass metric=None to analyze all available metrics
            self.analyze_boxplots(mode, metric_type=metric)
            print(f"✅ Boxplot analysis completed")
        except Exception as e:
            print(f"❌ Boxplot analysis failed: {e}")

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
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "control",
            "tnt_mb247",
            "tnt_lc10_2",
            "tnt_ddc",
            "tnt_mb247_pooled",
            "tnt_lc10_2_pooled",
            "tnt_ddc_pooled",
        ],
        help="Analysis mode (use '_pooled' suffix to include shared controls from other experiments)",
    )
    parser.add_argument(
        "--analysis",
        choices=["binary_metrics", "boxplots", "coordinates", "all"],
        help="Type of analysis to perform. If not specified, runs all available analyses.",
    )
    parser.add_argument(
        "--metric", help="Specific metric for boxplot analysis. If not specified, analyzes all available metrics."
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory for plots. Defaults based on mode: "
            "control=/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New, "
            "tnt_mb247=/mnt/upramdya_data/MD/F1_Tracks/MB247, "
            "tnt_lc10_2=/mnt/upramdya_data/MD/F1_Tracks/LC10-2, "
            "tnt_ddc=/mnt/upramdya_data/MD/F1_Tracks/DDC, "
            "pooled modes use same directories with '_pooled' suffix in filename"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots in addition to saving them. By default, plots are only saved.",
    )
    parser.add_argument(
        "--controls",
        choices=["EmptyGal4", "EmptySplit", "both", "none"],
        default=None,
        help=(
            "Filter which control genotypes to include in analysis. "
            "EmptyGal4: only TNTxEmptyGal4, "
            "EmptySplit: only TNTxEmptySplit, "
            "both: include both controls, "
            "none: exclude all controls (experimental genotypes only). "
            "Default: include all genotypes in brain_region_mapping config."
        ),
    )

    args = parser.parse_args()

    # Initialize framework with control filter
    framework = F1AnalysisFramework(args.config, args.output_dir, args.controls)

    # Override show_plots config based on --show argument
    framework.config["output"]["show_plots"] = args.show

    # Determine and display output directory
    if framework.custom_output_dir:
        output_dir = framework.custom_output_dir
    elif args.mode == "control":
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/Plots/F1_New")
    elif args.mode == "tnt_mb247" or args.mode == "tnt_mb247_pooled":
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/MB247")
    elif args.mode == "tnt_lc10_2" or args.mode == "tnt_lc10_2_pooled":
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/LC10-2")
    elif args.mode == "tnt_ddc" or args.mode == "tnt_ddc_pooled":
        output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/DDC")
    else:
        output_dir = Path(__file__).parent

    # Add pooled suffix to mode name for display
    mode_display = args.mode.replace("_", " ").title()
    if "pooled" in args.mode.lower():
        print(f"\n🔄 Running in POOLED mode - combining shared controls from multiple experiments")

    print(f"\n📁 Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # If no analysis specified, run all available analyses
    if args.analysis is None or args.analysis == "all":
        print(f"Running all available analyses for mode: {args.mode}")
        framework.run_all_analyses(args.mode, args.metric)
    else:
        # Run specific analysis
        if args.analysis == "binary_metrics":
            framework.analyze_binary_metrics(args.mode)
        elif args.analysis == "boxplots":
            if args.metric:
                # Single metric specified
                framework.analyze_boxplots(args.mode, args.metric)
            else:
                # No metric specified - analyze ALL available metrics using auto-discovery
                print(f"\nNo specific metric provided. Using auto-discovery to analyze all available metrics...")
                framework.analyze_boxplots(args.mode, metric_type=None)
        elif args.analysis == "coordinates":
            framework.analyze_coordinates(args.mode)

        print(f"Analysis completed for mode: {args.mode}, analysis: {args.analysis}")


if __name__ == "__main__":
    main()
