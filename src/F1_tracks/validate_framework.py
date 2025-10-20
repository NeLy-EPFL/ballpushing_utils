#!/usr/bin/env python3
"""
Utility script for testing and validating the unified F1 analysis framework.
Provides tools to compare outputs with original scripts and validate configurations.
"""

import yaml
import pandas as pd
from pathlib import Path
import argparse
from unified_f1_analysis import F1AnalysisFramework


class F1AnalysisValidator:
    """Validator for the unified F1 analysis framework."""

    def __init__(self, config_path=None):
        """Initialize validator with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "analysis_config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.framework = F1AnalysisFramework(config_path)

    def validate_datasets(self):
        """Validate that all configured datasets are accessible."""
        print("Validating dataset accessibility...")

        for mode, mode_config in self.config["analysis_modes"].items():
            dataset_path = mode_config["dataset_path"]
            print(f"\nChecking {mode} dataset: {dataset_path}")

            if not Path(dataset_path).exists():
                print(f"  ❌ Dataset not found: {dataset_path}")
                continue

            try:
                df = pd.read_feather(dataset_path)
                print(f"  ✅ Dataset loaded successfully: {df.shape}")

                # Check for required columns
                detected_cols = self.framework.detect_columns(df, mode)
                missing_cols = []

                for col_type, col_name in detected_cols.items():
                    if col_name is None:
                        missing_cols.append(col_type)

                if missing_cols:
                    print(f"  ⚠️  Could not detect columns: {missing_cols}")
                else:
                    print(f"  ✅ All required columns detected: {detected_cols}")

                # Check for binary metrics
                binary_metrics = self.config["analysis_parameters"]["binary_metrics"]
                available_metrics = [m for m in binary_metrics if m in df.columns]
                missing_metrics = [m for m in binary_metrics if m not in df.columns]

                print(f"  📊 Available binary metrics: {available_metrics}")
                if missing_metrics:
                    print(f"  ⚠️  Missing binary metrics: {missing_metrics}")

            except Exception as e:
                print(f"  ❌ Error loading dataset: {e}")

    def validate_config(self):
        """Validate configuration file structure and values."""
        print("Validating configuration structure...")

        required_sections = ["analysis_modes", "analysis_parameters", "plot_styling", "output", "column_detection"]

        for section in required_sections:
            if section in self.config:
                print(f"  ✅ Found section: {section}")
            else:
                print(f"  ❌ Missing section: {section}")

        # Validate analysis modes
        print("\nValidating analysis modes...")
        for mode, mode_config in self.config["analysis_modes"].items():
            print(f"  Mode: {mode}")

            required_keys = ["description", "dataset_path", "grouping_variables"]
            for key in required_keys:
                if key in mode_config:
                    print(f"    ✅ {key}")
                else:
                    print(f"    ❌ Missing {key}")

        # Validate plot styling
        print("\nValidating plot styling...")
        for mode in self.config["analysis_modes"].keys():
            if mode in self.config["plot_styling"]:
                print(f"  ✅ Styling defined for mode: {mode}")
            else:
                print(f"  ⚠️  No styling defined for mode: {mode}")

    def test_analysis_run(self, mode="control", dry_run=True):
        """Test running an analysis without saving plots."""
        print(f"Testing analysis run for mode: {mode}")

        try:
            # Temporarily disable plot saving and showing
            original_save = self.config["output"]["save_plots"]
            original_show = self.config["output"]["show_plots"]

            if dry_run:
                self.config["output"]["save_plots"] = False
                self.config["output"]["show_plots"] = False

            # Run binary metrics analysis
            self.framework.analyze_binary_metrics(mode)

            print(f"  ✅ Analysis completed successfully for mode: {mode}")

            # Restore original settings
            self.config["output"]["save_plots"] = original_save
            self.config["output"]["show_plots"] = original_show

        except Exception as e:
            print(f"  ❌ Analysis failed for mode {mode}: {e}")
            import traceback

            traceback.print_exc()

    def compare_with_original(self, mode, original_script_path):
        """Compare framework output with original script (placeholder for future implementation)."""
        print(f"Comparing framework output with original script...")
        print(f"Mode: {mode}")
        print(f"Original script: {original_script_path}")
        print("⚠️  Comparison functionality not yet implemented")
        # TODO: Implement comparison logic

    def generate_usage_examples(self):
        """Generate usage examples for all configured modes."""
        print("Usage Examples:")
        print("=" * 50)

        for mode in self.config["analysis_modes"].keys():
            print(f"\n# {mode.replace('_', ' ').title()} Analysis")
            print(f"python unified_f1_analysis.py --mode {mode} --analysis binary_metrics")
            print(f"python extended_f1_analysis.py --mode {mode} --analysis boxplots --metric interaction_rate")

        print("\n# Custom configuration")
        print("python unified_f1_analysis.py --mode control --analysis binary_metrics --config custom_config.yaml")

    def check_dependencies(self):
        """Check if all required dependencies are available."""
        print("Checking dependencies...")

        required_packages = ["pandas", "matplotlib", "seaborn", "numpy", "scipy", "yaml", "pathlib"]

        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} - not installed")

        # Check for brain region config
        try:
            import sys

            sys.path.append("/home/matthias/ballpushing_utils/src")
            from PCA import Config

            print(f"  ✅ Brain region Config - loaded {len(Config.SplitRegistry)} entries")
        except Exception as e:
            print(f"  ⚠️  Brain region Config - {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="F1 analysis framework validator")
    parser.add_argument(
        "--action",
        required=True,
        choices=["validate_datasets", "validate_config", "test_run", "check_deps", "examples", "full_validation"],
        help="Validation action to perform",
    )
    parser.add_argument(
        "--mode", default="control", choices=["control", "tnt_mb247", "tnt_lc10_2"], help="Mode for test run"
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run test without saving plots")

    args = parser.parse_args()

    # Initialize validator
    validator = F1AnalysisValidator(args.config)

    if args.action == "validate_datasets":
        validator.validate_datasets()
    elif args.action == "validate_config":
        validator.validate_config()
    elif args.action == "test_run":
        validator.test_analysis_run(args.mode, args.dry_run)
    elif args.action == "check_deps":
        validator.check_dependencies()
    elif args.action == "examples":
        validator.generate_usage_examples()
    elif args.action == "full_validation":
        print("Running full validation suite...")
        print("\n" + "=" * 60)
        validator.check_dependencies()
        print("\n" + "=" * 60)
        validator.validate_config()
        print("\n" + "=" * 60)
        validator.validate_datasets()
        print("\n" + "=" * 60)
        validator.test_analysis_run(args.mode, dry_run=True)
        print("\n" + "=" * 60)
        validator.generate_usage_examples()
        print("\n" + "=" * 60)
        print("✅ Full validation completed")


if __name__ == "__main__":
    main()
