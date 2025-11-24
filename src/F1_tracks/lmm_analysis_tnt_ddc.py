#!/usr/bin/env python3
"""
Linear Mixed-Effects Model (LMM) Analysis for TNT DDC experiments.

This script fits the core model: Genotype * Pretraining (with random effects for Date).

The model always includes:
- Genotype (main effect)
- Pretraining (main effect)
- Genotype × Pretraining interaction

This tests:
1. Does genotype affect the response variable?
2. Does pretraining affect the response variable?
3. Does the effect of pretraining depend on genotype (interaction)?

Usage:
    python lmm_analysis_tnt_ddc.py [--metric distance_moved] [--max-iter 200]
"""

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.anova import anova_lm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Suppress convergence warnings for display (we'll track them)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class LMMAnalysis:
    """
    Linear Mixed-Effects Model analysis with automated model selection.

    Similar to R's lme4 + MuMIn workflow for multivariate analysis.
    """

    def __init__(self, data, response_var, output_dir):
        """
        Initialize LMM analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with all variables
        response_var : str
            Name of response variable (e.g., 'distance_moved')
        output_dir : Path
            Directory to save outputs
        """
        self.data = data.copy()
        self.response_var = response_var
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.model_comparison = None
        self.best_model = None
        self.best_model_result = None

    def prepare_data(self, genotype_col, pretraining_col, date_col=None, fly_col=None):
        """
        Prepare data for LMM analysis.

        Parameters
        ----------
        genotype_col : str
            Column name for genotype
        pretraining_col : str
            Column name for pretraining
        date_col : str, optional
            Column name for date (random effect)
        fly_col : str, optional
            Column name for fly ID (random effect)
        """
        # Standardize column names for easier formula writing
        self.data["Genotype"] = self.data[genotype_col]
        self.data["Pretraining"] = self.data[pretraining_col]

        if date_col and date_col in self.data.columns:
            self.data["Date"] = self.data[date_col]
            self.has_date = True
        else:
            self.has_date = False

        if fly_col and fly_col in self.data.columns:
            self.data["fly"] = self.data[fly_col]
            self.has_fly = True
        else:
            self.has_fly = False

        # Ensure response variable exists
        if self.response_var not in self.data.columns:
            raise ValueError(f"Response variable '{self.response_var}' not found in dataset")

        # Clean data
        required_cols = ["Genotype", "Pretraining", self.response_var]
        if self.has_date:
            required_cols.append("Date")
        if self.has_fly:
            required_cols.append("fly")

        self.data = self.data[required_cols].dropna()

        # Remove infinite values
        self.data = self.data[~np.isinf(self.data[self.response_var])]

        print(f"\n📊 Data prepared for LMM analysis:")
        print(f"   Response variable: {self.response_var}")
        print(f"   Sample size: {len(self.data)}")
        print(f"   Genotypes: {sorted(self.data['Genotype'].unique())}")
        print(f"   Pretraining levels: {sorted(self.data['Pretraining'].unique())}")
        if self.has_date:
            print(f"   Dates: {len(self.data['Date'].unique())} levels")
        if self.has_fly:
            print(f"   Flies: {len(self.data['fly'].unique())} individuals")

    def fit_model(self, formula, model_name, max_iter=200):
        """
        Fit a single mixed-effects model.

        Parameters
        ----------
        formula : str
            Model formula (e.g., "distance_moved ~ Genotype + Pretraining")
        model_name : str
            Name for this model
        max_iter : int
            Maximum iterations for fitting

        Returns
        -------
        dict
            Model results and diagnostics
        """
        try:
            # Determine groups for random effects
            if self.has_date:
                groups = self.data["Date"]
            elif self.has_fly:
                groups = self.data["fly"]
            else:
                raise ValueError("Need at least one grouping variable (Date or fly)")

            # Fit model
            model = mixedlm(formula, self.data, groups=groups)
            result = model.fit(maxiter=max_iter, method="lbfgs")

            # Calculate additional metrics
            n_params = len(result.params)

            # Handle cases where AIC/BIC might be invalid (inf/-inf/nan)
            aic = result.aic if np.isfinite(result.aic) else np.nan
            bic = result.bic if np.isfinite(result.bic) else np.nan
            llf = result.llf if np.isfinite(result.llf) else np.nan

            # Check for singular covariance
            is_singular = False
            try:
                if hasattr(result, "cov_re") and result.cov_re is not None:
                    is_singular = np.abs(np.linalg.det(result.cov_re)) < 1e-10
            except:
                is_singular = True

            return {
                "name": model_name,
                "formula": formula,
                "result": result,
                "AIC": aic,
                "BIC": bic,
                "logLik": llf,
                "n_params": n_params,
                "converged": result.converged,
                "n_obs": len(self.data),
                "is_singular": is_singular,
            }

        except Exception as e:
            print(f"⚠️  Failed to fit model '{model_name}': {e}")
            return None

    def fit_core_model(self, max_iter=200, fallback_to_ols=True):
        """
        Fit the core model: Genotype * Pretraining (includes main effects + interaction).

        This is the primary model of interest for testing genotype and pretraining effects.
        Will attempt mixed-effects model first, then fall back to OLS if it fails.

        Parameters
        ----------
        max_iter : int
            Maximum iterations for model fitting
        fallback_to_ols : bool
            If True, use OLS when mixed model fails (recommended)

        Returns
        -------
        dict
            Model results
        """
        print(f"\n🔬 Fitting core model: Genotype * Pretraining...")

        # Genotype * Pretraining expands to: Genotype + Pretraining + Genotype:Pretraining
        formula = f"{self.response_var} ~ Genotype * Pretraining"
        model_name = "Core Model: Genotype * Pretraining"

        print(f"   Formula: {formula}")
        print(f"   (expands to: Genotype + Pretraining + Genotype:Pretraining)")

        # Try mixed model first
        print(f"\n   Attempting mixed-effects model with Date as random effect...")
        core_model = self.fit_model(formula, model_name + " (LMM)", max_iter)

        # If mixed model fails and fallback is enabled, use OLS
        if core_model is None and fallback_to_ols:
            print(f"\n   Mixed model failed. Falling back to OLS (Ordinary Least Squares)...")
            print(f"   This is appropriate when random effects explain zero variance.")

            try:
                # Fit OLS model
                ols_result = ols(formula, data=self.data).fit()

                # Package results in same format
                core_model = {
                    "name": model_name + " (OLS)",
                    "formula": formula,
                    "result": ols_result,
                    "AIC": ols_result.aic,
                    "BIC": ols_result.bic,
                    "logLik": ols_result.llf,
                    "n_params": len(ols_result.params),
                    "converged": True,
                    "n_obs": len(self.data),
                    "is_singular": False,
                    "model_type": "OLS",
                }

                print(f"\n✓ OLS model fitted successfully")
                print(f"   Note: Using regular linear regression (no random effects)")

            except Exception as e:
                print(f"\n❌ OLS model also failed: {e}")
                return None

        if core_model:
            self.best_model = core_model
            self.best_model_result = core_model["result"]

            # Create a simple "comparison" table with just this model
            self.model_comparison = pd.DataFrame(
                [
                    {
                        "Model": core_model["name"],
                        "Formula": core_model["formula"],
                        "AIC": core_model["AIC"],
                        "BIC": core_model["BIC"],
                        "logLik": core_model["logLik"],
                        "df": core_model["n_params"],
                        "Converged": "✓" if core_model["converged"] else "✗",
                    }
                ]
            )

            self.models = {core_model["name"]: core_model}

            model_type = core_model.get("model_type", "LMM")
            print(f"\n✓ Core model fitted successfully ({model_type})")

            if core_model.get("is_singular", False):
                print("\n⚠️  Note: Singular random effects covariance detected.")
                print("   This means Date/fly add no variance beyond fixed effects.")
                print("   Fixed effects are still valid and interpretable!")
        else:
            print("\n❌ Failed to fit core model")

        return core_model

    def compare_all_models(self, include_interactions=True, max_iter=200):
        """
        Compare all possible combinations of fixed effects (like R's MuMIn::dredge).

        NOTE: For your use case, consider using fit_core_model() instead,
        which always includes Genotype * Pretraining.

        Parameters
        ----------
        include_interactions : bool
            Whether to include interaction terms
        max_iter : int
            Maximum iterations for model fitting
        """
        print(f"\n🔬 Fitting and comparing all model combinations...")
        print("   (Note: If you always want Genotype*Pretraining, use fit_core_model() instead)")

        # Define possible fixed effects
        fixed_effects = ["Genotype", "Pretraining"]

        if include_interactions:
            fixed_effects.append("Genotype:Pretraining")

        all_models = []

        # Null model (intercept only)
        print(f"\n   Fitting null model (intercept only)...")
        null_model = self.fit_model(f"{self.response_var} ~ 1", "Null (Intercept only)", max_iter)
        if null_model:
            all_models.append(null_model)

        # All combinations of fixed effects
        model_count = 1
        total_models = sum(1 for r in range(1, len(fixed_effects) + 1) for _ in combinations(fixed_effects, r))

        for r in range(1, len(fixed_effects) + 1):
            for combo in combinations(fixed_effects, r):
                # Build formula
                fixed_part = " + ".join(combo)
                formula = f"{self.response_var} ~ {fixed_part}"
                model_name = f"Model {model_count}: {fixed_part}"

                print(f"   Fitting {model_name} ({model_count}/{total_models})...")

                model_result = self.fit_model(formula, model_name, max_iter)
                if model_result:
                    all_models.append(model_result)

                model_count += 1

        # Create comparison table
        comparison_data = []
        for model in all_models:
            comparison_data.append(
                {
                    "Model": model["name"],
                    "Formula": model["formula"],
                    "AIC": model["AIC"],
                    "BIC": model["BIC"],
                    "logLik": model["logLik"],
                    "df": model["n_params"],
                    "Converged": "✓" if model["converged"] else "✗",
                }
            )

        self.model_comparison = pd.DataFrame(comparison_data)

        # Check if we have valid AIC values
        valid_aic = self.model_comparison["AIC"].notna() & np.isfinite(self.model_comparison["AIC"])

        if not valid_aic.any():
            print("\n⚠️  Warning: No models with valid AIC values!")
            print("   This typically indicates singular random effects covariance.")
            print("   Models can still be compared using BIC or likelihood ratio tests.")

            # Try to use BIC instead
            valid_bic = self.model_comparison["BIC"].notna() & np.isfinite(self.model_comparison["BIC"])
            if valid_bic.any():
                print("   Falling back to BIC for model comparison.")
                self.model_comparison = self.model_comparison.sort_values("BIC").reset_index(drop=True)
                self.model_comparison["delta_BIC"] = self.model_comparison["BIC"] - self.model_comparison["BIC"].min()
                self.model_comparison["BIC_weight"] = np.exp(-0.5 * self.model_comparison["delta_BIC"])
                self.model_comparison["BIC_weight"] /= self.model_comparison["BIC_weight"].sum()
            else:
                # Just sort by number of parameters (prefer simpler models)
                self.model_comparison = self.model_comparison.sort_values("df").reset_index(drop=True)
        else:
            # Sort by AIC (lower is better)
            self.model_comparison = self.model_comparison.sort_values("AIC").reset_index(drop=True)

            # Calculate delta AIC and AIC weights
            self.model_comparison["delta_AIC"] = self.model_comparison["AIC"] - self.model_comparison["AIC"].min()

            # AIC weights (relative likelihood)
            self.model_comparison["AIC_weight"] = np.exp(-0.5 * self.model_comparison["delta_AIC"])
            self.model_comparison["AIC_weight"] /= self.model_comparison["AIC_weight"].sum()

        # Store all models
        self.models = {m["name"]: m for m in all_models}

        print(f"\n✓ Fitted {len(all_models)} models successfully")

        return self.model_comparison

    def get_best_model(self, criterion="AIC"):
        """
        Get the best model based on information criterion.

        Parameters
        ----------
        criterion : str
            'AIC' or 'BIC'
        """
        if self.model_comparison is None:
            raise ValueError("Run compare_all_models() first")

        # Check if criterion column has valid values
        if criterion not in self.model_comparison.columns:
            # Fallback to BIC if AIC not available
            if "BIC" in self.model_comparison.columns:
                criterion = "BIC"
                print(f"\n⚠️  {criterion} not available, using BIC instead")
            else:
                # Just use the first model
                best_row = self.model_comparison.iloc[0]
                best_model_name = best_row["Model"]
                self.best_model = self.models[best_model_name]
                self.best_model_result = self.best_model["result"]
                print(f"\n🏆 Best model (first in list):")
                print(f"   {best_model_name}")
                print(f"   Formula: {self.best_model['formula']}")
                return self.best_model

        # Filter out rows with invalid criterion values
        valid_models = self.model_comparison[
            self.model_comparison[criterion].notna() & np.isfinite(self.model_comparison[criterion])
        ]

        if valid_models.empty:
            # Fallback: use first model
            best_row = self.model_comparison.iloc[0]
        else:
            best_row = valid_models.sort_values(criterion).iloc[0]

        best_model_name = best_row["Model"]

        self.best_model = self.models[best_model_name]
        self.best_model_result = self.best_model["result"]

        print(f"\n🏆 Best model (by {criterion}):")
        print(f"   {best_model_name}")
        print(f"   Formula: {self.best_model['formula']}")

        if pd.notna(best_row[criterion]) and np.isfinite(best_row[criterion]):
            print(f"   {criterion}: {best_row[criterion]:.2f}")
        else:
            print(f"   {criterion}: Not available (singular covariance)")

        # Only print weight if it exists
        if "AIC_weight" in best_row.index and pd.notna(best_row["AIC_weight"]):
            print(f"   AIC weight: {best_row['AIC_weight']:.4f}")
        elif "BIC_weight" in best_row.index and pd.notna(best_row["BIC_weight"]):
            print(f"   BIC weight: {best_row['BIC_weight']:.4f}")

        return self.best_model

    def print_model_comparison(self, top_n=None):
        """
        Print model comparison table.

        Parameters
        ----------
        top_n : int, optional
            Only show top N models
        """
        if self.model_comparison is None:
            raise ValueError("Run compare_all_models() first")

        print("\n" + "=" * 120)

        # Determine which criterion was used
        has_aic = "delta_AIC" in self.model_comparison.columns
        has_bic = "delta_BIC" in self.model_comparison.columns

        if has_aic:
            print("MODEL COMPARISON TABLE (sorted by AIC)")
        elif has_bic:
            print("MODEL COMPARISON TABLE (sorted by BIC)")
        else:
            print("MODEL COMPARISON TABLE")
        print("=" * 120)

        display_df = self.model_comparison.copy()
        if top_n:
            display_df = display_df.head(top_n)

        # Format for display - only format columns that exist
        for col in ["AIC", "BIC", "logLik"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "N/A"
                )

        if "delta_AIC" in display_df.columns:
            display_df["delta_AIC"] = display_df["delta_AIC"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "N/A"
            )

        if "delta_BIC" in display_df.columns:
            display_df["delta_BIC"] = display_df["delta_BIC"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "N/A"
            )

        if "AIC_weight" in display_df.columns:
            display_df["AIC_weight"] = display_df["AIC_weight"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) else "N/A"
            )

        if "BIC_weight" in display_df.columns:
            display_df["BIC_weight"] = display_df["BIC_weight"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) else "N/A"
            )

        print(display_df.to_string(index=False))
        print("=" * 120)

        # Interpretation guide - adapt based on what's available
        if has_aic or has_bic:
            print("\nInterpretation Guide:")
            if has_aic:
                print("  • delta_AIC < 2:  Substantial support (models are competitive)")
                print("  • delta_AIC 4-7:  Considerably less support")
                print("  • delta_AIC > 10: Essentially no support")
                print("  • AIC_weight:     Relative probability that this is the best model")
            elif has_bic:
                print("  • delta_BIC < 2:  Substantial support (models are competitive)")
                print("  • delta_BIC 6-10: Considerably less support")
                print("  • delta_BIC > 10: Essentially no support")
                print("  • BIC_weight:     Relative probability that this is the best model")
        else:
            print("\nNote: Model comparison metrics unavailable due to singular covariance.")

    def print_best_model_summary(self):
        """Print detailed summary of the best model."""
        if self.best_model_result is None:
            raise ValueError("Run get_best_model() first")

        print("\n" + "=" * 120)
        print("BEST MODEL SUMMARY")
        print("=" * 120)
        print(self.best_model_result.summary())
        print("=" * 120)

    def plot_diagnostics(self):
        """Create diagnostic plots for the best model."""
        if self.best_model_result is None:
            raise ValueError("Run get_best_model() first")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Get residuals and fitted values
        # Handle singular covariance case
        try:
            residuals = self.best_model_result.resid
            fitted = self.best_model_result.fittedvalues
        except ValueError as e:
            if "singular covariance" in str(e).lower():
                print("\n⚠️  Warning: Singular random effects covariance detected.")
                print("   Using marginal residuals (without random effects) for diagnostics.")
                # Use marginal residuals instead
                residuals = (
                    self.data[self.response_var]
                    - self.best_model_result.fe_params @ self.best_model_result.model.exog.T
                )
                fitted = self.best_model_result.fe_params @ self.best_model_result.model.exog.T
                residuals = np.array(residuals)
                fitted = np.array(fitted)
            else:
                raise

        # 1. Residuals vs Fitted
        ax = axes[0, 0]
        ax.scatter(fitted, residuals, alpha=0.5, edgecolors="k", linewidth=0.5)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Fitted Values", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residuals", fontsize=12, fontweight="bold")
        ax.set_title("Residuals vs Fitted\n(Check for homoscedasticity)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add loess smoother
        from scipy.interpolate import make_interp_spline

        if len(fitted) > 10:
            try:
                sorted_idx = np.argsort(fitted)
                fitted_sorted = fitted[sorted_idx]
                residuals_sorted = residuals[sorted_idx]

                # Simple moving average as smoother
                window = max(3, len(fitted) // 20)
                smoothed = pd.Series(residuals_sorted).rolling(window, center=True).mean()
                ax.plot(fitted_sorted, smoothed, "r-", linewidth=2, alpha=0.7)
            except:
                pass

        # 2. Q-Q plot
        ax = axes[0, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot\n(Check for normality of residuals)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 3. Scale-Location plot
        ax = axes[1, 0]
        sqrt_abs_resid = np.sqrt(np.abs(residuals))
        ax.scatter(fitted, sqrt_abs_resid, alpha=0.5, edgecolors="k", linewidth=0.5)
        ax.set_xlabel("Fitted Values", fontsize=12, fontweight="bold")
        ax.set_ylabel("√|Residuals|", fontsize=12, fontweight="bold")
        ax.set_title("Scale-Location Plot\n(Check for homoscedasticity)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add smoother
        if len(fitted) > 10:
            try:
                sorted_idx = np.argsort(fitted)
                fitted_sorted = fitted[sorted_idx]
                sqrt_resid_sorted = sqrt_abs_resid[sorted_idx]

                window = max(3, len(fitted) // 20)
                smoothed = pd.Series(sqrt_resid_sorted).rolling(window, center=True).mean()
                ax.plot(fitted_sorted, smoothed, "r-", linewidth=2, alpha=0.7)
            except:
                pass

        # 4. Histogram of residuals
        ax = axes[1, 1]
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Residuals", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title("Histogram of Residuals\n(Check for normality)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(f'Diagnostic Plots: {self.best_model["name"]}', fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save
        plot_path = self.output_dir / f"lmm_diagnostics_{self.response_var}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 Diagnostic plots saved to: {plot_path}")

        plt.show()

    def plot_effects(self):
        """Plot fixed effects from the best model."""
        if self.best_model_result is None:
            raise ValueError("Run get_best_model() first")

        # Extract fixed effects
        params = self.best_model_result.params
        conf_int = self.best_model_result.conf_int()

        # Remove intercept and random effects
        fixed_params = params[params.index != "Group Var"]
        fixed_params = fixed_params[~fixed_params.index.str.contains("Group")]

        if "Intercept" in fixed_params.index:
            fixed_params = fixed_params.drop("Intercept")
            conf_int = conf_int.loc[fixed_params.index]

        if len(fixed_params) == 0:
            print("⚠️  No fixed effects to plot (intercept-only model)")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(fixed_params) * 0.5)))

        y_pos = np.arange(len(fixed_params))

        # Plot coefficients with confidence intervals
        ax.errorbar(
            fixed_params.values,
            y_pos,
            xerr=[fixed_params.values - conf_int[0].values, conf_int[1].values - fixed_params.values],
            fmt="o",
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        # Add vertical line at 0
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.7)

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fixed_params.index)
        ax.set_xlabel("Effect Size (Coefficient)", fontsize=12, fontweight="bold")
        ax.set_title(
            f'Fixed Effects: {self.best_model["name"]}\nwith 95% Confidence Intervals', fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for i, (coef, ci_low, ci_high) in enumerate(zip(fixed_params.values, conf_int[0].values, conf_int[1].values)):
            ax.text(coef, i, f"  {coef:.3f}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()

        # Save
        plot_path = self.output_dir / f"lmm_effects_{self.response_var}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Effects plot saved to: {plot_path}")

        plt.show()

    def export_results(self):
        """Export all results to files."""
        # 1. Model comparison table
        comparison_path = self.output_dir / f"lmm_model_comparison_{self.response_var}.csv"
        self.model_comparison.to_csv(comparison_path, index=False)
        print(f"\n📄 Model comparison table saved to: {comparison_path}")

        # 2. Best model summary
        summary_path = self.output_dir / f"lmm_best_model_summary_{self.response_var}.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 120 + "\n")
            f.write(f"BEST MODEL SUMMARY: {self.response_var}\n")
            f.write("=" * 120 + "\n\n")
            f.write(f"Best Model: {self.best_model['name']}\n")
            f.write(f"Formula: {self.best_model['formula']}\n")
            f.write(f"AIC: {self.best_model['AIC']:.2f}\n")
            f.write(f"BIC: {self.best_model['BIC']:.2f}\n")
            f.write(f"Log-Likelihood: {self.best_model['logLik']:.2f}\n")
            f.write(f"Converged: {'Yes' if self.best_model['converged'] else 'No'}\n\n")
            f.write(str(self.best_model_result.summary()))
            f.write("\n" + "=" * 120 + "\n")
        print(f"📄 Best model summary saved to: {summary_path}")

        # 3. Fixed effects coefficients
        params_df = pd.DataFrame(
            {
                "Parameter": self.best_model_result.params.index,
                "Coefficient": self.best_model_result.params.values,
                "Std_Error": self.best_model_result.bse.values,
                "z_value": self.best_model_result.tvalues.values,
                "p_value": self.best_model_result.pvalues.values,
                "CI_lower": self.best_model_result.conf_int()[0].values,
                "CI_upper": self.best_model_result.conf_int()[1].values,
            }
        )

        params_path = self.output_dir / f"lmm_coefficients_{self.response_var}.csv"
        params_df.to_csv(params_path, index=False)
        print(f"📄 Model coefficients saved to: {params_path}")


def main(metric="distance_moved", max_iter=200):
    """
    Main function to run LMM analysis.

    Parameters
    ----------
    metric : str
        Response variable to analyze
    max_iter : int
        Maximum iterations for model fitting
    """
    print("\n" + "=" * 120)
    print(f"LINEAR MIXED-EFFECTS MODEL ANALYSIS (TNT DDC): {metric}")
    print("=" * 120)

    # Load data
    dataset_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Datasets/F1_TNT_DDC_Data/summary/pooled_summary.feather")

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    print(f"\n📁 Loading dataset from: {dataset_path}")
    df = pd.read_feather(dataset_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Detect columns
    genotype_col = None
    for col in df.columns:
        if "genotype" in col.lower():
            genotype_col = col
            break

    pretraining_col = None
    for col in df.columns:
        if "pretrain" in col.lower():
            pretraining_col = col
            break

    date_col = None
    for col in ["Date", "date", "DATE"]:
        if col in df.columns:
            date_col = col
            break

    fly_col = None
    for col in ["fly", "Fly", "fly_id", "Fly_ID"]:
        if col in df.columns:
            fly_col = col
            break

    ball_condition_col = None
    for col in ["ball_condition", "ball_identity"]:
        if col in df.columns:
            ball_condition_col = col
            break

    if not all([genotype_col, pretraining_col]):
        print(f"❌ Error: Could not find required columns")
        print(f"   Genotype column: {genotype_col}")
        print(f"   Pretraining column: {pretraining_col}")
        return

    if metric not in df.columns:
        print(f"❌ Error: Metric '{metric}' not found in dataset")
        print(f"   Available numeric columns: {', '.join(df.select_dtypes(include=[np.number]).columns)}")
        return

    print(f"\n📊 Detected columns:")
    print(f"   Genotype: {genotype_col}")
    print(f"   Pretraining: {pretraining_col}")
    print(f"   Date: {date_col}")
    print(f"   Fly: {fly_col}")
    print(f"   Ball condition: {ball_condition_col}")
    print(f"   Response variable: {metric}")

    # Filter for test ball only
    if ball_condition_col:
        test_ball_values = ["test", "Test", "TEST", "1", 1, "test_ball", "testball"]
        df_test = pd.DataFrame()
        for val in test_ball_values:
            subset = df[df[ball_condition_col] == val]
            if not subset.empty:
                df_test = pd.concat([df_test, subset])

        if not df_test.empty:
            df = df_test
            print(f"✓ Filtered for test ball only: {len(df)} rows")

    # Initialize LMM analysis
    output_dir = Path("/mnt/upramdya_data/MD/F1_Tracks/DDC/lmm_analysis")
    lmm = LMMAnalysis(df, metric, output_dir)

    # Prepare data
    lmm.prepare_data(genotype_col, pretraining_col, date_col, fly_col)

    # Fit the core model: Genotype * Pretraining
    print("\n" + "=" * 120)
    print("STEP 1: Fit Core Model (Genotype * Pretraining)")
    print("=" * 120)
    print("\nThis model always includes:")
    print("  • Genotype effect (main effect)")
    print("  • Pretraining effect (main effect)")
    print("  • Genotype × Pretraining interaction")
    print("  • Random effects for Date (grouping variable)")

    core_model = lmm.fit_core_model(max_iter=max_iter)

    if core_model is None:
        print("\n❌ Failed to fit core model. Exiting.")
        return

    # Check for singular covariance warning
    has_singular = core_model.get("is_singular", False)
    if has_singular:
        print("\n" + "!" * 120)
        print("⚠️  IMPORTANT: Singular random effects covariance detected!")
        print("!" * 120)
        print("\nThis means:")
        print("  • The random effect variance is essentially zero (Date/fly explain no additional variance)")
        print("  • This happens when flies are perfectly nested within dates")
        print("  • The mixed model reduces to a regular linear model (OLS)")
        print("\nRecommendations:")
        print("  1. Interpret fixed effects normally - they are still valid!")
        print("  2. Consider using OLS (Ordinary Least Squares) instead for cleaner results")
        print("  3. For power analysis, use the existing power_analysis_tnt_ddc.py script")
        print("\nTo run OLS instead:")
        print("  from statsmodels.formula.api import ols")
        print(f"  model = ols('{metric} ~ Genotype * Pretraining', data=df).fit()")
        print("!" * 120 + "\n")

    # Display model summary
    print("\n" + "=" * 120)
    print("STEP 2: Model Summary")
    print("=" * 120)
    lmm.print_best_model_summary()

    # Create visualizations
    print("\n" + "=" * 120)
    print("STEP 3: Visualizations")
    print("=" * 120)
    lmm.plot_diagnostics()
    lmm.plot_effects()

    # Export results
    print("\n" + "=" * 120)
    print("STEP 4: Export Results")
    print("=" * 120)
    lmm.export_results()

    print("\n" + "=" * 120)
    print("✅ LMM analysis complete!")
    print("=" * 120)
    print("\nKey findings to interpret:")
    print("  • Genotype[T.TNTxDDC]: Effect of TNTxDDC vs control (TNTxEmptyGal4)")
    print("  • Pretraining[T.y]: Effect of pretraining (y vs n)")
    print("  • Genotype[T.TNTxDDC]:Pretraining[T.y]: Interaction (does pretraining effect differ by genotype?)")
    print("\nInterpretation:")
    print("  • If interaction p < 0.05: The effect of pretraining DEPENDS on genotype")
    print("  • If interaction p > 0.05: Main effects can be interpreted independently")

    return lmm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Mixed-Effects Model analysis for TNT DDC experiments")
    parser.add_argument(
        "--metric", type=str, default="distance_moved", help="Response variable to analyze (default: distance_moved)"
    )
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum iterations for model fitting (default: 200)")

    args = parser.parse_args()

    main(metric=args.metric, max_iter=args.max_iter)
