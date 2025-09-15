from pathlib import Path
import pandas as pd
import Config
import os
import numpy as np
import re
from utils_behavior import Processing, Utils
import argparse
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

data_path = Utils.get_data_server()


def load_nickname_mapping():
    """Load the simplified nickname mapping for visualization (same as in Mann-Whitney script)"""
    region_map_path = "/mnt/upramdya_data/MD/Region_map_250908.csv"
    print(f"📋 Loading nickname mapping from {region_map_path}")

    try:
        region_map = pd.read_csv(region_map_path)
        # Create mapping from Nickname to Simplified Nickname
        nickname_mapping = dict(zip(region_map["Nickname"], region_map["Simplified Nickname"]))
        print(f"📋 Loaded {len(nickname_mapping)} nickname mappings")

        # Also create brain region mapping for simplified nicknames
        simplified_to_region = dict(zip(region_map["Simplified Nickname"], region_map["Simplified region"]))

        return nickname_mapping, simplified_to_region
    except Exception as e:
        print(f"⚠️  Could not load region mapping: {e}")
        return {}, {}


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate ID card plots for nicknames.")
    parser.add_argument("--nickname", type=str, default=None, help="Process only this Nickname (exact match)")
    parser.add_argument("--nickname-yaml", type=str, default=None, help="YAML file with a list of nicknames to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plots if they exist")
    args = parser.parse_args()

    # Set your data_path and output root here
    data_path = (
        Utils.get_data_server()
        / "MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240120_short_contacts_no_cutoff_no_downsample_Data/coordinates_regions"
    )
    output_root = Utils.get_data_server() / "MD/Ballpushing_TNTScreen/Plots/ID_cards"

    registries = Config.registries
    split_registry = Config.SplitRegistry
    all_nicknames = split_registry["Nickname"].unique()

    # Determine nicknames to process
    if args.nickname_yaml is not None:
        with open(args.nickname_yaml, "r") as f:
            yaml_nicknames = yaml.safe_load(f)
        if not isinstance(yaml_nicknames, list):
            print(f"[ERROR] YAML file {args.nickname_yaml} does not contain a list.")
            exit(1)
        # Optionally filter to only those in the registry
        nicknames_to_process = [n for n in yaml_nicknames if n in all_nicknames]
        missing = [n for n in yaml_nicknames if n not in all_nicknames]
        if missing:
            print(f"[WARN] The following nicknames from YAML are not in the registry and will be skipped: {missing}")
        if not nicknames_to_process:
            print(f"[ERROR] No valid nicknames found in YAML file.")
            exit(1)
    elif args.nickname is not None:
        if args.nickname not in all_nicknames:
            print(f"[ERROR] Nickname '{args.nickname}' not found in registry. Available: {list(all_nicknames)}")
            exit(1)
        nicknames_to_process = [args.nickname]
    else:
        nicknames_to_process = all_nicknames

    for nickname in nicknames_to_process:
        try:
            # Check if composite plot already exists
            row = split_registry[split_registry["Nickname"] == nickname]
            if row.empty:
                print(f"[WARN] Nickname {nickname} not found in registry. Skipping.")
                continue
            brain_region = row["Simplified region"].iloc[0]
            output_dir = output_root / str(brain_region) / str(nickname)
            final_path = output_dir / f"IDCard_{nickname}_summary.png"
            if final_path.exists() and not args.overwrite:
                print(f"[SKIP] Composite plot already exists for {nickname}: {final_path}. Use --overwrite to remake.")
                continue
            print(f"\n=== Processing Nickname: {nickname} ===")
            # Find brain region for this nickname
            row = split_registry[split_registry["Nickname"] == nickname]
            if row.empty:
                print(f"[WARN] Nickname {nickname} not found in registry. Skipping.")
                continue
            brain_region = row["Simplified region"].iloc[0]

            # Set output_dir for this nickname
            output_dir = output_root / str(brain_region) / str(nickname)
            output_dir.mkdir(parents=True, exist_ok=True)
            # # Set parameters
            # nickname = "854 (OK107-Gal4)"
            # data_path = Path(
            #     "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240120_short_contacts_no_cutoff_no_downsample_Data/coordinates_regions"
            # )
            # output_dir = Path("")
            # output_dir.mkdir(exist_ok=True, parents=True)

            # registries = Config.registries
            # split_registry = Config.SplitRegistry

            # # Pick a random nickname if not set
            # if nickname is None:
            #     available_nicknames = split_registry["Nickname"].unique()
            #     nickname = np.random.choice(available_nicknames)
            #     print(f"Randomly selected Nickname: {nickname}")

            # else:
            #     print(f"Using specified Nickname: {nickname}")

            # Find the brain region for this nickname
            row = split_registry[split_registry["Nickname"] == nickname]
            if row.empty:
                raise ValueError(f"Nickname {nickname} not found in registry.")
            brain_region = row["Simplified region"].iloc[0]

            # Load the data for this brain region (and control)
            BallTrajectories = Config.load_datasets_for_brain_region(brain_region, data_path, registries)
            if BallTrajectories.empty:
                raise ValueError(f"No data found for brain region {brain_region}")

            # Filter for the Nickname and its associated control
            subset_data = Config.get_subset_data(BallTrajectories, col="Nickname", value=nickname)
            if subset_data.empty:
                raise ValueError(f"No data found for Nickname {nickname} or its control.")

            # Downsample the data
            subset_data = (
                subset_data.groupby("fly", group_keys=False).apply(lambda df: df.iloc[::290, :]).reset_index(drop=True)
            )

            # Output path for the plot
            output_path = output_dir / f"IDCard_Trajectories_{nickname}.png"

            # Plot using your existing function (single nickname mode)
            Config.create_and_save_plot(
                subset_data, [nickname], brain_region, output_path, registries, show_signif=True, test_nickname=nickname
            )

            print(f"Trajectory plot saved to {output_path}")

            # Events
            events_data = pd.read_feather(
                Utils.get_data_server()
                / "MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/event_metrics/pooled_event_metrics.feather"
            )

            # Subset events_data for the Nickname and its associated control
            events_subset = Config.get_subset_data(events_data, col="Nickname", value=nickname)
            if events_subset.empty:
                raise ValueError(f"No event data found for Nickname {nickname} or its control.")

            # Output path for the KDE/ECDF plot
            output_path_kde_ecdf = output_dir / f"IDCard_KDE_ECDF_{nickname}.png"

            # Get the brain region for the Nickname
            row = split_registry[split_registry["Nickname"] == nickname]
            if row.empty:
                raise ValueError(f"Nickname {nickname} not found in registry.")
            brain_region = row["Simplified region"].iloc[0]

            # Generate KDE and ECDF plots
            Config.create_and_save_kde_ecdf_plot(
                events_subset, [nickname], brain_region, output_path_kde_ecdf, registries
            )

            print(f"KDE/ECDF plot saved to {output_path_kde_ecdf}")

            # --- SUMMARY METRICS PANEL: JITTERBOXPLOTS FOR SIGNIFICANT METRICS ---
            import re
            from utils_behavior import Processing

            # Load summary metrics dataset (as in summary_notebooks.py)
            summary_data = pd.read_feather(
                Utils.get_data_server()
                / "MD/Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather"
            )

            # Add simplified nickname mappings (same as Mann-Whitney script)
            print("📋 Adding simplified nickname mappings...")
            nickname_mapping, simplified_to_region = load_nickname_mapping()
            if nickname_mapping:
                print("📋 Applying nickname mappings to dataset...")
                summary_data["Simplified Nickname"] = summary_data["Nickname"].map(nickname_mapping)
                print("📋 Applying brain region mappings...")
                summary_data["Simplified region"] = summary_data["Simplified Nickname"].map(simplified_to_region)

                # Report mapping success
                print("📋 Calculating mapping statistics...")
                mapped_count = summary_data["Simplified Nickname"].notna().sum()
                total_count = len(summary_data)
                print(
                    f"📋 Mapped {mapped_count}/{total_count} flies to simplified nicknames ({mapped_count/total_count*100:.1f}%)"
                )

                if mapped_count < total_count:
                    unmapped_nicknames = summary_data[summary_data["Simplified Nickname"].isna()]["Nickname"].unique()
                    print(f"⚠️  Unmapped nicknames: {list(unmapped_nicknames)}")
            else:
                print(f"⚠️  Could not load nickname mapping, using original nicknames")
                summary_data["Simplified Nickname"] = summary_data["Nickname"]
                summary_data["Simplified region"] = summary_data["Brain region"]

            # Load metrics from the same file as used in Mann-Whitney analysis
            metrics_file = "/home/matthias/ballpushing_utils/src/PCA/metrics_lists/final_metrics_for_pca_alt.txt"
            try:
                with open(metrics_file, "r") as f:
                    all_metric_names = [line.strip() for line in f if line.strip()]
                print(f"📋 Loaded {len(all_metric_names)} metrics from {metrics_file}")
            except Exception as e:
                print(f"⚠️  Could not load metrics file: {e}, using fallback metrics")
                # Fallback to core metrics if file loading fails
                all_metric_names = [
                    "nb_events",
                    "max_event",
                    "max_event_time",
                    "max_distance",
                    "final_event",
                    "final_event_time",
                    "nb_significant_events",
                    "significant_ratio",
                    "first_significant_event",
                    "first_significant_event_time",
                    "major_event",
                    "first_major_event_time",
                    "pulled",
                    "pulling_ratio",
                    "interaction_proportion",
                    "interaction_persistence",
                    "distance_moved",
                    "distance_ratio",
                    "chamber_exit_time",
                    "normalized_velocity",
                    "has_finished",
                    "has_major",
                    "has_significant",
                ]

            # Filter out binned metrics to avoid cluttering (same patterns as Mann-Whitney analysis)
            excluded_patterns = [
                "binned_",
                "r2",
                "slope",
                "_bin_",
                "logistic_",
                "learning_",
                "interaction_rate_bin",
                "binned_auc",
                "binned_slope",
            ]

            # Filter metrics by excluding problematic patterns
            metric_names = []
            for metric in all_metric_names:
                # Skip if it matches any excluded pattern
                skip_metric = False
                for pattern in excluded_patterns:
                    if pattern in metric:
                        skip_metric = True
                        break
                if skip_metric:
                    continue
                metric_names.append(metric)

            print(
                f"📊 Using {len(metric_names)} filtered metrics for ID card (excluded binned and problematic metrics)"
            )
            if len(metric_names) != len(all_metric_names):
                excluded_count = len(all_metric_names) - len(metric_names)
                print(f"   Excluded {excluded_count} metrics to avoid cluttering")

            # Subset for Nickname and its associated control
            summary_subset = Config.get_subset_data(summary_data, col="Nickname", value=nickname)
            if summary_subset.empty:
                raise ValueError(f"No summary data found for Nickname {nickname} or its control.")
            # Ensure unique index for seaborn plotting
            summary_subset = summary_subset.reset_index(drop=True)

            # Now filter the metrics based on what's available in the actual data
            available_metric_names = []
            for metric in metric_names:
                if metric not in summary_subset.columns:
                    continue
                # Skip if it's non-numeric
                if not pd.api.types.is_numeric_dtype(summary_subset[metric]):
                    continue
                available_metric_names.append(metric)

            metric_names = available_metric_names
            print(f"📊 Final metrics for ID card analysis: {len(metric_names)} metrics")

            # Get brain region for output path
            row = split_registry[split_registry["Nickname"] == nickname]
            if row.empty:
                raise ValueError(f"Nickname {nickname} not found in registry.")
            brain_region = row["Simplified region"].iloc[0]

            # Output directory for summary metric plots
            summary_dir = output_dir / f"IDCard_SummaryMetrics_{nickname}"
            os.makedirs(summary_dir, exist_ok=True)

            # Get simplified nickname for this genotype if available
            simplified_nickname = None
            if "Simplified Nickname" in summary_subset.columns:
                simplified_mapping = summary_subset[summary_subset["Nickname"] == nickname]["Simplified Nickname"]
                if not simplified_mapping.empty:
                    simplified_nickname = simplified_mapping.iloc[0]

            # For each metric, plot jitterboxplot if significant difference
            significant_metrics = []

            for metric in metric_names:
                # Separate data for Nickname and control using original nickname
                nickname_data = summary_subset[summary_subset["Nickname"] == nickname][metric].dropna()
                control_data = summary_subset[summary_subset["Nickname"] != nickname][metric].dropna()
                if nickname_data.empty or control_data.empty:
                    continue
                # Try to ensure numeric type and catch errors
                try:
                    nickname_data = pd.to_numeric(nickname_data, errors="coerce").dropna()
                    control_data = pd.to_numeric(control_data, errors="coerce").dropna()
                    if nickname_data.empty or control_data.empty:
                        print(f"[WARN] After numeric conversion, data empty for metric: {metric}")
                        continue
                    # Compute bootstrapped confidence intervals
                    nickname_ci = Processing.draw_bs_ci(
                        nickname_data.values, func=np.mean, rg=np.random.default_rng(), n_reps=300, show_progress=False
                    )
                    control_ci = Processing.draw_bs_ci(
                        control_data.values, func=np.mean, rg=np.random.default_rng(), n_reps=300, show_progress=False
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process metric '{metric}': {e}")
                    continue
                # Ensure CIs are arrays of length 2
                import numpy as np

                if not isinstance(nickname_ci, (list, tuple, np.ndarray)) or len(np.atleast_1d(nickname_ci)) != 2:
                    nickname_ci = [nickname_data.mean(), nickname_data.mean()]
                if not isinstance(control_ci, (list, tuple, np.ndarray)) or len(np.atleast_1d(control_ci)) != 2:
                    control_ci = [control_data.mean(), control_data.mean()]
                # Compute effect size and CI
                try:
                    effect_size, effect_size_interval = Processing.compute_effect_size(nickname_ci, control_ci)
                except Exception as e:
                    print(f"[ERROR] Effect size computation failed for metric '{metric}': {e}")
                    continue
                # Only plot if effect size CI does not include zero
                if (effect_size_interval[0] > 0 and effect_size_interval[1] > 0) or (
                    effect_size_interval[0] < 0 and effect_size_interval[1] < 0
                ):
                    significant_metrics.append(metric)
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    fig, ax = plt.subplots(figsize=(8, 3))

                    # Use simplified nicknames and brain region coloring like Mann-Whitney script
                    y_col = "Simplified Nickname" if "Simplified Nickname" in summary_subset.columns else "Nickname"
                    hue_col = "Simplified region" if "Simplified region" in summary_subset.columns else "Brain region"

                    stripplot = sns.stripplot(
                        data=summary_subset,
                        x=metric,
                        y=y_col,
                        hue=hue_col,
                        palette=Config.color_dict,
                        dodge=False,
                        alpha=0.6,
                        jitter=True,
                        size=5,
                        ax=ax,
                    )
                    y_positions = stripplot.get_yticks()
                    # Add bootstrapped confidence intervals as error bars
                    # Ensure correct order: Nickname first, Control second
                    yticklabels = [t.get_text() for t in ax.get_yticklabels()]

                    # Find y-positions for Nickname and Control
                    # Use simplified nickname for matching if available
                    target_nickname = simplified_nickname if simplified_nickname else nickname
                    y_nickname = None
                    y_control = None
                    for y, label in zip(ax.get_yticks(), yticklabels):
                        if label == target_nickname:
                            y_nickname = y
                        elif label != target_nickname:  # Any other label is control
                            y_control = y
                    # Fallback: use first two y-positions
                    y_positions = [y_nickname, y_control]
                    # Filter out None values and ensure at least two positions
                    y_positions = [y for y in y_positions if y is not None]
                    if len(y_positions) < 2:
                        y_positions = list(ax.get_yticks()[:2])
                    ax.errorbar(
                        x=[nickname_data.mean(), control_data.mean()],
                        y=y_positions,
                        xerr=[
                            [nickname_data.mean() - nickname_ci[0], control_data.mean() - control_ci[0]],
                            [nickname_ci[1] - nickname_data.mean(), control_ci[1] - control_data.mean()],
                        ],
                        fmt="o",
                        color="black",
                        capsize=5,
                        zorder=1,
                    )
                    # Add effect size annotation
                    ax.text(
                        0.8,
                        0.5,
                        f"Effect Size: {effect_size:.2f}\n95% CI: [{effect_size_interval[0]:.2f}, {effect_size_interval[1]:.2f}]",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                    # Add a red * for significance
                    ax.text(
                        0.98, 0.9, "*", color="red", fontsize=18, fontweight="bold", transform=ax.transAxes, ha="right"
                    )
                    # Create title showing both original nickname and simplified nickname if different
                    title_nickname = (
                        f"{simplified_nickname} ({nickname})"
                        if simplified_nickname and simplified_nickname != nickname
                        else nickname
                    )
                    ax.set_title(f"{metric} - {title_nickname} vs Control - {brain_region}", fontsize=14)
                    ax.set_xlabel(metric)
                    ax.set_ylabel("")

                    # Color y-axis labels by brain region (like Mann-Whitney script)
                    if "Simplified region" in summary_subset.columns:
                        for tick in ax.get_yticklabels():
                            tick_text = tick.get_text()
                            # Find brain region for this simplified nickname
                            tick_data = summary_subset[summary_subset[y_col] == tick_text]
                            if not tick_data.empty:
                                brain_region_for_tick = tick_data[hue_col].iloc[0]
                                if brain_region_for_tick in Config.color_dict:
                                    tick.set_color(Config.color_dict[brain_region_for_tick])

                    # Hide legend
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    plt.tight_layout()
                    # Save plot
                    safe_metric = re.sub(r"[^\w\-_\. ]", "_", metric)
                    plt.savefig(summary_dir / f"{safe_metric}_jitterboxplot.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

            print(f"Significant summary metrics for {nickname}: {significant_metrics}")

            # --- PCA HIT INFORMATION ---
            # Load best PCA analysis results to check if this nickname is a hit
            pca_hit_status = "Not analyzed"
            pca_hit_details = ""

            try:
                pca_stats_path = "/home/matthias/ballpushing_utils/src/PCA/best_pca_analysis/best_pca_stats_results.csv"
                pca_stats = pd.read_csv(pca_stats_path)

                # Check if nickname exists in the PCA results (match by genotype column)
                pca_row = pca_stats[pca_stats["genotype"] == nickname]
                if not pca_row.empty:
                    pca_data = pca_row.iloc[0]

                    # Check different significance criteria
                    is_mannwhitney_hit = pca_data.get("MannWhitney_any_dim_significant", False)
                    is_permutation_hit = pca_data.get("Permutation_FDR_significant", False)
                    is_mahalanobis_hit = pca_data.get("Mahalanobis_FDR_significant", False)

                    # Get significant dimensions if available
                    sig_dims = pca_data.get("MannWhitney_significant_dims", "[]")
                    if isinstance(sig_dims, str) and sig_dims.startswith("["):
                        try:
                            import ast

                            sig_dims_list = ast.literal_eval(sig_dims)
                            num_sig_dims = len(sig_dims_list) if sig_dims_list else 0
                        except:
                            num_sig_dims = pca_data.get("num_significant_PCs", 0)
                    else:
                        num_sig_dims = pca_data.get("num_significant_PCs", 0)

                    # Determine overall hit status
                    hit_methods = []
                    if is_mannwhitney_hit:
                        hit_methods.append("Mann-Whitney")
                    if is_permutation_hit:
                        hit_methods.append("Permutation")
                    if is_mahalanobis_hit:
                        hit_methods.append("Mahalanobis")

                    if hit_methods:
                        pca_hit_status = f"HIT ({', '.join(hit_methods)})"
                        if num_sig_dims > 0:
                            pca_hit_details = f"Significant PCs: {num_sig_dims}"
                        else:
                            pca_hit_details = "Multivariate significance detected"
                    else:
                        pca_hit_status = "Not significant"
                        pca_hit_details = "No significant behavioral differences detected"
                else:
                    pca_hit_status = "Not found in PCA analysis"
                    pca_hit_details = "Genotype not included in best PCA analysis"

                print(f"PCA analysis status for {nickname}: {pca_hit_status} - {pca_hit_details}")

            except Exception as e:
                print(f"Could not load PCA hit information: {e}")
                pca_hit_status = "Error loading PCA results"
                pca_hit_details = str(e)
            # --- FINAL PANEL: FLEXIBLE COMPOSITE ID CARD LAYOUT ---
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            from glob import glob

            def find_first_png(directory, pattern):
                files = sorted(glob(str(directory / pattern)))
                return files[0] if files else None

            # Paths to panel images
            traj_path = output_dir / f"IDCard_Trajectories_{nickname}.png"
            kde_path = output_dir / f"IDCard_KDE_ECDF_{nickname}.png"
            sum_dir = output_dir / f"IDCard_SummaryMetrics_{nickname}"

            # Find summary metric plots (up to 3 most significant)
            sum_pngs = sorted(glob(str(sum_dir / "*_jitterboxplot.png")))[:3]

            # Collect all available panels
            panels = []
            if traj_path.exists():
                panels.append(("Trajectories", traj_path))
            if kde_path.exists():
                panels.append(("KDE / ECDF", kde_path))
            for i, png in enumerate(sum_pngs):
                panels.append((f"Summary Metric {i+1}", png))

            n_panels = len(panels)
            # Choose layout: single column for <=4, 2-column grid for more
            if n_panels <= 4:
                ncols = 1
                nrows = n_panels
                figsize = (10, 4 * n_panels)
            else:
                ncols = 2
                nrows = (n_panels + 1) // 2
                figsize = (16, 4 * nrows)

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            axes = axes.flatten()
            for ax, (title, img_path) in zip(axes, panels):
                ax.imshow(mpimg.imread(img_path))
                ax.set_title(title, fontsize=15)
                ax.axis("off")
            # Hide unused axes
            for ax in axes[len(panels) :]:
                ax.axis("off")
            # Create a more informative title with PCA hit information
            # Show both simplified and original nickname if different
            if simplified_nickname and simplified_nickname != nickname:
                title_main = f"ID Card Summary: {simplified_nickname} ({nickname})"
            else:
                title_main = f"ID Card Summary: {nickname}"

            title_pca = f"PCA Analysis: {pca_hit_status}"
            if pca_hit_details:
                title_pca += f" ({pca_hit_details})"

            fig.suptitle(f"{title_main}\n{title_pca}", fontsize=20, fontweight="bold")
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            final_path = output_dir / f"IDCard_{nickname}_summary.png"
            plt.savefig(final_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Composite ID card summary saved to {final_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process {nickname}: {e}")
            import traceback

            traceback.print_exc()
            continue
