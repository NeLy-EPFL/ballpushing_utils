from pathlib import Path
import pandas as pd
import Config
import os
import numpy as np
from utils_behavior import Processing, Utils
import argparse
import yaml

data_path = Utils.get_data_server()

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
                / "MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
            )

            # List of metrics to check (from summary_notebooks.py)
            metric_names = [
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
                "major_event_first",
                "cumulated_breaks_duration",
                "chamber_time",
                "chamber_ratio",
                "pushed",
                "pulled",
                "pulling_ratio",
                "success_direction",
                "interaction_proportion",
                "interaction_persistence",
                "distance_moved",
                "distance_ratio",
                "chamber_exit_time",
                "number_of_pauses",
                "total_pause_duration",
                "learning_slope",
                "learning_slope_r2",
                "logistic_L",
                "logistic_k",
                "logistic_t0",
                "logistic_r2",
                "avg_displacement_after_success",
                "avg_displacement_after_failure",
                "influence_ratio",
                "normalized_velocity",
                "velocity_during_interactions",
                "velocity_trend",
            ]

            # Subset for Nickname and its associated control
            summary_subset = Config.get_subset_data(summary_data, col="Nickname", value=nickname)
            if summary_subset.empty:
                raise ValueError(f"No summary data found for Nickname {nickname} or its control.")
            # Ensure unique index for seaborn plotting
            summary_subset = summary_subset.reset_index(drop=True)
            # Get brain region for output path
            row = split_registry[split_registry["Nickname"] == nickname]
            if row.empty:
                raise ValueError(f"Nickname {nickname} not found in registry.")
            brain_region = row["Simplified region"].iloc[0]

            # Output directory for summary metric plots
            summary_dir = output_dir / f"IDCard_SummaryMetrics_{nickname}"
            os.makedirs(summary_dir, exist_ok=True)

            # For each metric, plot jitterboxplot if significant difference
            significant_metrics = []
            for metric in metric_names:
                # Separate data for Nickname and control
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
                    stripplot = sns.stripplot(
                        data=summary_subset,
                        x=metric,
                        y="Nickname",
                        hue="Brain region",
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
                    group_labels = list(summary_subset["Nickname"].unique())
                    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
                    # Find y-positions for Nickname and Control
                    y_nickname = None
                    y_control = None
                    for y, label in zip(ax.get_yticks(), yticklabels):
                        if label == nickname:
                            y_nickname = y
                        elif label == control_data.index[0] if not control_data.empty else None:
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
                    ax.set_title(f"{metric} - {nickname} vs Control - {brain_region}", fontsize=14)
                    ax.set_xlabel(metric)
                    ax.set_ylabel("")
                    # Hide legend
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    plt.tight_layout()
                    # Save plot
                    safe_metric = re.sub(r"[^\w\-_\. ]", "_", metric)
                    plt.savefig(summary_dir / f"{safe_metric}_jitterboxplot.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

            print(f"Significant summary metrics for {nickname}: {significant_metrics}")

            # --- PCA RESULTS PANEL: SIGNIFICANT PCS/DIMS FOR NICKNAME ---
            # This section loads tailored PCA stats and metadata for static PCA, temporal FPCA, and MFA.
            # For each PCA, it checks for significant PCs for the Nickname, loads the corresponding feather dataset,
            # and generates jitterboxplots for each significant PC/dimension, saving them in a dedicated output folder.
            import ast

            # Set suffix for PCA/FPCA/MFA file loading (should match what was used in PCA scripts)
            # Set to '_emptysplit' or '_tailoredctrls' as appropriate
            pca_suffix = "_tailoredctrls"  # or "_tailoredctrls" if not using force_control

            pca_experiments = [
                {
                    "name": "static_pca",
                    "stats_csv": f"src/PCA/static_pca_stats_results_allmethods{pca_suffix}.csv",
                    "feather": f"src/PCA/static_pca_with_metadata{pca_suffix}.feather",
                    "pc_prefix": "PCA",
                },
                {
                    "name": "fpca_temporal",
                    "stats_csv": f"src/PCA/fpca_temporal_stats_results_allmethods{pca_suffix}.csv",
                    "feather": f"src/PCA/fpca_temporal_with_metadata{pca_suffix}.feather",
                    "pc_prefix": "FPCA",
                },
                {
                    "name": "mfa",
                    "stats_csv": f"src/PCA/mfa_stats_results_allmethods{pca_suffix}.csv",
                    "feather": f"src/PCA/mfa_with_metadata{pca_suffix}.feather",
                    "pc_prefix": "Dim.",
                },
            ]

            pca_dir = output_dir / f"IDCard_PCA_{nickname}"
            os.makedirs(pca_dir, exist_ok=True)

            for pca in pca_experiments:
                # Load stats CSV
                try:
                    stats = pd.read_csv(pca["stats_csv"])
                except Exception as e:
                    print(f"Could not load {pca['stats_csv']}: {e}")
                    continue
                # Find row for this Nickname
                row = stats[stats["Nickname"] == nickname]
                if row.empty:
                    print(f"No PCA stats for {nickname} in {pca['name']}")
                    continue
                # Check if any significant PCs
                if not row.iloc[0].get("MannWhitney_any_dim_significant", False):
                    print(f"No significant PCs for {nickname} in {pca['name']}")
                    continue
                # Get list of significant PCs/dims
                sig_dims = row.iloc[0].get("MannWhitney_significant_dims", [])
                if isinstance(sig_dims, str):
                    try:
                        sig_dims = ast.literal_eval(sig_dims)
                    except Exception:
                        sig_dims = []
                if not sig_dims:
                    print(f"No significant dimensions listed for {nickname} in {pca['name']}")
                    continue
                # Check feather file exists
                if not os.path.exists(pca["feather"]):
                    print(f"Feather file not found: {pca['feather']}")
                    continue
                # Load feather dataset
                try:
                    df = pd.read_feather(pca["feather"])
                except Exception as e:
                    print(f"Could not load {pca['feather']}: {e}")
                    continue
                # For each significant PC/dim, plot Nickname vs control
                for dim in sig_dims:
                    if dim not in df.columns:
                        print(f"Dimension {dim} not found in {pca['feather']}")
                        continue
                    # Subset for Nickname and control
                    subset = df[df["Nickname"].isin([nickname, row.iloc[0]["Control"]])]
                    if subset.empty:
                        print(f"No data for {nickname} and control in {pca['feather']} for {dim}")
                        continue
                    # Ensure numeric and drop NaNs for the PC/dim
                    try:
                        nickname_data = pd.to_numeric(
                            subset[subset["Nickname"] == nickname][dim], errors="coerce"
                        ).dropna()
                        control_data = pd.to_numeric(
                            subset[subset["Nickname"] == row.iloc[0]["Control"]][dim], errors="coerce"
                        ).dropna()
                        if nickname_data.empty or control_data.empty:
                            print(f"[WARN] After numeric conversion, data empty for PCA dim: {dim}")
                            continue
                        # Compute bootstrapped confidence intervals
                        nickname_ci = Processing.draw_bs_ci(
                            nickname_data.values,
                            func=np.mean,
                            rg=np.random.default_rng(),
                            n_reps=300,
                            show_progress=False,
                        )
                        control_ci = Processing.draw_bs_ci(
                            control_data.values,
                            func=np.mean,
                            rg=np.random.default_rng(),
                            n_reps=300,
                            show_progress=False,
                        )
                        # Ensure CIs are arrays of length 2
                        import numpy as np

                        if (
                            not isinstance(nickname_ci, (list, tuple, np.ndarray))
                            or len(np.atleast_1d(nickname_ci)) != 2
                        ):
                            nickname_ci = [nickname_data.mean(), nickname_data.mean()]
                        if not isinstance(control_ci, (list, tuple, np.ndarray)) or len(np.atleast_1d(control_ci)) != 2:
                            control_ci = [control_data.mean(), control_data.mean()]
                    except Exception as e:
                        print(f"[ERROR] Failed to process PCA dim '{dim}': {e}")
                        continue
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.stripplot(
                        data=subset,
                        x=dim,
                        y="Nickname",
                        hue="Brain region" if "Brain region" in subset.columns else None,
                        palette=getattr(Config, "color_dict", None),
                        dodge=False,
                        alpha=0.6,
                        jitter=True,
                        size=5,
                        ax=ax,
                    )
                    y_positions = ax.get_yticks()
                    # Defensive: ensure CIs are arrays
                    nickname_ci = np.atleast_1d(nickname_ci)
                    control_ci = np.atleast_1d(control_ci)
                    # Add bootstrapped confidence intervals as error bars
                    # Ensure correct order: Nickname first, Control second
                    group_labels = list(subset["Nickname"].unique())
                    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
                    y_nickname = None
                    y_control = None
                    for y, label in zip(ax.get_yticks(), yticklabels):
                        if label == nickname:
                            y_nickname = y
                        elif label == row.iloc[0]["Control"]:
                            y_control = y
                    y_positions = [y_nickname, y_control]
                    # Filter out None values and ensure at least two positions
                    y_positions = [y for y in y_positions if y is not None]
                    if len(y_positions) < 2:
                        y_positions = list(ax.get_yticks()[:2])
                    try:
                        ax.errorbar(
                            x=[nickname_data.mean(), control_data.mean()],
                            y=y_positions,
                            xerr=[
                                [
                                    nickname_data.mean() - float(nickname_ci[0]),
                                    control_data.mean() - float(control_ci[0]),
                                ],
                                [
                                    float(nickname_ci[1]) - nickname_data.mean(),
                                    float(control_ci[1]) - control_data.mean(),
                                ],
                            ],
                            fmt="o",
                            color="black",
                            capsize=5,
                            zorder=1,
                        )
                    except Exception as e:
                        print(f"[WARN] Could not plot error bars for PCA dim '{dim}': {e}")
                    # Add effect size annotation (optional, for consistency)
                    try:
                        effect_size, effect_size_interval = Processing.compute_effect_size(nickname_ci, control_ci)
                        ax.text(
                            0.8,
                            0.5,
                            f"Effect Size: {effect_size:.2f}\n95% CI: [{effect_size_interval[0]:.2f}, {effect_size_interval[1]:.2f}]",
                            transform=ax.transAxes,
                            fontsize=10,
                            verticalalignment="center",
                            horizontalalignment="center",
                        )
                    except Exception as e:
                        print(f"[WARN] Could not compute effect size for PCA dim '{dim}': {e}")
                    ax.set_title(f"{pca['name']} {dim} - {nickname} vs Control", fontsize=14)
                    ax.set_xlabel(dim)
                    ax.set_ylabel("")
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    plt.tight_layout()
                    safe_dim = re.sub(r"[^\w\-_\. ]", "_", dim)
                    plt.savefig(pca_dir / f"{pca['name']}_{safe_dim}_jitterboxplot.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

            print(f"PCA results panel complete for {nickname}")

            # --- UMAP PANEL: 2D scatterplot and cluster proportions if hits ---
            import seaborn as sns
            import matplotlib.pyplot as plt

            # Path to UMAP data and hits CSV (portable, relative to repo root)
            repo_root = Path(__file__).resolve().parent.parent.parent
            umap_data_path = repo_root / "tests" / "integration" / "outputs" / "umap_TNT_1.feather"
            umap_hits_path = repo_root / "outputs" / "Umap_hits.csv"

            # Load UMAP data
            try:
                umap_data = pd.read_feather(umap_data_path)
            except Exception as e:
                print(f"[UMAP] Could not load UMAP data: {e}")
                umap_data = None

            # Load UMAP hits
            try:
                umap_hits = pd.read_csv(umap_hits_path)
            except Exception as e:
                print(f"[UMAP] Could not load UMAP hits: {e}")
                umap_hits = None

            if umap_data is not None:
                # Subset for Nickname and its control
                subset_data = Config.get_subset_data(umap_data, col="Nickname", value=nickname)
                # Dynamically set the control nickname based on Brain region == "Control"
                control_rows = subset_data[subset_data["Brain region"] == "Control"]
                if not control_rows.empty:
                    control = control_rows["Nickname"].iloc[0]
                else:
                    control = "Unknown Control"
                # 2D scatterplot UMAP1 x UMAP2 with cluster background
                fig, ax = plt.subplots(figsize=(8, 6))
                # --- Load full UMAP dataset for cluster background ---
                try:
                    full_umap_path = "/home/durrieu/ballpushing_utils/tests/integration/outputs/umap_TNT_1.feather"
                    full_umap_data = pd.read_feather(full_umap_path)
                except Exception as e:
                    print(f"[WARN] Could not load full UMAP dataset for cluster background: {e}")
                    full_umap_data = None

                fig, ax = plt.subplots(figsize=(8, 6))
                # --- Draw convex hull background for each cluster using full dataset ---
                legend_handles = []
                if full_umap_data is not None and "cluster" in full_umap_data.columns:
                    from scipy.spatial import ConvexHull
                    import matplotlib.patches as mpatches

                    # Use a palette with at least 23 distinct colors
                    n_clusters = full_umap_data["cluster"].nunique()
                    if n_clusters <= 20:
                        cluster_palette = sns.color_palette("tab20", n_colors=n_clusters)
                    elif n_clusters <= 30:
                        cluster_palette = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", n_clusters - 20)
                    else:
                        cluster_palette = sns.color_palette("husl", n_colors=n_clusters)
                    cluster_color_dict = {
                        c: cluster_palette[i % len(cluster_palette)]
                        for i, c in enumerate(sorted(full_umap_data["cluster"].unique()))
                    }
                    for i, cluster_id in enumerate(sorted(full_umap_data["cluster"].unique())):
                        cluster_points = full_umap_data[full_umap_data["cluster"] == cluster_id][
                            ["UMAP1", "UMAP2"]
                        ].values
                        color = cluster_palette[i % len(cluster_palette)]
                        if len(cluster_points) >= 3:
                            try:
                                hull = ConvexHull(cluster_points)
                                polygon = mpatches.Polygon(
                                    cluster_points[hull.vertices],
                                    closed=True,
                                    facecolor=color,
                                    edgecolor=None,
                                    alpha=0.18,
                                    zorder=0,
                                )
                                ax.add_patch(polygon)
                                legend_handles.append(
                                    mpatches.Patch(
                                        facecolor=color, edgecolor="k", label=f"Cluster {cluster_id}", alpha=0.5
                                    )
                                )
                            except Exception as e:
                                print(f"[WARN] Could not plot convex hull for cluster {cluster_id}: {e}")
                # Overlay Nickname/control points colored by Brain region
                sns.scatterplot(
                    data=subset_data,
                    x="UMAP1",
                    y="UMAP2",
                    hue="Brain region" if "Brain region" in subset_data.columns else "Nickname",
                    style="Nickname" if "Nickname" in subset_data.columns else None,
                    palette=getattr(Config, "color_dict", None),
                    alpha=0.7,
                    s=40,
                    ax=ax,
                    zorder=2,
                )
                # Add cluster legend
                if legend_handles:
                    ax.legend(
                        handles=legend_handles,
                        title="Clusters",
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                        borderaxespad=0.0,
                    )
                ax.set_title(f"UMAP 2D Scatter: {nickname} vs {control}", fontsize=14)
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                # Do not remove the legend here; keep the cluster legend visible
                plt.tight_layout()
                # Save UMAP plots in the correct subdirectory of output_dir
                umap_dir = output_dir / f"IDCard_UMAP_{nickname}"
                os.makedirs(umap_dir, exist_ok=True)
                plt.savefig(umap_dir / f"umap2d_{nickname}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"UMAP 2D scatterplot saved to {umap_dir / f'umap2d_{nickname}.png'}")
                # If any hits for this nickname, plot cluster proportions
                if umap_hits is not None and nickname in umap_hits["Nickname"].values:
                    hit_row = umap_hits[umap_hits["Nickname"] == nickname].iloc[0]
                    # Prepare cluster proportions as in UMAP_plotting.py
                    fly_proportions_df = pd.DataFrame()
                    all_clusters = sorted(subset_data["cluster"].unique())
                    unique_flies = subset_data["fly"].unique()
                    for fly_id in unique_flies:
                        fly_data = subset_data[subset_data["fly"] == fly_id]
                        cluster_counts = fly_data["cluster"].value_counts()
                        cluster_counts.name = "count"
                        cluster_proportions = cluster_counts / cluster_counts.sum()
                        cluster_proportions = cluster_proportions.reset_index()
                        cluster_proportions.columns = ["cluster", "proportion"]
                        cluster_proportions["fly"] = fly_id
                        cluster_proportions["Group"] = fly_data["Brain region"].iloc[0]
                        for cluster_id in all_clusters:
                            if cluster_id not in cluster_proportions["cluster"].values:
                                cluster_proportions = pd.concat(
                                    [
                                        cluster_proportions,
                                        pd.DataFrame(
                                            {
                                                "cluster": [cluster_id],
                                                "proportion": [0],
                                                "fly": [fly_id],
                                                "Group": [fly_data["Brain region"].iloc[0]],
                                            }
                                        ),
                                    ],
                                    ignore_index=True,
                                )
                        fly_proportions_df = pd.concat([fly_proportions_df, cluster_proportions], ignore_index=True)
                    if not fly_proportions_df.empty:
                        fly_proportions_df["cluster"] = fly_proportions_df["cluster"].astype(int)
                        fly_proportions_df = fly_proportions_df.sort_values(by="cluster")
                        fig, ax = plt.subplots(figsize=(18, 8))
                        sns.set_style("whitegrid")
                        # --- 2) Draw colored cluster background bands ---
                        cluster_palette = sns.color_palette("pastel", n_colors=len(all_clusters))
                        for i, cluster_id in enumerate(all_clusters):
                            ax.axvspan(i - 0.5, i + 0.5, color=cluster_palette[i], alpha=0.18, zorder=0)
                        # --- 1) Boxplot and stripplot as before ---
                        box = sns.boxplot(
                            data=fly_proportions_df,
                            x="cluster",
                            y="proportion",
                            hue="Group",
                            palette=getattr(Config, "color_dict", None),
                            width=0.7,
                            ax=ax,
                        )
                        strip = sns.stripplot(
                            data=fly_proportions_df,
                            x="cluster",
                            y="proportion",
                            hue="Group",
                            dodge=True,
                            size=3,
                            alpha=0.8,
                            color="black",
                            linewidth=0.5,
                            ax=ax,
                        )
                        # Remove duplicate legends
                        handles, labels = ax.get_legend_handles_labels()
                        if handles:
                            ax.legend([], [], frameon=False)
                        # --- 1) Add significance annotation for hit clusters ---
                        import ast

                        sig_clusters = []
                        sig_val = hit_row["significant_clusters"]
                        # If it's a pandas Series, get the first value
                        if hasattr(sig_val, "values") and len(sig_val.values) == 1:
                            sig_val = sig_val.values[0]
                        if isinstance(sig_val, str):
                            try:
                                sig_clusters = ast.literal_eval(sig_val)
                            except Exception:
                                sig_clusters = []
                        elif isinstance(sig_val, (list, tuple, np.ndarray)):
                            sig_clusters = list(sig_val)
                        else:
                            sig_clusters = [sig_val]
                        # Flatten any nested lists/arrays and filter to ints
                        flat_sig_clusters = []
                        for s in sig_clusters:
                            if isinstance(s, (list, tuple, np.ndarray, pd.Series)):
                                flat_sig_clusters.extend([int(x) for x in np.array(s).flatten() if str(x).isdigit()])
                            else:
                                try:
                                    flat_sig_clusters.append(int(s))
                                except Exception:
                                    continue
                        for sig_int in flat_sig_clusters:
                            if sig_int in all_clusters:
                                idx = all_clusters.index(sig_int)
                                y_max = fly_proportions_df[fly_proportions_df["cluster"] == sig_int]["proportion"].max()
                                ax.text(
                                    idx,
                                    y_max + 0.05,
                                    "*",
                                    color="red",
                                    fontsize=22,
                                    fontweight="bold",
                                    ha="center",
                                    va="bottom",
                                    zorder=10,
                                )
                        for i in range(len(all_clusters) - 1):
                            ax.axvline(x=i + 0.5, color="gray", linestyle="-", alpha=0.5, linewidth=1, zorder=1)
                        ax.grid(axis="y", linestyle="--", alpha=0.7)
                        ax.grid(axis="x", visible=False)
                        ax.axhline(0.5, color="red", linestyle="--", label="Equal proportion")
                        ax.set_title(f"Cluster Proportions - {nickname} vs {control}", fontsize=14)
                        ax.set_xlabel("Cluster Number", fontsize=12)
                        ax.set_ylabel("Proportion by Cluster", fontsize=12)
                        ax.set_xticks(range(len(all_clusters)))
                        ax.set_xticklabels([str(c) for c in all_clusters], fontsize=11, fontweight="bold")
                        plt.tight_layout()
                        plt.savefig(umap_dir / f"proportions_by_cluster_{nickname}.png", dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print(
                            f"UMAP cluster proportions plot saved to {umap_dir / f'proportions_by_cluster_{nickname}.png'}"
                        )

            # --- FINAL PANEL: FLEXIBLE COMPOSITE ID CARD LAYOUT ---
            import matplotlib.image as mpimg
            from glob import glob

            def find_first_png(directory, pattern):
                files = sorted(glob(str(directory / pattern)))
                return files[0] if files else None

            # Paths to panel images
            traj_path = output_dir / f"IDCard_Trajectories_{nickname}.png"
            kde_path = output_dir / f"IDCard_KDE_ECDF_{nickname}.png"
            sum_dir = output_dir / f"IDCard_SummaryMetrics_{nickname}"
            pca_dir = output_dir / f"IDCard_PCA_{nickname}"
            umap_dir = output_dir / f"IDCard_UMAP_{nickname}"

            # Find summary metric plots (up to 3 most significant)
            sum_pngs = sorted(glob(str(sum_dir / "*_jitterboxplot.png")))[:3]
            # Find PCA/FPCA/MFA plots (all significant, not grouped)
            pca_pngs = sorted(glob(str(pca_dir / "*_jitterboxplot.png")))
            # UMAP panels
            umap2d_path = umap_dir / f"umap2d_{nickname}.png"
            proportions_path = find_first_png(umap_dir, "proportions_by_cluster_*.png")

            # Collect all available panels
            panels = []
            if traj_path.exists():
                panels.append(("Trajectories", traj_path))
            if kde_path.exists():
                panels.append(("KDE / ECDF", kde_path))
            for i, png in enumerate(sum_pngs):
                panels.append((f"Summary Metric {i+1}", png))
            # For PCA/FPCA/MFA, use filename to extract method and dim for title
            import re

            for png in pca_pngs:
                fname = os.path.basename(png)
                m = re.match(r"(static_pca|fpca_temporal|mfa)_([\w\.]+)_jitterboxplot.png", fname)
                if m:
                    method, dim = m.groups()
                    panels.append((f"{method} {dim}", png))
                else:
                    panels.append((f"PCA/FPCA/MFA", png))
            if umap2d_path.exists():
                panels.append(("UMAP 2D Scatter", umap2d_path))
            if proportions_path and os.path.exists(proportions_path):
                panels.append(("UMAP Cluster Proportions", proportions_path))

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
            fig.suptitle(f"ID Card Summary: {nickname}", fontsize=22, fontweight="bold")
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
