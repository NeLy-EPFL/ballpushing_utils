#!/usr/bin/env python3
"""
Correlation analysis script for ballpushing metrics.
This script analyzes the correlation between different metrics to identify
redundant features before adding them to the PCA pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Family assignment helpers

FAMILY_KEYWORDS = {
    "timing": [
        "time",
        "duration",
        "_time",
        "latency",
        "first_",
        "final_",
        "exit_time",
        "first_major_event",
        "first_significant_event",
        "stop",
        "pause",
    ],
    "intensity": [
        "nb_",
        "count",
        "magnitude",
        "max_",
        "major_event",
        "distance_moved",
        "max_distance",
        "normalized_velocity",
        "velocity",
        "speed",
    ],
    "rates_ratios": [
        "rate",
        "ratio",
        "_ratio",
        "interaction_rate",
        "pulling_ratio",
        "significant_ratio",
        "distance_ratio",
        "head_pushing_ratio",
    ],
    "state_success": [
        "has_",
        "finished",
        "pulled",
        "success",
        "state",
        "completed",
    ],
    "persistence_dynamics": [
        "persistence",
        "learning",
        "trend",
        "auc",
    ],
    "orientation_kinematics": [
        "facing",
        "flailing",
        "head",
    ],
}


def assign_family(metric_name: str) -> str:
    name = metric_name.lower()
    for fam, keys in FAMILY_KEYWORDS.items():
        if any(k in name for k in keys):
            return fam
    return "other"


def group_metrics_by_family(metrics: list[str]) -> dict[str, list[str]]:
    fam_map: dict[str, list[str]] = {}
    for m in metrics:
        fam = assign_family(m)
        fam_map.setdefault(fam, []).append(m)
    return fam_map


def load_metrics_data(file_path):
    """Load the metrics dataset - NaN conversion is now handled by pre-cleaned dataset."""
    try:
        dataset = pd.read_feather(file_path)
        print(f"✓ Successfully loaded dataset with shape: {dataset.shape}")
        needs_update = False
        # Only add columns if they do not already exist
        if "has_major" not in dataset.columns:
            if "first_major_event" in dataset.columns:
                dataset["has_major"] = (~dataset["first_major_event"].isnull()).astype(int)
            else:
                dataset["has_major"] = 0
            needs_update = True
        if "has_significant" not in dataset.columns:
            if "first_significant_event" in dataset.columns:
                dataset["has_significant"] = (~dataset["first_significant_event"].isnull()).astype(int)
            else:
                dataset["has_significant"] = 0
            needs_update = True
        # Save updated dataset if new columns were added
        if needs_update:
            try:
                dataset.reset_index(drop=True, inplace=True)
                dataset.to_feather(file_path)
                print(f"✓ Updated dataset saved with new columns: {file_path}")
            except Exception as save_e:
                print(f"✗ Error saving updated dataset: {save_e}")
        return dataset
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def identify_metric_columns(dataset):
    """Identify which columns are metrics vs metadata."""
    potential_metrics = []
    base_metrics = [
        "nb_events",
        "max_event",
        "max_event_time",
        "max_distance",
        "final_event",
        "final_event_time",
        "nb_significant_events",
        "significant_ratio",
        "first_major_event",
        "first_major_event_time",
        "major_event",
        "major_event_time",
        "pulled",
        "pulling_ratio",
        "interaction_persistence",
        "distance_moved",
        "distance_ratio",
        "chamber_exit_time",
        "normalized_velocity",
        "auc",
        "overall_interaction_rate",
    ]
    for metric in base_metrics:
        if metric in dataset.columns:
            potential_metrics.append(metric)
    additional_patterns = [
        "velocity",
        "speed",
        "pause",
        "freeze",
        "persistence",
        "learning",
        "logistic",
        "influence",
        "trend",
        "interaction_rate",
        "finished",
        "chamber",
        "stop",
        "facing",
        "flailing",
        "head",
        "median_",
        "mean_",
        "has_",
    ]
    for pattern in additional_patterns:
        pattern_cols = [col for col in dataset.columns if pattern in col.lower()]
        potential_metrics.extend(pattern_cols)
    potential_metrics = sorted(list(set(potential_metrics)))
    metadata_keywords = [
        "nickname",
        "genotype",
        "condition",
        "experiment",
        "date",
        "path",
        "file",
        "id",
        "index",
        "fly_idx",
        "ball_idx",
    ]
    excluded_patterns = [
        "binned_",
        "_bin_",
        "r2",
        "slope",
        "logistic_",
        "logistic_L",
        "logistic_k",
        "logistic_t0",
        "logistic_r2",
        # "velocity_trend",
        # "overall_interaction_rate",
        # "max_distance",
        # "nb_significant_events",
        # "pulled",
        # "chamber_time",
        "raw_pauses",
    ]
    actual_metrics = []
    for col in potential_metrics:
        if any(keyword in col.lower() for keyword in metadata_keywords):
            continue
        if any(pattern in col.lower() for pattern in excluded_patterns):
            continue
        actual_metrics.append(col)
    return actual_metrics


def analyze_missing_values_by_nickname(dataset, metric_columns, nan_threshold=0.05):
    print("\n" + "=" * 80)
    print("PCA_STATIC FILTERING BIAS ANALYSIS")
    print("=" * 80)

    # Extract metrics data with nickname
    if "Nickname" not in dataset.columns:
        print("❌ 'Nickname' column not found in dataset. Skipping nickname-based analysis.")
        return None

    analysis_data = dataset[["Nickname"] + metric_columns].copy()

    # Convert boolean columns to numeric
    for col in metric_columns:
        if analysis_data[col].dtype == "bool":
            analysis_data[col] = analysis_data[col].astype(int)

    print(f"📊 Analyzing PCA_Static filtering method on {len(analysis_data['Nickname'].unique())} nicknames...")

    # STEP 1: Apply metric filtering (same as PCA_Static)
    print(f"\n🔍 STEP 1: METRIC FILTERING (remove metrics with >{nan_threshold*100}% missing)")
    print("-" * 70)

    metrics_data = analysis_data[metric_columns]
    total_rows = len(metrics_data)
    missing_counts = metrics_data.isnull().sum()
    missing_percentages = (missing_counts / total_rows) * 100

    # Filter metrics (same logic as PCA_Static)
    valid_metrics = missing_percentages[missing_percentages <= (nan_threshold * 100)].index.tolist()
    excluded_metrics = missing_percentages[missing_percentages > (nan_threshold * 100)].index.tolist()

    print(f"   • Original metrics: {len(metric_columns)}")
    print(f"   • Valid metrics (≤{nan_threshold*100}% missing): {len(valid_metrics)}")
    print(f"   • Excluded metrics (>{nan_threshold*100}% missing): {len(excluded_metrics)}")

    if excluded_metrics:
        print(f"\n❌ EXCLUDED METRICS:")
        for metric in excluded_metrics:
            percentage = missing_percentages[metric]
            print(f"   • {metric} ({percentage:.1f}% missing)")

    # STEP 2: Apply row filtering and analyze nickname bias
    print(f"\n🔍 STEP 2: ROW FILTERING BIAS ANALYSIS")
    print("-" * 50)
    print(f"Removing flies with ANY missing values in {len(valid_metrics)} valid metrics...")

    # For each nickname, calculate retention after row filtering
    nickname_impact = []

    for nickname in sorted(analysis_data["Nickname"].unique()):
        nickname_data = analysis_data[analysis_data["Nickname"] == nickname]
        n_total = len(nickname_data)

        # Count complete rows (no NaN in valid metrics)
        if len(valid_metrics) > 0:
            complete_mask = ~nickname_data[valid_metrics].isnull().any(axis=1)
            n_complete = complete_mask.sum()
        else:
            n_complete = n_total  # If no valid metrics, keep all rows

        retention_rate = (n_complete / n_total) * 100 if n_total > 0 else 0
        flies_lost = n_total - n_complete

        nickname_impact.append(
            {
                "nickname": nickname,
                "total_flies": n_total,
                "complete_flies": n_complete,
                "retention_rate": retention_rate,
                "flies_lost": flies_lost,
            }
        )

    impact_df = pd.DataFrame(nickname_impact)

    # Display results
    print(f"\n📋 NICKNAME RETENTION AFTER PCA_STATIC FILTERING:")
    print("-" * 70)
    print(f"{'Nickname':<25} | {'Total':<6} | {'Complete':<9} | {'Retention%':<11} | {'Lost':<5}")
    print("-" * 70)

    for nickname in sorted(impact_df["nickname"].unique()):
        row_data = impact_df[impact_df["nickname"] == nickname].iloc[0]
        total_flies = row_data["total_flies"]
        complete_flies = row_data["complete_flies"]
        retention_rate = row_data["retention_rate"]
        flies_lost = row_data["flies_lost"]

        # Flag problematic retention rates
        if retention_rate == 0:
            status = "❌"  # Completely lost
        elif retention_rate < 50:
            status = "⚠️ "  # Low retention
        else:
            status = "✓ "  # Good retention

        print(
            f"{status}{nickname:<24} | {total_flies:<6} | {complete_flies:<9} | "
            f"{retention_rate:<10.1f}% | {flies_lost:<5}"
        )

    # Summary statistics
    completely_lost = impact_df[impact_df["complete_flies"] == 0]
    low_retention = impact_df[impact_df["retention_rate"] < 50]
    good_retention = impact_df[impact_df["retention_rate"] >= 80]

    total_flies_before = impact_df["total_flies"].sum()
    total_flies_after = impact_df["complete_flies"].sum()
    overall_retention = (total_flies_after / total_flies_before) * 100

    print(f"\n� PCA_STATIC FILTERING IMPACT SUMMARY:")
    print("-" * 50)
    print(f"• Total nicknames analyzed: {len(impact_df)}")
    print(f"• Nicknames completely lost (0% retention): {len(completely_lost)}")
    print(f"• Nicknames with low retention (<50%): {len(low_retention)}")
    print(f"• Nicknames with good retention (≥80%): {len(good_retention)}")
    print(f"• Overall flies before filtering: {total_flies_before:,}")
    print(f"• Overall flies after filtering: {total_flies_after:,}")
    print(f"• Overall retention rate: {overall_retention:.1f}%")

    # Detailed bias assessment
    print(f"\n⚖️  BIAS RISK ASSESSMENT:")
    print("-" * 40)

    if len(completely_lost) > 0:
        print(f"❌ CRITICAL BIAS RISK: {len(completely_lost)} genotype(s) completely excluded!")
        print(f"   These nicknames will have NO representation in PCA:")
        for nickname in completely_lost["nickname"]:
            total = impact_df[impact_df["nickname"] == nickname]["total_flies"].iloc[0]
            print(f"   • {nickname} (lost all {total} flies)")

    if len(low_retention) > 0:
        print(f"\n⚠️  MODERATE BIAS RISK: {len(low_retention)} genotype(s) with low retention!")
        print(f"   These nicknames may be underrepresented in PCA:")
        for nickname in low_retention["nickname"]:
            row_data = impact_df[impact_df["nickname"] == nickname].iloc[0]
            if row_data["complete_flies"] > 0:  # Don't repeat completely lost ones
                print(
                    f"   • {nickname}: {row_data['retention_rate']:.1f}% retention ({row_data['complete_flies']}/{row_data['total_flies']})"
                )

    if len(completely_lost) == 0 and len(low_retention) <= 2:
        print(f"✅ LOW BIAS RISK: Most genotypes have good retention rates")
        print(f"   The filtering appears to affect nicknames relatively evenly.")

    # Distribution analysis
    retention_rates = impact_df["retention_rate"]
    print(f"\n📈 RETENTION RATE DISTRIBUTION:")
    print(f"   • Mean retention: {retention_rates.mean():.1f}%")
    print(f"   • Median retention: {retention_rates.median():.1f}%")
    print(f"   • Min retention: {retention_rates.min():.1f}%")
    print(f"   • Max retention: {retention_rates.max():.1f}%")
    print(f"   • Standard deviation: {retention_rates.std():.1f}%")

    # Final recommendation
    print(f"\n💡 RECOMMENDATION:")
    if len(completely_lost) > 0:
        print(f"   🚨 HIGH BIAS: Complete loss of genotypes will bias results!")
        print(f"   Consider alternative strategies:")
        print(f"   • Use imputation for missing values")
        print(f"   • Relax the 5% missing value threshold")
        print(f"   • Exclude problematic metrics instead of genotypes")
    elif len(low_retention) > 3:
        print(f"   ⚠️  MODERATE BIAS: Several genotypes underrepresented")
        print(f"   Monitor results for potential bias effects")
    else:
        print(f"   ✅ ACCEPTABLE: PCA_Static filtering method shows low bias risk")
        print(f"   The current approach appears to sample genotypes fairly")

    return {
        "filtering_impact": impact_df,
        "valid_metrics": valid_metrics,
        "excluded_metrics": excluded_metrics,
        "completely_lost": completely_lost["nickname"].tolist() if len(completely_lost) > 0 else [],
        "low_retention": low_retention["nickname"].tolist() if len(low_retention) > 0 else [],
        "overall_retention": overall_retention,
        "retention_stats": {
            "mean": retention_rates.mean(),
            "median": retention_rates.median(),
            "min": retention_rates.min(),
            "max": retention_rates.max(),
            "std": retention_rates.std(),
        },
    }


def analyze_correlations(dataset, metric_columns, correlation_threshold=0.8, nan_threshold=0.05):
    print(f"\nAnalyzing correlations for {len(metric_columns)} metrics (using Spearman)...")
    metrics_data = dataset[metric_columns].copy()
    for col in metrics_data.columns:
        if metrics_data[col].dtype == "bool":
            metrics_data[col] = metrics_data[col].astype(int)

    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)

    # === DEPENDENT_OF mapping here ===
    DEPENDENT_OF = {
        "final_event": "has_finished",
        "final_event_time": "has_finished",
        "max_event": "has_finished",
        "max_event_time": "has_finished",
        "first_major_event": "has_major",
        "first_major_event_time": "has_major",
        "first_significant_event": "has_significant",
        "first_significant_event_time": "has_significant",
    }

    # conditional missingness
    missing_percentages = {}
    missing_counts = {}
    for metric in metric_columns:
        if metric in DEPENDENT_OF and DEPENDENT_OF[metric] in dataset.columns:
            dep_col = DEPENDENT_OF[metric]
            valid_mask = dataset[dep_col] == True
            if valid_mask.sum() > 0:
                # Calculate conditional missing percentage
                conditional_miss_pct = dataset.loc[valid_mask, metric].isnull().mean() * 100
                conditional_miss_cnt = dataset.loc[valid_mask, metric].isnull().sum()

                # BUT ALSO calculate overall missing percentage for filtering decisions
                overall_miss_pct = dataset[metric].isnull().mean() * 100
                overall_miss_cnt = dataset[metric].isnull().sum()

                # Use overall percentage for filtering decisions to be consistent
                miss_pct = overall_miss_pct
                miss_cnt = overall_miss_cnt

                print(
                    f"{metric:<35} | {miss_cnt:5d} missing ({miss_pct:5.1f}%) [overall] | {'❌ EXCLUDE' if miss_pct > (nan_threshold * 100) else '✓ KEEP'}"
                )
                print(
                    f"{'':35} | {conditional_miss_cnt:5d} missing ({conditional_miss_pct:5.1f}%) [conditional on {dep_col}=True]"
                )
            else:
                miss_pct, miss_cnt = 100.0, valid_mask.size
        else:
            miss_pct = dataset[metric].isnull().mean() * 100
            miss_cnt = dataset[metric].isnull().sum()
            print(
                f"{metric:<35} | {miss_cnt:5d} missing ({miss_pct:5.1f}%) | {'❌ EXCLUDE' if miss_pct > (nan_threshold * 100) else '✓ KEEP'}"
            )

        missing_percentages[metric] = miss_pct
        missing_counts[metric] = miss_cnt

    total_rows = len(metrics_data)
    print(f"Total number of rows: {total_rows}")
    print(f"NaN threshold for exclusion: {nan_threshold*100}% (allowing up to {nan_threshold*100}% missing values)\n")
    print("Missing values per metric (sorted by % missing):")
    print("-" * 60)
    for metric in sorted(list(missing_percentages.keys()), key=lambda k: missing_percentages[k], reverse=True):
        count = missing_counts[metric]
        percentage = missing_percentages[metric]
        status = "❌ EXCLUDE" if percentage > (nan_threshold * 100) else "✓ KEEP"
        print(f"{metric:<35} | {count:5d} missing ({percentage:5.1f}%) | {status}")

    valid_metrics = [m for m, pct in missing_percentages.items() if pct <= nan_threshold * 100]
    excluded_metrics = [m for m in metric_columns if missing_percentages[m] > nan_threshold * 100]
    print(f"\n📊 SUMMARY:")
    print(f"   • Metrics to keep: {len(valid_metrics)}")
    print(f"   • Metrics to exclude: {len(excluded_metrics)}")

    metrics_data = metrics_data[valid_metrics]
    # Drop all-NaN columns and zero-variance columns
    metrics_data = metrics_data.dropna(axis=1, how="all")
    metrics_data = metrics_data.loc[:, metrics_data.std(skipna=True) > 0]

    # === Robust scaling for homogeneity (match PCA_Static) ===
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    metrics_data_scaled = pd.DataFrame(
        scaler.fit_transform(metrics_data), columns=metrics_data.columns, index=metrics_data.index
    )
    # Use scaled data for correlation analysis
    metrics_data = metrics_data_scaled

    if metrics_data.shape[1] < 2:
        raise ValueError(
            "Not enough valid metrics for clustering after filtering. " "Check your NaN threshold or input data."
        )

    # Create a temporary copy with median fill for correlation computation only
    tmp_for_corr = metrics_data.copy()
    tmp_for_corr = tmp_for_corr.fillna(tmp_for_corr.median(numeric_only=True))
    if metrics_data.shape[1] < 2:
        raise ValueError(
            "Not enough valid metrics for clustering after filtering. Check your NaN threshold or input data."
        )
    print(f"\n✓ Proceeding with {metrics_data.shape[1]} metrics for correlation analysis...")

    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster

    DOMAIN_PRIORITY = [
        "nb_events",
        "distance_moved",
        "max_distance",
        "has_finished",
        "chamber_exit_time",
        "first_major_event",
        "first_major_event_time",
        "max_event",
        "max_event_time",
        "nb_freeze",
        "number_of_pauses",
        "total_pause_duration",
        "normalized_velocity",
        "velocity_during_interactions",
        "fraction_not_facing_ball",
        "flailing",
        "head_pushing_ratio",
        "interaction_persistence",
        "persistence_at_end",
        "time_chamber_beginning",
        "chamber_ratio",
        "pulling_ratio",
        "significant_ratio",
    ]

    def cluster_select(
        metrics_data,
        valid_metrics,
        missing_percentages,
        cluster_threshold=0.2,
        stability_freq=None,
        domain_priority=None,
    ):
        corr = metrics_data.corr(method="spearman")
        D = 1 - corr.abs()
        np.fill_diagonal(D.values, 0)
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, fcluster

        D_condensed = squareform(D.values, checks=False)
        Z = linkage(D_condensed, method="average")
        cluster_labels = fcluster(Z, t=cluster_threshold, criterion="distance")

        clusters = {}
        cols = list(corr.columns)
        for col, label in zip(cols, cluster_labels):
            clusters.setdefault(label, []).append(col)

        selected_metrics = []
        for label, members in clusters.items():
            if len(members) == 1:
                selected_metrics.append(members[0])
            else:
                best_score, best_metric = -np.inf, None
                for m in members:
                    miss = missing_percentages[m]
                    dom = 1 if domain_priority and m in domain_priority else 0
                    stab = stability_freq[m] if stability_freq and m in stability_freq else 0
                    score = 2 * dom + stab - miss / 100
                    if score > best_score:
                        best_score, best_metric = score, m
                selected_metrics.append(best_metric)
        return selected_metrics, clusters

    print("\n🔁 Running stability analysis (bootstrap feature selection)...")
    n_boot = 100
    rng = np.random.default_rng(42)
    selection_counts = {m: 0 for m in valid_metrics}
    for i in range(n_boot):
        boot_idx = rng.integers(0, len(metrics_data), len(metrics_data))
        boot_data = metrics_data.iloc[boot_idx].reset_index(drop=True)
        boot_data_filled = boot_data.fillna(boot_data.median(numeric_only=True))
        boot_missing = (boot_data.isnull().sum() / len(boot_data)) * 100
        boot_valid = [
            col for col in valid_metrics if col in boot_missing.index and boot_missing[col] <= nan_threshold * 100
        ]
        if len(boot_valid) < 2:
            continue

        def cluster_select_boot(
            metrics_data,
            valid_metrics,
            missing_percentages,
            cluster_threshold=0.2,
            stability_freq=None,
            domain_priority=None,
        ):
            corr = boot_data_filled[boot_valid].corr(method="spearman")
            D = 1 - corr.abs()
            np.fill_diagonal(D.values, 0)
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import linkage, fcluster

            D_condensed = squareform(D.values, checks=False)
            Z = linkage(D_condensed, method="average")
            cluster_labels = fcluster(Z, t=cluster_threshold, criterion="distance")
            clusters = {}
            cols = list(corr.columns)
            for col, label in zip(cols, cluster_labels):
                clusters.setdefault(label, []).append(col)
            selected_metrics = []
            for label, members in clusters.items():
                if len(members) == 1:
                    selected_metrics.append(members[0])
                else:
                    best_score, best_metric = -np.inf, None
                    for m in members:
                        miss = missing_percentages[m]
                        dom = 1 if domain_priority and m in domain_priority else 0
                        stab = stability_freq[m] if stability_freq and m in stability_freq else 0
                        score = 2 * dom + stab - miss / 100
                        if score > best_score:
                            best_score, best_metric = score, m
                    selected_metrics.append(best_metric)
            return selected_metrics, clusters

        boot_selected, _ = cluster_select_boot(boot_data[boot_valid], boot_valid, boot_missing, cluster_threshold=0.2)
        for m in boot_selected:
            selection_counts[m] = selection_counts.get(m, 0) + 1
    stability_df = pd.DataFrame(
        {
            "metric": list(selection_counts.keys()),
            "selection_count": list(selection_counts.values()),
            "selection_freq": [selection_counts[m] / n_boot for m in selection_counts],
        }
    )
    stability_df = stability_df.sort_values("selection_freq", ascending=False)
    stability_df.to_csv("metrics_selection_stability.csv", index=False)
    print(f"\n📊 Stability analysis complete. Saved to metrics_selection_stability.csv")
    print("Top stable metrics (selected in >50% of bootstraps):")
    for _, row in stability_df[stability_df["selection_freq"] > 0.5].iterrows():
        print(f"  - {row['metric']}: {row['selection_freq']*100:.1f}%")

    stable_metrics = stability_df[stability_df["selection_freq"] > 0.5]["metric"].tolist()
    with open("stable_metrics_for_pca_alt.txt", "w") as f:
        for m in stable_metrics:
            f.write(m + "\n")
    print(f"\n💾 Saved stable metrics list for PCA to stable_metrics_for_pca_alt.txt")

    # ---- Per-family clustering and selection ----
    tmp_for_corr = metrics_data.copy()
    tmp_for_corr = tmp_for_corr.fillna(tmp_for_corr.median(numeric_only=True))

    print("\n🧭 Assigning families to valid metrics...")
    families = group_metrics_by_family(valid_metrics)
    for fam, cols in sorted(families.items()):
        print(f"  - {fam}: {len(cols)} metrics")

    # Per-family clustering and selection
    family_selected = {}
    family_clusters = {}

    print("\n🔗 Per-family clustering and representative selection:")
    for fam, fam_cols in families.items():
        if len(fam_cols) == 0:
            continue
        if len(fam_cols) == 1:
            family_selected[fam] = fam_cols
            family_clusters[fam] = {1: fam_cols}
            print(f"  • {fam}: only 1 metric → keep {fam_cols[0]}")
            continue

        fam_df = tmp_for_corr.loc[:, [c for c in fam_cols if c in tmp_for_corr.columns]]
        fam_df = fam_df.loc[:, fam_df.columns[fam_df.nunique(dropna=True) > 1]]
        if fam_df.shape[1] < 2:
            kept = [fam_cols[0]]
            family_selected[fam] = kept
            family_clusters[fam] = {1: [fam_cols[0]]}
            print(f"  • {fam}: low-variability family → keep {fam_cols[0]}")
            continue

        fam_valid = list(fam_df.columns)
        fam_missing = {m: missing_percentages[m] for m in fam_valid}
        fam_stab = dict(zip(stability_df["metric"], stability_df["selection_freq"]))

        sel, clus = cluster_select(
            fam_df,
            fam_valid,
            fam_missing,
            cluster_threshold=0.2,
            stability_freq=fam_stab,
            domain_priority=DOMAIN_PRIORITY,
        )
        family_selected[fam] = sel
        family_clusters[fam] = clus
        print(f"  • {fam}: {len(clus)} clusters → selected {len(sel)} reps")

    # Combine per-family representatives
    combined_selected = []
    for fam, reps in family_selected.items():
        combined_selected.extend(reps)

    # Optional cross-family de-duplication (loose threshold)
    ENABLE_CROSS_FAMILY_DEDUP = True
    CROSS_FAMILY_D = 0.10
    if ENABLE_CROSS_FAMILY_DEDUP and len(combined_selected) > 1:
        comb_df = tmp_for_corr.loc[:, combined_selected].copy()
        corr_comb = comb_df.corr(method="spearman").abs()
        rho_thresh = 1 - CROSS_FAMILY_D
        to_drop = set()

        def score_metric(m):
            miss = missing_percentages.get(m, 0)
            dom = 1 if DOMAIN_PRIORITY and m in DOMAIN_PRIORITY else 0
            stab = dict(zip(stability_df["metric"], stability_df["selection_freq"]))
            return 2 * dom + stab.get(m, 0) - miss / 100

        cols = list(corr_comb.columns)
        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue
                if corr_comb.iloc[i, j] >= rho_thresh:
                    mi, mj = cols[i], cols[j]
                    if score_metric(mi) >= score_metric(mj):
                        to_drop.add(mj)
                    else:
                        to_drop.add(mi)
        combined_selected = [m for m in combined_selected if m not in to_drop]
        if len(to_drop) > 0:
            print(f"\n🧹 Cross-family de-duplication removed {len(to_drop)} near-duplicates.")

    # Build overall correlation matrix for plotting/reporting (on tmp_for_corr for completeness)
    correlation_matrix = tmp_for_corr.loc[:, combined_selected].corr(method="spearman")

    # Save outputs
    with open("final_metrics_for_pca_alt.txt", "w") as f:
        for m in combined_selected:
            f.write(m + "\n")
    print(f"\n💾 Saved final metrics list for PCA to final_metrics_for_pca_alt.txt")

    with open("final_metrics_by_family.txt", "w") as f:
        for fam, reps in sorted(family_selected.items()):
            f.write(f"[{fam}] {', '.join(reps)}\n")
    print("💾 Saved per-family representatives to final_metrics_by_family.txt")

    selected_metrics = combined_selected
    clusters = {
        f"{fam}:{lab}": members
        for fam, fam_clusters in family_clusters.items()
        for lab, members in fam_clusters.items()
    }

    return correlation_matrix, clusters, selected_metrics


def create_correlation_heatmap(correlation_matrix, output_path="correlation_heatmap.png"):
    """Create and save a correlation heatmap."""
    plt.figure(figsize=(20, 16))

    # Create mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=False,  # Don't annotate due to size
        cmap="RdBu_r",
        center=0,
        square=True,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Metrics Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✓ Correlation heatmap saved to: {output_path}")


def suggest_metrics_to_remove(high_corr_pairs, correlation_matrix):
    """Suggest which metrics to remove based on correlation analysis."""
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS FOR REDUNDANT METRICS")
    print("=" * 50)

    if not high_corr_pairs:
        print("✓ No highly correlated metric pairs found!")
        return []

    # Count how many high correlations each metric has
    metric_corr_counts = {}
    for pair in high_corr_pairs:
        metric1, metric2 = pair["metric1"], pair["metric2"]
        metric_corr_counts[metric1] = metric_corr_counts.get(metric1, 0) + 1
        metric_corr_counts[metric2] = metric_corr_counts.get(metric2, 0) + 1

    # Suggest removal candidates (metrics with many high correlations)
    removal_candidates = []
    processed_pairs = set()

    print(f"\nFound {len(high_corr_pairs)} highly correlated pairs:")
    print("-" * 50)

    for pair in high_corr_pairs:
        metric1, metric2, corr = pair["metric1"], pair["metric2"], pair["correlation"]
        pair_key = tuple(sorted([metric1, metric2]))

        if pair_key not in processed_pairs:
            print(f"{metric1:<30} ↔ {metric2:<30} | r = {corr:6.3f}")

            # Suggest which one to remove based on various criteria
            if metric_corr_counts[metric1] > metric_corr_counts[metric2]:
                suggestion = metric1
            elif metric_corr_counts[metric2] > metric_corr_counts[metric1]:
                suggestion = metric2
            else:
                # If equal, prefer removing more complex/derived metrics
                if any(pattern in metric1 for pattern in ["binned_", "rate_", "ratio"]):
                    suggestion = metric1
                elif any(pattern in metric2 for pattern in ["binned_", "rate_", "ratio"]):
                    suggestion = metric2
                else:
                    suggestion = metric2  # Default to second one

            removal_candidates.append(suggestion)
            processed_pairs.add(pair_key)

    # Get unique removal candidates
    unique_candidates = list(set(removal_candidates))

    print(f"\n📋 SUGGESTED METRICS TO CONSIDER REMOVING:")
    print("-" * 50)
    for candidate in sorted(unique_candidates):
        corr_count = metric_corr_counts.get(candidate, 0)
        print(f"• {candidate:<30} (involved in {corr_count} high correlations)")

    return unique_candidates


def create_final_metrics_list(correlation_matrix, removal_candidates, keep_for_biological_significance=None):
    """Create the final list of metrics for PCA after removing highly correlated ones."""
    if keep_for_biological_significance is None:
        keep_for_biological_significance = ["pulled"]

    # Start with all metrics that passed the NaN filtering
    final_metrics = list(correlation_matrix.columns)

    # Remove the suggested candidates, except those we want to keep for biological significance
    metrics_to_remove = [metric for metric in removal_candidates if metric not in keep_for_biological_significance]

    final_metrics_for_pca = [metric for metric in final_metrics if metric not in metrics_to_remove]

    print(f"\n🎯 FINAL METRICS LIST FOR PCA")
    print("=" * 60)
    print(f"   • Started with: {len(correlation_matrix.columns)} metrics (after NaN filtering)")
    print(f"   • Removal candidates: {len(removal_candidates)}")
    print(f"   • Kept for biological significance: {keep_for_biological_significance}")
    print(f"   • Actually removed: {len(metrics_to_remove)}")
    print(f"   • Final metrics for PCA: {len(final_metrics_for_pca)}")

    if metrics_to_remove:
        print(f"\n❌ REMOVED METRICS:")
        for metric in sorted(metrics_to_remove):
            print(f"   • {metric}")

    if keep_for_biological_significance:
        kept_candidates = [m for m in removal_candidates if m in keep_for_biological_significance]
        if kept_candidates:
            print(f"\n🧬 KEPT FOR BIOLOGICAL SIGNIFICANCE:")
            for metric in sorted(kept_candidates):
                print(f"   • {metric}")

    print(f"\n✅ FINAL METRICS FOR PCA ({len(final_metrics_for_pca)} metrics):")
    print("-" * 60)
    for i, metric in enumerate(sorted(final_metrics_for_pca), 1):
        print(f"{i:3d}. {metric}")

    return final_metrics_for_pca


def save_correlation_analysis(correlation_matrix, high_corr_pairs, metric_columns, output_dir="."):
    """Save correlation analysis results."""
    output_dir = Path(output_dir)

    # Save correlation matrix
    correlation_matrix.to_csv(output_dir / "metrics_correlation_matrix.csv")

    # Save high correlation pairs
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df.to_csv(output_dir / "high_correlation_pairs.csv", index=False)

    # Save metrics list
    metrics_df = pd.DataFrame({"metric": metric_columns})
    metrics_df.to_csv(output_dir / "metrics_list.csv", index=False)

    print(f"\n✓ Analysis results saved to {output_dir}/")


def main():
    DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250924_14_summary_TNT_screen_Data/summary/pooled_summary.feather"
    CORRELATION_THRESHOLD = 0.8
    NAN_THRESHOLD = 0.05
    print("🔍 BALLPUSHING METRICS CORRELATION ANALYSIS\n" + "=" * 50)
    dataset = load_metrics_data(DATA_PATH)
    if dataset is None:
        return

    metric_columns = identify_metric_columns(dataset)
    print(f"\n📊 Found {len(metric_columns)} potential metrics:")
    for i, metric in enumerate(metric_columns, 1):
        print(f"{i:3d}. {metric}")

    analyze_missing_values_by_nickname(dataset, metric_columns, NAN_THRESHOLD)
    correlation_matrix, clusters, selected_metrics = analyze_correlations(
        dataset, metric_columns, CORRELATION_THRESHOLD, NAN_THRESHOLD
    )
    create_correlation_heatmap(correlation_matrix)
    print(f"\n✅ Analysis complete!")
    print(f"   • Initial metrics found: {len(metric_columns)}")
    print(f"   • Final metrics after cluster-based selection: {len(selected_metrics)}")  # fixed
    print(f"   • Cluster count: {len(clusters)}")
    print(f"   • See final_metrics_for_pca.txt and stable_metrics_for_pca.txt for PCA input.")


if __name__ == "__main__":
    main()
