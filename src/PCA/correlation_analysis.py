#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster


def load_metrics_data(file_path):
    try:
        dataset = pd.read_feather(file_path)
        print(f"✓ Loaded dataset: shape={dataset.shape}")

        needs_update = False
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
        if needs_update:
            try:
                dataset.reset_index(drop=True, inplace=True)
                dataset.to_feather(file_path)
                print(f"✓ Dataset updated with derived flags")
            except Exception as e:
                print(f"✗ Could not write back updated dataset: {e}")
        return dataset
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None


def identify_metric_columns(dataset):
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
    additional_patterns = [
        "velocity",
        "speed",
        "pause",
        "pause",
        "stop",
        "persistence",
        "learning",
        "logistic",
        "influence",
        "trend",
        "interaction_rate",
        "finished",
        "chamber",
        "facing",
        "flailing",
        "head",
        "median_",
        "mean_",
        "has_",
    ]
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
        "raw_pauses",
    ]

    potential = []
    for m in base_metrics:
        if m in dataset.columns:
            potential.append(m)
    for patt in additional_patterns:
        potential.extend([c for c in dataset.columns if patt in c.lower()])
    potential = sorted(set(potential))

    actual = []
    for col in potential:
        low = col.lower()
        if any(k in low for k in metadata_keywords):
            continue
        if any(p in low for p in excluded_patterns):
            continue
        actual.append(col)
    return sorted(set(actual))


def calculate_correlation_degrees(corr_matrix, threshold=0.8):
    """
    Calculate how many other metrics each metric is highly correlated with.
    Higher degree = more redundant = should be penalized.
    """
    high_corr = (corr_matrix.abs() >= threshold) & (corr_matrix.abs() < 1.0)  # Exclude self-correlation
    degrees = high_corr.sum(axis=1)  # Count high correlations per metric
    return degrees


def correlation_select_objective(
    dataset, metric_columns, nan_threshold=0.05, correlation_threshold=0.8, cluster_distance=0.2, n_boot=100, seed=42
):
    """
    Select metrics using objective scoring:
    - Lower missingness = better
    - Lower correlation degree (fewer high correlations) = better
    - Higher bootstrap stability = better
    """
    # Convert bool to int
    X = dataset[metric_columns].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    # Calculate missingness and filter out high-missing metrics
    miss_pct = (X.isnull().sum() / len(X)) * 100
    valid_metrics = miss_pct[miss_pct <= (nan_threshold * 100)].index.tolist()
    excluded_metrics = miss_pct[miss_pct > (nan_threshold * 100)].index.tolist()

    # Save all metrics with <5% missing data for comparison
    pd.Series(valid_metrics, name="metric").to_csv("full_metrics_pca.txt", header=False, index=False)

    print(f"\n📊 MISSINGNESS ANALYSIS")
    print(f"{'Metric':<35} | {'Missing':<7} | {'%':<6} | {'Status'}")
    print("-" * 60)
    for metric in sorted(metric_columns, key=lambda m: miss_pct[m], reverse=True):
        count = X[metric].isnull().sum()
        pct = miss_pct[metric]
        status = "❌ DROP" if pct > (nan_threshold * 100) else "✓ KEEP"
        print(f"{metric:<35} | {count:<7} | {pct:<6.1f} | {status}")

    print(f"\n• Keep (≤{nan_threshold*100:.1f}% missing): {len(valid_metrics)}")
    print(f"• Drop (>{nan_threshold*100:.1f}% missing): {len(excluded_metrics)}")

    if excluded_metrics:
        print(f"\n❌ DROPPED METRICS:")
        for metric in sorted(excluded_metrics):
            print(f"   • {metric} ({miss_pct[metric]:.1f}% missing)")

    if len(valid_metrics) < 2:
        raise ValueError("Not enough valid metrics after NaN filtering.")

    # Work only with valid metrics
    Xv = X[valid_metrics].copy()

    # Drop constant columns
    nonconst = Xv.columns[Xv.std(skipna=True) > 0]
    if len(nonconst) < len(Xv.columns):
        dropped_const = set(Xv.columns) - set(nonconst)
        print(f"\n⚠️  Dropped {len(dropped_const)} constant metrics: {sorted(dropped_const)}")
    Xv = Xv[nonconst]

    if Xv.shape[1] < 2:
        raise ValueError("Not enough non-constant metrics after filtering.")

    print(f"\n✓ Proceeding with {len(Xv.columns)} metrics for correlation analysis")

    # Robust scale
    scaler = RobustScaler()
    Xv_scaled = pd.DataFrame(scaler.fit_transform(Xv), columns=Xv.columns, index=Xv.index)

    # Calculate correlation degrees on full correlation matrix
    full_corr = Xv_scaled.corr(method="spearman").abs()
    correlation_degrees = calculate_correlation_degrees(full_corr, threshold=correlation_threshold)

    print(f"\n🕸️  CORRELATION DEGREE ANALYSIS (threshold ≥ {correlation_threshold})")
    print(f"{'Metric':<35} | {'Degree':<6} | {'High correlations with'}")
    print("-" * 80)
    for metric in sorted(correlation_degrees.index, key=lambda m: correlation_degrees[m], reverse=True):
        degree = correlation_degrees[metric]
        if degree > 0:
            # Find which metrics this one is highly correlated with
            high_corrs = full_corr.loc[metric][full_corr.loc[metric] >= correlation_threshold]
            high_corrs = high_corrs[high_corrs.index != metric]  # Remove self
            corr_with = ", ".join([f"{m}({full_corr.loc[metric, m]:.2f})" for m in high_corrs.index])
            print(f"{metric:<35} | {degree:<6} | {corr_with}")
        else:
            print(f"{metric:<35} | {degree:<6} | (no high correlations)")

    # Bootstrap stability analysis
    print(f"\n🔄 Running bootstrap stability analysis ({n_boot} iterations)...")
    stability = pd.Series(0.0, index=Xv.columns)
    rng = np.random.default_rng(seed)
    boots_completed = 0

    for i in range(n_boot):
        # Bootstrap sample
        idx = rng.integers(0, len(Xv_scaled), len(Xv_scaled))
        B = Xv_scaled.iloc[idx].reset_index(drop=True)

        # Check if bootstrap sample still has valid metrics
        miss_b = (B.isnull().sum() / len(B)) * 100
        keep_b = [c for c in Xv.columns if miss_b.get(c, 0) <= nan_threshold * 100]

        if len(keep_b) < 2:
            continue

        # Correlation on bootstrap sample
        corr_b = B[keep_b].corr(method="spearman").abs()

        # Skip if correlation matrix has issues
        if corr_b.isnull().all().all():
            continue

        # Calculate correlation degrees for bootstrap sample
        boot_degrees = calculate_correlation_degrees(corr_b, threshold=correlation_threshold)

        # Hierarchical clustering
        D = 1 - corr_b.fillna(0)
        np.fill_diagonal(D.values, 0)

        try:
            Z = linkage(squareform(D.values, checks=False), method="average")
            labels = fcluster(Z, t=cluster_distance, criterion="distance")

            # Group into clusters
            clusters = {}
            cols = list(corr_b.columns)
            for col, lab in zip(cols, labels):
                clusters.setdefault(lab, []).append(col)

            # Select representatives using objective scoring
            for members in clusters.values():
                if len(members) == 1:
                    stability[members[0]] += 1
                else:
                    best = min(
                        members,
                        key=lambda m: (
                            boot_degrees.get(m, 0),  # Lower correlation degree is better
                            miss_pct.get(m, 0),  # Lower missingness is better
                            m,  # Alphabetical tiebreaker for determinism
                        ),
                    )
                    stability[best] += 1

            boots_completed += 1

        except Exception as e:
            continue

    if boots_completed > 0:
        stability = stability / boots_completed
        print(f"✓ Completed {boots_completed}/{n_boot} bootstrap iterations")
    else:
        print("⚠️  No successful bootstrap iterations - using single clustering")

    # Final clustering on full data
    corr = Xv_scaled.corr(method="spearman").abs()
    D = 1 - corr.fillna(0)
    np.fill_diagonal(D.values, 0)

    Z = linkage(squareform(D.values, checks=False), method="average")
    labels = fcluster(Z, t=cluster_distance, criterion="distance")

    # Group into clusters
    clusters = {}
    cols = list(corr.columns)
    for col, lab in zip(cols, labels):
        clusters.setdefault(lab, []).append(col)

    print(f"\n🔗 CLUSTERING RESULTS")
    print(f"Found {len(clusters)} clusters:")
    for i, (lab, members) in enumerate(clusters.items(), 1):
        if len(members) > 1:
            print(f"  Cluster {lab}: {len(members)} metrics - {', '.join(sorted(members))}")

    # Objective scoring function
    def objective_score(m):
        """
        Objective scoring (lower is better for selection):
        - Correlation degree penalty (more connections = worse)
        - Missingness penalty
        - Stability bonus (higher stability = lower penalty)
        """
        degree_penalty = correlation_degrees.get(m, 0)
        miss_penalty = miss_pct.get(m, 0) / 100.0
        stability_bonus = -stability.get(m, 0.0)  # Negative because we want lower total score

        return degree_penalty + miss_penalty + stability_bonus

    # Selection per cluster
    selected = []
    selection_details = []

    for lab, members in clusters.items():
        if len(members) == 1:
            selected.append(members[0])
            selection_details.append(
                {"cluster": lab, "selected": members[0], "reason": "only_member", "score": objective_score(members[0])}
            )
        else:
            # Score all members and pick the best (lowest score)
            scored = [(m, objective_score(m)) for m in members]
            scored.sort(key=lambda x: (x[1], x[0]))  # Ascending score, then alphabetical
            best_metric = scored[0][0]
            selected.append(best_metric)
            selection_details.append(
                {
                    "cluster": lab,
                    "selected": best_metric,
                    "reason": "lowest_objective_score",
                    "score": scored[0][1],
                    "alternatives": [f"{m}({s:.3f})" for m, s in scored[1:]],
                }
            )

    selected = sorted(set(selected))

    print(f"\n🎯 OBJECTIVE SELECTION DETAILS")
    print(f"{'Selected Metric':<35} | {'Degree':<6} | {'Miss%':<6} | {'Stability':<9} | {'Score':<7}")
    print("-" * 80)

    for detail in selection_details:
        if len(clusters[detail["cluster"]]) > 1:
            m = detail["selected"]
            degree = correlation_degrees.get(m, 0)
            miss = miss_pct.get(m, 0)
            stab = stability.get(m, 0)
            score = detail["score"]
            print(f"{m:<35} | {degree:<6} | {miss:<6.1f} | {stab:<9.3f} | {score:<7.3f}")
            if detail.get("alternatives"):
                print(f"    Alternatives: {', '.join(detail['alternatives'])}")

    print(f"\n✅ Selected {len(selected)} representatives from {len(clusters)} clusters")

    # Generate outputs
    corr_sel = Xv_scaled[selected].corr(method="spearman")

    # Save results
    pd.Series(selected, name="metric").to_csv("final_metrics_for_pca.txt", header=False, index=False)
    corr_sel.to_csv("metrics_correlation_matrix_selected.csv", index=True)

    # Save detailed analysis
    analysis_df = pd.DataFrame(
        {
            "metric": correlation_degrees.index,
            "correlation_degree": correlation_degrees.values,
            "missingness_pct": [miss_pct.get(m, 0) for m in correlation_degrees.index],
            "stability": [stability.get(m, 0) for m in correlation_degrees.index],
            "objective_score": [objective_score(m) for m in correlation_degrees.index],
            "selected": [m in selected for m in correlation_degrees.index],
        }
    )
    analysis_df.to_csv("objective_metrics_analysis.csv", index=False)

    print(f"\n📊 OBJECTIVE SCORING SUMMARY:")
    print(
        f"   • Average correlation degree of selected metrics: {analysis_df[analysis_df['selected']]['correlation_degree'].mean():.2f}"
    )
    print(
        f"   • Average correlation degree of non-selected metrics: {analysis_df[~analysis_df['selected']]['correlation_degree'].mean():.2f}"
    )
    print(
        f"   • Average missingness of selected metrics: {analysis_df[analysis_df['selected']]['missingness_pct'].mean():.2f}%"
    )
    print(f"   • Average stability of selected metrics: {analysis_df[analysis_df['selected']]['stability'].mean():.3f}")

    return selected, corr_sel, clusters


def main():
    DATA_PATH = "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250924_14_summary_TNT_screen_Data/summary/pooled_summary.feather"
    NAN_THRESHOLD = 0.05  # 5% max missing
    CORRELATION_THRESHOLD = 0.8  # Threshold for counting high correlations
    CLUSTER_DISTANCE = 0.2  # corresponds to |rho| ≥ 0.8 within-cluster
    N_BOOT = 100

    print("🔍 OBJECTIVE CORRELATION-BASED METRIC SELECTION")
    print("=" * 60)

    ds = load_metrics_data(DATA_PATH)
    if ds is None:
        return

    metric_cols = identify_metric_columns(ds)
    print(f"Discovered {len(metric_cols)} candidate metrics")

    selected, corr_sel, clusters = correlation_select_objective(
        ds,
        metric_cols,
        nan_threshold=NAN_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        cluster_distance=CLUSTER_DISTANCE,
        n_boot=N_BOOT,
        seed=42,
    )

    print(f"\n📋 FINAL SUMMARY")
    print(f"   • Initial candidate metrics: {len(metric_cols)}")
    print(f"   • Final selected metrics:    {len(selected)}")
    print(f"   • Clusters formed:           {len(clusters)}")
    print(
        f"   • Max correlation in selected set: {corr_sel.abs().values[np.triu_indices_from(corr_sel.abs().values, k=1)].max():.3f}"
    )

    print(f"\n💾 FILES SAVED:")
    print(f"   • final_metrics_for_pca.txt - List of selected metrics")
    print(f"   • metrics_correlation_matrix_selected.csv - Correlation matrix of selected metrics")
    print(f"   • objective_metrics_analysis.csv - Full analysis with degrees and scores")


if __name__ == "__main__":
    main()
