import pandas as pd
import numpy as np
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn

import Config


def find_behavioral_hits(umap_data, config, alpha=0.05):
    """
    Identify Nicknames with significant differences in cluster proportions compared to their controls.

    Args:
        umap_data (pd.DataFrame): DataFrame containing UMAP embeddings, cluster labels, and Nicknames.
        config (module): Config module with the `get_subset` function.
        alpha (float): Significance level for statistical tests.

    Returns:
        pd.DataFrame: DataFrame of hits with significant differences and the clusters involved.
    """
    results = []

    # Get unique Nicknames
    nicknames = umap_data["Nickname"].unique()

    for nickname in nicknames:
        # Get subset data for the Nickname and its control
        subset_data = config.get_subset_data(umap_data, col="Nickname", value=nickname)
        if subset_data.empty:
            print(f"No data for Nickname: {nickname}")
            continue

        # print(f"Subset data for {nickname}:\n{subset_data}")

        # Calculate cluster proportions for each fly
        cluster_counts = subset_data.groupby(["fly", "cluster"]).size().unstack(fill_value=0)
        cluster_proportions = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

        # print(f"Cluster proportions for {nickname}:\n{cluster_proportions}")

        # Separate data for Nickname and its control
        nickname_data = cluster_proportions.loc[subset_data[subset_data["Nickname"] == nickname]["fly"].unique()]
        control_data = cluster_proportions.loc[
            subset_data[subset_data["Split"] == subset_data["Split"].iloc[0]]["fly"].unique()
        ]

        # Run Kruskal-Wallis test across all flies
        try:
            stat, p_value = kruskal(
                *[nickname_data[col].dropna() for col in nickname_data.columns],
                *[control_data[col].dropna() for col in control_data.columns],
            )
            print(f"Kruskal-Wallis test for {nickname}: stat={stat}, p_value={p_value}")
        except ValueError as e:
            print(f"Error in Kruskal-Wallis test for {nickname}: {e}")
            continue

        if p_value < alpha:
            significant_clusters = []
            for cluster in nickname_data.columns:
                # Combine data for the current cluster
                cluster_values = pd.concat(
                    [
                        pd.DataFrame({"values": nickname_data[cluster], "group": "Nickname"}),
                        pd.DataFrame({"values": control_data[cluster], "group": "Control"}),
                    ]
                )

                # Perform Dunn's test for the current cluster
                posthoc_results = posthoc_dunn(
                    cluster_values, val_col="values", group_col="group", p_adjust="bonferroni"
                )

                # Check if the p-value for the comparison is below alpha
                if posthoc_results.loc["Nickname", "Control"] < alpha:
                    significant_clusters.append(cluster)

            # Append results
            results.append(
                {
                    "Nickname": nickname,
                    "Control": subset_data["Split"].iloc[0],
                    "p_value": p_value,
                    "significant_clusters": significant_clusters,
                }
            )

    # Convert results to a DataFrame
    hits_df = pd.DataFrame(results)

    return hits_df


# Load UMAP data
umap_data = pd.read_feather("/home/matthias/ballpushing_utils/tests/integration/outputs/umap_TNT_1.feather")

# Find hits
hits = find_behavioral_hits(umap_data, Config, alpha=0.05)

# Save results to CSV
hits.to_csv("/home/matthias/ballpushing_utils/src/Plotting/Umap_hits.csv", index=False)

# Display results
print(hits)
