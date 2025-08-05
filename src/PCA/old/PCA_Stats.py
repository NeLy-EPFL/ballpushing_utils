from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import PCA.Config as Config

from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd

dataset = pd.read_feather("pca_with_metadata.feather")


def run_permanova_on_nicknames(df, pca_df, metadata_df, group_column="Nickname"):
    results = []

    for nickname in df[group_column].unique():
        print(f"Nickname selected: {nickname}")

        # Get the subset of data for the current Nickname and its control
        subset_data = Config.get_subset_data(df, col=group_column, value=nickname)

        if subset_data.empty:
            print(f"Skipping Nickname {nickname} due to empty subset.")
            continue

        # Extract PCA data and metadata directly from subset_data
        pca_columns = [f"PC{i+1}" for i in range(3)]  # Ensure these match the PCA column names
        subset_pca = subset_data[pca_columns]  # Extract PCA data
        subset_metadata = subset_data[group_column]  # Extract the group column directly

        # Count and remove NaN rows in subset_pca
        nan_count = subset_pca.isnull().sum().sum()  # Total number of NaN values
        if nan_count > 0:
            print(f"Removing {nan_count} NaN values from PCA data for Nickname {nickname}.")
            subset_pca = subset_pca.dropna()
            subset_metadata = subset_metadata.loc[subset_pca.index]  # Align metadata with cleaned PCA data

        # Check for non-finite values and remove them
        if not np.isfinite(subset_pca.values).all():
            print(f"Removing non-finite values from PCA data for Nickname {nickname}.")
            subset_pca = subset_pca[np.isfinite(subset_pca).all(axis=1)]
            subset_metadata = subset_metadata.loc[subset_pca.index]  # Align metadata with cleaned PCA data

        # Ensure there are exactly two groups
        unique_groups = subset_metadata.unique()
        if len(unique_groups) != 2:
            print(f"Skipping {nickname} due to insufficient groups: {unique_groups}")
            continue

        # Ensure IDs in the metadata are unique
        if subset_metadata.index.duplicated().any():
            print(f"Skipping {nickname} due to duplicate IDs in metadata.")
            continue

        # If the subset is empty after cleaning, skip this iteration
        if subset_pca.empty:
            print(f"Skipping {nickname} due to empty PCA data after cleaning.")
            continue

        # Debugging: Check group sizes
        group_sizes = subset_metadata.value_counts()
        print(f"Group sizes for {nickname}:")
        print(group_sizes)

        # Compute the distance matrix
        try:
            pca_distances = pdist(subset_pca.values, metric="euclidean")
            distance_matrix = DistanceMatrix(squareform(pca_distances), ids=subset_metadata.index.astype(str))
        except Exception as e:
            print(f"Error creating distance matrix for {nickname}: {e}")
            continue

        # Debugging: Check the distance matrix and grouping vector
        print(f"Distance matrix shape for {nickname}: {distance_matrix.shape}")
        print(f"Distance matrix contains NaN: {np.isnan(distance_matrix.data).any()}")
        print(f"Distance matrix contains infinite values: {np.isinf(distance_matrix.data).any()}")
        print(f"Grouping vector for {nickname}: {subset_metadata.values}")
        print(f"Unique groups: {np.unique(subset_metadata.values)}")

        # Perform PERMANOVA
        try:
            permanova_results = permanova(distance_matrix, subset_metadata.values, permutations=300)
            print(f"PERMANOVA raw results for {nickname}: {permanova_results}")

            # Extract results correctly from the pandas.Series
            test_statistic = permanova_results.loc["test statistic"]
            p_value = permanova_results.loc["p-value"]

            # Append results if valid
            results.append(
                {
                    "Nickname": nickname,
                    "Pseudo-F": test_statistic,
                    "p-value": p_value,
                }
            )
            print(f"PERMANOVA for {nickname}: Pseudo-F = {test_statistic:.4f}, p-value = {p_value:.4f}")
        except Exception as e:
            print(f"Error running PERMANOVA for {nickname}: {e}")

    # Convert results to a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    return results_df


# Run the PERMANOVA analysis
permanova_results_df = run_permanova_on_nicknames(final_dataset, pca_df, metadata_data)
print(permanova_results_df)
# Save the PERMANOVA results to a CSV file with high precision for p-values
permanova_results_df["p-value"] = permanova_results_df["p-value"].map(lambda x: f"{x:.7f}")
permanova_results_df.to_csv("permanova_results.csv", index=False)
