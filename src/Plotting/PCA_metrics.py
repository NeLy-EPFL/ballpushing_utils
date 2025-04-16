from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import Config

from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd


dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# dataset = Config.map_split_registry(dataset)

controls = ["Empty-Split", "Empty-Gal4", "TNTxPR"]

dataset = Config.cleanup_data(dataset)

# Convert boolean metrics to numeric (1 for True, 0 for False)
for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)


# Configure which are the metrics and what are the metadata

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
    "major_event_time",
    "major_event_first",
    # "insight_effect",
    # "insight_effect_log",
    "cumulated_breaks_duration",
    "chamber_time",
    "chamber_ratio",
    "pushed",
    "pulled",
    "pulling_ratio",
    # "success_direction",
    "interaction_proportion",
    "interaction_persistence",
    "distance_moved",
    "distance_ratio",
    # "exit_time",
    "chamber_exit_time",
    "number_of_pauses",
    "total_pause_duration",
    # "learning_slope",
    # "learning_slope_r2",
    # "logistic_L",
    # "logistic_k",
    # "logistic_t0",
    # "logistic_r2",
    "avg_displacement_after_success",
    "avg_displacement_after_failure",
    "influence_ratio",
    "normalized_velocity",
    "velocity_during_interactions",
    "velocity_trend",
]

# For each column, handle missing values gracefully

# For max event, first significant event, major event final event, if missing, set to -1
dataset["max_event"].fillna(-1, inplace=True)
dataset["first_significant_event"].fillna(-1, inplace=True)
dataset["major_event"].fillna(-1, inplace=True)
dataset["final_event"].fillna(-1, inplace=True)

# For max event time, first significant event time, major event final event time, if missing, set to 3600
dataset["max_event_time"].fillna(3600, inplace=True)
dataset["first_significant_event_time"].fillna(3600, inplace=True)
dataset["major_event_time"].fillna(3600, inplace=True)
dataset["final_event_time"].fillna(3600, inplace=True)

# Remove columns insight_effect, insight_effect_log, exit_time
dataset.drop(columns=["insight_effect", "insight_effect_log", "exit_time"], inplace=True)

# for pulling_ratio, avg_displacement_after_success, avg_displacement_before_success, and influence_ratio, if missing set to 0
dataset["pulling_ratio"].fillna(0, inplace=True)
dataset["avg_displacement_after_success"].fillna(0, inplace=True)
dataset["avg_displacement_after_failure"].fillna(0, inplace=True)
dataset["influence_ratio"].fillna(0, inplace=True)

# Check which metrics have NA values
na_metrics = dataset[metric_names].isna().sum()
print("Metrics with NA values:")
print(na_metrics[na_metrics > 0])

# Keep only columns without NaN values
valid_metric_names = [col for col in metric_names if col not in na_metrics[na_metrics > 0].index]
print(f"Valid metric columns (no NaNs): {valid_metric_names}")

# Define metadata columns as any column not in metric_names
metadata_names = [col for col in dataset.columns if col not in metric_names]

print(f"Metadata columns: {metadata_names}")
print(f"Metric columns: {metric_names}")

# Step 1: Extract metrics and metadata
metrics_data = dataset[valid_metric_names]  # Select only the valid metric columns
metadata_data = dataset[metadata_names]  # Select only the metadata columns

# Step 2: Drop rows with NaN values in the valid metrics
metrics_data = metrics_data.dropna()
metadata_data = metadata_data.loc[metrics_data.index]  # Align metadata with the cleaned metrics data

# Step 3: Standardize the metrics data
scaler = StandardScaler()
metrics_scaled = scaler.fit_transform(metrics_data)

# Step 4: Perform PCA
pca = PCA(n_components=3)  # Keep the first 3 principal components
pca_result = pca.fit_transform(metrics_scaled)

# Step 5: Create a DataFrame for the PCA results
pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_result, columns=pca_columns)

# Print the explained variance of each component
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each principal component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f}")

# Step 6: Save PCA loadings to a CSV file
loadings_df = pd.DataFrame(pca.components_.T, columns=pca_columns, index=valid_metric_names)
loadings_df.to_csv("pca_loadings.csv", index=True)

# # Check if the PC columns contain NaN values
# if pca_df.isnull().values.any():
#     print("PCA DataFrame contains NaN values")

#     # Count the number of NaN values in each column
#     na_counts = pca_df.isna().sum()
#     print("NaN counts in PCA DataFrame:")
#     print(na_counts[na_counts > 0])

# # Remove any rows with NaN values in the PCA DataFrame
# pca_df = pca_df.dropna()

# Step 7: Combine the first 3 PCs with metadata
final_dataset = pd.concat([pca_df, metadata_data.reset_index(drop=True)], axis=1)

# Step 8: Save the final dataset to a CSV file
final_dataset.to_feather("pca_with_metadata.feather")

print("PCA loadings and dataset with PCs and metadata have been saved.")


def run_permanova_on_nicknames(df, pca_df, metadata_df, group_column="Nickname"):
    results = []

    for nickname in df[group_column].unique():
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

        # If the subset is empty after cleaning, skip this iteration
        if subset_pca.empty:
            print(f"Skipping Nickname {nickname} due to empty PCA data after cleaning.")
            continue

        # Compute the distance matrix
        pca_distances = pdist(subset_pca.values, metric="euclidean")
        distance_matrix = DistanceMatrix(squareform(pca_distances), ids=subset_metadata.index.astype(str))

        # Perform PERMANOVA
        try:
            permanova_results = permanova(distance_matrix, subset_metadata.values, permutations=999)
            results.append(
                {
                    "Nickname": nickname,
                    "Pseudo-F": permanova_results["test_statistic"],
                    "p-value": permanova_results["p_value"],
                }
            )
            print(
                f"PERMANOVA for {nickname}: Pseudo-F = {permanova_results['test_statistic']:.4f}, p-value = {permanova_results['p_value']:.4f}"
            )
        except Exception as e:
            print(f"Error running PERMANOVA for {nickname}: {e}")

    # Convert results to a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    return results_df


# Run the PERMANOVA analysis
permanova_results_df = run_permanova_on_nicknames(final_dataset, pca_df, metadata_data)
print(permanova_results_df)
# Save the PERMANOVA results to a CSV file
permanova_results_df.to_csv("permanova_results.csv", index=False)
