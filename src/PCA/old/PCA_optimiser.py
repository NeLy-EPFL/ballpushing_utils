from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import Config

# Load and preprocess the dataset
dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# Clean up the dataset using Config
dataset = Config.cleanup_data(dataset)

# Convert boolean metrics to numeric (1 for True, 0 for False)
for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)

# Define the list of metrics
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
    "insight_effect",
    "insight_effect_log",
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
    "exit_time",
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

# Remove columns that are not in the dataset
metric_names = [col for col in metric_names if col in dataset.columns]

# Handle missing values for specific metrics
dataset["max_event"] = dataset["max_event"].fillna(-1)
dataset["first_significant_event"] = dataset["first_significant_event"].fillna(-1)
dataset["major_event"] = dataset["major_event"].fillna(-1)
dataset["final_event"] = dataset["final_event"].fillna(-1)
dataset["max_event_time"] = dataset["max_event_time"].fillna(3600)
dataset["first_significant_event_time"] = dataset["first_significant_event_time"].fillna(3600)
dataset["first_major_event_time"] = dataset["first_major_event_time"].fillna(3600)
dataset["final_event_time"] = dataset["final_event_time"].fillna(3600)
dataset["pulling_ratio"] = dataset["pulling_ratio"].fillna(0)
dataset["avg_displacement_after_success"] = dataset["avg_displacement_after_success"].fillna(0)
dataset["avg_displacement_after_failure"] = dataset["avg_displacement_after_failure"].fillna(0)
dataset["influence_ratio"] = dataset["influence_ratio"].fillna(0)

# Check which metrics have NA values
na_metrics = dataset[metric_names].isna().sum()
print("Metrics with NA values:")
print(na_metrics[na_metrics > 0])

# Keep only columns without NaN values
valid_metric_names = [col for col in metric_names if col not in na_metrics[na_metrics > 0].index]
print(f"Valid metric columns (no NaNs): {valid_metric_names}")

# Extract valid metrics data
metrics_data = dataset[valid_metric_names]

# Ensure all columns in metrics_data are numeric
metrics_data = metrics_data.apply(pd.to_numeric, errors="coerce")

# Drop columns with non-numeric data
non_numeric_columns = metrics_data.columns[metrics_data.isna().any()].tolist()
if non_numeric_columns:
    print(f"Non-numeric columns detected and removed: {non_numeric_columns}")
    metrics_data = metrics_data.drop(columns=non_numeric_columns)

# Proceed with variance filtering
feature_variance = metrics_data.var(axis=0)
selected_features = feature_variance[feature_variance > 0.01].index.tolist()
metrics_data = metrics_data[selected_features]

# Filter features based on correlation
corr_matrix = metrics_data.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
metrics_data = metrics_data.drop(columns=to_drop, errors="ignore")


# Function to find the best combination of metrics
def find_best_metric_combinations(metrics_data, max_metrics=None, min_metrics=10, top_n=10):
    """
    Find the top N combinations of metrics that maximize the sum of explained variance
    for the first three principal components (PC1, PC2, PC3).

    Args:
        metrics_data (pd.DataFrame): The metrics data to evaluate.
        max_metrics (int): The maximum number of metrics to consider in combinations.
                           If None, all metrics will be considered.
        min_metrics (int): The minimum number of metrics to include in combinations.
        top_n (int): The number of top combinations to return.

    Returns:
        list: A list of dictionaries with the top combinations and their explained variance.
    """
    results = []

    # Limit the number of metrics to consider if max_metrics is specified
    metric_names = metrics_data.columns.tolist()
    if max_metrics is not None:
        metric_names = metric_names[:max_metrics]

    # Iterate over all combinations of metrics
    for r in range(min_metrics, len(metric_names) + 1):
        for combination in combinations(metric_names, r):
            # Select the subset of metrics
            subset_data = metrics_data[list(combination)]

            # Standardize the data
            scaler = StandardScaler()
            subset_scaled = scaler.fit_transform(subset_data)

            # Perform PCA
            pca = PCA(n_components=3)
            pca.fit(subset_scaled)

            # Calculate the sum of explained variance for PC1, PC2, and PC3
            explained_variance = sum(pca.explained_variance_ratio_)

            # Store the result
            results.append(
                {
                    "combination": combination,
                    "explained_variance": explained_variance,
                }
            )

    # Sort results by explained variance in descending order
    results = sorted(results, key=lambda x: x["explained_variance"], reverse=True)

    # Return the top N combinations
    return results[:top_n]


# Example usage
print("Finding the top 10 combinations of metrics...")
top_combinations = find_best_metric_combinations(metrics_data, max_metrics=None, min_metrics=5, top_n=10)

# Print the top combinations
for i, result in enumerate(top_combinations, 1):
    print(f"Rank {i}: Combination: {result['combination']}, Explained Variance: {result['explained_variance']:.4f}")

# Save the top combinations to a CSV file
top_combinations_df = pd.DataFrame(
    [
        {
            "Rank": i + 1,
            "Combination": ", ".join(result["combination"]),
            "Explained Variance": result["explained_variance"],
        }
        for i, result in enumerate(top_combinations)
    ]
)
top_combinations_df.to_csv("top_combinations.csv", index=False)
