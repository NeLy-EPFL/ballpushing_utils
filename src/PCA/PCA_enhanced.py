# Updated PCA Analysis with Multiblock Integration and Robustness

# 1. Import Enhanced PCA Modules
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import RobustScaler  # Better for biological data
from prince import MFA  # Multiblock analysis
import numpy as np
import pandas as pd
import Config
import subprocess

import matplotlib.pyplot as plt

# 2. Data Segmentation
dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250520_summary_TNT_screen_Data/summary/pooled_summary.feather"
)

# dataset = Config.map_split_registry(dataset)

controls = ["Empty-Split", "Empty-Gal4", "TNTxPR"]

dataset = Config.cleanup_data(dataset)

exclude_nicknames = ["Ple-Gal4.F a.k.a TH-Gal4", "TNTxCS"]

dataset = dataset[~dataset["Nickname"].isin(exclude_nicknames)]


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
    # "first_significant_event",
    # "first_significant_event_time",
    "major_event",
    "major_event_time",
    # "major_event_first",
    # "insight_effect",
    # "insight_effect_log",
    # "cumulated_breaks_duration",
    # "chamber_time",
    # "chamber_ratio",
    # "pushed",
    "pulled",
    "pulling_ratio",
    # "success_direction",
    # "interaction_proportion",
    "interaction_persistence",
    "distance_moved",
    "distance_ratio",
    # "exit_time",
    "chamber_exit_time",
    # "number_of_pauses",
    # "total_pause_duration",
    # "learning_slope",
    # "learning_slope_r2",
    # "logistic_L",
    # "logistic_k",
    # "logistic_t0",
    # "logistic_r2",
    # "avg_displacement_after_success",
    # "avg_displacement_after_failure",
    # "influence_ratio",
    "normalized_velocity",
    # "velocity_during_interactions",
    # "velocity_trend",
    "auc",
    "overall_slope",
    "overall_interaction_rate",
]

# Find all columns with the desired prefixes
binned_slope_cols = [col for col in dataset.columns if col.startswith("binned_slope_")]
interaction_rate_cols = [col for col in dataset.columns if col.startswith("interaction_rate_bin_")]
binned_auc_cols = [col for col in dataset.columns if col.startswith("binned_auc_")]

# Add them to metric_names list
metric_names += binned_slope_cols + interaction_rate_cols + binned_auc_cols

# For each column, handle missing values gracefully

dataset["max_event"] = dataset["max_event"].fillna(-1)
dataset["first_significant_event"] = dataset["first_significant_event"].fillna(-1)
dataset["major_event"] = dataset["major_event"].fillna(-1)
dataset["final_event"] = dataset["final_event"].fillna(-1)
dataset["max_event_time"] = dataset["max_event_time"].fillna(3600)
dataset["first_significant_event_time"] = dataset["first_significant_event_time"].fillna(3600)
dataset["major_event_time"] = dataset["major_event_time"].fillna(3600)
dataset["final_event_time"] = dataset["final_event_time"].fillna(3600)

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

metrics_data = dataset[valid_metric_names]  # Select only the valid metric columns
metadata_data = dataset[metadata_names]  # Select only the metadata columns

# Split metrics into temporal vs static blocks
temporal_metrics = [
    col for col in valid_metric_names if col.startswith(("binned_slope_", "interaction_rate_bin_", "binned_auc_"))
]
static_metrics = [col for col in valid_metric_names if col not in temporal_metrics]

# 3. Robust Scaling per Block
scaler_temporal = RobustScaler(quantile_range=(25, 75))  # Resistant to outliers
metrics_temporal_scaled = scaler_temporal.fit_transform(metrics_data[temporal_metrics])

scaler_static = RobustScaler()
metrics_static_scaled = scaler_static.fit_transform(metrics_data[static_metrics])

# 4. Multiblock PCA Implementation
mfa = MFA(
    n_components=min(30, len(temporal_metrics) + len(static_metrics)),
    copy=True,
    random_state=42,
)

# Combine scaled data preserving block structure
combined_scaled = np.hstack((metrics_temporal_scaled, metrics_static_scaled))
combined_scaled_df = pd.DataFrame(combined_scaled, columns=temporal_metrics + static_metrics)

# Save scaled data for R
combined_scaled_df.to_csv("mfa_input.csv", index=False)

# Write group sizes (number of columns in each block) BEFORE calling R
with open("mfa_groups.txt", "w") as f:
    f.write(f"{len(temporal_metrics)} {len(static_metrics)}\n")

# Call R script to run MFA with FactoMineR
subprocess.run(["Rscript", "run_mfa.R"], check=True)

# Read R results
mfa_scores = pd.read_csv("mfa_scores.csv")
loadings_df = pd.read_csv("mfa_loadings.csv", index_col=0)
contrib_df = pd.read_csv("mfa_block_contrib.csv", index_col=0)

# Combine with metadata
final_dataset = pd.concat([mfa_scores, metadata_data.reset_index(drop=True)], axis=1)
final_dataset.to_feather("mfa_with_metadata.feather")

# Save block-aware loadings (already saved by R, but you can re-save if needed)
loadings_df.to_csv("mfa_block_loadings.csv")
contrib_df.to_csv("mfa_block_contributions.csv")

# Read explained variance from R
eig_df = pd.read_csv("mfa_eigenvalues.csv")
plt.plot(np.cumsum(eig_df["percentage of variance"]))
plt.xlabel("Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("MFA Cumulative Explained Variance (FactoMineR)")
plt.show()
