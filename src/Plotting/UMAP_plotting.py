import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from Config import get_subset_data, color_dict


# Function to sanitize filenames
def sanitize_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)


# Load hits from CSV
hits_df = pd.read_csv("/home/matthias/ballpushing_utils/src/Plotting/Umap_hits.csv")

# Filter out rows with empty significant_clusters
hits_df = hits_df[hits_df["significant_clusters"] != "[]"]

# Load UMAP data
umap_data = pd.read_feather("/home/matthias/ballpushing_utils/tests/integration/outputs/umap_TNT_1.feather")

# Process each hit
for _, hit in hits_df.iterrows():
    nickname = hit["Nickname"]

    # Get subset data for the Nickname and its control
    subset_data = get_subset_data(umap_data, col="Nickname", value=nickname)

    # Dynamically set the control nickname based on Brain region == "Control"
    control_rows = subset_data[subset_data["Brain region"] == "Control"]
    if not control_rows.empty:
        control = control_rows["Nickname"].iloc[0]
    else:
        control = "Unknown Control"  # Fallback if no control is found

    # Initialize a DataFrame to store the individual fly proportions for this nickname
    fly_proportions_df = pd.DataFrame()

    # Get all unique clusters in the dataset
    all_clusters = sorted(subset_data["cluster"].unique())

    # Get unique flies
    unique_flies = subset_data["fly"].unique()

    for fly_id in unique_flies:
        # Get data for this specific fly
        fly_data = subset_data[subset_data["fly"] == fly_id]

        # Count total occurrences of events in each cluster for this fly
        cluster_counts = fly_data["cluster"].value_counts()
        cluster_counts.name = "count"

        # Normalize by the total number of events for this fly to calculate proportions
        cluster_proportions = cluster_counts / cluster_counts.sum()
        cluster_proportions = cluster_proportions.reset_index()
        cluster_proportions.columns = ["cluster", "proportion"]
        cluster_proportions["fly"] = fly_id
        cluster_proportions["Group"] = fly_data["Brain region"].iloc[0]  # Assuming each fly has one Brain region

        # Add missing clusters with proportion 0
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

        # Append to the proportions DataFrame
        fly_proportions_df = pd.concat([fly_proportions_df, cluster_proportions], ignore_index=True)

    # Skip plotting if no data
    if fly_proportions_df.empty:
        print(f"No data to plot for Nickname: {nickname}")
        continue

    # Ensure all clusters are included in the final DataFrame
    fly_proportions_df["cluster"] = fly_proportions_df["cluster"].astype(int)
    fly_proportions_df = fly_proportions_df.sort_values(by="cluster")

    # Create figure with more width to accommodate separations
    plt.figure(figsize=(18, 8))  # Increased width

    # Set style for better grid visibility
    sns.set_style("whitegrid")

    # Add boxplot with explicit order
    ax = sns.boxplot(
        data=fly_proportions_df,
        x="cluster",
        y="proportion",
        hue="Group",
        palette=color_dict,
        width=0.7,  # Slightly narrower boxes
    )

    # Add stripplot with black color
    sns.stripplot(
        data=fly_proportions_df,
        x="cluster",
        y="proportion",
        hue="Group",
        dodge=True,
        size=3,
        alpha=0.8,
        color="black",  # Set stripplot color to black
        linewidth=0.5,
    )

    # Add vertical lines between cluster groups
    for i in range(len(all_clusters) - 1):
        plt.axvline(x=i + 0.5, color="gray", linestyle="-", alpha=0.5, linewidth=1)

    # Add alternating background for easier visual grouping
    for i in range(len(all_clusters)):
        if i % 2 == 0:  # Every other cluster gets a light background
            plt.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="gray")

    # Set proper grid
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.grid(axis="x", visible=False)  # Remove default x grid

    plt.axhline(0.5, color="red", linestyle="--", label="Equal proportion")
    plt.title(f"Cluster Proportions - {nickname} vs {control}", fontsize=14)  # Updated title
    plt.xlabel("Cluster Number", fontsize=12)
    plt.ylabel("Proportion by Cluster", fontsize=12)

    # Improve x-axis labels
    plt.xticks(range(len(all_clusters)), [str(c) for c in all_clusters], fontsize=11, fontweight="bold")

    # Remove the legend
    plt.legend([], [], frameon=False)  # Remove legend

    plt.tight_layout()
    # Save the plot
    output_dir = "/home/matthias/ballpushing_utils/src/Plotting/Umap_hits"
    sanitized_nickname = sanitize_filename(nickname)
    plt.savefig(f"{output_dir}/proportions_by_cluster_{sanitized_nickname}.png")
    plt.show()
    plt.close()
