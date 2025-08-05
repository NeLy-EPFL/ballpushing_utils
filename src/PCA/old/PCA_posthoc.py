import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import PCA.Config as Config
import re  # For sanitizing filenames


# Function to sanitize nicknames for filenames
def sanitize_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)  # Replace non-alphanumeric characters with underscores


# Function to determine significance stars
def get_significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


# Create the output folder if it doesn't exist
output_folder = "PCA_hits"
os.makedirs(output_folder, exist_ok=True)

# Load the posthoc comparisons results
posthoc_results = pd.read_csv("all_significant_results.csv")  # Adjust for each category if needed

# Load the PCA dataset
pca_with_metadata = pd.read_feather("pca_with_metadata.feather")

# Filter for significant comparisons
if "adjusted p-value" in posthoc_results.columns:
    significant_comparisons = posthoc_results[posthoc_results["adjusted p-value"] < 0.05]
else:
    significant_comparisons = posthoc_results[posthoc_results["p-value"] < 0.05]

# Iterate over each significant nickname
for nickname in significant_comparisons["Nickname"].unique():
    print(f"Processing {nickname}...")

    # Use get_subset_data to get both the nickname and its control
    subset_data = Config.get_subset_data(pca_with_metadata, col="Nickname", value=nickname)

    # Split the data into nickname and control groups
    nickname_data = subset_data[subset_data["Nickname"] == nickname]
    control_name = subset_data[subset_data["Nickname"] != nickname]["Nickname"].iloc[0]
    control_data = subset_data[subset_data["Nickname"] == control_name]

    if nickname_data.empty or control_data.empty:
        print(f"Skipping {nickname} due to empty data for nickname or control.")
        continue

    # Sanitize nickname and control name for filenames
    sanitized_nickname = sanitize_filename(nickname)
    sanitized_control_name = sanitize_filename(control_name)

    # Create a plot for PC1, PC2, and PC3
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, pc in enumerate(["PC1", "PC2", "PC3"]):
        sns.boxplot(
            data=subset_data,
            x=pc,
            y="Nickname",
            hue="Brain region",
            palette=Config.color_dict,
            ax=axes[i],
            showfliers=False,
        )
        sns.stripplot(
            data=subset_data,
            x=pc,
            y="Nickname",
            color="black",
            size=5,
            jitter=True,
            ax=axes[i],
        )
        axes[i].set_title(f"{pc} Comparison")
        axes[i].set_xlabel(pc)
        axes[i].set_ylabel("")

        # Add significance stars
        pc_comparison = significant_comparisons[
            (significant_comparisons["Nickname"] == nickname) & (significant_comparisons["PC"] == pc)
        ]
        if not pc_comparison.empty:
            p_value = pc_comparison["adjusted p-value"].iloc[0]
            stars = get_significance_stars(p_value)
            if stars:
                # Add stars to the plot
                max_value = subset_data[pc].max()
                axes[i].text(
                    x=max_value * 0.9,  # Position near the maximum value
                    y=0.5,  # Centered between nickname and control
                    s=stars,
                    fontsize=16,
                    color="red",
                    ha="center",
                )

    # Set the overall title and save the plot
    fig.suptitle(f"{nickname} vs {control_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_folder, f"{sanitized_nickname}_vs_{sanitized_control_name}.png")
    plt.savefig(output_file)
    plt.close(fig)

    print(f"Plot saved for {nickname} vs {control_name}: {output_file}")
