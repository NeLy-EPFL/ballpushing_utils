import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import Config

# Load the PCA dataset
pca_with_metadata = pd.read_feather("pca_with_metadata.feather")

# Create the output folder if it doesn't exist
output_folder = "PCA_hits"
os.makedirs(output_folder, exist_ok=True)

# Create a 3D scatterplot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Scatterplot for PC1, PC2, PC3, colored by Brain region
scatter = ax.scatter(
    pca_with_metadata["PC1"],
    pca_with_metadata["PC2"],
    pca_with_metadata["PC3"],
    c=pca_with_metadata["Brain region"].map(Config.color_dict),  # Map colors using Config.color_dict
    s=50,  # Marker size
    alpha=0.8,  # Transparency
)

# Set axis labels
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Global 3D PCA Scatterplot by Brain Region")

# Create a legend
unique_regions = pca_with_metadata["Brain region"].unique()
handles = [
    plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=10)
    for region, color in Config.color_dict.items()
    if region in unique_regions
]
labels = [region for region in Config.color_dict.keys() if region in unique_regions]
ax.legend(handles, labels, title="Brain Region", loc="upper left", bbox_to_anchor=(1.05, 1))

# Save the plot
output_file = os.path.join(output_folder, "global_3D_scatterplot.png")
plt.tight_layout()
plt.savefig(output_file, bbox_inches="tight")
plt.show()

print(f"Global 3D scatterplot saved to {output_file}")
