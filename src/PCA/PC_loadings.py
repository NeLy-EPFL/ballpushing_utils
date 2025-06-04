import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the PCA loadings
loadings_df = pd.read_csv("pca_loadings.csv", index_col=0)

# Select the top N PCs to visualize
top_n_pcs = 5  # Change as needed
pcs_to_plot = loadings_df.columns[:top_n_pcs]

# For each PC, plot the top positive and negative loadings
for pc in pcs_to_plot:
    # Sort by absolute loading value
    sorted_loadings = loadings_df[pc].sort_values(key=abs, ascending=False)
    top_features = sorted_loadings.head(10)  # Show top 10 features

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="vlag")
    plt.title(f"Top loadings for {pc}")
    plt.xlabel("Loading value")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(f"loadings_{pc}.png")
    plt.show()
    plt.close()
