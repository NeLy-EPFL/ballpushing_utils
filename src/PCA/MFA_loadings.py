import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MFA loadings (from R)
loadings_df = pd.read_csv("mfa_block_loadings.csv", index_col=0)

# Select the top N dimensions to visualize
top_n_dims = 5  # Change as needed
dims_to_plot = loadings_df.columns[:top_n_dims]

for dim in dims_to_plot:
    # Sort by absolute loading value
    sorted_loadings = loadings_df[dim].sort_values(key=abs, ascending=False)
    top_features = sorted_loadings.head(10)  # Show top 10 features

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="vlag")
    plt.title(f"Top loadings for {dim}")
    plt.xlabel("Loading value")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(f"mfa_loadings_{dim}.png")
    plt.show()
    plt.close()
