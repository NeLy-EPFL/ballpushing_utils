import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from itertools import product
from Ballpushing_utils.behavior_umap import BehaviorUMAP
from itertools import chain, combinations
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import os


class UMAPOptimizer:
    def __init__(self, dataset_path, output_path, param_grid):
        """
        Initialize the UMAPOptimizer class.

        Args:
            dataset_path (str): Path to the input dataset.
            output_path (str): Path to save the optimization results.
            param_grid (dict): Dictionary of UMAP and clustering parameters to test.
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.param_grid = param_grid
        self.results = []

        # Load existing results if the file exists
        if os.path.exists(self.output_path):
            self.existing_results = pd.read_csv(self.output_path)
        else:
            self.existing_results = pd.DataFrame()

    def has_been_tested(self, param_dict):
        """
        Check if a parameter combination has already been tested.

        Args:
            param_dict (dict): Parameter combination to check.

        Returns:
            bool: True if the combination has been tested, False otherwise.
        """
        if self.existing_results.empty:
            return False

        # Iterate through each row in the existing results
        for _, row in self.existing_results.iterrows():
            match = True
            for key, value in param_dict.items():
                if key not in row:
                    match = False
                    break
                # Handle list or non-scalar values
                if isinstance(value, list):
                    # Convert the string representation in the CSV back to a list for comparison
                    try:
                        csv_value = eval(row[key]) if not pd.isna(row[key]) else None
                    except Exception:
                        csv_value = None
                    if csv_value != value:
                        match = False
                        break
                else:
                    # Compare scalar values directly
                    if row[key] != value:
                        match = False
                        break
            if match:
                return True

        return False

    def evaluate_clustering(self, embeddings, cluster_labels):
        """
        Evaluate the quality of clustering using various metrics.

        Args:
            embeddings (np.ndarray): UMAP embeddings.
            cluster_labels (np.ndarray): Cluster labels.

        Returns:
            dict: Dictionary of clustering quality metrics.
        """
        metrics = {}
        metrics["silhouette_score"] = silhouette_score(embeddings, cluster_labels)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(embeddings, cluster_labels)
        metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, cluster_labels)
        return metrics

    def rank_results(self, top_n=3):
        """
        Rank and display the top parameter combinations based on each metric.

        Args:
            top_n (int): Number of top combinations to display for each metric.
        """
        if not self.results:
            print("No results to rank. Run the optimizer first.")
            return

        # Convert results to a DataFrame
        results_df = pd.DataFrame(self.results)

        # Define metrics and their sorting order
        metrics = {
            "silhouette_score": False,  # Higher is better
            "calinski_harabasz_score": False,  # Higher is better
            "davies_bouldin_score": True,  # Lower is better
        }

        print(f"\nTop {top_n} parameter combinations for each metric:\n")
        for metric, ascending in metrics.items():
            if metric in results_df.columns:
                print(f"--- {metric} (sorted {'ascending' if ascending else 'descending'}) ---")
                top_combinations = results_df.sort_values(by=metric, ascending=ascending).head(top_n)
                print(
                    top_combinations[
                        ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
                        + list(self.param_grid.keys())
                    ]
                )
                print("\n")

    def save_plots(self, embeddings, cluster_labels, param_dict, output_dir):
        """
        Save scatterplot and density plot of UMAP embeddings.

        Args:
            embeddings (np.ndarray): UMAP embeddings.
            cluster_labels (np.ndarray): Cluster labels.
            param_dict (dict): Dictionary of parameters for the current run.
            output_dir (str): Directory to save the plots.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Convert embeddings to a DataFrame for plotting
        umap_columns = [f"UMAP{i+1}" for i in range(embeddings.shape[1])]
        df = pd.DataFrame(embeddings, columns=umap_columns)
        df["cluster"] = cluster_labels

        # Generate a unique filename based on parameters
        param_str = "_".join([f"{key}={value}" for key, value in param_dict.items()])
        safe_param_str = param_str.replace("/", "-").replace(" ", "_")  # Make filename safe

        # Scatterplot
        scatterplot_path = os.path.join(output_dir, f"scatter_{safe_param_str}.png")
        if embeddings.shape[1] == 2:
            # 2D scatterplot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="cluster", palette="viridis", s=10)
            plt.title(f"Scatterplot: {param_str}")
            plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(scatterplot_path)
            plt.close()
            print(f"2D Scatterplot saved to {scatterplot_path}")
        elif embeddings.shape[1] == 3:
            # 3D scatterplot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(df["UMAP1"], df["UMAP2"], df["UMAP3"], c=df["cluster"], cmap="viridis", s=10)
            ax.set_title(f"3D Scatterplot: {param_str}")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_zlabel("UMAP3")
            fig.colorbar(scatter, ax=ax, label="Cluster")
            plt.tight_layout()
            plt.savefig(scatterplot_path)
            plt.close()
            print(f"3D Scatterplot saved to {scatterplot_path}")

        # Density plot (only for 2D embeddings)
        if embeddings.shape[1] == 2:
            densityplot_path = os.path.join(output_dir, f"density_{safe_param_str}.png")
            plt.figure(figsize=(8, 6))
            sns.kdeplot(data=df, x="UMAP1", y="UMAP2", fill=True, cmap="viridis")
            plt.title(f"Density Plot: {param_str}")
            plt.tight_layout()
            plt.savefig(densityplot_path)
            plt.close()
            print(f"Density plot saved to {densityplot_path}")

    def optimize(self):
        """
        Run the optimization process by testing all parameter combinations.
        """
        # Load the dataset
        print(f"Loading dataset from {self.dataset_path}...")
        data = pd.read_feather(self.dataset_path)

        # Replace any value that's 9999 with NaN
        print("Replacing 9999 with 0...")
        data = data.replace(9999, 0)

        # Check if there is any remaining 9999 and check if there are any NaN values

        if (data == 9999).any().any():
            print("Warning: 9999 values still present in the dataset.")
        if data.isna().any().any():
            print("Warning: NaN values present in the dataset.")

        # Generate all parameter combinations
        param_combinations = list(product(*self.param_grid.values()))
        param_names = list(self.param_grid.keys())

        print(f"Testing {len(param_combinations)} parameter combinations...")

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            print(f"Testing parameters: {param_dict}")

            # Skip if the parameter combination has already been tested
            if self.has_been_tested(param_dict):
                print(f"Skipping already tested parameters: {param_dict}")
                continue

            # Initialize BehaviorUMAP with current parameters
            umap_runner = BehaviorUMAP(
                n_neighbors=param_dict["n_neighbors"],
                min_dist=param_dict["min_dist"],
                n_components=param_dict["n_components"],
                n_clusters=param_dict["n_clusters"],
                filter_features=param_dict["filter_features"],
            )

            # Generate UMAP embeddings and clusters
            try:
                result_df = umap_runner.generate_umap_and_clusters(
                    data=data,
                    feature_groups=param_dict["feature_groups"],
                    include_ball=param_dict["include_ball"],
                    use_pca=param_dict["use_pca"],
                )

                embeddings = result_df[[f"UMAP{i+1}" for i in range(param_dict["n_components"])]].values
                cluster_labels = result_df["cluster"].values

                # Evaluate clustering quality
                metrics = self.evaluate_clustering(embeddings, cluster_labels)
                metrics.update(param_dict)  # Add parameters to the metrics
                self.results.append(metrics)

                # Save plots for visual inspection
                output_dir = os.path.join(os.path.dirname(self.output_path), "plots")
                self.save_plots(embeddings, cluster_labels, param_dict, output_dir)

                print(f"Metrics: {metrics}")
            except Exception as e:
                print(f"Error with parameters {param_dict}: {e}")

        # Save results to a CSV file
        if self.results:
            new_results_df = pd.DataFrame(self.results)
            if not self.existing_results.empty:
                combined_results = pd.concat([self.existing_results, new_results_df], ignore_index=True)
            else:
                combined_results = new_results_df
            combined_results.to_csv(self.output_path, index=False)
            print(f"Optimization results saved to {self.output_path}")
        else:
            print("No new results to save.")


if __name__ == "__main__":
    # Define the parameter grid
    features = [ "frame", "statistical", "fourier"]
    all_combinations = list(chain.from_iterable(combinations(features, r) for r in range(1, len(features) + 1)))
    all_combinations = [list(comb) for comb in all_combinations]

    sub_combinations = [["frame"], ["frame", "statistical"], ["frame", "statistical", "fourier"]]

    param_grid = {
        "n_neighbors": [50],
        "min_dist": [0.1],
        "n_components": [2],
        "n_clusters": [10],
        "filter_features": [False, True],
        "feature_groups": all_combinations,
        "include_ball": [False],
        "use_pca": [False, True],
    }

    # Paths
    dataset_path = "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250411_Transposed_Ctrls_Data/transposed/pooled_transposed.feather"
    output_path = (
        "/home/matthias/ballpushing_utils/tests/integration/outputs/umap_optimisation_withNan/umap_optimization_results.csv"
    )

    # Run the optimizer
    optimizer = UMAPOptimizer(dataset_path, output_path, param_grid)
    optimizer.optimize()

    # Rank and display the top combinations
    optimizer.rank_results(top_n=3)
