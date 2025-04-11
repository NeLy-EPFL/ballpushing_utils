import pandas as pd
import argparse
import os
from Ballpushing_utils.behavior_umap import BehaviorUMAP

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run UMAP on a dataset and log results.")
    parser.add_argument("input_file", type=str, help="Path to the input feather dataset.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the output CSV.")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance between points in UMAP.")
    parser.add_argument("--n_components", type=int, default=2, help="Number of dimensions for UMAP output.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMeans.")
    parser.add_argument("--use_pca", action="store_true", help="Whether to apply PCA before UMAP.")
    parser.add_argument("--filter_features", action="store_true", help="Enable feature filtering.")
    parser.add_argument("--include_ball", action="store_true", help="Include ball-related features.")
    parser.add_argument("--feature_groups", type=str, nargs="+", default=None, help="Feature groups to include.")
    args = parser.parse_args()

    # Load dataset
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return

    print(f"Loading dataset from {args.input_file}...")
    data = pd.read_feather(args.input_file)

    if args.output_file is None:
        # Set output file to input directory with umap name
        input_dir = os.path.dirname(args.input_file)
        input_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = os.path.join(input_dir, f"{input_name}_umap.feather")
        print(f"Output file set to {args.output_file}")

    # Initialize BehaviorUMAP
    umap_runner = BehaviorUMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        filter_features=args.filter_features,
    )

    # Run UMAP and clustering
    print("Running UMAP and clustering...")
    result_df = umap_runner.generate_umap_and_clusters(
        data=data,
        feature_groups=args.feature_groups,
        include_ball=args.include_ball,
        use_pca=args.use_pca,
        savepath=args.output_file,
    )

    print(f"UMAP and clustering completed. Results saved to {args.output_file}.")

if __name__ == "__main__":
    main()