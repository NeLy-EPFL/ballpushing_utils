import pandas as pd
import argparse
import os
from Ballpushing_utils.behavior_umap import BehaviorUMAP
import json
import yaml


def load_config(config_file):
    """
    Load configuration from a JSON or YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_file, "r") as f:
        if config_file.endswith(".json"):
            return json.load(f)
        elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            raise ValueError("Unsupported configuration file format. Use JSON or YAML.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run UMAP on a dataset and log results.")
    parser.add_argument("--config", type=str, help="Path to the configuration file (JSON or YAML).")
    parser.add_argument("--input_file", type=str, help="Path to the input feather dataset.")
    parser.add_argument("--output_file", type=str, help="Path to save the output CSV.")
    parser.add_argument("--n_neighbors", type=int, help="Number of neighbors for UMAP.")
    parser.add_argument("--min_dist", type=float, help="Minimum distance between points in UMAP.")
    parser.add_argument("--n_components", type=int, help="Number of dimensions for UMAP output.")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters for KMeans.")
    parser.add_argument("--use_pca", action="store_true", help="Whether to apply PCA before UMAP.")
    parser.add_argument("--filter_features", action="store_true", help="Enable feature filtering.")
    parser.add_argument("--include_ball", action="store_true", help="Include ball-related features.")
    parser.add_argument("--feature_groups", type=str, nargs="+", help="Feature groups to include.")
    parser.add_argument("--video_output_dir", type=str, help="Directory to save cluster videos.")
    parser.add_argument("--interaction_data", type=str, help="Path to interaction data for video generation.")
    parser.add_argument("--best_disp", action="store_true", help="Use events with the highest displacement for videos.")
    args = parser.parse_args()

    # Load configuration from file if provided
    config = vars(args)  # Convert argparse Namespace to dictionary
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file '{args.config}' does not exist.")
            return
        print(f"Loading configuration from {args.config}...")
        file_config = load_config(args.config)
        # Update config with values from the file, overriding defaults
        config.update({k: v for k, v in file_config.items() if v is not None})

    # Ensure required arguments are present
    if not config.get("input_file"):
        print("Error: 'input_file' is required.")
        return

    # Load dataset
    input_file = config["input_file"]
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    print(f"Loading dataset from {input_file}...")
    data = pd.read_feather(input_file)

    # Set default output file if not provided
    output_file = config.get("output_file")
    if not output_file:
        input_dir = os.path.dirname(input_file)
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(input_dir, f"{input_name}_umap.feather")
        print(f"Output file set to {output_file}")

    # Initialize BehaviorUMAP
    umap_runner = BehaviorUMAP(
        n_neighbors=config.get("n_neighbors", 15),
        min_dist=config.get("min_dist", 0.1),
        n_components=config.get("n_components", 2),
        n_clusters=config.get("n_clusters", 3),
        filter_features=config.get("filter_features", False),
    )

    # Run UMAP and clustering
    print("Running UMAP and clustering...")
    result_df = umap_runner.generate_umap_and_clusters(
        data=data,
        feature_groups=config.get("feature_groups"),
        include_ball=config.get("include_ball", False),
        use_pca=config.get("use_pca", False),
        savepath=output_file,
    )

    print(f"UMAP and clustering completed. Results saved to {output_file}.")

    # Generate cluster videos if interaction data and output directory are provided
    video_output_dir = config.get("video_output_dir")
    interaction_data_path = config.get("interaction_data")
    if video_output_dir and interaction_data_path:
        if not os.path.exists(interaction_data_path):
            print(f"Error: Interaction data file '{interaction_data_path}' does not exist.")
            return

        print(f"Loading interaction data from {interaction_data_path}...")
        interaction_data = pd.read_feather(interaction_data_path)

        print(f"Generating cluster videos in {video_output_dir}...")
        umap_runner.generate_cluster_videos(
            Umap=result_df,
            interaction_data=interaction_data,
            output_dir=video_output_dir,
            best_disp=config.get("best_disp", False),
        )
        print("Cluster videos generated successfully.")
    elif video_output_dir or interaction_data_path:
        print("Warning: Both 'video_output_dir' and 'interaction_data' must be provided to generate cluster videos.")


if __name__ == "__main__":
    main()