import argparse
from pathlib import Path
from Ballpushing_utils import Fly, Experiment, Dataset
from utils_behavior import Utils


def process_fly(fly_path):
    """
    Process a single fly and ensure the transposed dataset is properly generated.

    Args:
        fly_path (str): Path to the fly data.
    """
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    BehaviorMap = Dataset(ExampleFly, dataset_type="behavior_umap")

    # Print the transposed dataset
    if BehaviorMap.data is not None:
        print("behavior umap:")
        print(BehaviorMap.data.head())
    else:
        print("transposedData.data is None. Cannot call 'head()'.")

    return BehaviorMap


def process_experiment(experiment_path):
    """
    Process an experiment and ensure the transposed dataset is properly generated.

    Args:
        experiment_path (str): Path to the experiment data.
    """
    # Load an example experiment
    ExampleExperiment = Experiment(experiment_path)

    # Make a transposed dataset using the experiment's ballpushing metrics
    transposedData = Dataset(ExampleExperiment, dataset_type="transposed")

    # Print the transposed dataset
    if transposedData.data is not None:
        print("transposed dataset:")
        print(transposedData.data.head())
    else:
        print("transposedData.data is None. Cannot call 'head()'.")

    return transposedData


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single fly or an experiment.")
    parser.add_argument(
        "--mode",
        choices=["fly", "experiment"],
        required=True,
        help="Specify whether to process a single fly or an experiment.",
    )
    parser.add_argument("--path", required=True, help="Path to the fly or experiment data.")

    args = parser.parse_args()

    if args.mode == "fly":
        BehaviorMap = process_fly(Path(args.path))

    elif args.mode == "experiment":
        BehaviorMap = process_experiment(Path(args.path))

    # Save the transposed dataset in a csv file
    if BehaviorMap.data is not None:
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)  # Ensure the 'outputs' directory exists
        output_path = output_dir / "test_umap.csv"
        BehaviorMap.data.to_csv(output_path, index=False)
        print(f"Column names saved to {output_path}")
    else:
        print("No data available to save.")
