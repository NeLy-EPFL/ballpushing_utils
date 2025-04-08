import argparse
from pathlib import Path
from Ballpushing_utils import Fly, Experiment, Dataset
from utils_behavior import Utils


def process_fly(fly_path):
    """
    Process a single fly and ensure the transformed dataset is properly generated.

    Args:
        fly_path (str): Path to the fly data.
    """
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    # Get the fly's interaction metrics
    if ExampleFly.event_metrics is not None:
        print(f"Individual events metrics: {ExampleFly.event_metrics}")

    # Get the fly's ballpushing metrics
    if ExampleFly.event_summaries is not None:
        print(f"Summary metrics: {ExampleFly.event_summaries}")

    # Make a transformed dataset using the fly's ballpushing metrics
    TransformedData = Dataset(ExampleFly, dataset_type="transformed")

    # Print the transformed dataset
    if TransformedData.data is not None:
        print("Transformed dataset:")
        print(TransformedData.data.head())
    else:
        print("TransformedData.data is None. Cannot call 'head()'.")


def process_experiment(experiment_path):
    """
    Process an experiment and ensure the transformed dataset is properly generated.

    Args:
        experiment_path (str): Path to the experiment data.
    """
    # Load an example experiment
    ExampleExperiment = Experiment(experiment_path)

    # Make a transformed dataset using the experiment's ballpushing metrics
    TransformedData = Dataset(ExampleExperiment, dataset_type="transformed")

    # Print the transformed dataset
    if TransformedData.data is not None:
        print("Transformed dataset:")
        print(TransformedData.data.head())
    else:
        print("TransformedData.data is None. Cannot call 'head()'.")


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
        process_fly(Path(args.path))
    elif args.mode == "experiment":
        process_experiment(Path(args.path))
