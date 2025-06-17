import argparse
from pathlib import Path
from Ballpushing_utils import Fly, Experiment, Dataset
from utils_behavior import Utils
import cProfile
import pstats
import json


def process_fly(fly_path):
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    # Get the fly's interaction-based event metrics using Dataset
    event_dataset = Dataset(ExampleFly, dataset_type="event_metrics")

    # Prepare results
    if event_dataset.data is not None:
        results = event_dataset.data.to_dict(orient="list")
    else:
        results = "No event metrics found."

    print(json.dumps(results, indent=4))
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "interaction_event_metrics_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_dir / 'interaction_event_metrics_results.json'}")


def process_experiment(experiment_path):
    # Load an example experiment
    ExampleExperiment = Experiment(experiment_path)

    # Make a dataset using the experiment's ballpushing metrics
    ExampleData_summary = Dataset(ExampleExperiment, dataset_type="summary")

    ExampleData_individual = Dataset(ExampleExperiment, dataset_type="event_metrics")

    # Print the dataset
    if ExampleData_summary.data is not None:
        print(ExampleData_summary.data.head())
    else:
        print("ExampleData.data is None. Cannot call 'head()'.")

    if ExampleData_individual.data is not None:
        print(ExampleData_individual.data.head())
    else:
        print("ExampleData.data is None. Cannot call 'head()'.")


def check_metric(metric, path, mode):
    """Load a fly or experiment and check the interaction-based event metrics.

    Args:
        metric (str): The metric to check.
        path (Path): The path to the fly or experiment data.
        mode (str): The mode of operation ("fly" or "experiment").
    """
    # Load the appropriate object based on the mode
    if mode == "fly":
        data_object = Fly(path, as_individual=True)
    elif mode == "experiment":
        data_object = Experiment(path)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'fly' or 'experiment'.")

    # Build the event_metrics dataset
    event_dataset = Dataset(data_object, dataset_type="event_metrics")

    # Check if the event_metrics dataset contains data
    if event_dataset.data is not None:
        if metric:
            # If a specific metric is provided, print its values
            if metric in event_dataset.data.columns:
                print(f"Interaction-based Metric '{metric}' values:")
                print(event_dataset.data[metric].head())
            else:
                print(f"Interaction-based Metric '{metric}' not found in the event_metrics dataset.")
        else:
            # If no metric is provided, print all metrics
            print("No specific metric provided. Printing all interaction-based event metrics:")
            for column in event_dataset.data.columns:
                print(f"\nInteraction-based Metric '{column}' values:")
                print(event_dataset.data[column].head())
    else:
        print("Event metrics dataset is empty. Cannot check metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single fly or an experiment.")
    parser.add_argument(
        "--mode",
        choices=["fly", "experiment"],
        required=True,
        help="Specify whether to process a single fly or an experiment.",
    )
    parser.add_argument("--path", required=True, help="Path to the fly or experiment data.")
    parser.add_argument(
        "--metric", required=False, help="The metric to check. If not provided, all metrics will be printed."
    )

    args = parser.parse_args()

    check_metric(args.metric, Path(args.path), args.mode)
