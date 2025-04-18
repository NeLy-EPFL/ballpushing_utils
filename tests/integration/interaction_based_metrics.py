import argparse
from pathlib import Path
from Ballpushing_utils import Fly, Experiment, Dataset
from utils_behavior import Utils
import cProfile
import pstats


def process_fly(fly_path):
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    # Get the fly's interaction metrics

    if ExampleFly.event_metrics is not None:
        print(f"Individual events metrics: {ExampleFly.event_metrics}")

    # Get the fly's ballpushing metrics
    if ExampleFly.event_summaries is not None:
        print(f"Summary metrics: {ExampleFly.event_summaries}")

    # Make a dataset using the fly's ballpushing metrics
    ExampleData = Dataset(ExampleFly, dataset_type="summary")

    # Print the dataset
    if ExampleData.data is not None:
        print(ExampleData.data.head())
    else:
        print("ExampleData.data is None. Cannot call 'head()'.")


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
    """Load a fly or experiment and check the metric.

    Current metrics:
        "nb_events",
        "max_event",
        "max_event_time",
        "max_distance",
        "final_event",
        "final_event_time",
        "nb_significant_events",
        "significant_ratio",
        "first_significant_event",
        "first_significant_event_time",
        "major_event",
        "major_event_time",
        "major_event_first",
        "insight_effect",
        "insight_effect_log",
        "cumulated_breaks_duration",
        "chamber_time",
        "chamber_ratio",
        "pushed",
        "pulled",
        "pulling_ratio",
        "success_direction",
        "interaction_proportion",
        "distance_moved",
        "exit_time"
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

    # Build the dataset
    dataset = Dataset(data_object, dataset_type="summary")

    # Check if the dataset contains data
    if dataset.data is not None:
        if metric:
            # If a specific metric is provided, print its values
            if metric in dataset.data.columns:
                print(f"Metric '{metric}' values:")
                print(dataset.data[metric].head())
            else:
                print(f"Metric '{metric}' not found in the dataset.")
        else:
            # If no metric is provided, print all metrics
            print("No specific metric provided. Printing all metrics:")
            for column in dataset.data.columns:
                print(f"\nMetric '{column}' values:")
                print(dataset.data[column].head())
    else:
        print("Dataset is empty. Cannot check metrics.")

    individual_dataset = Dataset(data_object, dataset_type="event_metrics")

    # Check if the individual dataset contains data
    if individual_dataset.data is not None:
        if metric:
            # If a specific metric is provided, print its values
            if metric in individual_dataset.data.columns:
                print(f"Individual Metric '{metric}' values:")
                print(individual_dataset.data[metric].head())
            else:
                print(f"Individual Metric '{metric}' not found in the dataset.")
        else:
            # If no metric is provided, print all metrics
            print("No specific metric provided. Printing all metrics:")
            for column in individual_dataset.data.columns:
                print(f"\nIndividual Metric '{column}' values:")
                print(individual_dataset.data[column].head())
    else:
        print("Individual dataset is empty. Cannot check metrics.")


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
