from __future__ import annotations
from Ballpushing_utils.config import Config
from Ballpushing_utils.behavior_umap import BehaviorUMAP

import os
import pandas as pd
from collections import Counter
import traceback
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.fft import fft
import re


from moviepy.editor import (
    VideoFileClip,
    clips_array,
    ColorClip,
)


class Dataset:
    def __init__(
        self,
        source,
        brain_regions_path="/mnt/upramdya_data/MD/Region_map_240312.csv",
        dataset_type="coordinates",
    ):
        """
        A class to generate a Dataset from Experiments and Fly objects.

        It is in essence a list of Fly objects that can be used to generate a pandas DataFrame containing chosen metrics for each fly.

        Parameters
        ----------
        source : can either be a list of Experiment objects, one Experiment object, a list of Fly objects or one Fly object.

        """
        self.config = Config()

        self.source = source

        # Define the experiments and flies attributes
        if isinstance(source, list):
            # If the source is a list, check if it contains Experiment or Fly objects, otherwise raise an error
            if type(source[0]).__name__ == "Experiment":
                # If the source contains Experiment objects, generate a dataset from the experiments
                self.experiments = source

                self.flies = [fly for experiment in self.experiments for fly in experiment.flies]

            elif type(source[0]).__name__ == "Fly":
                # make a list of distinct experiments associated with the flies
                self.experiments = list(set([fly.experiment for fly in source]))

                self.flies = source

            else:
                raise TypeError(
                    "Invalid source format: source must be a (list of) Experiment objects or a list of Fly objects"
                )

        elif type(source).__name__ == "Experiment":
            # If the source is an Experiment object, generate a dataset from the experiment
            self.experiments = [source]

            self.flies = source.flies

        elif type(source).__name__ == "Fly":
            # If the source is a Fly object, generate a dataset from the fly
            self.experiments = [source.experiment]

            self.flies = [source]
        else:
            raise TypeError(
                "Invalid source format: source must be a (list of) Experiment objects or a list of Fly objects"
            )

        self.flies = [fly for fly in self.flies if fly._tracking_data.valid_data]

        self.brain_regions_path = brain_regions_path
        self.regions_map = pd.read_csv(self.brain_regions_path)

        self.metadata = []

        self.data = None

        self.dataset_type = dataset_type

        self.generate_dataset(metrics=dataset_type)

    def __str__(self):
        # Look for recurring words in the experiment names
        experiment_names = [experiment.directory.name for experiment in self.experiments]
        experiment_names = "_".join(experiment_names)
        experiment_names = experiment_names.split("_")  # Split by "_"

        # Ignore certain labels
        labels_to_ignore = {"Tracked", "Videos"}
        experiment_names = [name for name in experiment_names if name not in labels_to_ignore]

        # Ignore words that are found only once
        experiment_names = [name for name in experiment_names if experiment_names.count(name) > 1]

        experiment_names = Counter(experiment_names)
        experiment_names = experiment_names.most_common(3)
        experiment_names = [name for name, _ in experiment_names]
        experiment_names = ", ".join(experiment_names)

        return (
            f"Dataset with {len(self.flies)} flies and {len(self.experiments)} experiments\nkeyword: {experiment_names}"
        )

    def __repr__(self):

        from Ballpushing_utils import Experiment, Fly

        # Adapt the repr function to the source attribute
        # If the source is a list, check if it is Fly or Experiment objects
        if isinstance(self.source, list):
            if isinstance(self.source[0], Experiment):
                return f"Dataset({[experiment.directory for experiment in self.experiments]})"
            elif isinstance(self.source[0], Fly):
                return f"Dataset({[fly.directory for fly in self.flies]})"
        elif isinstance(self.source, Experiment):
            return f"Dataset({self.experiments[0].directory})"
        elif isinstance(self.source, Fly):
            return f"Dataset({self.flies[0].directory})"
        return "Dataset()"

    def find_flies(self, on, value):
        """
        Makes a list of Fly objects matching a certain criterion.

        Parameters
        ----------
        on : str
            The name of the attribute to filter on. Can be a nested attribute (e.g., 'metadata.name').

        value : str
            The value of the attribute to filter on.

        Returns
        ----------
        list
            A list of Fly objects matching the criterion.
        """

        def get_nested_attr(obj, attr):
            """Helper function to get nested attributes."""
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
            return obj

        return [fly for fly in self.flies if get_nested_attr(fly, on) == value]

    def generate_dataset(self, metrics="coordinates"):
        """Generates a pandas DataFrame from a list of Experiment objects. The dataframe contains the smoothed fly and ball positions for each experiment.

        Args:
            experiments (list): A list of Experiment objects.
            metrics (str): The kind of dataset to generate. Currently, the following metrics are available:
            - 'coordinates': The fly and ball coordinates for each frame.
            - 'summary': Summary metrics for each fly. These are single values for each fly (e.g. number of events, duration of the breaks between events, etc.). A list of available summary metrics can be found in _prepare_dataset_summary_metrics documentation.

        Returns:
            dict: A dictionary where keys are ball types and values are DataFrames containing selected metrics for each fly and associated metadata.
        """

        Dataset = []

        try:
            if metrics == "coordinates":
                for fly in self.flies:
                    data = self._prepare_dataset_coordinates(fly)
                    Dataset.append(data)

            elif metrics == "contact_data":
                for fly in self.flies:
                    data = self._prepare_dataset_contact_data(fly)
                    if not data.empty:
                        Dataset.append(data)

            elif metrics == "event_metrics":
                for fly in self.flies:
                    data = self._prepare_dataset_event_metrics(fly)
                    if data is not None and hasattr(data, "empty") and not data.empty:
                        Dataset.append(data)
                    else:
                        print(f"[WARN] Skipping fly {getattr(fly, 'directory', None)}: No valid event metrics.")
            elif metrics == "summary":
                for fly in self.flies:
                    # print("Preparing dataset for", fly.name)
                    data = self._prepare_dataset_summary_metrics(fly)
                    # print(f"Data : {data}")
                    Dataset.append(data)

            elif metrics == "F1_coordinates":
                for fly in self.flies:
                    data = self._prepare_dataset_F1_coordinates(fly)
                    Dataset.append(data)

            elif metrics == "F1_checkpoints":
                for fly in self.flies:
                    data = self._prepare_dataset_F1_checkpoints(fly)
                    Dataset.append(data)
            elif metrics == "Skeleton_contacts":
                for fly in self.flies:
                    data = self._prepare_dataset_skeleton_contacts(fly)
                    Dataset.append(data)
            elif metrics == "standardized_contacts":
                for fly in self.flies:
                    data = self._prepare_dataset_standardized_contacts(fly)
                    if not data.empty:
                        Dataset.append(data)

            elif metrics == "transposed":
                for fly in self.flies:
                    data = self._prepare_framewise_dataset(fly)
                    if not data.empty:
                        Dataset.append(data)

            elif metrics == "behavior_umap":
                data = self.generate_UMAP()
                if data is not None:
                    self.data = data
                    return self.data

            elif metrics == "transformed":
                for fly in self.flies:
                    data = self._prepare_transformed_dataset(fly)
                    Dataset.append(data)
                    if not data.empty:
                        Dataset.append(data)

            if Dataset:
                self.data = pd.concat(Dataset).reset_index()
            else:
                self.data = pd.DataFrame()  # Return an empty DataFrame if no data

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            self.data = pd.DataFrame()  # Return an empty DataFrame in case of error

        return self.data

    def generate_UMAP(self):

        # Check if self.data already exists and if self.dataset_type is "transposed"
        if self.data is None or self.dataset_type != "transposed":
            print("No data available for UMAP generation. Generating dataset...")
            input_data = self.generate_dataset(metrics="transposed")
        else:
            input_data = self.data

        # Check if the input data is empty
        if input_data.empty:
            print("Input data is empty. Cannot generate UMAP.")
            return None

        # Create the behavior UMAP object
        behavior_umap = BehaviorUMAP()

        # Generate UMAP embeddings
        embeddings = behavior_umap.generate_umap_and_clusters(input_data)

        return embeddings

    def _prepare_dataset_coordinates(self, fly, downsampling_factor=None, annotate_events=True):
        """
        Helper function to prepare individual fly dataset with fly and ball coordinates. It also adds the fly name, experiment name, and arena metadata as categorical data.

        Args:
            fly (Fly): A Fly object.
            downsampling_factor (int): The factor (in seconds) by which to downsample the dataset. Defaults to None.
            annotate_events (bool): Whether to annotate the dataset with interaction events. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's coordinates and associated metadata.
        """
        downsampling_factor = fly.config.downsampling_factor or downsampling_factor

        # Extract fly and ball tracking data
        flydata = [obj.dataset for obj in fly.tracking_data.flytrack.objects]
        balldata = [obj.dataset for obj in fly.tracking_data.balltrack.objects]

        # Initialize dataset with time and frame columns
        dataset = pd.DataFrame(
            {
                "time": flydata[0]["time"],
                "frame": flydata[0]["frame"],
                "adjusted_time": (fly.f1_metrics["adjusted_time"] if fly.tracking_data.exit_time else np.nan),
            }
        )

        # Add fly coordinates and distances
        for i, data in enumerate(flydata):
            dataset[f"x_fly_{i}"] = data["x_thorax"] - fly.tracking_data.start_x
            dataset[f"y_fly_{i}"] = data["y_thorax"] - fly.tracking_data.start_y
            dataset[f"distance_fly_{i}"] = np.sqrt(dataset[f"x_fly_{i}"].pow(2) + dataset[f"y_fly_{i}"].pow(2))

        # Add ball coordinates and distances
        for i, data in enumerate(balldata):
            dataset[f"x_ball_{i}"] = data["x_centre"] - fly.tracking_data.start_x
            dataset[f"y_ball_{i}"] = data["y_centre"] - fly.tracking_data.start_y
            dataset[f"distance_ball_{i}"] = np.sqrt(dataset[f"x_ball_{i}"].pow(2) + dataset[f"y_ball_{i}"].pow(2))

        # Downsample the dataset if required
        if downsampling_factor:
            dataset = dataset.iloc[:: downsampling_factor * fly.experiment.fps]

        # Annotate interaction events if required
        if annotate_events:
            self._annotate_interaction_events(dataset, fly)

        # Add trial information for learning experiments
        if fly.config.experiment_type == "Learning" and hasattr(fly, "learning_metrics"):
            self._add_trial_information(dataset, fly)

        # Add metadata
        dataset = self._add_metadata(dataset, fly)

        return dataset

    # TODO : Events should be reannotated if the dataset is subsetted

    # TODO: implement events durations

    def _annotate_interaction_events(self, dataset, fly):
        """
        Annotate the dataset with interaction events and their onsets.

        Args:
            dataset (pd.DataFrame): The dataset to annotate.
            fly (Fly): A Fly object.
        """
        interaction_events = fly.tracking_data.interaction_events
        event_frames = np.full(len(dataset), np.nan)
        event_onset_frames = np.full(len(dataset), np.nan)

        event_index = 0
        for fly_idx, ball_events in interaction_events.items():
            for ball_idx, events in ball_events.items():
                for event in events:
                    event_frames[event[0] : event[1]] = event_index
                    event_index += 1

        dataset["interaction_event"] = event_frames

        # Annotate interaction event onsets
        event_onset = 0
        for (
            fly_idx,
            ball_idx,
        ), onsets in fly.tracking_data.interactions_onsets.items():
            for onset in onsets:
                if onset is not None and 0 <= onset < len(event_onset_frames):
                    event_onset_frames[onset] = event_onset
                    event_onset += 1

        dataset["interaction_event_onset"] = event_onset_frames

    def _prepare_dataset_contact_data(self, fly, hidden_value=None):
        if hidden_value is None:
            hidden_value = self.config.hidden_value

        all_contact_data = []
        contact_indices = []

        if fly.skeleton_metrics is None:
            print(f"No skeleton metrics found for fly {fly.metadata.name}. Skipping...")
            return pd.DataFrame()  # Return an empty DataFrame
        else:
            skeleton_data = fly.tracking_data.skeletontrack.objects[0].dataset
            ball_data = fly.tracking_data.balltrack.objects[0].dataset

            contact_events = fly.skeleton_metrics.all_contacts

            # Apply success_cutoff if enabled
            if fly.config.success_cutoff:

                # print("applying success cutoff to dataset")

                final_contact_idx, _ = fly.skeleton_metrics.get_final_contact()
                if final_contact_idx is not None:
                    contact_events = contact_events[: final_contact_idx + 1]

                    # print(f"Final contact index in dataset: {final_contact_idx}")
                    # print(
                    #     f"Applying success cutoff. Number of contact events after cutoff: {len(contact_events)}"
                    # )

            for event_idx, event in enumerate(contact_events):
                start_idx, end_idx = event[0], event[1]
                contact_data = skeleton_data.iloc[start_idx:end_idx]
                event_ball_data = ball_data.iloc[start_idx:end_idx]
                contact_data["contact_index"] = event_idx  # Add contact_index column

                # Add ball data to the contact data
                for col in event_ball_data.columns:
                    contact_data[col] = event_ball_data[col].values

                all_contact_data.append(contact_data)
                contact_indices.append(event_idx)

            if all_contact_data:
                combined_data = pd.concat(all_contact_data, ignore_index=True)
                combined_data.fillna(hidden_value, inplace=True)
                combined_data = self._add_metadata(combined_data, fly)
            else:
                combined_data = pd.DataFrame()

            # nb_events = len(contact_indices)
            # print(f"Number of contact events: {nb_events}")

        return combined_data

    def _prepare_dataset_event_metrics(self, fly):
        """
        Prepare a dataset with all events and their associated metrics for a given fly.

        Args:
            fly (Fly): A Fly object.

        Returns:
            pandas.DataFrame: A DataFrame containing all events and their associated metrics.
        """
        # Initialize the InteractionsMetrics object
        try:
            interaction_metrics = fly.event_metrics
        except Exception as e:
            print(f"[WARN] Could not compute event_metrics for fly {getattr(fly, 'directory', None)}: {e}")
            return None

        if interaction_metrics is None:
            print(f"[WARN] event_metrics is None for fly {getattr(fly, 'directory', None)}")
            return None

        # Preprocess keys and events into a flat structure
        event_data = []
        for key, events in interaction_metrics.items():
            if events is None:
                continue
            for event_idx, metrics in events.items():
                if metrics is None:
                    continue
                try:
                    event_data.append(
                        {
                            **metrics,
                            "fly_idx": int(key.split("_")[1]),
                            "ball_idx": int(key.split("_")[3]),
                            "event_id": event_idx,
                            "event_type": metrics.get("event_type", "unknown"),
                        }
                    )
                except Exception as e:
                    print(f"[WARN] Error processing event metrics for fly {getattr(fly, 'directory', None)}: {e}")
                    continue

        # Convert the event data to a DataFrame
        if not event_data:
            print(f"[WARN] No event data for fly {getattr(fly, 'directory', None)}")
            return None
        event_df = pd.DataFrame.from_records(event_data)

        # Sort by start_time to ensure events are in temporal order
        if "start_time" in event_df.columns:
            event_df = event_df.sort_values("start_time").reset_index(drop=True)
        else:
            print(f"[WARN] 'start_time' not in event_df columns for fly {getattr(fly, 'directory', None)}")
            return None

        # Add metadata to the dataset
        event_df = self._add_metadata(event_df, fly)

        return event_df

    def _prepare_dataset_summary_metrics(
        self,
        fly,
        metrics=[],
    ):
        """
        Prepares a dataset with summary metrics for a given fly. The metrics are computed for all events, but only the ones specified in the 'metrics' argument are included in the returned DataFrame.

        Args:
            fly (Fly): A Fly object.
            metrics (list): A list of metrics to include in the dataset. The metrics require valid ball and fly tracking data. If the list is empty, all available metrics are included. Defaults to [].

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's summary metrics and associated metadata.
        """
        # Initialize an empty DataFrame
        dataset = pd.DataFrame()

        # If no specific metrics are provided, include all available metrics
        if not metrics:
            # Instead of just the first dict's keys, collect all keys from all event_summaries
            all_keys = set()
            for metric_dict in fly.event_summaries.values():
                all_keys.update(metric_dict.keys())
            metrics = list(all_keys)

        # For each pair of fly and ball, get the metrics from the Fly metrics
        for key, metric_dict in fly.event_summaries.items():

            for metric in metrics:
                if metric in metric_dict:
                    value = metric_dict[metric]
                    # Ensure the value is in the expected format
                    if isinstance(value, list) or isinstance(value, dict):
                        value = str(value)  # Convert lists and dicts to strings
                    dataset.at[key, metric] = value

        if fly.f1_metrics:
            dataset["direction_match"] = fly.f1_metrics["direction_match"]

            # Assign summaries condition based on F1 condition

            dataset["F1_condition"] = fly.metadata.F1_condition

            if dataset["F1_condition"].iloc[0] == "control":
                for key in fly.event_summaries.keys():
                    if key == "fly_0_ball_0":
                        dataset.at[key, "ball_condition"] = "test"
            else:
                for key in fly.event_summaries.keys():
                    if key == "fly_0_ball_0":
                        dataset.at[key, "ball_condition"] = "training"
                    elif key == "fly_0_ball_1":
                        dataset.at[key, "ball_condition"] = "test"

        # Add metadata to the dataset
        dataset = self._add_metadata(dataset, fly)

        # print(dataset.columns)

        return dataset

    def _prepare_dataset_F1_coordinates(self, fly, downsampling_factor=None):
        # Check if the fly ever exits the corridor
        if fly.tracking_data.exit_time is None:
            print(f"Fly {fly.metadata.name} never exits the corridor")
            return

        downsampling_factor = fly.config.downsampling_factor

        dataset = pd.DataFrame()

        dataset["time"] = fly.tracking_data.flytrack.objects[0].dataset["time"]
        dataset["frame"] = fly.tracking_data.flytrack.objects[0].dataset["frame"]

        dataset["adjusted_time"] = fly.f1_metrics["adjusted_time"]

        # Assign training_ball and test_ball distances separately

        if fly.metadata.F1_condition != "control":
            dataset["training_ball"] = fly.f1_metrics["training_ball_distances"]["euclidean_distance"]

        dataset["test_ball"] = fly.f1_metrics["test_ball_distances"]["euclidean_distance"]

        # Exclude the fly if test_ball_distances has moved > 10 px before adjusted time 0
        NegData = dataset[dataset["adjusted_time"] < 0]

        if NegData["test_ball"].max() > 10:
            print(f"Fly {fly.metadata.name} excluded due to premature ball movements")
            return

        if downsampling_factor:
            dataset = dataset.iloc[:: downsampling_factor * fly.experiment.fps]

        dataset = self._add_metadata(dataset, fly)

        return dataset

    def _prepare_dataset_F1_checkpoints(self, fly):
        if fly.tracking_data.exit_time is None:
            print(f"Fly {fly.metadata.name} never exits the corridor")
            return pd.DataFrame()  # Return an empty DataFrame

        # Create a list of dictionaries for each checkpoint
        data = [
            {
                "fly_exit_time": fly.tracking_data.exit_time,
                "distance": distance,
                "adjusted_time": adjusted_time,
            }
            for distance, adjusted_time in fly.f1_metrics["F1_checkpoints"].items()
        ]

        # Convert the list of dictionaries to a DataFrame
        dataset = pd.DataFrame(data)

        # Add metadata to the dataset
        dataset = self._add_metadata(dataset, fly)

        if fly.metadata.F1_condition == "control":
            dataset["success_direction"] = fly.event_summaries["fly_0_ball_0"]["success_direction"]

        else:
            dataset["success_direction"] = fly.event_summaries["fly_0_ball_1"]["success_direction"]

        return dataset

    def _prepare_dataset_skeleton_contacts(self, fly):
        """
        Prepares a dataset with the fly's contacts with the ball and associated ball displacement, + metadata
        """

        if not fly.skeleton_metrics:
            print(f"No skeleton metrics found for fly {fly.metadata.name}")
            return pd.DataFrame()

        # Check if there are contacts for this fly
        if not fly.skeleton_metrics.ball_displacements:
            print(f"No contacts found for fly {fly.metadata.name}")
            return pd.DataFrame()

        dataset = pd.DataFrame()

        # Get the ball displacements of the fly

        ball_displacements = fly.skeleton_metrics.ball_displacements

        # Use the list indices + 1 as the contact indices

        contact_indices = [i + 1 for i in range(len(ball_displacements))]

        # Create a DataFrame with the contact indices and the ball displacements

        dataset["contact_index"] = contact_indices
        dataset["ball_displacement"] = ball_displacements

        # Add metadata to the dataset

        dataset = self._add_metadata(dataset, fly)

        return dataset

    def _add_metadata(self, data, fly):
        """
        Adds the metadata to a dataset generated by a _prepare_dataset_... method.

        Args:
            data (pandas.DataFrame): A pandas DataFrame generated by a _prepare_dataset_... method.
            fly (Fly): A Fly object.

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's coordinates and associated metadata.
        """

        dataset = pd.DataFrame()

        try:
            dataset = data

            # Add a column with the fly name as categorical data
            dataset["fly"] = fly.metadata.name
            dataset["fly"] = dataset["fly"].astype("category")

            # Add a column with the path to the fly's folder
            dataset["flypath"] = fly.metadata.directory.as_posix()
            dataset["flypath"] = dataset["flypath"].astype("category")

            # Add a column with the experiment name as categorical data
            dataset["experiment"] = fly.experiment.directory.name
            dataset["experiment"] = dataset["experiment"].astype("category")

            # Handle missing values for 'Nickname' and 'Brain region'
            dataset["Nickname"] = fly.metadata.nickname if fly.metadata.nickname is not None else "Unknown"
            dataset["Brain region"] = fly.metadata.brain_region if fly.metadata.brain_region is not None else "Unknown"
            dataset["Simplified Nickname"] = (
                fly.metadata.simplified_nickname if fly.metadata.simplified_nickname is not None else "Unknown"
            )
            dataset["Split"] = fly.metadata.split if fly.metadata.split is not None else "Unknown"

            # Add the metadata for the fly's arena as columns
            for var, data in fly.metadata.arena_metadata.items():
                # Handle missing values in arena metadata
                data = data if data is not None else "Unknown"
                dataset[var] = data
                dataset[var] = dataset[var].astype("category")

                # If the variable name is not in the metadata list, add it
                if var not in self.metadata:
                    self.metadata.append(var)

        except Exception as e:
            print(f"Error occurred while adding metadata for fly {fly.metadata.name}: {str(e)}")
            print(f"Current dataset:\n{dataset}")

        return dataset

    def _prepare_dataset_standardized_contacts(self, fly):
        """Prepares standardized contact event windows for analysis"""
        if not hasattr(fly, "skeleton_metrics") or fly.skeleton_metrics is None:
            return pd.DataFrame()

        events_df = fly.skeleton_metrics.events_based_contacts

        # Add trial information for learning experiments
        if fly.config.experiment_type == "Learning" and hasattr(fly, "learning_metrics"):
            events_df = self._add_trial_information(events_df, fly)

        # Add metadata
        events_df = self._add_metadata(events_df, fly)

        return events_df

    def _add_trial_information(self, dataset, fly):
        """
        Add trial information (trial, trial_frame, trial_time) to the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to annotate.
            fly (Fly): A Fly object.

        Returns:
            pd.DataFrame: The dataset with trial information added.
        """
        if not hasattr(fly, "learning_metrics") or fly.learning_metrics is None:
            # If no learning metrics are available, return the dataset as is
            return dataset

        trials_data = fly.learning_metrics.trials_data

        # Add trial column with default value of 0
        dataset["trial"] = 0

        # Update with actual trial values - more efficient with vectorized operations
        common_indices = dataset.index.intersection(trials_data.index)
        if not common_indices.empty:
            dataset.loc[common_indices, "trial"] = trials_data.loc[common_indices, "trial"].values

        # Calculate trial_frame as the frame index relative to the start of each trial
        dataset["trial_frame"] = dataset.groupby("trial").cumcount()

        # Compute trial_time from the trial_frame and experiment FPS
        dataset["trial_time"] = dataset["trial_frame"] / fly.experiment.fps

        return dataset

    def _prepare_transformed_dataset(self, fly):
        """Prepares a dataset regrouping individual interactions metrics including standardized contacts, tracking data, displacements, duration, etc."""

        # Extract the events-based contacts data
        data = fly.skeleton_metrics.events_based_contacts

        # Identify keypoint and metadata columns
        keypoint_columns = data.filter(regex="^(x|y)_").columns
        metadata_columns = data.columns.difference(keypoint_columns)

        # Group data by event_id, and optionally event_type
        groups = (
            list(data.groupby(["event_id", "event_type"]))
            if "event_type" in data.columns
            else list(data.groupby(["event_id"]))
        )

        transformed_data = []

        # Fetch event-level metrics once to avoid redundant calculations
        event_metrics_df = self._prepare_dataset_event_metrics(fly)

        for group_key, group in groups:
            # Compute duration and metadata for the group
            duration = group["frame"].max() - group["frame"].min() + 1
            metadata = group[metadata_columns].iloc[0].to_dict()

            # Initialize the row with basic metadata and duration
            row = {
                "fly": fly.metadata.name,
                "event_id": group_key[0],
                "event_type": group_key[1] if len(group_key) > 1 else None,
                "start": group["time"].iloc[0],
                "end": group["time"].iloc[-1],
                "start_frame": group["frame"].iloc[0],
                "end_frame": group["frame"].iloc[-1],
                "duration": duration,
            }

            # Add event-level metrics (avoiding duplicates)
            if "event_id" in event_metrics_df.columns:
                event_metrics = event_metrics_df[event_metrics_df["event_id"] == group_key[0]].iloc[0].to_dict()
                row.update(event_metrics)

            # Add metadata columns
            for col in metadata_columns:
                row[col] = metadata[col] if col in metadata else None

            # Append the row to the transformed data
            transformed_data.append(row)

        # Convert the transformed data into a DataFrame
        transformed_df = pd.DataFrame(transformed_data)

        # Add trial information if applicable
        if fly.config.experiment_type == "Learning" and hasattr(fly, "learning_metrics"):
            transformed_df = self._add_trial_information(transformed_df, fly)

        return transformed_df

    def _prepare_framewise_dataset(self, fly):
        """
        Prepares a dataset regrouping individual interactions metrics including standardized contacts,
        tracking data, displacements, duration, etc., and integrates event metrics.
        """

        # Extract the events-based contacts data
        data = fly.skeleton_metrics.events_based_contacts

        # Filter keypoint columns once
        keypoint_columns = data.filter(regex="^(x|y)_").columns
        if fly.config.fly_only:
            keypoint_columns = [col for col in keypoint_columns if "centre" not in col]

        # Group data by event_id (and optionally event_type)
        groupby_columns = ["event_id", "event_type"] if "event_type" in data.columns else ["event_id"]
        groups = data.groupby(groupby_columns)

        # Fetch the event metrics dataset
        event_metrics_df = self._prepare_dataset_event_metrics(fly)

        # print(f"Event metrics DataFrame: {event_metrics_df.head}")

        framewise_data = []

        for group_key, group in groups:
            event_id = group_key[0]
            event_type = group_key[1] if len(group_key) > 1 else None

            event_df = data[(data["event_id"] == event_id) & (data["event_type"] == event_type)]

            # Initialize the row with basic metadata
            row = {
                "fly": fly.metadata.name,
                "event_id": event_id,
                "event_type": event_type,
            }

            # Pre-filtered group for fly_only
            if fly.config.fly_only:
                group = group.drop(columns=[col for col in group.columns if "centre" in col])

            # Add transposed keypoints
            row.update(self._transpose_keypoints(fly, event_df, event_id=event_id, event_type=event_type))

            # Add frame-based features (velocity, angles, etc.)
            row.update(self._calculate_frame_features(fly, event_df, event_id=event_id, event_type=event_type))

            # Add statistical measures
            row.update(
                self._calculate_statistical_measures(
                    fly, event_df, keypoint_columns, event_id=event_id, event_type=event_type
                )
            )

            # Add Fourier features
            row.update(
                self._calculate_fourier_features(
                    fly, event_df, keypoint_columns, event_id=event_id, event_type=event_type
                )
            )

            # Add event metrics if available
            if not event_metrics_df.empty:
                event_metrics = event_metrics_df[
                    (event_metrics_df["event_id"] == event_id) & (event_metrics_df["event_type"] == event_type)
                ]
                if not event_metrics.empty:
                    row.update(event_metrics.iloc[0].to_dict())

            # Append the row to the framewise data
            framewise_data.append(row)

        # Convert the transformed data into a DataFrame
        framewise_df = pd.DataFrame(framewise_data)

        framewise_df = framewise_df.fillna(fly.config.hidden_value)

        # Add trial information if applicable
        if fly.config.experiment_type == "Learning" and hasattr(fly, "learning_metrics"):
            framewise_df = self._add_trial_information(framewise_df, fly)

        # Add metadata to the transformed dataset
        framewise_df = self._add_metadata(framewise_df, fly)

        return framewise_df

    def _transpose_keypoints(self, fly, data, event_id, event_type):
        """Transpose tracking keypoints to end up with one row per event."""
        try:
            # Define regex patterns for keypoints
            keypoint_patterns = r"^(x|y)_.*_(fly|centre_preprocessed)$"
            keypoint_columns = data.filter(regex=keypoint_patterns).columns

            # Add frame count for pivoting
            data = data.assign(frame_count=data.groupby(["event_id", "event_type", "adjusted_frame"]).cumcount())

            # Pivot all keypoints at once
            pivoted = data.pivot(
                index=["event_id", "event_type"], columns=["adjusted_frame", "frame_count"], values=keypoint_columns
            )

            # Flatten the multi-index columns and rename them
            pivoted.columns = [
                f"{col[0].split('_', 1)[1]}_frame{col[1]}_{col[0].split('_')[0]}" for col in pivoted.columns
            ]

            # Convert to dictionary
            transposed = pivoted.loc[(event_id, event_type)].to_dict() if not pivoted.empty else {}

            return transposed
        except Exception as e:
            print(f"Error in _transpose_keypoints for event_id {event_id}, event_type {event_type}: {e}")
            return {}

    def _calculate_frame_features(self, fly, data, event_id, event_type):
        """Calculate per-frame velocity and angles using original coordinates."""
        features = {}

        try:
            # Filter data for the specific event_id and event_type
            event_df = data[(data["event_id"] == event_id) & (data["event_type"] == event_type)]
            event_df = event_df.sort_values("adjusted_frame").reset_index(drop=True)

            # Define keypoints
            base_keypoints = {
                "Head",
                "Thorax",
                "Abdomen",
                "Rfront",
                "Lfront",
                "Rmid",
                "Lmid",
                "Rhind",
                "Lhind",
                "Rwing",
                "Lwing",
            }
            if not fly.config.fly_only:
                base_keypoints.add("centre")

            # Vectorized calculations for all keypoints
            for kp in base_keypoints:
                x_col, y_col = f"x_{kp}", f"y_{kp}"
                if x_col not in event_df.columns or y_col not in event_df.columns:
                    continue

                x_vals, y_vals = event_df[x_col].values, event_df[y_col].values
                dx, dy = np.diff(x_vals, prepend=x_vals[0]), np.diff(y_vals, prepend=y_vals[0])

                velocities = np.hypot(dx, dy)
                angles = np.degrees(np.arctan2(dy, dx)) % 360
                angular_velocity = np.diff(angles, prepend=angles[0])

                # Add features for each frame
                frames = event_df["adjusted_frame"].values
                features.update({f"{kp}_frame{frame}_velocity": velocities[i] for i, frame in enumerate(frames)})
                features.update({f"{kp}_frame{frame}_angle": angles[i] for i, frame in enumerate(frames)})
                features.update(
                    {f"{kp}_frame{frame}_angular_velocity": angular_velocity[i] for i, frame in enumerate(frames)}
                )

        except Exception as e:
            print(f"Error in _calculate_frame_features for event_id {event_id}, event_type {event_type}: {e}")

        return features

    def _calculate_statistical_measures(self, fly, data, keypoint_columns, event_id, event_type, fly_relative=True):
        """Calculate statistical measures for keypoints."""
        results = {}

        try:
            # Filter data for the specific event_id and event_type
            event_df = data[(data["event_id"] == event_id) & (data["event_type"] == event_type)]

            # Filter keypoint columns based on fly_relative
            if fly_relative:
                keypoint_columns = [col for col in keypoint_columns if "_fly" in col]

            if not keypoint_columns:
                print(
                    f"No keypoint columns found for statistical measures in event_id {event_id}, event_type {event_type}."
                )
                return results

            # Vectorized statistical calculations
            for col in keypoint_columns:
                col_values = event_df[col].values
                results[f"{col}_mean"] = np.nanmean(col_values)
                results[f"{col}_std"] = np.nanstd(col_values)
                results[f"{col}_skew"] = pd.Series(col_values).skew()
                results[f"{col}_kurt"] = pd.Series(col_values).kurtosis()

        except Exception as e:
            print(f"Error in _calculate_statistical_measures for event_id {event_id}, event_type {event_type}: {e}")

        return results

    def _calculate_fourier_features(self, fly, data, keypoint_columns, event_id, event_type, fly_relative=True):
        """Calculate Fourier features for keypoints."""
        fft_results = {}

        try:
            # Filter data for the specific event_id and event_type
            event_df = data[(data["event_id"] == event_id) & (data["event_type"] == event_type)]

            # Filter keypoint columns based on fly_relative
            if fly_relative:
                keypoint_columns = [col for col in keypoint_columns if "_fly" in col]

            if not keypoint_columns:
                print(
                    f"No keypoint columns found for Fourier features in event_id {event_id}, event_type {event_type}."
                )
                return fft_results

            # Vectorized Fourier calculations
            for col in keypoint_columns:
                col_values = event_df[col].values
                if len(col_values) < 2:  # Ensure sufficient data for FFT
                    continue

                fft_vals = fft(col_values)
                fft_vals = np.array(fft_vals)
                dominant_freq = np.abs(fft_vals[1 : len(fft_vals) // 2]).argmax() + 1
                fft_results[f"{col}_dom_freq"] = dominant_freq
                fft_results[f"{col}_dom_freq_magnitude"] = np.abs(fft_vals[dominant_freq])

        except Exception as e:
            print(f"Error in _calculate_fourier_features for event_id {event_id}, event_type {event_type}: {e}")

        return fft_results

    def generate_clip(self, fly, event, outpath):
        """
        Make a video clip of a fly's event.
        """

        # Get the fly object
        flies = self.find_flies("metadata.name", fly)
        if not flies:
            raise ValueError(f"Fly with name {fly} not found.")
        fly = flies[0]

        # Get the event start and end frames
        event_start = fly.skeleton_metrics.contacts[event][0]
        event_end = fly.skeleton_metrics.contacts[event][1]

        # Get the video file
        video_file = fly.metadata.video

        # Open the video file
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        # Get the frame rate of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Set the start and end frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, event_start)
        start_frame = event_start
        end_frame = event_end

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

        # Get the video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create the output file
        out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

        try:
            # Go to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read the video frame by frame and write to the output file
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                        break
                else:
                    break
        finally:
            # Release the video capture and writer objects
            cap.release()
            out.release()

        return outpath

    def make_events_grid(self, events_dict, clip_path):
        """
        This function will generate a grid video of all the events selected in events_dict, identified by the fly name and the event index.

        Args:
            events_dict (list): List of dictionaries with keys "name" and "event_index".
            clip_path (str): The path where the grid video will be saved.
        """

        clips = []

        for event in events_dict:
            print(event)

            fly_name = event["name"]
            event_index = event["event_index"]

            try:
                # Generate the clip for the event using the new generate_clip method
                event_clip_path = self.generate_clip(
                    fly_name,
                    event_index,
                    outpath=f"{os.path.dirname(clip_path)}/{fly_name}_event_{event_index}.mp4",
                )
                clips.append(event_clip_path)
            except Exception as e:
                print(f"Error generating clip for fly {fly_name}, event {event_index}: {e}")
                continue

        if clips:
            # Concatenate the clips into a grid video
            self.concatenate_clips(clips, clip_path)

            # Remove the individual clip files
            for event_clip_path in clips:
                os.remove(event_clip_path)

            print(f"Finished processing events grid! Saved to {clip_path}")
        else:
            print("No clips were generated.")

    def concatenate_clips(self, clips, output_path):
        """
        Concatenate the clips into a grid video.

        Args:
            clips (list): List of paths to the video clips.
            output_path (str): The path where the grid video will be saved.
        """
        video_clips = [VideoFileClip(clip) for clip in clips]

        # Determine the number of rows and columns for the grid
        num_clips = len(video_clips)
        grid_size = int(np.ceil(np.sqrt(num_clips)))

        # Create a grid of clips
        clip_grid = []
        for i in range(0, num_clips, grid_size):
            clip_row = video_clips[i : i + grid_size]
            # Pad the row with empty clips if necessary
            while len(clip_row) < grid_size:
                empty_clip = ColorClip(
                    size=(video_clips[0].w, video_clips[0].h),
                    color=(0, 0, 0),
                    duration=video_clips[0].duration,
                )
                clip_row.append(empty_clip.set_duration(video_clips[0].duration))
            clip_grid.append(clip_row)

        # Pad the grid with empty rows if necessary
        while len(clip_grid) < grid_size:
            empty_row = [
                ColorClip(
                    size=(video_clips[0].w, video_clips[0].h),
                    color=(0, 0, 0),
                    duration=video_clips[0].duration,
                )
                for _ in range(grid_size)
            ]
            clip_grid.append(empty_row)

        # Stack the clips into a grid
        final_clip = clips_array(clip_grid)

        # Write the final video to the output path
        final_clip.write_videofile(output_path, fps=29)


def detect_boundaries(Fly, threshold1=30, threshold2=100):
    """Detects the start and end of the corridor in the video. This is later used to compute the relative distance of the fly from the start of the corridor.

    Args:
        threshold1 (int, optional): the first threshold for the hysteresis procedure in Canny edge detection. Defaults to 30.
        threshold2 (int, optional): the second threshold for the hysteresis procedure in Canny edge detection. Defaults to 100.

    Returns:
        frame (np.array): the last frame of the video.
        start (int): the start of the corridor.
        end (int): the end of the corridor.
    """

    video_file = Fly.metadata.video

    if not video_file.exists():
        print(f"Error: Video file {video_file} does not exist")
        return None, None, None

    # open the video
    cap = cv2.VideoCapture(str(video_file))

    # get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # set the current position to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    # read the last frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame from video {video_file}")
        return None, None, None
    elif frame is None:
        print(f"Error: Frame is None for video {video_file}")
        return None, None, None

    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Use the Canny edge detection method
    edges = cv2.Canny(frame, threshold1=threshold1, threshold2=threshold2)

    # Get the top and bottom edges of the corridor in y-direction
    top_edge = np.min(np.where(edges > 0)[0])
    bottom_edge = np.max(np.where(edges > 0)[0])

    start = bottom_edge - 110
    end = top_edge + 110

    # Save a .npy file with the start and end coordinates in the video folder
    np.save(video_file.parent / "coordinates.npy", [start, end])

    return frame, start, end


def generate_grid(Experiment, preview=False, overwrite=False):

    # Check if the grid image already exists
    if (Experiment.directory / "grid.png").exists() and not overwrite:
        print(f"Grid image already exists for {Experiment.directory.name}")
        return
    else:
        print(f"Generating grid image for {Experiment.directory.name}")

        frames = []
        starts = []
        paths = []

        for fly in Experiment.flies:
            frame, start, end = detect_boundaries(fly)
            if frame is not None and start is not None:
                frames.append(frame)
                starts.append(start)
                paths.append(fly.video)

        # Set the number of rows and columns for the grid

        nrows = 9
        ncols = 6

        # Create a figure with subplots
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))

        # Loop over the frames, minimum row indices, and video paths
        for i, (frame, start, flypath) in enumerate(zip(frames, starts, paths)):
            # Get the row and column index for this subplot
            row = i // ncols
            col = i % ncols

            # Plot the frame on this subplot
            try:
                axs[row, col].imshow(frame, cmap="gray", vmin=0, vmax=255)
            except Exception as e:
                print(f"Error: Could not plot frame {i} for video {flypath}. Exception: {e}")
                # go to the next folder
                continue

            # Plot the horizontal lines on this subplot
            axs[row, col].axhline(start, color="red")
            axs[row, col].axhline(start - 290, color="blue")

        # Remove the axis of each subplot and draw them closer together
        for ax in axs.flat:
            ax.axis("off")
        plt.subplots_adjust(wspace=0, hspace=0)

        # Save the grid image in the main folder
        plt.savefig(Experiment.directory / "grid.png")

        if preview:
            plt.show()
