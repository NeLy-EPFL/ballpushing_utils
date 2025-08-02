import numpy as np
import pandas as pd
from typing import Any
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from utils_behavior import Processing


class BallPushingMetrics:
    def __init__(self, tracking_data, compute_metrics_on_init=True):

        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        # Iterate over the indices of the flytrack objects list
        # self.chamber_exit_times = {
        #     fly_idx: self.get_chamber_exit_time(fly_idx, 0)
        #     for fly_idx in range(len(self.tracking_data.flytrack.objects))
        # }

        if self.fly.config.debugging:
            for fly_idx, exit_time in self.tracking_data.chamber_exit_times.items():
                print(f"Fly {fly_idx} chamber exit time: {exit_time}")

        self.metrics = {}
        if compute_metrics_on_init:
            self.compute_metrics()

    def compute_metrics(self):
        """
        Compute and store various metrics for each pair of fly and ball.
        """

        def safe_call(method, *args, default: Any = np.nan, **kwargs) -> Any:
            """
            Helper function to safely call a method and handle exceptions.

            Parameters
            ----------
            method : callable
                The method to call.
            *args : tuple
                Positional arguments to pass to the method.
            default : Any, optional
                Default value to return in case of an exception (default is np.nan).
            **kwargs : dict
                Keyword arguments to pass to the method.

            Returns
            -------
            Any
                The result of the method call, or the default value if an exception occurs.
            """
            try:
                return method(*args, **kwargs)
            except Exception as e:
                if self.fly.config.debugging:
                    print(f"Error in {method.__name__}: {e}")
                return default

        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}"

                # Define metrics and their corresponding methods
                metrics = {
                    "nb_events": lambda: safe_call(self.get_adjusted_nb_events, fly_idx, ball_idx, signif=False),
                    "max_event": lambda: safe_call(self.get_max_event, fly_idx, ball_idx),
                    "max_distance": lambda: safe_call(self.get_max_distance, fly_idx, ball_idx),
                    "significant_events": lambda: safe_call(self.get_significant_events, fly_idx, ball_idx, default=[]),
                    "nb_significant_events": lambda: (
                        safe_call(self.get_adjusted_nb_events, fly_idx, ball_idx, signif=True)
                        if self.fly.config.experiment_type == "F1"
                        else len(safe_call(self.get_significant_events, fly_idx, ball_idx, default=[]))
                    ),
                    "first_significant_event": lambda: safe_call(self.get_first_significant_event, fly_idx, ball_idx),
                    "first_major_event": lambda: safe_call(self.get_major_event, fly_idx, ball_idx),
                    "events_direction": lambda: safe_call(
                        self.find_events_direction, fly_idx, ball_idx, default=([], [])
                    ),
                    "final_event": lambda: safe_call(self.get_final_event, fly_idx, ball_idx),
                    "success_direction": lambda: safe_call(self.get_success_direction, fly_idx, ball_idx),
                    "cumulated_breaks_duration": lambda: safe_call(
                        self.get_cumulated_breaks_duration, fly_idx, ball_idx
                    ),
                    "chamber_time": lambda: safe_call(self.get_chamber_time, fly_idx),
                    "chamber_ratio": lambda: safe_call(self.chamber_ratio, fly_idx),
                    "distance_moved": lambda: safe_call(self.get_distance_moved, fly_idx, ball_idx),
                    "distance_ratio": lambda: safe_call(self.get_distance_ratio, fly_idx, ball_idx),
                    "insight_effect": lambda: (
                        safe_call(self.get_insight_effect, fly_idx, ball_idx)
                        if safe_call(self.get_major_event, fly_idx, ball_idx)
                        else {
                            "raw_effect": np.nan,
                            "log_effect": np.nan,
                            "classification": "none",
                            "first_event": False,
                            "post_aha_count": 0,
                        }
                    ),
                }

                # Compute metrics
                nb_events = metrics["nb_events"]()
                max_event = metrics["max_event"]()
                # Handle final event
                final_event = metrics["final_event"]()
                if final_event is None:
                    final_event_idx = -1
                    final_event_time = self.tracking_data.duration
                else:
                    final_event_idx, final_event_time, _ = final_event

                significant_events = metrics["significant_events"]()

                # Filter events and significant events up to and including the final event
                if final_event_idx >= 0:
                    filtered_events = events[: final_event_idx + 1]
                    filtered_significant_events = [e for e in significant_events if e[1] <= final_event_idx]
                else:
                    filtered_events = events  # If no final event, keep all
                    filtered_significant_events = significant_events

                # Filtered event directions for pulling/pushing ratio
                filtered_events_direction = self.find_events_direction(fly_idx, ball_idx)
                filtered_pushed = [
                    e for e in filtered_events_direction[0] if e in [ev[0] for ev in filtered_significant_events]
                ]
                filtered_pulled = [
                    e for e in filtered_events_direction[1] if e in [ev[0] for ev in filtered_significant_events]
                ]

                major_event = metrics["first_major_event"]()
                events_direction = metrics["events_direction"]()
                insight_effect = metrics["insight_effect"]()
                pause_metrics = safe_call(
                    self.compute_pause_metrics, fly_idx, default={"number_of_pauses": 0, "total_pause_duration": 0.0}
                )
                interaction_persistence = safe_call(self.compute_interaction_persistence, fly_idx, ball_idx)
                learning_slope = safe_call(self.compute_learning_slope, fly_idx, ball_idx)
                logistic_features = safe_call(self.compute_logistic_features, fly_idx, ball_idx)
                event_influence = safe_call(self.compute_event_influence, fly_idx, ball_idx)
                normalized_velocity = safe_call(self.compute_normalized_velocity, fly_idx, ball_idx)
                velocity_during_interactions = safe_call(self.compute_velocity_during_interactions, fly_idx, ball_idx)
                velocity_trend = safe_call(self.compute_velocity_trend, fly_idx)

                # New metrics
                has_finished = safe_call(self.get_has_finished, fly_idx, ball_idx, default=0)
                persistence_at_end = safe_call(self.compute_persistence_at_end, fly_idx, default=np.nan)
                fly_distance_moved = safe_call(self.compute_fly_distance_moved, fly_idx, default=np.nan)
                time_chamber_beginning = safe_call(self.get_time_chamber_beginning, fly_idx, default=np.nan)
                median_freeze_duration = safe_call(self.compute_median_freeze_duration, fly_idx, default=np.nan)

                binned_slope = safe_call(self.compute_binned_slope, fly_idx, ball_idx)
                interaction_rate_by_bin = safe_call(self.compute_interaction_rate_by_bin, fly_idx, ball_idx)

                binned_slope_dict = {f"binned_slope_{i}": val for i, val in enumerate(binned_slope or [])}
                interaction_rate_by_bin_dict = {
                    f"interaction_rate_bin_{i}": val for i, val in enumerate(interaction_rate_by_bin or [])
                }
                overall_slope = safe_call(self.compute_overall_slope, fly_idx, ball_idx)
                overall_interaction_rate = safe_call(self.compute_overall_interaction_rate, fly_idx, ball_idx)

                auc = safe_call(self.compute_auc, fly_idx, ball_idx)
                binned_auc = safe_call(self.compute_binned_auc, fly_idx, ball_idx)
                binned_auc_dict = {f"binned_auc_{i}": val for i, val in enumerate(binned_auc or [])}

                # Store metrics in the dictionary
                self.metrics[key] = {
                    "nb_events": nb_events,
                    "max_event": max_event[0],
                    "max_event_time": max_event[1],
                    "max_distance": metrics["max_distance"](),
                    "final_event": final_event_idx,
                    "final_event_time": final_event_time,
                    "nb_significant_events": len(filtered_significant_events),
                    "significant_ratio": (
                        len(filtered_significant_events) / len(filtered_events) if len(filtered_events) > 0 else np.nan
                    ),
                    "first_significant_event": metrics["first_significant_event"]()[0],
                    "first_significant_event_time": metrics["first_significant_event"]()[1],
                    "first_major_event": major_event[0],
                    "first_major_event_time": major_event[1],
                    "major_event_first": insight_effect["first_event"],
                    # "insight_effect": safe_call(self.get_insight_effect, fly_idx, ball_idx, subset=filtered_events)[
                    #     "raw_effect"
                    # ],
                    # "insight_effect_log": safe_call(self.get_insight_effect, fly_idx, ball_idx, subset=filtered_events)[
                    #     "log_effect"
                    # ],
                    "cumulated_breaks_duration": safe_call(
                        self.get_cumulated_breaks_duration, fly_idx, ball_idx, subset=filtered_events
                    ),
                    "chamber_time": safe_call(self.get_chamber_time, fly_idx, end_time=final_event_time),
                    "chamber_ratio": safe_call(self.chamber_ratio, fly_idx, end_time=final_event_time),
                    "pushed": len(filtered_pushed),
                    "pulled": len(filtered_pulled),
                    "pulling_ratio": (
                        len(filtered_pulled) / (len(filtered_pushed) + len(filtered_pulled))
                        if (len(filtered_pushed) + len(filtered_pulled)) > 0
                        else -1
                    ),
                    "success_direction": metrics["success_direction"](),
                    "interaction_proportion": (
                        sum([event[2] for event in filtered_events])
                        / (final_event_time if final_event_time is not None else self.tracking_data.duration)
                        if (final_event_time if final_event_time is not None else self.tracking_data.duration) > 0
                        else np.nan
                    ),
                    "interaction_persistence": safe_call(
                        self.compute_interaction_persistence, fly_idx, ball_idx, subset=filtered_events
                    ),
                    "distance_moved": safe_call(self.get_distance_moved, fly_idx, ball_idx, subset=filtered_events),
                    "distance_ratio": safe_call(self.get_distance_ratio, fly_idx, ball_idx, subset=filtered_events),
                    "exit_time": self.tracking_data.exit_time,
                    "chamber_exit_time": self.tracking_data.chamber_exit_times[fly_idx],
                    "number_of_pauses": safe_call(self.compute_pause_metrics, fly_idx, subset=filtered_events)[
                        "number_of_pauses"
                    ],
                    "total_pause_duration": safe_call(self.compute_pause_metrics, fly_idx, subset=filtered_events)[
                        "total_pause_duration"
                    ],
                    # Learning and logistic slopes: use full data
                    "learning_slope": learning_slope["slope"],
                    "learning_slope_r2": learning_slope["r2"],
                    "logistic_L": logistic_features["L"],
                    "logistic_k": logistic_features["k"],
                    "logistic_t0": logistic_features["t0"],
                    "logistic_r2": logistic_features["r2"],
                    # "avg_displacement_after_success": safe_call(
                    #     self.compute_event_influence, fly_idx, ball_idx, subset=filtered_events
                    # )["avg_displacement_after_success"],
                    # "avg_displacement_after_failure": safe_call(
                    #     self.compute_event_influence, fly_idx, ball_idx, subset=filtered_events
                    # )["avg_displacement_after_failure"],
                    # "influence_ratio": safe_call(
                    #     self.compute_event_influence, fly_idx, ball_idx, subset=filtered_events
                    # )["influence_ratio"],
                    "normalized_velocity": normalized_velocity,
                    "velocity_during_interactions": velocity_during_interactions,
                    "velocity_trend": velocity_trend,
                    "overall_slope": overall_slope,
                    "overall_interaction_rate": overall_interaction_rate,
                    **binned_slope_dict,
                    **interaction_rate_by_bin_dict,
                    "auc": auc,
                    **binned_auc_dict,
                    # New metrics
                    "has_finished": has_finished,
                    "persistence_at_end": persistence_at_end,
                    "fly_distance_moved": fly_distance_moved,
                    "time_chamber_beginning": time_chamber_beginning,
                    "median_freeze_duration": median_freeze_duration,
                }

    def get_adjusted_nb_events(self, fly_idx, ball_idx, signif=False):
        """
        Calculate the adjusted number of events for a given fly and ball. adjustment is based on the duration of the experiment.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        signif : bool, optional
            Whether to use significant events only (default is False).

        Returns
        -------
        float
            Adjusted number of events.
        """
        if signif:
            events = self.get_significant_events(fly_idx, ball_idx)
        else:
            events = self.tracking_data.interaction_events[fly_idx][ball_idx]

        if self.fly.config.debugging:
            print(f"Events for fly {fly_idx} ball {ball_idx}, signif:{signif}: {len(events)}; {events}")

        adjusted_nb_events = 0  # Initialize to 0 in case there are no events

        if hasattr(self.fly.metadata, "F1_condition"):

            if self.fly.metadata.F1_condition == "control":
                if ball_idx == 0 and self.tracking_data.exit_time is not None:
                    adjusted_nb_events = (
                        len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / (self.tracking_data.duration - self.tracking_data.exit_time)
                        if self.tracking_data.duration - self.tracking_data.exit_time > 0
                        else 0
                    )
            else:
                if ball_idx == 1 and self.tracking_data.exit_time is not None:
                    adjusted_nb_events = (
                        len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / (self.tracking_data.duration - self.tracking_data.exit_time)
                        if self.tracking_data.duration - self.tracking_data.exit_time > 0
                        else 0
                    )
                elif ball_idx == 0:
                    adjusted_nb_events = (
                        len(events) * self.fly.config.adjusted_events_normalisation / self.tracking_data.exit_time
                        if (self.tracking_data.exit_time and self.tracking_data.exit_time > 0)
                        else len(events) * self.fly.config.adjusted_events_normalisation / self.tracking_data.duration
                    )

        else:
            adjusted_nb_events = (
                len(events)
                * self.fly.config.adjusted_events_normalisation
                / (self.tracking_data.duration - self.tracking_data.chamber_exit_times[fly_idx])
                if self.tracking_data.duration > 0
                else 0
            )

        return adjusted_nb_events

    def _calculate_median_coordinates(self, data, start_idx=None, end_idx=None, window=10, keypoint="centre"):
        """
        Calculate the median coordinates for the start and/or end of a given range.

        Parameters
        ----------
        data : pandas.DataFrame
            Data containing the x and y coordinate columns.
        start_idx : int, optional
            Start index of the range. If None, the start median is not calculated.
        end_idx : int, optional
            End index of the range. If None, the end median is not calculated.
        window : int, optional
            Number of frames to consider for the median calculation (default is 10).
        keypoint : str, optional
            Keypoint prefix for the x and y columns (e.g., "centre" for "x_centre" and "y_centre",
            or "thorax" for "x_thorax" and "y_thorax"). Default is "centre".

        Returns
        -------
        tuple
            Median coordinates for the start and/or end of the range. If `start_idx` or `end_idx`
            is None, the corresponding value in the tuple will be None.
        """
        x_col = f"x_{keypoint}"
        y_col = f"y_{keypoint}"

        start_x, start_y, end_x, end_y = None, None, None, None

        # Calculate the median position of the first `window` frames if start_idx is provided
        if start_idx is not None:
            start_x = data[x_col].iloc[start_idx : start_idx + window].median()
            start_y = data[y_col].iloc[start_idx : start_idx + window].median()

        # Calculate the median position of the last `window` frames if end_idx is provided
        if end_idx is not None:
            end_x = data[x_col].iloc[end_idx - window : end_idx].median()
            end_y = data[y_col].iloc[end_idx - window : end_idx].median()

        return start_x, start_y, end_x, end_y

    def find_event_by_distance(self, fly_idx, ball_idx, threshold, distance_type="max"):
        """
        Find the event at which the ball has been moved a given amount of pixels for a given fly and ball.
        Threshold is the distance threshold to check, whereas max is the maximum distance reached by the ball for this particular fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float
            Distance threshold.
        distance_type : str, optional
            Type of distance to check ("max" or "threshold", default is "max").

        Returns
        -------
        tuple
            Event and event index.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if distance_type == "max":
            max_distance = ball_data["euclidean_distance"].max() - threshold

            distance_check = (
                lambda event: ball_data.loc[event[0] : event[1], "euclidean_distance"].max() >= max_distance
            )
        elif distance_type == "threshold":
            distance_check = lambda event: ball_data.loc[event[0] : event[1], "euclidean_distance"].max() >= threshold
        else:
            raise ValueError("Invalid distance_type. Use 'max' or 'threshold'.")

        try:
            event, event_index = next(
                (event, i)
                for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
                if distance_check(event)
            )
        except StopIteration:
            event, event_index = None, None

        return event, event_index

    # TODO: Implement adjustment by computing time to exit chamber and subtract it from the time of the event (perhaps keep both?)
    def get_chamber_exit_time(self, fly_idx, ball_idx):
        """
        Compute the time at which the fly left the chamber for a given fly and ball.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.

        Returns:
            float or None: The exit time in seconds, or None if the fly never left the chamber.
        """
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate the distance from the fly start position for each frame
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"], fly_data["y_thorax"], self.tracking_data.start_x, self.tracking_data.start_y
        )

        # Debugging: Print distances
        if self.fly.config.debugging:
            print(f"Distances for fly {fly_idx}, ball {ball_idx}: {distances}")

        # Determine the frames where the fly is within a certain radius of the start position
        in_chamber = distances <= self.fly.config.chamber_radius

        # Debugging: Print in_chamber array
        if self.fly.config.debugging:
            print(f"In-chamber status for fly {fly_idx}, ball {ball_idx}: {in_chamber}")

        # Check if the fly ever leaves the chamber
        if not np.any(~in_chamber):  # If all values in `in_chamber` are True
            if self.fly.config.debugging:
                print(f"Fly {fly_idx} never left the chamber.")
            return None

        # Get the first frame where the fly is outside the chamber
        exit_frame = np.argmax(~in_chamber)  # Find the first `False` in `in_chamber`

        # Debugging: Print exit_frame
        if self.fly.config.debugging:
            print(f"Exit frame for fly {fly_idx}, ball {ball_idx}: {exit_frame}")

        exit_time = exit_frame / self.fly.experiment.fps

        return exit_time

    def get_max_event(self, fly_idx, ball_idx, threshold=None):
        """
        Get the event at which the ball was moved at its maximum distance for a given fly and ball.
        Maximum here doesn't mean the ball has reached the end of the corridor.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float, optional
            Distance threshold (default is None).

        Returns
        -------
        tuple
            Maximum event index and maximum event time.
        """
        if threshold is None:
            threshold = self.fly.config.max_event_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Find the event with the maximum distance
        max_event, max_event_idx = self.find_event_by_distance(fly_idx, ball_idx, threshold, distance_type="max")

        # Get the chamber exit time for the current fly
        chamber_exit_time = self.tracking_data.chamber_exit_times.get(fly_idx, None)

        # Calculate the time of the maximum event
        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            if max_event:
                max_event_time = max_event[0] / self.fly.experiment.fps
                if chamber_exit_time is not None:
                    max_event_time -= chamber_exit_time
            else:
                max_event_time = self.tracking_data.duration
        else:
            if max_event:
                max_event_time = (max_event[0] / self.fly.experiment.fps) - self.tracking_data.exit_time
            else:
                max_event_time = self.tracking_data.duration

        return max_event_idx, max_event_time

    def get_max_distance(self, fly_idx, ball_idx):
        """
        Get the maximum distance moved by the ball for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Maximum distance moved by the ball.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Calculate the median initial position
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(ball_data, start_idx=0, window=10)

        # Calculate the Euclidean distance from the initial position
        distances = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
        )

        # Return the maximum distance
        return distances.max()

    def get_final_event(self, fly_idx, ball_idx, threshold=None, init=False):
        """
        Get the final event for a given fly and ball based on a distance threshold.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float, optional
            Distance threshold for the final event (default is None).
        init : bool, optional
            Whether to use initial conditions (default is False).

        Returns
        -------
        tuple or None
            A tuple (final_event_idx, final_event_time, final_event_end) if a final event is found,
            or None if no final event is found.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Check the ball data time boundaries
        ball_data_start = ball_data["time"].iloc[0]
        ball_data_end = ball_data["time"].iloc[-1]

        if self.fly.config.debugging:
            print(f"Ball data time range: {ball_data_start} to {ball_data_end}")

        # # Apply time range filtering
        # time_range = self.tracking_data.fly.config.time_range
        # if time_range:
        #     start = int(time_range[0] * self.tracking_data.fly.experiment.fps) if time_range[0] is not None else 0
        #     end = int(time_range[1] * self.tracking_data.fly.experiment.fps) if time_range[1] is not None else None
        #     ball_data = ball_data.iloc[start:end]

        # Get the chamber exit time for the current fly
        chamber_exit_time = self.tracking_data.chamber_exit_times.get(fly_idx, None)

        # Determine the appropriate threshold
        if threshold is None:
            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                threshold = self.fly.config.final_event_threshold
            else:
                threshold = self.fly.config.final_event_F1_threshold
        if self.fly.config.debugging:
            print(f"Threshold for fly {fly_idx}, ball {ball_idx}: {threshold}")

        # Find the final event based on the threshold
        final_event, final_event_idx = self.find_event_by_distance(
            fly_idx, ball_idx, threshold, distance_type="threshold"
        )
        if self.fly.config.debugging:
            print(f"Final event for fly {fly_idx}, ball {ball_idx}: {final_event}")

        # If no final event is found, return None
        if not final_event:
            return None

        # Calculate the time and end time of the final event
        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            final_event_time = final_event[0] / self.fly.experiment.fps
            final_event_end = final_event[1] / self.fly.experiment.fps

            if chamber_exit_time is not None:
                final_event_time -= chamber_exit_time
                final_event_end -= chamber_exit_time
        else:
            final_event_time = (final_event[0] / self.fly.experiment.fps) - self.tracking_data.exit_time
            final_event_end = (final_event[1] / self.fly.experiment.fps) - self.tracking_data.exit_time

        return final_event_idx, final_event_time, final_event_end

    def get_significant_events(self, fly_idx, ball_idx, distance=5):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = [
            (event, i)
            for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        return significant_events

    def get_first_significant_event(self, fly_idx, ball_idx, distance=5):
        """
        Get the first significant event for a given fly and ball based on a distance threshold.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        distance : float, optional
            Distance threshold for significant events (default is 5).

        Returns
        -------
        tuple
            First significant event index and first significant event time.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Get the chamber exit time for the current fly
        chamber_exit_time = self.tracking_data.chamber_exit_times.get(fly_idx, None)

        # Get significant events
        significant_events = self.get_significant_events(fly_idx, ball_idx, distance=distance)

        if significant_events:
            first_significant_event = significant_events[0]
            first_significant_event_idx = first_significant_event[1]

            # Calculate the time of the first significant event
            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                first_significant_event_time = first_significant_event[0][0] / self.fly.experiment.fps
                if chamber_exit_time is not None:
                    first_significant_event_time -= chamber_exit_time
            else:
                first_significant_event_time = (
                    first_significant_event[0][0] / self.fly.experiment.fps
                ) - self.tracking_data.exit_time

            return first_significant_event_idx, first_significant_event_time
        else:
            return -1, self.tracking_data.duration

    def check_yball_variation(self, event, ball_data, threshold=None):

        if threshold is None:
            threshold = self.fly.config.significant_threshold

        yball_event = ball_data.loc[event[0] : event[1], "y_centre"]
        variation = yball_event.max() - yball_event.min()
        return variation > threshold

    def find_breaks(self, fly_idx, ball_idx):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        breaks = []
        if not self.tracking_data.interaction_events[fly_idx][ball_idx]:
            breaks.append((0, len(ball_data), len(ball_data)))
            return breaks

        if self.tracking_data.interaction_events[fly_idx][ball_idx][0][0] > 0:
            breaks.append(
                (
                    0,
                    self.tracking_data.interaction_events[fly_idx][ball_idx][0][0],
                    self.tracking_data.interaction_events[fly_idx][ball_idx][0][0],
                )
            )

        for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx][:-1]):
            start = event[1]
            end = self.tracking_data.interaction_events[fly_idx][ball_idx][i + 1][0]
            duration = end - start
            breaks.append((start, end, duration))

        if self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1] < len(ball_data):
            breaks.append(
                (
                    self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1],
                    len(ball_data),
                    len(ball_data) - self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1],
                )
            )

        return breaks

    def get_cumulated_breaks_duration(self, fly_idx, ball_idx, subset=None):
        breaks = self.find_breaks(fly_idx, ball_idx)
        if subset is not None:
            # Only include breaks that overlap with any event in subset
            valid_ranges = [(event[0], event[1]) for event in subset]
            filtered_breaks = []
            for br in breaks:
                br_start, br_end, _ = br
                if any((br_start < end and br_end > start) for start, end in valid_ranges):
                    filtered_breaks.append(br)
            breaks = filtered_breaks
        cumulated_breaks_duration = sum([break_[2] for break_ in breaks])
        return cumulated_breaks_duration

    def get_chamber_time(self, fly_idx, end_time=None):
        """
        Compute the time spent by the fly in the chamber. The chamber is defined as the area within a certain radius of the fly start position.

        Args:
            fly_idx (int): Index of the fly.

        Returns:
            float: Time spent in the chamber in seconds.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Get the fly start position using the median of the first 10 frames
        start_x, start_y, _, _ = self._calculate_median_coordinates(fly_data, start_idx=0, window=10, keypoint="thorax")

        # Calculate the distance from the fly start position for each frame
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"], fly_data["y_thorax"], start_x, start_y
        )

        # Determine the frames where the fly is within a certain radius of the start position
        in_chamber = distances <= self.fly.config.chamber_radius

        if end_time is not None:
            in_chamber = in_chamber[: int(end_time * self.fly.experiment.fps)]

        # Calculate the time spent in the chamber
        time_in_chamber = np.sum(in_chamber) / self.fly.experiment.fps

        return time_in_chamber

    # TODO: Maybe use the non smoothed data to compute this kind of thing as it looks like there's a slight lag in the smoothed data.

    def chamber_ratio(self, fly_idx, end_time=None):
        """Compute the ratio of time spent in the chamber to the total time of the experiment.
        Args:
            fly_idx (int): Index of the fly.
        """

        time_in_chamber = self.get_chamber_time(fly_idx, end_time=end_time)
        total_time = end_time if end_time is not None else self.tracking_data.duration

        # Calculate the ratio
        if total_time > 0:
            ratio = time_in_chamber / total_time
        else:
            ratio = np.nan

        return ratio

    def find_events_direction(self, fly_idx, ball_idx):
        """
        Categorize significant events as pushing or pulling based on the change in distance
        between the ball and the fly during each event.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.

        Returns:
            tuple: Two lists - pushing_events and pulling_events.
        """
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = self.get_significant_events(fly_idx, ball_idx)

        pushing_events = []
        pulling_events = []

        for event in significant_events:
            event = event[0]

            start_roi = event[0]
            end_roi = event[1]

            # Calculate the median positions for the start and end of the event
            start_ball_x, start_ball_y, end_ball_x, end_ball_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_roi, end_idx=end_roi, window=10, keypoint="centre"
            )

            start_fly_x, start_fly_y, _, _ = self._calculate_median_coordinates(
                fly_data, start_idx=start_roi, window=10, keypoint="thorax"
            )

            # Calculate the distances using the median positions
            start_distance = Processing.calculate_euclidian_distance(
                start_ball_x, start_ball_y, start_fly_x, start_fly_y
            )
            end_distance = Processing.calculate_euclidian_distance(end_ball_x, end_ball_y, start_fly_x, start_fly_y)

            # Categorize the event based on the change in distance
            if end_distance > start_distance:
                pushing_events.append(event)
            else:
                pulling_events.append(event)

        return pushing_events, pulling_events

    def get_distance_moved(self, fly_idx, ball_idx, subset=None):
        """
        Calculate the total distance moved by the ball for a given fly and ball,
        using the median position of the first and last 10 frames of each event
        to reduce the impact of tracking aberrations.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.
            subset (list, optional): List of events to consider. Defaults to all interaction events.

        Returns:
            float: Total distance moved by the ball.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if subset is None:
            subset = self.tracking_data.interaction_events[fly_idx][ball_idx]

        total_distance = 0

        for event in subset:
            start_idx, end_idx = event[0], event[1]

            # Ensure the event duration is long enough to calculate medians
            if end_idx - start_idx < 10:
                continue

            # Use the helper method to calculate median coordinates
            start_x, start_y, end_x, end_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
            )

            # Calculate the Euclidean distance between the start and end positions
            distance = Processing.calculate_euclidian_distance(start_x, start_y, end_x, end_y)

            total_distance += distance

        return total_distance

    def get_distance_ratio(self, fly_idx, ball_idx, subset=None):
        """
        Calculate the ratio between the maximum distance moved from the start position
        and the total distance moved by the ball. This ratio highlights discrepancies
        caused by behaviors like pulling and pushing the ball repeatedly.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.

        Returns:
            float: The distance ratio. A value closer to 1 indicates consistent movement
                in one direction, while a lower value indicates more back-and-forth movement.
        """
        # Get the maximum distance moved from the start position
        max_distance = self.get_max_distance(fly_idx, ball_idx)

        # Get the total distance moved
        total_distance = self.get_distance_moved(fly_idx, ball_idx, subset=subset)

        # Calculate the ratio
        if total_distance > 0:
            distance_ratio = total_distance / max_distance
        else:
            distance_ratio = np.nan  # Handle cases where total distance is zero

        return distance_ratio

    def get_major_event(self, fly_idx, ball_idx, distance=None):
        """
        Identify the aha moment for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        distance : float, optional
            Distance threshold for the aha moment (default is None).

        Returns
        -------
        tuple
            Aha moment index and aha moment time.
        """
        if distance is None:
            distance = self.fly.config.major_event_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        major_event = [
            (event, i)
            for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        if major_event:
            # Select the event right before the event at which the ball was moved more than the threshold
            major_event_instance, major_event_idx = major_event[0]
            if major_event_idx > 0:
                previous_event = self.tracking_data.interaction_events[fly_idx][ball_idx][major_event_idx - 1]
                major_event_instance = previous_event
                major_event_idx -= 1

            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                first_major_event_time = major_event_instance[0] / self.fly.experiment.fps
            else:
                first_major_event_time = (
                    major_event_instance[0] / self.fly.experiment.fps
                ) - self.tracking_data.exit_time

            return major_event_idx, first_major_event_time
        else:
            return -1, self.tracking_data.duration

    def get_insight_effect(self, fly_idx, ball_idx, epsilon=1e-6, strength_threshold=2, subset=None):
        """
        Calculate enhanced insight effect with performance analytics, optionally on a subset of events.
        """
        if subset is not None:
            significant_events = [event[0] for event in subset]
        else:
            significant_events = [event[0] for event in self.get_significant_events(fly_idx, ball_idx)]
        major_event_index, _ = self.get_major_event(fly_idx, ball_idx)

        if not significant_events:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        if major_event_index is None:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        before_aha = significant_events[: major_event_index + 1]
        after_aha = significant_events[major_event_index + 1 :]

        avg_before = self._calculate_avg_distance(fly_idx, ball_idx, before_aha)
        avg_after = self._calculate_avg_distance(fly_idx, ball_idx, after_aha)

        if major_event_index == 0:
            insight_effect = 1.0
        elif avg_before == 0:
            insight_effect = np.nan
        else:
            insight_effect = avg_after / avg_before

        log_effect = np.log(insight_effect + 1) if insight_effect > 0 else np.nan
        classification = "strong" if insight_effect > strength_threshold else "weak"

        return {
            "raw_effect": insight_effect,
            "log_effect": log_effect,
            "classification": classification,
            "first_event": major_event_index == 0,
            "post_aha_count": len(after_aha),
        }

    def _calculate_avg_distance(self, fly_idx, ball_idx, events):
        """Helper method to safely calculate average distances"""
        if not events:
            return np.nan

        try:
            distances = self.get_distance_moved(fly_idx, ball_idx, subset=events)
            return np.mean(distances) / len(events)
        except (ValueError, ZeroDivisionError):
            return np.nan

    def get_success_direction(self, fly_idx, ball_idx, threshold=None):
        """
        Determine the success direction (push, pull, or both) based on the ball's movement.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.
            threshold (float, optional): Distance threshold for success. Defaults to the configured threshold.

        Returns:
            str: "push", "pull", "both", or None if no success direction is determined.
        """
        if threshold is None:
            threshold = self.fly.config.success_direction_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Get the median initial position of the ball
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, window=10, keypoint="centre"
        )

        # Calculate the Euclidean distance from the initial position
        ball_data["euclidean_distance"] = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
        )

        # Filter the data for frames where the ball has moved beyond the threshold
        moved_data = ball_data[ball_data["euclidean_distance"] >= threshold]

        if moved_data.empty:
            return None

        # Determine the success direction based on the y-coordinate movement
        if abs(initial_x - self.tracking_data.start_x) < 100:
            pushed = any(moved_data["y_centre"] < initial_y)
            pulled = any(moved_data["y_centre"] > initial_y)
        else:
            pushed = any(moved_data["y_centre"] > initial_y)
            pulled = any(moved_data["y_centre"] < initial_y)

        if pushed and pulled:
            return "both"
        elif pushed:
            return "push"
        elif pulled:
            return "pull"
        else:
            return None

    def detect_pauses(self, fly_idx, threshold=None, window=None, minimum_duration=None):
        """
        Detect pauses in the fly's movement based on skeleton keypoints, excluding pauses
        that occur when the fly is in the chamber.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        threshold : float, optional
            Movement threshold in pixels to consider as a pause (default is 5).
        window : int, optional
            Number of frames to use for calculating movement (default is 5).
        minimum_duration : float, optional
            Minimum duration (in seconds) for a pause to be considered valid (default is 2).

        Returns
        -------
        list of tuple
            List of pauses, where each pause is represented as a tuple (start_time, end_time, duration).
        """
        if threshold is None:
            threshold = self.fly.config.pause_threshold
        if window is None:
            window = self.fly.config.pause_window
        if minimum_duration is None:
            minimum_duration = self.fly.config.pause_min_duration

        # Get the skeleton data for the fly
        skeleton_data = self.tracking_data.skeletontrack.objects[fly_idx].dataset

        # Extract all keypoints (assuming columns are named like "x_<keypoint>" and "y_<keypoint>")
        keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]

        # Initialize a boolean array to track movement
        is_static = np.ones(len(skeleton_data), dtype=bool)

        for keypoint in keypoints:
            # Calculate the velocity (magnitude of movement) for each keypoint over a rolling window
            x_velocity = skeleton_data[f"x_{keypoint}"].diff().rolling(window=window).mean().abs()
            y_velocity = skeleton_data[f"y_{keypoint}"].diff().rolling(window=window).mean().abs()

            # Check if the velocity is below the threshold
            keypoint_static = (x_velocity <= threshold) & (y_velocity <= threshold)

            # Combine with the overall static status
            is_static &= keypoint_static

        # Smooth the static status using a rolling window to avoid noise
        is_static_smoothed = pd.Series(is_static).rolling(window=window, min_periods=1).mean() == 1

        # Identify chunks of static frames
        pauses = []
        start_frame = None

        for i, static in enumerate(is_static_smoothed):
            if static and start_frame is None:
                start_frame = i
            elif not static and start_frame is not None:
                end_frame = i
                duration = (end_frame - start_frame) / self.fly.experiment.fps
                if duration >= minimum_duration:
                    # Check if the fly is in the chamber during the pause
                    fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
                    start_x, start_y, _, _ = self._calculate_median_coordinates(
                        fly_data, start_idx=0, window=10, keypoint="thorax"
                    )
                    distances = Processing.calculate_euclidian_distance(
                        fly_data["x_thorax"].iloc[start_frame:end_frame],
                        fly_data["y_thorax"].iloc[start_frame:end_frame],
                        start_x,
                        start_y,
                    )
                    in_chamber = np.all(distances <= self.fly.config.chamber_radius)

                    if not in_chamber:  # Exclude pauses in the chamber
                        pauses.append((start_frame, end_frame, duration))
                start_frame = None

        # Handle the case where a pause extends to the end of the recording
        if start_frame is not None:
            end_frame = len(is_static_smoothed)
            duration = (end_frame - start_frame) / self.fly.experiment.fps
            if duration >= minimum_duration:
                fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
                start_x, start_y, _, _ = self._calculate_median_coordinates(
                    fly_data, start_idx=0, window=10, keypoint="thorax"
                )
                distances = Processing.calculate_euclidian_distance(
                    fly_data["x_thorax"].iloc[start_frame:end_frame],
                    fly_data["y_thorax"].iloc[start_frame:end_frame],
                    start_x,
                    start_y,
                )
                in_chamber = np.all(distances <= self.fly.config.chamber_radius)

                if not in_chamber:  # Exclude pauses in the chamber
                    pauses.append((start_frame, end_frame, duration))

        # Convert frame indices to timestamps
        pauses_timestamps = [
            (start_frame / self.fly.experiment.fps, end_frame / self.fly.experiment.fps, duration)
            for start_frame, end_frame, duration in pauses
        ]

        if self.fly.config.debugging:
            print(f"Detected pauses for fly {fly_idx}: {pauses_timestamps}")

        return pauses_timestamps

    def compute_pause_metrics(self, fly_idx, subset=None, threshold=5, window=5, minimum_duration=2):
        """
        Compute the number of pauses and total duration of pauses for a given fly, optionally within a subset of frames.
        """
        pauses = self.detect_pauses(fly_idx, threshold=threshold, window=window, minimum_duration=minimum_duration)

        if subset is not None:
            # Flatten all intervals in subset to a set of valid time ranges (in seconds)
            valid_ranges = []
            for event in subset:
                start_time = event[0] / self.fly.experiment.fps
                end_time = event[1] / self.fly.experiment.fps
                valid_ranges.append((start_time, end_time))
            # Filter pauses to only those that overlap with any valid range
            filtered_pauses = []
            for pause in pauses:
                pause_start, pause_end, duration = pause
                if any((pause_start < end and pause_end > start) for start, end in valid_ranges):
                    filtered_pauses.append(pause)
            pauses = filtered_pauses

        if not pauses:
            return {
                "number_of_pauses": 0,
                "total_pause_duration": 0.0,
            }

        number_of_pauses = len(pauses)
        total_pause_duration = sum(pause[2] for pause in pauses)

        return {
            "number_of_pauses": number_of_pauses,
            "total_pause_duration": total_pause_duration,
        }

    def compute_interaction_persistence(self, fly_idx, ball_idx, subset=None):
        """
        Compute the interaction persistence for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            The average duration of interaction events in seconds, or np.nan if no events exist.
        """
        # Get the interaction events for the fly and ball
        events = subset if subset is not None else self.tracking_data.interaction_events[fly_idx][ball_idx]
        if not events:
            # Return NaN if there are no interaction events
            return np.nan

        # Calculate the duration of each event
        event_durations = [(event[1] - event[0]) / self.fly.experiment.fps for event in events]

        # Compute the average duration
        average_duration = np.mean(event_durations)

        return average_duration

    def compute_learning_slope(self, fly_idx, ball_idx):
        """
        Compute the learning slope and R² for a given fly and ball based on ball position over time.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        dict
            Dictionary containing:
            - "slope": float, the slope of the linear regression.
            - "r2": float, the R² of the linear regression.
        """
        # Get the ball position data
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Extract time (in seconds) and position (e.g., y_centre)
        time = np.arange(len(ball_data)) / self.fly.experiment.fps  # Time in seconds
        position = ball_data["y_centre"].values  # Replace with "x_centre" if needed

        # Check if there is enough data to fit a regression
        if len(time) < 2 or np.all(position == position[0]):  # No movement
            return {"slope": np.nan, "r2": np.nan}

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(time.reshape(-1, 1), position)

        # Calculate R²
        r2 = model.score(time.reshape(-1, 1), position)

        # Return the slope and R²
        return {"slope": model.coef_[0], "r2": r2}

    def compute_logistic_features(self, fly_idx, ball_idx):
        """
        Compute logistic features and R² for a given fly and ball based on ball position over time.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        dict
            Dictionary containing:
            - "L": float, maximum value (plateau).
            - "k": float, growth rate (steepness of the curve).
            - "t0": float, midpoint (time at which the curve reaches half of L).
            - "r2": float, R² of the logistic regression.
        """
        # Get the ball position data
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Extract time (in seconds) and position (e.g., y_centre)
        time = np.arange(len(ball_data)) / self.fly.experiment.fps  # Time in seconds
        position = ball_data["y_centre"].values  # Replace with "x_centre" if needed

        # Check if there is enough data to fit a logistic function
        if len(time) < 3 or np.all(position == position[0]):  # No movement
            return {"L": np.nan, "k": np.nan, "t0": np.nan, "r2": np.nan}

        # Initial guesses for logistic parameters
        initial_guess = [position.max(), 1, time.mean()]  # L, k, t0

        try:
            # Fit the logistic function to the data
            params, _ = curve_fit(Processing.logistic_function, time, position, p0=initial_guess, maxfev=10000)
            L, k, t0 = params

            # Calculate predicted values
            predicted = Processing.logistic_function(time, L, k, t0)

            # Calculate R²
            ss_res = np.sum((position - predicted) ** 2)  # Residual sum of squares
            ss_tot = np.sum((position - np.mean(position)) ** 2)  # Total sum of squares
            r2 = 1 - (ss_res / ss_tot)

        except RuntimeError:
            # If the fit fails, return NaN
            return {"L": np.nan, "k": np.nan, "t0": np.nan, "r2": np.nan}

        return {"L": L, "k": k, "t0": t0, "r2": r2}

    def compute_event_influence(self, fly_idx, ball_idx, success_threshold=None):
        """
        Compute the influence of event n-1 on event n for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        success_threshold : float, optional
            Threshold for significant ball displacement to classify an event as successful (default is 5).

        Returns
        -------
        dict
            Dictionary containing:
            - "avg_displacement_after_success": float, average displacement of events following successful events.
            - "avg_displacement_after_failure": float, average displacement of events following unsuccessful events.
            - "influence_ratio": float, ratio of the two averages (success/failure).
        """
        if success_threshold is None:
            success_threshold = self.fly.config.success_threshold

        # Get the interaction events for the fly and ball
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if len(events) < 2:
            # Not enough events to compute influence
            return {
                "avg_displacement_after_success": np.nan,
                "avg_displacement_after_failure": np.nan,
                "influence_ratio": np.nan,
            }

        # Helper function to calculate displacement for an event
        def calculate_displacement(event):
            start_idx, end_idx = event[0], event[1]
            start_x, start_y, end_x, end_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
            )
            return Processing.calculate_euclidian_distance(start_x, start_y, end_x, end_y)

        # Classify events as successful or unsuccessful
        event_success = []
        for event in events:
            displacement = calculate_displacement(event)
            event_success.append(displacement >= success_threshold)

        # Analyze the influence of event n-1 on event n
        displacements_after_success = []
        displacements_after_failure = []

        for i in range(1, len(events)):
            displacement_n = calculate_displacement(events[i])
            if event_success[i - 1]:  # If event n-1 was successful
                displacements_after_success.append(displacement_n)
            else:  # If event n-1 was unsuccessful
                displacements_after_failure.append(displacement_n)

        # Compute averages
        avg_displacement_after_success = np.mean(displacements_after_success) if displacements_after_success else 0
        avg_displacement_after_failure = np.mean(displacements_after_failure) if displacements_after_failure else 0

        # Compute influence ratio
        if avg_displacement_after_failure > 0:
            influence_ratio = avg_displacement_after_success / avg_displacement_after_failure
        else:
            influence_ratio = 0

        return {
            "avg_displacement_after_success": avg_displacement_after_success,
            "avg_displacement_after_failure": avg_displacement_after_failure,
            "influence_ratio": influence_ratio,
        }

    def compute_normalized_velocity(self, fly_idx, ball_idx):
        """
        Compute the fly's velocity normalized by the available space.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            The average velocity normalized by the available space.
        """
        # Get the fly and ball data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Calculate the available space as the distance between the ball and the fly's starting position
        initial_ball_x, initial_ball_y, _, _ = self._calculate_median_coordinates(ball_data, start_idx=0, window=10)
        ball_distances = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_ball_x, initial_ball_y
        )

        # Calculate the fly's velocity
        fly_velocity = (
            np.sqrt(fly_data["x_thorax"].diff() ** 2 + fly_data["y_thorax"].diff() ** 2) * self.fly.experiment.fps
        )  # Convert to velocity in pixels/second

        # Normalize velocity by the available space
        normalized_velocity = fly_velocity / (ball_distances + 1e-6)  # Add epsilon to avoid division by zero

        # Return the average normalized velocity
        return np.nanmean(normalized_velocity)

    def compute_velocity_during_interactions(self, fly_idx, ball_idx):
        """
        Compute the fly's average velocity during interaction events.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            The average velocity during interaction events.
        """
        # Get the fly data and interaction events
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]

        if not events:
            return np.nan

        # Calculate the fly's velocity
        fly_velocity = (
            np.sqrt(fly_data["x_thorax"].diff() ** 2 + fly_data["y_thorax"].diff() ** 2) * self.fly.experiment.fps
        )  # Convert to velocity in pixels/second

        # Extract velocity during interaction events
        velocities_during_events = []
        for event in events:
            start_idx, end_idx = event[0], event[1]
            velocities_during_events.extend(fly_velocity[start_idx:end_idx])

        # Return the average velocity during interaction events
        return np.nanmean(velocities_during_events)

    def compute_velocity_trend(self, fly_idx):
        """
        Compute the trend (slope) of the fly's velocity over time.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            The slope of the velocity trend over time.
        """
        # Get the fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate the fly's velocity
        fly_velocity = (
            np.sqrt(fly_data["x_thorax"].diff() ** 2 + fly_data["y_thorax"].diff() ** 2) * self.fly.experiment.fps
        )  # Convert to velocity in pixels/second

        # Remove NaN values
        valid_indices = ~np.isnan(fly_velocity)
        time = np.arange(len(fly_velocity))[valid_indices] / self.fly.experiment.fps  # Time in seconds
        velocity = fly_velocity[valid_indices]

        if len(time) < 2:
            return np.nan

        # Fit a linear regression to the velocity trend
        model = LinearRegression()
        model.fit(time.reshape(-1, 1), velocity)

        # Return the slope of the trend
        return model.coef_[0]

    def compute_binned_slope(self, fly_idx, ball_idx, n_bins=12):
        """
        Compute the slope of the ball position curve in time bins.

        Returns
        -------
        list of float: Slope for each bin.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        position = ball_data["y_centre"].values

        bins = np.linspace(time[0], time[-1], n_bins + 1)
        slopes = []

        for i in range(n_bins):
            mask = (time >= bins[i]) & (time < bins[i + 1])
            t_bin = time[mask]
            p_bin = position[mask]
            if len(t_bin) > 1 and not np.all(p_bin == p_bin[0]):
                model = LinearRegression()
                model.fit(t_bin.reshape(-1, 1), p_bin)
                slopes.append(model.coef_[0])
            else:
                slopes.append(np.nan)
        return slopes

    def compute_overall_slope(self, fly_idx, ball_idx):
        """
        Compute the overall slope of the ball trajectory.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        position = ball_data["y_centre"].values
        if len(time) < 2 or np.all(position == position[0]):
            return np.nan
        model = LinearRegression()
        model.fit(time.reshape(-1, 1), position)
        return model.coef_[0]

    def compute_interaction_rate_by_bin(self, fly_idx, ball_idx, n_bins=12):
        """
        Compute the rate of interactions per time bin.
        Returns a list of rates (events per second) for each bin.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        bins = np.linspace(time[0], time[-1], n_bins + 1)
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]
        rates = []
        for i in range(n_bins):
            bin_start, bin_end = bins[i], bins[i + 1]
            # Count events whose start falls in this bin
            count = sum(
                (event[0] / self.fly.experiment.fps >= bin_start) and (event[0] / self.fly.experiment.fps < bin_end)
                for event in events
            )
            duration = bin_end - bin_start
            rates.append(count / duration if duration > 0 else np.nan)
        return rates

    def compute_overall_interaction_rate(self, fly_idx, ball_idx):
        """
        Compute the overall rate of interactions (events per second).
        """
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]
        duration = self.tracking_data.duration
        return len(events) / duration if duration > 0 else np.nan

    def compute_auc(self, fly_idx, ball_idx):
        """
        Compute the area under the y_centre vs. time curve for the ball.

        Returns
        -------
        float: The total AUC.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        position = ball_data["y_centre"].values
        auc = np.trapz(position, time)
        return auc

    def compute_binned_auc(self, fly_idx, ball_idx, n_bins=12):
        """
        Compute the AUC in each time bin.

        Returns
        -------
        list of float: AUC for each bin.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        position = ball_data["y_centre"].values
        bins = np.linspace(time[0], time[-1], n_bins + 1)
        aucs = []
        for i in range(n_bins):
            mask = (time >= bins[i]) & (time < bins[i + 1])
            t_bin = time[mask]
            p_bin = position[mask]
            if len(t_bin) > 1:
                aucs.append(np.trapz(p_bin, t_bin))
            else:
                aucs.append(np.nan)
        return aucs

    def get_has_finished(self, fly_idx, ball_idx):
        """
        Check if a final event exists for this fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        int
            1 if a final event exists, 0 otherwise.
        """
        final_event = self.get_final_event(fly_idx, ball_idx)
        return 1 if final_event is not None else 0

    def compute_persistence_at_end(self, fly_idx):
        """
        Compute the fraction of time the fly spent at a certain distance from start.
        This distance is defined by the corridor_end config parameter.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Fraction of time spent at corridor_end distance from start.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Get the fly start position using median of first 10 frames
        start_x, start_y, _, _ = self._calculate_median_coordinates(fly_data, start_idx=0, window=10, keypoint="thorax")

        # Calculate distance from start position for each frame
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"], fly_data["y_thorax"], start_x, start_y
        )

        # Get corridor_end distance from config (assuming it exists)
        corridor_end_distance = getattr(self.fly.config, "corridor_end_threshold", 170)  # Default to 170 if not set

        # Count frames where fly is at or beyond corridor_end distance
        at_end = distances >= corridor_end_distance

        # Calculate fraction of time
        total_frames = len(distances)
        if total_frames > 0:
            return np.sum(at_end) / total_frames
        else:
            return np.nan

    def compute_fly_distance_moved(self, fly_idx):
        """
        Compute the overall distance moved by the fly over the whole experiment.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Total distance moved by the fly in millimeters.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate frame-to-frame distances
        x_diff = fly_data["x_thorax"].diff()
        y_diff = fly_data["y_thorax"].diff()

        # Calculate Euclidean distance for each frame transition
        frame_distances = np.sqrt(x_diff**2 + y_diff**2)

        # Sum all distances (excluding NaN from first frame)
        total_distance_pixels = np.nansum(frame_distances)

        # Convert from pixels to millimeters
        pixels_per_mm = getattr(self.fly.config, "pixels_per_mm", 500 / 30)  # Default conversion factor
        total_distance_mm = total_distance_pixels / pixels_per_mm

        return total_distance_mm

    def get_time_chamber_beginning(self, fly_idx):
        """
        Compute the time spent by the fly in the chamber during the first 25% of the video.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Time spent in chamber during first 25% of video in seconds.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate the end frame for first 25% of video
        total_frames = len(fly_data)
        end_frame_25_percent = int(total_frames * 0.25)

        # Get the fly start position using median of first 10 frames
        start_x, start_y, _, _ = self._calculate_median_coordinates(fly_data, start_idx=0, window=10, keypoint="thorax")

        # Calculate distance from start position for first 25% of frames
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"].iloc[:end_frame_25_percent],
            fly_data["y_thorax"].iloc[:end_frame_25_percent],
            start_x,
            start_y,
        )

        # Determine frames where fly is in chamber
        in_chamber = distances <= self.fly.config.chamber_radius

        # Calculate time spent in chamber
        time_in_chamber = np.sum(in_chamber) / self.fly.experiment.fps

        return time_in_chamber

    def compute_median_freeze_duration(self, fly_idx):
        """
        Compute the median duration of pause events for a given fly.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Median duration of pause events in seconds.
        """
        # Get pause metrics which include individual pause durations
        pauses = self.detect_pauses(fly_idx)

        if not pauses:
            return np.nan

        # Extract durations from pause tuples (start_time, end_time, duration)
        durations = [pause[2] for pause in pauses]

        # Calculate median duration
        median_duration = np.median(durations)

        return median_duration
