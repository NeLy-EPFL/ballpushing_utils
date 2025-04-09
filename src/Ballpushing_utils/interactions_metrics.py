import numpy as np
import pandas as pd
from typing import Any
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from utils_behavior import Processing


class InteractionsMetrics:

    def __init__(self, tracking_data):

        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        # Iterate over the indices of the flytrack objects list
        self.chamber_exit_times = {
            fly_idx: self.get_chamber_exit_time(fly_idx, 0)
            for fly_idx in range(len(self.tracking_data.flytrack.objects))
        }

        if self.fly.config.debugging:
            for fly_idx, exit_time in self.chamber_exit_times.items():
                print(f"Fly {fly_idx} chamber exit time: {exit_time}")

        self.metrics = {}
        self.compute_all_event_metrics()

    def compute_all_event_metrics(self):
        """
        Compute metrics for all combinations of fly and ball.

        Returns:
            dict: Nested dictionary containing metrics for each fly-ball combination and their events.
        """
        all_metrics = {}

        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}"
                all_metrics[key] = self.compute_event_metrics(fly_idx, ball_idx)

        self.metrics = all_metrics
        return all_metrics

    def compute_event_metrics(self, fly_idx, ball_idx):
        """
        Compute metrics for each interaction event.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.

        Returns:
            dict: Dictionary where each key is an event index, and the value is a dictionary of metrics for that event.
        """
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate the initial position of the ball
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, end_idx=len(ball_data), window=10, keypoint="centre"
        )

        metrics = {}
        previous_success = None  # To store the success of the previous event

        for event_idx, event in enumerate(events):
            start_idx, end_idx = event[0], event[1]

            # Duration of the event
            duration = (end_idx - start_idx) / self.fly.experiment.fps

            # Ball displacement
            start_x, start_y, end_x, end_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
            )
            displacement = self._calculate_euclidean_distance(start_x, start_y, end_x, end_y)

            # Start distance and end distance from the initial position
            start_distance = self._calculate_euclidean_distance(start_x, start_y, initial_x, initial_y)
            end_distance = self._calculate_euclidean_distance(end_x, end_y, initial_x, initial_y)

            # Direction (push or pull)
            direction = self._determine_event_direction(fly_idx, ball_idx, start_idx, end_idx)

            # Significance (1 if displacement > threshold, else 0)
            significant = int(displacement > self.fly.config.significant_threshold)

            # Major event (1 if displacement > major event threshold, else 0)
            major_event = int(displacement > self.fly.config.major_event_threshold)

            # Ball velocity during the interaction
            ball_velocity = displacement / duration if duration > 0 else 0

            # Correlation with the previous event
            efficiency_diff = None
            if previous_success is not None:
                efficiency_diff = ball_velocity - previous_success

            # Update the previous success for the next iteration
            previous_success = ball_velocity

            # Store metrics for this event
            metrics[event_idx] = {
                "start_time": start_idx / self.fly.experiment.fps,
                "end_time": end_idx / self.fly.experiment.fps,
                "duration": duration,
                "displacement": displacement,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "direction": direction,
                "significant": significant,
                "major_event": major_event,
                "ball_velocity": ball_velocity,
                "efficiency_diff": efficiency_diff,
            }

        return metrics

    def _calculate_median_coordinates(self, data, start_idx, end_idx, window, keypoint):
        """
        Calculate the median coordinates for the start and end of an event.

        Args:
            data (pd.DataFrame): DataFrame containing tracking data.
            start_idx (int): Start index of the event.
            end_idx (int): End index of the event.
            window (int): Number of frames to use for median calculation.
            keypoint (str): Keypoint to calculate coordinates for.

        Returns:
            tuple: Median x and y coordinates for the start and end of the event.
        """
        start_x = data[f"x_{keypoint}"].iloc[start_idx : start_idx + window].median()
        start_y = data[f"y_{keypoint}"].iloc[start_idx : start_idx + window].median()
        end_x = data[f"x_{keypoint}"].iloc[end_idx - window : end_idx].median()
        end_y = data[f"y_{keypoint}"].iloc[end_idx - window : end_idx].median()
        return start_x, start_y, end_x, end_y

    def _calculate_euclidean_distance(self, x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            x1, y1, x2, y2 (float): Coordinates of the two points.

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _determine_event_direction(self, fly_idx, ball_idx, start_idx, end_idx):
        """
        Determine the direction of an event (push or pull).

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.
            start_idx (int): Start index of the event.
            end_idx (int): End index of the event.

        Returns:
            str: "push", "pull", or "unknown".
        """
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        start_fly_x, start_fly_y, _, _ = self._calculate_median_coordinates(
            fly_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="thorax"
        )
        start_ball_x, start_ball_y, end_ball_x, end_ball_y = self._calculate_median_coordinates(
            ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
        )

        start_distance = self._calculate_euclidean_distance(start_fly_x, start_fly_y, start_ball_x, start_ball_y)
        end_distance = self._calculate_euclidean_distance(start_fly_x, start_fly_y, end_ball_x, end_ball_y)

        if end_distance > start_distance:
            return 1
        elif end_distance < start_distance:
            return -1
        else:
            return None

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
