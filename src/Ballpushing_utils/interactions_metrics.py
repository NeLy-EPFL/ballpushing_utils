import numpy as np
from Ballpushing_utils.ballpushing_metrics import BallPushingMetrics


class InteractionsMetrics:

    def __init__(self, tracking_data):
        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.summary = BallPushingMetrics(self.fly.tracking_data, compute_metrics_on_init=False)
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
        Compute metrics for all combinations of fly and ball, including interaction and random events.

        Returns:
            dict: Nested dictionary containing metrics for each fly-ball combination and their events.
        """
        all_metrics = {}

        # Compute metrics for interaction events
        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}_interaction"
                if self.fly.config.debugging:
                    print(f"Computing metrics for {key}...")
                all_metrics[key] = self.compute_event_metrics(fly_idx, ball_idx, event_type="interaction")

        if self.fly.config.generate_random:
            # Compute metrics for random events if available
            if hasattr(self.tracking_data, "random_events"):
                for fly_idx, ball_dict in self.tracking_data.random_events.items():
                    for ball_idx, events in ball_dict.items():
                        key = f"fly_{fly_idx}_ball_{ball_idx}_random"
                        if self.fly.config.debugging:
                            print(f"Computing metrics for {key}...")
                        all_metrics[key] = self.compute_event_metrics(fly_idx, ball_idx, event_type="random")

        self.metrics = all_metrics
        if self.fly.config.debugging:
            print("Finished compute_all_event_metrics.")
        return all_metrics

    def compute_event_metrics(self, fly_idx, ball_idx, event_type="interaction"):
        """
        Compute metrics for each event (interaction or random).

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.
            event_type (str): Type of event ("interaction" or "random").

        Returns:
            dict: Dictionary where each key is an event index, and the value is a dictionary of metrics for that event.
        """
        if event_type == "interaction":
            events = self.tracking_data.interaction_events[fly_idx][ball_idx]
        elif event_type == "random":
            events = self.tracking_data.random_events[fly_idx][ball_idx]
        else:
            raise ValueError(f"Invalid event_type: {event_type}")

        if self.fly.config.debugging:
            print(f"Number of events: {len(events)}")

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        if self.fly.config.debugging:
            print("Precomputing rolling medians for ball and fly data...")
        ball_data = self.precompute_rolling_medians(ball_data, window=10, keypoint="centre")
        fly_data = self.precompute_rolling_medians(fly_data, window=10, keypoint="thorax")
        if self.fly.config.debugging:
            print("Rolling medians precomputed.")

        # Calculate the initial position of the ball
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, end_idx=len(ball_data) - 1, keypoint="centre"
        )
        if self.fly.config.debugging:
            print(f"Initial ball position: ({initial_x}, {initial_y})")

        max_event = self.summary.get_max_event(fly_idx, ball_idx)
        final_event = self.summary.get_final_event(fly_idx, ball_idx)

        metrics = {}
        previous_success = None  # To store the success of the previous event

        for event_idx, event in enumerate(events):
            if self.fly.config.debugging:
                print(f"Processing event {event_idx}...")
            start_idx, end_idx = event[0], event[1]

            # Duration of the event
            duration = (end_idx - start_idx) / self.fly.experiment.fps

            # Ball displacement
            start_x, start_y, end_x, end_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_idx, end_idx=end_idx, keypoint="centre"
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

            # Max event (1 if it is, 0 if not)
            is_max_event = 1 if event_idx == max_event[0] else 0

            # Final event (1 if it is, 0 if not)
            is_final_event = 1 if event_idx == final_event[0] else 0

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
                "max_event": is_max_event,
                "final_event": is_final_event,
                "ball_velocity": ball_velocity,
                "efficiency_diff": efficiency_diff,
                "event_type": event_type,
            }
            if self.fly.config.debugging:
                print(f"Finished processing event {event_idx}.")
        if self.fly.config.debugging:
            print(f"Finished compute_event_metrics for fly_idx={fly_idx}, ball_idx={ball_idx}, event_type={event_type}")
        return metrics

    def precompute_rolling_medians(self, data, window, keypoint):
        """
        Precompute rolling medians for the given keypoint and store them in the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing tracking data.
            window (int): Number of frames to use for rolling median calculation.
            keypoint (str): Keypoint to calculate rolling medians for.

        Returns:
            pd.DataFrame: DataFrame with added rolling median columns.
        """
        data[f"x_{keypoint}_rolling_median"] = data[f"x_{keypoint}"].rolling(window=window, center=True).median()
        data[f"y_{keypoint}_rolling_median"] = data[f"y_{keypoint}"].rolling(window=window, center=True).median()

        # Replace NaN values with forward fill and backward fill
        data[f"x_{keypoint}_rolling_median"] = data[f"x_{keypoint}_rolling_median"].ffill().bfill()
        data[f"y_{keypoint}_rolling_median"] = data[f"y_{keypoint}_rolling_median"].ffill().bfill()

        if self.fly.config.debugging:
            print(f"Finished precompute_rolling_medians for keypoint={keypoint}")
        return data

    def _calculate_median_coordinates(self, data, start_idx, end_idx, keypoint):
        """
        Retrieve the precomputed median coordinates for the start and end of an event.

        Args:
            data (pd.DataFrame): DataFrame containing tracking data with precomputed rolling medians.
            start_idx (int): Start index of the event.
            end_idx (int): End index of the event.
            keypoint (str): Keypoint to calculate coordinates for.

        Returns:
            tuple: Median x and y coordinates for the start and end of the event.
        """
        start_x = data[f"x_{keypoint}_rolling_median"].iloc[start_idx]
        start_y = data[f"y_{keypoint}_rolling_median"].iloc[start_idx]
        end_x = data[f"x_{keypoint}_rolling_median"].iloc[end_idx]
        end_y = data[f"y_{keypoint}_rolling_median"].iloc[end_idx]

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
        Determine the direction of an event (push or pull) if the event is significant.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.
            start_idx (int): Start index of the event.
            end_idx (int): End index of the event.

        Returns:
            int: 1 for "push", -1 for "pull", or 0 if the event is not significant.
        """
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Calculate the median coordinates for the fly and ball
        start_fly_x, start_fly_y, _, _ = self._calculate_median_coordinates(
            fly_data, start_idx=start_idx, end_idx=end_idx, keypoint="thorax"
        )
        start_ball_x, start_ball_y, end_ball_x, end_ball_y = self._calculate_median_coordinates(
            ball_data, start_idx=start_idx, end_idx=end_idx, keypoint="centre"
        )

        # Calculate the displacement of the ball during the event
        displacement = self._calculate_euclidean_distance(start_ball_x, start_ball_y, end_ball_x, end_ball_y)

        # Check if the event is significant
        if displacement <= self.fly.config.significant_threshold:
            return 0  # Not significant, set direction to 0

        # Calculate the distances between the fly and the ball
        start_distance = self._calculate_euclidean_distance(start_fly_x, start_fly_y, start_ball_x, start_ball_y)
        end_distance = self._calculate_euclidean_distance(start_fly_x, start_fly_y, end_ball_x, end_ball_y)

        # Determine the direction
        if end_distance > start_distance:
            return 1  # Push
        elif end_distance < start_distance:
            return -1  # Pull
        else:
            return 0  # No change

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
        distances = np.sqrt(
            (fly_data["x_thorax"] - self.tracking_data.start_x) ** 2
            + (fly_data["y_thorax"] - self.tracking_data.start_y) ** 2
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
