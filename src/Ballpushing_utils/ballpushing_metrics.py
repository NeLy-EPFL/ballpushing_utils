class BallPushingMetrics:
    def __init__(self, tracking_data):

        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.metrics = {}
        self.compute_metrics()
        # TODO: Compute maximum distance pushed (corresponding to max_event)

    def compute_metrics(self):
        """
        Compute and store various metrics for each pair of fly and ball.
        """
        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}"

                try:
                    nb_events = self.get_adjusted_nb_events(fly_idx, ball_idx, signif=False)
                except Exception as e:
                    nb_events = np.nan

                    if self.fly.config.debugging:
                        print(f"Error in get_adjusted_nb_events: {e}")

                try:
                    max_event = self.get_max_event(fly_idx, ball_idx)
                except Exception as e:
                    max_event = (np.nan, np.nan)

                    if self.fly.config.debugging:
                        print(f"Error in get_max_event: {e}")

                try:
                    max_distance = self.get_max_distance(fly_idx, ball_idx)
                except Exception as e:
                    max_distance = np.nan

                    if self.fly.config.debugging:
                        print(f"Error in get_max_distance: {e}")

                try:
                    significant_events = self.get_significant_events(fly_idx, ball_idx)
                except Exception as e:
                    significant_events = []

                    if self.fly.config.debugging:
                        print(f"Error in get_significant_events: {e}")

                try:
                    if self.fly.config.experiment_type == "F1":
                        nb_significant_events = self.get_adjusted_nb_events(fly_idx, ball_idx, signif=True)
                    else:
                        nb_significant_events = len(significant_events)
                except Exception as e:
                    nb_significant_events = np.nan

                    if self.fly.config.debugging:
                        print(f"Error in get_adjusted_nb_events: {e}")

                try:
                    first_significant_event = self.get_first_significant_event(fly_idx, ball_idx)
                except Exception as e:
                    first_significant_event = (np.nan, np.nan)

                    if self.fly.config.debugging:
                        print(f"Error in get_first_significant_event: {e}")

                try:
                    aha_moment = self.get_aha_moment(fly_idx, ball_idx)
                except Exception as e:
                    aha_moment = (np.nan, np.nan)

                    if self.fly.config.debugging:
                        print(f"Error in get_aha_moment: {e}")

                try:
                    events_direction = self.find_events_direction(fly_idx, ball_idx)
                except Exception as e:
                    events_direction = ([], [])

                    if self.fly.config.debugging:
                        print(f"Error in find_events_direction: {e}")

                try:
                    final_event = self.get_final_event(fly_idx, ball_idx)
                except Exception as e:
                    final_event = (np.nan, np.nan)

                    if self.fly.config.debugging:
                        print(f"Error in get_final_event: {e}")

                try:
                    success_direction = self.get_success_direction(fly_idx, ball_idx)
                except Exception as e:
                    success_direction = np.nan

                    if self.fly.config.debugging:
                        print(f"Error in get_success_direction: {e}")

                try:
                    cumulated_breaks_duration = self.get_cumulated_breaks_duration(fly_idx, ball_idx)
                except Exception as e:
                    cumulated_breaks_duration = np.nan

                    if self.fly.config.debugging:
                        print(f"Error in get_cumulated_breaks_duration: {e}")

                try:
                    distance_moved = self.get_distance_moved(fly_idx, ball_idx)
                except Exception as e:
                    distance_moved = np.nan

                    if self.fly.config.debugging:
                        print(f"Error in get_distance_moved: {e}")

                try:
                    insight_effect = self.get_insight_effect(fly_idx, ball_idx) if aha_moment else np.nan
                except Exception as e:
                    insight_effect = np.nan

                    if self.fly.config.debugging:
                        print(f"Error in get_insight_effect: {e}")

                self.metrics[key] = {
                    "nb_events": nb_events,
                    "max_event": max_event[0],
                    "max_event_time": max_event[1],
                    "max_distance": max_distance,
                    "final_event": final_event[0],
                    "final_event_time": final_event[1],
                    "nb_significant_events": nb_significant_events,
                    "significant_ratio": (nb_significant_events / nb_events if nb_events > 0 else np.nan),
                    "first_significant_event": first_significant_event[0],
                    "first_significant_event_time": first_significant_event[1],
                    "aha_moment": aha_moment[0],
                    "aha_moment_time": aha_moment[1],
                    "aha_moment_first": insight_effect["first_event"],
                    "insight_effect": insight_effect["raw_effect"],
                    "insight_effect_log": insight_effect["log_effect"],
                    "cumulated_breaks_duration": cumulated_breaks_duration,
                    "pushed": len(events_direction[0]),
                    "pulled": len(events_direction[1]),
                    "pulling_ratio": (
                        len(events_direction[1]) / (len(events_direction[0]) + len(events_direction[1]))
                        if (len(events_direction[0]) + len(events_direction[1])) > 0
                        else np.nan
                    ),
                    "success_direction": success_direction,
                    "interaction_proportion": (
                        sum([event[2] for event in events])
                        / (sum([event[2] for event in events]) + cumulated_breaks_duration)
                        if cumulated_breaks_duration > 0
                        else np.nan
                    ),
                    "distance_moved": distance_moved,
                    "exit_time": self.tracking_data.exit_time,
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
                len(events) * self.fly.config.adjusted_events_normalisation / self.tracking_data.duration
                if self.tracking_data.duration > 0
                else 0
            )

        return adjusted_nb_events

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

        # ball_data["euclidean_distance"] = np.sqrt(
        #     (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
        #     + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
        # )

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

    def get_max_event(self, fly_idx, ball_idx, threshold=None):
        """
        Get the event at which the ball was moved at its maximum distance for a given fly and ball. Maximum here doesn't mean the ball has reached the end of the corridor.

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

        max_event, max_event_idx = self.find_event_by_distance(fly_idx, ball_idx, threshold, distance_type="max")

        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            max_event_time = max_event[0] / self.fly.experiment.fps if max_event else None
        else:
            max_event_time = (
                (max_event[0] / self.fly.experiment.fps) - self.tracking_data.exit_time if max_event else None
            )

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

        max_distance = np.sqrt(
            (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
            + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
        ).max()

        return max_distance

    def get_final_event(self, fly_idx, ball_idx, threshold=None, init=False):

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            threshold = self.fly.config.final_event_threshold

            final_event, final_event_idx = self.find_event_by_distance(
                fly_idx, ball_idx, threshold, distance_type="threshold"
            )

            final_event_time = final_event[0] / self.fly.experiment.fps if final_event else None

        else:
            threshold = self.fly.config.final_event_F1_threshold

            final_event, final_event_idx = self.find_event_by_distance(
                fly_idx, ball_idx, threshold, distance_type="threshold"
            )

            final_event_time = (
                (final_event[0] / self.fly.experiment.fps) - self.tracking_data.exit_time if final_event else None
            )

        return final_event_idx, final_event_time

    def get_significant_events(self, fly_idx, ball_idx, distance=5):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = [
            (event, i)
            for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        return significant_events

    def get_first_significant_event(self, fly_idx, ball_idx, distance=5):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = self.get_significant_events(fly_idx, ball_idx, distance=distance)

        if significant_events:
            first_significant_event = significant_events[0]
            first_significant_event_idx = first_significant_event[1]

            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                first_significant_event_time = first_significant_event[0][0] / self.fly.experiment.fps
            else:
                first_significant_event_time = (
                    first_significant_event[0][0] / self.fly.experiment.fps
                ) - self.tracking_data.exit_time

            return first_significant_event_idx, first_significant_event_time
        else:
            return None, None

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

    def get_cumulated_breaks_duration(self, fly_idx, ball_idx):
        breaks = self.find_breaks(fly_idx, ball_idx)
        cumulated_breaks_duration = sum([break_[2] for break_ in breaks])
        return cumulated_breaks_duration

    def find_events_direction(self, fly_idx, ball_idx):
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = self.get_significant_events(fly_idx, ball_idx)

        pushing_events = []
        pulling_events = []

        for event in significant_events:
            event = event[0]

            start_roi = event[0]
            end_roi = event[1]

            start_distance = np.sqrt(
                (ball_data.loc[start_roi, "x_centre"] - fly_data.loc[start_roi, "x_thorax"]) ** 2
                + (ball_data.loc[start_roi, "y_centre"] - fly_data.loc[start_roi, "y_thorax"]) ** 2
            )
            end_distance = np.sqrt(
                (ball_data.loc[end_roi, "x_centre"] - fly_data.loc[start_roi, "x_thorax"]) ** 2
                + (ball_data.loc[end_roi, "y_centre"] - fly_data.loc[start_roi, "y_thorax"]) ** 2
            )

            if end_distance > start_distance:
                pushing_events.append(event)
            else:
                pulling_events.append(event)

        return pushing_events, pulling_events

    def get_distance_moved(self, fly_idx, ball_idx, subset=None):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if subset is None:
            subset = self.tracking_data.interaction_events[fly_idx][ball_idx]

        for event in subset:
            ball_data.loc[event[0] : event[1], "euclidean_distance"] = np.sqrt(
                (ball_data["x_centre"].iloc[event[1]] - ball_data["x_centre"].iloc[event[0]]) ** 2
                + (ball_data["y_centre"].iloc[event[1]] - ball_data["y_centre"].iloc[event[0]]) ** 2
            )

        return ball_data["euclidean_distance"].sum()

    def get_aha_moment(self, fly_idx, ball_idx, distance=None):
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
            distance = self.fly.config.aha_moment_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        aha_moment = [
            (event, i)
            for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        if aha_moment:
            # Select the event right before the event at which the ball was moved more than the threshold
            aha_moment_event, aha_moment_idx = aha_moment[0]
            if aha_moment_idx > 0:
                previous_event = self.tracking_data.interaction_events[fly_idx][ball_idx][aha_moment_idx - 1]
                aha_moment_event = previous_event
                aha_moment_idx -= 1

            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                aha_moment_time = aha_moment_event[0] / self.fly.experiment.fps
            else:
                aha_moment_time = (aha_moment_event[0] / self.fly.experiment.fps) - self.tracking_data.exit_time

            return aha_moment_idx, aha_moment_time
        else:
            return None, None

    def get_insight_effect(self, fly_idx, ball_idx, epsilon=1e-6, strength_threshold=2):
        """
        Calculate enhanced insight effect with performance analytics.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        epsilon : float, optional
            Smoothing factor to prevent division by zero.
        strength_threshold : float, optional
            Threshold for strong/weak classification.

        Returns
        -------
        dict
            Dictionary containing multiple insight metrics:
            - raw_effect: Base ratio of post/pre aha performance
            - log_effect: Log-transformed effect for normal distribution
            - classification: Strong/weak based on threshold
            - first_event: Flag for aha moment as first interaction
            - post_aha_count: Number of post-aha events
        """
        significant_events = [event[0] for event in self.get_significant_events(fly_idx, ball_idx)]
        aha_moment_index, _ = self.get_aha_moment(fly_idx, ball_idx)

        # Handle no significant events case early
        if not significant_events:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        # Handle no aha moment case early
        if aha_moment_index is None:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        # Segment events with aha moment in before period
        before_aha = significant_events[: aha_moment_index + 1]
        after_aha = significant_events[aha_moment_index + 1 :]

        # Calculate average distances with safety checks
        avg_before = self._calculate_avg_distance(fly_idx, ball_idx, before_aha)
        avg_after = self._calculate_avg_distance(fly_idx, ball_idx, after_aha)

        # Core insight calculation
        if aha_moment_index == 0:
            insight_effect = 1.0
        elif avg_before == 0:
            insight_effect = np.nan
        else:
            insight_effect = avg_after / avg_before

        # Transformations and classifications
        log_effect = np.log(insight_effect + 1) if insight_effect > 0 else np.nan
        classification = "strong" if insight_effect > strength_threshold else "weak"

        return {
            "raw_effect": insight_effect,
            "log_effect": log_effect,
            "classification": classification,
            "first_event": aha_moment_index == 0,
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

        if threshold is None:
            threshold = self.fly.config.success_direction_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        initial_y = ball_data["y_centre"].iloc[0]

        ball_data["euclidean_distance"] = np.sqrt(
            (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
            + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
        )

        moved_data = ball_data[ball_data["euclidean_distance"] >= threshold]

        if moved_data.empty:
            return None

        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
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

    def get_time_chamber(self):
        """
        Get the time spent by the fly in the chamber, meaning within a 50 px radius of the fly start position.

        Returns
        -------
        float
            Time spent in the chamber in seconds.
        """
        # Get the tracking data for the fly
        tracking_data = self.tracking_data

        # Determine the start position by averaging the first 10-20 frames
        num_frames_to_average = 20
        start_position_x = np.mean(tracking_data.x[:num_frames_to_average])
        start_position_y = np.mean(tracking_data.y[:num_frames_to_average])

        # Calculate the distance from the start position for each frame
        distances = np.sqrt((tracking_data.x - start_position_x) ** 2 + (tracking_data.y - start_position_y) ** 2)

        # Determine the frames where the fly is within a 50 px radius of the start position
        in_chamber = distances <= 50

        # Calculate the time spent in the chamber
        time_in_chamber = np.sum(in_chamber) / self.fly.experiment.fps

        return time_in_chamber
