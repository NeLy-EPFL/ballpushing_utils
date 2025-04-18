from utils_behavior import Sleap_utils
from Ballpushing_utils import utilities
from Ballpushing_utils.ballpushing_metrics import BallPushingMetrics
import numpy as np
import warnings


class FlyTrackingData:
    def __init__(self, fly, time_range=None, log_missing=False, keep_idle=False):
        self.fly = fly
        self.flytrack = None
        self.balltrack = None
        self.skeletontrack = None
        self.valid_data = True
        self.log_missing = log_missing
        self.keep_idle = keep_idle
        self.cutoff_reference = None  # New cutoff reference storage

        try:
            # Load tracking files
            self.balltrack = self.load_tracking_file("*ball*.h5", "ball")
            self.flytrack = self.load_tracking_file("*fly*.h5", "fly")
            self.skeletontrack = self.load_tracking_file(
                "*full_body*.h5",
                "fly",
                smoothing=self.fly.config.skeleton_tracks_smoothing,
            )

            if self.balltrack is None or (self.flytrack is None and self.skeletontrack is None):
                print(f"Missing tracking files for {self.fly.metadata.name}")
                self.valid_data = False
                if self.log_missing:
                    self.log_missing_fly()
                return

            # Calculate euclidean distances for all balls (new line)
            self.calculate_euclidean_distances()

        except Exception as e:
            print(f"Error loading files for {self.fly.metadata.name}: {e}")
            self.valid_data = False
            if self.log_missing:
                self.log_missing_fly()
            return

        # Data quality checks
        self.valid_data = self.check_data_quality()
        self.check_dying()

        if self.valid_data or self.keep_idle:
            self.duration = self.balltrack.objects[0].dataset["time"].iloc[-1]
            self.start_x, self.start_y = self.get_initial_position()
            if self.fly.config.debugging:
                print(f"Initial position for {self.fly.metadata.name}: ({self.start_x}, {self.start_y})")

            self.fly_skeleton = self.get_skeleton()
            self.exit_time = self.get_exit_time()
            self.adjusted_time = self.compute_adjusted_time()

        # Apply initial time range filter if specified in config
        if self.fly.config.time_range:
            self.filter_tracking_data(self.fly.config.time_range)

        # Determine success cutoff reference
        if self.fly.config.success_cutoff:
            self._determine_success_cutoff()

    def load_tracking_file(
        self,
        pattern,
        object_type,
        smoothing=None,
    ):
        """Load a tracking file for the fly."""

        if smoothing is None:
            smoothing = self.fly.config.tracks_smoothing

        try:
            tracking_file = list(self.fly.directory.glob(pattern))[0]
            return Sleap_utils.Sleap_Tracks(
                tracking_file,
                object_type=object_type,
                smoothed_tracks=smoothing,
                debug=False,
            )
        except IndexError:
            return None

    def calculate_euclidean_distances(self):
        """
        Calculate euclidean distance for all balls in the tracking data.
        This is the distance from the ball's initial position.
        """
        if self.balltrack is None:
            return

        for ball_idx in range(len(self.balltrack.objects)):
            ball_data = self.balltrack.objects[ball_idx].dataset

            # Calculate euclidean distance from initial position
            ball_data["euclidean_distance"] = np.sqrt(
                (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
                + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
            )

            # Also add an alias column for learning experiments
            ball_data[f"distance_ball_{ball_idx}"] = ball_data["euclidean_distance"]

    def _determine_success_cutoff(self):
        """
        Calculate success cutoff based on the final event using ball-pushing metrics.
        This method computes the final event (if any) and crops the data up to the end of the final event.
        """
        if self.balltrack is None or self.balltrack.objects is None:
            warnings.warn("Balltrack or its objects are not initialized.")
            return

        if self.flytrack is None or self.flytrack.objects is None:
            warnings.warn("Flytrack or its objects are not initialized.")
            return

        ballpushing_metrics = BallPushingMetrics(self, compute_metrics_on_init=False)

        # TODO: that's gonna be an issue for F1, we'll look at it later.
        final_event = ballpushing_metrics.get_final_event(0, 0)

        # Check if the final event is not None
        if final_event is not None:
            # Set the cutoff reference to the end of the final event
            self.cutoff_reference = final_event[2] + 15  # Adding 1 to include the last frame

        # Reset cached calculations
        for attr in [
            "_interaction_events",
            "_interactions_onsets",
            "_interactions_offsets",
            "_std_interactions",
            "_random_events",
            "_random_events_onsets",
            "_random_events_offsets",
            "_std_random_events",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def interaction_events(self):
        """Chunks of time where the fly is interacting with the ball."""
        if not hasattr(self, "_interaction_events"):
            time_range = (0, self.cutoff_reference) if self.cutoff_reference else None

            self._interaction_events = self._calculate_interactions(time_range)
        return self._interaction_events

    def _calculate_interactions(self, time_range=None):
        """Actual event detection with optional time range"""
        if self.balltrack is None or self.balltrack.objects is None:
            warnings.warn("Balltrack or its objects are not initialized.")
            return
        if self.flytrack is None or self.flytrack.objects is None:
            warnings.warn("Flytrack or its objects are not initialized.")
            return

        events = {}
        for fly_idx in range(len(self.flytrack.objects)):
            events[fly_idx] = {}
            for ball_idx in range(len(self.balltrack.objects)):
                events[fly_idx][ball_idx] = self.find_flyball_interactions(
                    time_range=time_range, fly_idx=fly_idx, ball_idx=ball_idx
                )
        return events

    def find_flyball_interactions(
        self,
        gap_between_events=None,
        event_min_length=None,
        thresh=None,
        time_range=None,
        fly_idx=0,
        ball_idx=0,
    ):
        """
        Find interaction events between the fly and the ball.
        It uses the find_interaction_events function from the Ballpushing_utils module to find chunks of time where the fly ball distance is below a threshold thresh for a minimum of event_min_length seconds. It also merge together events that are separated by less than gap_between_events seconds.
        """
        # Parameter handling unchanged
        if thresh is None:
            thresh = self.fly.config.interaction_threshold
        if gap_between_events is None:
            gap_between_events = self.fly.config.gap_between_events
        if event_min_length is None:
            event_min_length = self.fly.config.events_min_length

        if self.flytrack is None or self.balltrack is None:
            return []

        # Apply time range filtering at detection time
        fly_data = self.flytrack.objects[fly_idx].dataset
        ball_data = self.balltrack.objects[ball_idx].dataset

        if time_range:
            # Convert frame indices to integers for iloc
            start = int(time_range[0] * self.fly.experiment.fps)
            end = int(time_range[1] * self.fly.experiment.fps) if time_range[1] else None
            fly_data = fly_data.iloc[start:end]
            ball_data = ball_data.iloc[start:end]

        # Original event detection logic
        interaction_events = utilities.find_interaction_events(
            fly_data,
            ball_data,
            nodes1=["thorax"],
            nodes2=["centre"],
            threshold=thresh,
            gap_between_events=gap_between_events,
            event_min_length=event_min_length,
            fps=self.fly.experiment.fps,
        )
        return interaction_events

    @property
    def events_before_cutoff(self):
        """Number of events before cutoff."""
        if self.cutoff_reference:
            all_events = self._calculate_interactions(None)
            filtered_events = self._calculate_interactions((0, self.cutoff_reference))

            # Ensure all_events and filtered_events are not None
            if all_events is None or filtered_events is None:
                warnings.warn("Interaction events could not be calculated.")
                return None, None

            all_count = sum(len(events) for fly_dict in all_events.values() for events in fly_dict.values())
            filtered_count = sum(len(events) for fly_dict in filtered_events.values() for events in fly_dict.values())
            return filtered_count, all_count

        return None, None

    @property
    def interactions_onsets(self):
        """
        For each interaction event, get the standardized onset of the fly interaction with the ball.
        """
        if not hasattr(self, "_interactions_onsets"):
            self._interactions_onsets = self._calculate_interactions_boundaries()[0]
        return self._interactions_onsets

    @property
    def interactions_offsets(self):
        """
        For each interaction event, get the standardized offset of the fly interaction with the ball.
        """
        if not hasattr(self, "_interactions_offsets"):
            self._interactions_offsets = self._calculate_interactions_boundaries()[1]
        return self._interactions_offsets

    def _calculate_interactions_boundaries(self):
        """
        Calculate standardized interaction onsets, offsets, and centers based on event centers.
        """
        if self.flytrack is None or self.balltrack is None:
            print(f"Skipping interaction events for {self.fly.metadata.name} due to missing tracking data.")
            return {}, {}, {}

        interactions_onsets = {}
        interactions_offsets = {}
        interactions_centers = {}

        for fly_idx in range(len(self.flytrack.objects)):
            fly_data = self.flytrack.objects[fly_idx].dataset

            for ball_idx in range(len(self.balltrack.objects)):
                ball_data = self.balltrack.objects[ball_idx].dataset

                # Ensure interaction_events is not None and contains the required keys
                if (
                    self.interaction_events is None
                    or fly_idx not in self.interaction_events
                    or ball_idx not in self.interaction_events[fly_idx]
                ):
                    warnings.warn(f"Interaction events missing for fly_idx={fly_idx}, ball_idx={ball_idx}.")
                    continue

                interaction_events = self.interaction_events[fly_idx][ball_idx]

                onsets = []
                offsets = []
                centers = []
                for event in interaction_events:
                    # Calculate the center of the event
                    event_center = (event[0] + event[1]) // 2

                    # Calculate standardized onset and offset without clamping
                    onset = event_center - self.fly.config.frames_before_onset
                    offset = event_center + self.fly.config.frames_after_onset

                    onsets.append(onset)
                    offsets.append(offset)
                    centers.append(event_center)

                interactions_onsets[(fly_idx, ball_idx)] = onsets
                interactions_offsets[(fly_idx, ball_idx)] = offsets
                interactions_centers[(fly_idx, ball_idx)] = centers

        return interactions_onsets, interactions_offsets, interactions_centers

    @property
    def standardized_interactions(self):
        """Standardized interaction events based on frames before and after onset."""
        if not hasattr(self, "_std_interactions") or (
            hasattr(self, "_prev_cutoff") and self._prev_cutoff != self.cutoff_reference
        ):
            # Make sure interaction_events exist and contain data
            if not self.interaction_events or not any(
                events for fly_dict in self.interaction_events.values() for events in fly_dict.values()
            ):
                print(f"No interaction events found for {self.fly.metadata.name}")
                self._std_interactions = {}
            else:
                self._std_interactions = self._calculate_standardized_interactions()
            self._prev_cutoff = self.cutoff_reference
        return self._std_interactions

    def _calculate_standardized_interactions(self):
        standardized = {}

        # Check if onsets exist
        if not self.interactions_onsets:
            print(f"No interaction onsets found for {self.fly.metadata.name}")
            return {}

        for (fly_idx, ball_idx), onsets in self.interactions_onsets.items():
            events = []
            for onset in onsets:
                if onset is None:
                    continue
                if self.flytrack is None or self.flytrack.objects is None:
                    warnings.warn("Flytrack or its objects are not initialized.")
                    continue

                start = max(0, onset - self.fly.config.frames_before_onset)
                end = min(
                    len(self.flytrack.objects[fly_idx].dataset),
                    onset + self.fly.config.frames_after_onset,
                )
                events.append((start, end))

            # Add deduplication here
            unique_events = []
            for event in events:
                if event not in unique_events:
                    unique_events.append(event)

            standardized[(fly_idx, ball_idx)] = unique_events

        return standardized

    @property
    def random_events(self):
        """Random chunks of time with the same lengths as interaction events."""
        if not hasattr(self, "_random_events"):
            self._random_events = self._generate_random_events()
        return self._random_events

    def _generate_random_events(self):
        """Generate random events of the same lengths as interaction events."""
        if not self.interaction_events:
            warnings.warn(f"No interaction events found for {self.fly.metadata.name}. Cannot generate random events.")
            return {}

        random_events = {}

        # Determine the valid frame range based on the filtered dataset
        if self.flytrack:
            fly_data = self.flytrack.objects[0].dataset
            valid_start = fly_data.index[0]  # First valid frame
            valid_end = fly_data.index[-1]  # Last valid frame
            total_frames = valid_end - valid_start + 1
        else:
            total_frames = 0

        for fly_idx, ball_events in self.interaction_events.items():
            random_events[fly_idx] = {}
            for ball_idx, events in ball_events.items():
                random_events[fly_idx][ball_idx] = []
                for event in events:
                    event_duration = event[1] - event[0]
                    max_start = total_frames - event_duration

                    if max_start <= 0:
                        warnings.warn(
                            f"Event duration exceeds available frames for fly_idx={fly_idx}, ball_idx={ball_idx}."
                        )
                        continue

                    # Attempt to generate a random event with a maximum number of retries
                    max_retries = 1000
                    for _ in range(max_retries):
                        random_start = np.random.randint(valid_start, valid_start + max_start)
                        random_end = random_start + event_duration

                        # Ensure no overlap with existing interaction events
                        if not any(
                            random_start <= existing_event[1] and random_end >= existing_event[0]
                            for existing_event in events
                        ):
                            random_events[fly_idx][ball_idx].append((random_start, random_end))
                            break
                    else:
                        # If no valid random event could be generated, log a warning and skip
                        warnings.warn(
                            f"Could not generate a random event for fly_idx={fly_idx}, ball_idx={ball_idx} "
                            f"after {max_retries} attempts. Skipping."
                        )

        return random_events

    @property
    def random_events_onsets(self):
        """Standardized onsets for random events."""
        if not hasattr(self, "_random_events_onsets"):
            self._random_events_onsets = self._calculate_random_events_boundaries()[0]
        return self._random_events_onsets

    @property
    def random_events_offsets(self):
        """Standardized offsets for random events."""
        if not hasattr(self, "_random_events_offsets"):
            self._random_events_offsets = self._calculate_random_events_boundaries()[1]
        return self._random_events_offsets

    def _calculate_random_events_boundaries(self):
        """Calculate standardized onsets, offsets, and centers for random events."""
        if self.flytrack is None or self.balltrack is None:
            print(f"Skipping random events for {self.fly.metadata.name} due to missing tracking data.")
            return {}, {}, {}

        random_onsets = {}
        random_offsets = {}
        random_centers = {}

        for fly_idx in range(len(self.flytrack.objects)):
            for ball_idx in range(len(self.balltrack.objects)):
                if (
                    self.random_events is None
                    or fly_idx not in self.random_events
                    or ball_idx not in self.random_events[fly_idx]
                ):
                    warnings.warn(f"Random events missing for fly_idx={fly_idx}, ball_idx={ball_idx}.")
                    continue

                random_events = self.random_events[fly_idx][ball_idx]

                onsets = []
                offsets = []
                centers = []
                for event in random_events:
                    # Calculate the center of the event
                    event_center = (event[0] + event[1]) // 2

                    # Calculate standardized onset and offset without clamping
                    onset = event_center - self.fly.config.frames_before_onset
                    offset = event_center + self.fly.config.frames_after_onset

                    onsets.append(onset)
                    offsets.append(offset)
                    centers.append(event_center)

                random_onsets[(fly_idx, ball_idx)] = onsets
                random_offsets[(fly_idx, ball_idx)] = offsets
                random_centers[(fly_idx, ball_idx)] = centers

        return random_onsets, random_offsets, random_centers

    @property
    def standardized_random_events(self):
        """Standardized random events based on frames before and after onset."""
        if not hasattr(self, "_std_random_events"):
            if not self.random_events or not any(
                events for fly_dict in self.random_events.values() for events in fly_dict.values()
            ):
                print(f"No random events found for {self.fly.metadata.name}")
                self._std_random_events = {}
            else:
                self._std_random_events = self._calculate_standardized_random_events()
        return self._std_random_events

    def _calculate_standardized_random_events(self):
        standardized = {}

        # Check if onsets exist
        if not self.random_events_onsets:
            print(f"No random event onsets found for {self.fly.metadata.name}")
            return {}

        for (fly_idx, ball_idx), onsets in self.random_events_onsets.items():
            events = []
            for onset in onsets:
                if onset is None:
                    continue
                if self.flytrack is None or self.flytrack.objects is None:
                    warnings.warn("Flytrack or its objects are not initialized.")
                    continue

                start = max(0, onset - self.fly.config.frames_before_onset)
                end = min(
                    len(self.flytrack.objects[fly_idx].dataset),
                    onset + self.fly.config.frames_after_onset,
                )
                events.append((start, end))

            # Add deduplication here
            unique_events = []
            for event in events:
                if event not in unique_events:
                    unique_events.append(event)

            standardized[(fly_idx, ball_idx)] = unique_events

        return standardized

    def detect_trials(self):
        """
        Detect trials for learning experiments.
        """
        if self.fly.config.experiment_type != "Learning":
            return None

        # This will call the LearningMetrics class methods
        return self.fly.learning_metrics.trials_data

    def get_trial_events(self, trial_number):
        """
        Get interaction events for a specific trial.

        Args:
            trial_number (int): The trial number.

        Returns:
            dict: A dictionary of events for the specified trial.
        """
        if self.fly.config.experiment_type != "Learning" or not hasattr(self.fly, "learning_metrics"):
            return {}

        trial_interactions = self.fly.learning_metrics.metrics.get("trial_interactions", {})
        return trial_interactions.get(trial_number, [])

    def get_trial_standardized_interactions(self, trial_number):
        """
        Get standardized interaction events for a specific trial.

        Args:
            trial_number (int): The trial number.

        Returns:
            dict: A dictionary of standardized events for the specified trial.
        """
        if self.fly.config.experiment_type != "Learning" or not hasattr(self.fly, "learning_metrics"):
            return {}

        # Filter standardized interactions by trial
        trial_events = self.get_trial_events(trial_number)

        # Create a set of event bounds for quick lookup
        event_bounds = {(event[0], event[1]) for event in trial_events}

        # Filter standardized interactions
        trial_std_interactions = {}

        for (fly_idx, ball_idx), events in self.standardized_interactions.items():
            matching_events = [event for event in events if (event[0], event[1]) in event_bounds]

            if matching_events:
                trial_std_interactions[(fly_idx, ball_idx)] = matching_events

        return trial_std_interactions

    def check_data_quality(self):
        """Check if the fly is dead or in poor condition.

        This method loads the smoothed fly tracking data and checks if the fly moved more than 30 pixels in the y or x direction. If it did, it means the fly is alive and in good condition.

        Returns:
            bool: True if the fly is dead or in poor condition, False otherwise.
        """
        # Ensure that flytrack is not None
        if self.flytrack is None:
            print(f"{self.fly.metadata.name} has no tracking data.")
            return False

        # Use the flytrack dataset
        fly_data = self.flytrack.objects[0].dataset

        if "y_thorax" not in fly_data.columns or "x_thorax" not in fly_data.columns:
            warnings.warn(
                f"'y_thorax' or 'x_thorax' missing for {self.fly.metadata.name}. Skipping data quality check."
            )
            return False

        # Check if any of the smoothed fly x and y coordinates are more than 30 pixels away from their initial position
        moved_y = np.any(abs(fly_data["y_thorax"] - fly_data["y_thorax"].iloc[0]) > self.fly.config.dead_threshold)
        moved_x = np.any(abs(fly_data["x_thorax"] - fly_data["x_thorax"].iloc[0]) > self.fly.config.dead_threshold)

        if not moved_y and not moved_x:
            print(f"{self.fly.metadata.name} did not move significantly.")
            return False

        # Check if the interaction events dictionary is empty
        if not self.interaction_events or not any(self.interaction_events.values()):
            print(f"{self.fly.metadata.name} did not interact with the ball.")

            if not self.keep_idle:
                return False

        return True

    def check_dying(self):
        """
        Check if in the fly tracking data, there is any time where the fly doesn't move more than 30 pixels for 15 minutes.
        """
        # Ensure flytrack and its objects are not None
        if self.flytrack is None or self.flytrack.objects is None:
            warnings.warn("Flytrack or its objects are not initialized.")
            return False

        # Ensure there is at least one object in flytrack
        if len(self.flytrack.objects) == 0:
            warnings.warn("Flytrack contains no objects.")
            return False

        fly_data = self.flytrack.objects[0].dataset

        if "y_thorax" not in fly_data.columns or "x_thorax" not in fly_data.columns:
            warnings.warn(f"'y_thorax' or 'x_thorax' missing for {self.fly.metadata.name}. Skipping dying check.")
            return False

        # Get the velocity of the fly
        velocity = np.sqrt(
            np.diff(fly_data["x_thorax"], prepend=np.nan) ** 2 + np.diff(fly_data["y_thorax"], prepend=np.nan) ** 2
        )

        # Ensure the length of the velocity array matches the length of the DataFrame index
        if len(velocity) != len(fly_data):
            velocity = np.append(velocity, np.nan)

        fly_data["velocity"] = velocity

        # Get the time points where the fly's velocity is less than 2 px/s
        low_velocity = fly_data[fly_data["velocity"] < 2]

        # Get consecutive time points where the fly's velocity is less than 2 px/s
        consecutive_points = np.split(low_velocity.index, np.where(np.diff(low_velocity.index) != 1)[0] + 1)

        # Get the duration of each consecutive period
        durations = [len(group) for group in consecutive_points]

        # Check if there is any consecutive period of 15 minutes where the fly's velocity is less than 2 px/s
        for group, events in zip(consecutive_points, durations):
            if events > 15 * 60 * self.fly.experiment.fps:
                # Get the corresponding time
                time = fly_data.loc[group[0], "time"]
                # print(f"Warning: {self.fly.metadata.name} is inactive at {time}")
                return True

        return False

    def get_initial_position(self):
        """
        Get the initial x and y positions of the fly. First, try to use the fly tracking data.
        If not available, use the skeleton data.

        Returns:
            tuple: The median initial x and y positions of the fly over the first 10 frames.
        """
        # Check if fly tracking data is available
        if hasattr(self, "flytrack") and self.flytrack is not None:
            fly_data = self.flytrack.objects[0].dataset
            if "y_thorax" in fly_data.columns and "x_thorax" in fly_data.columns:
                # Compute the median over the first 10 frames
                initial_x = fly_data["x_thorax"].iloc[:10].median()
                initial_y = fly_data["y_thorax"].iloc[:10].median()
                return initial_x, initial_y
            else:
                warnings.warn(f"'y_thorax' or 'x_thorax' missing for {self.fly.metadata.name}. Skipping.")
                return None, None

        # Fallback to skeleton data if fly tracking data is not available
        if self.fly_skeleton is not None:
            if "y_thorax" in self.fly_skeleton.columns and "x_thorax" in self.fly_skeleton.columns:
                # Compute the median over the first 10 frames
                initial_x = self.fly_skeleton["x_thorax"].iloc[:10].median()
                initial_y = self.fly_skeleton["y_thorax"].iloc[:10].median()
                return initial_x, initial_y
            else:
                warnings.warn(f"'y_thorax' or 'x_thorax' missing for {self.fly.metadata.name}. Skipping.")
                return None, None

        raise ValueError(f"No valid position data found for {self.fly.metadata.name}.")

    def check_yball_variation(self, event, threshold=None):
        """
        Check if the yball value varies more than a given threshold during an event.

        Args:
            event (list): A list containing the start and end indices of the event.
            threshold (float): The maximum allowed variation in yball value.

        Returns:
            tuple: A boolean indicating if the variation exceeds the threshold, and the actual variation.
        """
        if self.balltrack is None or self.balltrack.objects is None:
            warnings.warn("Balltrack or its objects are not initialized.")
            return False, 0

        # Set threshold using config

        if threshold is None:
            threshold = self.fly.config.significant_threshold

        # Get the ball data
        ball_data = self.balltrack.objects[0].dataset

        # Ensure the event indices are within the bounds of the dataset
        start_idx, end_idx = event
        if start_idx < 0 or end_idx > len(ball_data):
            warnings.warn(f"Event indices {event} are out of bounds.")
            return False, 0

        # Extract the yball values for the event
        yball_values = ball_data["y_centre"].iloc[start_idx:end_idx]

        # Calculate the variation
        variation = yball_values.max() - yball_values.min()

        # Check if the variation exceeds the threshold
        exceeds_threshold = variation > threshold

        return exceeds_threshold, variation

    def get_skeleton(self):
        """
        Extracts the coordinates of the fly's skeleton from the full body tracking data.

        Returns:
            DataFrame: A DataFrame containing the coordinates of the fly's skeleton.
        """

        if self.skeletontrack is None:
            warnings.warn(f"No skeleton tracking file found for {self.fly.metadata.name}.")
            return None

        # Get the first track
        full_body_data = self.skeletontrack.objects[0].dataset

        return full_body_data

    def get_exit_time(self):
        """
        Get the exit time, which is the first time at which the fly x position has been 100 px away from the initial fly x position.

        Returns:
            float: The exit time, or None if the fly did not move 100 px away from the initial position.
        """

        if self.flytrack is None:
            return None

        fly_data = self.flytrack.objects[0].dataset

        # Get the initial x position of the fly
        initial_x = self.start_x

        if "x_thorax" not in fly_data.columns:
            warnings.warn(f"'x_thorax' missing for {self.fly.metadata.name}. Skipping exit time calculation.")
            return None

        # Get the x position of the fly
        x = fly_data["x_thorax"]

        # Find the first time at which the fly x position has been 100 px away from the initial fly x position
        if initial_x is None:
            warnings.warn(f"Initial x position is None for {self.fly.metadata.name}. Skipping exit time calculation.")
            return None

        exit_condition = x > initial_x + 100
        if not exit_condition.any():
            return None

        exit_time = x[exit_condition].index[0] / self.fly.experiment.fps

        return exit_time

    def compute_adjusted_time(self):
        """
        Compute adjusted time based on the fly's exit time if any, otherwise return None.
        """
        if self.exit_time is not None:
            # Ensure flytrack and its objects are not None
            if self.flytrack is None or self.flytrack.objects is None:
                warnings.warn("Flytrack or its objects are not initialized.")
                return None

            # Ensure there is at least one object in flytrack
            if len(self.flytrack.objects) == 0:
                warnings.warn("Flytrack contains no objects.")
                return None

            flydata = self.flytrack.objects[0].dataset

            # Compute adjusted time
            flydata["adjusted_time"] = flydata["time"] - self.exit_time

            return flydata["adjusted_time"]
        else:
            return None

    def filter_tracking_data(self, time_range):
        """Filter the tracking data based on the time range."""
        if self.flytrack is not None:
            self.flytrack.filter_data(time_range)
        if self.balltrack is not None:
            self.balltrack.filter_data(time_range)
        if self.skeletontrack is not None:
            self.skeletontrack.filter_data(time_range)

    def log_missing_fly(self):
        """Log the metadata of flies that do not pass the validity test."""
        log_path = self.fly.config.log_path

        # Get the metadata from the fly.FlyMetadata and write it to a log file

        # Get the fly's metadata
        name = self.fly.metadata.name
        metadata = self.fly.metadata.get_arena_metadata()

        # Write the metadata to a log file
        with open(f"{log_path}/missing_flies.log", "a") as f:
            f.write(f"{name}: {metadata}\n")


# TODO : Test the valid_data function in conditions where I know the fly is dead or the arena is empty or not to check success
