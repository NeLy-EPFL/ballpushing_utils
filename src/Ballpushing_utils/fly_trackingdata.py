from functools import lru_cache
from utils_behavior import Sleap_utils
from Ballpushing_utils import utilities
from utils_behavior import Processing
from Ballpushing_utils.ballpushing_metrics import BallPushingMetrics
import numpy as np
import warnings


class FlyTrackingData:
    def __init__(self, fly, time_range=None, log_missing=False, keep_idle=False):
        self.fly = fly
        self.flytrack = None
        self.balltrack = None
        self.raw_balltrack = None  # Raw (unsmoothed) balltrack
        self.skeletontrack = None
        self.valid_data = True
        self.log_missing = log_missing
        self.keep_idle = keep_idle
        self.cutoff_reference = None  # New cutoff reference storage

        try:
            # Load tracking files
            self.balltrack = self.load_tracking_file("*ball*.h5", "ball")
            self.raw_balltrack = self.load_tracking_file("*ball*.h5", "ball")
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

            self.start_x, self.start_y = self.get_initial_position()

            # Assign ball identities based on position relative to fly
            self.assign_ball_identities()

            self.calculate_relative_positions()

            time_range_start = self.fly.config.time_range[0] if self.fly.config.time_range else None
            exit_time = self.get_chamber_exit_time()

            # Handle case where exit time is before time range start
            if exit_time is not None and time_range_start is not None and exit_time < time_range_start:
                self.chamber_exit_time = 0
            else:
                self.chamber_exit_time = exit_time

            # Apply initial time range filter if specified in config
            if self.fly.config.time_range:
                # if self.fly.config.debugging:
                # print(f"Applying time range filter for {self.fly.metadata.name}: {self.fly.config.time_range}")
                self.filter_tracking_data(self.fly.config.time_range)

            # Compute duration as the difference between last and first time
            # Use training ball if available, otherwise use first ball
            ball_idx = self.training_ball_idx if self.training_ball_idx is not None else 0
            times = self.balltrack.objects[ball_idx].dataset["time"]
            if not times.empty:
                self.duration = times.iloc[-1] - times.iloc[0]

            if self.fly.config.debugging:
                print(f"Duration for {self.fly.metadata.name}: {self.duration}")
                print(f"Initial position for {self.fly.metadata.name}: ({self.start_x}, {self.start_y})")

            self.fly_skeleton = self.get_skeleton()
            self.f1_exit_time = self.get_f1_exit_time()
            self.adjusted_time = self.compute_adjusted_time()

            # Check for F1 premature exit (flies that exit before 55 minutes should be discarded)
            if hasattr(self.fly.config, "experiment_type") and self.fly.config.experiment_type == "F1":
                if self.check_f1_premature_exit():
                    print(f"F1 premature exit detected for {self.fly.metadata.name}: exited before 55 minutes")
                    self.valid_data = False
                    return

            # Determine success cutoff reference
            if self.fly.config.success_cutoff:
                self._determine_success_cutoff()

    def load_tracking_file(
        self,
        pattern,
        object_type,
        smoothing=None,
    ):
        """Load a tracking file for the fly with arena/corridor validation."""

        if smoothing is None:
            smoothing = self.fly.config.tracks_smoothing

        try:
            tracking_file = list(self.fly.directory.glob(pattern))[0]

            # Load the tracking data
            sleap_tracks = Sleap_utils.Sleap_Tracks(
                tracking_file,
                object_type=object_type,
                smoothed_tracks=smoothing,
                debug=False,
            )

            # Validate that the H5 file's video path matches the current fly's arena/corridor
            if not self._validate_tracking_file_consistency(tracking_file, sleap_tracks):
                print(
                    f"⚠️  WARNING: Tracking file {tracking_file.name} has mismatched arena/corridor for {self.fly.metadata.name}"
                )
                print(f"   This file may be corrupted or misplaced. Consider moving it to /invalid directory.")
                print(f"   Skipping this tracking file to prevent corrupted analysis.")
                return None  # Skip mismatched files

            return sleap_tracks

        except IndexError:
            return None

    def _validate_tracking_file_consistency(self, tracking_file, sleap_tracks):
        """
        Validate that the tracking file's video path matches the current fly's arena/corridor.

        Parameters
        ----------
        tracking_file : Path
            Path to the H5 tracking file
        sleap_tracks : Sleap_Tracks
            Loaded SLEAP tracking data

        Returns
        -------
        bool
            True if validation passes, False if there's a mismatch
        """
        try:
            # Get the video path from the H5 file
            video_path = sleap_tracks.video

            # Extract arena/corridor from H5 file path
            h5_arena_corridor = self._parse_arena_corridor_from_path(str(tracking_file))

            # Extract arena/corridor from video path
            video_arena_corridor = self._parse_arena_corridor_from_path(str(video_path))

            # Get expected arena/corridor from current fly
            expected_arena = self.fly.metadata.arena  # e.g., "arena2"
            expected_corridor = self.fly.metadata.corridor  # e.g., "corridor5" or "Left"/"Right" for F1

            # Parse expected values to integers, handling different experiment types
            expected_arena_num = None
            expected_corridor_num = None

            # Handle arena parsing (should be consistent across experiment types)
            if expected_arena and isinstance(expected_arena, str) and expected_arena.startswith("arena"):
                try:
                    expected_arena_num = int(expected_arena.replace("arena", ""))
                except (ValueError, IndexError):
                    expected_arena_num = None

            # Handle corridor parsing - depends on experiment type
            if expected_corridor and isinstance(expected_corridor, str):
                # Check if this is an F1 experiment
                is_f1_experiment = (
                    hasattr(self.fly.config, "experiment_type") and self.fly.config.experiment_type == "F1"
                )

                if is_f1_experiment and expected_corridor.lower() in ["left", "right"]:
                    # For F1 experiments, map Left/Right to corridor numbers for validation
                    expected_corridor_num = 1 if expected_corridor.lower() == "left" else 2
                elif expected_corridor.startswith("corridor"):
                    # For regular experiments with corridor1, corridor2, etc.
                    try:
                        expected_corridor_num = int(expected_corridor.replace("corridor", ""))
                    except (ValueError, IndexError):
                        expected_corridor_num = None
                else:
                    # Unknown corridor format, skip validation
                    expected_corridor_num = None

            expected_arena_corridor = (expected_arena_num, expected_corridor_num)

            # Check if video path matches expected arena/corridor
            if video_arena_corridor and expected_arena_corridor:
                if video_arena_corridor != expected_arena_corridor:
                    print(f"   H5 file: {tracking_file}")
                    print(f"   Expected arena/corridor: {expected_arena_corridor}")
                    print(f"   Video points to: {video_arena_corridor}")
                    return False

            return True

        except Exception as e:
            print(f"   Error during validation: {e}")
            return True  # Continue processing if validation fails

    def _parse_arena_corridor_from_path(self, path_str):
        """
        Extract arena and corridor numbers from a path string.
        Handles both regular experiments (arena1/corridor2) and F1 experiments (arena1/Left, arena1/Right).

        Parameters
        ----------
        path_str : str
            Path string to parse

        Returns
        -------
        tuple or None
            Tuple of (arena_number, corridor_number) or None if not found
        """
        import re

        # Look for arena pattern
        arena_match = re.search(r"arena(\d+)", path_str, re.IGNORECASE)

        if not arena_match:
            return None

        arena_num = int(arena_match.group(1))

        # Look for corridor pattern - try both regular and F1 experiment formats
        corridor_match = re.search(r"corridor(\d+)", path_str, re.IGNORECASE)

        if corridor_match:
            # Regular experiment format: corridor1, corridor2, etc.
            corridor_num = int(corridor_match.group(1))
        else:
            # F1 experiment format: Left/Right
            if re.search(r"left", path_str, re.IGNORECASE):
                corridor_num = 1  # Map Left to corridor 1
            elif re.search(r"right", path_str, re.IGNORECASE):
                corridor_num = 2  # Map Right to corridor 2
            else:
                return None  # No recognizable corridor pattern found

        return (arena_num, corridor_num)

    def get_chamber_exit_time(self):
        """
        Compute the time at which the fly left the chamber.
        Since we have only one fly in one chamber, this returns a single value.

        Returns:
            float or None: The exit time in seconds, or None if the fly never left the chamber.
        """
        if self.flytrack is None or self.flytrack.objects is None or len(self.flytrack.objects) == 0:
            if self.fly.config.debugging:
                print(f"No flytrack data available for chamber exit calculation.")
            return None

        fly_data = self.flytrack.objects[0].dataset

        # Calculate the distance from the fly start position for each frame
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"], fly_data["y_thorax"], self.start_x, self.start_y
        )

        # Debugging: Print distances
        if self.fly.config.debugging:
            print(f"Distances for chamber exit calculation: {distances}")

        # Determine the frames where the fly is within a certain radius of the start position
        in_chamber = distances <= self.fly.config.chamber_radius

        # Debugging: Print in_chamber array
        if self.fly.config.debugging:
            print(f"In-chamber status: {in_chamber}")

        # Check if the fly ever leaves the chamber
        if not np.any(~in_chamber):  # If all values in `in_chamber` are True
            if self.fly.config.debugging:
                print(f"Fly never left the chamber.")
            return None

        # Get the first frame where the fly is outside the chamber
        exit_frame = np.argmax(~in_chamber)  # Find the first `False` in `in_chamber`

        # Debugging: Print exit_frame
        if self.fly.config.debugging:
            print(f"Exit frame: {exit_frame}")

        exit_time = exit_frame / self.fly.experiment.fps

        return exit_time

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

    def assign_ball_identities(self):
        """
        Assign ball identities based on available SLEAP track names.

        Behavior:
        1. Single ball (regular experiments): No identity system needed, use ball_idx as before
        2. Multiple balls + track names: Use track names as identities
        3. Multiple balls + no track names: Issue warning, proceed without identities

        This method updates the balltrack objects to include ball identity information
        and creates mapping dictionaries for easy access when applicable.
        """
        if self.balltrack is None:
            print(f"Cannot assign ball identities for {self.fly.metadata.name}: missing balltrack")
            return

        # Initialize ball identity mappings
        self.training_ball_idx = None
        self.test_ball_idx = None
        self.ball_identities = {}  # Maps ball_idx -> identity
        self.identity_to_idx = {}  # Maps identity -> ball_idx

        num_balls = len(self.balltrack.objects)

        # Check for track names regardless of number of balls
        # This is important for F1 control experiments which have single test_ball
        has_track_names = (
            hasattr(self.balltrack, "track_names")
            and self.balltrack.track_names
            and len(self.balltrack.track_names) > 0
        )

        if not has_track_names:
            # No track names - handle based on number of balls
            if num_balls == 1:
                # Single ball without track names - regular experiment
                self.training_ball_idx = 0
                if self.fly.config.debugging:
                    print(
                        f"Single ball experiment for {self.fly.metadata.name}: using ball_idx=0 as training (no identity system)"
                    )
                return
            else:
                # Multiple balls without track names - issue warning
                print(
                    f"⚠️  WARNING: Multiple balls ({num_balls}) detected for {self.fly.metadata.name} but no SLEAP track names found!"
                )
                print(
                    f"   For F1 experiments, please ensure SLEAP tracking includes track names (e.g., 'training', 'test')"
                )
                print(f"   Proceeding without ball identity assignment - using ball indices for analysis")

                # Set first ball as training for basic compatibility but warn about it
                self.training_ball_idx = 0
                if self.fly.config.debugging:
                    print(f"   Defaulting to ball_idx=0 as training ball for basic compatibility")
                return

        # Has track names - use track names as identities (works for single or multiple balls)
        if self.fly.config.debugging:
            print(f"Using SLEAP track names for {self.fly.metadata.name}: {self.balltrack.track_names}")

        for ball_idx in range(num_balls):
            if ball_idx < len(self.balltrack.track_names):
                track_name = self.balltrack.track_names[ball_idx]
                track_name_lower = track_name.lower()

                # Map common track names to standard identities
                if track_name_lower in ["training", "train", "training_ball"]:
                    identity = "training"
                    self.training_ball_idx = ball_idx
                    self.identity_to_idx["training"] = ball_idx
                elif track_name_lower in ["test", "testing", "test_ball"]:
                    identity = "test"
                    self.test_ball_idx = ball_idx
                    self.identity_to_idx["test"] = ball_idx
                else:
                    # Use the track name directly as identity
                    identity = track_name_lower

                    # For F1 experiments with generic track names, assign training/test roles
                    if num_balls == 2:
                        if ball_idx == 0 and self.training_ball_idx is None:
                            # First ball becomes training
                            self.training_ball_idx = ball_idx
                            self.identity_to_idx["training"] = ball_idx
                        elif ball_idx == 1 and self.test_ball_idx is None:
                            # Second ball becomes test
                            self.test_ball_idx = ball_idx
                            self.identity_to_idx["test"] = ball_idx
                    else:
                        # For other cases, use the original logic
                        if self.training_ball_idx is None and ball_idx == 0:
                            self.training_ball_idx = ball_idx
                            self.identity_to_idx["training"] = ball_idx

                self.ball_identities[ball_idx] = identity

                # Add identity to ball dataset
                ball_data = self.balltrack.objects[ball_idx].dataset
                ball_data["ball_identity"] = identity
                ball_data["track_name"] = track_name

                if self.fly.config.debugging:
                    print(f"  Ball {ball_idx}: '{track_name}' → identity '{identity}'")
            else:
                # More balls than track names - use generic naming with warning
                identity = f"unnamed_ball_{ball_idx}"
                self.ball_identities[ball_idx] = identity

                ball_data = self.balltrack.objects[ball_idx].dataset
                ball_data["ball_identity"] = identity

                print(f"⚠️  WARNING: Ball {ball_idx} has no track name, assigned generic identity '{identity}'")

        # Ensure we have a training ball for backward compatibility
        if self.training_ball_idx is None and num_balls > 0:
            # Assign first ball as training if no explicit training ball found
            self.training_ball_idx = 0
            if "training" not in self.identity_to_idx:
                self.identity_to_idx["training"] = 0
            if self.fly.config.debugging:
                print(f"  No explicit training ball found, using ball 0 as training for compatibility")

        if self.fly.config.debugging:
            print(f"Ball identity assignment summary for {self.fly.metadata.name}:")
            print(f"  Training ball: {self.training_ball_idx}")
            print(f"  Test ball: {self.test_ball_idx}")
            print(f"  All identities: {self.ball_identities}")
            print(f"  Identity→Index mappings: {self.identity_to_idx}")

    def get_ball_by_identity(self, identity):
        """
        Get ball index by identity ('training' or 'test').

        Args:
            identity (str): 'training' or 'test'

        Returns:
            int or None: Ball index if found, None otherwise
        """
        return self.identity_to_idx.get(identity, None)

    def get_training_ball_data(self):
        """
        Get the dataset for the training ball.

        Returns:
            DataFrame or None: Training ball dataset if available
        """
        if self.training_ball_idx is not None and self.balltrack is not None:
            return self.balltrack.objects[self.training_ball_idx].dataset
        return None

    def get_test_ball_data(self):
        """
        Get the dataset for the test ball.

        Returns:
            DataFrame or None: Test ball dataset if available
        """
        if self.test_ball_idx is not None and self.balltrack is not None:
            return self.balltrack.objects[self.test_ball_idx].dataset
        return None

    def has_training_ball(self):
        """Check if a training ball is available."""
        return self.training_ball_idx is not None

    def has_test_ball(self):
        """Check if a test ball is available."""
        return self.test_ball_idx is not None

    def get_ball_identity(self, ball_idx):
        """
        Get the identity of a ball by its index.

        Args:
            ball_idx (int): Ball index

        Returns:
            str or None: Ball identity ('training', 'test', etc.) or None if not found
        """
        return self.ball_identities.get(ball_idx, None)

    def get_all_ball_identities(self):
        """
        Get all available ball identities.

        Returns:
            list: List of all ball identities (e.g., ['training', 'test'] or custom track names)
        """
        return list(self.ball_identities.values())

    def get_balls_by_identity_pattern(self, pattern):
        """
        Get ball indices that match a pattern in their identity.

        Args:
            pattern (str): Pattern to match (e.g., 'training', 'test', 'ball')

        Returns:
            list: List of ball indices whose identities contain the pattern
        """
        matching_balls = []
        for ball_idx, identity in self.ball_identities.items():
            if pattern.lower() in identity.lower():
                matching_balls.append(ball_idx)
        return matching_balls

    def has_identity(self, identity):
        """
        Check if a specific identity exists.

        Args:
            identity (str): Identity to check for

        Returns:
            bool: True if the identity exists
        """
        return identity in self.identity_to_idx or identity in self.ball_identities.values()

    def get_interactions_with_training_ball(self, fly_idx=0):
        """
        Get interaction events with the training ball specifically.

        Args:
            fly_idx (int): Fly index (default 0)

        Returns:
            list: List of interaction events with training ball
        """
        if (
            self.training_ball_idx is not None
            and hasattr(self, "_interaction_events")
            and self.interaction_events is not None
        ):
            if fly_idx in self.interaction_events and self.training_ball_idx in self.interaction_events[fly_idx]:
                return self.interaction_events[fly_idx][self.training_ball_idx]
        return []

    def get_interactions_with_test_ball(self, fly_idx=0):
        """
        Get interaction events with the test ball specifically.

        Args:
            fly_idx (int): Fly index (default 0)

        Returns:
            list: List of interaction events with test ball
        """
        if (
            self.test_ball_idx is not None
            and hasattr(self, "_interaction_events")
            and self.interaction_events is not None
        ):
            if fly_idx in self.interaction_events and self.test_ball_idx in self.interaction_events[fly_idx]:
                return self.interaction_events[fly_idx][self.test_ball_idx]
        return []

    def get_interactions_by_identity(self, identity="training", fly_idx=0):
        """
        Get interaction events with a ball by its identity.

        Args:
            identity (str): Ball identity ('training' or 'test')
            fly_idx (int): Fly index (default 0)

        Returns:
            list: List of interaction events with specified ball identity
        """
        ball_idx = self.get_ball_by_identity(identity)
        if ball_idx is not None and hasattr(self, "_interaction_events") and self.interaction_events is not None:
            if fly_idx in self.interaction_events and ball_idx in self.interaction_events[fly_idx]:
                return self.interaction_events[fly_idx][ball_idx]
        return []

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

        # Apply initial time range filter if specified
        # if self.fly.config.time_range is not None:
        #     time_range = self.fly.config.time_range
        #     self.filter_tracking_data(time_range)

        ballpushing_metrics = BallPushingMetrics(self, compute_metrics_on_init=False)

        # Use test ball for final event detection (prioritize test ball over training ball)
        # For control experiments with only test ball, this will work correctly
        # For F1 experiments, we want the test ball's final event for success cutoff
        if self.test_ball_idx is not None:
            ball_idx = self.test_ball_idx
        elif self.training_ball_idx is not None:
            ball_idx = self.training_ball_idx
        else:
            ball_idx = 0  # Fallback to first ball

        final_event = ballpushing_metrics.get_final_event(0, ball_idx)
        if self.fly.config.debugging:
            ball_type = (
                "test"
                if ball_idx == self.test_ball_idx
                else ("training" if ball_idx == self.training_ball_idx else f"ball_{ball_idx}")
            )
            print(f"Final event for {self.fly.metadata.name} using {ball_type} ball (idx={ball_idx}): {final_event}")
        # Check if the final event is valid
        if final_event is not None and final_event[2] is not None:
            if self.fly.config.debugging:
                print(f"Final event detected for {self.fly.metadata.name}: {final_event}")
            # Set the cutoff reference to the end of the final event + safety frames
            self.cutoff_reference = final_event[2] + 15
        else:
            print(f"No valid final event detected for {self.fly.metadata.name}.")
            # If no final event, set cutoff reference to the end of the time range or dataset
            if self.fly.config.time_range and self.fly.config.time_range[1] is not None:
                self.cutoff_reference = self.fly.config.time_range[1]
            else:
                self.cutoff_reference = None

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
            if self.fly.config.time_range is not None:
                time_range = (
                    (self.fly.config.time_range[0], self.cutoff_reference)
                    if self.cutoff_reference is not None
                    else (self.fly.config.time_range[0], None)
                )
            else:
                time_range = (None, self.cutoff_reference) if self.cutoff_reference is not None else None

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
        """
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

        # if time_range:
        #     # Convert frame indices to integers for iloc
        #     start = int(time_range[0] * self.fly.experiment.fps) if time_range[0] is not None else 0
        #     end = int(time_range[1] * self.fly.experiment.fps) if time_range[1] is not None else None
        #     fly_data = fly_data.iloc[start:end]
        #     ball_data = ball_data.iloc[start:end]

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
        # Check if random event generation is disabled in the config
        if not self.fly.config.generate_random:
            warnings.warn(f"Random event generation is disabled for {self.fly.metadata.name}.")
            return {}

        if not hasattr(self, "_random_events"):
            self._random_events = self._generate_random_events()
        return self._random_events

    def _generate_random_events(self):
        """Generate random events of the same lengths as interaction events."""
        # Check if random event generation is disabled in the config
        if not self.fly.config.generate_random:
            warnings.warn(f"Random event generation is disabled for {self.fly.metadata.name}.")
            return {}

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

    def check_f1_premature_exit(self):
        """
        Check if the fly exits the first corridor before 55 minutes in F1 experiments.
        Uses the existing exit_time calculation which detects when the fly moves 100px
        in the x direction from its initial position.

        Returns:
            bool: True if fly exits before 55 minutes (should be discarded), False otherwise
        """
        # Only apply this check to F1 experiments
        if not (hasattr(self.fly.config, "experiment_type") and self.fly.config.experiment_type == "F1"):
            return False

        # If no F1 exit time detected, fly never left first corridor (valid)
        if self.f1_exit_time is None:
            if self.fly.config.debugging:
                print(f"F1 fly {self.fly.metadata.name} never exited first corridor - keeping")
            return False

        # Convert 55 minutes to seconds
        premature_exit_threshold = 55 * 60  # 55 minutes in seconds

        # Check if fly exited before threshold
        if self.f1_exit_time < premature_exit_threshold:
            if self.fly.config.debugging:
                print(
                    f"F1 fly {self.fly.metadata.name} exited at {self.f1_exit_time/60:.1f} minutes (before {premature_exit_threshold/60} min threshold)"
                )
            return True

        if self.fly.config.debugging:
            print(
                f"F1 fly {self.fly.metadata.name} exited at {self.f1_exit_time/60:.1f} minutes (after threshold) - keeping"
            )
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

    def check_yball_variation(self, event, threshold=None, ball_identity="training"):
        """
        Check if the yball value varies more than a given threshold during an event.

        Args:
            event (list): A list containing the start and end indices of the event.
            threshold (float): The maximum allowed variation in yball value.
            ball_identity (str): Which ball to use ('training' or 'test'). Defaults to 'training'.

        Returns:
            tuple: A boolean indicating if the variation exceeds the threshold, and the actual variation.
        """
        if self.balltrack is None or self.balltrack.objects is None:
            warnings.warn("Balltrack or its objects are not initialized.")
            return False, 0

        # Set threshold using config
        if threshold is None:
            threshold = self.fly.config.significant_threshold

        # Get the appropriate ball data
        ball_idx = self.get_ball_by_identity(ball_identity)
        if ball_idx is None:
            # Fallback to first ball if identity not found
            ball_idx = 0
            if self.fly.config.debugging:
                print(f"Ball identity '{ball_identity}' not found, using ball_idx=0")

        ball_data = self.balltrack.objects[ball_idx].dataset

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

    def get_f1_exit_time(self):
        """
        Get the F1 corridor exit time, which is the first time at which the fly x position has been 100 px away from the initial fly x position.
        This is specifically for F1 experiments to detect when the fly exits the first corridor to enter the second arena.
        Uses raw positions with sustained exit (2 seconds) requirement to reduce false positives from tracking errors.
        The 2-second persistence requirement provides robustness without the computational overhead of median smoothing.

        Returns:
            float: The F1 exit time, or None if the fly did not move 100 px away from the initial position sustainedly.
        """

        if self.flytrack is None:
            return None

        fly_data = self.flytrack.objects[0].dataset

        # Get the initial x position of the fly (already uses median of first 10 frames)
        initial_x = self.start_x

        if "x_thorax" not in fly_data.columns:
            warnings.warn(f"'x_thorax' missing for {self.fly.metadata.name}. Skipping exit time calculation.")
            return None

        if initial_x is None:
            warnings.warn(f"Initial x position is None for {self.fly.metadata.name}. Skipping exit time calculation.")
            return None

        # Use raw x positions (no smoothing) - the persistence check will handle noise
        x_positions = fly_data["x_thorax"]

        # Find frames where the fly x position is 100 px away from the initial position
        exit_condition = x_positions > initial_x + 100
        if not exit_condition.any():
            return None

        # Require sustained exit: fly must stay beyond threshold for at least 2 seconds (60 frames)
        persistence_frames = int(2 * self.fly.experiment.fps)  # 2 seconds in frames

        # Find consecutive sequences of frames where exit condition is True
        exit_indices = np.where(exit_condition)[0]

        if len(exit_indices) == 0:
            return None

        # Group consecutive indices
        consecutive_groups = []
        current_group = [exit_indices[0]]

        for i in range(1, len(exit_indices)):
            if exit_indices[i] == exit_indices[i - 1] + 1:
                # Consecutive frame
                current_group.append(exit_indices[i])
            else:
                # Gap found, start new group
                consecutive_groups.append(current_group)
                current_group = [exit_indices[i]]

        # Don't forget the last group
        consecutive_groups.append(current_group)

        # Find the first group that meets persistence requirement
        for group in consecutive_groups:
            if len(group) >= persistence_frames:
                # Found a sustained exit - use the first frame of this group
                exit_frame_idx = group[0]
                exit_time = fly_data.index[exit_frame_idx] / self.fly.experiment.fps

                if self.fly.config.debugging:
                    raw_x_at_exit = fly_data["x_thorax"].iloc[exit_frame_idx]
                    sustained_duration = len(group) / self.fly.experiment.fps
                    print(
                        f"Sustained exit detected for {self.fly.metadata.name}: frame={exit_frame_idx}, time={exit_time:.1f}s"
                    )
                    print(f"  Initial x: {initial_x:.1f}, Raw x: {raw_x_at_exit:.1f}")
                    print(
                        f"  Sustained for: {sustained_duration:.1f}s ({len(group)} frames, required: {persistence_frames})"
                    )

                return exit_time

        # No sustained exit found
        if self.fly.config.debugging:
            max_consecutive = max(len(group) for group in consecutive_groups) if consecutive_groups else 0
            max_duration = max_consecutive / self.fly.experiment.fps
            print(
                f"No sustained exit for {self.fly.metadata.name}: max consecutive = {max_consecutive} frames ({max_duration:.1f}s), required: {persistence_frames} frames ({persistence_frames/self.fly.experiment.fps:.1f}s)"
            )

        return None

    def compute_adjusted_time(self):
        """
        Compute adjusted time based on the fly's F1 exit time if any, otherwise return None.
        """
        if self.f1_exit_time is not None:
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
            flydata["adjusted_time"] = flydata["time"] - self.f1_exit_time

            return flydata["adjusted_time"]
        else:
            return None

    def filter_tracking_data(self, time_range):
        """Filter the tracking data based on the time range."""
        if self.flytrack is not None:
            if self.fly.config.debugging:
                print(f"Filtering flytrack data for {self.fly.metadata.name} with time range: {time_range}")
            self.flytrack.filter_data(time_range)
            if self.flytrack.objects[0].dataset.empty:
                raise ValueError(f"Flytrack data is empty after filtering for {self.fly.metadata.name}.")

        if self.balltrack is not None:
            self.balltrack.filter_data(time_range)
            if self.balltrack.objects[0].dataset.empty:
                raise ValueError(f"Balltrack data is empty after filtering for {self.fly.metadata.name}.")

        if self.skeletontrack is not None:
            self.skeletontrack.filter_data(time_range)
            if self.skeletontrack.objects[0].dataset.empty:
                raise ValueError(f"Skeletontrack data is empty after filtering for {self.fly.metadata.name}.")

        # Reset cached interaction events to force recomputation with filtered data
        if hasattr(self, "_interaction_events"):
            del self._interaction_events
        if hasattr(self, "_interactions_onsets"):
            del self._interactions_onsets
        if hasattr(self, "_interactions_offsets"):
            del self._interactions_offsets

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

    def calculate_relative_positions(self):
        """
        Add relative x and y columns to flytrack and balltrack datasets, using start_x and start_y as reference.
        The relative positions are the absolute distance from the start position for each frame.
        """
        # Fly relative positions
        if self.flytrack is not None and self.start_x is not None and self.start_y is not None:
            for obj in self.flytrack.objects:
                df = obj.dataset
                if "x_thorax" in df.columns and "y_thorax" in df.columns:
                    df["rel_x_fly"] = np.abs(df["x_thorax"] - self.start_x)
                    df["rel_y_fly"] = np.abs(df["y_thorax"] - self.start_y)
        # Ball relative positions
        if self.balltrack is not None and self.start_x is not None and self.start_y is not None:
            for obj in self.balltrack.objects:
                df = obj.dataset
                if "x_centre" in df.columns and "y_centre" in df.columns:
                    df["rel_x_ball"] = np.abs(df["x_centre"] - self.start_x)
                    df["rel_y_ball"] = np.abs(df["y_centre"] - self.start_y)

    def clear_caches(self):
        """
        Clear all cached calculations to free up memory. This should be called between processing different experiments.
        """
        # Clear all cached interaction-related attributes
        cached_attributes = [
            "_interaction_events",
            "_interactions_onsets",
            "_interactions_offsets",
            "_std_interactions",
            "_random_events",
            "_random_events_onsets",
            "_random_events_offsets",
            "_std_random_events",
        ]

        for attr in cached_attributes:
            if hasattr(self, attr):
                delattr(self, attr)

        # Force garbage collection
        import gc

        gc.collect()
