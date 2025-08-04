from dataclasses import dataclass
from utils_behavior import Utils
from typing import Optional

# Pixel size: 30 mm = 500 pixels, 4 mm = 70 pixels, 1.5 mm = 25 pixels
# Conversion factor: 500 pixels = 30 mm, so 1 mm = 16.67 pixels
PIXELS_PER_MM = 500 / 30  # 16.67 pixels per mm


@dataclass
class Config:
    """
    Configuration class for the Ball pushing experiments

    Attributes:
    interaction_threshold: tuple: The lower and upper limit values (in pixels) for the signal to be considered an event. Defaults to (0, 70).
    time_range: tuple: The time range (in seconds) to filter the tracking data. Defaults to None.
    success_cutoff: bool: Whether to filter the tracking data based on the success cutoff time range. Defaults to False. If True, the success_cutoff_time_range computed in the FlyTrackingData class will be used to filter the tracking data.
    tracks_smoothing: bool: Whether to apply smoothing to the tracking data. Defaults to True.
    dead_threshold: int: The threshold value (in pixels traveled) for the fly to be considered dead. Defaults to 30.
    adjusted_events_normalisation: int: The normalisation value for the adjusted number of events. It's an arbitrary value that multiplies all the adjusted events values to make it more easily readable. Defaults to 1000.
    significant_threshold: int: The threshold value (in pixels) for an event to be considered significant. Defaults to 5.
    major_event_threshold: int: The threshold value (in pixels) for an event to be considered an Aha moment. Defaults to 20.
    success_direction_threshold: int: The threshold value (in pixels) for an event to be considered a success direction, which is used to check whether a fly is a "pusher" or a "puller", or "both. Defaults to 25.
    final_event_threshold: int: The threshold value (in pixels) for an event to be considered a final event. Defaults to 170.
    final_event_F1_threshold: int: The threshold value (in pixels) for an event to be considered a final event in the F1 condition. Defaults to 100.
    max_event_threshold: int: The threshold value (in pixels) for an event to be considered a maximum event. Defaults to 10.

    """

    debugging: bool = False

    # General configuration attributes

    experiment_type: str = "TNT"
    time_range: Optional[tuple] = None
    success_cutoff: bool = False
    success_cutoff_method: str = "final_event"
    tracks_smoothing: bool = False
    chamber_radius: int = 50
    rolling_window: int = 10

    # Pixel to mm conversion factor (500 pixels = 30 mm)
    pixels_per_mm: float = 500 / 30  # 16.67 pixels per mm

    log_missing = True
    log_path = Utils.get_data_server() / "MD/MultiMazeRecorder"

    keep_idle = True

    # Coordinates dataset attributes

    downsampling_factor: Optional[int] = None  # Classic values used are 5 or 10.

    # Random events attributes

    generate_random: bool = False
    random_exclude_interactions: bool = True
    random_interaction_map: str = "full"  # Options: "full" or "onset"

    # Events related thresholds

    corridor_end_threshold: int = 170  # Threshold for corridor end detection in pixels
    interaction_threshold: tuple = (0, 45)  # Default was 70
    gap_between_events: int = 1  # Default was 2
    events_min_length: int = 1  # Default was 2

    frames_before_onset = 60  # Default was 20
    frames_after_onset = 60  # Default was 20

    dead_threshold: int = 30
    adjusted_events_normalisation: int = 1000
    significant_threshold: int = 5
    success_threshold: int = 5
    major_event_threshold: int = 20
    success_direction_threshold: int = 25
    final_event_threshold: int = 170
    final_event_F1_threshold: int = 100
    max_event_threshold: int = 10

    pause_threshold: int = 5
    pause_window: int = 5
    pause_min_duration: int = 5

    backoff_threshold: int = 30

    # Skeleton tracking configuration attributes

    # skeleton_tracks_smoothing: bool = False

    # Template size
    template_width: int = 96
    template_height: int = 516

    padding: int = 20
    y_crop: tuple = (74, 0)

    # # Skeleton metrics

    contact_nodes = ["Rfront", "Lfront", "Head"]

    contact_threshold: tuple = (0, 14)  # Was 13 before
    gap_between_contacts: Optional[float] = None  # 1 / 4  # Set to None to allow disabling gap merging
    contact_min_length: Optional[float] = None  # 1 / 4  # Set to None to allow disabling min length filtering

    # Head pushing detection configuration
    exclude_hidden_legs: bool = True  # If True, exclude contacts where legs are not visible (NaN tracking data)
    head_pushing_threshold: float = 0.6  # Fraction of frames required for clear head/leg pushing classification
    late_contact_window: float = 0.3  # Fraction of contact duration to analyze for late leg extension (0.3 = last 30%)

    # Standardized events mode: "interaction_events" uses standardized interactions from tracking data,
    # "contact_events" uses skeleton-based contact detection for creating standardized events
    standardized_events_mode: str = "contact_events"  # Options: "interaction_events", "contact_events"

    # Skeleton metrics: longer

    skeleton_tracks_smoothing: bool = False

    # contact_nodes = ["Thorax",  "Head"]

    # contact_threshold: tuple = (0, 40)
    # gap_between_contacts: int = 3 / 2
    # contact_min_length: int = 2

    fly_only = False

    # hidden_value: int = -1
    hidden_value: int = -9999

    def print_config(self):
        """
        Print the current configuration parameters.
        """
        print("Config loaded with the following parameters:")
        for field_name, field_value in self.__dict__.items():
            print(f"{field_name}: {field_value}")

    def set_experiment_config(self, experiment_type):
        """
        Set the configuration for the experiment based on the experiment type.

        Args:
            experiment_type (str): The type of experiment, e.g., 'MagnetBlock', 'Training', etc.
        """
        if experiment_type == "MagnetBlock":
            self.time_range = (3600, None)
            self.final_event_threshold = 100

            print("MagnetBlock experiment configuration set.")
            # Add other specific settings for MagnetBlock
        elif experiment_type == "Learning":
            # Learning experiment specific settings
            self.trial_peak_height = 0.23  # Height threshold for peak detection
            self.trial_peak_distance = 500  # Minimum distance between peaks
            self.trial_skip_frames = 500  # Initial frames to skip in a trial
            self.trial_min_count = 2  # Minimum number of trials for a valid fly

    def set_property(self, property_name, value):
        """
        Set the value of a property in the configuration.

        Args:
            property_name (str): The name of the property to set.
            value: The value to set for the property.
        """
        if hasattr(self, property_name):
            setattr(self, property_name, value)
        else:
            raise AttributeError(f"Config has no attribute named '{property_name}'")
