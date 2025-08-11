from __future__ import annotations
from pathlib import Path
import cv2
import datetime
import numpy as np
import pygame
import os
import gc
from functools import lru_cache
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.fx.speedx import speedx

from utils_behavior import Utils

from Ballpushing_utils.fly_trackingdata import FlyTrackingData
from Ballpushing_utils.ballpushing_metrics import BallPushingMetrics
from Ballpushing_utils.skeleton_metrics import SkeletonMetrics
from Ballpushing_utils.learning_metrics import LearningMetrics
from Ballpushing_utils.f1_metrics import F1Metrics
from Ballpushing_utils.interactions_metrics import InteractionsMetrics
from Ballpushing_utils.trajectory_metrics import TrajectoryMetrics


class Fly:
    """
    A class for a single fly. This represents a folder containing a video, associated tracking files, and metadata files. It is usually contained in an Experiment object, and inherits the Experiment object's metadata.
    """

    def __init__(
        self,
        directory,
        experiment=None,
        as_individual=False,
        custom_config=None,
    ):
        """
        Initialize a Fly object.

        Args:
            directory (Path): The path to the fly directory.
            experiment (Experiment, optional): An optional Experiment object. If not provided, an Experiment object will be created based on the parent directory of the given directory.

        Attributes:
            directory (Path): The path to the fly directory.
            experiment (Experiment): The Experiment object associated with the Fly.
            arena (str): The name of the parent directory of the fly directory.
            corridor (str): The name of the fly directory.
            name (str): A string combining the name of the experiment directory, the arena, and the corridor.
            arena_metadata (dict): Metadata for the arena, obtained by calling the get_arena_metadata method.
            video (Path): The path to the video file in the fly directory.
            flytrack (Path): The path to the fly tracking file in the fly directory. If not found, a message is printed and this attribute is not set.
            balltrack (Path): The path to the ball tracking file in the fly directory. If not found, a message is printed and this attribute is not set.
            flyball_positions (DataFrame): The coordinates of the fly and the ball, obtained by calling the get_coordinates function with the flytrack and balltrack paths.
        """

        from Ballpushing_utils.experiment import Experiment
        from Ballpushing_utils.fly_metadata import FlyMetadata

        self.directory = Path(directory)

        if experiment is not None:
            self.experiment = experiment
        elif as_individual:
            self.experiment = Experiment(self.directory.parent.parent, metadata_only=True)
        else:
            self.experiment = Experiment(self.directory.parent.parent)

        self.config = self.experiment.config

        # self.experiment_type = experiment_type

        if self.config.experiment_type:
            self.config.set_experiment_config(self.config.experiment_type)

        # Apply custom configuration if provided
        if custom_config:
            self._apply_custom_config(custom_config)

        if self.config.debugging:
            print(f" Time range: {self.config.time_range}")

        self.metadata = FlyMetadata(self)

        self.Genotype = self.metadata.arena_metadata.get("Genotype", "Unknown")

        self._tracking_data = None
        self._trajectory_metrics = None

        # Check if the fly has valid tracking data
        try:
            if as_individual:
                if self.tracking_data is None or not self.tracking_data.valid_data:
                    if self.config.debugging:
                        print(f"Invalid data for: {self.metadata.name}. Skipping.")
                    return
        except Exception as e:
            print(f"Error initializing tracking data for {self.metadata.name}: {e}")
            return

        self.flyball_positions = None
        self.fly_skeleton = None

        # After tracking data is loaded and valid, build flyball_positions with both fly and ball data
        if self.tracking_data is not None and self.tracking_data.valid_data:
            # Get fly and ball DataFrames (first object for each)
            fly_df = None
            ball_df = None
            if self.tracking_data.flytrack is not None and len(self.tracking_data.flytrack.objects) > 0:
                fly_df = self.tracking_data.flytrack.objects[0].dataset
            if self.tracking_data.balltrack is not None and len(self.tracking_data.balltrack.objects) > 0:
                ball_df = self.tracking_data.balltrack.objects[0].dataset
            if fly_df is not None and ball_df is not None:
                # Merge on index, suffixing overlapping columns
                self.flyball_positions = fly_df.join(ball_df, lsuffix="_fly", rsuffix="_ball", how="outer")
            elif fly_df is not None:
                self.flyball_positions = fly_df.copy()
            elif ball_df is not None:
                self.flyball_positions = ball_df.copy()
            # Add percent_completion column if rel_x_ball is present
            if self.flyball_positions is not None and "rel_x_ball" in self.flyball_positions:
                from Ballpushing_utils.trajectory_metrics import TrajectoryMetrics

                ball_x_metrics = TrajectoryMetrics(self)
                self.flyball_positions["percent_completion"] = ball_x_metrics.percent_completion()

        self._event_metrics = None

        self._event_summaries = None

        self._f1_metrics = None

        self._learning_metrics = None

        self._skeleton_metrics = None

    def _apply_custom_config(self, custom_config):
        """
        Apply custom configuration values to the Fly's config.

        Args:
            custom_config (str or dict): A path to a JSON/YAML file or a dictionary of custom configuration values.
        """
        if isinstance(custom_config, str):
            # Load configuration from a JSON or YAML file
            config_path = Path(custom_config)
            if config_path.suffix in [".json"]:
                with open(config_path, "r") as f:
                    custom_config = json.load(f)
            elif config_path.suffix in [".yaml", ".yml"]:
                with open(config_path, "r") as f:
                    custom_config = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format. Use JSON or YAML.")

        if not isinstance(custom_config, dict):
            raise ValueError("Custom configuration must be a dictionary or a path to a JSON/YAML file.")

        # Update the Fly's config with the custom values
        for key, value in custom_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise AttributeError(f"Config has no attribute named '{key}'")

    @property
    def tracking_data(self):
        if self._tracking_data is None:
            self._tracking_data = FlyTrackingData(self)
            if not self._tracking_data.valid_data:
                if self.config.debugging:
                    print(f"Invalid data for: {self.metadata.name}. Skipping.")
                self._tracking_data = None
        return self._tracking_data

    @property
    def event_metrics(self):
        if self._event_metrics is None:
            # print("Computing events metrics...")
            self._event_metrics = InteractionsMetrics(self.tracking_data).metrics
        return self._event_metrics

    @property
    def event_summaries(self):
        if self._event_summaries is None:
            # print("Computing events metrics...")
            # Cache the entire BallPushingMetrics object to reuse computations
            if not hasattr(self, "_ballpushing_metrics_obj"):
                self._ballpushing_metrics_obj = BallPushingMetrics(self.tracking_data)
            self._event_summaries = self._ballpushing_metrics_obj.metrics
        return self._event_summaries

    @property
    def f1_metrics(self):
        if self._f1_metrics is None and self.config.experiment_type == "F1":
            self._f1_metrics = F1Metrics(self.tracking_data).metrics
        return self._f1_metrics

    @property
    def learning_metrics(self):
        if self._learning_metrics is None and self.config.experiment_type == "Learning":
            self._learning_metrics = LearningMetrics(self.tracking_data)
        return self._learning_metrics

    @property
    def skeleton_metrics(self):
        # Check if we have a cached result first to avoid recomputation
        if hasattr(self, "_skeleton_metrics_cached"):
            return self._skeleton_metrics_cached

        if self.tracking_data is None:
            if self.config.debugging:
                print("No tracking data available.")
            return None

        if self.tracking_data.skeletontrack is None:
            if self.config.debugging:
                print("No skeleton data available.")
            return None

        if self._skeleton_metrics is None:
            self._skeleton_metrics = SkeletonMetrics(self)

        # Cache the result to prevent expensive recomputation
        self._skeleton_metrics_cached = self._skeleton_metrics
        return self._skeleton_metrics_cached

    @property
    def trajectory_metrics(self):
        """
        Lazy-loaded property for TrajectoryMetrics, using the fly's y position (y_thorax) by default.
        Returns:
            TrajectoryMetrics or None: Instance if data available, else None.
        """
        if self._trajectory_metrics is None:
            if self.flyball_positions is not None and "y_thorax" in self.flyball_positions:
                self._trajectory_metrics = TrajectoryMetrics(self.flyball_positions["y_thorax"].values)
            else:
                if self.config.debugging:
                    print(f"No 'y_thorax' in flyball_positions for {self.metadata.name}.")
                self._trajectory_metrics = None
        return self._trajectory_metrics

    def __str__(self):
        # Handle cases where tracking data might be None
        flytrack = self.tracking_data.flytrack if self.tracking_data is not None else "No Flytrack"
        balltrack = self.tracking_data.balltrack if self.tracking_data is not None else "No Balltrack"

        return (
            f"Fly: {self.metadata.name}\n"
            f"Arena: {self.metadata.arena}\n"
            f"Corridor: {self.metadata.corridor}\n"
            f"Video: {self.metadata.video}\n"
            f"Flytrack: {flytrack}\n"
            f"Balltrack: {balltrack}\n"
            f"Genotype: {self.Genotype}"
        )

    def __repr__(self):
        return f"Fly({self.directory})"

    def clear_caches(self):
        """
        Clear all caches to free up memory. This should be called between processing different flies.
        """
        # Clear cached metrics objects
        self._event_summaries = None
        self._f1_metrics = None
        self._learning_metrics = None
        self._skeleton_metrics = None
        self._trajectory_metrics = None

        # Clear cached skeleton metrics (specific cache)
        if hasattr(self, "_skeleton_metrics_cached"):
            delattr(self, "_skeleton_metrics_cached")

        # Clear BallPushingMetrics object cache if it exists
        if hasattr(self, "_ballpushing_metrics_obj"):
            self._ballpushing_metrics_obj.clear_caches()
            delattr(self, "_ballpushing_metrics_obj")

        # Clear tracking data caches if tracking data exists
        if self._tracking_data is not None:
            # Clear FlyTrackingData internal caches
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
                if hasattr(self._tracking_data, attr):
                    delattr(self._tracking_data, attr)

        # Force garbage collection
        gc.collect()

    def generate_clip(self, event, outpath=None, fps=None, width=None, height=None, tracks=False):
        """
        Generate a video clip for a given event.

        This method creates a video clip from the original video for the duration of the event. It also adds text to each frame indicating the event number and start time. If the 'yball' value varies more than a certain threshold during the event, a red dot is added to the frame.

        Args:
            event (list or int): A list containing the start and end indices of the event in the 'flyball_positions' DataFrame. Alternatively, an integer can be provided to indicate the index of the event in the 'interaction_events' list.
            outpath (Path): The directory where the output video clip should be saved.
            fps (int): The frames per second of the original video.
            width (int): The width of the output video frames.
            height (int): The height of the output video frames.

        Returns:
            str: The path to the output video clip.
        """
        # Ensure tracking data is available
        if self.tracking_data is None or self.tracking_data.interaction_events is None:
            if self.config.debugging:
                print(f"No tracking data or interaction events available for {self.metadata.name}.")
            return None

        # Ensure the video file exists
        if not self.metadata.video or not Path(self.metadata.video).exists():
            if self.config.debugging:
                print(f"Video file not found for {self.metadata.name}.")
            return None

        # If no outpath is provided, use a default path based on the fly's name and the event number
        if not outpath:
            outpath = Utils.get_labserver() / "Videos"

        # Check if the event is an integer or a list
        if isinstance(event, int):
            try:
                event = self.tracking_data.interaction_events[event - 1]
            except IndexError:
                if self.config.debugging:
                    print(f"Invalid event index {event} for {self.metadata.name}.")
                return None

        start_frame, end_frame = event[0], event[1]
        cap = cv2.VideoCapture(str(self.metadata.video))

        # If no fps, width, or height is provided, use the original video's fps, width, and height
        if not fps:
            fps = cap.get(cv2.CAP_PROP_FPS)
        if not width:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if not height:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened():
            if self.config.debugging:
                print(f"Failed to open video file for {self.metadata.name}.")
            return None

        try:
            start_time = start_frame / fps
            start_time_str = str(datetime.timedelta(seconds=int(start_time)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

            # Get the index of the event in the list to apply it to the output file name
            try:
                if self.tracking_data is None or self.tracking_data.interaction_events is None:
                    if self.config.debugging:
                        print(f"No tracking data or interaction events available for {self.metadata.name}.")
                    return None

                # Ensure interaction_events is a list
                if isinstance(self.tracking_data.interaction_events, list):
                    event_index = self.tracking_data.interaction_events.index(event)
                else:
                    if self.config.debugging:
                        print(f"interaction_events is not a list for {self.metadata.name}.")
                    return None
            except ValueError:
                if self.config.debugging:
                    print(f"Event not found in interaction events for {self.metadata.name}.")
                return None

            if outpath == Utils.get_labserver() / "Videos":
                clip_path = outpath.joinpath(f"{self.metadata.name}_{event_index}.mp4").as_posix()
            else:
                clip_path = outpath.joinpath(f"output_{event_index}.mp4").as_posix()

            out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for frame_idx in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        if self.config.debugging:
                            print(f"Failed to read frame {frame_idx} for {self.metadata.name}.")
                        break

                    # If tracks is True, add the tracking data to the frame
                    if tracks and self.flyball_positions is not None:
                        if frame_idx in self.flyball_positions.index:
                            flyball_coordinates = self.flyball_positions.loc[frame_idx]
                            frame = self.draw_circles(frame, flyball_coordinates)

                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    # Write some text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"Event:{event_index + 1} - start:{start_time_str}"
                    font_scale = width / 150
                    thickness = int(4 * font_scale)
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

                    # Position the text at the top center of the frame
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = 25
                    cv2.putText(
                        frame,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

                    # Check if yball value varies more than threshold
                    if self.tracking_data.check_yball_variation(event)[0]:  # You need to implement this function
                        # Add red dot to segment
                        dot = np.zeros((10, 10, 3), dtype=np.uint8)
                        dot[:, :, 2] = 255
                        dot = cv2.resize(dot, (20, 20))

                        # Position the dot right next to the text at the top of the frame
                        dot_x = text_x + text_size[0] + 10
                        dot_y = text_y - int(dot.shape[0] * 1.2) + text_size[1] // 2

                        frame[dot_y : dot_y + dot.shape[0], dot_x : dot_x + dot.shape[1]] = dot

                    # Write the frame into the output file
                    out.write(frame)

            finally:
                out.release()
        finally:
            cap.release()

        return clip_path

    def concatenate_clips(self, clips, outpath, fps, width, height, vidname):
        """
        Concatenate multiple video clips into a single video.

        This method takes a list of video clip paths, reads each clip frame by frame, and writes the frames into a new video file. The new video file is saved in the specified output directory with the specified name.

        Args:
            clips (list): A list of paths to the video clips to be concatenated.
            outpath (Path): The directory where the output video should be saved.
            fps (int): The frames per second for the output video.
            width (int): The width of the output video frames.
            height (int): The height of the output video frames.
            vidname (str): The name of the output video file (without the extension).

        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(outpath.joinpath(f"{vidname}.mp4").as_posix(), fourcc, fps, (height, width))
        try:
            for clip_path in clips:
                cap = cv2.VideoCapture(clip_path)
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                finally:
                    cap.release()
        finally:
            out.release()

    def generate_interactions_video(self, outpath=None, tracks=False):
        """
        Generate a video of the fly's interactions with the ball.

        This method detects interaction events, generates a video clip for each event, concatenates the clips into a single video, and saves the video in the specified output directory. The video is named after the fly's name and genotype. If the genotype is not defined, 'undefined' is used instead. After the video is created, the individual clips are deleted.

        Args:
            outpath (Path, optional): The directory where the output video should be saved. If None, the video is saved in the fly's directory. Defaults to None.
        """

        if self.flyball_positions is None:
            if self.config.debugging:
                print(f"No tracking data available for {self.metadata.name}. Skipping...")
            return

        if outpath is None:
            outpath = self.directory
        if self.tracking_data is None:
            if self.config.debugging:
                print(f"No tracking data available for {self.metadata.name}. Skipping...")
            return
        events = self.tracking_data.interaction_events
        clips = []

        cap = cv2.VideoCapture(str(self.metadata.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        vidname = f"{self.metadata.name}_{self.Genotype if self.Genotype else 'undefined'}"

        if events is None:
            if self.config.debugging:
                print(f"No interaction events available for {self.metadata.name}. Skipping...")
            return

        for i, event in enumerate(events):
            clip_path = self.generate_clip(event, outpath, fps, width, height, tracks)
            clips.append(clip_path)
        self.concatenate_clips(clips, outpath, fps, width, height, vidname)
        for clip_path in clips:
            os.remove(clip_path)
        print(f"Finished processing {vidname}!")

    def draw_circles(self, frame, flyball_coordinates):
        """
        Draw circles at the positions of the fly and ball in a video frame.

        Parameters:
        frame (numpy.ndarray): The video frame.
        fly_coordinates (pandas.Series): The tracking data for the fly.
        ball_coordinates (pandas.Series): The tracking data for the ball.
        frame_number (int): The number of the frame.

        Returns:
        numpy.ndarray: The video frame with the circles.
        """

        # Extract the x and y coordinates from the pandas Series
        fly_pos = tuple(map(int, [flyball_coordinates["xfly"], flyball_coordinates["yfly"]]))
        ball_pos = tuple(map(int, [flyball_coordinates["xball"], flyball_coordinates["yball"]]))

        # Draw a smaller circle at the fly's position
        cv2.circle(frame, fly_pos, 5, (0, 0, 255), -1)  # Adjust the radius to 5

        # Draw a smaller circle at the ball's position
        cv2.circle(frame, ball_pos, 5, (255, 0, 0), -1)  # Adjust the radius to 5

        # Draw an empty circle around the ball
        cv2.circle(frame, ball_pos, 11, (255, 0, 0), 2)  # Adjust the radius to 25 and the thickness to 2

        return frame

    def generate_preview(self, speed=60.0, save=False, preview=False, output_path=None, tracks=True):
        """
        Generate an accelerated version of the video using moviepy.

        This method uses the moviepy library to speed up the video, add circles at the positions of the fly and ball in each frame if tracking data is provided, and either save the resulting video or preview it using Pygame.

        Parameters:
        speed (float): The speedup factor. For example, 2.0 will double the speed of the video.
        save (bool, optional): Whether to save the sped up video. If True, the video is saved. If False, the video is previewed using Pygame. Defaults to False.
        output_path (str, optional): The path to save the sped up video. If not provided and save is True, a default path will be used. Defaults to None.
        tracks (dict, optional): A dictionary containing the tracking data for the fly and ball. Each key should be a string ('fly' or 'ball') and each value should be a numpy array with the x and y coordinates for each frame. Defaults to None.
        """

        if save and output_path is None:
            # Use the default output path
            output_path = (
                Utils.get_labserver()
                / "Videos"
                / "Previews"
                / f"{self.metadata.name}_{self.Genotype if self.Genotype else 'undefined'}_x{speed}.mp4"
            )

        # Load the video file
        clip = VideoFileClip(self.metadata.video.as_posix())

        # If tracks is True, add circles to the video

        # Create a new video clip with the draw_circles function applied to each frame
        if tracks:
            # Check if tracking data is available
            if self.flyball_positions is None:
                raise ValueError("No tracking data available.")

            # Create a new video clip with the draw_circles function applied to each frame
            marked_clip = VideoClip(
                lambda t: self.draw_circles(
                    clip.get_frame(t),
                    self.flyball_positions.loc[round(t * clip.fps)] if self.flyball_positions is not None else None,
                ),
                duration=clip.duration,
            )
        else:
            marked_clip = clip

        sped_up_clip = marked_clip.fx(speedx, speed)

        # If saving, write the new video clip to a file
        if save:
            if output_path is None:
                raise ValueError("Output path must be provided when save is True.")
            print(f"Saving {self.metadata.video.name} at {speed}x speed in {output_path.parent}")
            sped_up_clip.write_videofile(str(output_path), fps=clip.fps)  # Save the sped-up clip

        # If not saving, preview the new video clip
        if preview:
            # Check if running over SSH
            if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
                raise EnvironmentError(
                    "Preview mode shouldn't be run over SSH. Set preview argument to False or run locally."
                )

            # Initialize Pygame display
            pygame.display.init()

            # Set the title of the Pygame window
            pygame.display.set_caption(f"Preview (speed = x{speed})")

            print(f"Previewing {self.metadata.video.name} at {speed}x speed")

            sped_up_clip.preview(fps=self.experiment.fps * speed)

            # Manually close the preview window
            pygame.quit()

        # Close the video file to release resources
        clip.close()

        if not save and not preview:
            print("No action specified. Set save or preview argument to True.")
