import pandas as pd
from Ballpushing_utils import utilities
import numpy as np
from utils_behavior import Sleap_utils
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from matplotlib import pyplot as plt
from moviepy.video.fx.speedx import speedx


class SkeletonMetrics:
    """
    A class for computing metrics from the skeleton data. It requires to have a fly object with a valid skeleton data, and check whether there is a "preprocessed" ball data available.
    """

    def __init__(self, fly):
        self.fly = fly
        self.ball = self.fly.tracking_data.raw_balltrack
        self.preprocess_ball()

        # Find all contact events
        self.all_contacts = self.find_contact_events()

        # print(f"Number of contact events: {len(self.all_contacts)}")

        # Determine the final contact if success_cutoff is enabled
        if self.fly.config.success_cutoff:
            final_contact_idx, _ = self.get_final_contact()
            if final_contact_idx is not None:
                self.contacts = self.all_contacts[: final_contact_idx + 1]
            else:
                self.contacts = self.all_contacts
        else:
            self.contacts = self.all_contacts

        # print(f"Number of final contact events: {len(self.contacts)}")

        self.flyball_dataset = self.get_contact_annotated_dataset()

        self.ball_displacements = self.compute_ball_displacements()

        self.fly_centered_tracks = self.compute_fly_centered_tracks()

        self.events_based_contacts = self.compute_events_based_contacts()

    def resize_coordinates(self, x, y, original_width, original_height, new_width, new_height):
        """Resize the coordinates according to the new frame size."""
        x_scale = new_width / original_width
        y_scale = new_height / original_height
        return int(x * x_scale), int(y * y_scale)

    def apply_arena_mask_to_labels(self, x, y, mask_padding, crop_top, crop_bottom, new_height):
        """Adjust the coordinates according to the cropping and padding applied to the frame."""
        # Crop from top and bottom
        if crop_top <= y < (new_height - crop_bottom):
            y -= crop_top
        else:
            return None, None

        # Add padding to the left and right
        x += mask_padding

        return x, y

    def resize_and_transform_coordinate(
        self,
        x,
        y,
        original_width,
        original_height,
        new_width,
        new_height,
        mask_padding,
        crop_top,
        crop_bottom,
    ):
        """Resize and transform the coordinate to match the preprocessed frame."""
        # Resize the coordinate
        x, y = self.resize_coordinates(x, y, original_width, original_height, new_width, new_height)

        # Apply cropping offset and padding
        x, y = self.apply_arena_mask_to_labels(x, y, mask_padding, crop_top, crop_bottom, new_height)

        return x, y

    def preprocess_ball(self):
        """Transform the ball coordinates to match the skeleton data."""

        ball_data = self.ball.objects[0].dataset

        ball_coords = [(x, y) for x, y in zip(ball_data["x_centre"], ball_data["y_centre"])]

        # Apply resizing, cropping, and padding to the ball tracking data
        ball_coords = [
            self.resize_and_transform_coordinate(
                x,
                y,
                self.fly.metadata.original_size[0],
                self.fly.metadata.original_size[1],
                self.fly.config.template_width,
                self.fly.config.template_height,
                self.fly.config.padding,
                self.fly.config.y_crop[0],
                self.fly.config.y_crop[1],
            )
            for x, y in ball_coords
        ]

        ball_data["x_centre_preprocessed"] = [x for x, y in ball_coords if x is not None and y is not None]
        ball_data["y_centre_preprocessed"] = [y for x, y in ball_coords if x is not None and y is not None]

        # Add x_centre_preprocessed and y_centre_preprocessed in the node_names

        self.ball.node_names.extend(["centre_preprocessed"])

        # print(self.ball.objects[0].dataset)
        # print(self.ball.node_names)

        self.ball

    def find_contact_events(self, threshold=None, gap_between_events=None, event_min_length=None):
        if threshold is None:
            threshold = self.fly.experiment.config.contact_threshold
        if gap_between_events is None:
            gap_between_events = 0  # 0 means no gap merging
        if event_min_length is None:
            event_min_length = 0  # 0 means no min length filtering

        fly_data = self.fly.tracking_data.skeletontrack.objects[0].dataset
        ball_data = self.ball.objects[0].dataset

        # Find all contact events
        contact_events = utilities.find_interaction_events(
            fly_data,
            ball_data,
            nodes1=self.fly.experiment.config.contact_nodes,
            nodes2=["centre_preprocessed"],
            threshold=threshold,
            gap_between_events=gap_between_events,
            event_min_length=event_min_length,
        )

        return contact_events

    def find_contact_by_distance(self, threshold, distance_type="max"):
        ball_data = self.ball.objects[0].dataset

        # ball_data["euclidean_distance"] = np.sqrt(
        #     (
        #         ball_data["x_centre_preprocessed"]
        #         - ball_data["x_centre_preprocessed"].iloc[0]
        #     )
        #     ** 2
        #     + (
        #         ball_data["y_centre_preprocessed"]
        #         - ball_data["y_centre_preprocessed"].iloc[0]
        #     )
        #     ** 2
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
            event, event_index = next((event, i) for i, event in enumerate(self.all_contacts) if distance_check(event))
        except StopIteration:
            event, event_index = None, None

        return event, event_index

    def compute_events_based_contacts(self):
        """
        List fly relative tracking data associated with interaction events and random negative examples.
        """

        events = []

        # Process standardized interaction events
        if hasattr(self.fly.tracking_data, "standardized_interactions"):
            for (fly_idx, ball_idx), event_list in self.fly.tracking_data.standardized_interactions.items():
                for event_counter, event in enumerate(event_list):
                    if len(event) != 2:
                        print(f"Skipping malformed event: {event}")
                        continue

                    start_frame, end_frame = int(event[0]), int(event[1])

                    # Validate indices
                    if start_frame >= len(self.fly_centered_tracks) or end_frame > len(self.fly_centered_tracks):
                        print(
                            f"Invalid event bounds {start_frame}-{end_frame} for track length {len(self.fly_centered_tracks)}"
                        )
                        continue

                    # Process interaction event data
                    event_data = self.fly_centered_tracks.iloc[start_frame:end_frame].copy()
                    event_data["event_id"] = event_counter
                    event_data["time_rel_onset"] = (event_data.index - start_frame) / self.fly.experiment.fps
                    event_data["fly_idx"] = fly_idx
                    event_data["ball_idx"] = ball_idx
                    event_data["adjusted_frame"] = range(end_frame - start_frame)
                    event_data["event_type"] = "interaction"

                    # Calculate ball displacement
                    ball_disp = np.sqrt(
                        (event_data["x_centre_preprocessed"] - event_data["x_centre_preprocessed"].iloc[0]) ** 2
                        + (event_data["y_centre_preprocessed"] - event_data["y_centre_preprocessed"].iloc[0]) ** 2
                    )
                    event_data["ball_displacement"] = ball_disp

                    events.append(event_data)

        else:
            print("No standardized interactions found")

        if self.fly.config.generate_random:
            # Process standardized random events
            if hasattr(self.fly.tracking_data, "standardized_random_events"):
                for (fly_idx, ball_idx), random_event_list in self.fly.tracking_data.standardized_random_events.items():
                    for event_counter, random_event in enumerate(random_event_list):
                        if len(random_event) != 2:
                            print(f"Skipping malformed random event: {random_event}")
                            continue

                        start_frame, end_frame = int(random_event[0]), int(random_event[1])

                        # Validate indices
                        if start_frame >= len(self.fly_centered_tracks) or end_frame > len(self.fly_centered_tracks):
                            print(
                                f"Invalid random event bounds {start_frame}-{end_frame} for track length {len(self.fly_centered_tracks)}"
                            )
                            continue

                        # Process random event data
                        random_data = self.fly_centered_tracks.iloc[start_frame:end_frame].copy()
                        random_data["event_id"] = event_counter
                        random_data["time_rel_onset"] = (random_data.index - start_frame) / self.fly.experiment.fps
                        random_data["fly_idx"] = fly_idx
                        random_data["ball_idx"] = ball_idx
                        random_data["adjusted_frame"] = range(end_frame - start_frame)
                        random_data["event_type"] = "random"

                        # Calculate ball displacement
                        ball_disp = np.sqrt(
                            (random_data["x_centre_preprocessed"] - random_data["x_centre_preprocessed"].iloc[0]) ** 2
                            + (random_data["y_centre_preprocessed"] - random_data["y_centre_preprocessed"].iloc[0]) ** 2
                        )
                        random_data["ball_displacement"] = ball_disp

                        events.append(random_data)

        else:
            print("No standardized random events found")

        return pd.concat(events).reset_index(drop=True) if events else pd.DataFrame()

    def get_final_contact(self, threshold=None, init=False):
        if threshold is None:
            threshold = self.fly.config.final_event_threshold

        final_contact, final_contact_idx = self.find_contact_by_distance(threshold, distance_type="threshold")

        final_contact_time = final_contact[0] / self.fly.experiment.fps if final_contact else None

        # print(f"final_contact_idx from skeleton metric: {final_contact_idx}")

        return final_contact_idx, final_contact_time

    def compute_ball_displacements(self):
        """
        Compute the derivative of the ball position for each contact event
        """

        self.ball_displacements = []

        for event in self.contacts:
            # Get the ball positions for the event
            ball_positions = self.ball.objects[0].dataset.loc[event[0] : event[1]]
            # Get the derivative of the ball positions

            ball_velocity = np.mean(abs(np.diff(ball_positions["y_centre_preprocessed"], axis=0)))

            self.ball_displacements.append(ball_velocity)

        return self.ball_displacements

    def compute_fly_centered_tracks(self):
        """
        Compute fly-centric coordinates by:
        1. Translating all points to have thorax as origin
        2. Rotating to align thorax-head direction with positive y-axis

        Returns DataFrame with '_fly' suffix columns for transformed coordinates
        """
        tracking_data = self.fly.tracking_data.skeletontrack.objects[0].dataset.copy()

        # Add ball coordinates to tracking data
        tracking_data["x_centre_preprocessed"] = self.ball.objects[0].dataset["x_centre_preprocessed"]
        tracking_data["y_centre_preprocessed"] = self.ball.objects[0].dataset["y_centre_preprocessed"]

        # Get reference points
        thorax_x = tracking_data["x_Thorax"].values
        thorax_y = tracking_data["y_Thorax"].values
        head_x = tracking_data["x_Head"].values
        head_y = tracking_data["y_Head"].values

        # Calculate direction vector components
        dx = head_x - thorax_x
        dy = head_y - thorax_y
        mag = np.hypot(dx, dy)
        valid = mag > 1e-6  # Valid frames with measurable head direction

        # Calculate rotation components (vectorized operations)
        cos_theta = dy / mag
        sin_theta = dx / mag
        cos_theta[~valid] = 0  # Handle invalid frames
        sin_theta[~valid] = 0

        # Get all trackable nodes (excluding existing '_fly' columns)
        nodes = [col[2:] for col in tracking_data.columns if col.startswith("x_") and not col.endswith("_fly")]

        # Create transformed dataframe
        transformed = tracking_data.copy()

        for node in nodes:
            x_col = f"x_{node}"
            y_col = f"y_{node}"

            # 1. Translate to thorax-centric coordinates
            x_trans = tracking_data[x_col] - thorax_x
            y_trans = tracking_data[y_col] - thorax_y

            # 2. Apply rotation matrix (vectorized)
            # New x = x_trans * cosθ - y_trans * sinθ
            # New y = x_trans * sinθ + y_trans * cosθ
            transformed[f"{x_col}_fly"] = x_trans * cos_theta - y_trans * sin_theta
            transformed[f"{y_col}_fly"] = x_trans * sin_theta + y_trans * cos_theta

            # Handle invalid frames using config value
            transformed.loc[~valid, [f"{x_col}_fly", f"{y_col}_fly"]] = self.fly.config.hidden_value

        return transformed

    def compute_fly_centered_tracks_old(self):
        """
        Compute the fly-relative tracks for each skeleton tracking datapoint
        """

        tracking_data = self.fly.tracking_data.skeletontrack.objects[0].dataset
        # Add the ball tracking data

        tracking_data["x_centre_preprocessed"] = self.ball.objects[0].dataset["x_centre_preprocessed"]
        tracking_data["y_centre_preprocessed"] = self.ball.objects[0].dataset["y_centre_preprocessed"]

        thorax = tracking_data[["x_Thorax", "y_Thorax"]].values
        head = tracking_data[["x_Head", "y_Head"]].values

        # Vectorized calculations
        dxdy = head - thorax
        mag = np.linalg.norm(dxdy, axis=1)
        valid = mag > 1e-6  # Filter frames with valid head direction

        # Only calculate rotation where valid
        cos_theta = np.zeros_like(mag)
        sin_theta = np.zeros_like(mag)
        cos_theta[valid] = dxdy[valid, 1] / mag[valid]
        sin_theta[valid] = dxdy[valid, 0] / mag[valid]

        # Transform all points using matrix operations
        translated = tracking_data.filter(like="x_").values - thorax[:, 0][:, None]
        rotated = np.empty_like(translated)
        rotated[valid] = translated[valid] * cos_theta[valid, None] - translated[valid] * sin_theta[valid, None]

        # Create transformed dataframe
        transformed = tracking_data.copy()
        for i, col in enumerate([c for c in tracking_data.columns if c.startswith("x_")]):
            transformed[f"{col}_fly"] = rotated[:, i]

        return transformed

    def plot_skeleton_and_ball(self, frame=2039):
        """
        Plot the skeleton and ball tracking data on a given frame.
        """

        nodes_list = self.fly.tracking_data.skeletontrack.node_names + [
            node for node in self.ball.node_names if "preprocessed" in node
        ]

        annotated_frame = Sleap_utils.generate_annotated_frame(
            video=self.fly.tracking_data.skeletontrack.video,
            sleap_tracks_list=[self.fly.tracking_data.skeletontrack, self.ball],
            frame=frame,
            nodes=nodes_list,
        )

        # Plot the frame with the skeleton and ball tracking data
        plt.imshow(annotated_frame)
        plt.axis("off")
        plt.show()

        return annotated_frame

    def generate_contacts_video(self, output_path):
        """
        Generate a video of all contact events concatenated with annotations.
        """
        video_clips = []
        video = VideoFileClip(str(self.fly.tracking_data.skeletontrack.video))

        for idx, event in enumerate(self.contacts):
            start_frame, end_frame = event[0], event[1]
            start_time = start_frame / self.fly.experiment.fps
            end_time = end_frame / self.fly.experiment.fps
            clip = video.subclip(start_time, end_time)

            # Calculate the start time in seconds
            start_time_seconds = start_frame / 29  # Assuming 29 fps

            # Convert start time to mm:ss format
            minutes, seconds = divmod(start_time_seconds, 60)
            start_time_formatted = f"{int(minutes):02d}:{int(seconds):02d}"

            # Create a text annotation with the contact index and start time
            annotation_text = f"Contact {idx + 1}\nStart Time: {start_time_formatted}"
            annotation = TextClip(
                annotation_text,
                fontsize=8,
                color="white",
                bg_color="black",
                font="Arial",  # Specify a font that's definitely installed
                method="label",
            )
            annotation = annotation.set_position(("center", "bottom")).set_duration(clip.duration)

            # Overlay the annotation on the video clip
            annotated_clip = CompositeVideoClip([clip, annotation])
            video_clips.append(annotated_clip)

        concatenated_clip = concatenate_videoclips(video_clips)
        concatenated_clip.write_videofile(str(output_path), codec="libx264")

        print(f"Contacts video saved to {output_path}")

    def generate_contacts_grid_video(self, output_path, min_duration_sec=1.0, slow_factor=4, grid_max_contacts=12):
        """
        Generate a grid video of contact events, slowed down, with padding for short contacts.
        Args:
            output_path (str): Path to save the output video.
            min_duration_sec (float): Minimum duration (in seconds) for each contact clip.
            slow_factor (int): Slowdown factor (e.g., 4 means 4x slower).
            grid_max_contacts (int): Max number of contacts to show in the grid (default 12).
        """
        import math
        from moviepy.editor import VideoFileClip, vfx, clips_array

        # Get contact-annotated dataset and video
        annotated_df = self.get_contact_annotated_dataset()
        video = VideoFileClip(str(self.fly.tracking_data.skeletontrack.video))
        fps = self.fly.experiment.fps

        # Find contact events
        contact_events = self.find_contact_events()
        clips = []
        for event in contact_events[:grid_max_contacts]:
            start, end = event[0], event[1]
            duration_frames = end - start
            duration_sec = duration_frames / fps
            # Pad if too short
            if duration_sec < min_duration_sec:
                pad_frames = int((min_duration_sec * fps - duration_frames) / 2)
                start = max(0, start - pad_frames)
                end = min(int(video.duration * fps), end + pad_frames)
            # Extract and slow down
            clip = video.subclip(start / fps, end / fps)
            clip = clip.fx(speedx, 1.0 / slow_factor)
            clips.append(clip)

        # Compute grid size for 16:9 aspect ratio
        n = len(clips)
        if n == 0:
            print("No contacts to display.")
            return
        best_rows, best_cols, best_diff = 1, n, float("inf")
        for rows in range(1, n + 1):
            cols = math.ceil(n / rows)
            ratio = (cols * clip.w) / (rows * clip.h)
            diff = abs(ratio - 16 / 9)
            if diff < best_diff:
                best_rows, best_cols, best_diff = rows, cols, diff
        # Pad clips to fill grid
        while len(clips) < best_rows * best_cols:
            clips.append(clips[0].fx(vfx.colorx, 0.2).set_opacity(0))  # Invisible filler
        # Arrange in grid
        grid = []
        for r in range(best_rows):
            row = clips[r * best_cols : (r + 1) * best_cols]
            grid.append(row)
        grid_clip = clips_array(grid)
        grid_clip.write_videofile(str(output_path), codec="libx264")
        print(f"Contacts grid video saved to {output_path}")

    def generate_contact_annotated_interaction_events_video(self, output_path, min_duration_sec=1.0, slow_factor=1):
        """
        Generate a video of interaction events (from fly tracking),
        annotated with blue (no contact) and red (contact) overlays per frame based on skeleton metrics.
        Args:
            output_path (str): Path to save the output video.
            min_duration_sec (float): Minimum duration (in seconds) for each event.
            slow_factor (int): Slowdown factor (default 1, i.e., no slow down).
        """
        from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, concatenate_videoclips, vfx
        import numpy as np

        video = VideoFileClip(str(self.fly.tracking_data.skeletontrack.video))
        fps = self.fly.experiment.fps
        annotated_df = self.get_contact_annotated_dataset()
        interaction_events = self.fly.tracking_data.interaction_events
        # Flatten all events (may be nested dict)
        events = []
        if isinstance(interaction_events, dict):
            for fly_dict in interaction_events.values():
                if isinstance(fly_dict, dict):
                    for evlist in fly_dict.values():
                        events.extend(evlist)
                else:
                    events.extend(fly_dict)
        else:
            events = interaction_events

        video_clips = []
        for idx, event in enumerate(events):
            start, end = event[0], event[1]
            duration_frames = end - start
            duration_sec = duration_frames / fps
            # Pad if too short
            if duration_sec < min_duration_sec:
                pad_frames = int((min_duration_sec * fps - duration_frames) / 2)
                start = max(0, start - pad_frames)
                end = min(int(video.duration * fps), end + pad_frames)
            # Get contact status for each frame in this event
            contact_frames = annotated_df.loc[start:end, "is_contact"].values if "is_contact" in annotated_df else None
            if contact_frames is None or len(contact_frames) == 0:
                continue
            # Find segments of consecutive contact status
            segments = []
            prev_status = contact_frames[0]
            seg_start = start
            for i, status in enumerate(contact_frames):
                if status != prev_status or i == len(contact_frames) - 1:
                    seg_end = start + i if status != prev_status else start + i + 1
                    segments.append((seg_start, seg_end, prev_status))
                    seg_start = seg_end
                    prev_status = status
            # Create subclips for each segment
            seg_clips = []
            for seg_start, seg_end, status in segments:
                if seg_end <= seg_start:
                    continue
                clip = video.subclip(seg_start / fps, seg_end / fps)
                if slow_factor != 1:
                    clip = clip.fx(speedx, 1.0 / slow_factor)
                color = (255, 0, 0) if status else (0, 0, 255)
                bar = ColorClip(size=(clip.w, 20), color=color, duration=clip.duration)
                bar = bar.set_position((0, clip.h - 20))
                annotated_clip = CompositeVideoClip([clip, bar])
                seg_clips.append(annotated_clip)
            if seg_clips:
                event_clip = concatenate_videoclips(seg_clips)
                video_clips.append(event_clip)
        if not video_clips:
            print("No interaction events to display.")
            return
        final_clip = concatenate_videoclips(video_clips)
        final_clip.write_videofile(str(output_path), codec="libx264")
        print(f"Contact-annotated interaction events video saved to {output_path}")

    def get_contact_annotated_dataset(self):
        """
        Generate a DataFrame containing the skeleton tracks, and only the non-overlapping columns from raw balltracks, plus a column indicating contact frames.
        Returns:
            pd.DataFrame: Combined DataFrame with 'is_contact' column.
        """
        skeleton_df = self.fly.tracking_data.skeletontrack.objects[0].dataset.copy()
        ball_df = self.fly.tracking_data.raw_balltrack.objects[0].dataset.copy()

        # Find columns in ball_df not already in skeleton_df
        new_ball_cols = [col for col in ball_df.columns if col not in skeleton_df.columns]
        # Merge only these columns (plus index alignment)
        merged = skeleton_df.join(ball_df[new_ball_cols])

        # Get contact events (list of [start, end, duration] lists)
        contact_events = self.find_contact_events()

        # Create a boolean array for contact frames
        is_contact = np.zeros(len(merged), dtype=bool)
        for event in contact_events:
            if isinstance(event, (tuple, list)) and len(event) >= 2:
                start, end = event[0], event[1]
                is_contact[start:end] = True
            else:
                print(f"[WARN] Malformed contact event (expected list/tuple of at least 2): {event}")
        merged["is_contact"] = is_contact

        return merged
