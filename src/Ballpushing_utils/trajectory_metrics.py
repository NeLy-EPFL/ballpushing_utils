import numpy as np


class TrajectoryMetrics:
    """
    Computes trajectory-based metrics for a Fly, focusing on relative Y positions and percent completion.
    """

    def __init__(self, fly):
        self.fly = fly

        self.contact_dataset = (
            self.fly.skeleton_trackting.contact_dataset if hasattr(fly, "skeleton_tracking") else None
        )

        # Load relative Y positions for fly and ball
        if (
            hasattr(fly, "flyball_positions")
            and fly.flyball_positions is not None
            and "rel_y_fly" in fly.flyball_positions
            and "rel_y_ball" in fly.flyball_positions
        ):
            self.rel_y_fly = np.array(fly.flyball_positions["rel_y_fly"].values)
            self.rel_y_ball = np.array(fly.flyball_positions["rel_y_ball"].values)
        else:
            self.rel_y_fly = np.array([])
            self.rel_y_ball = np.array([])

    def percent_completion(self):
        """
        Returns the percent completion (0-100) for each frame/timepoint, as rel_y_ball / max(rel_y_ball) * 100.
        """
        if len(self.rel_y_ball) == 0:
            return np.array([])
        max_y = np.nanmax(self.rel_y_ball)
        if max_y == 0:
            return np.zeros_like(self.rel_y_ball)
        percent = 100 * self.rel_y_ball / max_y
        percent = np.clip(percent, 0, 100)
        return percent

    def plot_y_positions_over_time(self, output_path, zoom_major=True, zoom_separate=False):
        """
        Plots y_centre_preprocessed of the ball and y_Head of the fly over time from the annotated contact dataset and saves the plot.
        Args:
            output_path (str): Path to save the plot (e.g., 'output.png')
            zoom_major (bool): If True, show an inset zoom around the major event. If False, only show the main plot.
            zoom_separate (bool): If True and zoom_major is True, save the zoomed plot as a separate file instead of as an inset.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Try to get the annotated contact dataset
        annotated_df = None
        if hasattr(self.fly, "skeleton_metrics") and hasattr(
            self.fly.skeleton_metrics, "get_contact_annotated_dataset"
        ):
            annotated_df = self.fly.skeleton_metrics.get_contact_annotated_dataset()
        if annotated_df is None or annotated_df.empty:
            raise ValueError("Annotated contact dataset is not available for this fly.")

        if "y_centre_preprocessed" not in annotated_df.columns or "y_Head" not in annotated_df.columns:
            raise ValueError(
                "Required columns ('y_centre_preprocessed', 'y_Head') not found in annotated contact dataset."
            )

        y_ball = annotated_df["y_centre_preprocessed"].values
        y_head = annotated_df["y_Head"].values
        # Use time column if available, else fallback to frame/fps
        if "time" in annotated_df.columns:
            time = annotated_df["time"].values
        elif (
            "frame" in annotated_df.columns and hasattr(self.fly, "experiment") and hasattr(self.fly.experiment, "fps")
        ):
            time = annotated_df["frame"].values / float(self.fly.experiment.fps)
        else:
            # fallback: use index as frame, try to get fps
            fps = getattr(getattr(self.fly, "experiment", None), "fps", 30)
            time = np.arange(len(annotated_df)) / float(fps)
        is_contact = (
            annotated_df["is_contact"].values
            if "is_contact" in annotated_df.columns
            else np.zeros_like(y_ball, dtype=bool)
        )

        # --- Use only event_metrics for event-based trimming ---
        config = getattr(self.fly, "config", None)
        major_push_thresh = getattr(config, "major_event_threshold", 20) if config is not None else 20
        trim_end_time = time[-1]
        event_start_time = None
        event_end_time = None
        found_major_event = False
        if hasattr(self.fly, "event_metrics") and self.fly.event_metrics is not None:
            event_metrics = self.fly.event_metrics
            # Flatten all events into a list
            all_events = []
            for k, v in event_metrics.items():
                if isinstance(v, dict):
                    for ev in v.values():
                        all_events.append(ev)
            # Sort by start_time
            all_events = sorted(all_events, key=lambda e: e.get("start_time", 0))
            # Find first event with displacement >= threshold
            for ev in all_events:
                if ev.get("displacement", 0) >= major_push_thresh:
                    event_start_time = ev.get("start_time", None)
                    event_end_time = ev.get("end_time", None)
                    if event_end_time is not None:
                        trim_end_time = event_end_time + 30
                        found_major_event = True
                        print(f"[DEBUG] Major event found: start_time={event_start_time}, end_time={event_end_time}")
                    break
            if not found_major_event:
                print("[INFO] No major event (displacement >= threshold) found in event metrics.")
        else:
            print("[INFO] No event_metrics attribute found on fly object.")
        # Mask for time range
        mask = (time >= time[0]) & (time <= trim_end_time)

        # --- Plotting ---
        import matplotlib.patches as mpatches

        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        # Plot fly trace: thin, semi-transparent, background
        ax.plot(
            time[mask],
            -y_head[mask],
            label="Fly y_Head",
            color="tab:orange",
            linewidth=1,
            alpha=0.4,
            zorder=1,
        )
        # Plot ball trace: emphasize contact (red) over no-contact (blue)
        main_start_idx = None
        main_current_contact = None
        mask_indices = np.where(mask)[0]
        for i, idx in enumerate(mask_indices):
            contact = is_contact[idx]
            if main_start_idx is None:
                main_start_idx = i
                main_current_contact = contact
            elif contact != main_current_contact or i == len(mask_indices) - 1:
                seg_end = i if i < len(mask_indices) - 1 else i + 1
                if main_current_contact:
                    # Contact segments: bright red, thick, prominent
                    color = "red"
                    linewidth = 4.0
                    alpha = 1.0
                    zorder = 3
                else:
                    # No-contact segments: muted blue, thinner, less prominent
                    color = "lightsteelblue"
                    linewidth = 2.0
                    alpha = 0.6
                    zorder = 1
                ax.plot(
                    time[mask][main_start_idx:seg_end],
                    -y_ball[mask][main_start_idx:seg_end],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                    label=(
                        None
                        if main_start_idx > 0
                        else (
                            "Ball y_centre_preprocessed (contact)"
                            if main_current_contact
                            else "Ball y_centre_preprocessed (no contact)"
                        )
                    ),
                )
                main_start_idx = i
                main_current_contact = contact
        # Determine fly name for directory grouping (move this up before using fly_name)
        fly_name = getattr(getattr(self.fly, "metadata", None), "name", None)
        if not fly_name:
            fly_name = getattr(self.fly, "name", None)
        if not fly_name:
            fly_name = "Unknown"
        # Highlight the major event window on the main plot
        if found_major_event and event_start_time is not None and event_end_time is not None:
            ax.axvspan(event_start_time, event_end_time, color="yellow", alpha=0.2, label="Major event window")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y Position (inverted, image coordinates)")
        ax.set_title(f"Y Positions Over Time — {fly_name}")
        # Custom legend handles for ball contact/no contact with updated colors
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches

        legend_handles = [
            Line2D([0], [0], color="tab:orange", lw=1, alpha=0.4, label="Fly y_Head"),
            Line2D([0], [0], color="lightsteelblue", lw=2.0, alpha=0.6, label="Ball trajectory (no contact)"),
            Line2D([0], [0], color="red", lw=4.0, label="Ball trajectory (contact)"),
        ]
        if found_major_event and event_start_time is not None and event_end_time is not None:
            # Add patch to list of mixed legend elements
            legend_handles = legend_handles + [mpatches.Patch(color="yellow", alpha=0.2, label="Major event window")]
        ax.legend(handles=legend_handles)
        plt.tight_layout()
        # Always create a directory for the fly and save all plots there
        fly_dir = None
        if output_path is not None:
            base_dir = os.path.dirname(output_path)
            fly_dir = os.path.join(base_dir, fly_name)
            os.makedirs(fly_dir, exist_ok=True)
            # Main plot filename (no fly name in filename)
            main_plot_path = os.path.join(fly_dir, "fly_ball_y_positions_contact_annotated.png")
            output_path = main_plot_path
        # Add inset zoom around the major event (±10s) if zoom_major is True and zoom_separate is False
        if zoom_major and found_major_event and event_start_time is not None and event_end_time is not None:
            zoom_start = max(time[0], event_start_time - 10)
            zoom_end = min(time[-1], event_end_time + 10)
            inset_mask = (time >= zoom_start) & (time <= zoom_end)
            inset_indices = np.where(inset_mask)[0]
            if not zoom_separate:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=2)
                # Fly trace in inset
                axins.plot(
                    time[inset_mask],
                    -y_head[inset_mask],
                    color="tab:orange",
                    linewidth=1,
                    alpha=0.4,
                    zorder=1,
                )
                # Ball trace in inset with improved styling
                inset_start_idx = None
                inset_current_contact = None
                for j, idx in enumerate(inset_indices):
                    contact = is_contact[idx]
                    if inset_start_idx is None:
                        inset_start_idx = j
                        inset_current_contact = contact
                    elif contact != inset_current_contact or j == len(inset_indices) - 1:
                        seg_end = j if j < len(inset_indices) - 1 else j + 1
                        if inset_current_contact:
                            # Contact segments: bright red, thick
                            color = "red"
                            linewidth = 4.0
                            alpha = 1.0
                            zorder = 3
                        else:
                            # No-contact segments: muted blue, thinner
                            color = "lightsteelblue"
                            linewidth = 2.0
                            alpha = 0.6
                            zorder = 1
                        axins.plot(
                            time[inset_mask][inset_start_idx:seg_end],
                            -y_ball[inset_mask][inset_start_idx:seg_end],
                            color=color,
                            linewidth=linewidth,
                            alpha=alpha,
                            zorder=zorder,
                        )
                        inset_start_idx = j
                        inset_current_contact = contact
                axins.axvspan(event_start_time, event_end_time, color="yellow", alpha=0.2)
                axins.set_xlim(zoom_start, zoom_end)
                axins.set_ylabel("")
                axins.set_xlabel("")
                axins.set_title("Zoom on major event", fontsize=9)
                axins.tick_params(axis="both", which="major", labelsize=8)
                # Save the plot with the inset in the fly directory as 'with_inset.png'
                if fly_dir is not None:
                    inset_plot_path = os.path.join(fly_dir, "with_inset.png")
                    plt.savefig(inset_plot_path, dpi=150)
            else:
                # Save zoomed plot as a separate file in the fly directory as 'zoom.png'
                fig_zoom, ax_zoom = plt.subplots(figsize=(8, 4))
                ax_zoom.plot(
                    time[inset_mask],
                    -y_head[inset_mask],
                    color="tab:orange",
                    linewidth=1,
                    alpha=0.4,
                    zorder=1,
                    label="Fly y_Head",
                )
                # Ball trace in separate zoom with improved styling
                inset_start_idx = None
                inset_current_contact = None
                for j, idx in enumerate(inset_indices):
                    contact = is_contact[idx]
                    if inset_start_idx is None:
                        inset_start_idx = j
                        inset_current_contact = contact
                    elif contact != inset_current_contact or j == len(inset_indices) - 1:
                        seg_end = j if j < len(inset_indices) - 1 else j + 1
                        if inset_current_contact:
                            # Contact segments: bright red, thick
                            color = "red"
                            linewidth = 4.0
                            alpha = 1.0
                            zorder = 3
                        else:
                            # No-contact segments: muted blue, thinner
                            color = "lightsteelblue"
                            linewidth = 2.0
                            alpha = 0.6
                            zorder = 1
                        ax_zoom.plot(
                            time[inset_mask][inset_start_idx:seg_end],
                            -y_ball[inset_mask][inset_start_idx:seg_end],
                            color=color,
                            linewidth=linewidth,
                            alpha=alpha,
                            zorder=zorder,
                            label=None,
                        )
                        inset_start_idx = j
                        inset_current_contact = contact
                ax_zoom.axvspan(event_start_time, event_end_time, color="yellow", alpha=0.2)
                ax_zoom.set_xlim(zoom_start, zoom_end)
                ax_zoom.set_xlabel("Time (s)")
                ax_zoom.set_ylabel("Y Position (inverted, image coordinates)")
                ax_zoom.set_title("Zoom on major event")
                ax_zoom.legend()
                plt.tight_layout()
                if fly_dir is not None:
                    zoom_path = os.path.join(fly_dir, "zoom.png")
                    fig_zoom.savefig(zoom_path, dpi=150)
                plt.close(fig_zoom)
        plt.savefig(output_path, dpi=150)
        plt.close()

    def contact_proportion(self):
        """
        Returns the proportion of frames with contact during the major event.

        Returns:
            float: Proportion of contact frames during major event (0.0 to 1.0),
                   or None if no major event found or no contact data available.
        """
        # Get the major event timing
        major_event_info = self._get_major_event_info()
        if not major_event_info["found"]:
            return None

        event_start_time = major_event_info["start_time"]
        event_end_time = major_event_info["end_time"]

        # Get annotated contact dataset
        annotated_df = self._get_annotated_contact_dataset()
        if annotated_df is None:
            return None

        # Get time array
        time = self._get_time_array(annotated_df)

        # Mask for major event time window
        event_mask = (time >= event_start_time) & (time <= event_end_time)

        # Get contact data
        is_contact = (
            annotated_df["is_contact"].values
            if "is_contact" in annotated_df.columns
            else np.zeros_like(time, dtype=bool)
        )

        # Calculate proportion of contact frames during major event
        event_frames = np.sum(event_mask)
        if event_frames == 0:
            return None

        contact_frames_in_event = np.sum(is_contact[event_mask])
        proportion = contact_frames_in_event / event_frames

        return proportion

    def major_event_length(self):
        """
        Returns the duration of the major event in seconds.

        Returns:
            float: Duration of major event in seconds, or None if no major event found.
        """
        major_event_info = self._get_major_event_info()
        if not major_event_info["found"]:
            return None

        event_start_time = major_event_info["start_time"]
        event_end_time = major_event_info["end_time"]

        if event_start_time is None or event_end_time is None:
            return None

        return event_end_time - event_start_time

    def ball_displacement_to_major_event(self):
        """
        Returns the total displacement of the ball from the beginning of the recording
        until the start of the major event.

        Returns:
            float: Ball displacement in pixels from start to major event,
                   or None if no major event found or no ball position data available.
        """
        # Get the major event timing
        major_event_info = self._get_major_event_info()
        if not major_event_info["found"]:
            return None

        event_start_time = major_event_info["start_time"]

        # Get annotated contact dataset
        annotated_df = self._get_annotated_contact_dataset()
        if annotated_df is None:
            return None

        # Check if required ball position columns exist
        if "y_centre_preprocessed" not in annotated_df.columns:
            return None

        # Get time array
        time = self._get_time_array(annotated_df)

        # Get ball y positions (assuming movement is primarily along y-axis for ball pushing)
        y_ball = annotated_df["y_centre_preprocessed"].values

        # Find the frame closest to the start of recording and major event start
        start_frame_idx = 0  # Beginning of recording
        event_start_idx = np.argmin(np.abs(time - event_start_time))

        # Calculate displacement from start to major event
        if event_start_idx <= start_frame_idx:
            return 0.0  # Major event starts at the beginning

        start_position = y_ball[start_frame_idx]
        event_start_position = y_ball[event_start_idx]

        # Calculate absolute displacement
        displacement = abs(event_start_position - start_position)

        return displacement

    def ball_displacement_during_major_event(self):
        """
        Returns the total displacement of the ball during the major event.

        Returns:
            float: Ball displacement in pixels during the major event,
                   or None if no major event found or no ball position data available.
        """
        # Get the major event timing
        major_event_info = self._get_major_event_info()
        if not major_event_info["found"]:
            return None

        event_start_time = major_event_info["start_time"]
        event_end_time = major_event_info["end_time"]

        if event_start_time is None or event_end_time is None:
            return None

        # Get annotated contact dataset
        annotated_df = self._get_annotated_contact_dataset()
        if annotated_df is None:
            return None

        # Check if required ball position columns exist
        if "y_centre_preprocessed" not in annotated_df.columns:
            return None

        # Get time array
        time = self._get_time_array(annotated_df)

        # Get ball y positions (assuming movement is primarily along y-axis for ball pushing)
        y_ball = annotated_df["y_centre_preprocessed"].values

        # Find frames corresponding to the major event window
        event_mask = (time >= event_start_time) & (time <= event_end_time)
        event_indices = np.where(event_mask)[0]

        if len(event_indices) < 2:
            return 0.0  # Event too short to measure displacement

        # Get ball positions during the event
        event_y_positions = y_ball[event_indices]

        # Calculate displacement as the absolute difference between start and end of event
        event_start_position = event_y_positions[0]
        event_end_position = event_y_positions[-1]

        displacement = abs(event_end_position - event_start_position)

        return displacement

    def _get_major_event_info(self):
        """
        Helper method to extract major event information from fly's event_metrics.

        Returns:
            dict: Dictionary with keys 'found', 'start_time', 'end_time', 'displacement'
        """
        config = getattr(self.fly, "config", None)
        major_push_thresh = getattr(config, "major_event_threshold", 20) if config is not None else 20

        result = {
            "found": False,
            "start_time": None,
            "end_time": None,
            "displacement": None
        }

        if not (hasattr(self.fly, "event_metrics") and self.fly.event_metrics is not None):
            return result

        event_metrics = self.fly.event_metrics
        # Flatten all events into a list
        all_events = []
        for k, v in event_metrics.items():
            if isinstance(v, dict):
                for ev in v.values():
                    all_events.append(ev)

        # Sort by start_time
        all_events = sorted(all_events, key=lambda e: e.get("start_time", 0))

        # Find first event with displacement >= threshold
        for ev in all_events:
            if ev.get("displacement", 0) >= major_push_thresh:
                result["found"] = True
                result["start_time"] = ev.get("start_time", None)
                result["end_time"] = ev.get("end_time", None)
                result["displacement"] = ev.get("displacement", None)
                break

        return result

    def _get_annotated_contact_dataset(self):
        """
        Helper method to get the annotated contact dataset.

        Returns:
            pandas.DataFrame or None: The annotated contact dataset if available.
        """
        if hasattr(self.fly, "skeleton_metrics") and hasattr(
            self.fly.skeleton_metrics, "get_contact_annotated_dataset"
        ):
            annotated_df = self.fly.skeleton_metrics.get_contact_annotated_dataset()
            if annotated_df is not None and not annotated_df.empty:
                return annotated_df
        return None

    def _get_time_array(self, annotated_df):
        """
        Helper method to extract time array from annotated dataset.

        Args:
            annotated_df: The annotated contact dataset

        Returns:
            numpy.ndarray: Time array in seconds
        """
        # Use time column if available, else fallback to frame/fps
        if "time" in annotated_df.columns:
            return annotated_df["time"].values
        elif (
            "frame" in annotated_df.columns and hasattr(self.fly, "experiment") and hasattr(self.fly.experiment, "fps")
        ):
            return annotated_df["frame"].values / float(self.fly.experiment.fps)
        else:
            # fallback: use index as frame, try to get fps
            fps = getattr(getattr(self.fly, "experiment", None), "fps", 30)
            return np.arange(len(annotated_df)) / float(fps)
