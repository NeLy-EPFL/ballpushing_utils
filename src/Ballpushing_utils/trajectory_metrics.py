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

    def plot_y_positions_over_time(self, output_path, zoom_major=True):
        """
        Plots y_centre_preprocessed of the ball and y_Head of the fly over time from the annotated contact dataset and saves the plot.
        Args:
            output_path (str): Path to save the plot (e.g., 'output.png')
            zoom_major (bool): If True, show an inset zoom around the major event. If False, only show the main plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

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
        # Plot ball trace: thick, foreground, blue (no contact), red (contact)
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
                color = "red" if main_current_contact else "tab:blue"
                ax.plot(
                    time[mask][main_start_idx:seg_end],
                    -y_ball[mask][main_start_idx:seg_end],
                    color=color,
                    linewidth=3.5,
                    alpha=1.0,
                    zorder=2,
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
        # Highlight the major event window on the main plot
        if found_major_event and event_start_time is not None and event_end_time is not None:
            ax.axvspan(event_start_time, event_end_time, color="yellow", alpha=0.2, label="Major event window")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y Position (inverted, image coordinates)")
        ax.set_title("Y Positions Over Time (Ball & Fly, Intuitive Y)\nBall: red=in contact, blue=not in contact")
        ax.legend()
        plt.tight_layout()
        # Add inset zoom around the major event (±10s) if zoom_major is True
        if zoom_major and found_major_event and event_start_time is not None and event_end_time is not None:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            zoom_start = max(time[0], event_start_time - 10)
            zoom_end = min(time[-1], event_end_time + 10)
            inset_mask = (time >= zoom_start) & (time <= zoom_end)
            inset_indices = np.where(inset_mask)[0]
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
            # Ball trace in inset
            inset_start_idx = None
            inset_current_contact = None
            for j, idx in enumerate(inset_indices):
                contact = is_contact[idx]
                if inset_start_idx is None:
                    inset_start_idx = j
                    inset_current_contact = contact
                elif contact != inset_current_contact or j == len(inset_indices) - 1:
                    seg_end = j if j < len(inset_indices) - 1 else j + 1
                    color = "red" if inset_current_contact else "tab:blue"
                    axins.plot(
                        time[inset_mask][inset_start_idx:seg_end],
                        -y_ball[inset_mask][inset_start_idx:seg_end],
                        color=color,
                        linewidth=3.5,
                        alpha=1.0,
                        zorder=2,
                    )
                    inset_start_idx = j
                    inset_current_contact = contact
            axins.axvspan(event_start_time, event_end_time, color="yellow", alpha=0.2)
            axins.set_xlim(zoom_start, zoom_end)
            axins.set_ylabel("")
            axins.set_xlabel("")
            axins.set_title("Zoom on major event", fontsize=9)
            axins.tick_params(axis="both", which="major", labelsize=8)
        plt.savefig(output_path, dpi=150)
        plt.close()
