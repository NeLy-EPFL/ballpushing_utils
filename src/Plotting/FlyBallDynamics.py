import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from Ballpushing_utils.fly import Fly
from Ballpushing_utils.dataset import Dataset
from Ballpushing_utils.interactions_metrics import InteractionsMetrics

import matplotlib.pyplot as plt
import numpy as np


def plot_fly_interaction_backoff(fly_path):
    fly = Fly(fly_path, as_individual=True)
    event_dataset = Dataset(fly, dataset_type="event_metrics")
    event_metrics = event_dataset.data
    if event_metrics is None:
        print("No event metrics found.")
        return

    # Use InteractionsMetrics to get rolling median data
    interactions_metrics = InteractionsMetrics(fly.tracking_data)
    # For now, assume single fly and ball (index 0)
    fly_data, ball_data = interactions_metrics.get_rolling_median_data(0, 0)
    fps = fly.experiment.fps

    # Filter out rows with invalid or missing start_time before plotting
    event_metrics = event_metrics[event_metrics["start_time"].notnull()]

    # --- Create a layout with 1 plot per row ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 18))

    # Plot 1: Ball Y position relative to fly initial Y, event starts, and first backoff moments, plus fly trajectory
    all_times = []
    all_ball_y_rel = []
    event_start_times = []
    event_start_ball_y_rel = []
    backoff_times = []
    backoff_ball_y_rel = []
    fly_traj_y = []
    fly_traj_times = []

    # Get fly initial position from tracking data
    fly_initial_x = fly.tracking_data.start_x
    fly_initial_y = fly.tracking_data.start_y

    # Sample fly Y trajectory every 290 frames (about 10 seconds at 29 fps), relative to initial Y, use absolute value
    step = 290
    fly_y_med = fly_data["y_thorax_rolling_median"].values
    y0 = fly_initial_y if fly_initial_y is not None else (fly_y_med[0] if len(fly_y_med) > 0 else 0)
    n_frames = len(fly_y_med)
    for i in range(0, n_frames, step):
        fly_traj_y.append(abs(fly_y_med[i] - y0))
        fly_traj_times.append(i / fps)

    for idx, row in event_metrics.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]
        start_idx = int(np.round(start_time * fps))
        end_idx = int(np.round(end_time * fps))
        # Ball Y position relative to fly initial Y for this event (absolute value)
        ball_y = ball_data["y_centre_rolling_median"].iloc[start_idx : end_idx + 1].values
        ball_y_rel = np.abs(ball_y - y0)
        times = np.arange(start_idx, end_idx + 1) / fps
        all_times.extend(times)
        all_ball_y_rel.extend(ball_y_rel)
        # Use actual ball Y rel at event start for event start marker
        event_start_times.append(start_time)
        event_start_ball_y_rel.append(ball_y_rel[0] if len(ball_y_rel) > 0 else np.nan)
        # Mark only the first backoff moment
        if isinstance(row["backoff_intervals"], list) and len(row["backoff_intervals"]) > 0:
            interval = row["backoff_intervals"][0]
            t_backoff = start_time + interval["time_from_event_start"]
            frame_backoff = int(np.round(t_backoff * fps))
            idx_in_event = frame_backoff - start_idx
            # Robust bounds check for annotation
            if 0 <= idx_in_event < len(ball_y_rel):
                backoff_times.append(t_backoff)
                backoff_ball_y_rel.append(ball_y_rel[idx_in_event])
            elif idx_in_event >= len(ball_y_rel):
                # Clamp to last value if slightly out of bounds
                backoff_times.append(times[-1])
                backoff_ball_y_rel.append(ball_y_rel[-1])
        # Annotate final event with a special mark above the event start
        if "final_event" in row and row["final_event"] == 1:
            # Place a star above the event start marker
            axes[0].annotate(
                "★",
                (start_time, (ball_y_rel[0] if len(ball_y_rel) > 0 else 0) + 20),
                color="purple",
                fontsize=18,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Plot ball Y position relative to fly initial Y
    axes[0].plot(
        all_times, all_ball_y_rel, label="Ball Y rel. to fly initial Y (rolling median)", color="blue", alpha=0.7
    )
    # Plot fly Y trajectory (sampled, absolute relative to initial)
    axes[0].plot(fly_traj_times, fly_traj_y, label="Fly Y (|rolling median - start|)", color="orange", alpha=0.7)
    # Plot event and backoff markers
    axes[0].scatter(event_start_times, event_start_ball_y_rel, marker="x", color="red", label="Event start")
    axes[0].scatter(backoff_times, backoff_ball_y_rel, marker="o", color="green", label="First backoff moment")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Y position rel. to fly initial (pixels)")
    axes[0].set_title(
        "Ball Y position (rel. to fly initial) and fly trajectory over time with event starts and first backoff moments"
    )
    axes[0].legend()

    # Plot 2: Event index vs. time to first backoff
    times_to_first_backoff = []
    x_labels = []
    for idx, row in event_metrics.iterrows():
        if isinstance(row["backoff_intervals"], list) and len(row["backoff_intervals"]) > 0:
            # Use the time_from_event_start, which is already relative to event start
            times_to_first_backoff.append(row["backoff_intervals"][0]["time_from_event_start"])
        else:
            times_to_first_backoff.append(np.nan)
        # Add a star to the x label if this is the final event
        if "final_event" in row and row["final_event"] == 1:
            x_labels.append(f"{idx}★")
        else:
            x_labels.append(str(idx))
    axes[1].plot(range(len(times_to_first_backoff)), times_to_first_backoff, marker="o", linestyle="-")
    axes[1].set_xlabel("Event index")
    axes[1].set_ylabel("Time to first backoff (s) (relative to event start)")
    axes[1].set_title("Time from event start to first backoff per event")
    axes[1].set_xticks(range(len(x_labels)))
    axes[1].set_xticklabels(x_labels)

    # Plot 3: Number of backoffs per event
    n_backoffs = []
    for idx, row in event_metrics.iterrows():
        if isinstance(row["backoff_intervals"], list):
            n_backoffs.append(len(row["backoff_intervals"]))
        else:
            n_backoffs.append(0)
    axes[2].bar(range(len(n_backoffs)), n_backoffs, color="teal", alpha=0.7)
    axes[2].set_xlabel("Event index")
    axes[2].set_ylabel("Number of backoffs")
    axes[2].set_title("Number of backoff events per interaction event")

    # Plot 4: Ball displacement per event
    displacements = []
    for idx, row in event_metrics.iterrows():
        displacements.append(row["displacement"] if "displacement" in row else np.nan)
    axes[3].bar(range(len(displacements)), displacements, color="purple", alpha=0.7)
    axes[3].set_xlabel("Event index")
    axes[3].set_ylabel("Ball displacement (pixels)")
    axes[3].set_title("Ball displacement per event")

    plt.tight_layout()
    plt.show()


testfly = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5")
# Example usage:
plot_fly_interaction_backoff(testfly)
