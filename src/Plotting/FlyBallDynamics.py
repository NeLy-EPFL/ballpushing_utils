import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from Ballpushing_utils.fly import Fly
from Ballpushing_utils.dataset import Dataset
from Ballpushing_utils.interactions_metrics import InteractionsMetrics

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import argparse


def plot_fly_interaction_backoff(fly_path, output_dir=None, save=False):
    try:
        fly = Fly(fly_path, as_individual=True)
        event_dataset = Dataset(fly, dataset_type="event_metrics")
        event_metrics = event_dataset.data
        if event_metrics is None or len(event_metrics) == 0:
            print(f"[WARN] No event metrics found for {fly_path}.")
            return
        # Use InteractionsMetrics to get rolling median data
        interactions_metrics = InteractionsMetrics(fly.tracking_data)
        fly_data, ball_data = interactions_metrics.get_rolling_median_data(0, 0)
        fps = fly.experiment.fps
        if fly_data is None or ball_data is None or len(fly_data) == 0 or len(ball_data) == 0:
            print(f"[WARN] No tracking data for {fly_path}.")
            return
        # Filter out rows with invalid or missing start_time before plotting
        if "start_time" not in event_metrics.columns:
            print(f"[WARN] 'start_time' column missing for {fly_path}.")
            return
        event_metrics = event_metrics[event_metrics["start_time"].notnull()]
        if len(event_metrics) == 0:
            print(f"[WARN] No valid events with 'start_time' for {fly_path}.")
            return

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

        # After plotting all subplots, set y and x axis limits
        # axes[0, 0] is time-based (percent completion vs backoff time)
        axes[0, 0].set_ylim(0, 350)
        if all_end_times:
            max_event_end = max(all_end_times)
            x_max = max_event_end + 300  # 5 min post final event
        else:
            x_max = 300  # fallback if no events
        axes[0, 0].set_xlim(0, x_max)
        # For event-index plots, set xlim based on number of unique event indices
        n_events = max(event_idxs_bv) + 1 if event_idxs_bv else 0
        for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
            ax.set_ylim(0, 350)
            if n_events > 0:
                ax.set_xlim(-0.5, n_events - 0.5)
        plt.tight_layout()
        if save:
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                fly_name = fly.metadata.name
                fig.savefig(os.path.join(output_dir, f"{fly_name}_summary.png"))
                plt.close(fig)
        else:
            plt.show()
        # Plot 5: Backoff time vs percent completion (scatter, all backoffs in all events)
        backoff_times = []
        percent_completions = []
        for idx, row in event_metrics.iterrows():
            if isinstance(row["backoff_intervals"], list):
                for b in row["backoff_intervals"]:
                    t = b.get("time_from_event_start", np.nan)
                    pc = b.get("percent_completion", np.nan)
                    backoff_times.append(t)
                    percent_completions.append(pc)
        if backoff_times and percent_completions:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.scatter(percent_completions, backoff_times, color="darkred", alpha=0.7)
            ax2.set_xlabel("Percent completion at backoff (fraction of event displacement)")
            ax2.set_ylabel("Time to backoff (s, rel. to event start)")
            ax2.set_title("Backoff time vs. percent completion (all backoffs, all events)")
            plt.tight_layout()
            if save:
                if output_dir is not None:
                    fig2.savefig(os.path.join(output_dir, f"{fly_name}_backoff_vs_completion.png"))
                    plt.close(fig2)
            else:
                plt.show()
    except Exception as e:
        print(f"[ERROR] Failed to plot for {fly_path}: {e}")


def plot_group_backoff_vs_completion(fly_dirs):
    """
    Plot backoff time vs percent completion for all backoff events from all flies in the list.
    """
    all_backoff_times = []
    all_percent_completions = []
    fly_labels = []
    for fly_dir in fly_dirs:
        try:
            fly = Fly(Path(fly_dir), as_individual=True)
            event_dataset = Dataset(fly, dataset_type="event_metrics")
            event_metrics = event_dataset.data
            if event_metrics is None or len(event_metrics) == 0:
                continue
            if "start_time" not in event_metrics.columns:
                continue
            event_metrics = event_metrics[event_metrics["start_time"].notnull()]
            for idx, row in event_metrics.iterrows():
                if isinstance(row["backoff_intervals"], list):
                    for b in row["backoff_intervals"]:
                        t = b.get("time_from_event_start", np.nan)
                        pc = b.get("percent_completion", np.nan)
                        all_backoff_times.append(t)
                        all_percent_completions.append(pc)
                        fly_labels.append(str(fly_dir))
        except Exception as e:
            print(f"[ERROR] Failed to process {fly_dir}: {e}")
    if all_backoff_times and all_percent_completions:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(all_percent_completions, all_backoff_times, c="tab:blue", alpha=0.6)
        ax.set_xlabel("Percent completion at backoff (fraction of max y ball displacement)")
        ax.set_ylabel("Time to backoff (s, rel. to event start)")
        ax.set_title("Backoff time vs. percent completion (all backoffs, all flies)")
        plt.tight_layout()
        plt.show()
    else:
        print("[WARN] No backoff events found for group plot.")


def plot_group_summary(fly_dirs, save=False, output_dir=None):
    """
    Group summary: 4 plots -
    1. Backoff time vs percent completion (all backoffs, all events, all flies)
    2. Backoff time vs event_idx (first backoff per event, all flies)
    3. Ball velocity vs event_idx (all events, all flies)
    4. Percent completion at first backoff vs event_idx (all events, all flies)
    """
    all_backoff_times = []
    all_percent_completions = []
    all_event_idxs = []
    all_ball_velocities = []
    all_first_backoff_times = []
    all_first_backoff_percent = []
    all_first_backoff_event_idxs = []
    # Collect all end times for axis scaling
    all_end_times = []
    for fly_dir in fly_dirs:
        try:
            fly = Fly(Path(fly_dir), as_individual=True)
            event_dataset = Dataset(fly, dataset_type="event_metrics")
            event_metrics = event_dataset.data
            if event_metrics is not None and len(event_metrics) > 0 and "end_time" in event_metrics.columns:
                all_end_times.extend(event_metrics["end_time"].dropna().tolist())
        except Exception as e:
            print(f"[WARN] Could not compute event_metrics for fly {fly_dir}: {e}")
    for fly_dir in fly_dirs:
        print(f"Processing fly: {fly_dir}")
        try:
            fly = Fly(Path(fly_dir), as_individual=True)
            event_dataset = Dataset(fly, dataset_type="event_metrics")
            event_metrics = event_dataset.data
            if event_metrics is None or len(event_metrics) == 0:
                print(f"[WARN] Skipping fly {fly_dir}: No valid event metrics.")
                continue
            if "start_time" not in event_metrics.columns:
                print(f"[WARN] Skipping fly {fly_dir}: No 'start_time' column.")
                continue
            event_metrics = event_metrics[event_metrics["start_time"].notnull()]
            for idx, row in event_metrics.iterrows():
                event_idx = row.get("event_idx", idx)
                # All backoffs for scatter
                if isinstance(row["backoff_intervals"], list):
                    for b in row["backoff_intervals"]:
                        t = b.get("time_from_event_start", np.nan)
                        pc = b.get("percent_completion", np.nan)
                        all_backoff_times.append(t)
                        all_percent_completions.append(pc)
                        all_event_idxs.append(event_idx)
                # First backoff per event for line/box plots
                if isinstance(row["backoff_intervals"], list) and len(row["backoff_intervals"]) > 0:
                    b = row["backoff_intervals"][0]
                    t = b.get("time_from_event_start", np.nan)
                    pc = b.get("percent_completion", np.nan)
                    all_first_backoff_times.append(t)
                    all_first_backoff_percent.append(pc)
                    all_first_backoff_event_idxs.append(event_idx)
                # Ball velocity (only if not nan)
                bv = row.get("ball_velocity", np.nan)
                if not np.isnan(bv):
                    all_ball_velocities.append((event_idx, bv))
        except Exception as e:
            print(f"[WARN] Could not compute event_metrics for fly {fly_dir}: {e}")
    # Unpack event_idx and ball_velocity for plotting
    event_idxs_bv = [x[0] for x in all_ball_velocities]
    ball_velocities = [x[1] for x in all_ball_velocities]
    import matplotlib.pyplot as plt

    # Group mode summary layout
    print("Plotting group summary for all flies...")
    if save and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    elif save:
        output_dir = "outputs/group_plots"
        os.makedirs(output_dir, exist_ok=True)
    # Save the 2x2 summary layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(all_percent_completions, all_backoff_times, color="tab:blue", alpha=0.6)
    axes[0, 0].set_xlabel("Percent completion at backoff (fraction of max y ball displacement)")
    axes[0, 0].set_ylabel("Time to backoff (s, rel. to event start)")
    axes[0, 0].set_title("Backoff time vs. percent completion\n(all backoffs, all events, all flies)")
    axes[0, 1].scatter(all_first_backoff_event_idxs, all_first_backoff_times, color="tab:green", alpha=0.6)
    axes[0, 1].set_xlabel("Event index")
    axes[0, 1].set_ylabel("Time to first backoff (s)")
    axes[0, 1].set_title("Time to first backoff vs. event index\n(first backoff per event, all flies)")
    axes[1, 0].scatter(event_idxs_bv, ball_velocities, color="tab:orange", alpha=0.6)
    axes[1, 0].set_xlabel("Event index")
    axes[1, 0].set_ylabel("Ball velocity (pixels/s)")
    axes[1, 0].set_title("Ball velocity vs. event index\n(all events, all flies)")
    bv_vs_pc = [
        (pc, bv)
        for pc, bv in zip(
            all_first_backoff_percent, [x[1] for x in all_ball_velocities if x[0] in all_first_backoff_event_idxs]
        )
        if not np.isnan(pc) and not np.isnan(bv)
    ]
    if bv_vs_pc:
        x_pc, y_bv = zip(*bv_vs_pc)
        axes[1, 1].scatter(x_pc, y_bv, color="tab:red", alpha=0.6)
    else:
        axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center")
    axes[1, 1].set_xlabel("Percent completion at first backoff")
    axes[1, 1].set_ylabel("Ball velocity (pixels/s)")
    axes[1, 1].set_title("Ball velocity vs. percent completion at first backoff\n(first backoff per event, all flies)")
    # After plotting all subplots, set y and x axis limits
    # axes[0, 0] is time-based (percent completion vs backoff time)
    axes[0, 0].set_ylim(0, 350)
    if all_end_times:
        max_event_end = max(all_end_times)
        x_max = max_event_end + 300  # 5 min post final event
    else:
        x_max = 300  # fallback if no events
    axes[0, 0].set_xlim(0, x_max)
    # For event-index plots, set xlim based on number of unique event indices
    n_events = max(event_idxs_bv) + 1 if event_idxs_bv else 0
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set_ylim(0, 350)
        if n_events > 0:
            ax.set_xlim(-0.5, n_events - 0.5)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "group_summary.png"))
        plt.close(fig)
    else:
        plt.show()
    # Jointplot: percent completion at first backoff vs ball velocity and time to first backoff
    import seaborn as sns
    import pandas as pd

    joint_data = []
    for pc, t, bv in zip(
        all_first_backoff_percent,
        all_first_backoff_times,
        [x[1] for x in all_ball_velocities if x[0] in all_first_backoff_event_idxs],
    ):
        if not np.isnan(pc) and not np.isnan(t) and not np.isnan(bv):
            joint_data.append({"percent_completion": pc, "time_to_first_backoff": t, "ball_velocity": bv})
    if joint_data:
        df_joint = pd.DataFrame(joint_data)
        # Ball velocity vs percent completion (scatter as before, now with log scale and density)
        jp1 = sns.jointplot(
            data=df_joint, x="percent_completion", y="ball_velocity", kind="scatter", color="tab:orange"
        )
        # Overlay KDE for density
        try:
            sns.kdeplot(
                data=df_joint,
                x="percent_completion",
                y="ball_velocity",
                ax=jp1.ax_joint,
                fill=True,
                cmap="Oranges",
                alpha=0.4,
                levels=10,
            )
        except Exception as e:
            print(f"[WARN] KDE overlay failed for velocity: {e}")
        jp1.ax_joint.set_yscale("log")
        jp1.ax_joint.set_ylabel("Ball velocity (pixels/s, log scale)")
        plt.suptitle("Ball velocity vs. percent completion at first backoff (all flies)\n(log scale, density)", y=1.02)
        plt.tight_layout()
        if save:
            jp1.savefig(os.path.join(output_dir, "jointplot_ball_velocity_vs_percent_completion.png"))
            plt.close(jp1.fig)
        else:
            plt.show()
        # Time to first backoff vs percent completion: log scale and density
        jp2 = sns.jointplot(
            data=df_joint, x="percent_completion", y="time_to_first_backoff", kind="scatter", color="tab:blue"
        )
        # Overlay KDE for density
        try:
            sns.kdeplot(
                data=df_joint,
                x="percent_completion",
                y="time_to_first_backoff",
                ax=jp2.ax_joint,
                fill=True,
                cmap="Blues",
                alpha=0.4,
                levels=10,
            )
        except Exception as e:
            print(f"[WARN] KDE overlay failed: {e}")
        jp2.ax_joint.set_yscale("log")
        jp2.ax_joint.set_ylabel("Time to first backoff (s, log scale)")
        plt.suptitle(
            "Time to first backoff vs. percent completion at first backoff (all flies)\n(log scale, density)", y=1.02
        )
        plt.tight_layout()
        if save:
            jp2.savefig(os.path.join(output_dir, "jointplot_time_to_backoff_vs_percent_completion.png"))
            plt.close(jp2.fig)
        else:
            plt.show()
        # Additional grouped jointplot: backoff time (log) vs event index (all backoffs, all flies)
        import seaborn as sns
        import pandas as pd

        if all_backoff_times and all_event_idxs:
            df_backoff_event = pd.DataFrame({"event_idx": all_event_idxs, "backoff_time": all_backoff_times})
            jp3 = sns.jointplot(
                data=df_backoff_event, x="event_idx", y="backoff_time", kind="scatter", color="tab:purple"
            )
            # Overlay KDE for density
            try:
                sns.kdeplot(
                    data=df_backoff_event,
                    x="event_idx",
                    y="backoff_time",
                    ax=jp3.ax_joint,
                    fill=True,
                    cmap="Purples",
                    alpha=0.4,
                    levels=10,
                )
            except Exception as e:
                print(f"[WARN] KDE overlay failed for backoff time vs event idx: {e}")
            jp3.ax_joint.set_yscale("log")
            jp3.ax_joint.set_ylabel("Time to backoff (s, log scale)")
            jp3.ax_joint.set_xlabel("Event index")
            plt.suptitle("Backoff time vs. event index (all backoffs, all flies)\n(log scale, density)", y=1.02)
            plt.tight_layout()
            if save:
                jp3.savefig(os.path.join(output_dir, "jointplot_backoff_time_vs_event_idx_log.png"))
                plt.close(jp3.fig)
            else:
                plt.show()
    else:
        print("[WARN] No valid data for jointplots.")
    # Additional grouped plot: backoff time (log scale) vs event index (all backoffs, all flies)
    if all_backoff_times and all_event_idxs:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(all_event_idxs, all_backoff_times, color="tab:purple", alpha=0.6)
        ax3.set_xlabel("Event index")
        ax3.set_ylabel("Time to backoff (s, log scale)")
        ax3.set_yscale("log")
        ax3.set_title("Backoff time vs. event index (all backoffs, all flies, log scale)")
        plt.tight_layout()
        if save:
            fig3.savefig(os.path.join(output_dir, "groupplot_backoff_time_vs_event_idx_log.png"))
            plt.close(fig3)
        else:
            plt.show()
    # Additional grouped boxplot: backoff time per event index (all backoffs, all flies)
    if all_backoff_times and all_event_idxs:
        import seaborn as sns
        import pandas as pd

        df_box = pd.DataFrame({"event_idx": all_event_idxs, "backoff_time": all_backoff_times})
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="event_idx", y="backoff_time", data=df_box, ax=ax4, color="lightgray", showfliers=False)
        sns.stripplot(
            x="event_idx", y="backoff_time", data=df_box, ax=ax4, color="tab:blue", alpha=0.5, jitter=True, size=4
        )
        ax4.set_yscale("log")
        ax4.set_xlabel("Event index")
        ax4.set_ylabel("Time to backoff (s, log scale)")
        ax4.set_title("Backoff time per event index (all backoffs, all flies, log scale)")
        plt.tight_layout()
        if save:
            fig4.savefig(os.path.join(output_dir, "boxplot_backoff_time_per_event_idx_log.png"))
            plt.close(fig4)
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ball-pushing experiment data for Drosophila.")
    parser.add_argument(
        "--yaml",
        type=str,
        default="/home/durrieu/ballpushing_utils/experiments_yaml/TNT_Genotypes/EmptySplit.yaml",
        help="Path to YAML file listing experiment directories.",
    )
    parser.add_argument(
        "--save", action="store_true", help="If set, save plots to disk instead of showing interactively."
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory for plots.")
    args = parser.parse_args()
    yaml_path = args.yaml
    save = args.save
    base_output_dir = args.output_dir
    with open(yaml_path, "r") as f:
        fly_dirs = yaml.safe_load(f).get("directories", [])
    # Group mode summary layout
    print("Plotting group summary for all flies...")
    group_dir = os.path.join(base_output_dir, "group_plots") if save else None
    plot_group_summary(fly_dirs, save=save, output_dir=group_dir)
    # Individual mode plots (save to .../individual_plots if save is True)
    indiv_dir = os.path.join(base_output_dir, "individual_plots") if save else None
    for fly_dir in fly_dirs:
        print(f"Plotting for {fly_dir}")
        plot_fly_interaction_backoff(Path(fly_dir), output_dir=indiv_dir, save=save)
