#!/usr/bin/env python3
"""
Test script to investigate distance_moved vs max_distance discrepancy.

This script loads a fly, computes both metrics, and provides detailed analysis
of individual events to understand why max_distance might be higher than distance_moved.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from Ballpushing_utils.fly import Fly
from Ballpushing_utils.ballpushing_metrics import BallPushingMetrics
from utils_behavior import Processing


def analyze_distance_metrics(fly_path, output_dir="outputs/distance_metrics"):
    """
    Analyze distance metrics for a single fly.

    Parameters
    ----------
    fly_path : str
        Path to the fly directory
    output_dir : str
        Directory to save output plots
    """
    print("=" * 80)
    print("DISTANCE METRICS ANALYSIS")
    print("=" * 80)
    print(f"Fly path: {fly_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load fly
    print("\nLoading fly...")
    fly = Fly(fly_path, as_individual=True)

    # Create metrics object
    print("Computing metrics...")
    metrics = BallPushingMetrics(fly.tracking_data, compute_metrics_on_init=True)

    # Get tracking data
    tracking_data = fly.tracking_data

    # Analyze each fly-ball pair
    for fly_idx, ball_dict in tracking_data.interaction_events.items():
        for ball_idx, events in ball_dict.items():
            print(f"\n{'='*80}")
            print(f"Analyzing Fly {fly_idx}, Ball {ball_idx}")
            print(f"{'='*80}")

            if len(events) == 0:
                print("No interaction events found. Skipping.")
                continue

            # Get ball data
            ball_data = tracking_data.balltrack.objects[ball_idx].dataset

            # Compute metrics
            max_distance = metrics.get_max_distance(fly_idx, ball_idx)
            distance_moved = metrics.get_distance_moved(fly_idx, ball_idx)

            print(f"\n📊 SUMMARY METRICS:")
            print(f"  Max Distance:     {max_distance:.2f} px ({max_distance / (500/30):.2f} mm)")
            print(f"  Distance Moved:   {distance_moved:.2f} px ({distance_moved / (500/30):.2f} mm)")
            print(f"  Ratio (moved/max): {distance_moved/max_distance if max_distance > 0 else 0:.3f}")
            print(f"  Difference:       {max_distance - distance_moved:.2f} px")

            if max_distance > distance_moved:
                print(f"  ⚠️  MAX_DISTANCE > DISTANCE_MOVED by {max_distance - distance_moved:.2f} px")
            else:
                print(f"  ✓  Distance moved >= max distance (expected)")

            # Analyze individual events
            analyze_individual_events(fly_idx, ball_idx, events, ball_data, metrics, fly, output_path)

            # Plot trajectory and events
            plot_ball_trajectory(fly_idx, ball_idx, events, ball_data, metrics, fly, output_path)

            # Plot cumulative distance progression
            plot_cumulative_distance(fly_idx, ball_idx, events, ball_data, metrics, fly, output_path)


def analyze_individual_events(fly_idx, ball_idx, events, ball_data, metrics, fly, output_path):
    """Analyze each event individually."""
    print(f"\n📋 INDIVIDUAL EVENT ANALYSIS ({len(events)} events):")
    print(f"{'Event':<8} {'Start':<8} {'End':<8} {'Duration':<10} {'Frames':<8} {'Distance':<12} {'Cumulative':<12}")
    print("-" * 90)

    # Get initial position
    initial_x, initial_y, _, _ = metrics._calculate_median_coordinates(ball_data, start_idx=0, window=10)

    cumulative_distance = 0
    event_details = []

    for event_idx, event in enumerate(events):
        start_idx, end_idx = event[0], event[1]
        duration = (end_idx - start_idx) / fly.experiment.fps
        n_frames = end_idx - start_idx

        # Extract ball positions during this event
        event_x = ball_data["x_centre"].iloc[start_idx : end_idx + 1].values
        event_y = ball_data["y_centre"].iloc[start_idx : end_idx + 1].values

        # Calculate frame-to-frame distances
        dx = np.diff(event_x)
        dy = np.diff(event_y)
        frame_distances = np.sqrt(dx**2 + dy**2)
        event_distance = np.sum(frame_distances)

        cumulative_distance += event_distance

        # Get median positions at event boundaries
        start_x, start_y, end_x, end_y = metrics._calculate_median_coordinates(
            ball_data, start_idx=start_idx, end_idx=end_idx, window=10
        )

        # Calculate distance from initial position at end of event
        dist_from_initial = Processing.calculate_euclidian_distance(end_x, end_y, initial_x, initial_y)

        print(
            f"{event_idx:<8} {start_idx:<8} {end_idx:<8} {duration:<10.2f} "
            f"{n_frames:<8} {event_distance:<12.2f} {cumulative_distance:<12.2f}"
        )

        event_details.append(
            {
                "event_idx": event_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration": duration,
                "n_frames": n_frames,
                "event_distance": event_distance,
                "cumulative_distance": cumulative_distance,
                "dist_from_initial": dist_from_initial,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            }
        )

    print(f"\n  Total cumulative distance: {cumulative_distance:.2f} px")

    # Find event with maximum distance from initial
    max_dist_event = max(event_details, key=lambda x: x["dist_from_initial"])
    print(f"\n  Event reaching max distance from initial: Event {max_dist_event['event_idx']}")
    print(f"    Distance from initial: {max_dist_event['dist_from_initial']:.2f} px")

    return event_details


def plot_ball_trajectory(fly_idx, ball_idx, events, ball_data, metrics, fly, output_path):
    """Plot the ball trajectory with events highlighted."""
    print("\n📈 Generating ball trajectory plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get initial position
    initial_x, initial_y, _, _ = metrics._calculate_median_coordinates(ball_data, start_idx=0, window=10)

    # Plot 1: Full trajectory
    x_all = ball_data["x_centre"].values
    y_all = ball_data["y_centre"].values

    ax1.plot(x_all, y_all, "lightgray", alpha=0.3, linewidth=0.5, label="Full trajectory")
    ax1.plot(initial_x, initial_y, "go", markersize=10, label="Start", zorder=10)

    # Highlight events
    colors = plt.cm.viridis(np.linspace(0, 1, len(events)))
    for event_idx, event in enumerate(events):
        start_idx, end_idx = event[0], event[1]
        x_event = ball_data["x_centre"].iloc[start_idx : end_idx + 1].values
        y_event = ball_data["y_centre"].iloc[start_idx : end_idx + 1].values

        ax1.plot(
            x_event,
            y_event,
            color=colors[event_idx],
            linewidth=2,
            alpha=0.7,
            label=f"Event {event_idx}" if event_idx < 5 else None,
        )

        # Mark event boundaries
        start_x, start_y, end_x, end_y = metrics._calculate_median_coordinates(
            ball_data, start_idx=start_idx, end_idx=end_idx, window=10
        )
        ax1.plot(start_x, start_y, "o", color=colors[event_idx], markersize=6)
        ax1.plot(end_x, end_y, "s", color=colors[event_idx], markersize=6)

    ax1.set_xlabel("X position (px)")
    ax1.set_ylabel("Y position (px)")
    ax1.set_title(f"Ball Trajectory - Fly {fly_idx}, Ball {ball_idx}")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Plot 2: Distance from initial position over time
    distances_from_initial = Processing.calculate_euclidian_distance(
        ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
    )

    frames = np.arange(len(ball_data))
    time = frames / fly.experiment.fps

    ax2.plot(time, distances_from_initial, "k-", linewidth=0.5, alpha=0.5)

    # Highlight events
    for event_idx, event in enumerate(events):
        start_idx, end_idx = event[0], event[1]
        event_time = frames[start_idx : end_idx + 1] / fly.experiment.fps
        event_dist = distances_from_initial.iloc[start_idx : end_idx + 1].values
        ax2.plot(event_time, event_dist, color=colors[event_idx], linewidth=2, alpha=0.7)

    # Mark max distance
    max_dist_idx = distances_from_initial.idxmax()
    max_dist_time = frames[max_dist_idx] / fly.experiment.fps
    max_dist_val = distances_from_initial.iloc[max_dist_idx]
    ax2.axhline(max_dist_val, color="r", linestyle="--", linewidth=1, label=f"Max distance: {max_dist_val:.2f} px")
    ax2.plot(max_dist_time, max_dist_val, "r*", markersize=15, zorder=10)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance from initial position (px)")
    ax2.set_title("Distance from Initial Position Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_path / f"trajectory_fly{fly_idx}_ball{ball_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def plot_cumulative_distance(fly_idx, ball_idx, events, ball_data, metrics, fly, output_path):
    """Plot cumulative distance moved vs max distance progression."""
    print("\n📈 Generating cumulative distance plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Get initial position
    initial_x, initial_y, _, _ = metrics._calculate_median_coordinates(ball_data, start_idx=0, window=10)

    # Calculate cumulative distance and max distance for each event
    cumulative_distances = []
    max_distances_at_event = []
    event_times = []

    cumulative = 0

    for event_idx, event in enumerate(events):
        start_idx, end_idx = event[0], event[1]

        # Extract ball positions during this event
        event_x = ball_data["x_centre"].iloc[start_idx : end_idx + 1].values
        event_y = ball_data["y_centre"].iloc[start_idx : end_idx + 1].values

        # Calculate frame-to-frame distances
        dx = np.diff(event_x)
        dy = np.diff(event_y)
        frame_distances = np.sqrt(dx**2 + dy**2)
        event_distance = np.sum(frame_distances)

        cumulative += event_distance

        # Get median position at end of event
        _, _, end_x, end_y = metrics._calculate_median_coordinates(
            ball_data, start_idx=start_idx, end_idx=end_idx, window=10
        )

        # Calculate distance from initial at end of this event
        dist_from_initial = Processing.calculate_euclidian_distance(end_x, end_y, initial_x, initial_y)

        event_time = end_idx / fly.experiment.fps

        cumulative_distances.append(cumulative)
        max_distances_at_event.append(dist_from_initial)
        event_times.append(event_time)

    # Plot 1: Cumulative distance vs max distance over events
    event_numbers = np.arange(len(events))

    ax1.plot(event_numbers, cumulative_distances, "b-o", linewidth=2, markersize=6, label="Cumulative distance moved")
    ax1.plot(
        event_numbers,
        max_distances_at_event,
        "r-s",
        linewidth=2,
        markersize=6,
        label="Distance from initial (at event end)",
    )

    # Get final max distance
    final_max = metrics.get_max_distance(fly_idx, ball_idx)
    ax1.axhline(
        final_max, color="r", linestyle="--", linewidth=1, alpha=0.5, label=f"Final max distance: {final_max:.2f} px"
    )

    ax1.set_xlabel("Event Number")
    ax1.set_ylabel("Distance (px)")
    ax1.set_title("Cumulative Distance Moved vs Distance from Initial")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Individual event distances
    event_distances_only = []
    for i in range(len(cumulative_distances)):
        if i == 0:
            event_distances_only.append(cumulative_distances[0])
        else:
            event_distances_only.append(cumulative_distances[i] - cumulative_distances[i - 1])

    colors = ["green" if d > 0 else "red" for d in event_distances_only]
    bars = ax2.bar(event_numbers, event_distances_only, color=colors, alpha=0.6, edgecolor="black")

    ax2.set_xlabel("Event Number")
    ax2.set_ylabel("Event Distance (px)")
    ax2.set_title("Individual Event Distances (frame-to-frame sum)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0, color="black", linewidth=0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, event_distances_only)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=8,
            rotation=0,
        )

    plt.tight_layout()

    output_file = output_path / f"cumulative_distance_fly{fly_idx}_ball{ball_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()

    # Print summary
    print(f"\n  Final cumulative distance: {cumulative_distances[-1]:.2f} px")
    print(f"  Final max distance:        {final_max:.2f} px")
    print(f"  Difference:                {final_max - cumulative_distances[-1]:.2f} px")


def main():
    parser = argparse.ArgumentParser(description="Analyze distance metrics for a fly")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5",
        help="Path to fly directory",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/distance_metrics", help="Output directory for plots")

    args = parser.parse_args()

    analyze_distance_metrics(args.path, args.output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
