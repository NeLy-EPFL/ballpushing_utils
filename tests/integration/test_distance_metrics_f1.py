#!/usr/bin/env python3
"""
F1 Distance Metrics Test Script

Test script to investigate distance_moved vs max_distance discrepancy for F1 experiments.
This handles both training and test balls, analyzing each separately and comparing them.

For F1 experiments, flies interact with a training ball first, then a test ball.
This script analyzes distance metrics for both balls to understand:
- Whether distance_moved properly captures cumulative movement
- Why max_distance might be higher than distance_moved
- Differences in behavior between training and test balls
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


def analyze_f1_distance_metrics(fly_path, output_dir="outputs/distance_metrics_f1"):
    """
    Analyze distance metrics for an F1 fly (training and test balls).

    Parameters
    ----------
    fly_path : str
        Path to the F1 fly directory
    output_dir : str
        Directory to save output plots
    """
    print("=" * 80)
    print("F1 DISTANCE METRICS ANALYSIS")
    print("=" * 80)
    print(f"Fly path: {fly_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load fly
    print("\nLoading F1 fly...")
    fly = Fly(fly_path, as_individual=True)

    # Assign ball identities for F1 experiment
    if hasattr(fly.tracking_data, "assign_ball_identities"):
        fly.tracking_data.assign_ball_identities()
        print("✅ Ball identities assigned")
        print(f"   Ball identities: {fly.tracking_data.ball_identities}")
    else:
        print("⚠️  Could not assign ball identities - this may not be an F1 experiment")
        return

    # Create metrics object
    print("Computing metrics...")
    metrics = BallPushingMetrics(fly.tracking_data, compute_metrics_on_init=True)

    # Get tracking data
    tracking_data = fly.tracking_data

    # Check which balls are available
    has_training = tracking_data.has_training_ball() if hasattr(tracking_data, "has_training_ball") else False
    has_test = tracking_data.has_test_ball() if hasattr(tracking_data, "has_test_ball") else False

    print(f"\n🎾 Ball Availability:")
    print(f"   Training ball: {'✅ Available' if has_training else '❌ Not found'}")
    print(f"   Test ball: {'✅ Available' if has_test else '❌ Not found'}")

    if not (has_training or has_test):
        print("\n⚠️  No training or test balls found. Cannot proceed with F1 analysis.")
        return

    # Store results for comparison
    ball_results = {}

    # Analyze each fly-ball pair
    for fly_idx, ball_dict in tracking_data.interaction_events.items():
        for ball_idx, events in ball_dict.items():
            # Get ball identity
            ball_identity = tracking_data.ball_identities.get(ball_idx, f"ball_{ball_idx}")

            print(f"\n{'='*80}")
            print(f"Analyzing Fly {fly_idx}, Ball {ball_idx} ({ball_identity})")
            print(f"{'='*80}")

            if len(events) == 0:
                print("No interaction events found. Skipping.")
                continue

            # Get ball data
            ball_data = tracking_data.balltrack.objects[ball_idx].dataset

            # Compute metrics
            max_distance = metrics.get_max_distance(fly_idx, ball_idx)
            distance_moved = metrics.get_distance_moved(fly_idx, ball_idx)

            print(f"\n📊 SUMMARY METRICS ({ball_identity.upper()}):")
            print(f"  Max Distance:     {max_distance:.2f} px ({max_distance / (500/30):.2f} mm)")
            print(f"  Distance Moved:   {distance_moved:.2f} px ({distance_moved / (500/30):.2f} mm)")
            print(f"  Ratio (moved/max): {distance_moved/max_distance if max_distance > 0 else 0:.3f}")
            print(f"  Difference:       {max_distance - distance_moved:.2f} px")

            if max_distance > distance_moved:
                print(f"  ⚠️  MAX_DISTANCE > DISTANCE_MOVED by {max_distance - distance_moved:.2f} px")
            else:
                print(f"  ✓  Distance moved >= max distance (expected)")

            # Store results for comparison
            ball_results[ball_identity] = {
                "ball_idx": ball_idx,
                "max_distance": max_distance,
                "distance_moved": distance_moved,
                "ratio": distance_moved / max_distance if max_distance > 0 else 0,
                "difference": max_distance - distance_moved,
                "events": events,
                "ball_data": ball_data,
            }

            # Analyze individual events
            event_details = analyze_individual_events(
                fly_idx, ball_idx, ball_identity, events, ball_data, metrics, fly, output_path
            )
            ball_results[ball_identity]["event_details"] = event_details

            # Plot trajectory and events
            plot_ball_trajectory(fly_idx, ball_idx, ball_identity, events, ball_data, metrics, fly, output_path)

            # Plot cumulative distance progression
            plot_cumulative_distance(fly_idx, ball_idx, ball_identity, events, ball_data, metrics, fly, output_path)

    # Compare training vs test ball if both are available
    if "training" in ball_results and "test" in ball_results:
        compare_training_vs_test(ball_results, fly, output_path)


def analyze_individual_events(fly_idx, ball_idx, ball_identity, events, ball_data, metrics, fly, output_path):
    """Analyze each event individually."""
    print(f"\n📋 INDIVIDUAL EVENT ANALYSIS - {ball_identity.upper()} ({len(events)} events):")
    print(f"{'Event':<8} {'Start':<8} {'End':<8} {'Duration':<10} {'Frames':<8} {'Distance':<12} {'Cumulative':<12}")
    print("-" * 90)

    # Get initial position from FIRST EVENT (not frame 0)
    # This matches the corrected get_max_distance implementation
    if len(events) == 0:
        return []

    first_event = events[0]
    initial_x, initial_y, _, _ = metrics._calculate_median_coordinates(ball_data, start_idx=first_event[0], window=10)

    print(f"  Initial position (from first event frame {first_event[0]}): ({initial_x:.2f}, {initial_y:.2f})")

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

        if event_idx < 3:  # Debug first few events
            print(f"    DEBUG: End position: ({end_x:.2f}, {end_y:.2f}), Dist from initial: {dist_from_initial:.2f} px")

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


def plot_ball_trajectory(fly_idx, ball_idx, ball_identity, events, ball_data, metrics, fly, output_path):
    """Plot the ball trajectory with events highlighted."""
    print(f"\n📈 Generating ball trajectory plot for {ball_identity}...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get initial position from FIRST EVENT (not frame 0)
    # This matches the corrected get_max_distance implementation
    if len(events) == 0:
        plt.close()
        return

    first_event = events[0]
    initial_x, initial_y, _, _ = metrics._calculate_median_coordinates(ball_data, start_idx=first_event[0], window=10)

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
    ax1.set_title(f"Ball Trajectory - {ball_identity.upper()} - Fly {fly_idx}")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Plot 2: Distance from initial position over time
    # NOTE: In F1 experiments with ball identity, ball_data contains all frames
    # but only frames during events correspond to this specific ball identity
    distances_from_initial = Processing.calculate_euclidian_distance(
        ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
    )

    frames = np.arange(len(ball_data))
    time = frames / fly.experiment.fps

    # Plot full trajectory in light gray (may include other ball)
    ax2.plot(time, distances_from_initial, "k-", linewidth=0.5, alpha=0.2, label="All frames (may include other ball)")

    # Highlight events (only these frames are guaranteed to be this ball)
    for event_idx, event in enumerate(events):
        start_idx, end_idx = event[0], event[1]
        event_time = frames[start_idx : end_idx + 1] / fly.experiment.fps
        event_dist = distances_from_initial.iloc[start_idx : end_idx + 1].values
        ax2.plot(event_time, event_dist, color=colors[event_idx], linewidth=2, alpha=0.7)

    # Mark max distance FROM EVENTS ONLY
    max_dist_from_events = 0
    max_dist_time_from_events = 0
    for event in events:
        start_idx, end_idx = event[0], event[1]
        event_distances = distances_from_initial.iloc[start_idx : end_idx + 1]
        event_max = event_distances.max()
        if event_max > max_dist_from_events:
            max_dist_from_events = event_max
            max_dist_time_from_events = event_distances.idxmax() / fly.experiment.fps

    ax2.axhline(
        max_dist_from_events,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"Max distance (events only): {max_dist_from_events:.2f} px",
    )
    ax2.plot(max_dist_time_from_events, max_dist_from_events, "r*", markersize=15, zorder=10)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance from initial position (px)")
    ax2.set_title(f"Distance from Initial Position - {ball_identity.upper()}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_path / f"trajectory_{ball_identity}_fly{fly_idx}_ball{ball_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def plot_cumulative_distance(fly_idx, ball_idx, ball_identity, events, ball_data, metrics, fly, output_path):
    """Plot cumulative distance moved vs max distance progression."""
    print(f"\n📈 Generating cumulative distance plot for {ball_identity}...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Get initial position from FIRST EVENT (not frame 0)
    # This matches the corrected get_max_distance implementation
    if len(events) == 0:
        plt.close()
        return

    first_event = events[0]
    initial_x, initial_y, _, _ = metrics._calculate_median_coordinates(ball_data, start_idx=first_event[0], window=10)

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
    ax1.set_title(f"Cumulative Distance vs Distance from Initial - {ball_identity.upper()}")
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
    ax2.set_title(f"Individual Event Distances - {ball_identity.upper()} (frame-to-frame sum)")
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

    output_file = output_path / f"cumulative_distance_{ball_identity}_fly{fly_idx}_ball{ball_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()

    # Print summary
    print(f"\n  Final cumulative distance: {cumulative_distances[-1]:.2f} px")
    print(f"  Final max distance:        {final_max:.2f} px")
    print(f"  Difference:                {final_max - cumulative_distances[-1]:.2f} px")


def compare_training_vs_test(ball_results, fly, output_path):
    """Compare training ball vs test ball metrics and behaviors."""
    print(f"\n{'='*80}")
    print(f"🔄 TRAINING vs TEST BALL COMPARISON")
    print(f"{'='*80}")

    training = ball_results["training"]
    test = ball_results["test"]

    # Print comparison table
    print(f"\n📊 METRIC COMPARISON:")
    print(f"{'Metric':<30} {'Training':<20} {'Test':<20} {'Difference':<15}")
    print("-" * 85)

    print(
        f"{'Max Distance (px)':<30} {training['max_distance']:<20.2f} {test['max_distance']:<20.2f} "
        f"{test['max_distance'] - training['max_distance']:<15.2f}"
    )

    print(
        f"{'Distance Moved (px)':<30} {training['distance_moved']:<20.2f} {test['distance_moved']:<20.2f} "
        f"{test['distance_moved'] - training['distance_moved']:<15.2f}"
    )

    print(
        f"{'Ratio (moved/max)':<30} {training['ratio']:<20.3f} {test['ratio']:<20.3f} "
        f"{test['ratio'] - training['ratio']:<15.3f}"
    )

    print(
        f"{'Max - Moved (px)':<30} {training['difference']:<20.2f} {test['difference']:<20.2f} "
        f"{test['difference'] - training['difference']:<15.2f}"
    )

    print(
        f"{'Number of Events':<30} {len(training['events']):<20} {len(test['events']):<20} "
        f"{len(test['events']) - len(training['events']):<15}"
    )

    # Create comparison plots
    print(f"\n📈 Generating comparison plots...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Bar chart of key metrics
    metrics_names = ["Max Distance", "Distance Moved", "Max - Moved"]
    training_vals = [training["max_distance"], training["distance_moved"], training["difference"]]
    test_vals = [test["max_distance"], test["distance_moved"], test["difference"]]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax1.bar(x - width / 2, training_vals, width, label="Training", color="steelblue", alpha=0.8)
    ax1.bar(x + width / 2, test_vals, width, label="Test", color="orange", alpha=0.8)

    ax1.set_xlabel("Metric")
    ax1.set_ylabel("Distance (px)")
    ax1.set_title("Training vs Test: Distance Metrics Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Ratio comparison
    ax2.bar(
        ["Training", "Test"],
        [training["ratio"], test["ratio"]],
        color=["steelblue", "orange"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.axhline(1.0, color="r", linestyle="--", linewidth=1, label="Expected (moved = max)")
    ax2.set_ylabel("Ratio (Distance Moved / Max Distance)")
    ax2.set_title("Ratio Comparison (should be ≥ 1.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Event count comparison
    ax3.bar(
        ["Training", "Test"],
        [len(training["events"]), len(test["events"])],
        color=["steelblue", "orange"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax3.set_ylabel("Number of Events")
    ax3.set_title("Event Count Comparison")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Cumulative distance progression comparison
    if "event_details" in training and "event_details" in test:
        training_cum = [e["cumulative_distance"] for e in training["event_details"]]
        test_cum = [e["cumulative_distance"] for e in test["event_details"]]

        ax4.plot(
            range(len(training_cum)),
            training_cum,
            "o-",
            color="steelblue",
            linewidth=2,
            markersize=6,
            label="Training",
            alpha=0.8,
        )
        ax4.plot(
            range(len(test_cum)), test_cum, "s-", color="orange", linewidth=2, markersize=6, label="Test", alpha=0.8
        )

        ax4.set_xlabel("Event Number")
        ax4.set_ylabel("Cumulative Distance (px)")
        ax4.set_title("Cumulative Distance Progression")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_path / "training_vs_test_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()

    # Print interpretation
    print(f"\n🎯 INTERPRETATION:")
    if training["ratio"] < 1.0 or test["ratio"] < 1.0:
        print(f"  ⚠️  At least one ball has ratio < 1.0 (max_distance > distance_moved)")
        print(f"      This suggests cumulative frame-to-frame distance may be underestimating movement")
    else:
        print(f"  ✅ Both balls have ratio >= 1.0 (expected behavior)")

    if test["distance_moved"] > training["distance_moved"]:
        print(f"  📈 Test ball shows MORE movement than training ball")
        print(f"      This could indicate learning/improvement")
    elif test["distance_moved"] < training["distance_moved"]:
        print(f"  📉 Test ball shows LESS movement than training ball")
        print(f"      This could indicate different strategies or conditions")
    else:
        print(f"  ➡️  Test and training balls show similar movement")


def main():
    parser = argparse.ArgumentParser(description="Analyze F1 distance metrics (training and test balls)")
    parser.add_argument(
        "--path",
        type=str,
        default="/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena5/Right",
        help="Path to F1 fly directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/distance_metrics_f1", help="Output directory for plots"
    )

    args = parser.parse_args()

    analyze_f1_distance_metrics(args.path, args.output_dir)

    print("\n" + "=" * 80)
    print("F1 Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
