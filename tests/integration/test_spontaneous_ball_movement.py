#!/usr/bin/env python3
"""
Test script for spontaneous ball movement detection.
This demonstrates the new functionality that checks for ball movement outside of interaction events.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from Ballpushing_utils import Experiment, Fly, Config


def test_spontaneous_movement_detection():
    """Test the spontaneous ball movement detection on an experiment."""

    print("=" * 80)
    print("TESTING SPONTANEOUS BALL MOVEMENT DETECTION")
    print("=" * 80)
    print()

    # Test with a regular TNT experiment
    print("Testing with regular TNT experiment...")
    print("-" * 80)

    experiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_1_Videos_Tracked")

    if not experiment_path.exists():
        print(f"❌ Experiment path does not exist: {experiment_path}")
        print("Please update the path to a valid experiment directory.")
        return

    # Create custom config with spontaneous movement checking enabled
    config = Config()
    config.check_spontaneous_ball_movement = True
    config.spontaneous_movement_threshold = 10.0  # 10 pixels
    config.debugging = False  # Set to True for more verbose output

    print(f"\nConfiguration:")
    print(f"  check_spontaneous_ball_movement: {config.check_spontaneous_ball_movement}")
    print(f"  spontaneous_movement_threshold: {config.spontaneous_movement_threshold} pixels")
    print(f"  spontaneous_movement_window: {config.spontaneous_movement_window} frames")
    print()

    # Load experiment
    print(f"Loading experiment from: {experiment_path.name}")
    experiment = Experiment(experiment_path, custom_config=config)

    if not experiment.flies:
        print("❌ No flies found in experiment")
        return

    print(f"Found {len(experiment.flies)} flies in experiment")
    print()

    # Test first few flies
    num_to_test = min(5, len(experiment.flies))
    print(f"Testing spontaneous movement detection on first {num_to_test} flies:")
    print("=" * 80)
    print()

    flies_with_spontaneous_movement = []

    for i, fly in enumerate(experiment.flies[:num_to_test]):
        print(f"\n[{i+1}/{num_to_test}] Fly: {fly.metadata.name}")
        print("-" * 60)

        if not fly.tracking_data or not fly.tracking_data.valid_data:
            print("  ❌ Invalid tracking data, skipping...")
            continue

        # Get spontaneous movement summary for each ball
        num_balls = len(fly.tracking_data.balltrack.objects) if fly.tracking_data.balltrack else 0
        print(f"  Number of balls: {num_balls}")

        for ball_idx in range(num_balls):
            ball_identity = fly.tracking_data.get_ball_identity(ball_idx)
            ball_name = ball_identity if ball_identity else f"ball_{ball_idx}"

            summary = fly.tracking_data.get_spontaneous_movement_summary(ball_idx)

            if summary and summary["num_spontaneous_frames"] > 0:
                print(f"\n  ⚠️  {ball_name.upper()} - Spontaneous Movement Detected:")
                print(f"     Frames with movement: {summary['num_spontaneous_frames']}")
                print(f"     Total displacement: {summary['total_displacement']:.1f} pixels")
                print(f"     Max single-frame displacement: {summary['max_displacement']:.1f} pixels")
                print(
                    f"     Max consecutive frames: {summary['max_consecutive_frames']} ({summary['max_consecutive_frames']/experiment.fps:.1f}s)"
                )

                if summary["spontaneous_movement_times"]:
                    example_times = summary["spontaneous_movement_times"][:3]
                    print(f"     Example times: {', '.join([f'{t:.1f}s' for t in example_times])}", end="")
                    if len(summary["spontaneous_movement_times"]) > 3:
                        print(f" (and {len(summary['spontaneous_movement_times']) - 3} more)")
                    else:
                        print()

                flies_with_spontaneous_movement.append((fly.metadata.name, ball_name, summary))
            else:
                print(f"  ✅ {ball_name}: No spontaneous movement detected")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Flies tested: {num_to_test}")
    print(f"Flies with spontaneous ball movement: {len(flies_with_spontaneous_movement)}")

    if flies_with_spontaneous_movement:
        print("\nFlies flagged for review:")
        for fly_name, ball_name, summary in flies_with_spontaneous_movement:
            print(
                f"  • {fly_name} ({ball_name}): {summary['num_spontaneous_frames']} frames, "
                f"{summary['total_displacement']:.1f}px total displacement"
            )
    else:
        print("\n✅ No spontaneous ball movement detected in tested flies")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


def test_f1_experiment():
    """Test spontaneous movement detection on F1 experiment."""

    print("\n" + "=" * 80)
    print("TESTING F1 EXPERIMENT")
    print("=" * 80)
    print()

    f1_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/241014_F1_Pretrained_1_Videos_Tracked")

    if not f1_path.exists():
        print(f"❌ F1 experiment path does not exist: {f1_path}")
        return

    config = Config()
    config.experiment_type = "F1"
    config.check_spontaneous_ball_movement = True
    config.spontaneous_movement_threshold = 10.0

    print(f"Loading F1 experiment from: {f1_path.name}")
    experiment = Experiment(f1_path, custom_config=config)

    if not experiment.flies:
        print("❌ No flies found in F1 experiment")
        return

    print(f"Found {len(experiment.flies)} flies")
    print()

    # Test first fly
    fly = experiment.flies[0]
    print(f"Testing fly: {fly.metadata.name}")

    if not fly.tracking_data or not fly.tracking_data.valid_data:
        print("  ❌ Invalid tracking data")
        return

    num_balls = len(fly.tracking_data.balltrack.objects) if fly.tracking_data.balltrack else 0
    print(f"Number of balls: {num_balls}")

    # Check each ball
    for ball_idx in range(num_balls):
        ball_identity = fly.tracking_data.get_ball_identity(ball_idx)
        print(f"\n  Ball {ball_idx} ({ball_identity}):")

        summary = fly.tracking_data.get_spontaneous_movement_summary(ball_idx)

        if summary and summary["num_spontaneous_frames"] > 0:
            print(f"    ⚠️  Spontaneous movement detected")
            print(f"    Frames: {summary['num_spontaneous_frames']}")
            print(f"    Total displacement: {summary['total_displacement']:.1f} pixels")
        else:
            print(f"    ✅ No spontaneous movement")


if __name__ == "__main__":
    # Run tests
    test_spontaneous_movement_detection()

    # Uncomment to test F1 experiments
    # test_f1_experiment()

    print("\n✅ All tests completed!")
