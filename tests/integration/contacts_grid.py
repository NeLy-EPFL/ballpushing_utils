import sys
from pathlib import Path
from Ballpushing_utils.fly import Fly
from Ballpushing_utils.skeleton_metrics import SkeletonMetrics

if __name__ == "__main__":
    # Example usage: python contacts_grid.py --fly_dir /path/to/fly_dir --output /tmp/contacts_grid.mp4
    import argparse

    parser = argparse.ArgumentParser(description="Test contacts grid video generation.")
    parser.add_argument("--fly_dir", type=str, required=True, help="Path to fly directory")
    parser.add_argument("--output", type=str, default="contacts_grid_test.mp4", help="Output video path")
    args = parser.parse_args()

    fly = Fly(args.fly_dir, as_individual=True)
    metrics = SkeletonMetrics(fly)
    print(f"Generating contacts grid video for {args.fly_dir}...")
    metrics.generate_contacts_grid_video(args.output)
    out_path = Path(args.output)
    if out_path.exists():
        print(f"Test passed: Output video created at {args.output}")
    else:
        print(f"Test failed: Output video not found at {args.output}")
        sys.exit(1)
