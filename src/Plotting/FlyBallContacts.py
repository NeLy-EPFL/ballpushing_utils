from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))
from Ballpushing_utils.fly import Fly
from Ballpushing_utils.trajectory_metrics import TrajectoryMetrics
import argparse
import yaml
import os

# Example fly directory (adjust as needed)
exFly = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/240104_TNT_Fine_1_Videos_Tracked/arena9/corridor5")

# Load the Fly object
fly = Fly(exFly, as_individual=True)

# Create TrajectoryMetrics instance
metrics = TrajectoryMetrics(fly)

# Use the plotting routine (make sure the method exists in TrajectoryMetrics)
output_path = "outputs/fly_ball_y_positions_contact_annotated.png"
try:
    metrics.plot_y_positions_over_time(output_path)
    print(f"Plot saved to {output_path}")
except Exception as e:
    print(f"Failed to plot and save: {e}")


def plot_fly_contact(fly_dir, output_path=None, save=False, yaml_stem=None, output_dir="outputs"):
    fly = Fly(Path(fly_dir), as_individual=True)
    metrics = TrajectoryMetrics(fly)
    fly_name = getattr(fly.metadata, "name", Path(fly_dir).stem)
    if output_path is None:
        if yaml_stem is not None:
            dir_path = os.path.join(output_dir, yaml_stem)
            os.makedirs(dir_path, exist_ok=True)
            output_path = os.path.join(dir_path, f"{fly_name}_fly_ball_y_positions_contact_annotated.png")
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{fly_name}_fly_ball_y_positions_contact_annotated.png")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        metrics.plot_y_positions_over_time(output_path if save else None, zoom_separate=True)
        if save:
            print(f"Plot saved to {output_path}")
        else:
            print(f"Plot shown for {fly_dir}")
    except Exception as e:
        print(f"Failed to plot and save for {fly_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot fly-ball contacts for a list of flies from a YAML file or a single example."
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Path to YAML file listing experiment directories.",
    )
    parser.add_argument(
        "--save", action="store_true", help="If set, save plots to disk instead of showing interactively."
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory for plots.")
    args = parser.parse_args()
    if args.yaml is not None:
        with open(args.yaml, "r") as f:
            fly_dirs = yaml.safe_load(f).get("directories", [])
        yaml_base = os.path.splitext(os.path.basename(args.yaml))[0]
        for fly_dir in fly_dirs:
            plot_fly_contact(fly_dir, output_path=None, save=args.save, yaml_stem=yaml_base, output_dir=args.output_dir)
    else:
        fly_name = getattr(fly.metadata, "name", Path(exFly).stem)
        output_path = os.path.join(
            args.output_dir, "example_fly", f"{fly_name}_fly_ball_y_positions_contact_annotated.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plot_fly_contact(exFly, output_path=output_path, save=True)
