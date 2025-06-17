from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from Ballpushing_utils.fly import Fly
from Ballpushing_utils.trajectory_metrics import TrajectoryMetrics
from Ballpushing_utils.skeleton_metrics import SkeletonMetrics

# Example fly directory
exFly = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/240104_TNT_Fine_1_Videos_Tracked/arena9/corridor5")

# Load the Fly object
fly = Fly(exFly, as_individual=True)

# Ensure flyball_positions is loaded
if fly.flyball_positions is None and hasattr(fly, "tracking_data") and fly.tracking_data is not None:
    if hasattr(fly.tracking_data, "flytrack") and fly.tracking_data.flytrack is not None:
        try:
            flyball_df = fly.tracking_data.flytrack.objects[0].dataset.copy()
            fly.flyball_positions = flyball_df
        except Exception as e:
            print(f"Could not extract flyball_positions: {e}")
    else:
        print("No tracking data available for this fly.")

# Check the columns in flyball_positions
if fly.flyball_positions is not None:
    print("Flyball positions columns:", fly.flyball_positions.columns.tolist())
else:
    print("Flyball positions not available. Ensure tracking data is loaded correctly.")

# --- Plot fly and ball Y position across % ball completion on the same plot ---

metrics = TrajectoryMetrics(fly)
rel_y_fly = np.abs(metrics.rel_y_fly)
rel_y_ball = np.abs(metrics.rel_y_ball)
percent_completion = metrics.percent_completion()

if (
    len(rel_y_fly) > 0
    and len(rel_y_ball) > 0
    and len(percent_completion) > 0
    and fly.flyball_positions is not None
    and "time_fly" in fly.flyball_positions
):
    time = fly.flyball_positions["time_fly"].values
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=100)

    plt.figure(figsize=(10, 5))
    sc1 = plt.scatter(time, rel_y_fly, color="red", label="Fly Y (abs rel to start)", s=10)
    sc2 = plt.scatter(
        time,
        rel_y_ball,
        c=percent_completion,
        cmap=cmap,
        norm=norm,
        label="Ball Y (abs rel to fly start)",
        s=10,
        marker="x",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("|Y Position| (pixels, abs rel to fly start)")
    plt.title("Absolute Fly and Ball Y Position Over Time (Color: % Ball Completion)")
    plt.legend()
    cbar = plt.colorbar(sc2, label="% Ball Completion (Y, normalized)")
    plt.tight_layout()
    plt.show()
else:
    print("Could not plot: required data missing.")

# --- Plot fly and ball Y position from contact-annotated dataset ---

# Instantiate SkeletonMetrics and get the contact-annotated dataset
try:
    skeleton_metrics = SkeletonMetrics(fly)
    flyball_df = skeleton_metrics.get_contact_annotated_dataset()
except Exception as e:
    flyball_df = None
    print(f"Could not create SkeletonMetrics or get contact-annotated dataset: {e}")

if flyball_df is not None and "y_Head" in flyball_df and "y_centre_preprocessed" in flyball_df:
    time = flyball_df["time"] if "time" in flyball_df else flyball_df.index
    plt.figure(figsize=(10, 5))
    plt.plot(time, flyball_df["y_Head"], label="Fly y_Head", color="red")
    plt.plot(time, flyball_df["y_centre_preprocessed"], label="Ball y_centre_preprocessed", color="blue")
    plt.xlabel("Time (s)" if "time" in flyball_df else "Frame")
    plt.ylabel("Y Position (pixels)")
    plt.title("Fly y_Head and Ball y_centre_preprocessed Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Could not plot: contact-annotated dataset or required columns missing.")
