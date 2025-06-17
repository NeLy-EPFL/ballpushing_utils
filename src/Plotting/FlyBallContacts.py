from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))
from Ballpushing_utils.fly import Fly
from Ballpushing_utils.trajectory_metrics import TrajectoryMetrics

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
