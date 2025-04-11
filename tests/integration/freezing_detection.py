import argparse
from pathlib import Path
from Ballpushing_utils import Fly, Experiment, Dataset, BallPushingMetrics
from utils_behavior import Utils

ExampleFly_path = (
    Utils.get_data_server() / "MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5"
)


def process_fly(fly_path):
    # Load an example fly
    ExampleFly = Fly(fly_path, as_individual=True)

    # Get the fly's interaction metrics

    metrics = BallPushingMetrics(ExampleFly.tracking_data)

    pauses = metrics.detect_pauses(0)
    if pauses is not None:
        print(f"Pauses: {pauses}")


process_fly(ExampleFly_path)
