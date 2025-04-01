# This test is to check whether the central script correctly loads the config and initializes the experiment and fly objects.

import pytest
from pathlib import Path

from Ballpushing_utils import Experiment, Fly

# Load an example fly

ExampleFly_path = Path(
    "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5"
)

ExampleFly = Fly(ExampleFly_path, as_individual=True)

# Load an example experiment

ExampleExperiment_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked")

ExampleExperiment = Experiment(ExampleExperiment_path)
