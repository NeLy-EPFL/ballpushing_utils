import pytest
from pathlib import Path
from Ballpushing_utils import Experiment, Fly
from utils_behavior import Utils

if __name__ == "__main__":

    # Load an example fly

    ExampleFly_path = (
        Utils.get_data_server() / "MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5"
    )

    ExampleFly = Fly(ExampleFly_path, as_individual=True)

    # Load an example experiment

    ExampleExperiment_path = (
        Utils.get_data_server() / "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked"
    )

    ExampleExperiment = Experiment(ExampleExperiment_path)

else:

    @pytest.fixture(scope="module")
    def example_fly():
        path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5")
        return Fly(path, as_individual=True)

    @pytest.fixture(scope="module")
    def example_experiment():
        path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked")
        return Experiment(path)
