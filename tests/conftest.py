# conftest.py (shared fixtures)
import pytest
from pathlib import Path
from Ballpushing_utils import Experiment, Fly


@pytest.fixture(scope="module")
def example_fly():
    path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5")
    return Fly(path, as_individual=True)


@pytest.fixture(scope="module")
def example_experiment():
    path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked")
    return Experiment(path)
