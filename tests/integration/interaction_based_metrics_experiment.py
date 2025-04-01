import pytest
from pathlib import Path
from Ballpushing_utils import Experiment, Dataset
from utils_behavior import Utils

if __name__ == "__main__":

    # Load an example experiment
    ExampleExperiment_path = (
        Utils.get_data_server() / "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked"
    )

    ExampleExperiment = Experiment(ExampleExperiment_path)

    # Make a dataset using the fly's ballpushing metrics

    ExampleData = Dataset(ExampleExperiment, dataset_type="summary")

    # Print the dataset

    print(ExampleData.data.head())
