import pytest
from pathlib import Path
from Ballpushing_utils import Fly, Dataset
from utils_behavior import Utils

if __name__ == "__main__":
    # Load an example fly
    ExampleFly_path = (
        Utils.get_data_server() / "MD/MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked/arena2/corridor5"
    )
    ExampleFly = Fly(ExampleFly_path, as_individual=True)

    # Get the fly's ballpushing metrics

    ExampleFly.events_metrics

    # Make a dataset using the fly's ballpushing metrics

    ExampleData = Dataset(ExampleFly, dataset_type="summary")

    # Print the dataset

    print(ExampleData.data.head())
