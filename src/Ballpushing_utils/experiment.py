from __future__ import annotations
from pathlib import Path
from .config import Config
import json
import os
import numpy as np
from multiprocessing import Pool


class Experiment:
    """
    A class for an experiment. This represents a folder containing multiple flies, each of which is represented by a Fly object.
    """

    def __init__(
        self,
        directory,
        metadata_only=False,
        custom_config=None,
        multiprocess=False,
        num_processes=None,
    ):
        """
        Parameters
        ----------
        directory : Path
            The path to the experiment directory.
        metadata_only : bool
            If True, only load metadata without loading flies.
        custom_config : object
            Custom configuration for fly loading.
        multiprocess : bool
            Whether to use multiprocessing for loading flies.
        num_processes : int
            Number of processes to use for multiprocessing. If None, uses os.cpu_count().

        Attributes
        ----------
        directory : Path
            The path to the experiment directory.
        metadata : dict
            A dictionary containing the metadata for the experiment.
        fps : str
            The frame rate of the videos.
        """

        self.config = Config()
        self.directory = Path(directory)
        self.metadata = self.load_metadata()
        self.fps = self.load_fps()

        # If metadata_only is True, don't load the flies
        if not metadata_only:
            self.flies = self.load_flies(
                custom_config=custom_config, multiprocess=multiprocess, num_processes=num_processes
            )

    def __str__(self):
        # Generate a list of unique genotypes from the flies in the experiment
        tested_genotypes = list(set([fly.Genotype for fly in self.flies]))

        return f"Experiment: {self.directory.name}\n  Genotypes: {', '.join(tested_genotypes)}\n  Flies: {len(self.flies)}\n  FPS: {self.fps}"

    def __repr__(self):
        return f"Experiment({self.directory})"

    def load_metadata(self):
        """
        Loads the metadata for the experiment. The metadata is stored in a JSON file in the experiment directory. The file is loaded as a dictionary and each variable is stored as a key in the dictionary. Each variable key contains a dictionary with the arena number as the key and the value for that variable in that arena as the value.

        Returns:
            dict: A dictionary containing the metadata for the experiment.
        """
        with open(self.directory / "Metadata.json", "r") as f:
            metadata = json.load(f)
            variables = metadata["Variable"]
            metadata_dict = {}
            for var in variables:
                metadata_dict[var] = {}
                for arena in range(1, 10):
                    arena_key = f"Arena{arena}"
                    var_index = variables.index(var)
                    metadata_dict[var][arena_key] = metadata[arena_key][var_index]

            # In the metadata_dict, make all they Arena subkeys lower case

            for var in variables:
                metadata_dict[var] = {k.lower(): v for k, v in metadata_dict[var].items()}
            # print(metadata_dict)
            return metadata_dict

    def load_fps(self):
        """
        Loads the frame rate of the videos in the experiment directory.

        Returns:
            int: The frame rate of the videos.
        """
        # Load the fps value from the fps.npy file in the experiment directory
        fps_file = self.directory / "fps.npy"
        if fps_file.exists():
            fps = np.load(fps_file)

        else:
            fps = 30
            # print(f"Warning: fps.npy file not found in {self.directory}; Defaulting to 30 fps.")

        return fps

    def load_flies(
        self,
        custom_config,
        multiprocess=False,
        num_processes=None,
    ):
        """
        Loads all flies in the experiment directory. Find subdirectories containing at least one .mp4 file, then find all .mp4 files that are named the same as their parent directory. Create a Fly object for each found folder.

        Parameters:
            custom_config: Custom configuration for fly loading
            multiprocess (bool): Whether to use multiprocessing for loading flies
            num_processes (int): Number of processes to use. If None, uses os.cpu_count()

        Returns:
            list: A list of Fly objects.
        """
        # Find all directories containing at least one .mp4 file
        mp4_directories = [dir for dir in self.directory.glob("**/*") if any(dir.glob("*.mp4"))]

        # Find all .mp4 files that are named the same as their parent directory
        mp4_files = []
        for dir in mp4_directories:
            if dir.is_dir():
                try:
                    # Look for a video file named the same as the directory
                    mp4_file = list(dir.glob(f"{dir.name}.mp4"))[0]
                except IndexError:
                    try:
                        # Look for a video file named with the parent directory and corridor
                        mp4_file = list(dir.glob(f"{dir.parent.name}_corridor_{dir.name[-1]}.mp4"))[0]
                    except IndexError:
                        try:
                            # Look for any .mp4 file in the directory
                            mp4_file = list(dir.glob("*.mp4"))[0]
                        except IndexError:
                            print(f"No video found for {dir.name}. Moving to the next directory.")
                            continue  # Move on to the next directory
                # print(f"Found video {mp4_file.name} for {dir.name}")
                mp4_files.append(mp4_file)

        # Create a Fly object for each .mp4 file
        flies = []
        if multiprocess:
            # Use multiprocessing for parallel loading
            if num_processes is None:
                num_processes = os.cpu_count()

            # Create a serializable experiment data dict for workers
            experiment_data = {"directory": self.directory, "metadata": self.metadata, "fps": self.fps}

            with Pool(processes=num_processes) as pool:
                results = [
                    pool.apply_async(
                        load_fly_worker,
                        args=(
                            mp4_file,
                            experiment_data,
                            custom_config,
                        ),
                    )
                    for mp4_file in mp4_files
                ]
                for result in results:
                    try:
                        fly = result.get(timeout=60)  # 60 second timeout per fly
                        if fly is not None:
                            flies.append(fly)
                    except Exception as e:
                        print(f"Error loading fly in multiprocessing: {e}")
        else:
            # Sequential loading
            for mp4_file in mp4_files:
                fly = load_fly(
                    mp4_file,
                    self,
                    custom_config,
                )
                if fly is not None:
                    flies.append(fly)

        return flies

    def find_flies(self, on, value):
        """
        Makes a list of Fly objects matching a certain criterion.

        Parameters
        ----------
        on : str
            The name of the attribute to filter on.

        value : str
            The value of the attribute to filter on.

        Returns
        ----------
        list
            A list of Fly objects matching the criterion.
        """

        return [fly for fly in self.flies if getattr(fly, on, None) == value]


def filter_experiments(source, **criteria):
    """Generates a list of Experiment objects based on criteria.

    Args:
        source (list): A list of flies, experiments or folders to create Experiment objects from.
        criteria (dict): A dictionary of criteria to filter the experiments.

    Returns:
        list: A list of Experiment objects that match the criteria.
    """
    from Ballpushing_utils import Fly

    flies = []

    # If the source is a list of flies, check directly for the criteria in flies
    if isinstance(source[0], Fly):
        for fly in source:
            if all(fly.arena_metadata.get(key) == value for key, value in criteria.items()):
                flies.append(fly)

    else:
        if isinstance(source[0], Experiment):
            Exps = source

        else:
            # Create Experiment objects from the folders
            Exps = [Experiment(f) for f in source]

        # Get a list of flies based on the criteria
        for exp in Exps:
            for fly in exp.flies:
                if all(fly.arena_metadata.get(key) == value for key, value in criteria.items()):
                    flies.append(fly)

    return flies


def load_fly(
    mp4_file,
    experiment,
    # experiment_type,
    custom_config,
):
    from Ballpushing_utils import Fly

    print(f"Loading fly from {mp4_file.parent}")
    try:
        fly = Fly(
            mp4_file.parent,
            experiment=experiment,
            # experiment_type=experiment_type,
            custom_config=custom_config,
        )
        try:
            if fly.tracking_data and fly.tracking_data.valid_data:
                return fly
        except Exception as e:
            print(f"Error while validating tracking data for {mp4_file.parent}: {e}")
        return None
    except TypeError as e:
        print(f"Error while loading fly from {mp4_file.parent}: {e}")
    return None


def load_fly_worker(mp4_file, experiment_data, custom_config):
    """
    Worker function for multiprocessing fly loading.

    Parameters:
        mp4_file: Path to the MP4 file
        experiment_data: Serializable dictionary containing experiment information
        custom_config: Custom configuration for fly loading

    Returns:
        Fly object if successful, None otherwise
    """
    from Ballpushing_utils import Fly

    print(f"Loading fly from {mp4_file.parent} (worker)")
    try:
        # Create a minimal experiment object for the fly
        class MockExperiment:
            def __init__(self, data, custom_config):
                self.directory = data["directory"]
                self.metadata = data["metadata"]
                self.fps = data["fps"]
                # Add the config attribute that Fly expects
                try:
                    from .config import Config

                    self.config = Config()
                except:
                    # Fallback if config import fails
                    self.config = None

        mock_experiment = MockExperiment(experiment_data, custom_config)

        fly = Fly(
            mp4_file.parent,
            experiment=mock_experiment,
            custom_config=custom_config,
        )

        try:
            if fly.tracking_data and fly.tracking_data.valid_data:
                return fly
        except Exception as e:
            print(f"Worker error validating tracking data for {mp4_file.parent}: {e}")
        return None
    except Exception as e:
        print(f"Worker error loading fly from {mp4_file.parent}: {e}")
    return None
