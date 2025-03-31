class Experiment:
    """
    A class for an experiment. This represents a folder containing multiple flies, each of which is represented by a Fly object.
    """

    def __init__(
        self,
        directory,
        metadata_only=False,
        experiment_type=None,
    ):
        """
        Parameters
        ----------
        directory : Path
            The path to the experiment directory.

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

        # self.experiment_type = experiment_type

        # If metadata_only is True, don't load the flies
        if not metadata_only:
            self.flies = self.load_flies()

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
                metadata_dict[var] = {
                    k.lower(): v for k, v in metadata_dict[var].items()
                }
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
            print(
                f"Warning: fps.npy file not found in {self.directory}; Defaulting to 30 fps."
            )

        return fps

    def load_flies(self, multithreading=False):
        """
        Loads all flies in the experiment directory. Find subdirectories containing at least one .mp4 file, then find all .mp4 files that are named the same as their parent directory. Create a Fly object for each found folder.

        Returns:
            list: A list of Fly objects.
        """
        # Find all directories containing at least one .mp4 file
        mp4_directories = [
            dir for dir in self.directory.glob("**/*") if any(dir.glob("*.mp4"))
        ]

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
                        mp4_file = list(
                            dir.glob(f"{dir.parent.name}_corridor_{dir.name[-1]}.mp4")
                        )[0]
                    except IndexError:
                        try:
                            # Look for any .mp4 file in the directory
                            mp4_file = list(dir.glob("*.mp4"))[0]
                        except IndexError:
                            print(
                                f"No video found for {dir.name}. Moving to the next directory."
                            )
                            continue  # Move on to the next directory
                # print(f"Found video {mp4_file.name} for {dir.name}")
                mp4_files.append(mp4_file)

        # Create a Fly object for each .mp4 file using multiprocessing
        flies = []
        if multithreading:
            with Pool(processes=os.cpu_count()) as pool:
                results = [
                    pool.apply_async(
                        load_fly,
                        args=(
                            mp4_file,
                            self,
                            # self.experiment_type,
                        ),
                    )
                    for mp4_file in mp4_files
                ]
                for result in results:
                    fly = result.get()
                    if fly is not None:
                        flies.append(fly)
        else:
            for mp4_file in mp4_files:
                fly = load_fly(
                    mp4_file,
                    self,
                    # experiment_type=self.experiment_type,
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