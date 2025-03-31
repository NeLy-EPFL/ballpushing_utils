# @dataclass
class FlyMetadata:
    def __init__(self, fly):

        self.fly = fly
        self.directory = self.fly.directory
        self.experiment = self.fly.experiment
        self.arena = self.directory.parent.name
        self.corridor = self.directory.name
        self.name = f"{self.experiment.directory.name}_{self.arena}_{self.corridor}"
        self.arena_metadata = self.get_arena_metadata()

        # For each value in the arena metadata, add it as an attribute of the fly
        for var, data in self.arena_metadata.items():
            setattr(self, var, data)

        if fly.config.experiment_type == "F1":
            self.compute_F1_condition()

        self.nickname, self.brain_region = self.load_brain_regions(brain_regions_path)

        self.video = self.load_video()
        self.fps = self.experiment.fps

        self.original_size = self.get_video_size()

    def get_arena_metadata(self):
        """
        Retrieve the metadata for the Fly object's arena.

        This method looks up the arena's metadata in the experiment's metadata dictionary.
        The arena's name is converted to lowercase and used as the key to find the corresponding metadata.

        Returns:
            dict: A dictionary containing the metadata for the arena. The keys are the metadata variable names and the values are the corresponding metadata values. If no metadata is found for the arena, an empty dictionary is returned.
        """
        # Get the metadata for this fly's arena
        arena_key = self.arena.lower()
        return {
            var: data[arena_key]
            for var, data in self.experiment.metadata.items()
            if arena_key in data
        }

    def load_brain_regions(self, brain_regions_path):
        # Get the brain regions table
        brain_regions = pd.read_csv(brain_regions_path, index_col=0)

        # If the fly's genotype is defined in the arena metadata, find the associated nickname and brain region from the brain_regions_path file
        if "Genotype" in self.arena_metadata:
            try:
                genotype = self.arena_metadata["Genotype"]

                # If the genotype is None, skip the fly
                if genotype.lower() == "none":
                    print(f"Genotype is None: {self.name} is empty.")
                    return

                # Convert to lowercase for comparison
                lowercase_index = brain_regions.index.str.lower()
                matched_index = lowercase_index.get_loc(genotype.lower())

                self.nickname = brain_regions.iloc[matched_index]["Nickname"]
                self.brain_region = brain_regions.iloc[matched_index][
                    "Simplified region"
                ]
                self.simplified_nickname = brain_regions.iloc[matched_index][
                    "Simplified Nickname"
                ]
                self.split = brain_regions.iloc[matched_index]["Split"]
            except KeyError:
                print(
                    f"Genotype {genotype} not found in brain regions table for {self.name}. Defaulting to PR"
                )
                self.nickname = "PR"
                self.brain_region = "Control"
                self.simplified_nickname = "PR"
                self.split = "m"

        return self.nickname, self.brain_region

    def load_video(self):
        """Load the video file for the fly."""
        try:
            return list(self.directory.glob(f"{self.corridor}.mp4"))[0]
        except IndexError:
            try:
                return list(
                    self.directory.glob(
                        f"{self.directory.parent.name}_corridor_{self.corridor[-1]}.mp4"
                    )
                )[0]
            except IndexError:
                try:
                    # Look for a video file in the corridor directory
                    return list(self.directory.glob("*.mp4"))[0]
                except IndexError:
                    raise FileNotFoundError(f"No video found for {self.name}.")

    def get_video_size(self):
        """Get the size of the video."""

        # Load the video
        video = cv2.VideoCapture(str(self.video))

        return (
            video.get(cv2.CAP_PROP_FRAME_WIDTH),
            video.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )

    def compute_F1_condition(self):
        if "Pretraining" in self.arena_metadata and "Unlocked" in self.arena_metadata:
            pretraining = self.arena_metadata["Pretraining"]
            unlocked = self.arena_metadata["Unlocked"]

            if "n" in pretraining:
                self.arena_metadata["F1_condition"] = "control"
            elif "y" in pretraining:
                if "Left" in self.corridor:
                    if unlocked[0] == "y":
                        self.arena_metadata["F1_condition"] = "pretrained_unlocked"
                    else:
                        self.arena_metadata["F1_condition"] = "pretrained"
                elif "Right" in self.corridor:
                    if unlocked[1] == "y":
                        self.arena_metadata["F1_condition"] = "pretrained_unlocked"
                    else:
                        self.arena_metadata["F1_condition"] = "pretrained"
            else:
                print(f"Error: Pretraining value not valid for {self.name}")

        # Add F1_condition as an attribute of the fly
        if "F1_condition" in self.arena_metadata:
            setattr(self, "F1_condition", self.arena_metadata["F1_condition"])

    def display_metadata(self):
        """
        Print the metadata for the Fly object's arena.

        This method iterates over the arena's metadata dictionary and prints each key-value pair.

        Prints:
            str: The metadata variable name and its corresponding value, formatted as 'variable: value'.
        """
        # Print the metadata for this fly's arena
        for var, data in self.metadata.arena_metadata.items():
            print(f"{var}: {data}")