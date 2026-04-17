from __future__ import annotations
from pathlib import Path
import pandas as pd
from .utilities import brain_regions_path

import cv2


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
        return {var: data[arena_key] for var, data in self.experiment.metadata.items() if arena_key in data}

    def load_brain_regions(self, brain_regions_path):
        # Safe defaults
        self.nickname = "PR"
        self.brain_region = "Control"
        self.simplified_nickname = "PR"
        self.split = "m"

        # Load brain regions lookup table
        brain_regions = pd.read_csv(brain_regions_path, index_col=0)

        # Get genotype from arena metadata (may be missing)
        genotype = self.arena_metadata.get("Genotype", None)

        # Rule 1: If no Genotype entry, keep defaults (PR/Control)
        if genotype is None or str(genotype).strip() == "":
            return self.nickname, self.brain_region

        genotype_str = str(genotype).strip()

        # Rule 2: If Genotype exists but is "none", this fly is empty -> skip
        if genotype_str.lower() == "none":
            print(f"Genotype is None: {self.name} is empty.")
            # Raise TypeError so upstream loader skips this fly gracefully
            raise TypeError("Empty fly (Genotype == none)")

        # Rule 3/4: If genotype maps in table, set dynamically; else nickname := genotype
        try:
            lowercase_index = brain_regions.index.str.lower()
            matched_index = lowercase_index.get_loc(genotype_str.lower())
            row = brain_regions.iloc[matched_index]

            self.nickname = row["Nickname"]
            self.brain_region = row["Simplified region"]
            # Optional columns with safe fallbacks
            self.simplified_nickname = row.get("Simplified Nickname", self.nickname)
            self.split = row.get("Split", self.split)
        except KeyError:
            # Unmapped genotype: nickname becomes the genotype; brain region unknown
            self.nickname = genotype_str
            self.brain_region = "Unknown"
            self.simplified_nickname = genotype_str
            # self.split remains default

        return self.nickname, self.brain_region

    def load_video(self):
        """Load the video file for the fly."""
        try:
            return list(self.directory.glob(f"{self.corridor}.mp4"))[0]
        except IndexError:
            try:
                return list(self.directory.glob(f"{self.directory.parent.name}_corridor_{self.corridor[-1]}.mp4"))[0]
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
