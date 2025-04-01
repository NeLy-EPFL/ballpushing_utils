# Internal classes import

import Ballpushing_utils.fly as fly, Ballpushing_utils.experiment as experiment, Ballpushing_utils.dataset as dataset, Ballpushing_utils.config as config, Ballpushing_utils.fly_metadata as fly_metadata, Ballpushing_utils.fly_trackingdata as fly_trackingdata, Ballpushing_utils.ballpushing_metrics as ballpushing_metrics, Ballpushing_utils.skeleton_metrics as skeleton_metrics, Ballpushing_utils.learning_metrics as learning_metrics, Ballpushing_utils.f1_metrics as f1_metrics, Ballpushing_utils.utilities as utilities

# Internal classes from Nely github utils_behavior

import utils_behavior

# External libraries import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool

from pathlib import Path
import sys
import traceback
import json
import datetime
from collections import Counter
import pickle
import os
import cv2
import moviepy
import moviepy.config as mpconfig

from moviepy.editor import (
    VideoFileClip,
    clips_array,
    ColorClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip,
)
from moviepy.editor import VideoClip
from moviepy.video.fx import all as vfx

import pygame

import warnings

from dataclasses import dataclass

from scipy.signal import find_peaks, savgol_filter

moviepy.config.change_settings({"IMAGEMAGICK_BINARY": "magick"})
mpconfig.change_settings({"IMAGEMAGICK_BINARY": "/home/durrieu/miniforge3/envs/tracking_analysis/bin/magick"})

os.environ["MAGICK_FONT_PATH"] = "/etc/ImageMagick-6"
# os.environ['MAGICK_CONFIGURE_PATH'] = '/etc/ImageMagick-6'


sys.modules["Ballpushing_utils"] = sys.modules[__name__]
# This line creates an alias for utils_behavior.Ballpushing_utils to
# utils_behavior.__init__ so that the previously made pkl files can be loaded.

print("Loading BallPushing utils version 10 Mar 2025")

print(f"Current configuration : {config.Config.__dict__}")

brain_regions_path = "/mnt/upramdya_data/MD/Region_map_250116.csv"
