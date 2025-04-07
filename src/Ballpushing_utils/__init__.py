"""Python Package Template"""

from __future__ import annotations

__version__ = "0.0.1"

from .fly import Fly
from .experiment import Experiment
from .config import Config
from .fly_metadata import FlyMetadata
from .fly_trackingdata import FlyTrackingData
from .interactions_metrics import InteractionsMetrics
from .ballpushing_metrics import BallPushingMetrics
from .skeleton_metrics import SkeletonMetrics
from .learning_metrics import LearningMetrics
from .f1_metrics import F1Metrics
from .dataset import Dataset

# Add other main classes as needed

__all__ = [
    "Fly",
    "Experiment",
    "Config",
    "FlyMetadata",
    "FlyTrackingData",
    "InteractionsMetrics",
    "BallPushingMetrics",
    "SkeletonMetrics",
    "LearningMetrics",
    "F1Metrics",
    "Dataset",
]
