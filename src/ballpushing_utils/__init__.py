"""ballpushing_utils — analysis toolkit for Drosophila ball-pushing experiments.

Companion package for Durrieu et al. (2026), *Object manipulation and
affordance learning in Drosophila*. Provides data structures for flies,
experiments, and datasets; metrics for ball pushing, F1, interactions,
trajectories, skeleton kinematics, and learning; plus plotting and
statistics utilities used to generate the paper figures.

See the top-level :file:`README.md` for an overview and the
:file:`figures/` folder for per-figure reproduction scripts.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .ballpushing_metrics import BallPushingMetrics
from .behavior_umap import BehaviorUMAP  # noqa: F401 (re-exported helper)
from .config import Config
from .dataset import Dataset
from .experiment import Experiment
from .f1_metrics import F1Metrics
from .fly import Fly
from .fly_metadata import FlyMetadata
from .fly_trackingdata import FlyTrackingData
from .interactions_metrics import InteractionsMetrics
from .learning_metrics import LearningMetrics
from .skeleton_metrics import SkeletonMetrics

__all__ = [
    "BallPushingMetrics",
    "BehaviorUMAP",
    "Config",
    "Dataset",
    "Experiment",
    "F1Metrics",
    "Fly",
    "FlyMetadata",
    "FlyTrackingData",
    "InteractionsMetrics",
    "LearningMetrics",
    "SkeletonMetrics",
    "__version__",
]
