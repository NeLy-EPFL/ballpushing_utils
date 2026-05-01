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
from .compat import normalize_legacy_columns, read_feather
from .config import Config
from .dataset import Dataset
from .dataverse import (
    ARCHIVE_PREFIX_RECIPES,
    CONDITION_TRANSFORMERS,
    DEFAULT_CONDITION_FIELD,
    DEFAULT_VIDEO_SIZE,
    DataverseFly,
    default_condition_field,
    detect_dataverse_experiment_type,
    expand_condition,
    is_dataverse_layout,
    iter_dataverse_flies,
    parse_archive_name,
    synthesize_experiment_metadata,
)
from .experiment import Experiment
from .f1_metrics import F1Metrics
from .fly import Fly
from .fly_metadata import FlyMetadata
from .fly_trackingdata import FlyTrackingData
from .interactions_metrics import InteractionsMetrics
from .learning_metrics import LearningMetrics
from .paths import (
    data_root,
    dataset,
    detect_layout,
    figure_output_dir,
    figures_root,
    find_feather,
    load_dotenv,
    missing_data_message,
)
from .skeleton_metrics import SkeletonMetrics

__all__ = [
    "ARCHIVE_PREFIX_RECIPES",
    "BallPushingMetrics",
    "BehaviorUMAP",
    "CONDITION_TRANSFORMERS",
    "Config",
    "DEFAULT_CONDITION_FIELD",
    "DEFAULT_VIDEO_SIZE",
    "Dataset",
    "DataverseFly",
    "Experiment",
    "F1Metrics",
    "Fly",
    "FlyMetadata",
    "FlyTrackingData",
    "InteractionsMetrics",
    "LearningMetrics",
    "SkeletonMetrics",
    "__version__",
    "data_root",
    "dataset",
    "default_condition_field",
    "detect_dataverse_experiment_type",
    "detect_layout",
    "expand_condition",
    "figure_output_dir",
    "figures_root",
    "find_feather",
    "is_dataverse_layout",
    "iter_dataverse_flies",
    "load_dotenv",
    "missing_data_message",
    "normalize_legacy_columns",
    "parse_archive_name",
    "read_feather",
    "synthesize_experiment_metadata",
]
