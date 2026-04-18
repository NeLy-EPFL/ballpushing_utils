"""Shared pytest fixtures.

Fixtures that depend on real SLEAP/video data are resolved through
:func:`ballpushing_utils.paths.dataset` — i.e. relative to the
``BALLPUSHING_DATA_ROOT`` environment variable. If the data isn't
reachable (external user, CI without the lab share mounted, etc.) the
fixture calls :func:`pytest.skip` so the rest of the suite still runs.

Inside the Ramdya lab, setting ``BALLPUSHING_DATA_ROOT=/mnt/upramdya_data/MD``
is enough. For external users, see ``.env.example`` and the README.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ballpushing_utils import Experiment, Fly, dataset


# Relative paths under ``BALLPUSHING_DATA_ROOT`` that the fixtures point at.
# Kept here so the whole test suite agrees on which recordings count as the
# "canonical" example fly and example experiment.
_EXAMPLE_EXPERIMENT_REL = "MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked"
_EXAMPLE_FLY_REL = f"{_EXAMPLE_EXPERIMENT_REL}/arena2/corridor5"


def _require(path: Path) -> Path:
    """Return *path* if it exists, else skip the test cleanly."""
    if not path.exists():
        pytest.skip(
            f"Required data not found: {path}. "
            "Set BALLPUSHING_DATA_ROOT to a location that contains "
            f"'{_EXAMPLE_EXPERIMENT_REL}' (see .env.example)."
        )
    return path


@pytest.fixture(scope="module")
def example_fly_path() -> Path:
    """Absolute path to the canonical example fly."""
    return _require(dataset(_EXAMPLE_FLY_REL))


@pytest.fixture(scope="module")
def example_experiment_path() -> Path:
    """Absolute path to the canonical example experiment."""
    return _require(dataset(_EXAMPLE_EXPERIMENT_REL))


@pytest.fixture(scope="module")
def example_fly(example_fly_path: Path) -> Fly:
    return Fly(example_fly_path, as_individual=True)


@pytest.fixture(scope="module")
def example_experiment(example_experiment_path: Path) -> Experiment:
    return Experiment(example_experiment_path)
