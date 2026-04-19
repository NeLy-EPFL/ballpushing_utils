"""Shared pytest fixtures.

Fixtures that depend on real SLEAP/video data are resolved with the
following priority:

1. ``$BALLPUSHING_DATA_ROOT`` (the real lab share / workstation copy)
   via :func:`ballpushing_utils.paths.dataset`.
2. The Git-LFS-bundled sample set under
   ``tests/fixtures/sample_data/<same relative path>``.
3. If neither resolves, the fixture calls :func:`pytest.skip` so the
   rest of the suite still runs.

Inside the Ramdya lab, ``export BALLPUSHING_DATA_ROOT=/mnt/upramdya_data/MD``
keeps everything pointing at the full recordings. External users just
need ``git lfs pull`` to get the bundled sample fly — see
``tests/fixtures/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ballpushing_utils import Experiment, Fly, dataset


# ---------------------------------------------------------------------------
# Canonical relative paths. Kept here so the whole test suite agrees on
# which recordings count as the "canonical" fixtures. Each path is relative
# to either ``$BALLPUSHING_DATA_ROOT`` (real data) or
# ``tests/fixtures/sample_data/`` (bundled).
# ---------------------------------------------------------------------------

# Non-F1 (TNT_Fine). Canonical fly for the bulk of unit + integration tests.
_EXAMPLE_EXPERIMENT_REL = "MultiMazeRecorder/Videos/231130_TNT_Fine_2_Videos_Tracked"
_EXAMPLE_FLY_REL = f"{_EXAMPLE_EXPERIMENT_REL}/arena2/corridor5"

# F1 paradigm. Used by the F1-specific tests + the F1 subsections of the
# dataset_types_guide notebook.
_F1_EXPERIMENT_REL = "F1_Tracks/Videos/250904_F1_New_Videos_Checked"
_F1_FLY_REL = f"{_F1_EXPERIMENT_REL}/arena5/Right"


_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "sample_data"


def _resolve(relative: str) -> Path | None:
    """Return the first existing location of *relative*, or ``None``.

    Checks ``BALLPUSHING_DATA_ROOT`` first (via
    :func:`ballpushing_utils.paths.dataset`), then the bundled Git-LFS
    fixture tree.
    """
    for candidate in (dataset(relative), _FIXTURES_ROOT / relative):
        if candidate.exists():
            return candidate
    return None


def _require(relative: str) -> Path:
    """Return the resolved path for *relative*, else skip the test cleanly."""
    path = _resolve(relative)
    if path is None:
        pytest.skip(
            f"Required data not found: {relative!r}. Either set "
            f"BALLPUSHING_DATA_ROOT to a location that contains it, or run "
            f"`git lfs pull` to fetch the bundled fixture under "
            f"tests/fixtures/sample_data/. See tests/fixtures/README.md."
        )
    return path


# ---------------------------------------------------------------------------
# Non-F1 fixtures (canonical defaults used throughout the suite).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_fly_path() -> Path:
    """Absolute path to the canonical example fly."""
    return _require(_EXAMPLE_FLY_REL)


@pytest.fixture(scope="module")
def example_experiment_path() -> Path:
    """Absolute path to the canonical example experiment."""
    return _require(_EXAMPLE_EXPERIMENT_REL)


@pytest.fixture(scope="module")
def example_fly(example_fly_path: Path) -> Fly:
    return Fly(example_fly_path, as_individual=True)


@pytest.fixture(scope="module")
def example_experiment(example_experiment_path: Path) -> Experiment:
    return Experiment(example_experiment_path)


# ---------------------------------------------------------------------------
# F1 fixtures. Opt-in for tests that exercise the F1 paradigm. Skip cleanly
# on installs where neither the real lab share nor the F1 LFS fixture is
# available.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_f1_fly_path() -> Path:
    return _require(_F1_FLY_REL)


@pytest.fixture(scope="module")
def example_f1_experiment_path() -> Path:
    return _require(_F1_EXPERIMENT_REL)


@pytest.fixture(scope="module")
def example_f1_fly(example_f1_fly_path: Path) -> Fly:
    return Fly(example_f1_fly_path, as_individual=True, custom_config={"experiment_type": "F1"})


@pytest.fixture(scope="module")
def example_f1_experiment(example_f1_experiment_path: Path) -> Experiment:
    return Experiment(example_f1_experiment_path, custom_config={"experiment_type": "F1"})
