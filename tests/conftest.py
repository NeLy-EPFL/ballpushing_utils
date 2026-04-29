"""Shared pytest fixtures.

Fixtures that depend on real SLEAP/video data are resolved with the
following priority:

1. ``$BALLPUSHING_DATA_ROOT`` if it is *explicitly* set in the
   environment — i.e. you've pointed pytest at a real data tree. We do
   **not** reuse :func:`ballpushing_utils.paths.dataset` here because
   that helper falls back to a hardcoded lab path
   (``/mnt/upramdya_data/MD``) when the env var is unset, which would
   mask the bundled fixtures on workstations where the lab share is
   mounted.
2. The Git-LFS-bundled sample set under
   ``tests/fixtures/sample_data/<same relative path>``.
3. If neither resolves, the fixture calls :func:`pytest.skip` so the
   rest of the suite still runs.

Inside the Ramdya lab, ``export BALLPUSHING_DATA_ROOT=/mnt/upramdya_data/MD``
keeps everything pointing at the full recordings. External users (or
anyone who wants to validate the fixture path) just need ``git lfs pull``
and an unset env var — see ``tests/fixtures/README.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ballpushing_utils import Experiment, Fly


# ---------------------------------------------------------------------------
# Pytest collection ignores.
#
# ``tests/F1_tracks/`` is a set of standalone dev/debug scripts (each has a
# ``main()`` + ``if __name__ == "__main__"`` and hardcoded paths under
# ``/mnt/upramdya_data/MD/F1_Tracks/``). They were named ``test_*.py`` by
# accident and are not pytest tests. They should be moved to ``tools/`` or
# ``scripts/dev/`` in a follow-up commit; for now we just ignore the
# directory so collection doesn't choke on a syntax error inside one of
# the scripts.
# ---------------------------------------------------------------------------
collect_ignore_glob = ["F1_tracks/*"]


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

# MagnetBlock paradigm. Used by tests that need a magnet-block-style
# recording (the analysis window typically starts at 1 h, so callers
# should set ``time_range=(3600, ...)`` in their config).
_MAGNETBLOCK_EXPERIMENT_REL = "MultiMazeRecorder/Videos/240711_MagnetBlock_Videos_Tracked"
_MAGNETBLOCK_FLY_REL = f"{_MAGNETBLOCK_EXPERIMENT_REL}/arena4/corridor4"


_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "sample_data"


_LFS_POINTER_MAX_BYTES = 512  # real LFS payloads (.h5, .mp4) are always >> this


def _lfs_pulled(directory: Path) -> bool:
    """Return True only if the large binary fixtures look like real data.

    Git-LFS pointer stubs are ~134 bytes.  Actual SLEAP .h5 files are
    several MB.  Checking size is therefore a reliable proxy for whether
    ``git lfs pull`` has been run.  If the directory has no .h5 or .mp4
    files at all we assume it's fine (e.g. a metadata-only folder).
    """
    binary_files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix in {".h5", ".mp4"}
    ]
    if not binary_files:
        return True
    return all(p.stat().st_size > _LFS_POINTER_MAX_BYTES for p in binary_files)


def _resolve(relative: str) -> Path | None:
    """Return the first existing location of *relative*, or ``None``.

    Honours an *explicit* ``BALLPUSHING_DATA_ROOT`` first; otherwise
    prefers the bundled Git-LFS fixture tree. The explicit-only check is
    deliberate — :func:`ballpushing_utils.paths.dataset` falls back to a
    hardcoded lab path when the env var is unset, which would silently
    bypass the fixtures on workstations where the lab share is mounted.

    Also returns ``None`` (causing a clean :func:`pytest.skip`) when the
    fixture directory exists but its LFS payloads are still pointer stubs
    — i.e. ``git lfs pull`` has not been run.
    """
    candidates: list[Path] = []
    explicit_root = os.environ.get("BALLPUSHING_DATA_ROOT", "").strip()
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser() / relative)
    candidates.append(_FIXTURES_ROOT / relative)
    for candidate in candidates:
        if candidate.exists() and _lfs_pulled(candidate):
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


# ---------------------------------------------------------------------------
# MagnetBlock fixtures. Opt-in for tests that exercise the MagnetBlock
# paradigm. The analysis window for these recordings typically starts at
# 1 h, so callers usually layer ``custom_config={"time_range": (3600, ...)}``
# on top.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_magnetblock_fly_path() -> Path:
    return _require(_MAGNETBLOCK_FLY_REL)


@pytest.fixture(scope="module")
def example_magnetblock_experiment_path() -> Path:
    return _require(_MAGNETBLOCK_EXPERIMENT_REL)


@pytest.fixture(scope="module")
def example_magnetblock_fly(example_magnetblock_fly_path: Path) -> Fly:
    return Fly(example_magnetblock_fly_path, as_individual=True)


@pytest.fixture(scope="module")
def example_magnetblock_experiment(example_magnetblock_experiment_path: Path) -> Experiment:
    return Experiment(example_magnetblock_experiment_path)
