"""Unit tests for ballpushing_utils.preprocess_screen_data._find_source_paths.

Covers the two runtime code-paths that ``_find_source_paths`` takes:

1. Dataverse fallback — lab-share unavailable, brain-region feathers from the
   screen Dataverse archive are present in ``$BALLPUSHING_DATA_ROOT``.
2. Neither source available — ``FileNotFoundError`` is raised with a helpful
   message that mentions both the lab-share path and the ``ballpushing-fetch``
   command.

All tests are hermetic: no LFS fixtures, no lab share, no network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ballpushing_utils.dataverse_naming import SCREEN_STANDARDIZED_CONTACTS_FEATHERS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _isolate_lab_share(monkeypatch, tmp_path: Path) -> None:
    """Point the lab-share env var at a non-existent path so the first
    branch of ``_find_source_paths`` always falls through to the Dataverse
    fallback."""
    monkeypatch.setenv("BALLPUSHING_SCREEN_DATASETS_DIR", str(tmp_path / "no_such_dir"))
    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)


# ---------------------------------------------------------------------------
# Dataverse fallback — brain-region feathers present in data root.
# ---------------------------------------------------------------------------


def test_find_source_paths_dataverse_fallback_all_single_files(tmp_path, monkeypatch):
    """When lab share is unavailable and all 9 brain-region feathers are
    present as single files in the data root, every file is returned and
    the result is sorted by name."""
    _isolate_lab_share(monkeypatch, tmp_path)

    # Place stub feathers in the data root (content doesn't matter for this test).
    for name in SCREEN_STANDARDIZED_CONTACTS_FEATHERS:
        (tmp_path / name).touch()

    from ballpushing_utils.preprocess_screen_data import _find_source_paths

    paths = _find_source_paths()
    names = [p.name for p in paths]
    assert sorted(names) == names, "paths should be sorted by name"
    assert set(names) == set(SCREEN_STANDARDIZED_CONTACTS_FEATHERS)


def test_find_source_paths_dataverse_fallback_split_parts(tmp_path, monkeypatch):
    """When the first brain-region feather is split across numbered parts
    (``<stem>-1.feather``, ``<stem>-2.feather``, …), all parts are included."""
    _isolate_lab_share(monkeypatch, tmp_path)

    # Split the first feather into two parts; leave the rest as single files.
    first = SCREEN_STANDARDIZED_CONTACTS_FEATHERS[0]
    stem = Path(first).stem
    (tmp_path / f"{stem}-1.feather").touch()
    (tmp_path / f"{stem}-2.feather").touch()
    for name in SCREEN_STANDARDIZED_CONTACTS_FEATHERS[1:]:
        (tmp_path / name).touch()

    from ballpushing_utils.preprocess_screen_data import _find_source_paths

    paths = _find_source_paths()
    names = [p.name for p in paths]
    # Two split parts for the first feather, one each for the rest.
    assert f"{stem}-1.feather" in names
    assert f"{stem}-2.feather" in names
    # Unsplit originals are absent (only the numbered parts exist on disk).
    assert first not in names
    assert (
        len(paths) == len(SCREEN_STANDARDIZED_CONTACTS_FEATHERS) + 1
    )  # +1 extra split part


# ---------------------------------------------------------------------------
# Neither source available — FileNotFoundError.
# ---------------------------------------------------------------------------


def test_find_source_paths_raises_when_nothing_available(tmp_path, monkeypatch):
    """When neither the lab share nor any Dataverse feathers are found,
    FileNotFoundError is raised and the message mentions ballpushing-fetch."""
    _isolate_lab_share(monkeypatch, tmp_path)
    # tmp_path is empty — no feathers present.

    from ballpushing_utils.preprocess_screen_data import _find_source_paths

    with pytest.raises(FileNotFoundError, match="ballpushing-fetch"):
        _find_source_paths()
