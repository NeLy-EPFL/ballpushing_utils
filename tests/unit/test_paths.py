"""Unit tests for ``ballpushing_utils.paths`` helpers.

Covers the resolution / breadcrumb / mount-mismatch surface that user-
facing scripts depend on:

- :func:`find_feather` — 4-rule resolution: literal → data_root →
  ``BALLPUSHING_FEATHER_SEARCH`` → recursive rglob (single-match).
- :func:`detect_layout` — ``"server"`` / ``"dataverse"`` / ``None``
  classifier on synthetic trees.
- :func:`missing_data_message` — produces the structured three-source
  breadcrumb users see when data isn't reachable.
- :func:`require_path` — friendly error for hardcoded lab paths that
  may not resolve on machines with a different mount layout.
- :func:`get_cache_dir` — returns ``<repo>/.cache``.

All tests are hermetic; no LFS fixtures, no lab share.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ballpushing_utils.paths import (
    detect_layout,
    find_feather,
    get_cache_dir,
    missing_data_message,
    require_path,
)


# ---------------------------------------------------------------------------
# Pytest helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolate_env(monkeypatch, tmp_path):
    """Point BALLPUSHING_DATA_ROOT at an empty tmp dir + clear feather-search.

    The default ``BALLPUSHING_DATA_ROOT`` is the lab share, which would
    confound the find_feather tests if it happens to be mounted.
    """
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(data_root))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)
    return data_root


# ---------------------------------------------------------------------------
# find_feather — four-rule resolution.
# ---------------------------------------------------------------------------


def test_find_feather_rule1_literal_existing_path(tmp_path, isolate_env):
    """Rule 1: a literal path that already exists is returned as-is."""
    feather = tmp_path / "foo.feather"
    feather.write_bytes(b"")
    assert find_feather(feather) == feather.resolve()


def test_find_feather_rule2_relative_to_data_root(isolate_env):
    """Rule 2: relative paths resolve under BALLPUSHING_DATA_ROOT."""
    target = isolate_env / "MagnetBlock" / "Datasets" / "X" / "summary" / "pooled.feather"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    rel = "MagnetBlock/Datasets/X/summary/pooled.feather"
    assert find_feather(rel) == target.resolve()


def test_find_feather_rule3_feather_search_env(monkeypatch, tmp_path, isolate_env):
    """Rule 3: BALLPUSHING_FEATHER_SEARCH dirs match by basename."""
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    target = downloads / "pooled.feather"
    target.write_bytes(b"")
    monkeypatch.setenv("BALLPUSHING_FEATHER_SEARCH", str(downloads))
    # A relative path that wouldn't resolve under DATA_ROOT — basename
    # search picks it up under BALLPUSHING_FEATHER_SEARCH.
    assert find_feather("Some/Other/Path/pooled.feather") == target.resolve()


def test_find_feather_rule3_feather_search_multiple_dirs(monkeypatch, tmp_path, isolate_env):
    """``BALLPUSHING_FEATHER_SEARCH`` accepts colon-separated dirs."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (b / "wanted.feather").write_bytes(b"")
    monkeypatch.setenv("BALLPUSHING_FEATHER_SEARCH", os.pathsep.join([str(a), str(b)]))
    assert find_feather("anywhere/wanted.feather") == (b / "wanted.feather").resolve()


def test_find_feather_rule4_recursive_rglob_single_match(isolate_env):
    """Rule 4: single rglob match under BALLPUSHING_DATA_ROOT is accepted."""
    target = isolate_env / "deep" / "nested" / "wanted.feather"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    assert find_feather("anywhere/wanted.feather") == target.resolve()


def test_find_feather_rule4_ambiguous_match_returns_none(isolate_env, caplog):
    """Multiple rglob matches → returns None (caller produces breadcrumb)."""
    (isolate_env / "a").mkdir()
    (isolate_env / "b").mkdir()
    (isolate_env / "a" / "dup.feather").write_bytes(b"")
    (isolate_env / "b" / "dup.feather").write_bytes(b"")
    assert find_feather("anywhere/dup.feather") is None
    # And we logged the fact that the basename was ambiguous.
    assert any("matched basename" in r.message for r in caplog.records)


def test_find_feather_no_match_returns_none(isolate_env):
    assert find_feather("does/not/exist.feather") is None


# ---------------------------------------------------------------------------
# detect_layout — server / dataverse / None.
# ---------------------------------------------------------------------------


def test_detect_layout_server(tmp_path):
    """A directory containing an experiment folder with Metadata.json → 'server'."""
    exp = tmp_path / "231222_TNT_Fine_1_Videos_Tracked"
    exp.mkdir()
    (exp / "Metadata.json").write_text("{}")
    (exp / "arena1" / "corridor1").mkdir(parents=True)
    (exp / "arena1" / "corridor1" / "ball.h5").write_bytes(b"")
    assert detect_layout(tmp_path) == "server"


def test_detect_layout_server_root_itself_is_experiment(tmp_path):
    """Root contains Metadata.json directly → server."""
    (tmp_path / "Metadata.json").write_text("{}")
    assert detect_layout(tmp_path) == "server"


def test_detect_layout_dataverse(tmp_path):
    """Condition/date/arena/corridor with *ball*.h5 + no Metadata.json → 'dataverse'."""
    fly_dir = tmp_path / "MB247xTNT" / "231222" / "arena6" / "corridor5"
    fly_dir.mkdir(parents=True)
    (fly_dir / "ball.h5").write_bytes(b"")
    assert detect_layout(tmp_path) == "dataverse"


def test_detect_layout_returns_none_for_empty(tmp_path):
    assert detect_layout(tmp_path) is None


def test_detect_layout_returns_none_for_nonexistent(tmp_path):
    assert detect_layout(tmp_path / "nope") is None


# ---------------------------------------------------------------------------
# missing_data_message — structured breadcrumb.
# ---------------------------------------------------------------------------


def test_missing_data_message_lists_three_sources():
    msg = missing_data_message("foo.feather", context="feather")
    # The breadcrumb must mention all three escape hatches by name so a
    # user without lab access knows what to do next.
    assert "BALLPUSHING_DATA_ROOT" in msg
    assert "Dataverse" in msg
    assert "sample" in msg.lower() or "fixtures/sample_data" in msg


def test_missing_data_message_includes_requested_path():
    msg = missing_data_message("specific/path/foo.feather", context="feather")
    assert "specific/path/foo.feather" in msg


# ---------------------------------------------------------------------------
# require_path — friendly mount-mismatch handling.
# ---------------------------------------------------------------------------


def test_require_path_returns_existing_path(tmp_path):
    out = require_path(tmp_path, description="test root")
    assert out == tmp_path


def test_require_path_raises_with_three_fixes_listed(tmp_path):
    missing = tmp_path / "no" / "such" / "dir"
    with pytest.raises(FileNotFoundError) as excinfo:
        require_path(missing, description="confocal stacks")
    msg = str(excinfo.value)
    assert "confocal stacks" in msg  # the description appears
    assert "Mount the lab NFS share" in msg
    assert "Symlink" in msg
    assert "Edit the script" in msg
    assert str(missing) in msg


def test_require_path_env_var_override_takes_precedence(tmp_path, monkeypatch):
    """When env_var is set AND points at an existing path, use that."""
    monkeypatch.setenv("MY_OVERRIDE", str(tmp_path))
    out = require_path("/no/such/hardcoded/path", env_var="MY_OVERRIDE")
    assert out == tmp_path


def test_require_path_env_var_set_but_invalid_falls_through_to_error(monkeypatch):
    """Override that doesn't exist → fall through to the breadcrumb error."""
    monkeypatch.setenv("MY_OVERRIDE", "/also/not/here")
    with pytest.raises(FileNotFoundError) as excinfo:
        require_path("/not/here/either", env_var="MY_OVERRIDE")
    # The error mentions the env_var so the user knows that knob exists.
    assert "MY_OVERRIDE" in str(excinfo.value)


def test_require_path_expands_tilde(tmp_path, monkeypatch):
    """``~`` in the input path is expanded to $HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    out = require_path("~")
    assert out == tmp_path


# ---------------------------------------------------------------------------
# get_cache_dir — repo-anchored.
# ---------------------------------------------------------------------------


def test_get_cache_dir_anchored_to_repo():
    """Resolves to <repo_root>/.cache regardless of cwd, and creates it."""
    out = get_cache_dir()
    assert out.is_dir()
    assert out.name == ".cache"
    # The cache lives one level above ``src/`` (the repo root).
    repo_root = Path(__file__).resolve().parents[2]
    assert out == repo_root / ".cache"
