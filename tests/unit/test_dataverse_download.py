"""Unit tests for the Dataverse download CLI.

These don't hit the network — they check the AST walker that discovers
required basenames from ``figures/**/*.py`` and the destination
resolution logic. The end-to-end download is exercised manually.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ballpushing_utils.dataverse_download import (
    collect_required_basenames,
    resolve_destination,
)
from ballpushing_utils.dataverse_naming import BASENAME_TO_ARCHIVE


def test_collect_required_basenames_returns_archives_for_known_files():
    """Every figure-required basename rolls up to a known archive."""
    required = collect_required_basenames()
    # At minimum every paper archive contributes at least one feather.
    assert "affordance" in required
    assert "screen" in required
    assert "exploration" in required
    # And every returned basename has a registered archive mapping.
    for archive, basenames in required.items():
        for basename in basenames:
            assert BASENAME_TO_ARCHIVE.get(basename) == archive


def test_collect_required_basenames_includes_magnetblock_and_screen():
    required = collect_required_basenames()
    assert (
        "Magnetblock_ballpushing_metrics.feather" in required["affordance"]
    )
    assert (
        "ballpushing_metrics_silencing_screen.feather" in required["screen"]
    )
    assert (
        "Wild-Type_ballpushing_metrics.feather" in required["exploration"]
    )


def test_resolve_destination_explicit_arg(tmp_path):
    out = resolve_destination(tmp_path / "explicit")
    assert out == (tmp_path / "explicit").resolve() or out == tmp_path / "explicit"
    assert out.is_dir()


def test_resolve_destination_env_var(monkeypatch, tmp_path):
    env_dest = tmp_path / "env_dest"
    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(env_dest))
    out = resolve_destination(None)
    assert out == env_dest
    assert out.is_dir()


def test_resolve_destination_repo_fallback(monkeypatch):
    monkeypatch.delenv("BALLPUSHING_DATA_ROOT", raising=False)
    out = resolve_destination(None)
    # When the env var is unset, fall back to <repo>/Datasets/.
    assert out.name == "Datasets"
    assert out.is_dir()
