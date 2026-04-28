"""Unit tests for the legacy ``velocity → speed`` column-rename shim.

Datasets published on Harvard Dataverse before the 2026 rename
(``83f939d``) still carry the old column names. The
:mod:`ballpushing_utils.compat` module provides the back-compat
``normalize_legacy_columns`` helper that aliases the new column names
on top of legacy ones at load time. These tests guard the behaviour
that:

  * legacy-only frames get all the modern aliases added;
  * the per-keypoint per-frame ``{kp}_frame{N}_velocity`` pattern is
    handled too, but ``angular_velocity`` (a signed quantity, kept as
    a real velocity) is **not** rewritten;
  * modern frames are unmodified;
  * the helper is idempotent;
  * pre-existing modern columns are never clobbered by the legacy
    fallback (so user data wins over alias).
"""

from __future__ import annotations

import pandas as pd
import pytest

from ballpushing_utils.compat import (
    LEGACY_COLUMN_RENAMES,
    normalize_legacy_columns,
)


def test_legacy_only_frame_gets_modern_aliases():
    df = pd.DataFrame(
        {
            "velocity": [1.0, 2.0, 3.0],
            "normalized_velocity": [0.1, 0.2, 0.3],
            "velocity_trend": [10.0, 20.0, 30.0],
            "velocity_during_interactions": [5.0, 6.0, 7.0],
        }
    )
    normalize_legacy_columns(df)
    for new in LEGACY_COLUMN_RENAMES:
        assert new in df.columns, f"missing alias {new}"
        legacy = LEGACY_COLUMN_RENAMES[new]
        # Aliased values match the legacy column.
        pd.testing.assert_series_equal(
            df[new].reset_index(drop=True),
            df[legacy].reset_index(drop=True),
            check_names=False,
        )


def test_per_keypoint_per_frame_velocity_renames():
    """``{kp}_frame{N}_velocity`` becomes ``{kp}_frame{N}_speed``."""
    df = pd.DataFrame(
        {
            "thorax_frame0_velocity": [1.0],
            "head_frame100_velocity": [2.0],
        }
    )
    normalize_legacy_columns(df)
    assert "thorax_frame0_speed" in df.columns
    assert "head_frame100_speed" in df.columns
    # Originals stay too — we only add aliases, never drop columns.
    assert "thorax_frame0_velocity" in df.columns
    assert "head_frame100_velocity" in df.columns


def test_angular_velocity_is_preserved():
    """``angular_velocity`` is signed (np.diff of angles) — keep it as-is."""
    df = pd.DataFrame(
        {
            "thorax_frame0_angular_velocity": [-0.1, 0.0, 0.1],
        }
    )
    normalize_legacy_columns(df)
    # No ``..._speed`` alias should appear for the angular velocity column.
    assert "thorax_frame0_angular_speed" not in df.columns
    assert "thorax_frame0_speed" not in df.columns


def test_modern_only_frame_is_noop():
    """Frames that already have ``speed*`` columns are returned unchanged."""
    df = pd.DataFrame({"speed": [1, 2], "normalized_speed": [0.5, 0.6]})
    before = list(df.columns)
    normalize_legacy_columns(df)
    assert list(df.columns) == before


def test_idempotent():
    """Calling the helper twice yields the same column set as once."""
    df = pd.DataFrame({"velocity": [1, 2, 3]})
    normalize_legacy_columns(df)
    cols_after_one = list(df.columns)
    normalize_legacy_columns(df)
    assert list(df.columns) == cols_after_one


def test_modern_column_not_clobbered_by_alias():
    """If both old and new are present, the existing modern data wins."""
    df = pd.DataFrame({"speed": [99], "velocity": [1]})
    normalize_legacy_columns(df)
    # ``speed`` was already present — must not be overwritten with
    # the legacy ``velocity`` column.
    assert df["speed"].iloc[0] == 99


def test_copy_returns_new_frame():
    """``copy=True`` leaves the input untouched."""
    df = pd.DataFrame({"velocity": [1, 2]})
    out = normalize_legacy_columns(df, copy=True)
    assert "speed" not in df.columns
    assert "speed" in out.columns


@pytest.mark.parametrize(
    "modern, legacy",
    sorted(LEGACY_COLUMN_RENAMES.items()),
    ids=sorted(LEGACY_COLUMN_RENAMES),
)
def test_each_documented_rename(modern, legacy):
    """Each entry in ``LEGACY_COLUMN_RENAMES`` actually works."""
    df = pd.DataFrame({legacy: [1.0, 2.0]})
    normalize_legacy_columns(df)
    assert modern in df.columns
