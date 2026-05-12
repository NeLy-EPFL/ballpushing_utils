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


# ---------------------------------------------------------------------------
# read_feather — file-not-found behaviour + basename fallback search.
# These exercise the path that turns "data not on this machine" from a bare
# Arrow IO error into the structured three-source breadcrumb.
# ---------------------------------------------------------------------------


def test_read_feather_missing_file_raises_breadcrumb(tmp_path, monkeypatch):
    """When the feather doesn't resolve anywhere, raise FileNotFoundError
    with the standard 'three sources' breadcrumb that names DATA_ROOT,
    Dataverse, and sample_data.
    """
    from ballpushing_utils import read_feather

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        read_feather("nope/nothing-here.feather")
    msg = str(excinfo.value)
    assert "BALLPUSHING_DATA_ROOT" in msg
    assert "Dataverse" in msg
    assert "sample" in msg.lower()


def test_read_feather_basename_fallback_under_data_root(tmp_path, monkeypatch):
    """If the literal path doesn't resolve, read_feather walks
    BALLPUSHING_DATA_ROOT recursively for the basename and uses the
    single match it finds.
    """
    from ballpushing_utils import read_feather

    # Build a real (but tiny) feather under DATA_ROOT at a different path
    # than what the script will ask for.
    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    real = tmp_path / "deep" / "nested" / "pooled_summary.feather"
    real.parent.mkdir(parents=True)
    pd.DataFrame({"a": [1, 2, 3]}).to_feather(real)

    df = read_feather("any/script/relative/path/pooled_summary.feather")
    assert list(df["a"]) == [1, 2, 3]


def test_read_feather_basename_fallback_under_search_env(tmp_path, monkeypatch):
    """The same fallback works via BALLPUSHING_FEATHER_SEARCH for users who
    keep their feathers outside the data root.
    """
    from ballpushing_utils import read_feather

    # Empty data root so rule-2 / rule-4 produce nothing.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(empty))

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    pd.DataFrame({"speed_trend": [0.5]}).to_feather(downloads / "pooled_summary.feather")
    monkeypatch.setenv("BALLPUSHING_FEATHER_SEARCH", str(downloads))

    df = read_feather("Some/Other/Path/pooled_summary.feather")
    assert "speed_trend" in df.columns


def test_read_feather_ambiguous_basename_raises_breadcrumb(tmp_path, monkeypatch):
    """Multiple basename matches under DATA_ROOT → the search punts and the
    breadcrumb fires (so the user can disambiguate via FEATHER_SEARCH).
    """
    from ballpushing_utils import read_feather

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    a = tmp_path / "a" / "pooled.feather"
    b = tmp_path / "b" / "pooled.feather"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        pd.DataFrame({"x": [0]}).to_feather(p)

    with pytest.raises(FileNotFoundError):
        read_feather("anywhere/pooled.feather")


def test_read_feather_dataverse_alias_single_file(tmp_path, monkeypatch):
    """A server-style path that maps to a Dataverse basename loads cleanly."""
    from ballpushing_utils import read_feather

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    pd.DataFrame({"speed_trend": [0.1, 0.2]}).to_feather(
        tmp_path / "Magnetblock_ballpushing_metrics.feather"
    )

    df = read_feather(
        "MagnetBlock/Datasets/some_label_Data/summary/pooled_summary.feather"
    )
    assert list(df["speed_trend"]) == [0.1, 0.2]


def test_read_feather_dataverse_split_parts_concatenated(tmp_path, monkeypatch):
    """A multi-part Dataverse feather is loaded by concatenating parts in
    numeric order, with legacy column shimming applied to the result.
    """
    from ballpushing_utils import read_feather

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    # Three parts with legacy column name; the shim must alias on the
    # concatenated frame.
    for i, vals in [(1, [1, 2]), (2, [3, 4]), (3, [5])]:
        pd.DataFrame({"velocity_trend": vals}).to_feather(
            tmp_path / f"Generalisation-TNT_trajectories-{i}.feather"
        )

    df = read_feather(
        "F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data"
        "/coordinates/pooled_coordinates.feather"
    )
    assert list(df["velocity_trend"]) == [1, 2, 3, 4, 5]
    assert "speed_trend" in df.columns
    assert list(df["speed_trend"]) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# iter_coordinate_feathers — on-server vs Dataverse layout.
# ---------------------------------------------------------------------------


def test_iter_coordinate_feathers_on_server_layout(tmp_path, monkeypatch):
    """If the directory exists with *_coordinates.feather, iterate them."""
    from ballpushing_utils import iter_coordinate_feathers

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    coords_dir = (
        tmp_path
        / "Ballpushing_Exploration"
        / "Datasets"
        / "260220_10_summary_control_folders_Data"
        / "coordinates"
    )
    coords_dir.mkdir(parents=True)
    for stem in ("230704_A", "230705_B"):
        pd.DataFrame(
            {"fly": [1, 2], "distance_ball_0": [0.1, 0.2]}
        ).to_feather(coords_dir / f"{stem}_coordinates.feather")

    pairs = list(iter_coordinate_feathers(coords_dir))
    labels = [label for label, _ in pairs]
    assert sorted(labels) == ["230704_A_coordinates", "230705_B_coordinates"]
    for _, df in pairs:
        assert set(df.columns) >= {"fly", "distance_ball_0"}


def test_iter_coordinate_feathers_dataverse_layout(tmp_path, monkeypatch):
    """If the directory is missing, fall back to the per-condition pools."""
    from ballpushing_utils import iter_coordinate_feathers

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    # Two Wild-type condition feathers with two distinct experiments each.
    pd.DataFrame(
        {
            "experiment": [
                "230704_FeedingState_1_AM_Videos_Tracked",
                "230704_FeedingState_1_AM_Videos_Tracked",
                "230612_FeedingState_FedDay_Videos_Tracked",
            ],
            "fly": [1, 2, 3],
            "distance_ball_0": [0.1, 0.2, 0.3],
        }
    ).to_feather(tmp_path / "Wild-type_Lights-on_Fed_trajectories.feather")
    pd.DataFrame(
        {
            "experiment": [
                "230705_FeedingState_Starved_Videos_Tracked",
                "230705_FeedingState_Starved_Videos_Tracked",
            ],
            "fly": [4, 5],
            "distance_ball_0": [0.4, 0.5],
        }
    ).to_feather(tmp_path / "Wild-type_Lights-on_Starved_trajectories.feather")

    coords_dir = tmp_path / "Ballpushing_Exploration/Datasets/X_Data/coordinates"
    # Don't create the directory — exercise the Dataverse fallback.
    pairs = list(iter_coordinate_feathers(coords_dir))
    labels = sorted(label for label, _ in pairs)
    # One yield per (experiment, condition-feather) — three experiments total.
    assert labels == [
        "230612_FeedingState_FedDay_Videos_Tracked",
        "230704_FeedingState_1_AM_Videos_Tracked",
        "230705_FeedingState_Starved_Videos_Tracked",
    ]
    by_label = {label: df for label, df in pairs}
    assert len(by_label["230704_FeedingState_1_AM_Videos_Tracked"]) == 2
    assert len(by_label["230705_FeedingState_Starved_Videos_Tracked"]) == 2


def test_iter_coordinate_feathers_unknown_directory_raises(tmp_path, monkeypatch):
    from ballpushing_utils import iter_coordinate_feathers

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        list(iter_coordinate_feathers(tmp_path / "Some/Unmapped/Path/coordinates"))


def test_load_wildtype_experiment_slices_pool(tmp_path, monkeypatch):
    from ballpushing_utils import load_wildtype_experiment

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    pd.DataFrame(
        {
            "experiment": [
                "230704_FeedingState_1_AM_Videos_Tracked",
                "OTHER",
                "230704_FeedingState_1_AM_Videos_Tracked",
            ],
            "fly": [1, 2, 3],
            "y_ball_0": [10.0, 20.0, 15.0],
        }
    ).to_feather(tmp_path / "Wild-type_Lights-on_Fed_trajectories.feather")

    df = load_wildtype_experiment("230704_FeedingState_1_AM_Videos_Tracked")
    assert list(df["fly"]) == [1, 3]


def test_load_wildtype_experiment_missing_raises(tmp_path, monkeypatch):
    from ballpushing_utils import load_wildtype_experiment

    monkeypatch.setenv("BALLPUSHING_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("BALLPUSHING_FEATHER_SEARCH", raising=False)

    # No pool feathers at all.
    with pytest.raises(FileNotFoundError):
        load_wildtype_experiment("anything")

    # Pool present but experiment not in it.
    pd.DataFrame({"experiment": ["A"], "fly": [1]}).to_feather(
        tmp_path / "Wild-type_Lights-on_Fed_trajectories.feather"
    )
    with pytest.raises(ValueError):
        load_wildtype_experiment("B")
    assert load_wildtype_experiment("B", allow_empty=True).empty
