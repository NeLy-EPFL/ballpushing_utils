"""Unit tests for ``ballpushing_utils.dataverse``.

Covers the public API users hit when running the figure pipeline against
the published Dataverse archives:

- :func:`parse_archive_name` — decodes the archive folder name into
  ``(experiment_type, fields)`` for every paradigm-naming convention.
- :func:`expand_condition` — single-column default + F1 + MagnetBlock
  transformers.
- :func:`synthesize_experiment_metadata` — round-trips with the
  ``FlyMetadata.get_arena_metadata`` lookup.
- :func:`iter_dataverse_flies` — both layouts (multi-condition + flat
  single-condition) on synthetic trees under ``tmp_path``.
- :func:`is_dataverse_layout` — heuristic on a corridor-shaped folder.
- :func:`detect_dataverse_experiment_type` — paradigm inference from
  folder shape.

All tests are hermetic: no LFS fixtures, no lab share, no h5py.
"""

from __future__ import annotations

import pytest

from ballpushing_utils import (
    ARCHIVE_PREFIX_RECIPES,
    CONDITION_TRANSFORMERS,
    DEFAULT_CONDITION_FIELD,
    detect_dataverse_experiment_type,
    expand_condition,
    is_dataverse_layout,
    iter_dataverse_flies,
    parse_archive_name,
    synthesize_experiment_metadata,
)


# ---------------------------------------------------------------------------
# parse_archive_name — table-driven over every published naming pattern.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_etype, expected_fields",
    [
        # MagnetBlock — case-insensitive exact match.
        ("MagnetBlock-Blocked", "MagnetBlock", {"Magnet": "y"}),
        ("MagnetBlock-Control", "MagnetBlock", {"Magnet": "n"}),
        ("magnetblock-blocked", "MagnetBlock", {"Magnet": "y"}),  # lowercase OK
        # F1 (Generalisation) — multi-column transformer; genotype can have hyphens.
        (
            "Generalisation-Wild-type-control",
            "F1",
            {"Pretraining": "n", "F1_condition": "control", "Genotype": "Wild-type"},
        ),
        (
            "Generalisation-Wild-type-pretrained",
            "F1",
            {"Pretraining": "y", "F1_condition": "pretrained", "Genotype": "Wild-type"},
        ),
        (
            "Generalisation-TNTxLC10-2-pretrained_unlocked",
            "F1",
            {
                "Pretraining": "y",
                "F1_condition": "pretrained_unlocked",
                "Genotype": "TNTxLC10-2",
            },
        ),
        (
            "Generalisation-MB247xTNT-control",
            "F1",
            {"Pretraining": "n", "F1_condition": "control", "Genotype": "MB247xTNT"},
        ),
        # Single-prefix supplementary archives.
        ("Ballscents-New", "TNT", {"BallScent": "New"}),
        ("Ballscents-New-plus-Pre-exposed", "TNT", {"BallScent": "New-plus-Pre-exposed"}),
        ("Balltype-Marble", "TNT", {"BallType": "Marble"}),
        ("Balltype-Rusty", "TNT", {"BallType": "Rusty"}),
        ("Multi-trials-1", "Learning", {"Trial_duration": "1"}),
        ("Multi-trials-3", "Learning", {"Trial_duration": "3"}),
        # Triple-field feeding-state pattern: <Genotype>_Lights-<L>_<FS>[-N].
        (
            "Wild-type_Lights-off_Fed-1",
            "TNT",
            {"Genotype": "Wild-type", "Light": "off", "FeedingState": "Fed"},
        ),
        (
            "Wild-type_Lights-on_Starved",
            "TNT",
            {"Genotype": "Wild-type", "Light": "on", "FeedingState": "Starved"},
        ),
        (
            "Wild-type_Lights-on_Starved-without-water-2",
            "TNT",
            {
                "Genotype": "Wild-type",
                "Light": "on",
                "FeedingState": "Starved-without-water",
            },
        ),
        # <Genotype>-Light-{on,off} dual-field olfaction-dark pattern.
        (
            "TNTxEmptyGal4-Light-off",
            "TNT",
            {"Genotype": "TNTxEmptyGal4", "Light": "off"},
        ),
        ("TNTxIR8a-Light-on", "TNT", {"Genotype": "TNTxIR8a", "Light": "on"}),
        # Default (silencing screen): single-token genotype name.
        ("LC6", "TNT", {"Genotype": "LC6"}),
        ("MB247xTNT", "TNT", {"Genotype": "MB247xTNT"}),
        # Critical disambiguation: 2-underscore wild-type strain names must
        # NOT match the feeding-state parser (which requires 3+ tokens AND
        # a Lights-prefixed second token).
        ("Wild-type_PR", "TNT", {"Genotype": "Wild-type_PR"}),
        ("Wild-type_CS", "TNT", {"Genotype": "Wild-type_CS"}),
        # Mixed hyphens + underscores in genotype names fall through cleanly.
        (
            "MBON-gamma1pedc_to_alpha_beta",
            "TNT",
            {"Genotype": "MBON-gamma1pedc_to_alpha_beta"},
        ),
    ],
)
def test_parse_archive_name(name, expected_etype, expected_fields):
    etype, fields = parse_archive_name(name)
    assert etype == expected_etype
    assert fields == expected_fields


# ---------------------------------------------------------------------------
# expand_condition — single-column default + multi-column transformers.
# ---------------------------------------------------------------------------


def test_expand_condition_single_column_default_per_experiment_type():
    # Defaults: TNT -> Genotype, MagnetBlock -> Magnet (transformer overrides
    # this to map Blocked/Control to y/n), F1 -> F1_condition, Learning -> Pretraining.
    assert expand_condition("TNT", "MB247xTNT") == {"Genotype": "MB247xTNT"}
    assert expand_condition("Learning", "y") == {"Pretraining": "y"}


def test_expand_condition_condition_field_override():
    """For paradigms WITHOUT a registered transformer, --condition-field
    overrides the per-experiment-type default column.
    """
    assert expand_condition("TNT", "Marble", condition_field="BallType") == {
        "BallType": "Marble"
    }
    assert expand_condition("TNT", "starved_noWater", condition_field="FeedingState") == {
        "FeedingState": "starved_noWater"
    }


def test_expand_condition_f1_transformer_emits_two_columns():
    """F1 has a registered transformer that emits Pretraining + F1_condition
    from a single condition value.
    """
    assert expand_condition("F1", "control") == {
        "Pretraining": "n",
        "F1_condition": "control",
    }
    assert expand_condition("F1", "pretrained") == {
        "Pretraining": "y",
        "F1_condition": "pretrained",
    }
    assert expand_condition("F1", "pretrained_unlocked") == {
        "Pretraining": "y",
        "F1_condition": "pretrained_unlocked",
    }


def test_expand_condition_magnetblock_transformer_maps_to_y_n():
    """MagnetBlock transformer maps Blocked/Unblocked → y/n."""
    assert expand_condition("MagnetBlock", "Blocked") == {"Magnet": "y"}
    assert expand_condition("MagnetBlock", "Unblocked") == {"Magnet": "n"}


def test_expand_condition_unknown_experiment_type_raises():
    with pytest.raises(ValueError, match="No default condition field"):
        expand_condition("NotAParadigm", "value")


# ---------------------------------------------------------------------------
# synthesize_experiment_metadata — round-trip with FlyMetadata's lookup.
# ---------------------------------------------------------------------------


def test_synthesize_experiment_metadata_round_trip():
    """Output schema must match what ``FlyMetadata.get_arena_metadata``
    expects: a dict keyed by metadata variable, each holding a per-arena
    dict with lower-cased arena keys.
    """
    fields = {"Genotype": "MB247xTNT", "Magnet": "y"}
    meta = synthesize_experiment_metadata("Arena3", fields)

    assert meta == {
        "Genotype": {"arena3": "MB247xTNT"},
        "Magnet": {"arena3": "y"},
    }

    # Replicate the FlyMetadata lookup.
    arena_metadata = {var: data["arena3"] for var, data in meta.items() if "arena3" in data}
    assert arena_metadata == fields


def test_synthesize_experiment_metadata_lowercases_arena_key():
    meta = synthesize_experiment_metadata("ARENA1", {"X": "v"})
    assert "arena1" in meta["X"]
    assert "ARENA1" not in meta["X"]


# ---------------------------------------------------------------------------
# iter_dataverse_flies + is_dataverse_layout on synthetic trees.
# ---------------------------------------------------------------------------


def _make_corridor(parent, h5_files=("ball.h5", "fly.h5", "skeleton.h5")):
    parent.mkdir(parents=True, exist_ok=True)
    for name in h5_files:
        (parent / name).write_bytes(b"")  # zero-length is fine; only filename is matched


def test_iter_dataverse_flies_multi_condition_layout(tmp_path):
    """<root>/<condition>/<date>/arenaN/corridor/*.h5 — the standard layout."""
    _make_corridor(tmp_path / "MB247xTNT" / "231222" / "arena6" / "corridor1")
    _make_corridor(tmp_path / "MB247xTNT" / "231222" / "arena6" / "corridor5")
    _make_corridor(tmp_path / "MB247xTNT" / "240131-2" / "arena9" / "corridor3")
    _make_corridor(tmp_path / "EmptySplit" / "231130" / "arena2" / "corridor5")
    # Junk folders that should be ignored:
    (tmp_path / "MB247xTNT" / "README_metadata").mkdir()
    (tmp_path / "MB247xTNT" / "231222" / "junk_not_arena").mkdir()  # missing 'arena' prefix

    flies = list(iter_dataverse_flies(tmp_path))
    assert len(flies) == 4
    conditions = {f.condition for f in flies}
    assert conditions == {"MB247xTNT", "EmptySplit"}
    dates = {f.date_folder.name for f in flies}
    assert dates == {"231222", "240131-2", "231130"}
    arenas = {f.arena for f in flies}
    assert arenas == {"arena6", "arena9", "arena2"}


def test_iter_dataverse_flies_flat_layout(tmp_path):
    """<root>/<date>/arenaN/corridor/*.h5 where root IS the condition folder.

    The condition is then extracted from the last hyphen-segment of the
    root's name, e.g. ``Generalisation-Wild-type-pretrained`` → ``pretrained``.
    """
    root = tmp_path / "Generalisation-Wild-type-pretrained"
    _make_corridor(root / "251008" / "arena5" / "Right")
    _make_corridor(root / "251008" / "arena5" / "Left")

    flies = list(iter_dataverse_flies(root))
    assert len(flies) == 2
    # In the flat layout the .condition is taken from the last hyphen-segment
    # of root.name (legacy behaviour). The fly's date_folder is the date dir.
    assert {f.condition for f in flies} == {"pretrained"}
    assert {f.corridor for f in flies} == {"Right", "Left"}


def test_iter_dataverse_flies_skips_corridor_without_ball_h5(tmp_path):
    """A corridor folder without any *ball*.h5 is skipped silently."""
    _make_corridor(tmp_path / "G" / "231222" / "arena1" / "corridor1")
    (tmp_path / "G" / "231222" / "arena1" / "corridor2").mkdir(parents=True)
    # corridor2 has no ball file — should be skipped.
    flies = list(iter_dataverse_flies(tmp_path))
    assert len(flies) == 1
    assert flies[0].corridor == "corridor1"


def test_iter_dataverse_flies_raises_on_nonexistent_root(tmp_path):
    with pytest.raises(NotADirectoryError):
        list(iter_dataverse_flies(tmp_path / "no" / "such" / "dir"))


def test_is_dataverse_layout_positive(tmp_path):
    corridor = tmp_path / "MB247xTNT" / "231222" / "arena6" / "corridor1"
    _make_corridor(corridor)
    assert is_dataverse_layout(corridor) is True


def test_is_dataverse_layout_rejects_when_metadata_json_present(tmp_path):
    corridor = tmp_path / "exp" / "arena1" / "corridor1"
    _make_corridor(corridor)
    # Adding Metadata.json at <date>'s parent (= what would be the experiment
    # root in the on-server layout) means this is server-style, not dataverse.
    (corridor.parent.parent / "Metadata.json").write_text("{}")
    assert is_dataverse_layout(corridor) is False


def test_is_dataverse_layout_rejects_non_date_grandparent(tmp_path):
    corridor = tmp_path / "MB247xTNT" / "not-a-date" / "arena6" / "corridor1"
    _make_corridor(corridor)
    assert is_dataverse_layout(corridor) is False


# ---------------------------------------------------------------------------
# detect_dataverse_experiment_type — paradigm inference.
# ---------------------------------------------------------------------------


def test_detect_dataverse_experiment_type_magnetblock_keyword(tmp_path):
    """The string 'magnet' anywhere in the root path triggers MagnetBlock."""
    root = tmp_path / "MagnetBlock_extract"
    _make_corridor(root / "Blocked" / "240711" / "arena4" / "corridor4")
    assert detect_dataverse_experiment_type(root) == "MagnetBlock"


def test_detect_dataverse_experiment_type_magnetblock_y_n_folders(tmp_path):
    """All immediate children named y/n → MagnetBlock."""
    _make_corridor(tmp_path / "y" / "240711" / "arena4" / "corridor4")
    _make_corridor(tmp_path / "n" / "240711" / "arena5" / "corridor1")
    assert detect_dataverse_experiment_type(tmp_path) == "MagnetBlock"


def test_detect_dataverse_experiment_type_f1_left_right(tmp_path):
    """Left/Right corridor names → F1 paradigm."""
    _make_corridor(tmp_path / "pretrained" / "251008" / "arena5" / "Right")
    _make_corridor(tmp_path / "pretrained" / "251008" / "arena5" / "Left")
    assert detect_dataverse_experiment_type(tmp_path) == "F1"


def test_detect_dataverse_experiment_type_default_tnt(tmp_path):
    """Anything else falls back to TNT (the silencing-screen default)."""
    _make_corridor(tmp_path / "LC6" / "231222" / "arena1" / "corridor1")
    assert detect_dataverse_experiment_type(tmp_path) == "TNT"


# ---------------------------------------------------------------------------
# Sanity: registry tables stay in sync with what parse_archive_name uses.
# ---------------------------------------------------------------------------


def test_registry_tables_consistent():
    """The simple-prefix recipes table only contains paradigms we have a
    DEFAULT_CONDITION_FIELD entry for, and the F1/MagnetBlock paradigms
    have transformers (which is why they're NOT in ARCHIVE_PREFIX_RECIPES).
    """
    for _prefix, (etype, _column) in ARCHIVE_PREFIX_RECIPES.items():
        assert etype in DEFAULT_CONDITION_FIELD, etype
    # F1 + MagnetBlock get multi-column treatment.
    assert "F1" in CONDITION_TRANSFORMERS
    assert "MagnetBlock" in CONDITION_TRANSFORMERS
