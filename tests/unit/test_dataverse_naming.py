"""Compile-time invariants for the Dataverse name mapping.

These tests are hermetic — they never touch the network or
``$BALLPUSHING_DATA_ROOT``. They check that:

1. Every ``dataset(...)`` literal in ``figures/**/*.py`` is either
   (a) covered by ``SERVER_TO_DATAVERSE`` so external users can
   reproduce the figure from the Dataverse download, or (b) listed in
   :data:`KNOWN_UNMAPPED` with the reason it's deliberately not
   covered.
2. Every basename in the value lists is tagged in
   :data:`BASENAME_TO_ARCHIVE` so the download CLI knows which DOI to
   query for it.
3. The :func:`expand_split_parts` helper handles both single-file and
   numerically-suffixed split layouts and ignores look-alike basenames.

When :data:`SERVER_TO_DATAVERSE` grows, this test grows with it. Edit
:data:`KNOWN_UNMAPPED` only when adding a deliberate exception.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ballpushing_utils.dataverse_naming import (
    BASENAME_TO_ARCHIVE,
    SCREEN_STANDARDIZED_CONTACTS_FEATHERS,
    SERVER_DIRECTORY_TO_DATAVERSE,
    SERVER_TO_DATAVERSE,
    dataverse_candidates,
    dataverse_directory_candidates,
    expand_split_parts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "figures"


# Literal paths in figure scripts that resolve to either a directory or a
# per-fly artefact rather than a published Dataverse basename. Each entry
# is annotated with the reason it can't (yet) be mapped one-to-one to a
# basename. These scripts still work for lab users with on-server data;
# converting them to the Dataverse layout is tracked separately.
KNOWN_UNMAPPED: dict[str, str] = {
    # Directory paths — handled by
    # :func:`ballpushing_utils.iter_coordinate_feathers` at runtime
    # (resolves to per-condition Dataverse pools or per-experiment
    # on-server files). Listed here so the test still rejects unmapped
    # *file* literals that slip in.
    "Ballpushing_Exploration/Datasets/260220_10_summary_control_folders_Data/coordinates": (
        "directory iteration handled by iter_coordinate_feathers (SERVER_DIRECTORY_TO_DATAVERSE)"
    ),
    "Ballpushing_Exploration/Datasets/250806_10_coordinates_control_folders_Data/Other/coordinates": (
        "directory iteration handled by iter_coordinate_feathers (SERVER_DIRECTORY_TO_DATAVERSE)"
    ),
    # Per-fly coordinate feather — Fig 1 default reads a specific
    # cohort. Each script catches FileNotFoundError and falls back to
    # ``load_wildtype_experiment(DEFAULT_EXPERIMENT_NAME)`` which slices
    # the published Wild-type pools by experiment column.
    (
        "Ballpushing_Exploration/Datasets/260220_10_summary_control_folders_Data"
        "/coordinates/230704_FeedingState_1_AM_Videos_Tracked_coordinates.feather"
    ): "per-fly cohort; script falls back to load_wildtype_experiment()",
}


def test_directory_keys_match_server_dir_patterns() -> None:
    """Every server-style directory literal in KNOWN_UNMAPPED should match
    a pattern in SERVER_DIRECTORY_TO_DATAVERSE so iter_coordinate_feathers
    knows what to fall back to.
    """
    for literal, reason in KNOWN_UNMAPPED.items():
        if "iter_coordinate_feathers" not in reason:
            continue
        assert dataverse_directory_candidates(literal) is not None, (
            f"{literal!r} is listed as handled by iter_coordinate_feathers "
            f"but no pattern in SERVER_DIRECTORY_TO_DATAVERSE matches it."
        )


def _dataset_literals_in(py_path: Path) -> list[str]:
    """Return every string literal passed to ``dataset(...)`` in the file."""
    try:
        tree = ast.parse(py_path.read_text())
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dataset"
            and node.args
        ):
            arg = node.args[0]
            # Single literal: dataset("foo/bar.feather")
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg.value)
            # Implicit string concat: dataset("foo/" "bar.feather")
            elif isinstance(arg, ast.JoinedStr):
                pieces = [
                    v.value
                    for v in arg.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                ]
                if pieces:
                    out.append("".join(pieces))
    return out


def _all_figure_dataset_literals() -> list[tuple[str, str]]:
    """Walk ``figures/**/*.py`` skipping ``old/`` paths.

    Returns a list of ``(repo-relative file path, literal)``.
    """
    out: list[tuple[str, str]] = []
    for py in sorted(FIGURES_DIR.rglob("*.py")):
        if "/old/" in str(py) or py.name == "run_all_panels.py":
            continue
        rel = str(py.relative_to(REPO_ROOT))
        for lit in _dataset_literals_in(py):
            out.append((rel, lit))
    return out


def test_every_figure_dataset_call_resolves_or_is_listed() -> None:
    """Every ``dataset(...)`` string in figures/ is mapped or listed as unmapped."""
    unmapped: list[tuple[str, str]] = []
    for source, lit in _all_figure_dataset_literals():
        if dataverse_candidates(lit) is not None:
            continue
        if lit in KNOWN_UNMAPPED:
            continue
        unmapped.append((source, lit))
    assert not unmapped, (
        "Unmapped dataset() literals — either add an entry to "
        "SERVER_TO_DATAVERSE in dataverse_naming.py, or document the "
        "exception in KNOWN_UNMAPPED in this test:\n"
        + "\n".join(f"  {f}: {lit}" for f, lit in unmapped)
    )


def test_every_basename_has_an_archive() -> None:
    expected = {b for v in SERVER_TO_DATAVERSE.values() for b in v}
    missing = expected - set(BASENAME_TO_ARCHIVE)
    assert not missing, f"Basenames missing from BASENAME_TO_ARCHIVE: {sorted(missing)}"


def test_archive_keys_are_known() -> None:
    from ballpushing_utils.dataverse_naming import DATAVERSE_DOIS

    bad = {b: a for b, a in BASENAME_TO_ARCHIVE.items() if a not in DATAVERSE_DOIS}
    assert not bad, f"Unknown archive labels in BASENAME_TO_ARCHIVE: {bad}"


def test_expand_split_parts_single_file(tmp_path: Path) -> None:
    (tmp_path / "foo.feather").touch()
    parts = expand_split_parts("foo.feather", tmp_path)
    assert parts == [(tmp_path / "foo.feather").resolve()]


def test_expand_split_parts_numeric_split(tmp_path: Path) -> None:
    for i in (1, 2, 3):
        (tmp_path / f"foo-{i}.feather").touch()
    parts = expand_split_parts("foo.feather", tmp_path)
    assert [p.name for p in parts] == [
        "foo-1.feather",
        "foo-2.feather",
        "foo-3.feather",
    ]


def test_expand_split_parts_numeric_split_orders_double_digits(tmp_path: Path) -> None:
    for i in (1, 2, 10, 11):
        (tmp_path / f"foo-{i}.feather").touch()
    parts = expand_split_parts("foo.feather", tmp_path)
    assert [p.name for p in parts] == [
        "foo-1.feather",
        "foo-2.feather",
        "foo-10.feather",
        "foo-11.feather",
    ]


def test_expand_split_parts_ignores_lookalikes(tmp_path: Path) -> None:
    for i in (1, 2):
        (tmp_path / f"foo-{i}.feather").touch()
    (tmp_path / "foo-old-backup.feather").touch()
    (tmp_path / "foobar-1.feather").touch()
    parts = expand_split_parts("foo.feather", tmp_path)
    assert [p.name for p in parts] == ["foo-1.feather", "foo-2.feather"]


def test_expand_split_parts_empty_when_no_match(tmp_path: Path) -> None:
    assert expand_split_parts("foo.feather", tmp_path) == []


@pytest.mark.parametrize(
    "server_path, expected_first_basename",
    [
        (
            "MagnetBlock/Datasets/251126_11_summary_magnet_block_folders_Data/summary/pooled_summary.feather",
            "Magnetblock_ballpushing_metrics.feather",
        ),
        (
            "F1_Tracks/Datasets/260121_16_F1_coordinates_F1_New_Data/summary/pooled_summary.feather",
            "Generalisation-Wild-type_ballpushing_metrics.feather",
        ),
        (
            "F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/summary/pooled_summary.feather",
            "Generalisation-TNT_ballpushing_metrics.feather",
        ),
        (
            "Ballpushing_TNTScreen/Datasets/250811_18_summary_TNT_screen_Data/summary/pooled_summary.feather",
            "ballpushing_metrics_silencing_screen.feather",
        ),
    ],
)
def test_dataverse_candidates_known_paths(
    server_path: str, expected_first_basename: str
) -> None:
    result = dataverse_candidates(server_path)
    assert result is not None
    assert result[0] == expected_first_basename


def test_dataverse_candidates_unknown_returns_none() -> None:
    assert dataverse_candidates("does/not/exist/in/mapping.feather") is None


# ---------------------------------------------------------------------------
# standardized contacts — directory resolution and archive tagging.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server_dir",
    [
        "Ballpushing_TNTScreen/Datasets/250809_02_standardized_contacts_TNT_screen_Data/standardized_contacts",
        # Wildcard path as used at runtime (timestamp segment varies).
        "Ballpushing_TNTScreen/Datasets/250809_02_standardized_contacts_Data/standardized_contacts",
    ],
)
def test_dataverse_directory_candidates_standardized_contacts(server_dir: str) -> None:
    """dataverse_directory_candidates resolves any timestamped standardized_contacts
    directory under the TNT screen to the full set of brain-region feathers."""
    result = dataverse_directory_candidates(server_dir)
    assert result is not None, (
        f"dataverse_directory_candidates returned None for {server_dir!r}; "
        "SERVER_DIRECTORY_TO_DATAVERSE may be missing the standardized_contacts pattern."
    )
    assert set(result) == set(SCREEN_STANDARDIZED_CONTACTS_FEATHERS), (
        "dataverse_directory_candidates should return exactly the "
        "SCREEN_STANDARDIZED_CONTACTS_FEATHERS list."
    )


def test_all_standardized_contacts_feathers_in_basename_to_archive() -> None:
    """Every SCREEN_STANDARDIZED_CONTACTS_FEATHERS entry must be tagged as
    'screen' in BASENAME_TO_ARCHIVE so the download CLI knows which DOI to use."""
    missing = [
        name
        for name in SCREEN_STANDARDIZED_CONTACTS_FEATHERS
        if name not in BASENAME_TO_ARCHIVE
    ]
    assert not missing, (
        f"BASENAME_TO_ARCHIVE is missing entries for: {missing}. "
        "Add them with archive='screen'."
    )
    wrong_archive = [
        name
        for name in SCREEN_STANDARDIZED_CONTACTS_FEATHERS
        if BASENAME_TO_ARCHIVE.get(name) != "screen"
    ]
    assert (
        not wrong_archive
    ), f"These standardized_contacts feathers are not tagged as 'screen': {wrong_archive}"
