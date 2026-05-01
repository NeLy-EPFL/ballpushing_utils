"""Helpers for loading flies from the published Dataverse layout.

The Dataverse archive that accompanies Durrieu et al. (2026) is organised by
*condition* rather than by *experiment*: each top-level folder is a single
condition (e.g. genotype ``MB247xTNT``, magnet state ``y``/``n``, F1
pretraining bucket), and every fly recorded in that condition — across many
acquisition dates — is placed underneath. The on-server layout, by contrast,
is organised by acquisition date, with a per-experiment ``Metadata.json``
mapping arena → condition. The two layouts therefore look like::

    # On-server (canonical):
    <experiment>/Metadata.json
    <experiment>/arenaN/corridorM/{ball,fly,full_body}*.h5
    <experiment>/arenaN/corridorM/<corridor>.mp4

    # Dataverse (this module):
    <dataset_root>/<condition>/<date>[-N]/arenaN/corridorM/{ball,fly,skeleton}*.h5

The Dataverse layout intentionally ships only the SLEAP HDF5 tracks (no
``Metadata.json``, no ``.mp4``, no ``fps.npy``). Re-running the figure
pipeline against it therefore needs three things:

1. A way to discover fly directories under ``<dataset_root>``.
2. A way to synthesise the per-arena metadata dict that ``FlyMetadata``
   would normally read from the parent experiment's ``Metadata.json``.
3. Tolerance for the missing video / fps assets in the rest of the package.

This module provides (1) and (2). The minimal Experiment/FlyMetadata patches
that handle (3) live next to the original code. See ``src/dataset_builder.py``
``--dataverse-root`` for the CLI entry point and the README for an end-user
walkthrough.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator, Optional

__all__ = [
    "ARCHIVE_PREFIX_RECIPES",
    "CONDITION_TRANSFORMERS",
    "DEFAULT_CONDITION_FIELD",
    "DEFAULT_VIDEO_SIZE",
    "DataverseFly",
    "default_condition_field",
    "detect_dataverse_experiment_type",
    "expand_condition",
    "iter_dataverse_flies",
    "is_dataverse_layout",
    "parse_archive_name",
    "synthesize_experiment_metadata",
]


#: Default mapping of experiment_type → arena_metadata column that the
#: Dataverse condition folder name maps to. The Dataverse archive is sorted
#: by the *primary* condition variable for each paradigm; figure scripts
#: filter on these columns. Override per-run with ``--condition-field``.
#:
#: Paradigm-specific archives (balltype, ballscent, light, feeding state,
#: dark olfaction, learning mutants…) all use ``experiment_type="TNT"``
#: under the hood — pass ``--condition-field BallType`` /
#: ``--condition-field FeedingState`` / etc. explicitly when running them.
DEFAULT_CONDITION_FIELD = {
    "TNT": "Genotype",
    "MagnetBlock": "Magnet",
    "F1": "F1_condition",
    "Learning": "Pretraining",
}


def _expand_f1_condition(condition_value: str) -> dict[str, str]:
    """Derive Pretraining + F1_condition from a single F1 folder name.

    The Dataverse F1 archive sorts flies into folders named after their
    F1 condition (``control``, ``pretrained``, ``pretrained_unlocked``).
    Figure scripts filter on both ``Pretraining`` (``y`` / ``n``) and
    ``F1_condition`` — the former groups the two ``pretrained*`` buckets
    together — so we set both columns here. Anything other than
    ``control`` is treated as a pretrained variant (``Pretraining=y``).
    """
    f1_condition = condition_value
    pretraining = "n" if condition_value.lower().strip() == "control" else "y"
    return {"Pretraining": pretraining, "F1_condition": f1_condition}


#: Map from Dataverse MagnetBlock condition folder names to the on-server
#: ``Magnet`` column value.  The Dataverse uses human-readable labels
#: (``Blocked`` / ``Unblocked``) while the server metadata uses ``y``/``n``.
_MAGNETBLOCK_CONDITION_MAP: dict[str, str] = {
    "blocked": "y",
    "unblocked": "n",
}


def _expand_magnetblock_condition(condition_value: str) -> dict[str, str]:
    """Map a MagnetBlock Dataverse folder name to ``{"Magnet": "y"|"n"}``.

    The Dataverse archive uses ``Blocked`` / ``Unblocked`` (or
    ``Blocked_<variant>`` / ``Unblocked_<variant>``) while the on-server
    metadata column is ``Magnet`` with values ``"y"`` / ``"n"``.  Any
    unrecognised prefix is passed through verbatim so novel variants don't
    silently drop data.
    """
    key = condition_value.lower().split("_")[0]
    magnet = _MAGNETBLOCK_CONDITION_MAP.get(key, condition_value)
    return {"Magnet": magnet}


#: Per-experiment-type hooks that take a single condition folder name and
#: expand it into multiple ``{field: value}`` metadata pairs. Use this when
#: one archive folder needs to populate more than one feather column —
#: e.g. F1's ``Pretraining`` (binary) layered over ``F1_condition`` (3
#: levels). Paradigms not listed here fall back to the single-column
#: default in :data:`DEFAULT_CONDITION_FIELD`.
CONDITION_TRANSFORMERS: dict[str, Callable[[str], dict[str, str]]] = {
    "F1": _expand_f1_condition,
    "MagnetBlock": _expand_magnetblock_condition,
}


#: Default video frame size (W, H) used when the recording's actual ``.mp4``
#: is unavailable (the Dataverse layout ships only HDF5 tracks). This is the
#: standard MultiMazeRecorder rig output. ``SkeletonMetrics`` reads this for
#: ball-coordinate normalisation; if your recordings used a different
#: resolution, set ``Config.default_video_size`` explicitly.
DEFAULT_VIDEO_SIZE = (1280, 1024)


# Matches a Dataverse date folder. Acquisition dates are 6-digit
# ``YYMMDD``; if multiple experiments share the same calendar date they are
# disambiguated with a ``-2``, ``-3`` … suffix (e.g. ``230617-2``).
_DATE_FOLDER_RE = re.compile(r"^\d{6}(?:-\d+)?$")


class DataverseFly:
    """Lightweight record describing one fly directory in the Dataverse layout.

    Attributes
    ----------
    directory:
        Absolute path to the corridor folder (``…/<condition>/<date>/arenaN/
        corridorM``) that contains the per-fly HDF5 tracks.
    condition:
        Name of the condition folder (e.g. ``"MB247xTNT"`` for a screen
        genotype, ``"y"``/``"n"`` for MagnetBlock, ``"y_unrestrained"`` for
        an F1 pretraining bucket). Used as the value for the per-arena
        metadata column selected by :data:`DEFAULT_CONDITION_FIELD`.
    date_folder:
        Path to the date folder two levels up (``…/<condition>/<date>``).
        ``Fly`` treats this as the synthetic experiment directory: its name
        becomes the ``experiment`` column in the feather and the prefix of
        ``fly.metadata.name``.
    arena, corridor:
        Arena and corridor folder names, mirroring the on-server layout.
    """

    __slots__ = ("directory", "condition", "date_folder", "arena", "corridor")

    def __init__(self, directory: Path, condition: str) -> None:
        self.directory = directory
        self.condition = condition
        self.date_folder = directory.parent.parent
        self.arena = directory.parent.name
        self.corridor = directory.name

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"DataverseFly(condition={self.condition!r}, "
            f"date={self.date_folder.name!r}, arena={self.arena!r}, "
            f"corridor={self.corridor!r})"
        )


def default_condition_field(experiment_type: str) -> str:
    """Return the default arena-metadata column for ``experiment_type``.

    Raises ``KeyError`` if ``experiment_type`` is unknown so callers fail
    loudly instead of silently dropping the condition column.
    """
    return DEFAULT_CONDITION_FIELD[experiment_type]


def is_dataverse_layout(path: Path) -> bool:
    """Return True if ``path`` looks like a Dataverse fly directory.

    Heuristic: ``path`` contains at least one ``*ball*.h5``, the parent
    chain has the shape ``corridor → arena → date → condition`` (where
    ``date`` matches ``YYMMDD`` optionally followed by ``-N``), and there is
    no ``Metadata.json`` two levels up. Used by the dataset builder to
    detect when synthetic metadata is required.
    """
    path = Path(path)
    if not any(path.glob("*ball*.h5")):
        return False
    grandparent = path.parent.parent
    if (grandparent / "Metadata.json").exists():
        return False
    return bool(_DATE_FOLDER_RE.match(grandparent.name))


def _is_flat_layout(root: Path) -> bool:
    """Return True if ``root``'s immediate subdirectories look like date folders.

    A *flat* layout has the shape ``<root>/<date>/arena*/fly_dir/`` — the root
    itself acts as the condition folder (e.g.
    ``Generalisation-Wild-type-pretrained/251008/Arena4/Left/``).  Contrast
    with the standard *multi-condition* layout where an extra condition layer
    sits between root and the date folders.
    """
    return any(d.is_dir() and _DATE_FOLDER_RE.match(d.name) for d in root.iterdir())


def _condition_from_folder_name(name: str) -> str:
    """Extract the condition token from a folder name.

    The naming convention for Dataverse condition folders is
    ``<ExperimentLabel>-<Subtype>-<condition>`` where the condition is the
    last hyphen-delimited token.  Examples::

        Generalisation-Wild-type-pretrained   → pretrained
        Generalisation-Wild-type-naive        → naive
        Screen-MB247xTNT                      → MB247xTNT
    """
    return name.rsplit("-", 1)[-1]


def iter_dataverse_flies(root: Path) -> Iterator[DataverseFly]:
    """Walk a Dataverse subtree and yield one :class:`DataverseFly` per fly directory.

    Two layouts are supported:

    * **Multi-condition** (standard): ``<root>/<condition>/<date>/arena*/fly_dir/``
      where ``root``'s immediate children are condition folders.  The condition
      folder name is stored verbatim on :attr:`DataverseFly.condition`.
    * **Flat** (single-condition): ``<root>/<date>/arena*/fly_dir/`` where
      ``root`` is already the condition folder (e.g.
      ``Generalisation-Wild-type-pretrained``).  The condition is extracted
      from the last hyphen-delimited token of ``root.name``.

    In both cases the walker skips any fly directory without a ``*ball*.h5``
    track, and date folders whose name doesn't match ``YYMMDD[-N]``.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataverse root is not a directory: {root}")

    if _is_flat_layout(root):
        # Root IS the condition folder; extract condition from its name.
        condition_value = _condition_from_folder_name(root.name)
        date_parents: list[tuple[Path, str]] = [(root, condition_value)]
    else:
        # Standard multi-condition: <root>/<condition>/<date>/...
        date_parents = [
            (condition_dir, condition_dir.name)
            for condition_dir in sorted(root.iterdir())
            if condition_dir.is_dir() and not condition_dir.name.startswith(".")
        ]

    for condition_parent, condition_value in date_parents:
        for date_dir in sorted(condition_parent.iterdir()):
            if not date_dir.is_dir() or not _DATE_FOLDER_RE.match(date_dir.name):
                continue
            for arena_dir in sorted(date_dir.iterdir()):
                if not arena_dir.is_dir() or not arena_dir.name.lower().startswith("arena"):
                    continue
                for corridor_dir in sorted(arena_dir.iterdir()):
                    if not corridor_dir.is_dir():
                        continue
                    if not any(corridor_dir.glob("*ball*.h5")):
                        continue
                    yield DataverseFly(corridor_dir, condition_value)


def detect_dataverse_experiment_type(root: Path) -> str:
    """Infer the experiment type from the structure of a Dataverse root.

    Heuristics (applied in order):

    1. **MagnetBlock** — ``"magnet"`` (case-insensitive) appears anywhere in
       ``root``'s path, **or** any immediate child directory is named ``"y"``
       or ``"n"`` (the two MagnetBlock conditions).
    2. **F1** — under any arena directory the child folders are named
       ``"Left"`` or ``"Right"`` (not ``"corridor*"``).  A *ball*.h5 must be
       present so we don't trigger on empty arena stubs.
    3. **TNT** — default fallback.

    Parameters
    ----------
    root:
        Path to the Dataverse root whose immediate children are condition
        folders (the same path passed to ``--dataverse-root``).

    Returns
    -------
    str
        One of ``"MagnetBlock"``, ``"F1"``, ``"TNT"``.
    """
    root = Path(root)

    # 1. MagnetBlock: keyword anywhere in the path.
    if "magnet" in str(root).lower():
        return "MagnetBlock"

    all_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    f1_names = {"left", "right"}
    magnet_condition_names = {"y", "n", "y_unrestrained", "n_unrestrained"}

    if _is_flat_layout(root):
        # Root IS the condition folder; inspect the condition name token.
        last_token = _condition_from_folder_name(root.name).lower()
        if last_token in magnet_condition_names:
            return "MagnetBlock"
        # F1: look for Left/Right under <root>/<date>/arena*/
        for date_dir in all_dirs:
            if not _DATE_FOLDER_RE.match(date_dir.name):
                continue
            for arena_dir in date_dir.iterdir():
                if not arena_dir.is_dir() or not arena_dir.name.lower().startswith("arena"):
                    continue
                for fly_dir in arena_dir.iterdir():
                    if not fly_dir.is_dir():
                        continue
                    if fly_dir.name.lower() in f1_names and any(fly_dir.glob("*ball*.h5")):
                        return "F1"
        return "TNT"

    # Standard multi-condition layout.
    condition_dirs = all_dirs
    # 2. MagnetBlock: all condition folders named y/n.
    if condition_dirs and all(d.name.lower() in magnet_condition_names for d in condition_dirs):
        return "MagnetBlock"

    # 3. F1: fly-level dirs inside arenas are Left/Right.
    for condition_dir in condition_dirs:
        for date_dir in condition_dir.iterdir():
            if not date_dir.is_dir() or not _DATE_FOLDER_RE.match(date_dir.name):
                continue
            for arena_dir in date_dir.iterdir():
                if not arena_dir.is_dir() or not arena_dir.name.lower().startswith("arena"):
                    continue
                for fly_dir in arena_dir.iterdir():
                    if not fly_dir.is_dir():
                        continue
                    if fly_dir.name.lower() in f1_names and any(fly_dir.glob("*ball*.h5")):
                        return "F1"

    # 4. Default.
    return "TNT"


def expand_condition(
    experiment_type: str, condition_value: str, *, condition_field: Optional[str] = None
) -> dict[str, str]:
    """Map a Dataverse condition folder name to one or more metadata columns.

    Most paradigms produce a single ``{<default_field>: condition_value}``
    pair (Genotype for the screen, BallType for the balltype archive, …).
    Paradigms registered in :data:`CONDITION_TRANSFORMERS` produce more
    than one — e.g. F1 emits both ``Pretraining`` and ``F1_condition``
    from a single folder name.

    Parameters
    ----------
    experiment_type:
        Must be a key of :data:`DEFAULT_CONDITION_FIELD`.
    condition_value:
        The condition folder name as it appears in the archive.
    condition_field:
        Override the default ``DEFAULT_CONDITION_FIELD[experiment_type]``.
        Use this for paradigms that share an experiment_type but encode
        a different primary column (e.g. balltype + ballscent + light all
        run as ``experiment_type=TNT`` but populate ``BallType`` /
        ``FeedingState`` / ``Light`` columns respectively). When a
        transformer is registered for ``experiment_type``, its output
        wins — pass ``condition_field`` only with experiment types that
        don't have a transformer.

    Returns
    -------
    dict
        Flat ``{column_name: value}`` mapping for the current arena.
    """
    transformer = CONDITION_TRANSFORMERS.get(experiment_type)
    if transformer is not None:
        return transformer(condition_value)

    field = condition_field or DEFAULT_CONDITION_FIELD.get(experiment_type)
    if field is None:
        raise ValueError(
            f"No default condition field for experiment_type {experiment_type!r}. " f"Pass condition_field explicitly."
        )
    return {field: condition_value}


#: Recipe table for paradigms whose archive folder name encodes a single
#: ``<prefix>-<value>`` pair. Each entry maps ``"Prefix-"`` →
#: ``(experiment_type, condition_column)``. The condition value is whatever
#: comes after the prefix (verbatim, hyphens preserved). Add an entry here
#: when you publish a new paradigm-specific Dataverse archive — no other
#: code change is required for the simple-prefix case.
#:
#: Multi-field paradigms (MagnetBlock, F1/Generalisation, the
#: ``Wild-type_Lights-..._...`` feeding-state archives, and the
#: ``<genotype>-Light-{on,off}`` olfaction-dark archives) are NOT here;
#: they live in :func:`parse_archive_name` because one folder name
#: populates more than one column.
ARCHIVE_PREFIX_RECIPES: dict[str, tuple[str, str]] = {
    "Ballscents-": ("TNT", "BallScent"),
    "Balltype-": ("TNT", "BallType"),
    "Multi-trials-": ("Learning", "Trial_duration"),
}


def parse_archive_name(name: str) -> tuple[str, dict[str, str]]:
    """Decode a Dataverse archive folder name into ``(experiment_type, fields)``.

    The Dataverse archives are named after the experiment paradigm and the
    condition they pack:

    - ``Magnetblock-Blocked`` / ``Magnetblock-Control`` → MagnetBlock with
      ``Magnet=y`` / ``Magnet=n``.
    - ``Generalisation-<genotype>-<f1_condition>`` → F1 with
      ``Genotype``, ``F1_condition`` and derived ``Pretraining``.  The
      genotype slot may itself contain hyphens (``TNTxLC10-2``,
      ``Wild-type``); split from the right so the last hyphen-segment is
      the F1 condition.
    - ``Ballscents-<v>``, ``Balltypes-<v>``, ``Feedingstate-<v>``,
      ``Light-<v>``, ``Trial_duration-<v>`` → TNT (or Learning) with the
      paradigm-specific column populated. See
      :data:`ARCHIVE_PREFIX_RECIPES`.
    - ``TNT-olfaction-dark-<genotype>-<Light>`` → TNT with both
      ``Genotype`` and ``Light`` populated.
    - Anything else → silencing screen, treated as ``("TNT", {"Genotype":
      name})`` so single-token archive names like ``LC6`` work
      out-of-the-box.

    Returns
    -------
    (str, dict)
        Experiment type (``"TNT"`` / ``"MagnetBlock"`` / ``"F1"`` /
        ``"Learning"``) and a flat ``{column: value}`` dict ready to feed
        into :func:`synthesize_experiment_metadata`.
    """
    # MagnetBlock — exact match, case-insensitive (the published archives
    # use ``MagnetBlock-`` with a capital B; older drafts used the all
    # lowercase ``Magnetblock-`` form).
    if name.lower() == "magnetblock-blocked":
        return "MagnetBlock", {"Magnet": "y"}
    if name.lower() == "magnetblock-control":
        return "MagnetBlock", {"Magnet": "n"}

    # F1 (Generalisation alias). Genotype may contain hyphens — split
    # from the right so the last token is the F1 condition.
    if name.startswith("Generalisation-"):
        rest = name[len("Generalisation-") :]
        head, sep, f1_cond = rest.rpartition("-")
        if not sep:
            f1_cond, head = rest, ""
        pretraining = "n" if f1_cond.lower() == "control" else "y"
        fields: dict[str, str] = {"Pretraining": pretraining, "F1_condition": f1_cond}
        if head:
            fields["Genotype"] = head
        return "F1", fields

    # Feeding-state archives: ``<Genotype>_Lights-<on|off>_<FeedingState>[-<replicate>]``
    # Examples: Wild-type_Lights-off_Fed-1, Wild-type_Lights-on_Starved,
    # Wild-type_Lights-off_Starved-without-water-2.
    # Distinguished from ``Wild-type_PR`` / ``Wild-type_CS`` (silencing
    # screen, two underscore tokens) by requiring three+ underscore
    # tokens AND a ``Lights-`` prefix on the second one.
    underscore_parts = name.split("_")
    if len(underscore_parts) >= 3 and underscore_parts[1].startswith("Lights-"):
        genotype = underscore_parts[0]
        light = underscore_parts[1][len("Lights-") :]  # "on" or "off"
        feeding_state = "_".join(underscore_parts[2:])
        # Strip a trailing ``-<digit>`` replicate suffix so two replicates
        # of the same condition collapse onto one feather column value.
        m = re.match(r"^(.*?)-(\d+)$", feeding_state)
        if m:
            feeding_state = m.group(1)
        return "TNT", {
            "Genotype": genotype,
            "Light": light,
            "FeedingState": feeding_state,
        }

    # Olfaction-dark dual-field: ``<Genotype>-Light-{on,off}``.
    # Examples: TNTxEmptyGal4-Light-off, TNTxIR8a-Light-on.
    olf_match = re.match(r"^(.+)-Light-(on|off)$", name)
    if olf_match:
        return "TNT", {"Genotype": olf_match.group(1), "Light": olf_match.group(2)}

    # Single-prefix supplementary archives — see ARCHIVE_PREFIX_RECIPES.
    for prefix, (etype, column) in ARCHIVE_PREFIX_RECIPES.items():
        if name.startswith(prefix):
            return etype, {column: name[len(prefix) :]}

    # Default: silencing-screen archives are named after the genotype.
    return "TNT", {"Genotype": name}


def synthesize_experiment_metadata(arena: str, fields: dict[str, str]) -> dict:
    """Build the metadata dict ``FlyMetadata`` would normally read from JSON.

    The on-server schema is ``{<field>: {arena1: <value>, arena2: <value>, …}}``
    with arena keys lower-cased. Here we populate one entry per supplied
    ``(field, value)`` pair for the arena currently being loaded.

    Parameters
    ----------
    arena:
        The arena folder name (e.g. ``"arena1"``, case-insensitive — the
        on-server loader lower-cases its keys, so we mirror that here).
    fields:
        Flat ``{column_name: value}`` mapping. Typically the output of
        :func:`expand_condition`.

    Returns
    -------
    dict
        Schema matching :meth:`Experiment.load_metadata`'s output.
    """
    arena_key = arena.lower()
    return {field: {arena_key: value} for field, value in fields.items()}
