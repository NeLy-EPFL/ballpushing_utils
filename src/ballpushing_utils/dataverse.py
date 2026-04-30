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
    "CONDITION_TRANSFORMERS",
    "DEFAULT_CONDITION_FIELD",
    "DEFAULT_VIDEO_SIZE",
    "DataverseFly",
    "default_condition_field",
    "expand_condition",
    "iter_dataverse_flies",
    "is_dataverse_layout",
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


#: Per-experiment-type hooks that take a single condition folder name and
#: expand it into multiple ``{field: value}`` metadata pairs. Use this when
#: one archive folder needs to populate more than one feather column —
#: e.g. F1's ``Pretraining`` (binary) layered over ``F1_condition`` (3
#: levels). Paradigms not listed here fall back to the single-column
#: default in :data:`DEFAULT_CONDITION_FIELD`.
CONDITION_TRANSFORMERS: dict[str, Callable[[str], dict[str, str]]] = {
    "F1": _expand_f1_condition,
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


def iter_dataverse_flies(root: Path) -> Iterator[DataverseFly]:
    """Walk a Dataverse subtree and yield one :class:`DataverseFly` per corridor.

    ``root`` must point at a directory whose immediate children are
    *condition* folders (e.g. ``…/Affordance``, ``…/Screen``). The walker
    descends ``<root>/<condition>/<date>/arena*/corridor*`` and skips any
    corridor without at least one ``*ball*.h5`` track. Date folders whose
    name doesn't match ``YYMMDD[-N]`` are silently skipped — that prevents
    accidental descent into auxiliary archive folders (``README``,
    ``checksums``, …) that Dataverse sometimes ships next to the data.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataverse root is not a directory: {root}")

    for condition_dir in sorted(root.iterdir()):
        if not condition_dir.is_dir() or condition_dir.name.startswith("."):
            continue
        for date_dir in sorted(condition_dir.iterdir()):
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
                    yield DataverseFly(corridor_dir, condition_dir.name)


def expand_condition(experiment_type: str, condition_value: str, *, condition_field: Optional[str] = None) -> dict[str, str]:
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
            f"No default condition field for experiment_type {experiment_type!r}. "
            f"Pass condition_field explicitly."
        )
    return {field: condition_value}


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
