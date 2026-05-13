"""Server-path → Dataverse-basename resolution for the published feathers.

The figure scripts in ``figures/`` request feathers with their
*server-side* names (e.g.
``MagnetBlock/Datasets/<ts>/summary/pooled_summary.feather``) because
that's what existed when they were written. External users download
the **Dataverse archive** instead, which uses different filenames and
a flat layout (no ``<Subdir>/Datasets/<timestamp>_Data/<kind>/``
subtree).

This module is the single place that knows how to translate between
the two. :func:`ballpushing_utils.paths.find_feather` consults
:func:`dataverse_candidates` as its final resolution step;
:func:`ballpushing_utils.compat.read_feather` concatenates the parts
when a multi-part split feather is returned.

Add a new entry to ``SERVER_TO_DATAVERSE`` whenever a new figure
script reads a feather that's also published on Dataverse. The unit
test ``tests/unit/test_dataverse_naming.py`` walks ``figures/**/*.py``
and surfaces every ``dataset(...)`` literal that isn't covered.

Naming conventions on the Dataverse
-----------------------------------
- **Flat layout**: all feathers live at the archive root; no
  ``Datasets/<timestamp>/<kind>/`` subtree.
- **Filename pattern**: ``<paradigm-prefix>_<kind>.feather`` with
  ``<kind>`` ∈ ``{ballpushing_metrics, trajectories}``. (Note the
  *underscore* before ``<kind>`` — the paradigm prefix itself may
  contain hyphens, e.g. ``Generalisation-Wild-type_trajectories.feather``.)
- **Capitalisation drift**: the published archive has a few historical
  inconsistencies (``Magnetblock_*`` with a lowercase ``b``;
  ``Wild-Type_ballpushing_metrics.feather`` with a capital ``T`` but
  ``Wild-type_Lights-...`` for the per-condition trajectories). The
  table below records the exact published basenames verbatim.
- **Split parts**: any feather may be present as either
  ``<stem>.feather`` (single file) or ``<stem>-1.feather``,
  ``<stem>-2.feather``, … (split for >2.5 GiB files). The resolver
  tries both forms; the reader concatenates parts.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

__all__ = [
    "BASENAME_TO_ARCHIVE",
    "DATAVERSE_DOIS",
    "SCREEN_STANDARDIZED_CONTACTS_FEATHERS",
    "SERVER_DIRECTORY_TO_DATAVERSE",
    "SERVER_TO_DATAVERSE",
    "WILDTYPE_TRAJECTORY_FEATHERS",
    "dataverse_candidates",
    "dataverse_directory_candidates",
    "expand_split_parts",
]


#: DOIs of the three published Dataverse archives. The download CLI
#: queries each by DOI to discover what's actually on the server.
DATAVERSE_DOIS: dict[str, str] = {
    "affordance": "doi:10.7910/DVN/91R87T",
    "screen": "doi:10.7910/DVN/SPBKKJ",
    "exploration": "doi:10.7910/DVN/VB4UI5",
}

#: DOI for the confocal-stacks dataset (ED Figure 4).
#: Contains per-genotype ``.tiff`` files (flat layout, one file per genotype
#: key with the leading date prefix stripped) plus the ``stack_infos.yaml``
#: manifest.  Not included in the main :data:`DATAVERSE_DOIS` dict because
#: it is consumed by a dedicated helper in the figure script rather than by
#: the generic ``ballpushing-fetch`` feather downloader.
CONFOCAL_DOI: str = "doi:10.7910/DVN/MY4GN5"


#: Server-relative path (as written in the figure scripts' ``dataset(...)``
#: literals) → list of Dataverse basenames that, after concat, reproduce
#: the server feather.
#:
#: Globs in the keys (``*``) match the timestamped server label between
#: ``Datasets/`` and ``_Data/`` so we don't have to update the table when
#: a re-build moves the timestamp.
SERVER_TO_DATAVERSE: dict[str, list[str]] = {
    # ---- Affordance archive ----
    "MagnetBlock/Datasets/*/summary/pooled_summary.feather": [
        "Magnetblock_ballpushing_metrics.feather",
    ],
    "MagnetBlock/Datasets/*/coordinates/pooled_coordinates.feather": [
        "Magnetblock_trajectories.feather",
    ],
    # F1 wild-type (Affordance archive)
    "F1_Tracks/Datasets/*F1_New_Data/summary/pooled_summary.feather": [
        "Generalisation-Wild-type_ballpushing_metrics.feather",
    ],
    "F1_Tracks/Datasets/*F1_New_Data/coordinates/pooled_coordinates.feather": [
        "Generalisation-Wild-type_trajectories.feather",
    ],
    # F1-TNT — lives in the Affordance archive (despite the F1-TNT
    # panel appearing in Fig. 3, the uploaded data is in Affordance).
    "F1_Tracks/Datasets/*F1_TNT_Full_Data/summary/pooled_summary.feather": [
        "Generalisation-TNT_ballpushing_metrics.feather",
    ],
    "F1_Tracks/Datasets/*F1_TNT_Full_Data/coordinates/pooled_coordinates.feather": [
        # Published as a split feather (>2.5 GiB).
        "Generalisation-TNT_trajectories.feather",
    ],
    # ---- Screen archive ----
    # Screen summary is a bare-named file (no paradigm prefix).
    # Per-region trajectories are out of scope for current panels and
    # are handled by a separate helper (see "Per-region screen
    # trajectories" in docs/HANDOFF_dataverse_workflow.md).
    "Ballpushing_TNTScreen/Datasets/*/summary/pooled_summary.feather": [
        "ballpushing_metrics_silencing_screen.feather",
    ],
    # ---- Exploration archive ----
    "Ballpushing_Balltypes/Datasets/*/summary/pooled_summary.feather": [
        "Balltypes_ballpushing_metrics.feather",
    ],
    "Ballpushing_Balltypes/Datasets/*/coordinates/pooled_coordinates.feather": [
        "Balltypes_trajectories.feather",
    ],
    "Ball_scents/Datasets/*/summary/pooled_summary.feather": [
        "Ballscents_ballpushing_metrics.feather",
    ],
    "Ball_scents/Datasets/*/coordinates/pooled_coordinates.feather": [
        "Ballscents_trajectories.feather",
    ],
    # Wild-type × light × feeding state — the Exploration archive bundles
    # all six feeding-state × light combinations into one pooled
    # ``Wild-Type_ballpushing_metrics.feather`` (condition encoded as
    # columns inside; note the capital ``T``).
    "Ballpushing_Exploration/Datasets/*/summary/pooled_summary.feather": [
        "Wild-Type_ballpushing_metrics.feather",
    ],
    # Olfaction × dark — single file covers BOTH genotypes
    # (TNTxIR8a *and* TNTxEmptyGal4) for the whole olfaction-dark
    # experiment. Despite the "TNTxIR8a" token in the filename, the
    # control is in the same feather — do not assume genotype filtering
    # is required.
    "TNT_Olfaction_Dark/Datasets/*/summary/pooled_summary.feather": [
        "TNTxIR8a-dark_ballpushing_metrics.feather",
    ],
    "TNT_Olfaction_Dark/Datasets/*/coordinates/pooled_coordinates.feather": [
        "TNTxIR8a-dark_trajectories.feather",
    ],
    # Trial-duration / learning paradigm — single annotated dataset
    # rather than the pooled_summary convention. Published as a split
    # feather (>2.5 GiB).
    "BallPushing_Learning/Datasets/250318_Datasets/250320_Annotated_data.feather": [
        "Multi-trials_trajectories.feather",
    ],
}


#: Per-brain-region standardized-contacts feathers (Screen archive).
#: Built by ``src/build_region_standardized_contacts.py`` from the
#: per-date ``*_standardized_contacts.feather`` files in
#: ``Ballpushing_TNTScreen/Datasets/250809_02_standardized_contacts_TNT_screen_Data/standardized_contacts/``.
#: Each entry may be a single file or the first of a set of numbered
#: split parts (``<stem>-1.feather``, ``<stem>-2.feather``, …);
#: :func:`expand_split_parts` resolves both forms transparently.
#: Used by the UMAP pipeline (Fig. 3, ED Fig. 5, Supp. File 2) via
#: :mod:`ballpushing_utils.preprocess_screen_data`.
SCREEN_STANDARDIZED_CONTACTS_FEATHERS: tuple[str, ...] = (
    "Central_Complex_standardized_contacts.feather",
    "Control_standardized_contacts.feather",
    "Lateral_Horn_standardized_contacts.feather",
    "MB_extrinsic_standardized_contacts.feather",
    "Mushroom_Body_standardized_contacts.feather",
    "Neuropeptide_standardized_contacts.feather",
    "Olfaction_standardized_contacts.feather",
    "Others_standardized_contacts.feather",
    "Vision_standardized_contacts.feather",
)


#: Per-condition wild-type trajectory feathers (Exploration archive).
#: One file per (Light × FeedingState) combination. Iterating these in
#: place of an on-server ``coordinates/*_coordinates.feather`` directory
#: yields the same set of per-fly trajectory rows (just pooled by
#: condition rather than by experiment), so the figure scripts can run
#: against either layout — see :func:`iter_coordinate_feathers` in
#: :mod:`ballpushing_utils.compat`.
WILDTYPE_TRAJECTORY_FEATHERS: tuple[str, ...] = (
    "Wild-type_Lights-off_Fed_trajectories.feather",
    "Wild-type_Lights-off_Starved_trajectories.feather",
    "Wild-type_Lights-off_Starved-without-water_trajectories.feather",
    "Wild-type_Lights-on_Fed_trajectories.feather",
    "Wild-type_Lights-on_Starved_trajectories.feather",
    "Wild-type_Lights-on_Starved-without-water_trajectories.feather",
)


#: Server-side *directory* paths (as written in the figure scripts'
#: ``dataset(...)`` literals) → list of Dataverse basenames covering the
#: same per-fly trajectory data. The on-server layout has one
#: ``*_coordinates.feather`` per experiment in a single directory; the
#: Dataverse layout has one pooled feather per condition.
#:
#: ``iter_coordinate_feathers`` in :mod:`ballpushing_utils.compat`
#: consumes this table to expose a single iteration API to figure
#: scripts.
SERVER_DIRECTORY_TO_DATAVERSE: dict[str, list[str]] = {
    # Wild-type baseline trajectories — used by Fig 1, ED Fig 3, ED Fig 10.
    "Ballpushing_Exploration/Datasets/*/coordinates": list(
        WILDTYPE_TRAJECTORY_FEATHERS
    ),
    "Ballpushing_Exploration/Datasets/*/Other/coordinates": list(
        WILDTYPE_TRAJECTORY_FEATHERS
    ),
    # Silencing-screen standardized contacts — used by the UMAP pipeline
    # (Fig. 3, ED Fig. 5, Supp. File 2). The on-server layout is 74 per-date
    # feathers; the Dataverse layout is one pooled feather per brain region.
    # Built by src/build_region_standardized_contacts.py.
    "Ballpushing_TNTScreen/Datasets/*/standardized_contacts": list(
        SCREEN_STANDARDIZED_CONTACTS_FEATHERS
    ),
}


#: Map each Dataverse basename to its source archive. The download CLI
#: uses this to know which DOI to query for each file. Keep in sync with
#: :data:`SERVER_TO_DATAVERSE` and :data:`SERVER_DIRECTORY_TO_DATAVERSE`;
#: the assertion at import time below catches obvious "added a basename
#: to one table but not the other" bugs.
BASENAME_TO_ARCHIVE: dict[str, str] = {
    # Affordance
    "Magnetblock_ballpushing_metrics.feather": "affordance",
    "Magnetblock_trajectories.feather": "affordance",
    "Generalisation-Wild-type_ballpushing_metrics.feather": "affordance",
    "Generalisation-Wild-type_trajectories.feather": "affordance",
    "Generalisation-TNT_ballpushing_metrics.feather": "affordance",
    "Generalisation-TNT_trajectories.feather": "affordance",
    # Screen
    "ballpushing_metrics_silencing_screen.feather": "screen",
    **{name: "screen" for name in SCREEN_STANDARDIZED_CONTACTS_FEATHERS},
    # Exploration
    "Balltypes_ballpushing_metrics.feather": "exploration",
    "Balltypes_trajectories.feather": "exploration",
    "Ballscents_ballpushing_metrics.feather": "exploration",
    "Ballscents_trajectories.feather": "exploration",
    "Wild-Type_ballpushing_metrics.feather": "exploration",
    "TNTxIR8a-dark_ballpushing_metrics.feather": "exploration",
    "TNTxIR8a-dark_trajectories.feather": "exploration",
    "Multi-trials_trajectories.feather": "exploration",
    **{name: "exploration" for name in WILDTYPE_TRAJECTORY_FEATHERS},
}


# Sanity-check at import time: every basename mentioned in the mapping
# tables must have an archive assigned. Catches the obvious typo where a
# new basename was added to SERVER_TO_DATAVERSE (or
# SERVER_DIRECTORY_TO_DATAVERSE) but not BASENAME_TO_ARCHIVE.
_expected_basenames = {b for v in SERVER_TO_DATAVERSE.values() for b in v} | {
    b for v in SERVER_DIRECTORY_TO_DATAVERSE.values() for b in v
}
_missing = _expected_basenames - set(BASENAME_TO_ARCHIVE)
if _missing:  # pragma: no cover - import-time guard
    raise RuntimeError(
        f"dataverse_naming: basenames missing from BASENAME_TO_ARCHIVE: {sorted(_missing)}. "
        "Every basename in SERVER_TO_DATAVERSE / SERVER_DIRECTORY_TO_DATAVERSE "
        "must be tagged with its source archive."
    )
del _expected_basenames, _missing


def dataverse_candidates(server_relative: str | Path) -> list[str] | None:
    """Return the Dataverse basenames that satisfy a server-relative request.

    Returns ``None`` if no rule matches — callers should treat that as
    "not a published figure feather; let the existing search continue".
    Returns a non-empty list of basenames otherwise. The basenames may
    not all exist on disk; the resolver tries each in turn.
    """
    s = str(server_relative).replace("\\", "/")
    for pattern, candidates in SERVER_TO_DATAVERSE.items():
        if fnmatch.fnmatchcase(s, pattern):
            return list(candidates)
    return None


def dataverse_directory_candidates(server_relative: str | Path) -> list[str] | None:
    """Return the Dataverse basenames covering a server-side directory.

    Mirrors :func:`dataverse_candidates` but for the directory-level
    iteration pattern used by figure scripts that walk a
    ``coordinates/`` folder for per-experiment ``*_coordinates.feather``
    files. The values in :data:`SERVER_DIRECTORY_TO_DATAVERSE` are the
    *per-condition* Dataverse feathers that, taken together, contain
    the same per-fly rows.

    Returns ``None`` when no rule matches.
    """
    s = str(server_relative).replace("\\", "/")
    for pattern, candidates in SERVER_DIRECTORY_TO_DATAVERSE.items():
        if fnmatch.fnmatchcase(s, pattern):
            return list(candidates)
    return None


def expand_split_parts(basename: str, search_dir: Path) -> list[Path]:
    """Expand a single basename into the matching files on disk.

    If ``<stem>.feather`` exists, return ``[that path]``.
    Else if ``<stem>-1.feather``, ``<stem>-2.feather``, … exist, return
    them in numeric order.
    Else return ``[]``.

    Single source of truth for the split-part convention used by the
    Dataverse archives (see :mod:`ballpushing_utils.archive_split`).
    """
    direct = search_dir / basename
    if direct.exists():
        return [direct.resolve()]
    stem = Path(basename).stem
    suffix = Path(basename).suffix  # ".feather"
    keep: list[tuple[int, Path]] = []
    for p in search_dir.glob(f"{stem}-*{suffix}"):
        tail = p.stem[len(stem) + 1 :]  # everything after "<stem>-"
        if tail.isdigit():
            keep.append((int(tail), p.resolve()))
    keep.sort(key=lambda pair: pair[0])
    return [p for _, p in keep]
