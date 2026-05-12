"""Backwards-compatibility helpers for legacy column names.

The 2026 ``velocity → speed`` rename (commit ``83f939d``) renamed
several columns in summary / coordinates / transposed feathers. Datasets
published on the Harvard Dataverse before that rename still use the
old ``velocity*`` names. This module provides:

* :func:`normalize_legacy_columns` — adds ``speed*`` aliases on a
  DataFrame whose source feather still uses the legacy ``velocity*``
  names. Idempotent: if a frame already has the modern columns (or
  doesn't have the legacy ones at all), the call is a no-op.

* :func:`read_feather` — drop-in replacement for ``pandas.read_feather``
  that calls :func:`normalize_legacy_columns` before returning, so any
  downstream code can read the modern column names regardless of
  dataset vintage.

Use the wrapper at every script-level feather-read in figure / plot /
analysis scripts. Package-internal code that *generates* fresh data
(e.g. :class:`ballpushing_utils.dataset.Dataset`) never needs it,
since fresh data already uses the new names.

Example::

    from ballpushing_utils import read_feather
    df = read_feather("Ballpushing_TNTScreen/.../pooled_summary.feather")
    df["speed_trend"]    # works on both old and new feathers

The shim has no expiry — keep it as long as legacy datasets remain
referenced. Removing it later is a one-line edit per call site
(``read_feather`` → ``pd.read_feather``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "LEGACY_COLUMN_RENAMES",
    "iter_coordinate_feathers",
    "load_wildtype_experiment",
    "normalize_legacy_columns",
    "read_feather",
]


# Direct one-to-one renames from the velocity → speed pass. The mapping
# is keyed by the modern (preferred) name and points at the legacy name
# we'd accept as a fallback.
LEGACY_COLUMN_RENAMES: dict[str, str] = {
    "speed": "velocity",
    "normalized_speed": "normalized_velocity",
    "speed_during_interactions": "velocity_during_interactions",
    "speed_trend": "velocity_trend",
}

# Per-keypoint per-frame columns produced by the transposed / transformed
# datasets follow the pattern ``{kp}_frame{N}_speed`` after the rename
# (was ``{kp}_frame{N}_velocity``). Note ``{kp}_frame{N}_angular_velocity``
# is NOT renamed — angular velocity is a real signed quantity — and the
# regex deliberately requires ``_velocity`` exactly at end-of-string so
# we don't match ``_angular_velocity``.
_LEGACY_FRAME_RE = re.compile(r"^(?P<base>.+_frame\d+)_velocity$")


def normalize_legacy_columns(df: pd.DataFrame, *, copy: bool = False) -> pd.DataFrame:
    """Add modern ``speed*`` aliases for legacy ``velocity*`` columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to normalize. By default mutated in place.
    copy : bool, optional
        If True, work on a deep copy and return that instead. Use this
        when the caller wants to keep the original frame unchanged.

    Returns
    -------
    pandas.DataFrame
        The (possibly modified) frame. The new columns are added as
        plain references to the legacy data — no values are copied,
        so the memory cost is one extra pandas Series per added alias.
    """
    if copy:
        df = df.copy()

    cols = set(df.columns)

    for new_name, legacy_name in LEGACY_COLUMN_RENAMES.items():
        if new_name not in cols and legacy_name in cols:
            df[new_name] = df[legacy_name]

    # Frame-indexed kinematic columns from transposed / transformed.
    for legacy in list(cols):
        m = _LEGACY_FRAME_RE.match(legacy)
        if m is None:
            continue
        new_name = f"{m.group('base')}_speed"
        if new_name not in cols:
            df[new_name] = df[legacy]

    return df


def read_feather(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> pd.DataFrame:
    """:func:`pandas.read_feather` + :func:`normalize_legacy_columns`.

    Drop-in replacement: same signature as ``pandas.read_feather``.
    The returned frame is guaranteed to have the modern ``speed*``
    column names (either because the feather already had them or
    because they were aliased from the legacy ``velocity*`` columns).

    When ``path`` doesn't resolve directly, this falls back to
    :func:`ballpushing_utils.paths.find_feather`, which searches
    ``BALLPUSHING_DATA_ROOT`` / ``BALLPUSHING_FEATHER_SEARCH`` and the
    Dataverse alias table. That makes the figure scripts work for
    users who downloaded the published Dataverse archive — they don't
    need to recreate the on-server directory hierarchy.

    When the Dataverse publishes a feather as numerically-suffixed
    split parts (``<stem>-1.feather``, ``<stem>-2.feather``, … —
    happens for files >2.5 GiB; see
    :mod:`ballpushing_utils.archive_split`), :func:`find_feather`
    returns the list and this wrapper concatenates the parts in
    numeric order before applying the column shim. Each part is
    self-contained (whole flies are bin-packed during the split), so
    concat reconstructs the original feather exactly.

    If the search fails (or is ambiguous), raises
    :class:`FileNotFoundError` with the standard three-option
    breadcrumb from
    :func:`ballpushing_utils.paths.missing_data_message`.
    """
    from pathlib import Path

    actual_path = Path(path)
    if not actual_path.exists():
        from .paths import find_feather, missing_data_message

        resolved = find_feather(path)
        if resolved is None:
            raise FileNotFoundError(missing_data_message(path, context="feather"))
        if isinstance(resolved, list):
            frames = [pd.read_feather(p, *args, **kwargs) for p in resolved]
            df = pd.concat(frames, ignore_index=True)
            return normalize_legacy_columns(df)
        actual_path = resolved

    df = pd.read_feather(actual_path, *args, **kwargs)
    return normalize_legacy_columns(df)


def iter_coordinate_feathers(
    server_directory: str | os.PathLike[str],
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(label, dataframe)`` pairs covering ``server_directory``.

    Figure scripts that read trajectory data historically walk a
    ``coordinates/`` directory and iterate one ``*_coordinates.feather``
    per experiment. The Dataverse archive publishes the same data
    pooled by *condition* (e.g.
    ``Wild-type_Lights-on_Fed_trajectories.feather``) rather than by
    experiment. This helper hides the layout difference behind a single
    iteration API so the same downstream logic (filter / downsample /
    namespace fly IDs) works for both:

    - **On-server layout** (the directory exists and contains
      per-experiment files): yields ``(file_stem, df)`` per file.
    - **Dataverse layout** (the directory is empty, missing, or
      doesn't contain ``*_coordinates.feather`` files): yields
      ``(experiment_name, df)`` for each ``experiment`` value found in
      the per-condition pooled feathers mapped via
      :data:`ballpushing_utils.dataverse_naming.SERVER_DIRECTORY_TO_DATAVERSE`.

    The label is the right namespace prefix for fly IDs in either
    layout — the on-server file stem maps 1:1 to one experiment, and
    the Dataverse ``experiment`` column gives the same identifier.

    Raises :class:`FileNotFoundError` when neither layout resolves
    anything for ``server_directory``.
    """
    from .dataverse_naming import dataverse_directory_candidates
    from .paths import data_root

    abs_path = Path(server_directory)

    # On-server layout: a real directory with per-experiment files.
    if abs_path.is_dir():
        files = sorted(abs_path.glob("*_coordinates.feather"))
        if files:
            for fp in files:
                yield fp.stem, read_feather(fp)
            return

    # Dataverse layout: walk the per-condition pooled feathers.
    try:
        rel = abs_path.relative_to(data_root())
    except ValueError:
        rel = abs_path

    candidates = dataverse_directory_candidates(rel)
    if not candidates:
        raise FileNotFoundError(
            f"iter_coordinate_feathers({server_directory!r}): no on-server "
            f"*_coordinates.feather files found and no Dataverse alias "
            f"covers this path. Either populate the directory, set "
            f"BALLPUSHING_DATA_ROOT, or run ``ballpushing-fetch`` to "
            f"download the matching pooled trajectory feathers."
        )

    yielded_any = False
    for basename in candidates:
        try:
            pooled = read_feather(basename)
        except FileNotFoundError:
            # A condition feather may be missing from a partial download.
            # Skip it rather than aborting — downstream code will surface
            # the gap via empty FeedingState/Light counts.
            continue
        if "experiment" not in pooled.columns:
            # Defensive — should never happen for a published pool.
            yield Path(basename).stem, pooled
            yielded_any = True
            continue
        for exp_name, group in pooled.groupby("experiment", sort=True):
            yield str(exp_name), group.reset_index(drop=True)
            yielded_any = True

    if not yielded_any:
        raise FileNotFoundError(
            f"iter_coordinate_feathers({server_directory!r}): none of the "
            f"Dataverse-aliased feathers {candidates!r} resolved. Run "
            f"``ballpushing-fetch`` or check BALLPUSHING_DATA_ROOT."
        )


def load_wildtype_experiment(
    experiment_name: str, *, allow_empty: bool = False
) -> pd.DataFrame:
    """Load all trajectory rows for a single wild-type experiment.

    Slice equivalent of an on-server per-fly coordinates feather
    (``<experiment>_coordinates.feather``) when only the Dataverse-
    published per-condition pools are available. Walks the six
    ``Wild-type_Lights-*_*_trajectories.feather`` files,
    concatenates rows whose ``experiment`` column matches
    ``experiment_name``, and returns the result.

    Used by figure scripts that read a specific cohort in their default
    mode (e.g. ``230704_FeedingState_1_AM_Videos_Tracked`` for Fig 1e).

    Parameters
    ----------
    experiment_name : str
        Value of the ``experiment`` column to filter on, typically the
        on-server experiment folder name.
    allow_empty : bool, optional
        If True, return an empty frame instead of raising when no rows
        match. Default False.

    Raises
    ------
    FileNotFoundError
        When none of the wild-type pooled feathers resolve at all (the
        user hasn't run ``ballpushing-fetch``).
    ValueError
        When the feathers resolve but no rows match ``experiment_name``
        and ``allow_empty`` is False.
    """
    from .dataverse_naming import WILDTYPE_TRAJECTORY_FEATHERS

    matches: list[pd.DataFrame] = []
    any_resolved = False
    for basename in WILDTYPE_TRAJECTORY_FEATHERS:
        try:
            pooled = read_feather(basename)
            any_resolved = True
        except FileNotFoundError:
            continue
        if "experiment" not in pooled.columns:
            continue
        slice_ = pooled[pooled["experiment"] == experiment_name]
        if not slice_.empty:
            matches.append(slice_.reset_index(drop=True))

    if not any_resolved:
        raise FileNotFoundError(
            f"load_wildtype_experiment({experiment_name!r}): none of the "
            f"Wild-type_Lights-*_trajectories.feather files resolved. Run "
            f"``ballpushing-fetch`` first, or set BALLPUSHING_DATA_ROOT."
        )

    if not matches:
        if allow_empty:
            return pd.DataFrame()
        raise ValueError(
            f"load_wildtype_experiment({experiment_name!r}): experiment not "
            f"found in any Wild-type_Lights-*_trajectories.feather. Check the "
            f"experiment name; the 'experiment' column on Dataverse pools "
            f"holds on-server folder names verbatim."
        )

    return pd.concat(matches, ignore_index=True)
