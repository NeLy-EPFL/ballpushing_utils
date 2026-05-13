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


def _feather_column_names(path: Path) -> list[str]:
    """Return column names from a feather file without loading any data.

    Uses the Arrow IPC file footer — a few bytes at the end of the file —
    so the cost is a seek + small read regardless of file size.
    """
    import pyarrow as pa

    reader = pa.ipc.open_file(str(path))
    return reader.schema.names


def _columns_for_file(requested: list[str], on_disk: set[str]) -> list[str]:
    """Translate requested column names to their on-disk equivalents.

    Maps modern ``speed*`` names to the legacy ``velocity*`` names when the
    legacy name is present on disk but the modern one isn't. Columns that
    exist under their modern name are left unchanged. Columns not found in
    either form are passed through as-is so pandas raises a clear error.
    """
    result = []
    for col in requested:
        if col in on_disk:
            result.append(col)
        elif col in LEGACY_COLUMN_RENAMES and LEGACY_COLUMN_RENAMES[col] in on_disk:
            result.append(LEGACY_COLUMN_RENAMES[col])
        else:
            result.append(col)
    return result


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


def read_feather(
    path: str | os.PathLike[str],
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """:func:`pandas.read_feather` + :func:`normalize_legacy_columns`.

    Drop-in replacement for ``pandas.read_feather``.  The returned frame
    is guaranteed to have the modern ``speed*`` column names regardless of
    dataset vintage.

    Parameters
    ----------
    path : str or path-like
        Path to the feather file, or a server-relative path resolved via
        ``BALLPUSHING_DATA_ROOT`` / the Dataverse alias table.
    columns : list of str, optional
        Subset of columns to load.  Only those columns are read off disk —
        the rest never enter RAM.  Modern ``speed*`` names are translated
        to their legacy ``velocity*`` equivalents automatically when the
        file predates the rename, so callers always use modern names.
        ``None`` (default) loads all columns.

    Notes
    -----
    When ``path`` doesn't resolve directly this falls back to
    :func:`ballpushing_utils.paths.find_feather`.  Split Dataverse parts
    (``<stem>-1.feather``, ``<stem>-2.feather``, …) are concatenated in
    order; ``columns`` is applied to every part.

    Raises :class:`FileNotFoundError` when the path can't be resolved.
    """
    from pathlib import Path

    actual_path = Path(path)
    if not actual_path.exists():
        from .paths import find_feather, missing_data_message

        resolved = find_feather(path)
        if resolved is None:
            raise FileNotFoundError(missing_data_message(path, context="feather"))
        if isinstance(resolved, list):
            if columns is not None:
                on_disk = set(_feather_column_names(resolved[0]))
                actual_cols = _columns_for_file(columns, on_disk)
                frames = [pd.read_feather(p, columns=actual_cols, **kwargs) for p in resolved]
            else:
                frames = [pd.read_feather(p, **kwargs) for p in resolved]
            df = pd.concat(frames, ignore_index=True)
            return normalize_legacy_columns(df)
        actual_path = resolved

    if columns is not None:
        on_disk = set(_feather_column_names(actual_path))
        actual_cols = _columns_for_file(columns, on_disk)
        df = pd.read_feather(actual_path, columns=actual_cols, **kwargs)
    else:
        df = pd.read_feather(actual_path, **kwargs)
    return normalize_legacy_columns(df)


def iter_coordinate_feathers(
    server_directory: str | os.PathLike[str],
    *,
    columns: list[str] | None = None,
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

    Parameters
    ----------
    server_directory : str or path-like
        Path to the coordinates directory (on-server) or a server-relative
        path covered by the Dataverse alias table.
    columns : list of str, optional
        Subset of columns to yield in each dataframe.  Only those columns
        are read off disk.  Legacy ``velocity*`` → ``speed*`` translation
        is applied automatically.  ``None`` (default) yields all columns.

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
                yield fp.stem, read_feather(fp, columns=columns)
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
        # Always include "experiment" for the internal groupby split, even
        # if the caller didn't request it.
        if columns is not None and "experiment" not in columns:
            read_cols: list[str] | None = ["experiment"] + list(columns)
        else:
            read_cols = columns
        try:
            pooled = read_feather(basename, columns=read_cols)
        except FileNotFoundError:
            # A condition feather may be missing from a partial download.
            # Skip it rather than aborting — downstream code will surface
            # the gap via empty FeedingState/Light counts.
            continue
        if "experiment" not in pooled.columns:
            # Defensive — should never happen for a published pool.
            out = pooled[columns] if (columns is not None) else pooled
            yield Path(basename).stem, out
            yielded_any = True
            continue
        for exp_name, group in pooled.groupby("experiment", sort=True):
            group = group.reset_index(drop=True)
            # Drop "experiment" if the caller didn't ask for it.
            if columns is not None and "experiment" not in columns:
                group = group.drop(columns=["experiment"])
            yield str(exp_name), group
            yielded_any = True

    if not yielded_any:
        raise FileNotFoundError(
            f"iter_coordinate_feathers({server_directory!r}): none of the "
            f"Dataverse-aliased feathers {candidates!r} resolved. Run "
            f"``ballpushing-fetch`` or check BALLPUSHING_DATA_ROOT."
        )


def load_wildtype_experiment(
    experiment_name: str,
    *,
    columns: list[str] | None = None,
    allow_empty: bool = False,
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
    columns : list of str, optional
        Subset of columns to return.  ``"experiment"`` is always loaded
        internally for the row filter but dropped from the result unless
        explicitly requested.  ``None`` returns all columns.
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

    # Always include "experiment" for row filtering; strip it from the
    # result later if the caller didn't ask for it.
    want_experiment = columns is None or "experiment" in columns
    if columns is not None and not want_experiment:
        read_cols: list[str] | None = ["experiment"] + list(columns)
    else:
        read_cols = columns

    matches: list[pd.DataFrame] = []
    any_resolved = False
    for basename in WILDTYPE_TRAJECTORY_FEATHERS:
        try:
            pooled = read_feather(basename, columns=read_cols)
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

    result = pd.concat(matches, ignore_index=True)
    if columns is not None and not want_experiment and "experiment" in result.columns:
        result = result.drop(columns=["experiment"])
    return result
