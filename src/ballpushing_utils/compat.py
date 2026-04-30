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
from typing import Any

import pandas as pd

__all__ = ["LEGACY_COLUMN_RENAMES", "normalize_legacy_columns", "read_feather"]


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
    ``BALLPUSHING_DATA_ROOT`` and any directory in
    ``BALLPUSHING_FEATHER_SEARCH`` for a matching basename. That makes
    the figure scripts work for users who downloaded a single feather
    from the Dataverse and dropped it anywhere on disk — they don't
    need to recreate the on-server directory hierarchy. If the search
    still fails (or is ambiguous), raises :class:`FileNotFoundError`
    with the standard three-option breadcrumb from
    :func:`ballpushing_utils.paths.missing_data_message`.
    """
    from pathlib import Path

    actual_path = Path(path)
    if not actual_path.exists():
        from .paths import find_feather, missing_data_message

        resolved = find_feather(path)
        if resolved is None:
            raise FileNotFoundError(missing_data_message(path, context="feather"))
        actual_path = resolved

    df = pd.read_feather(actual_path, *args, **kwargs)
    return normalize_legacy_columns(df)
