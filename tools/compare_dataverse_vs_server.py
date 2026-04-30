"""Compare per-fly summary feathers built from Dataverse vs server data.

This script verifies that anyone who downloads the Dataverse archive and runs
``dataset_builder.py --dataverse-root`` produces numerically identical summary
metrics to the canonical server-side pipeline.

Usage
-----
Step 1 – build server feathers (run from repo root):

    python src/dataset_builder.py \\
        --mode flies \\
        --yaml experiments_yaml/lc6_server_flies.yaml \\
        --datasets summary \\
        --experiment-type TNT

    (outputs land in $BALLPUSHING_DATA_ROOT/MultiMazeRecorder/Datasets/summary/
     unless you override CONFIG["PATHS"]["dataset_dir"] in dataset_builder.py
     or pass --output-dir if that flag exists)

    For a self-contained test, temporarily set dataset_dir to a local path
    inside the repo, e.g. outputs/test_comparison/server, before running.

Step 2 – build Dataverse feathers:

    python src/dataset_builder.py \\
        --dataverse-root ~/Downloads \\
        --experiment-type TNT \\
        --datasets summary

    (The LC6 subfolder is the condition folder; the root must be ~/Downloads
     so iter_dataverse_flies sees <root>/LC6/<date>/Arena*/Corridor*)

    Direct output to outputs/test_comparison/dataverse the same way.

Step 3 – compare:

    python tools/compare_dataverse_vs_server.py \\
        --server   outputs/test_comparison/server/summary \\
        --dataverse outputs/test_comparison/dataverse/summary

The script exits 0 if all matched pairs are identical on every metric column,
non-zero if any mismatch or unmatched fly is found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Columns that are EXPECTED to differ between server and Dataverse output
# and should not be compared as metrics.
# ---------------------------------------------------------------------------
_SKIP_COLS = frozenset(
    [
        "fly",  # full path-derived identifier
        "experiment",  # experiment folder name (contains date + rig suffix on server)
        "Genotype",  # server: TNTxG56 / TNTxG74 / TNTxG76; Dataverse: LC6
        # Any other path-derived columns you have:
        "dataset_dir",
        "source",
    ]
)

# Regex to pull (date, arena_number, corridor_number) from a feather filename.
# Handles both:
#   231115_TNT_Fine_1_Videos_Tracked_arena4_corridor1_summary.feather  (server)
#   231115_Arena4_Corridor1_summary.feather                            (dataverse)
_KEY_RE = re.compile(
    r"^(\d{6})"  # date  (group 1)
    r".*?"  # anything in between (greedy-lazy)
    r"[Aa]rena(\d+)"  # arena number (group 2)
    r".*?"
    r"[Cc]orridor(\d+)",  # corridor number (group 3)
    re.IGNORECASE,
)


def parse_key(feather_path: Path) -> tuple[str, str, str] | None:
    """Return (date, arena_num, corridor_num) parsed from a feather filename."""
    m = _KEY_RE.match(feather_path.stem)
    if m is None:
        return None
    return (m.group(1), m.group(2), m.group(3))


def load_feathers(directory: Path) -> dict[tuple, pd.DataFrame]:
    """Load all *_summary.feather files in ``directory``, keyed by (date, arena, corridor)."""
    feathers: dict[tuple, pd.DataFrame] = {}
    for f in sorted(directory.glob("*_summary.feather")):
        key = parse_key(f)
        if key is None:
            print(f"  [WARN] Cannot parse key from filename: {f.name} — skipping")
            continue
        feathers[key] = pd.read_feather(f)
    return feathers


def compare_pair(
    key: tuple,
    server_df: pd.DataFrame,
    dv_df: pd.DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[str]:
    """Return a list of failure messages for this fly pair (empty = pass)."""
    date, arena, corridor = key
    label = f"date={date} arena={arena} corridor={corridor}"
    failures: list[str] = []

    # Determine columns to compare: numeric, present in both, not in skip list
    server_numeric = set(server_df.select_dtypes(include=[np.number]).columns)
    dv_numeric = set(dv_df.select_dtypes(include=[np.number]).columns)
    common = (server_numeric & dv_numeric) - _SKIP_COLS

    missing_in_dv = (server_numeric - dv_numeric) - _SKIP_COLS
    missing_in_server = (dv_numeric - server_numeric) - _SKIP_COLS

    if missing_in_dv:
        failures.append(f"  {label}: columns in server but missing in Dataverse: {sorted(missing_in_dv)}")
    if missing_in_server:
        failures.append(f"  {label}: columns in Dataverse but missing in server: {sorted(missing_in_server)}")

    if server_df.shape[0] != dv_df.shape[0]:
        failures.append(f"  {label}: row count mismatch — server={server_df.shape[0]} dataverse={dv_df.shape[0]}")
        return failures  # no point comparing values if shapes differ

    for col in sorted(common):
        sv = server_df[col].to_numpy(dtype=float, na_value=np.nan)
        dv = dv_df[col].to_numpy(dtype=float, na_value=np.nan)

        nan_match = np.isnan(sv) == np.isnan(dv)
        if not nan_match.all():
            failures.append(
                f"  {label} [{col}]: NaN pattern differs — "
                f"server NaNs at {np.where(np.isnan(sv))[0].tolist()}, "
                f"dataverse NaNs at {np.where(np.isnan(dv))[0].tolist()}"
            )
            continue

        non_nan = ~np.isnan(sv)
        if not np.allclose(sv[non_nan], dv[non_nan], rtol=rtol, atol=atol, equal_nan=False):
            max_diff = np.nanmax(np.abs(sv - dv))
            failures.append(f"  {label} [{col}]: values differ (max abs diff={max_diff:.3e})")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare server-built vs Dataverse-built summary feathers.")
    parser.add_argument(
        "--server",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing server-side per-fly *_summary.feather files",
    )
    parser.add_argument(
        "--dataverse",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing Dataverse-side per-fly *_summary.feather files",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for numeric comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for numeric comparison (default: 1e-8)",
    )
    args = parser.parse_args()

    print(f"Loading server feathers from:    {args.server}")
    server_map = load_feathers(args.server)
    print(f"  Found {len(server_map)} flies")

    print(f"Loading Dataverse feathers from: {args.dataverse}")
    dv_map = load_feathers(args.dataverse)
    print(f"  Found {len(dv_map)} flies")

    server_keys = set(server_map)
    dv_keys = set(dv_map)
    matched = server_keys & dv_keys
    only_server = server_keys - dv_keys
    only_dv = dv_keys - server_keys

    if only_server:
        print(f"\n[WARN] {len(only_server)} flies only in SERVER (no Dataverse match):")
        for k in sorted(only_server):
            print(f"  date={k[0]} arena={k[1]} corridor={k[2]}")

    if only_dv:
        print(f"\n[WARN] {len(only_dv)} flies only in DATAVERSE (no server match):")
        for k in sorted(only_dv):
            print(f"  date={k[0]} arena={k[1]} corridor={k[2]}")

    print(f"\nComparing {len(matched)} matched pairs …")
    all_failures: list[str] = []
    passes = 0

    for key in sorted(matched):
        failures = compare_pair(key, server_map[key], dv_map[key], rtol=args.rtol, atol=args.atol)
        if failures:
            all_failures.extend(failures)
        else:
            passes += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passes}/{len(matched)} flies PASSED")

    if all_failures:
        print(f"FAILURES ({len(all_failures)} issues):")
        for msg in all_failures:
            print(msg)
        return 1

    if only_server or only_dv:
        print("All matched pairs are identical, but some flies are unmatched (see warnings above).")
        return 1

    print("All flies matched and all metric columns are numerically identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
