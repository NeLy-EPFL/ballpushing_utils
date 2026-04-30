"""Compare summary metrics between server-built and Dataverse-built feathers.

Loads both pooled feathers, optionally filters the server one to a specific
condition (e.g. F1_condition=='pretrained' or Magnet=='y'), aligns on
(fly key, ball_idx), and prints per-metric deltas.

Usage
-----
    # F1 pretrained comparison
    python tools/compare_server_dataverse.py \\
        --server   outputs/f1_server/... \\
        --dataverse outputs/f1_dv/... \\
        --condition-filter F1_condition=pretrained

    # MagnetBlock Blocked comparison
    python tools/compare_server_dataverse.py \\
        --server   outputs/mb_blocked_server_test/... \\
        --dataverse outputs/mb_blocked_dv_test/... \\
        --condition-filter Magnet=y

The script auto-discovers the single pooled feather under each directory if a
directory is passed instead of a file.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Metrics to compare (a representative subset of the summary columns).
METRICS = [
    "distance_moved",
    "max_distance",
    "nb_events",
    "nb_significant_events",
    "f1_exit_time",
    "flailing",
    "head_pushing_ratio",
    "fraction_not_facing_ball",
    "interaction_proportion",
    "overall_interaction_rate",
    "persistence_at_end",
    "pushed",
    "pulled",
]


def _resolve_feather(path: Path) -> Path:
    """Return the feather file itself, or discover it under a directory."""
    if path.is_file():
        return path
    candidates = list(path.rglob("pooled_summary.feather"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        sys.exit(f"Multiple pooled_summary.feather files under {path}. " "Pass the exact file path.")
    sys.exit(f"No pooled_summary.feather found under {path}.")


def _fly_key(df: pd.DataFrame) -> pd.Series:
    """Derive a stable join key: '<YYMMDD>_<ArenaN>_<SlotOrSide>'.

    Handles two naming conventions:
    * F1 / Left-Right:  ``251008_Arena1_Left``  (Dataverse)
                        ``251008_F1_New_Videos_Checked_arena1_Left``  (server)
    * MagnetBlock:      ``240710_Arena2_Corridor1``  (Dataverse)
                        ``240710_MagnetBlock_Videos_Tracked_arena2_corridor1``  (server)
    """

    def _normalise(s: str) -> str:
        parts = s.split("_")
        # Date is always the first 6-digit token.
        date = parts[0]
        # Arena: case-insensitive 'arenaN' where N is an integer.
        arena = next(
            (p for p in parts if p.lower().startswith("arena") and p[5:].isdigit()),
            None,
        )
        # Slot: either Left/Right (F1) or corridorN (MagnetBlock).
        slot = next((p for p in parts if p in ("Left", "Right")), None)
        if slot is None:
            # Try corridorN (case-insensitive).
            slot = next(
                (p for p in parts if p.lower().startswith("corridor") and p[8:].isdigit()),
                None,
            )
            if slot is not None:
                slot = "Corridor" + slot[8:]  # normalise capitalisation
        if arena and slot:
            arena = "Arena" + arena[5:]  # normalise capitalisation
            return f"{date}_{arena}_{slot}"
        return s  # fallback: keep original

    return df["fly"].apply(_normalise)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        required=True,
        type=Path,
        help="Path to the server-built pooled feather (or its parent directory).",
    )
    parser.add_argument(
        "--dataverse",
        required=True,
        type=Path,
        help="Path to the Dataverse-built pooled feather (or its parent directory).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV file to save per-fly metric deltas (optional).",
    )
    parser.add_argument(
        "--ball-idx",
        type=int,
        default=None,
        help=(
            "Restrict comparison to a specific ball_idx row per fly "
            "(0 = training ball, 1 = test ball). "
            "Default: compare both rows independently."
        ),
    )
    parser.add_argument(
        "--condition-filter",
        default=None,
        metavar="COL=VALUE",
        help=(
            "Filter the server feather to rows where COL==VALUE before comparing. "
            "E.g. 'F1_condition=pretrained' or 'Magnet=y'. "
            "No filter is applied to the Dataverse feather (assumed already condition-specific)."
        ),
    )
    args = parser.parse_args()

    server_path = _resolve_feather(args.server)
    dv_path = _resolve_feather(args.dataverse)

    print(f"Server feather : {server_path}")
    print(f"Dataverse feather: {dv_path}")

    srv = pd.read_feather(server_path)
    dv = pd.read_feather(dv_path)

    # Optionally filter the server feather to a specific condition.
    n_before = len(srv)
    if args.condition_filter:
        col, _, val = args.condition_filter.partition("=")
        col = col.strip()
        val = val.strip()
        if col not in srv.columns:
            sys.exit(f"Column '{col}' not found in server feather. Available: {list(srv.columns)}")
        srv = srv[srv[col] == val].copy()
        print(f"\nServer : {n_before} rows total → {len(srv)} after filtering to {col}=='{val}'")
    else:
        print(f"\nServer : {n_before} rows (no condition filter applied)")
    print(f"Dataverse: {len(dv)} rows")

    if args.ball_idx is not None:
        srv = srv[srv["ball_idx"] == args.ball_idx].copy()
        dv = dv[dv["ball_idx"] == args.ball_idx].copy()
        print(f"Both filtered to ball_idx == {args.ball_idx}")

    # Build join key.
    srv["_key"] = _fly_key(srv)
    dv["_key"] = _fly_key(dv)

    # Include ball_idx in the key if both rows per fly are present.
    join_cols = ["_key", "ball_idx"] if args.ball_idx is None else ["_key"]

    srv_keyed = srv.set_index(join_cols)
    dv_keyed = dv.set_index(join_cols)

    common_keys = srv_keyed.index.intersection(dv_keyed.index)
    only_server = srv_keyed.index.difference(dv_keyed.index)
    only_dv = dv_keyed.index.difference(srv_keyed.index)

    print(f"\nMatched fly×ball_idx pairs : {len(common_keys)}")
    if len(only_server):
        print(f"  Only in server  ({len(only_server)}): {list(only_server)[:5]}")
    if len(only_dv):
        print(f"  Only in Dataverse ({len(only_dv)}): {list(only_dv)[:5]}")

    if not len(common_keys):
        print("\nNo matched pairs — cannot compare. Check fly key parsing.")
        sys.exit(1)

    # Align and compute deltas.
    available = [m for m in METRICS if m in srv_keyed.columns and m in dv_keyed.columns]
    missing = [m for m in METRICS if m not in available]
    if missing:
        print(f"\n  Metrics not in both feathers (skipped): {missing}")

    srv_aligned = srv_keyed.loc[common_keys, available]
    dv_aligned = dv_keyed.loc[common_keys, available]

    delta = dv_aligned - srv_aligned

    print(f"\n{'Metric':<35} {'n':>4} {'mean_delta':>12} {'max|delta|':>12} {'matched':>8}")
    print("-" * 75)
    rows = []
    for m in available:
        d = delta[m].dropna()
        if d.empty:
            continue
        print(
            f"{m:<35} {len(d):>4} {d.mean():>12.4f} {d.abs().max():>12.4f}" f" {'✓' if d.abs().max() < 1e-6 else '':>8}"
        )
        # per-fly rows for the CSV
        for idx, val in delta[m].items():
            rows.append(
                {"key": idx, "metric": m, "dv": dv_aligned.loc[idx, m], "server": srv_aligned.loc[idx, m], "delta": val}
            )

    if args.out and rows:
        out_df = pd.DataFrame(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nPer-fly delta table saved to {args.out}")


if __name__ == "__main__":
    main()
