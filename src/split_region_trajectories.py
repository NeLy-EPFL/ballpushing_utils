#!/usr/bin/env python3
"""
Split large region trajectory feather files into ≤ 2.5 GB parts for upload
to Harvard Dataverse, keeping all rows of the same genotype in the same part.

For a file like  Mushroom_Body_trajectories.feather  that exceeds the limit the
script produces:
    Mushroom_Body_trajectories-1.feather
    Mushroom_Body_trajectories-2.feather
    ...
and removes the original oversized file.

Files that are already within the size limit are left untouched.

Usage
-----
Dry run (no files written, just shows what would happen):
    python split_region_trajectories.py --dir /path/to/region_datasets --dry-run

Full run:
    python split_region_trajectories.py --dir /path/to/region_datasets

Custom size limit (e.g. 1 GB):
    python split_region_trajectories.py --dir /path/to/region_datasets --max-gb 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/region_trajectories")
DEFAULT_MAX_GB = 2.5
GENOTYPE_COL = "Genotype"  # column used to keep flies together

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


# ---------------------------------------------------------------------------
# Bin-packing helpers
# ---------------------------------------------------------------------------


def _assign_genotypes_to_parts(
    geno_sizes: list[tuple[str, int]],
    max_rows_per_part: int,
) -> list[list[str]]:
    """First-fit-decreasing bin packing.

    Parameters
    ----------
    geno_sizes:
        List of (genotype, row_count) sorted descending by row_count.
    max_rows_per_part:
        Maximum rows allowed per part (derived from size estimate).

    Returns
    -------
    List of parts, each part being a list of genotype strings.
    """
    parts: list[list[str]] = []
    part_totals: list[int] = []

    for geno, n_rows in geno_sizes:
        placed = False
        for i, total in enumerate(part_totals):
            if total + n_rows <= max_rows_per_part:
                parts[i].append(geno)
                part_totals[i] += n_rows
                placed = True
                break
        if not placed:
            parts.append([geno])
            part_totals.append(n_rows)

    return parts


def split_file(
    path: Path,
    max_bytes: int,
    dry_run: bool = False,
) -> None:
    """Split a single feather file if it exceeds max_bytes."""

    file_size = path.stat().st_size
    size_gb = file_size / 1e9

    if file_size <= max_bytes:
        logging.info(f"  OK  {path.name}  ({size_gb:.2f} GB) – no split needed")
        return

    logging.info(f"  LARGE  {path.name}  ({size_gb:.2f} GB) – loading for split…")
    df = pd.read_feather(path)

    if GENOTYPE_COL not in df.columns:
        logging.warning(f"  No '{GENOTYPE_COL}' column in {path.name}; " "will split by equal row chunks instead.")
        _split_by_chunks(path, df, max_bytes, dry_run)
        return

    total_rows = len(df)
    bytes_per_row = file_size / total_rows
    max_rows_per_part = int(max_bytes / bytes_per_row)

    # Build per-genotype row counts, sorted largest-first for bin packing
    geno_counts = df[GENOTYPE_COL].value_counts().sort_values(ascending=False).items()
    geno_sizes = [(g, n) for g, n in geno_counts]

    # Warn if any single genotype already exceeds the limit
    oversized = [(g, n) for g, n in geno_sizes if n > max_rows_per_part]
    if oversized:
        logging.warning(
            f"  {len(oversized)} genotype(s) exceed the part size on their own "
            f"(they will each become a separate part): "
            f"{[g for g, _ in oversized]}"
        )

    parts = _assign_genotypes_to_parts(geno_sizes, max_rows_per_part)
    n_parts = len(parts)
    est_part_gb = size_gb / n_parts

    logging.info(f"  → splitting into {n_parts} part(s) " f"(~{est_part_gb:.2f} GB each, {total_rows} rows total)")

    if dry_run:
        for i, part_genos in enumerate(parts, start=1):
            part_rows = df[df[GENOTYPE_COL].isin(part_genos)]
            est_bytes = len(part_rows) * bytes_per_row / 1e9
            print(
                f"    [dry-run] part {i}: {len(part_genos)} genotype(s), " f"{len(part_rows)} rows, ~{est_bytes:.2f} GB"
            )
            for g in sorted(part_genos):
                print(f"      {g}")
        return

    # Write parts
    stem = path.stem  # e.g. "Mushroom_Body_trajectories"
    suffix = path.suffix  # ".feather"
    out_dir = path.parent

    written: list[Path] = []
    for i, part_genos in enumerate(parts, start=1):
        part_df = df[df[GENOTYPE_COL].isin(part_genos)].copy().reset_index(drop=True)
        part_path = out_dir / f"{stem}-{i}{suffix}"
        part_df.to_feather(part_path)
        actual_gb = part_path.stat().st_size / 1e9
        logging.info(f"  Wrote part {i}/{n_parts}: {part_path.name}  " f"({len(part_df)} rows, {actual_gb:.2f} GB)")
        written.append(part_path)

    # Only remove original if all parts were written successfully
    path.unlink()
    logging.info(f"  Removed original: {path.name}")


def _split_by_chunks(
    path: Path,
    df: pd.DataFrame,
    max_bytes: int,
    dry_run: bool,
) -> None:
    """Fallback: split into equal row chunks when no Genotype column exists."""
    total_rows = len(df)
    file_size = path.stat().st_size
    bytes_per_row = file_size / total_rows
    max_rows = int(max_bytes / bytes_per_row)
    n_parts = -(-total_rows // max_rows)  # ceiling division

    logging.info(f"  → splitting into {n_parts} equal chunk(s)")
    if dry_run:
        for i in range(n_parts):
            start = i * max_rows
            end = min(start + max_rows, total_rows)
            est_gb = (end - start) * bytes_per_row / 1e9
            print(f"    [dry-run] chunk {i+1}: rows {start}–{end}, ~{est_gb:.2f} GB")
        return

    stem, suffix = path.stem, path.suffix
    for i in range(n_parts):
        start = i * max_rows
        chunk = df.iloc[start : start + max_rows].copy().reset_index(drop=True)
        part_path = path.parent / f"{stem}-{i+1}{suffix}"
        chunk.to_feather(part_path)
        actual_gb = part_path.stat().st_size / 1e9
        logging.info(f"  Wrote chunk {i+1}/{n_parts}: {part_path.name} ({actual_gb:.2f} GB)")

    path.unlink()
    logging.info(f"  Removed original: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split oversized region trajectory feather files for Dataverse upload."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_DIR),
        help=f"Directory containing *_trajectories.feather files (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=DEFAULT_MAX_GB,
        help=f"Maximum file size in GB per part (default: {DEFAULT_MAX_GB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split plan without writing or deleting any files.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    target_dir = Path(args.dir)
    if not target_dir.exists():
        logging.error(f"Directory not found: {target_dir}")
        sys.exit(1)

    max_bytes = int(args.max_gb * 1e9)

    # Find all *_trajectories.feather files (exclude already-split parts like -1.feather)
    candidates = sorted(
        f for f in target_dir.glob("*_trajectories.feather") if not any(f.stem.endswith(f"-{i}") for i in range(1, 50))
    )

    if not candidates:
        logging.warning(f"No *_trajectories.feather files found in {target_dir}")
        sys.exit(0)

    logging.info(f"Found {len(candidates)} trajectory file(s) in {target_dir}")
    if args.dry_run:
        print("\n[DRY RUN – no files will be written or deleted]\n")

    for path in candidates:
        split_file(path, max_bytes=max_bytes, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
