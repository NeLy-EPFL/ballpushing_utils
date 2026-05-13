#!/usr/bin/env python3
"""
Build standardized_contacts datasets grouped by brain region for the TNT screen.

Reads the per-date ``*_standardized_contacts.feather`` files produced by
``dataset_builder.py`` and reorganises them into one feather file per brain
region, sized for upload to Harvard Dataverse (≤ 2.5 GB each).

Output filenames:
    <output_dir>/<RegionName>_standardized_contacts.feather

Files exceeding the size limit are split by genotype (first-fit-decreasing
bin packing, same logic as ``split_region_trajectories.py``):

    <output_dir>/<RegionName>_standardized_contacts-1.feather
    <output_dir>/<RegionName>_standardized_contacts-2.feather
    ...

Source files are never modified. The original per-date layout remains intact
for lab users; the brain-region feathers are for Dataverse distribution.

After running this script, upload the output directory to the silencing-screen
Dataverse archive, then update ``SCREEN_STANDARDIZED_CONTACTS_FEATHERS`` in
``src/ballpushing_utils/dataverse_naming.py`` if any filenames changed.

Usage
-----
Dry run (show region summary, no files written):
    python build_region_standardized_contacts.py --dry-run

Full run (write to default output dir):
    python build_region_standardized_contacts.py

Custom output:
    python build_region_standardized_contacts.py \\
        --output-dir /path/to/output

Process only specific regions (use remapped names):
    python build_region_standardized_contacts.py --regions MB Olfaction
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ballpushing_utils.paths import dataset

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(
    "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/"
    "250809_02_standardized_contacts_TNT_screen_Data/standardized_contacts"
)
DEFAULT_OUTPUT_DIR = dataset(
    "Affordance_article_dataverse/silencing_screen/standardized_contacts"
)
DEFAULT_MAX_GB = 2.5
BRAIN_REGION_COL = "Brain region"
GENOTYPE_COL = "Genotype"

# ---------------------------------------------------------------------------
# Brain-region name normalisation
# Same mapping as build_region_trajectories.py for consistency.
# Regions not listed keep their original name (spaces → underscores).
# ---------------------------------------------------------------------------

REGION_REMAP: dict[str, str] = {
    "CX": "Central_Complex",
    "LH": "Lateral_Horn",
    "MB": "Mushroom_Body",
    "MB extrinsic neurons": "MB_extrinsic",
    "DN": "Others",
    "JON": "Others",
    "fchON": "Others",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def remap_region(raw: str) -> str:
    """Return the output-filename stem for a raw ``Brain region`` value."""
    mapped = REGION_REMAP.get(raw, raw)
    return mapped.replace(" ", "_").replace("/", "-")


def _assign_genotypes_to_parts(
    geno_sizes: list[tuple[str, int]],
    max_rows_per_part: int,
) -> list[list[str]]:
    """First-fit-decreasing bin packing: assign genotypes to parts.

    Keeps all rows of the same genotype in the same part so that external
    users can load a single part and have complete per-genotype data.
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


# ---------------------------------------------------------------------------
# Core: write one region (with optional splitting)
# ---------------------------------------------------------------------------


def write_region(
    region: str,
    df: pd.DataFrame,
    output_dir: Path,
    max_bytes: int,
    *,
    overwrite: bool = False,
) -> None:
    """Write ``df`` (all rows for ``region``) to the output directory.

    If the resulting feather exceeds ``max_bytes`` the data is split across
    multiple numbered parts; otherwise a single file is written.  Existing
    files are skipped unless ``overwrite`` is True.
    """
    safe_name = remap_region(region)
    stem = f"{safe_name}_standardized_contacts"
    single_path = output_dir / f"{stem}.feather"

    if not overwrite:
        if single_path.exists() or any(output_dir.glob(f"{stem}-*.feather")):
            logging.info(
                f"[{region}] Already written, skipping (use --overwrite to replace)."
            )
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to a temp file to measure the actual on-disk size before committing.
    tmp_path = output_dir / f".{stem}.tmp.feather"
    logging.info(f"[{region}] Writing {len(df)} rows to temporary file…")
    df.reset_index(drop=True).to_feather(tmp_path)
    file_size = tmp_path.stat().st_size
    size_gb = file_size / 1e9

    if file_size <= max_bytes:
        tmp_path.rename(single_path)
        logging.info(
            f"[{region}] ✅  {single_path.name}  ({size_gb:.2f} GB,  {len(df)} rows)"
        )
        return

    # --- Split required ---------------------------------------------------
    logging.info(
        f"[{region}] {size_gb:.2f} GB exceeds limit — splitting by {GENOTYPE_COL}…"
    )
    tmp_path.unlink()

    if GENOTYPE_COL not in df.columns:
        logging.warning(
            f"[{region}] No '{GENOTYPE_COL}' column — splitting into equal chunks."
        )
        _split_equal_chunks(region, stem, df, output_dir, max_bytes, file_size)
        return

    bytes_per_row = file_size / len(df)
    # Apply a 15 % safety margin: per-genotype feather compression varies from
    # the whole-region average, so a purely row-count-based budget can overshoot.
    effective_max_bytes = int(max_bytes * 0.85)
    max_rows_per_part = int(effective_max_bytes / bytes_per_row)

    geno_counts = df[GENOTYPE_COL].value_counts().sort_values(ascending=False)
    geno_sizes = list(geno_counts.items())

    oversized = [(g, n) for g, n in geno_sizes if n > max_rows_per_part]
    if oversized:
        logging.warning(
            f"[{region}] {len(oversized)} genotype(s) exceed the part size on their own "
            f"and will each become a separate part: {[g for g, _ in oversized]}"
        )

    parts = _assign_genotypes_to_parts(geno_sizes, max_rows_per_part)
    n_parts = len(parts)
    logging.info(
        f"[{region}] → {n_parts} part(s) (~{size_gb / n_parts:.2f} GB each,  {len(df)} rows total)"
    )

    for i, part_genos in enumerate(parts, start=1):
        part_df = df[df[GENOTYPE_COL].isin(part_genos)].copy().reset_index(drop=True)
        part_path = output_dir / f"{stem}-{i}.feather"
        part_df.to_feather(part_path)
        actual_gb = part_path.stat().st_size / 1e9
        status = "⚠️  OVER LIMIT" if part_path.stat().st_size > max_bytes else "✅"
        logging.info(
            f"[{region}]   part {i}/{n_parts}: {part_path.name}  "
            f"({len(part_df)} rows,  {actual_gb:.2f} GB)  {status}"
        )
        if part_path.stat().st_size > max_bytes:
            logging.warning(
                f"[{region}] Part {i} is {actual_gb:.2f} GB — exceeds the {max_bytes/1e9:.1f} GB "
                f"limit. The genotype(s) in this part may have above-average row density. "
                f"Consider splitting this part manually or reducing --max-gb."
            )


def _split_equal_chunks(
    region: str,
    stem: str,
    df: pd.DataFrame,
    output_dir: Path,
    max_bytes: int,
    file_size: int,
) -> None:
    """Fallback: equal-row chunks when no Genotype column is present."""
    n_rows = len(df)
    bytes_per_row = file_size / n_rows
    max_rows = int(max_bytes * 0.85 / bytes_per_row)  # 15 % safety margin
    n_parts = -(-n_rows // max_rows)  # ceiling division
    for i in range(n_parts):
        chunk = df.iloc[i * max_rows : (i + 1) * max_rows].copy().reset_index(drop=True)
        part_path = output_dir / f"{stem}-{i + 1}.feather"
        chunk.to_feather(part_path)
        actual_gb = part_path.stat().st_size / 1e9
        logging.info(
            f"[{region}]   chunk {i + 1}/{n_parts}: {part_path.name}  ({actual_gb:.2f} GB)"
        )


# ---------------------------------------------------------------------------
# Discovery: scan source files cheaply (one column only)
# ---------------------------------------------------------------------------


def discover_regions(feather_files: list[Path]) -> dict[str, list[Path]]:
    """Return ``{remapped_region: [files_that_contain_it]}``.

    Reads only the ``Brain region`` column from each feather — very cheap
    with feather's columnar format.
    """
    region_to_files: dict[str, list[Path]] = {}
    for path in feather_files:
        try:
            df_col = pd.read_feather(path, columns=[BRAIN_REGION_COL])
        except Exception as exc:
            logging.warning(f"Could not read {path.name}: {exc}")
            continue
        for raw in df_col[BRAIN_REGION_COL].dropna().unique():
            mapped = remap_region(str(raw))
            region_to_files.setdefault(mapped, []).append(path)
    return region_to_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one standardized_contacts feather per brain region from "
            "the TNT silencing-screen per-date feather files."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help=f"Directory with per-date *_standardized_contacts.feather files (default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for brain-region feathers (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=DEFAULT_MAX_GB,
        help=f"Maximum file size in GB before splitting (default: {DEFAULT_MAX_GB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a region summary without writing any files.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        metavar="REGION",
        help=(
            "Only process these brain regions (use remapped names, e.g. "
            "--regions Central_Complex Olfaction). Default: all."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    source_dir: Path = args.source_dir
    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    feather_files = sorted(source_dir.glob("2*.feather"))
    if not feather_files:
        logging.error(f"No 2*.feather files found in {source_dir}")
        sys.exit(1)

    logging.info(f"Found {len(feather_files)} per-date feather file(s) in {source_dir}")
    max_bytes = int(args.max_gb * 1e9)

    # ---- Discovery pass (cheap: one column per file) -----
    logging.info("Scanning brain regions across source files…")
    region_to_files = discover_regions(feather_files)
    all_regions = sorted(region_to_files)
    logging.info(f"Brain regions found: {all_regions}")

    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN")
        print(f"{'='*60}")
        print(f"Source  : {source_dir}  ({len(feather_files)} files)")
        print(f"Output  : {args.output_dir}")
        print(f"Max size: {args.max_gb} GB\n")
        print(f"{'Region':<30} {'#files':>7}  {'Expected output name'}")
        print("-" * 80)
        for region in all_regions:
            if args.regions and region not in args.regions:
                continue
            n_files = len(region_to_files[region])
            stem = f"{region}_standardized_contacts"
            print(
                f"  {region:<28} {n_files:>7}  {stem}.feather  (or -1.feather/-2.feather if >2.5 GB)"
            )
        print(f"\n{'='*60}")
        print("Dry run complete. No files written.")
        return

    # ---- Filter to requested regions ----
    regions_to_process = all_regions
    if args.regions:
        requested = set(args.regions)
        regions_to_process = [r for r in all_regions if r in requested]
        missing = requested - set(regions_to_process)
        if missing:
            logging.warning(
                f"Requested region(s) not found in data: {sorted(missing)}"
                f"  Available: {all_regions}"
            )
        if not regions_to_process:
            logging.error("No matching regions to process.")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Processing: one region at a time to limit peak memory ----
    for region in regions_to_process:
        files_for_region = region_to_files[region]
        logging.info(f"[{region}] Loading from {len(files_for_region)} file(s)…")
        chunks: list[pd.DataFrame] = []
        for path in files_for_region:
            df_full = pd.read_feather(path)
            # Re-map the Brain region column for correct grouping
            mask = (
                df_full[BRAIN_REGION_COL].map(lambda r: remap_region(str(r))) == region
            )
            chunk = df_full[mask]
            if not chunk.empty:
                chunks.append(chunk)
            del df_full  # free memory

        if not chunks:
            logging.warning(f"[{region}] No rows found after filtering — skipping.")
            continue

        logging.info(f"[{region}] Concatenating {len(chunks)} chunk(s)…")
        df_region = pd.concat(chunks, ignore_index=True)
        del chunks

        write_region(
            region, df_region, args.output_dir, max_bytes, overwrite=args.overwrite
        )
        del df_region

    print("\nDone.")


if __name__ == "__main__":
    main()
