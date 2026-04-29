#!/usr/bin/env python3
"""
Build coordinates datasets grouped by brain region for the TNT screen.

For each brain region found in the split registry, all flies whose genotype maps
to that region are processed and their coordinate data is concatenated into a
single feather file:
    <output_dir>/<RegionName>_trajectories.feather

Usage
-----
Dry run (no files written, just prints the fly manifest):
    python build_region_trajectories.py --dry-run

Full run:
    python build_region_trajectories.py --output-dir /path/to/output

Full run with custom YAML:
    python build_region_trajectories.py \
        --yaml experiments_yaml/TNT_screen.yaml \
        --split-registry /mnt/upramdya_data/MD/Region_map_250908.csv \
        --output-dir /mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/region_trajectories \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

import ballpushing_utils

# ---------------------------------------------------------------------------
# Default paths – override with CLI flags
# ---------------------------------------------------------------------------
DEFAULT_YAML = Path(__file__).parent.parent / "experiments_yaml" / "TNT_screen.yaml"
DEFAULT_SPLIT_REGISTRY = Path("/mnt/upramdya_data/MD/Region_map_250908.csv")
DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/region_trajectories")
DATASET_TYPE = "coordinates"

# ---------------------------------------------------------------------------
# Brain-region name remapping
# Regions not listed here keep their original name from the split registry.
# Multiple registry names that map to the same key are merged into one file.
# ---------------------------------------------------------------------------
REGION_REMAP: dict[str, str] = {
    "CX": "Central_Complex",
    "LH": "Lateral_Horn",
    "MB": "Mushroom_Body",
    "MB extrinsic neurons": "MB_extrinsic",
    # Small / sparse regions grouped together
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


def load_experiment_dirs(yaml_path: Path) -> list[Path]:
    with open(yaml_path) as f:
        payload = yaml.safe_load(f) or {}
    return [Path(p) for p in payload.get("directories", [])]


def build_genotype_region_map(split_registry_path: Path) -> dict[str, str]:
    """Return {genotype_string: simplified_region}."""
    reg = pd.read_csv(split_registry_path)
    reg["Genotype"] = reg["Genotype"].astype(str).str.strip()
    reg["Simplified region"] = reg["Simplified region"].astype(str).str.strip()
    return dict(zip(reg["Genotype"], reg["Simplified region"]))


def parse_metadata(metadata_path: Path) -> dict[str, str]:
    """Parse Metadata.json and return {arena_name_lower: genotype}."""
    with open(metadata_path) as f:
        meta = json.load(f)

    variables: list[str] = meta.get("Variable", [])
    try:
        genotype_idx = [v.lower() for v in variables].index("genotype")
    except ValueError:
        logging.warning(f"No 'Genotype' variable in {metadata_path}")
        return {}

    arena_genotypes: dict[str, str] = {}
    for key, values in meta.items():
        if key.lower().startswith("arena") and isinstance(values, list):
            if len(values) > genotype_idx:
                raw = values[genotype_idx]
                # None / null in the JSON means the arena/corridor is empty – skip it
                if raw is None or str(raw).strip().lower() in ("none", "null", ""):
                    continue
                arena_genotypes[key.lower()] = str(raw).strip()
    return arena_genotypes


def build_fly_manifest(
    experiment_dirs: list[Path],
    genotype_region_map: dict[str, str],
) -> pd.DataFrame:
    """
    Walk all experiment dirs and build a table with columns:
        fly_dir, experiment, arena, corridor, genotype, brain_region
    """
    rows = []
    missing_metadata: list[Path] = []
    unmapped_genotypes: set[str] = set()

    for exp_dir in experiment_dirs:
        if not exp_dir.exists():
            logging.warning(f"Experiment dir not found, skipping: {exp_dir}")
            continue

        meta_path = exp_dir / "Metadata.json"
        if not meta_path.exists():
            missing_metadata.append(exp_dir)
            logging.warning(f"No Metadata.json in {exp_dir}")
            continue

        arena_geno = parse_metadata(meta_path)
        if not arena_geno:
            continue

        # Iterate arena dirs
        for arena_dir in sorted(exp_dir.iterdir()):
            if not arena_dir.is_dir():
                continue
            arena_key = arena_dir.name.lower()
            if not arena_key.startswith("arena"):
                continue

            genotype = arena_geno.get(arena_key)
            if genotype is None:
                logging.debug(f"Arena {arena_dir.name} not in metadata for {exp_dir.name}")
                continue

            region = genotype_region_map.get(genotype)
            if region is None:
                unmapped_genotypes.add(genotype)
                region = "UNMAPPED"
            else:
                region = REGION_REMAP.get(region, region)

            # Iterate corridor dirs
            for corridor_dir in sorted(arena_dir.iterdir()):
                if not corridor_dir.is_dir():
                    continue
                if not corridor_dir.name.lower().startswith("corridor"):
                    continue

                rows.append(
                    {
                        "fly_dir": corridor_dir,
                        "experiment": exp_dir.name,
                        "arena": arena_dir.name,
                        "corridor": corridor_dir.name,
                        "genotype": genotype,
                        "brain_region": region,
                    }
                )

    if missing_metadata:
        logging.warning(f"{len(missing_metadata)} experiment(s) had no Metadata.json")

    if unmapped_genotypes:
        logging.warning(
            f"{len(unmapped_genotypes)} genotype(s) not found in split registry "
            f"(tagged as UNMAPPED): {sorted(unmapped_genotypes)}"
        )

    return pd.DataFrame(rows)


def dry_run_report(manifest: pd.DataFrame) -> None:
    """Print a detailed summary without processing any flies."""
    total_flies = len(manifest)
    print(f"\n{'='*70}")
    print(f"DRY RUN REPORT")
    print(f"{'='*70}")
    print(f"Total fly directories found: {total_flies}")

    unmapped = manifest[manifest["brain_region"] == "UNMAPPED"]
    if not unmapped.empty:
        print(
            f"\n⚠️  UNMAPPED GENOTYPES ({len(unmapped)} flies, " f"{unmapped['genotype'].nunique()} unique genotypes):"
        )
        for geno, grp in unmapped.groupby("genotype"):
            print(f"    {geno:30s}  → {len(grp)} flies")

    print(f"\nFlies per brain region:")
    region_summary = (
        manifest.groupby("brain_region")
        .agg(n_flies=("fly_dir", "count"), n_genotypes=("genotype", "nunique"))
        .sort_values("brain_region")
    )
    print(region_summary.to_string())

    print(f"\nFlies per region/genotype:")
    geno_summary = (
        manifest.groupby(["brain_region", "genotype"])
        .agg(n_flies=("fly_dir", "count"))
        .sort_values(["brain_region", "genotype"])
    )
    print(geno_summary.to_string())

    print(f"\nExperiments processed: {manifest['experiment'].nunique()}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_region(
    region: str,
    region_manifest: pd.DataFrame,
    output_dir: Path,
    overwrite: bool = False,
) -> bool:
    """Process all flies in a brain region and save a single feather file.

    Returns True on success.
    """
    # Region names are already clean after REGION_REMAP; just guard against
    # any residual spaces or slashes that might appear for unmapped names.
    safe_name = region.replace(" ", "_").replace("/", "-")
    out_path = output_dir / f"{safe_name}_trajectories.feather"

    if out_path.exists() and not overwrite:
        logging.info(f"[{region}] Output already exists, skipping: {out_path}")
        return True

    # Lazy import to keep dry-run fast
    try:
        import sys as _sys

        _sys.path.append(str(Path(__file__).parent))
        import ballpushing_utils
    except ImportError as exc:
        logging.error(f"Cannot import ballpushing_utils: {exc}")
        return False

    fly_dirs = region_manifest["fly_dir"].tolist()
    logging.info(f"[{region}] Processing {len(fly_dirs)} flies…")

    chunks: list[pd.DataFrame] = []
    n_ok = 0
    n_fail = 0

    for fly_dir in fly_dirs:
        fly_dir = Path(fly_dir)
        fly_label = f"{fly_dir.parent.parent.name}/{fly_dir.parent.name}/{fly_dir.name}"
        try:
            fly = ballpushing_utils.Fly(fly_dir, as_individual=True)
            ds = ballpushing_utils.Dataset(fly, dataset_type=DATASET_TYPE)
            if ds.data is None or ds.data.empty:
                logging.warning(f"[{region}] No data for {fly_label}")
                n_fail += 1
                continue
            chunks.append(ds.data)
            n_ok += 1
        except Exception as exc:
            logging.warning(f"[{region}] Skipping {fly_label}: {exc}")
            n_fail += 1
            continue
        finally:
            # Free memory
            if "fly" in dir():
                try:
                    if hasattr(fly, "clear_caches"):
                        fly.clear_caches()
                    del fly
                except Exception:
                    pass

    if not chunks:
        logging.error(f"[{region}] No valid data found.")
        return False

    logging.info(f"[{region}] Concatenating {len(chunks)} DataFrames…")
    pooled = pd.concat(chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.reset_index(drop=True).to_feather(out_path)
    logging.info(f"[{region}] ✅ Saved {len(pooled)} rows ({n_ok} flies, {n_fail} skipped) → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one coordinates feather file per brain region from TNT screen experiments."
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=str(DEFAULT_YAML),
        help=f"Path to YAML file listing experiment directories (default: {DEFAULT_YAML})",
    )
    parser.add_argument(
        "--split-registry",
        type=str,
        default=str(DEFAULT_SPLIT_REGISTRY),
        help=f"CSV file mapping genotypes to brain regions (default: {DEFAULT_SPLIT_REGISTRY})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to write region feather files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest and region summary without processing any flies.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        metavar="REGION",
        help="Only process these brain regions (e.g. --regions MB CX). Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output feather files.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    yaml_path = Path(args.yaml)
    split_registry_path = Path(args.split_registry)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not yaml_path.exists():
        logging.error(f"YAML file not found: {yaml_path}")
        sys.exit(1)
    if not split_registry_path.exists():
        logging.error(f"Split registry not found: {split_registry_path}")
        sys.exit(1)

    # Build manifest
    logging.info(f"Loading experiment directories from {yaml_path}")
    exp_dirs = load_experiment_dirs(yaml_path)
    logging.info(f"Found {len(exp_dirs)} experiment directories in YAML")

    logging.info(f"Loading genotype→region map from {split_registry_path}")
    genotype_region_map = build_genotype_region_map(split_registry_path)
    logging.info(f"Region map has {len(genotype_region_map)} entries")

    logging.info("Building fly manifest…")
    manifest = build_fly_manifest(exp_dirs, genotype_region_map)

    if manifest.empty:
        logging.error("Manifest is empty – no flies found. Check your YAML and experiment paths.")
        sys.exit(1)

    logging.info(
        f"Manifest built: {len(manifest)} fly directories across " f"{manifest['brain_region'].nunique()} brain regions"
    )

    # Always show dry-run style summary
    dry_run_report(manifest)

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    # Filter to requested regions
    regions_to_process = sorted(manifest["brain_region"].unique())
    if args.regions:
        requested = set(args.regions)
        regions_to_process = [r for r in regions_to_process if r in requested]
        missing = requested - set(regions_to_process)
        if missing:
            logging.warning(f"Requested regions not found in manifest: {missing}")
        if not regions_to_process:
            logging.error("No matching regions to process.")
            sys.exit(1)

    # Skip UNMAPPED unless explicitly requested
    if "UNMAPPED" in regions_to_process and (args.regions is None or "UNMAPPED" not in (args.regions or [])):
        logging.warning("Skipping UNMAPPED flies. To process them, use --regions UNMAPPED.")
        regions_to_process = [r for r in regions_to_process if r != "UNMAPPED"]

    logging.info(f"Regions to process: {regions_to_process}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    for region in regions_to_process:
        region_manifest = manifest[manifest["brain_region"] == region]
        results[region] = process_region(
            region=region,
            region_manifest=region_manifest,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for region, ok in results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"  {status}  {region}")
    n_failed = sum(1 for ok in results.values() if not ok)
    if n_failed:
        print(f"\n{n_failed} region(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} region(s) completed successfully.")


if __name__ == "__main__":
    main()
