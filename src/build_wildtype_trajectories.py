#!/usr/bin/env python3
"""
Build coordinates datasets grouped by light condition and feeding state
for wild-type animals recorded in control experiments.

For each (Light, FeedingState) combination found across all experiment
directories, all fly corridors are processed and their coordinate data is
concatenated into a single feather file:
    <output_dir>/Wild-type_<Light>_<FeedingState>_trajectories.feather

e.g.:
    Wild-type_Lights-on_Starved_trajectories.feather
    Wild-type_Lights-on_Fed_trajectories.feather
    Wild-type_Lights-on_Starved-without-water_trajectories.feather
    Wild-type_Lights-off_Starved_trajectories.feather
    Wild-type_Lights-off_Fed_trajectories.feather
    Wild-type_Lights-off_Starved-without-water_trajectories.feather

Usage
-----
Dry run (no files written, just prints the fly manifest):
    python build_wildtype_trajectories.py --dry-run

Full run:
    python build_wildtype_trajectories.py --output-dir /path/to/output

Full run with custom YAML:
    python build_wildtype_trajectories.py \\
        --yaml experiments_yaml/control_folders.yaml \\
        --output-dir /mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/wildtype_trajectories \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Default paths – override with CLI flags
# ---------------------------------------------------------------------------
DEFAULT_YAML = Path(__file__).parent.parent / "experiments_yaml" / "control_folders.yaml"
DEFAULT_OUTPUT_DIR = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/wildtype_trajectories")
DATASET_TYPE = "coordinates"

N_ARENAS = 9
N_CORRIDORS = 6
CONDITION_PREFIX = "Wild-type"

# Human-readable labels for raw metadata values.
# None means skip the arena.
LIGHT_LABELS: dict[str, str | None] = {
    "on": "Lights-on",
    "off": "Lights-off",
    "": None,
}

FEEDING_LABELS: dict[str, str | None] = {
    "fed": "Fed",
    "Fed": "Fed",
    "starved": "Starved",
    "starved_noWater": "Starved-without-water",
    "": None,
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


def condition_label(light_raw: str, feeding_raw: str) -> str | None:
    """Return the condition directory name, or None to skip this arena."""
    light = LIGHT_LABELS.get(light_raw)
    feeding = FEEDING_LABELS.get(feeding_raw)
    if light is None or feeding is None:
        return None
    return f"{CONDITION_PREFIX}_{light}_{feeding}"


def parse_metadata(metadata_path: Path) -> dict:
    """Load metadata JSON from a path (handles both dict-of-lists and list-of-dicts)."""
    with open(metadata_path) as f:
        return json.load(f)


def _metadata_get(meta: dict, arena_key: str, variable: str) -> str:
    """Extract a variable value for a given arena from the metadata dict."""
    variables: list = meta["Variable"]
    values: list = meta[arena_key]
    idx = [v.lower() for v in variables].index(variable.lower())
    raw = values[idx]
    if raw is None:
        return ""
    return str(raw).strip()


def build_fly_manifest(experiment_dirs: list[Path]) -> pd.DataFrame:
    """
    Walk all experiment dirs and build a table with columns:
        fly_dir, experiment, arena, corridor, light, feeding_state, condition
    """
    rows = []
    unmapped_light: set[str] = set()
    unmapped_feeding: set[str] = set()

    for exp_dir in experiment_dirs:
        if not exp_dir.exists():
            logging.warning(f"Experiment dir not found, skipping: {exp_dir}")
            continue

        # Support both 'Metadata.json' and 'metadata.json'
        meta_path = exp_dir / "Metadata.json"
        if not meta_path.exists():
            meta_path = exp_dir / "metadata.json"
        if not meta_path.exists():
            logging.warning(f"No metadata file in {exp_dir}")
            continue

        try:
            meta = parse_metadata(meta_path)
        except Exception as exc:
            logging.warning(f"Could not parse metadata in {exp_dir}: {exc}")
            continue

        if "Variable" not in meta:
            logging.warning(f"No 'Variable' key in metadata of {exp_dir}")
            continue

        for arena_idx in range(1, N_ARENAS + 1):
            arena_key = f"Arena{arena_idx}"
            if arena_key not in meta:
                continue

            try:
                light_raw = _metadata_get(meta, arena_key, "Light")
                feeding_raw = _metadata_get(meta, arena_key, "FeedingState")
            except (ValueError, IndexError, KeyError):
                logging.debug(f"Missing Light/FeedingState for {arena_key} in {exp_dir.name}")
                continue

            # Skip arenas where genotype or other critical value is None/empty
            try:
                genotype_raw = _metadata_get(meta, arena_key, "Genotype")
                if genotype_raw.lower() in ("none", "null", ""):
                    logging.debug(f"Empty genotype for {arena_key} in {exp_dir.name}, skipping")
                    continue
            except (ValueError, IndexError, KeyError):
                pass

            if light_raw not in LIGHT_LABELS:
                unmapped_light.add(light_raw)
            if feeding_raw not in FEEDING_LABELS:
                unmapped_feeding.add(feeding_raw)

            cond = condition_label(light_raw, feeding_raw)
            if cond is None:
                logging.debug(
                    f"Skipping {arena_key} in {exp_dir.name}: " f"Light={light_raw!r}, FeedingState={feeding_raw!r}"
                )
                continue

            for corridor_idx in range(1, N_CORRIDORS + 1):
                corridor_dir = exp_dir / f"arena{arena_idx}" / f"corridor{corridor_idx}"
                if not corridor_dir.is_dir():
                    continue

                rows.append(
                    {
                        "fly_dir": corridor_dir,
                        "experiment": exp_dir.name,
                        "arena": f"Arena{arena_idx}",
                        "corridor": f"Corridor{corridor_idx}",
                        "light": light_raw,
                        "feeding_state": feeding_raw,
                        "condition": cond,
                    }
                )

    if unmapped_light:
        logging.warning(f"Unknown Light value(s) encountered (arenas skipped): {sorted(unmapped_light)}")
    if unmapped_feeding:
        logging.warning(f"Unknown FeedingState value(s) encountered (arenas skipped): {sorted(unmapped_feeding)}")

    return pd.DataFrame(rows)


def dry_run_report(manifest: pd.DataFrame) -> None:
    """Print a detailed summary without processing any flies."""
    print(f"\n{'='*70}")
    print("DRY RUN REPORT")
    print(f"{'='*70}")
    print(f"Total fly directories found: {len(manifest)}")

    print(f"\nFlies per condition:")
    cond_summary = (
        manifest.groupby("condition")
        .agg(
            n_flies=("fly_dir", "count"),
            n_experiments=("experiment", "nunique"),
        )
        .sort_values("condition")
    )
    print(cond_summary.to_string())

    print(f"\nFlies per condition / experiment:")
    exp_summary = (
        manifest.groupby(["condition", "experiment"])
        .agg(n_flies=("fly_dir", "count"))
        .sort_values(["condition", "experiment"])
    )
    print(exp_summary.to_string())

    print(f"\nExperiments processed: {manifest['experiment'].nunique()}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_condition(
    condition: str,
    condition_manifest: pd.DataFrame,
    output_dir: Path,
    overwrite: bool = False,
) -> bool:
    """Process all flies in a condition and save a single feather file.

    Returns True on success.
    """
    safe_name = condition.replace(" ", "_").replace("/", "-")
    out_path = output_dir / f"{safe_name}_trajectories.feather"

    if out_path.exists() and not overwrite:
        logging.info(f"[{condition}] Output already exists, skipping: {out_path}")
        return True

    try:
        import sys as _sys

        _sys.path.append(str(Path(__file__).parent))
        import Ballpushing_utils
    except ImportError as exc:
        logging.error(f"Cannot import Ballpushing_utils: {exc}")
        return False

    fly_dirs = condition_manifest["fly_dir"].tolist()
    logging.info(f"[{condition}] Processing {len(fly_dirs)} flies…")

    chunks: list[pd.DataFrame] = []
    n_ok = 0
    n_fail = 0

    for fly_dir in fly_dirs:
        fly_dir = Path(fly_dir)
        fly_label = f"{fly_dir.parent.parent.name}/{fly_dir.parent.name}/{fly_dir.name}"
        try:
            fly = Ballpushing_utils.Fly(fly_dir, as_individual=True)
            ds = Ballpushing_utils.Dataset(fly, dataset_type=DATASET_TYPE)
            if ds.data is None or ds.data.empty:
                logging.warning(f"[{condition}] No data for {fly_label}")
                n_fail += 1
                continue
            chunks.append(ds.data)
            n_ok += 1
        except Exception as exc:
            logging.warning(f"[{condition}] Skipping {fly_label}: {exc}")
            n_fail += 1
            continue
        finally:
            if "fly" in dir():
                try:
                    if hasattr(fly, "clear_caches"):
                        fly.clear_caches()
                    del fly
                except Exception:
                    pass

    if not chunks:
        logging.error(f"[{condition}] No valid data found.")
        return False

    logging.info(f"[{condition}] Concatenating {len(chunks)} DataFrames…")
    pooled = pd.concat(chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.reset_index(drop=True).to_feather(out_path)
    logging.info(f"[{condition}] Saved {len(pooled)} rows " f"({n_ok} flies, {n_fail} skipped) → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one coordinates feather file per light/feeding-state condition "
            "from wild-type control experiments."
        )
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=str(DEFAULT_YAML),
        help=f"Path to YAML file listing experiment directories (default: {DEFAULT_YAML})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to write condition feather files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest and condition summary without processing any flies.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        metavar="CONDITION",
        help=("Only process these conditions (e.g. --conditions Wild-type_Lights-on_Starved). " "Default: all."),
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
    output_dir = Path(args.output_dir)

    if not yaml_path.exists():
        logging.error(f"YAML file not found: {yaml_path}")
        sys.exit(1)

    logging.info(f"Loading experiment directories from {yaml_path}")
    exp_dirs = load_experiment_dirs(yaml_path)
    logging.info(f"Found {len(exp_dirs)} experiment directories in YAML")

    logging.info("Building fly manifest…")
    manifest = build_fly_manifest(exp_dirs)

    if manifest.empty:
        logging.error("Manifest is empty – no flies found. Check your YAML and experiment paths.")
        sys.exit(1)

    logging.info(
        f"Manifest built: {len(manifest)} fly directories across " f"{manifest['condition'].nunique()} conditions"
    )

    dry_run_report(manifest)

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    conditions_to_process = sorted(manifest["condition"].unique())
    if args.conditions:
        requested = set(args.conditions)
        conditions_to_process = [c for c in conditions_to_process if c in requested]
        missing = requested - set(conditions_to_process)
        if missing:
            logging.warning(f"Requested conditions not found in manifest: {missing}")
        if not conditions_to_process:
            logging.error("No matching conditions to process.")
            sys.exit(1)

    logging.info(f"Conditions to process: {conditions_to_process}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    for condition in conditions_to_process:
        condition_manifest = manifest[manifest["condition"] == condition]
        results[condition] = process_condition(
            condition=condition,
            condition_manifest=condition_manifest,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for condition, ok in results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        print(f"  {status}  {condition}")
    n_failed = sum(1 for ok in results.values() if not ok)
    if n_failed:
        print(f"\n{n_failed} condition(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} condition(s) completed successfully.")


if __name__ == "__main__":
    main()
