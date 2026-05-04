#!/usr/bin/env python3
"""
Prepare TNT olfaction (dark/light) raw H5 files for Dataverse upload.

Layout:
  <OUTPUT_ROOT>/<Genotype>-Light-<on|off>/<Date>/<Arena>/<Corridor>/<h5 files>

Example:
  TNTxIR8a-Light-on/251104/Arena1/Corridor1/...

Each experiment folder in TNT_Olfaction_Dark.yaml contains:
  - metadata.json  (Arena# entries include Genotype, Light, Date)
  - arena{1..9}/corridor{1..6}/*tracked*.h5 and optional *full_body.h5

Usage
-----
  # Dry-run for a single experiment (index 0 in YAML):
  python prepare_tnt_olfaction_dark_h5.py --dry-run --experiment-index 0

  # Run all pre-flight checks without copying:
  python prepare_tnt_olfaction_dark_h5.py --validate

  # Full copy (all experiments):
  python prepare_tnt_olfaction_dark_h5.py --copy

  # Full copy + archive each genotype-light folder to .tar:
  python prepare_tnt_olfaction_dark_h5.py --copy --compress
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from ballpushing_utils.paths import dataset
from ballpushing_utils.archive_split import archive_directory_with_split
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YAML_PATH = Path(__file__).resolve().parents[2] / "experiments_yaml/TNT_Olfaction_Dark.yaml"
OUTPUT_ROOT = dataset("Affordance_article_dataverse/TNT_olfaction_dark/raw_h5")
N_ARENAS = 9
N_CORRIDORS = 6

LIGHT_LABELS: dict[str, str | None] = {
    "on": "on",
    "off": "off",
    "": None,
}

MAX_ARCHIVE_SIZE_BYTES = 2 * 1024**3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_dirname(name: str) -> str:
    """Convert arbitrary text to a safe directory name."""
    text = str(name).strip()
    text = re.sub(r"[\s/\\]+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def load_experiments(yaml_path: Path) -> list[Path]:
    """Return list of experiment directory Paths from the YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [Path(d) for d in data["directories"]]


def load_metadata(exp_dir: Path) -> dict:
    """Load metadata.json from an experiment directory."""
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {exp_dir}")
    with open(meta_path) as f:
        return json.load(f)


def metadata_get(metadata: dict, arena_key: str, variable: str) -> str:
    """Extract a variable value for a given arena key from metadata dict."""
    variables: list = metadata["Variable"]
    values: list = metadata[arena_key]
    try:
        idx = variables.index(variable)
    except ValueError:
        raise KeyError(
            f"Variable '{variable}' not found in metadata for arena '{arena_key}' "
            f"- variables present: {variables}"
        )
    return str(values[idx]).strip()


def genotype_light_dir(genotype_raw: str, light_raw: str) -> str | None:
    """Return <Genotype>-Light-<on|off> directory name, or None to skip."""
    genotype = sanitize_dirname(genotype_raw)
    if not genotype:
        return None
    light_label = LIGHT_LABELS.get(light_raw)
    if light_label is None:
        return None
    return f"{genotype}-Light-{light_label}"


def find_h5_files(corridor_dir: Path) -> list[Path]:
    """Return H5 files for Dataverse export."""
    return sorted(
        f
        for f in corridor_dir.glob("*.h5")
        if (
            "_tracked_fly." in f.name
            or "_tracked_ball." in f.name
            or "_preprocessed_full_body" in f.name
        )
    )


def _collect_slots(
    experiments: list[Path], metadata_cache: dict[Path, dict] | None = None
) -> dict[tuple[str, str, int], list[Path]]:
    """Return {(group, date, arena_idx): [exp_dirs]} for collision detection."""
    slot_exps: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for exp_dir in sorted(experiments):
        if not exp_dir.exists():
            continue
        if metadata_cache is not None:
            meta = metadata_cache.get(exp_dir)
            if meta is None:
                continue
        else:
            try:
                meta = load_metadata(exp_dir)
            except (FileNotFoundError, KeyError):
                continue

        for arena_idx in range(1, N_ARENAS + 1):
            arena_key = f"Arena{arena_idx}"
            if arena_key not in meta:
                continue
            try:
                genotype_raw = metadata_get(meta, arena_key, "Genotype")
                light_raw = metadata_get(meta, arena_key, "Light")
                date = metadata_get(meta, arena_key, "Date")
            except (KeyError, IndexError):
                continue
            group = genotype_light_dir(genotype_raw, light_raw)
            if group is None or not date or date == "None":
                continue
            slot_exps[(group, date, arena_idx)].append(exp_dir)
    return slot_exps


def plan_experiment(exp_dir: Path, output_root: Path) -> list[tuple[Path, Path]]:
    """Build a list of (src, dst) copy pairs for one experiment."""
    if not exp_dir.exists():
        return []

    metadata = load_metadata(exp_dir)
    pairs: list[tuple[Path, Path]] = []

    for arena_idx in range(1, N_ARENAS + 1):
        arena_key = f"Arena{arena_idx}"
        if arena_key not in metadata:
            continue

        genotype_raw = metadata_get(metadata, arena_key, "Genotype")
        light_raw = metadata_get(metadata, arena_key, "Light")
        date = metadata_get(metadata, arena_key, "Date")
        group = genotype_light_dir(genotype_raw, light_raw)
        if group is None or not date or date == "None":
            continue

        for corridor_idx in range(1, N_CORRIDORS + 1):
            corridor_dir = exp_dir / f"arena{arena_idx}" / f"corridor{corridor_idx}"
            if not corridor_dir.exists():
                continue
            if next(corridor_dir.glob("*_tracked_fly.*.h5"), None) is None:
                continue

            for src in find_h5_files(corridor_dir):
                dst_dir = (
                    output_root
                    / group
                    / date
                    / f"Arena{arena_idx}"
                    / f"Corridor{corridor_idx}"
                )
                pairs.append((src, dst_dir / src.name))

    return pairs


def plan_all_experiments(
    experiments: list[Path], output_root: Path
) -> list[tuple[Path, Path]]:
    """Build all (src, dst) copy pairs across every experiment.

    Handles same-day same-group same-arena collisions by appending numeric suffix
    to date (e.g. 251030-1, 251030-2).
    """
    experiments_sorted = sorted(experiments)

    metadata_cache: dict[Path, dict] = {}
    for exp_dir in experiments_sorted:
        if not exp_dir.exists():
            continue
        try:
            metadata_cache[exp_dir] = load_metadata(exp_dir)
        except (FileNotFoundError, KeyError):
            continue

    slot_exps = _collect_slots(experiments, metadata_cache=metadata_cache)

    date_labels: dict[tuple[tuple[str, str, int], Path], str] = {}
    for (group, date, arena_idx), exps in slot_exps.items():
        slot = (group, date, arena_idx)
        if len(exps) == 1:
            date_labels[(slot, exps[0])] = date
        else:
            for i, exp_dir in enumerate(exps, 1):
                date_labels[(slot, exp_dir)] = f"{date}-{i}"
                print(
                    f"  [INFO] Date collision: {group}/Arena{arena_idx} on {date} "
                    f"in {len(exps)} experiments "
                    f"-> assigning '{date}-{i}' to {exp_dir.name}"
                )

    all_pairs: list[tuple[Path, Path]] = []
    for exp_dir, metadata in metadata_cache.items():
        for arena_idx in range(1, N_ARENAS + 1):
            arena_key = f"Arena{arena_idx}"
            if arena_key not in metadata:
                continue
            try:
                genotype_raw = metadata_get(metadata, arena_key, "Genotype")
                light_raw = metadata_get(metadata, arena_key, "Light")
                date = metadata_get(metadata, arena_key, "Date")
            except (KeyError, IndexError):
                continue
            group = genotype_light_dir(genotype_raw, light_raw)
            if group is None or not date or date == "None":
                continue
            slot = (group, date, arena_idx)
            resolved_date = date_labels.get((slot, exp_dir), date)

            for corridor_idx in range(1, N_CORRIDORS + 1):
                corridor_dir = exp_dir / f"arena{arena_idx}" / f"corridor{corridor_idx}"
                if not corridor_dir.exists():
                    continue
                if next(corridor_dir.glob("*_tracked_fly.*.h5"), None) is None:
                    continue
                for src in find_h5_files(corridor_dir):
                    dst_dir = (
                        output_root
                        / group
                        / resolved_date
                        / f"Arena{arena_idx}"
                        / f"Corridor{corridor_idx}"
                    )
                    all_pairs.append((src, dst_dir / src.name))

    return all_pairs


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def validate(
    experiments: list[Path],
    output_root: Path,
    all_pairs: list[tuple[Path, Path]] | None = None,
) -> bool:
    """Run all pre-flight checks. Returns True if everything looks clean."""
    issues: list[str] = []

    print("\n[Test 1] Checking Genotype/Light/Date metadata values...")
    missing_exps: list[Path] = []
    unknown_light: set[str] = set()
    groups_seen: set[str] = set()

    for exp_dir in experiments:
        if not exp_dir.exists():
            missing_exps.append(exp_dir)
            continue
        try:
            meta = load_metadata(exp_dir)
        except FileNotFoundError as e:
            msg = f"  [ERROR] {e}"
            print(msg)
            issues.append(msg)
            continue

        for arena_idx in range(1, N_ARENAS + 1):
            arena_key = f"Arena{arena_idx}"
            if arena_key not in meta:
                continue
            try:
                genotype_raw = metadata_get(meta, arena_key, "Genotype")
                light_raw = metadata_get(meta, arena_key, "Light")
                _ = metadata_get(meta, arena_key, "Date")
            except KeyError as e:
                msg = f"  [ERROR] {exp_dir}: {e}"
                print(msg)
                issues.append(msg)
                break

            if light_raw not in LIGHT_LABELS:
                unknown_light.add(light_raw)

            group = genotype_light_dir(genotype_raw, light_raw)
            if group:
                groups_seen.add(group)

    if missing_exps:
        for p in missing_exps:
            msg = f"  [ERROR] Experiment directory missing: {p}"
            print(msg)
            issues.append(msg)

    if unknown_light:
        for l in sorted(unknown_light):
            print(f"  [WARN] Unknown Light value: {l!r} - arena will be skipped")
    else:
        print("  OK - all Light values are recognized")

    print(f"  INFO - groups found: {sorted(groups_seen)}")

    print("\n[Test 2] Checking for destination path collisions...")
    if all_pairs is None:
        all_pairs = plan_all_experiments(experiments, output_root)
    dst_to_srcs: dict[Path, list[Path]] = defaultdict(list)
    for src, dst in all_pairs:
        dst_to_srcs[dst].append(src)
    dup_dsts = {dst: srcs for dst, srcs in dst_to_srcs.items() if len(srcs) > 1}
    if dup_dsts:
        for dst, srcs in sorted(dup_dsts.items()):
            msg = f"  [ERROR] {len(srcs)} sources -> {dst}\n    srcs: {srcs}"
            print(msg)
            issues.append(msg)
    else:
        print("  OK - no destination path collisions")

    print(
        "\n[Test 3] Checking H5 file counts per corridor with fly tracking (expected 2 or 3)..."
    )
    unexpected: list[str] = []
    skipped_no_fly = 0
    for exp_dir in experiments:
        if not exp_dir.exists():
            continue
        for arena_idx in range(1, N_ARENAS + 1):
            for corridor_idx in range(1, N_CORRIDORS + 1):
                corridor_dir = exp_dir / f"arena{arena_idx}" / f"corridor{corridor_idx}"
                if not corridor_dir.exists():
                    continue
                if next(corridor_dir.glob("*_tracked_fly.*.h5"), None) is None:
                    skipped_no_fly += 1
                    continue
                n = len(find_h5_files(corridor_dir))
                if n not in (2, 3):
                    msg = f"  [WARN] {corridor_dir}: {n} H5 files"
                    print(msg)
                    unexpected.append(msg)
    if skipped_no_fly:
        print(f"  INFO - {skipped_no_fly} corridors skipped (no fly tracking file)")
    if not unexpected:
        print("  OK - all corridors with fly tracking have 2 or 3 H5 files")

    print("\n[Test 4] Checking total planned file count...")
    total = len(all_pairs)
    print(f"  Total H5 files planned for copy: {total}")
    if total == 0:
        msg = "  [ERROR] No files planned - check YAML and metadata"
        print(msg)
        issues.append(msg)
    else:
        print("  OK")

    print("\n[Info] Files per genotype-light group:")
    group_counts: dict[str, int] = defaultdict(int)
    for _, dst in all_pairs:
        group_name = dst.relative_to(output_root).parts[0]
        group_counts[group_name] += 1
    for group_name, count in sorted(group_counts.items()):
        print(f"  {group_name}: {count} files")

    print(f"\n{'='*60}")
    if issues:
        print(f"Pre-flight FAILED - {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("Pre-flight PASSED - safe to run --copy")
    return True


def dry_run(exp_dir: Path, experiments: list[Path], output_root: Path) -> None:
    """Print a summary for a single experiment without touching filesystem."""
    print(f"\n{'='*70}")
    print(f"DRY RUN  -  {exp_dir.name}")
    print(f"{'='*70}")

    if not exp_dir.exists():
        print(f"  [ERROR] Directory does not exist: {exp_dir}")
        return

    metadata = load_metadata(exp_dir)
    slot_exps = _collect_slots(experiments)
    exp_has_collision = any(
        exp_dir in exps and len(exps) > 1 for exps in slot_exps.values()
    )
    if exp_has_collision:
        all_global = plan_all_experiments(experiments, output_root)
        pairs = [(src, dst) for src, dst in all_global if src.is_relative_to(exp_dir)]
    else:
        pairs = plan_experiment(exp_dir, output_root)

    print("\nGenotype/Light per arena:")
    for arena_idx in range(1, N_ARENAS + 1):
        arena_key = f"Arena{arena_idx}"
        if arena_key not in metadata:
            continue
        try:
            genotype_raw = metadata_get(metadata, arena_key, "Genotype")
            light_raw = metadata_get(metadata, arena_key, "Light")
            date = metadata_get(metadata, arena_key, "Date")
        except KeyError:
            continue
        group = genotype_light_dir(genotype_raw, light_raw)
        label = group if group else "[SKIPPED]"
        print(
            f"  Arena{arena_idx}: Genotype={genotype_raw!r}, Light={light_raw!r} "
            f"-> {label} (date={date})"
        )

    print(f"\nH5 files to copy ({len(pairs)}):")
    for src, dst in pairs[:20]:
        print(f"  {src}")
        print(f"    -> {dst.relative_to(output_root)}")
    if len(pairs) > 20:
        print(f"  ... and {len(pairs) - 20} more")


def _copy_one(src: Path, dst: Path) -> str:
    """Copy one file atomically. Returns 'copied', 'skipped', or 'warn'."""
    if dst.exists():
        return "warn" if dst.stat().st_size != src.stat().st_size else "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    try:
        shutil.copy2(src, tmp)
        tmp.rename(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return "copied"


def copy_all_experiments(
    experiments: list[Path],
    output_root: Path,
    workers: int = 8,
    all_pairs: list[tuple[Path, Path]] | None = None,
) -> int:
    """Copy all H5 files in parallel. Returns number of files copied."""
    if all_pairs is None:
        all_pairs = plan_all_experiments(experiments, output_root)
    total = len(all_pairs)
    copied = skipped = warns = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_copy_one, src, dst): dst for src, dst in all_pairs}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result == "copied":
                copied += 1
            elif result == "skipped":
                skipped += 1
            elif result == "warn":
                print(f"  [WARN] Size mismatch on existing file: {futures[future]}")
                warns += 1
            if i % 200 == 0 or i == total:
                print(
                    f"  Progress: {i}/{total} "
                    f"({copied} copied, {skipped} skipped, {warns} warnings)..."
                )
    print(
        f"\n  Done: {copied} copied, {skipped} already present, "
        f"{warns} size warnings (total planned: {total})"
    )
    return copied


def compress_groups(output_root: Path) -> None:
    """Archive each top-level genotype-light folder into one or more .tar files."""
    group_dirs = sorted(
        [d for d in output_root.iterdir() if d.is_dir()], key=lambda p: p.name
    )
    print(f"\nArchiving {len(group_dirs)} genotype-light folders...")
    for gdir in group_dirs:
        archive_directory_with_split(
            source_dir=gdir,
            output_root=output_root,
            max_archive_size_bytes=MAX_ARCHIVE_SIZE_BYTES,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TNT olfaction dark H5 files for Dataverse upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied for one experiment (use --experiment-index)",
    )
    mode.add_argument(
        "--copy",
        action="store_true",
        help="Copy H5 files for all experiments",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run all pre-flight checks without copying anything",
    )

    parser.add_argument(
        "--experiment-index",
        type=int,
        default=0,
        help="Index (0-based) of experiment for --dry-run (default: 0)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="After copying, archive each genotype-light folder to .tar (used with --copy)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel copy workers (default: 8, used with --copy)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip pre-flight validation and copy anyway (use with caution)",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=YAML_PATH,
        help=f"Path to experiments YAML (default: {YAML_PATH})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Root output directory (default: {OUTPUT_ROOT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading experiments from {args.yaml} ...")
    experiments = load_experiments(args.yaml)
    print(f"  {len(experiments)} experiments found")

    if args.dry_run:
        exp = experiments[args.experiment_index]
        dry_run(exp, experiments, args.output_root)
        return

    if args.validate:
        validate(experiments, args.output_root)
        return

    all_pairs = plan_all_experiments(experiments, args.output_root)
    if not args.force:
        print("Running pre-flight validation before copy...")
        if not validate(experiments, args.output_root, all_pairs=all_pairs):
            print(
                "\nAborting --copy due to pre-flight errors. "
                "Fix the issues above or use --force to proceed anyway."
            )
            return

    args.output_root.mkdir(parents=True, exist_ok=True)
    orphans = list(args.output_root.rglob("*.tmp"))
    if orphans:
        print(
            f"  [INFO] Cleaning {len(orphans)} orphaned .tmp files from previous run..."
        )
        for f in orphans:
            f.unlink(missing_ok=True)

    copy_all_experiments(
        experiments,
        args.output_root,
        workers=args.workers,
        all_pairs=all_pairs,
    )

    if args.compress:
        compress_groups(args.output_root)


if __name__ == "__main__":
    main()
