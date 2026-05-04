#!/usr/bin/env python3
"""
Prepare ball-scent raw H5 files for Dataverse upload.

Layout:
  <OUTPUT_ROOT>/<BallScentCondition>/<Date>/<Arena>/<Corridor>/<h5 files>

Where BallScentCondition uses prefixed canonical labels, e.g.:
  Ballscents-New
  Ballscents-New-plus-Pre-exposed
  Ballscents-Washed
  Ballscents-Washed-plus-Pre-exposed
  Ballscents-Pre-exposed
  Ballscents-Ctrl

Each experiment folder in ballscents.yaml contains:
  - metadata.json  (Arena# entries include BallScent and Date)
  - arena{1..9}/corridor{1..6}/*tracked*.h5 and *full_body.h5

Usage
-----
  # Dry-run for a single experiment (index 0 in YAML):
  python prepare_ballscents_h5.py --dry-run --experiment-index 0

  # Run all pre-flight checks without copying:
  python prepare_ballscents_h5.py --validate

  # Full copy (all experiments):
  python prepare_ballscents_h5.py --copy

  # Full copy + archive each condition folder to .tar:
  python prepare_ballscents_h5.py --copy --compress
"""

from __future__ import annotations

import argparse
import json
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

YAML_PATH = Path(__file__).resolve().parents[2] / "experiments_yaml/ballscents.yaml"
OUTPUT_ROOT = dataset("Affordance_article_dataverse/ball_scents/raw_h5")
N_ARENAS = 9
N_CORRIDORS = 6
MAX_ARCHIVE_SIZE_BYTES = 2 * 1024**3

# Canonical BallScent keys and output directory labels.
# Matches semantics used in the analysis script's label normalization.
BALLSCENT_DIRS: dict[str, str] = {
    "Ctrl": "Ballscents-Ctrl",
    "CtrlScent": "Ballscents-Pre-exposed",
    "Washed": "Ballscents-Washed",
    "Scented": "Ballscents-Washed-plus-Pre-exposed",
    "New": "Ballscents-New",
    "NewScent": "Ballscents-New-plus-Pre-exposed",
}

EXPECTED_CONDITIONS: set[str] = set(BALLSCENT_DIRS.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def canonical_ballscent(raw: str) -> str | None:
    """Map raw BallScent value to canonical key used for directory naming."""
    if raw is None:
        return None
    value = str(raw).strip()
    if value == "":
        return None

    # Direct canonical keys.
    if value in BALLSCENT_DIRS:
        return value

    # Common aliases (from processed datasets / plotting labels).
    alias_map = {
        "New + Pre-exposed": "NewScent",
        "Washed + Pre-exposed": "Scented",
        "Pre-exposed": "CtrlScent",
    }
    if value in alias_map:
        return alias_map[value]

    # Lightweight heuristic fallback.
    v = value.lower().replace("_", " ")
    if "ctrl" in v and "scent" in v:
        return "CtrlScent"
    if "ctrl" in v:
        return "Ctrl"
    if "new" in v and "scent" in v:
        return "NewScent"
    if "new" in v:
        return "New"
    if "wash" in v and "scent" in v:
        return "Scented"
    if "wash" in v:
        return "Washed"
    if "scent" in v:
        return "Scented"

    return None


def ballscent_dir(raw: str) -> str | None:
    """Return destination directory label for BallScent value, or None to skip."""
    canonical = canonical_ballscent(raw)
    if canonical is None:
        return None
    return BALLSCENT_DIRS[canonical]


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
    """Return {(condition, date, arena_idx): [exp_dirs]} for collision detection."""
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
                ballscent_raw = metadata_get(meta, arena_key, "BallScent")
                date = metadata_get(meta, arena_key, "Date")
            except (KeyError, IndexError):
                continue
            condition = ballscent_dir(ballscent_raw)
            if condition is None or not date or date == "None":
                continue
            slot_exps[(condition, date, arena_idx)].append(exp_dir)
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

        ballscent_raw = metadata_get(metadata, arena_key, "BallScent")
        date = metadata_get(metadata, arena_key, "Date")
        condition = ballscent_dir(ballscent_raw)
        if condition is None or not date or date == "None":
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
                    / condition
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

    Handles same-day same-condition same-arena collisions by appending a numeric
    suffix to date (e.g. 251017-1, 251017-2).
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
    for (condition, date, arena_idx), exps in slot_exps.items():
        slot = (condition, date, arena_idx)
        if len(exps) == 1:
            date_labels[(slot, exps[0])] = date
        else:
            for i, exp_dir in enumerate(exps, 1):
                date_labels[(slot, exp_dir)] = f"{date}-{i}"
                print(
                    f"  [INFO] Date collision: {condition}/Arena{arena_idx} on {date} "
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
                ballscent_raw = metadata_get(metadata, arena_key, "BallScent")
                date = metadata_get(metadata, arena_key, "Date")
            except (KeyError, IndexError):
                continue
            condition = ballscent_dir(ballscent_raw)
            if condition is None or not date or date == "None":
                continue
            slot = (condition, date, arena_idx)
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
                        / condition
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

    print("\n[Test 1] Checking BallScent values and metadata availability...")
    missing_exps: list[Path] = []
    unknown_values: set[str] = set()
    conditions_seen: set[str] = set()

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
                ballscent_raw = metadata_get(meta, arena_key, "BallScent")
            except KeyError:
                msg = f"  [ERROR] 'BallScent' missing in metadata for {exp_dir}"
                print(msg)
                issues.append(msg)
                break

            cond = ballscent_dir(ballscent_raw)
            if cond is None:
                unknown_values.add(ballscent_raw)
            else:
                conditions_seen.add(cond)

    if missing_exps:
        for p in missing_exps:
            msg = f"  [ERROR] Experiment directory missing: {p}"
            print(msg)
            issues.append(msg)

    if unknown_values:
        for v in sorted(unknown_values):
            print(f"  [WARN] Unknown BallScent value: {v!r} - arena will be skipped")
    else:
        print("  OK - all BallScent values are recognized")

    missing_conditions = EXPECTED_CONDITIONS - conditions_seen
    if missing_conditions:
        for c in sorted(missing_conditions):
            print(f"  [WARN] Condition not present in this dataset: {c}")

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

    print("\n[Info] Files per BallScent condition:")
    cond_counts: dict[str, int] = defaultdict(int)
    for _, dst in all_pairs:
        cond_name = dst.relative_to(output_root).parts[0]
        cond_counts[cond_name] += 1
    for cond_name, count in sorted(cond_counts.items()):
        print(f"  {cond_name}: {count} files")

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

    print("\nBallScent per arena:")
    for arena_idx in range(1, N_ARENAS + 1):
        arena_key = f"Arena{arena_idx}"
        if arena_key not in metadata:
            continue
        try:
            ballscent_raw = metadata_get(metadata, arena_key, "BallScent")
            date = metadata_get(metadata, arena_key, "Date")
        except KeyError:
            continue
        mapped = ballscent_dir(ballscent_raw)
        label = mapped if mapped else "[SKIPPED]"
        print(
            f"  Arena{arena_idx}: BallScent={ballscent_raw!r} -> {label} (date={date})"
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


def compress_conditions(output_root: Path) -> None:
    """Archive each BallScent condition folder, splitting archives above 2 GB."""
    condition_dirs = sorted(
        [d for d in output_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    print(f"\nArchiving {len(condition_dirs)} condition folders...")
    for cdir in condition_dirs:
        archive_directory_with_split(cdir, output_root, MAX_ARCHIVE_SIZE_BYTES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ball-scent H5 files for Dataverse upload",
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
        help="After copying, archive each condition folder to .tar (used with --copy)",
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
        compress_conditions(args.output_root)


if __name__ == "__main__":
    main()
