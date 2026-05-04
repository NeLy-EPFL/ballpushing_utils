#!/usr/bin/env python3
"""
Prepare silencing screen raw H5 files for Dataverse upload.

Layout:
  <OUTPUT_ROOT>/<Nickname>/<Date>/<Arena>/<Corridor>/<h5 files>

Each experiment folder in TNT_screen.yaml contains:
  - metadata.json  (maps Arena# -> [Date, Genotype, ...])
  - arena{1..9}/corridor{1..6}/*tracked*.h5  and *full_body.h5

Genotype codes are resolved to directory-safe nicknames via Region_map CSV.

Usage
-----
  # Dry-run for a single experiment (index 0 in YAML):
  python prepare_screen_h5.py --dry-run --experiment-index 0

  # Full copy (all experiments):
  python prepare_screen_h5.py --copy

  # Full copy + archive each genotype folder to .tar:
  python prepare_screen_h5.py --copy --compress
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from ballpushing_utils.paths import dataset
from ballpushing_utils.archive_split import archive_directory_with_split
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YAML_PATH = Path(__file__).resolve().parents[2] / "experiments_yaml/TNT_screen.yaml"
MAPPING_CSV_PATH = dataset("Region_map_250908.csv")
OUTPUT_ROOT = dataset("Affordance_article_dataverse/silencing_screen/raw_h5")
# Patterns of H5 files to copy (matched against filename, not full path)
H5_PATTERNS = [
    "*_tracked_fly.*.h5",
    "*_tracked_ball.*.h5",
    "*_preprocessed_full_body.h5",
]

N_ARENAS = 9
N_CORRIDORS = 6
MAX_ARCHIVE_SIZE_BYTES = 2 * 1024**3

# Manual overrides for genotypes whose Simplified Nickname is ambiguous after
# sanitization (e.g. two distinct GAL4 drivers targeting the same region).
# These take precedence over the CSV mapping.
GENOTYPE_OVERRIDES: dict[str, str] = {
    "TNTxZ1979": "ER3m_EB_R28D01",
    "TNTxZ1990": "ER3m_EB_R28E01",
    # MBON-11-GaL4 targets MBON-γ1pedc>α/β; the CSV Simplified Nickname
    # ("MBON 11") loses this information, so we use the full descriptive name.
    "TNTxG80": "MBON-gamma1pedc_to_alpha_beta",
    # Only one DDC line in the screen, so the '-2' suffix is unnecessary.
    "TNTxDDC": "DDC",
    # Control lines crossed with TNT: named by the background only.
    "TNTxM6": "CS",
    "TNTxM7": "PR",
    # Plain wild-type backgrounds (not crossed with TNT).
    "M6": "Wild-type_CS",
    "M7": "Wild-type_PR",
    "PR": "Wild-type_PR",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_dirname(name: str) -> str:
    """Convert an arbitrary string into a safe directory name.

    Greek letters and the prime symbol are spelled out in full before
    other characters are handled, so that e.g. 'MBON-α′1' becomes
    'MBON-alpha_prime1' rather than losing information.

    Opening parentheses are replaced with '_' (so 'Wild-type(PR)'
    becomes 'Wild-type_PR') and closing parentheses are dropped.
    Remaining whitespace/slashes become '_'; all other non-alphanumeric
    characters (except '-' and '_') are removed.
    """
    GREEK = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "τ": "tau",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
        # prime / apostrophe variants
        "\u2032": "_prime",  # ′
        "\u2019": "_prime",  # '  (right single quotation mark, sometimes used)
    }
    name = str(name).strip()
    for char, replacement in GREEK.items():
        name = name.replace(char, replacement)
    name = re.sub(r"\(", "_", name)  # ( → _
    name = re.sub(r"\)", "", name)  # ) → removed
    name = re.sub(r"[\s/\\]+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def load_mapping(csv_path: Path) -> dict[str, str]:
    """Return {Genotype: sanitized_directory_name} from the mapping CSV.

    Uses the 'Simplified Nickname' column as the human-readable label,
    then sanitizes it for use as a directory name.
    """
    df = pd.read_csv(csv_path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        genotype = str(row["Genotype"]).strip()
        nickname = sanitize_dirname(str(row["Simplified Nickname"]))
        mapping[genotype] = nickname
    return mapping


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
            f"— variables present: {variables}"
        )
    return str(values[idx])


def find_h5_files(corridor_dir: Path) -> list[Path]:
    """Return all H5 files matching the expected patterns in a corridor dir."""
    _SUFFIXES = ("_tracked_fly", "_tracked_ball", "_preprocessed_full_body")
    return sorted(
        f for f in corridor_dir.glob("*.h5") if any(s in f.name for s in _SUFFIXES)
    )


def plan_experiment(
    exp_dir: Path,
    mapping: dict[str, str],
    output_root: Path,
) -> list[tuple[Path, Path]]:
    """Build a list of (src, dst) copy pairs for one experiment.

    Returns empty list if experiment directory does not exist.
    Missing genotypes (not in mapping) are handled gracefully.
    """
    if not exp_dir.exists():
        return []

    metadata = load_metadata(exp_dir)
    pairs: list[tuple[Path, Path]] = []

    for arena_idx in range(1, N_ARENAS + 1):
        arena_key = f"Arena{arena_idx}"
        if arena_key not in metadata:
            continue

        genotype = metadata_get(metadata, arena_key, "Genotype")
        date = metadata_get(metadata, arena_key, "Date")

        if not genotype or genotype == "None":
            continue

        nickname_dir = GENOTYPE_OVERRIDES.get(genotype) or mapping.get(genotype)
        if nickname_dir is None:
            # Fallback: sanitize the raw genotype string
            nickname_dir = sanitize_dirname(genotype)
            print(
                f"  [WARN] Unmapped genotype: '{genotype}' -> '{nickname_dir}' (fallback)"
            )

        for corridor_idx in range(1, N_CORRIDORS + 1):
            corridor_dir = exp_dir / f"arena{arena_idx}" / f"corridor{corridor_idx}"
            if not corridor_dir.exists():
                continue
            if next(corridor_dir.glob("*_tracked_fly.*.h5"), None) is None:
                continue  # no fly tracking → empty/ball-only corridor, skip

            h5_files = find_h5_files(corridor_dir)
            for src in h5_files:
                dst_dir = (
                    output_root
                    / nickname_dir
                    / date
                    / f"Arena{arena_idx}"
                    / f"Corridor{corridor_idx}"
                )
                dst = dst_dir / src.name
                pairs.append((src, dst))

    return pairs


def _detect_collision_slots(
    experiments: list[Path],
    mapping: dict[str, str],
    metadata_cache: dict[Path, dict] | None = None,
) -> dict[tuple, list[Path]]:
    """Return {(nickname, date, arena_idx): [exp_dirs]} with metadata reads only.

    Shared by plan_all_experiments (first pass) and dry_run (collision check).
    No corridor glob calls are made.

    If metadata_cache is provided (a {Path: dict} map already built by the
    caller), it is used instead of re-reading each metadata.json from disk.
    """
    from collections import defaultdict

    slot_exps: dict[tuple, list[Path]] = defaultdict(list)
    for exp_dir in sorted(experiments):
        if not exp_dir.exists():
            continue
        if metadata_cache is not None:
            if exp_dir not in metadata_cache:
                continue
            meta = metadata_cache[exp_dir]
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
                genotype = metadata_get(meta, arena_key, "Genotype")
                date = metadata_get(meta, arena_key, "Date")
            except (KeyError, IndexError):
                continue
            if not genotype or genotype == "None":
                continue
            nick = (
                GENOTYPE_OVERRIDES.get(genotype)
                or mapping.get(genotype)
                or sanitize_dirname(genotype)
            )
            slot_exps[(nick, date, arena_idx)].append(exp_dir)
    return slot_exps


def plan_all_experiments(
    experiments: list[Path],
    mapping: dict[str, str],
    output_root: Path,
) -> list[tuple[Path, Path]]:
    """Build all (src, dst) copy pairs across every experiment in the list.

    Handles same-day same-genotype same-arena collisions by appending a
    numeric suffix to the date component (e.g. 240109-1, 240109-2).
    Experiments are sorted by directory name for deterministic ordering.
    None/empty genotypes are silently skipped.
    """
    from collections import defaultdict

    experiments_sorted = sorted(experiments)

    # Pre-load all metadata once; skip experiments that fail to load.
    metadata_cache: dict[Path, dict] = {}
    for exp_dir in experiments_sorted:
        if not exp_dir.exists():
            continue
        try:
            metadata_cache[exp_dir] = load_metadata(exp_dir)
        except (FileNotFoundError, KeyError):
            continue

    # --- First pass: detect slots with >1 experiment -----------------------
    slot_exps = _detect_collision_slots(
        experiments, mapping, metadata_cache=metadata_cache
    )

    # Build (slot, exp_dir) -> date_label, disambiguating collisions
    date_labels: dict[tuple, str] = {}
    for (nickname_dir, date, arena_idx), exps in slot_exps.items():
        slot = (nickname_dir, date, arena_idx)
        if len(exps) == 1:
            date_labels[(slot, exps[0])] = date
        else:
            for i, exp_dir in enumerate(exps, 1):
                date_labels[(slot, exp_dir)] = f"{date}-{i}"
                print(
                    f"  [INFO] Date collision: {nickname_dir}/Arena{arena_idx} "
                    f"on {date} in {len(exps)} experiments "
                    f"\u2192 assigning '{date}-{i}' to {exp_dir.name}"
                )

    # --- Second pass: build pairs with resolved date labels ----------------
    all_pairs: list[tuple[Path, Path]] = []
    for exp_dir, metadata in metadata_cache.items():
        for arena_idx in range(1, N_ARENAS + 1):
            arena_key = f"Arena{arena_idx}"
            if arena_key not in metadata:
                continue
            try:
                genotype = metadata_get(metadata, arena_key, "Genotype")
                date = metadata_get(metadata, arena_key, "Date")
            except (KeyError, IndexError):
                continue
            if not genotype or genotype == "None":
                continue
            nickname_dir = GENOTYPE_OVERRIDES.get(genotype) or mapping.get(genotype)
            if nickname_dir is None:
                nickname_dir = sanitize_dirname(genotype)
            slot = (nickname_dir, date, arena_idx)
            resolved_date = date_labels.get((slot, exp_dir), date)
            for corridor_idx in range(1, N_CORRIDORS + 1):
                corridor_dir = exp_dir / f"arena{arena_idx}" / f"corridor{corridor_idx}"
                if not corridor_dir.exists():
                    continue
                if next(corridor_dir.glob("*_tracked_fly.*.h5"), None) is None:
                    continue  # no fly tracking → empty/ball-only corridor, skip
                for src in find_h5_files(corridor_dir):
                    dst_dir = (
                        output_root
                        / nickname_dir
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
    mapping: dict[str, str],
    output_root: Path,
    all_pairs: list[tuple[Path, Path]] | None = None,
) -> bool:
    """Run all pre-flight checks. Returns True if everything looks clean."""
    from collections import Counter

    issues: list[str] = []
    ok = print  # alias for readability

    # --- Test 1: Nickname collision across genotypes -------------------------
    ok("\n[Test 1] Checking nickname collisions across all genotypes...")
    full_map = {**mapping, **GENOTYPE_OVERRIDES}
    nickname_counts: Counter = Counter(full_map.values())
    collisions = {
        nick: [g for g, n in full_map.items() if n == nick]
        for nick, count in nickname_counts.items()
        if count > 1
    }
    # Intentional shared folders: same biological line, multiple genotype codes.
    intentional_shared = {
        "Wild-type_PR",  # M7 and PR are the same line
        "DDC",  # TNTxZ1661 (GMR70G12-GAL4) and TNTxDDC both target DDC neurons
        "SS32230_LAL-1",  # TNTxZ1711 and TNTxLAL1 are the same split line
        "SS32219_LAL-2",  # TNTxZ1712 and TNTxLAL2 are the same split line
    }
    for nick, genotypes in collisions.items():
        if nick not in intentional_shared:
            msg = f"  COLLISION: '{nick}' <- {genotypes}"
            print(msg)
            issues.append(msg)
    if not collisions or all(n in intentional_shared for n in collisions):
        print("  OK — no unintentional nickname collisions")

    # --- Test 2: Unmapped genotypes across all experiments ------------------
    ok("\n[Test 2] Checking for unmapped genotypes across all experiments...")
    unmapped_seen: set[str] = set()
    missing_exps: list[Path] = []
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
        variables = meta["Variable"]
        try:
            genotype_idx = variables.index("Genotype")
        except ValueError:
            msg = f"  [ERROR] 'Genotype' missing from Variable list in {exp_dir}"
            print(msg)
            issues.append(msg)
            continue
        for key, values in meta.items():
            if not key.startswith("Arena"):
                continue
            genotype = str(values[genotype_idx]).strip()
            if not genotype or genotype == "None":
                continue
            if (
                GENOTYPE_OVERRIDES.get(genotype) is None
                and mapping.get(genotype) is None
            ):
                unmapped_seen.add(genotype)
    if missing_exps:
        for p in missing_exps:
            msg = f"  [ERROR] Experiment directory missing: {p}"
            print(msg)
            issues.append(msg)
    if unmapped_seen:
        for g in sorted(unmapped_seen):
            msg = f"  [WARN] Unmapped genotype (will use raw name): '{g}'"
            print(msg)
    else:
        print("  OK — all genotypes mapped")

    # --- Test 3: Destination path collisions (two srcs -> same dst) ---------
    ok("\n[Test 3] Checking for destination path collisions across all experiments...")
    from collections import defaultdict as _dd

    if all_pairs is None:
        all_pairs = plan_all_experiments(experiments, mapping, output_root)
    dst_to_srcs: dict[Path, list[Path]] = _dd(list)
    for src, dst in all_pairs:
        dst_to_srcs[dst].append(src)
    dup_dsts = {dst: srcs for dst, srcs in dst_to_srcs.items() if len(srcs) > 1}
    if dup_dsts:
        for dst, srcs in sorted(dup_dsts.items()):
            msg = f"  [ERROR] {len(srcs)} sources -> {dst}\n    srcs: {srcs}"
            print(msg)
            issues.append(msg)
    else:
        print("  OK — no destination path collisions")

    # --- Test 4: Expected H5 file counts per corridor -----------------------
    ok(
        "\n[Test 4] Checking H5 file counts per corridor with fly tracking (expected 2 or 3)..."
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
                    continue  # expected: empty/ball-only corridor
                n = len(find_h5_files(corridor_dir))
                if n not in (2, 3):
                    msg = f"  [WARN] {corridor_dir}: {n} H5 files"
                    print(msg)
                    unexpected.append(msg)
    if skipped_no_fly:
        print(f"  INFO — {skipped_no_fly} corridors skipped (no fly tracking file)")
    if not unexpected:
        print("  OK — all corridors with fly tracking have 2 or 3 H5 files")

    # --- Test 5: Sanity-check total planned file count ----------------------
    ok("\n[Test 5] Checking total planned file count...")
    total = len(all_pairs)
    print(f"  Total H5 files planned for copy: {total}")
    if total < 10_000:
        msg = f"  [WARN] Total file count ({total}) seems low — expected ~12k-18k"
        print(msg)
    elif total > 25_000:
        msg = f"  [WARN] Total file count ({total}) seems high — expected ~12k-18k"
        print(msg)
    else:
        print("  OK — file count is in the expected range (10k-25k)")

    # --- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    if issues:
        print(f"Pre-flight FAILED — {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("Pre-flight PASSED — safe to run --copy")
        return True


def dry_run(
    exp_dir: Path,
    experiments: list[Path],
    mapping: dict[str, str],
    output_root: Path,
) -> None:
    """Print a summary for a single experiment without touching the filesystem.

    Uses plan_all_experiments so that date-disambiguation suffixes (-1/-2)
    are applied exactly as they would be during --copy.
    """
    print(f"\n{'='*70}")
    print(f"DRY RUN  —  {exp_dir.name}")
    print(f"{'='*70}")

    if not exp_dir.exists():
        print(f"  [ERROR] Directory does not exist: {exp_dir}")
        return

    metadata = load_metadata(exp_dir)
    # Use shared slot-detection helper (metadata reads only, no corridor globs).
    # If this experiment is not involved in any collision, use the cheap
    # single-experiment plan instead of scanning all experiments' corridors.
    slot_exps = _detect_collision_slots(experiments, mapping)
    exp_has_collision = any(
        exp_dir in exps and len(exps) > 1 for exps in slot_exps.values()
    )
    if exp_has_collision:
        # Full plan needed for correct -1/-2 date suffixes.
        all_global = plan_all_experiments(experiments, mapping, output_root)
        pairs = [(src, dst) for src, dst in all_global if src.is_relative_to(exp_dir)]
    else:
        # Fast path: only glob this experiment's corridors (~54 calls vs ~4000).
        pairs = plan_experiment(exp_dir, mapping, output_root)

    # Summarise genotypes found
    genotypes_found: dict[str, str] = {}
    for arena_idx in range(1, N_ARENAS + 1):
        arena_key = f"Arena{arena_idx}"
        if arena_key not in metadata:
            continue
        genotype = metadata_get(metadata, arena_key, "Genotype")
        nickname = GENOTYPE_OVERRIDES.get(genotype) or mapping.get(
            genotype, f"[UNMAPPED] {sanitize_dirname(genotype)}"
        )
        genotypes_found[genotype] = nickname

    print(f"\nGenotypes found ({len(genotypes_found)}):")
    for genotype, nickname in genotypes_found.items():
        print(f"  {genotype:30s} -> {nickname}")

    print(f"\nH5 files to copy ({len(pairs)}):")
    for src, dst in pairs[:20]:  # show first 20
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
    mapping: dict[str, str],
    output_root: Path,
    workers: int = 8,
    all_pairs: list[tuple[Path, Path]] | None = None,
) -> int:
    """Copy all H5 files in parallel. Returns number of files copied.

    Uses ThreadPoolExecutor to overlap I/O waits on network filesystems.
    Each file is written atomically via a .tmp intermediate.
    """
    if all_pairs is None:
        all_pairs = plan_all_experiments(experiments, mapping, output_root)
    total = len(all_pairs)
    copied = skipped = warns = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_copy_one, src, dst): dst for src, dst in all_pairs}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()  # re-raises any copy exception
            if result == "copied":
                copied += 1
            elif result == "skipped":
                skipped += 1
            elif result == "warn":
                print(f"  [WARN] Size mismatch on existing file: {futures[future]}")
                warns += 1
            if i % 500 == 0 or i == total:
                print(
                    f"  Progress: {i}/{total} "
                    f"({copied} copied, {skipped} skipped, {warns} warnings)..."
                )
    print(
        f"\n  Done: {copied} copied, {skipped} already present, "
        f"{warns} size warnings (total planned: {total})"
    )
    return copied


def compress_genotypes(output_root: Path) -> None:
    """Archive each top-level genotype folder into one or more .tar files.

    HDF5 files are already internally compressed or binary, so gzip achieves
    virtually nothing and wastes significant CPU time. Plain .tar bundles
    files for upload without the overhead. The source directory is deleted
    after successful archive creation so only .tar files remain.
    """
    genotype_dirs = sorted(
        [d for d in output_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    print(f"\nArchiving {len(genotype_dirs)} genotype folders...")
    for gdir in genotype_dirs:
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
        description="Prepare silencing screen H5 files for Dataverse upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: show what would be copied for one experiment (use --experiment-index)",
    )
    mode.add_argument(
        "--copy",
        action="store_true",
        help="Copy H5 files for all experiments",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run all pre-flight checks (nickname collisions, unmapped genotypes, "
        "destination collisions, H5 counts, total file count) without copying anything",
    )

    parser.add_argument(
        "--experiment-index",
        type=int,
        default=0,
        help="Index (0-based) of experiment to use for --dry-run (default: 0)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="After copying, archive each genotype folder to .tar (used with --copy)",
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
        "--mapping-csv",
        type=Path,
        default=MAPPING_CSV_PATH,
        help=f"Path to genotype->nickname CSV (default: {MAPPING_CSV_PATH})",
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

    print(f"Loading genotype mapping from {args.mapping_csv} ...")
    mapping = load_mapping(args.mapping_csv)
    print(f"  {len(mapping)} genotypes mapped")

    print(f"Loading experiments from {args.yaml} ...")
    experiments = load_experiments(args.yaml)
    print(f"  {len(experiments)} experiments found")

    if args.dry_run:
        exp = experiments[args.experiment_index]
        dry_run(exp, experiments, mapping, args.output_root)
        return

    if args.validate:
        validate(experiments, mapping, args.output_root)
        return

    # --copy mode
    # Compute the plan once; pass it to both validate and copy to avoid
    # re-reading metadata and re-globbing corridors twice.
    all_pairs = plan_all_experiments(experiments, mapping, args.output_root)
    if not args.force:
        print("Running pre-flight validation before copy...")
        if not validate(experiments, mapping, args.output_root, all_pairs=all_pairs):
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
        mapping,
        args.output_root,
        workers=args.workers,
        all_pairs=all_pairs,
    )

    if args.compress:
        compress_genotypes(args.output_root)


if __name__ == "__main__":
    main()
