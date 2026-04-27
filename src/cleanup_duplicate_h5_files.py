#!/usr/bin/env python3
"""
Cleanup script to handle duplicate H5 files in target directories.

After running the SLEAP mismatch fix, this script checks all directories for multiple
H5 files of the same type and moves extras to /invalid directories, keeping only
the most recently modified file.

Usage:
    python cleanup_duplicate_h5_files.py [--file-type TYPE] [--dry-run] [--yaml-file PATH]
"""

import argparse
import yaml
from pathlib import Path
import sys
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(__file__).parent.parent / "outputs" / f"cleanup_duplicates_log_{timestamp}.log"
    log_file.parent.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"📝 Log file: {log_file}")
    return logger


def load_directories_from_yaml(yaml_file: Path) -> List[Path]:
    """
    Load directories from a YAML file.

    Parameters
    ----------
    yaml_file : Path
        Path to the YAML file

    Returns
    -------
    List[Path]
        List of directory paths
    """
    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        # Handle different YAML structures
        directories = []
        if isinstance(data, dict):
            if "directories" in data:
                directories = data["directories"]
            elif "folders" in data:
                directories = data["folders"]
            else:
                # Try to extract paths from any list values
                for key, value in data.items():
                    if isinstance(value, list):
                        directories.extend(value)
        elif isinstance(data, list):
            directories = data

        return [Path(d) for d in directories if d]

    except Exception as e:
        print(f"❌ Error loading YAML file {yaml_file}: {e}")
        return []


def find_directories_with_duplicates(directories: List[Path], file_type: str) -> Dict[Path, List[Path]]:
    """
    Find directories that contain multiple H5 files of the same type.

    Parameters
    ----------
    directories : List[Path]
        List of base directories to search
    file_type : str
        Type of H5 files to check (full_body, ball, fly)

    Returns
    -------
    Dict[Path, List[Path]]
        Dictionary mapping directory paths to lists of H5 files found
    """
    pattern = f"*{file_type}*.h5"
    duplicates_found = {}

    print(f"🔍 Searching for duplicate {file_type} files...")

    for base_dir in directories:
        if not base_dir.exists():
            print(f"⚠️  Directory does not exist: {base_dir}")
            continue

        # Find all arena/corridor directories
        for arena_dir in base_dir.glob("arena*"):
            if not arena_dir.is_dir():
                continue

            for corridor_dir in arena_dir.glob("corridor*"):
                if not corridor_dir.is_dir():
                    continue

                # Find H5 files of the specified type in this corridor
                h5_files = list(corridor_dir.glob(pattern))

                if len(h5_files) > 1:
                    duplicates_found[corridor_dir] = h5_files
                    print(f"📁 Found {len(h5_files)} {file_type} files in {corridor_dir.relative_to(base_dir)}")
                    for h5_file in h5_files:
                        print(f"   - {h5_file.name} (modified: {datetime.fromtimestamp(h5_file.stat().st_mtime)})")

    return duplicates_found


def create_invalid_directory(corridor_dir: Path) -> Path:
    """
    Create an /invalid directory at the arena level.

    Parameters
    ----------
    corridor_dir : Path
        Path to the corridor directory

    Returns
    -------
    Path
        Path to the invalid directory
    """
    arena_dir = corridor_dir.parent
    invalid_dir = arena_dir / "invalid"
    invalid_dir.mkdir(exist_ok=True)
    return invalid_dir


def select_file_to_keep(h5_files: List[Path]) -> Path:
    """
    Select which H5 file to keep based on modification time (most recent).

    Parameters
    ----------
    h5_files : List[Path]
        List of H5 files to choose from

    Returns
    -------
    Path
        The file to keep
    """
    # Keep the most recently modified file
    return max(h5_files, key=lambda f: f.stat().st_mtime)


def cleanup_duplicates(
    duplicates: Dict[Path, List[Path]], file_type: str, dry_run: bool, logger: logging.Logger
) -> Dict[str, int]:
    """
    Clean up duplicate H5 files by moving extras to /invalid directories.

    Parameters
    ----------
    duplicates : Dict[Path, List[Path]]
        Dictionary of directories with their duplicate files
    file_type : str
        Type of files being cleaned up
    dry_run : bool
        If True, only simulate the cleanup
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, int]
        Statistics about the cleanup operation
    """
    stats = {"directories_processed": 0, "files_moved_to_invalid": 0, "files_kept": 0, "errors": 0}

    if dry_run:
        logger.info("🔍 DRY RUN MODE: No files will actually be moved")
        print("🔍 DRY RUN MODE: Simulating cleanup...")
    else:
        logger.info("🧹 EXECUTING CLEANUP")
        print("🧹 EXECUTING CLEANUP...")

    for corridor_dir, h5_files in duplicates.items():
        stats["directories_processed"] += 1
        logger.info(f"📁 Processing {corridor_dir} with {len(h5_files)} {file_type} files")
        print(f"\n📁 Processing {corridor_dir.name} ({len(h5_files)} files)")

        try:
            # Select file to keep (most recent)
            file_to_keep = select_file_to_keep(h5_files)
            files_to_move = [f for f in h5_files if f != file_to_keep]

            logger.info(
                f"   ✅ Keeping: {file_to_keep.name} (modified: {datetime.fromtimestamp(file_to_keep.stat().st_mtime)})"
            )
            print(f"   ✅ Keeping: {file_to_keep.name}")
            stats["files_kept"] += 1

            # Create invalid directory
            invalid_dir = create_invalid_directory(corridor_dir)

            # Move extra files to invalid
            for file_to_move in files_to_move:
                invalid_target = invalid_dir / file_to_move.name

                # Handle name conflicts in /invalid directory
                counter = 1
                while invalid_target.exists():
                    stem = file_to_move.stem
                    suffix = file_to_move.suffix
                    invalid_target = invalid_dir / f"{stem}_duplicate_{counter}{suffix}"
                    counter += 1

                if dry_run:
                    logger.info(f"   📦 Would move to invalid: {file_to_move.name} → {invalid_target.name}")
                    print(f"   📦 Would move to invalid: {file_to_move.name}")
                else:
                    logger.info(f"   📦 Moving to invalid: {file_to_move.name} → {invalid_target}")
                    shutil.move(str(file_to_move), str(invalid_target))
                    print(f"   📦 Moved to invalid: {file_to_move.name}")
                    stats["files_moved_to_invalid"] += 1

        except Exception as e:
            logger.error(f"❌ Error processing {corridor_dir}: {e}")
            print(f"   ❌ Error: {e}")
            stats["errors"] += 1

    return stats


def save_cleanup_report(duplicates: Dict[Path, List[Path]], stats: Dict[str, int], file_type: str) -> None:
    """
    Save a detailed cleanup report to CSV.

    Parameters
    ----------
    duplicates : Dict[Path, List[Path]]
        Dictionary of directories with their duplicate files (before cleanup)
    stats : Dict[str, int]
        Cleanup statistics
    file_type : str
        Type of files that were cleaned up
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(__file__).parent.parent / "outputs" / f"cleanup_duplicates_report_{file_type}_{timestamp}.csv"

    report_data = []
    for corridor_dir, h5_files in duplicates.items():
        # Get file stats before determining which to keep
        # Handle case where files may have been moved already
        file_stats = []
        for h5_file in h5_files:
            try:
                if h5_file.exists():
                    file_stats.append((h5_file, h5_file.stat().st_mtime))
                else:
                    # File was moved, use current time as placeholder
                    file_stats.append((h5_file, 0))
            except Exception:
                # Fallback for any file access issues
                file_stats.append((h5_file, 0))

        # Sort by modification time to determine which was kept
        file_stats.sort(key=lambda x: x[1], reverse=True)
        file_to_keep = file_stats[0][0] if file_stats else h5_files[0]

        # Get experiment info from path
        path_parts = corridor_dir.parts
        experiment_name = ""
        for part in path_parts:
            if "_Videos_Tracked" in part:
                experiment_name = part
                break

        for i, (h5_file, mtime) in enumerate(file_stats):
            action = "keep" if h5_file == file_to_keep else "move_to_invalid"

            report_data.append(
                {
                    "experiment": experiment_name,
                    "corridor_directory": str(corridor_dir),
                    "h5_filename": h5_file.name,
                    "h5_full_path": str(h5_file),
                    "file_modification_time": datetime.fromtimestamp(mtime) if mtime > 0 else "File moved",
                    "action": action,
                    "file_type": file_type,
                    "total_files_in_directory": len(h5_files),
                    "file_rank_by_modification": i + 1,
                }
            )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(report_data)
    df.to_csv(report_file, index=False)

    print(f"\n💾 Cleanup report saved to: {report_file}")
    print(f"📊 Report contains details for {len(report_data)} files across {len(duplicates)} directories")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up duplicate H5 files in target directories")
    parser.add_argument(
        "--file-type",
        "-t",
        type=str,
        default="full_body",
        choices=["full_body", "ball", "fly"],
        help="Type of H5 files to clean up (default: full_body)",
    )
    parser.add_argument("--yaml-file", "-y", type=str, help="YAML file containing list of directories to check")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Perform a dry run without actually moving files")
    parser.add_argument("--specific-directory", "-s", type=str, help="Clean up duplicates in a specific directory")

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()

    logger.info("🧹 Duplicate H5 File Cleanup Started")
    logger.info(f"🏷️  File type: {args.file_type}")
    logger.info(f"🧪 Dry run: {args.dry_run}")

    print(f"🧹 Cleaning up duplicate {args.file_type} files")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be moved")

    # Determine directories to check
    directories = []

    if args.specific_directory:
        specific_dir = Path(args.specific_directory)
        if specific_dir.exists():
            directories = [specific_dir]
        else:
            print(f"❌ Directory does not exist: {specific_dir}")
            sys.exit(1)

    elif args.yaml_file:
        yaml_file = Path(args.yaml_file)
        if not yaml_file.exists():
            print(f"❌ YAML file does not exist: {yaml_file}")
            sys.exit(1)
        directories = load_directories_from_yaml(yaml_file)
        logger.info(f"📁 Loaded {len(directories)} directories from YAML")

    else:
        # Default: look for YAML files in experiments_yaml directory
        yaml_dir = Path(__file__).parent.parent / "experiments_yaml"
        if yaml_dir.exists():
            yaml_files = list(yaml_dir.glob("*.yaml"))
            if yaml_files:
                print(f"🔍 Found {len(yaml_files)} YAML files in experiments_yaml directory:")
                for i, yaml_file in enumerate(yaml_files):
                    print(f"   {i+1}. {yaml_file.name}")
                print()

                # Ask user to select
                try:
                    choice = input(
                        "Enter the number of the YAML file to use (or press Enter to use the first one): "
                    ).strip()
                    if choice:
                        yaml_file = yaml_files[int(choice) - 1]
                    else:
                        yaml_file = yaml_files[0]

                    print(f"Using YAML file: {yaml_file}")
                    directories = load_directories_from_yaml(yaml_file)

                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    sys.exit(1)
            else:
                print("❌ No YAML files found in experiments_yaml directory")
                sys.exit(1)
        else:
            print("❌ No experiments_yaml directory found and no arguments provided")
            print("Usage: python cleanup_duplicate_h5_files.py --yaml-file PATH or --specific-directory PATH")
            sys.exit(1)

    if not directories:
        print("❌ No directories to process")
        sys.exit(1)

    print(f"📋 Processing {len(directories)} base directories")

    # Find directories with duplicate files
    duplicates = find_directories_with_duplicates(directories, args.file_type)

    if not duplicates:
        print(f"✅ No duplicate {args.file_type} files found!")
        logger.info(f"✅ No duplicate {args.file_type} files found")
        return

    print(f"\n📊 Found {len(duplicates)} directories with duplicate {args.file_type} files")
    logger.info(f"📊 Found {len(duplicates)} directories with duplicates")

    # Confirm before proceeding (unless dry run)
    if not args.dry_run:
        total_files_to_move = sum(len(files) - 1 for files in duplicates.values())
        print(f"\n⚠️  This will move {total_files_to_move} files to /invalid directories")
        confirm = input("Continue? [y/N]: ").lower().strip()
        if confirm != "y":
            print("❌ Operation cancelled")
            return

    # Perform cleanup
    stats = cleanup_duplicates(duplicates, args.file_type, args.dry_run, logger)

    # Save detailed report
    save_cleanup_report(duplicates, stats, args.file_type)

    # Summary
    logger.info("============================================================")
    logger.info("📊 CLEANUP COMPLETE")
    logger.info("============================================================")
    logger.info(f"📁 Directories processed: {stats['directories_processed']}")
    logger.info(f"✅ Files kept: {stats['files_kept']}")
    logger.info(f"📦 Files moved to /invalid: {stats['files_moved_to_invalid']}")
    logger.info(f"❌ Errors: {stats['errors']}")

    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   - {stats['directories_processed']} directories processed")
    print(f"   - {stats['files_kept']} files kept (most recent)")
    print(f"   - {stats['files_moved_to_invalid']} files moved to /invalid")
    print(f"   - {stats['errors']} errors")

    if args.dry_run:
        print(f"\n🧪 This was a dry run - remove --dry-run flag to perform actual cleanup")

    if stats["errors"] > 0:
        print(f"⚠️  {stats['errors']} errors occurred - check logs for details")


if __name__ == "__main__":
    main()
