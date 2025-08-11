#!/usr/bin/env python3
"""
Script to fix SLEAP H5 file mismatches by moving files to their correct locations.

This script reads the mismatch CSV files and moves H5 files to match their referenced video paths.
Files that would be overwritten are moved to an /invalid directory to preserve data.

Usage:
    python fix_sleap_mismatches.py [--csv-dir PATH] [--dry-run] [--file-type TYPE]
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import shutil
from typing import Dict, List, Tuple, Set
from datetime import datetime
import logging


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger("sleap_fixer")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_mismatch_data(csv_files: List[Path]) -> pd.DataFrame:
    """
    Load and combine mismatch CSV files.

    Parameters
    ----------
    csv_files : List[Path]
        List of CSV files containing mismatch data

    Returns
    -------
    pd.DataFrame
        Combined mismatch data
    """
    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_csv"] = csv_file.name
            dfs.append(df)
            print(f"📁 Loaded {len(df)} mismatches from {csv_file.name}")
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"📊 Total mismatches loaded: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()


def calculate_target_path(row: pd.Series) -> Path:
    """
    Calculate the target path where the H5 file should be moved.

    Parameters
    ----------
    row : pd.Series
        Row from the mismatch DataFrame

    Returns
    -------
    Path
        Target path for the H5 file
    """
    h5_path = Path(row["h5_full_path"])
    # Convert to int to remove any decimals from the CSV data
    video_arena = int(float(str(row["video_arena"])))
    video_corridor = int(float(str(row["video_corridor"])))

    # Get the experiment directory (parent of arena directory)
    experiment_dir = h5_path.parent.parent.parent

    # Construct target path
    target_dir = experiment_dir / f"arena{video_arena}" / f"corridor{video_corridor}"
    target_path = target_dir / h5_path.name

    return target_path


def analyze_move_plan(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, List]:
    """
    Analyze the moves needed and detect potential conflicts.

    Parameters
    ----------
    df : pd.DataFrame
        Mismatch data
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, List]
        Dictionary containing move plan analysis
    """
    moves = []
    conflicts = []

    for idx, row in df.iterrows():
        source_path = Path(str(row["h5_full_path"]))
        target_path = calculate_target_path(row)

        move_info = {
            "source": source_path,
            "target": target_path,
            "experiment": str(row["experiment_date"]),
            "pattern": f"{row['h5_arena']}/{row['h5_corridor']} → {row['video_arena']}/{row['video_corridor']}",
            "exists_at_target": target_path.exists(),
            "file_type": str(row.get("file_type", "full_body")),  # Extract file type from CSV
            "idx": idx,
        }

        moves.append(move_info)

        # Check for conflicts
        if target_path.exists():
            conflicts.append(move_info)

    logger.info(f"📋 Planned moves: {len(moves)}")
    logger.info(f"⚠️  Potential conflicts: {len(conflicts)}")

    return {"moves": moves, "conflicts": conflicts}


def find_move_chains(moves: List[Dict]) -> List[List[Dict]]:
    """
    Find chains of moves where file A needs to go to location B,
    but location B has file C that needs to go somewhere else.

    Parameters
    ----------
    moves : List[Dict]
        List of move operations

    Returns
    -------
    List[List[Dict]]
        List of move chains
    """
    # Create mapping of source locations to moves
    source_to_move = {str(move["source"]): move for move in moves}

    # Create mapping of target locations to moves
    target_to_move = {str(move["target"]): move for move in moves}

    chains = []
    processed = set()

    for move in moves:
        if move["idx"] in processed:
            continue

        # Start a new chain
        chain = []
        current_move = move

        while current_move is not None:
            if current_move["idx"] in processed:
                break

            chain.append(current_move)
            processed.add(current_move["idx"])

            # Look for the next move in the chain
            target_str = str(current_move["target"])
            next_move = source_to_move.get(target_str)
            current_move = next_move

        if len(chain) > 1:
            chains.append(chain)

    return chains


def check_existing_files_in_target(target_path: Path, file_type: str) -> List[Path]:
    """
    Check if there are existing files of the same type in the target directory.

    Parameters
    ----------
    target_path : Path
        Target path where file will be moved
    file_type : str
        Type of file (full_body, ball, fly)

    Returns
    -------
    List[Path]
        List of existing files of the same type in target directory
    """
    target_dir = target_path.parent
    pattern = f"*{file_type}*.h5"

    if target_dir.exists():
        return list(target_dir.glob(pattern))
    else:
        return []


def create_invalid_directory(base_path: Path) -> Path:
    """
    Create an /invalid directory in the experiment directory.

    Parameters
    ----------
    base_path : Path
        Path to an H5 file (to extract experiment directory)

    Returns
    -------
    Path
        Path to the invalid directory
    """
    # Get experiment directory (3 levels up from H5 file)
    experiment_dir = base_path.parent.parent.parent
    invalid_dir = experiment_dir / "invalid"
    invalid_dir.mkdir(exist_ok=True)
    return invalid_dir


def execute_moves(plan: Dict[str, List], dry_run: bool, logger: logging.Logger) -> Dict[str, int]:
    """
    Execute the move plan.

    Parameters
    ----------
    plan : Dict[str, List]
        Move plan from analyze_move_plan
    dry_run : bool
        If True, only simulate the moves
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, int]
        Statistics about the operations performed
    """
    stats = {"moves_completed": 0, "files_moved_to_invalid": 0, "errors": 0, "skipped": 0}

    moves = plan["moves"]

    # Find move chains to handle swaps properly
    chains = find_move_chains(moves)
    single_moves = [move for move in moves if not any(move in chain for chain in chains)]

    logger.info(f"🔗 Found {len(chains)} move chains and {len(single_moves)} single moves")

    # Handle move chains (swaps)
    for i, chain in enumerate(chains):
        logger.info(f"🔗 Processing chain {i+1}/{len(chains)} with {len(chain)} moves")

        if not dry_run:
            # For chains, we need to use temporary files
            temp_files = []
            try:
                # Move files to temporary locations first
                for j, move in enumerate(chain):
                    source = move["source"]
                    temp_path = source.parent / f"temp_{datetime.now().strftime('%H%M%S')}_{j}_{source.name}"

                    logger.info(f"  📦 Moving {source.name} to temporary location")
                    shutil.move(str(source), str(temp_path))
                    temp_files.append((temp_path, move["target"]))

                # Now move from temporary locations to final destinations
                for temp_path, final_target in temp_files:
                    final_target.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"  ✅ Moving {temp_path.name} to {final_target}")
                    shutil.move(str(temp_path), str(final_target))
                    stats["moves_completed"] += 1

            except Exception as e:
                logger.error(f"❌ Error in chain processing: {e}")
                stats["errors"] += 1

                # Clean up temp files if something went wrong
                for temp_path, _ in temp_files:
                    if temp_path.exists():
                        logger.info(f"🧹 Cleaning up temp file: {temp_path}")
                        temp_path.unlink()
        else:
            logger.info(f"  [DRY RUN] Would process chain with {len(chain)} swaps")
            for move in chain:
                logger.info(f"    {move['source'].name} → {move['target']}")

    # Handle single moves
    for move in single_moves:
        source = move["source"]
        target = move["target"]
        file_type = move.get("file_type", "full_body")  # Default to full_body for backward compatibility

        if not source.exists():
            logger.warning(f"⚠️  Source file does not exist: {source}")
            stats["skipped"] += 1
            continue

        # Check for existing files of the same type in target directory
        existing_files = check_existing_files_in_target(target, file_type)

        if existing_files:
            logger.info(f"⚠️  Found {len(existing_files)} existing {file_type} files in target directory")

            # Move all existing files of the same type to invalid directory
            for existing_file in existing_files:
                invalid_dir = create_invalid_directory(source)
                invalid_path = invalid_dir / existing_file.name

                # Handle name conflicts in invalid directory
                counter = 1
                while invalid_path.exists():
                    stem = existing_file.stem
                    suffix = existing_file.suffix
                    invalid_path = invalid_dir / f"{stem}_conflict_{counter}{suffix}"
                    counter += 1

                if not dry_run:
                    logger.info(
                        f"🗂️  Moving existing {file_type} file to invalid: {existing_file.name} → {invalid_path}"
                    )
                    shutil.move(str(existing_file), str(invalid_path))
                    stats["files_moved_to_invalid"] += 1
                else:
                    logger.info(f"  [DRY RUN] Would move existing {file_type} file to invalid: {existing_file.name}")

        # Check if target exists (exact same file name conflict)
        if target.exists():
            invalid_dir = create_invalid_directory(source)
            invalid_path = invalid_dir / target.name

            # Handle name conflicts in invalid directory
            counter = 1
            while invalid_path.exists():
                stem = target.stem
                suffix = target.suffix
                invalid_path = invalid_dir / f"{stem}_conflict_{counter}{suffix}"
                counter += 1

            if not dry_run:
                logger.info(f"🗂️  Moving existing file to invalid: {target.name} → {invalid_path}")
                shutil.move(str(target), str(invalid_path))
                stats["files_moved_to_invalid"] += 1
            else:
                logger.info(f"  [DRY RUN] Would move existing file to invalid: {target.name}")

        # Move the source file to target
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Moving: {source.name} → {target}")
            shutil.move(str(source), str(target))
            stats["moves_completed"] += 1
        else:
            logger.info(f"  [DRY RUN] Would move: {source.name} → {target}")

    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix SLEAP H5 file mismatches by moving files to correct locations")
    parser.add_argument(
        "--csv-dir",
        "-c",
        type=str,
        default=None,
        help="Directory containing mismatch CSV files (default: outputs/tnt_screen_analysis)",
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Perform a dry run without actually moving files")
    parser.add_argument(
        "--file-type",
        "-t",
        type=str,
        default="full_body",
        choices=["full_body", "ball", "fly"],
        help="Type of H5 files to fix (default: full_body)",
    )
    parser.add_argument(
        "--log-file", "-l", type=str, default=None, help="Log file path (default: outputs/sleap_fix_log_TIMESTAMP.log)"
    )

    args = parser.parse_args()

    # Determine CSV directory
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
    else:
        script_dir = Path(__file__).parent
        csv_dir = script_dir.parent / "outputs" / "tnt_screen_analysis"

    # Determine log file
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(__file__).parent.parent / "outputs" / f"sleap_fix_log_{timestamp}.log"

    # Ensure log directory exists
    log_file.parent.mkdir(exist_ok=True)

    # Set up logging
    logger = setup_logging(log_file)

    logger.info(f"🔧 SLEAP Mismatch Fixer Started")
    logger.info(f"📁 CSV directory: {csv_dir}")
    logger.info(f"🏷️  File type: {args.file_type}")
    logger.info(f"🧪 Dry run: {args.dry_run}")
    logger.info(f"📝 Log file: {log_file}")

    if not csv_dir.exists():
        logger.error(f"❌ CSV directory does not exist: {csv_dir}")
        sys.exit(1)

    # Find CSV files for the specified file type
    csv_pattern = f"sleap_mismatches_{args.file_type}_*.csv"
    csv_files = list(csv_dir.glob(csv_pattern))

    if not csv_files:
        logger.error(f"❌ No CSV files found matching pattern: {csv_pattern}")
        sys.exit(1)

    logger.info(f"📊 Found {len(csv_files)} CSV files for {args.file_type} files")

    # Load mismatch data
    df = load_mismatch_data(csv_files)

    if df.empty:
        logger.info("✅ No mismatches found, nothing to fix!")
        return

    # Filter for arena/corridor mismatches only (skip file corruption issues)
    arena_mismatches = df[df["error_type"] == "arena_corridor_mismatch"].copy()

    if arena_mismatches.empty:
        logger.info("✅ No arena/corridor mismatches found, nothing to fix!")
        return

    logger.info(f"🎯 Processing {len(arena_mismatches)} arena/corridor mismatches")

    # Analyze move plan
    plan = analyze_move_plan(arena_mismatches, logger)

    # Show summary before execution
    logger.info("\n" + "=" * 60)
    logger.info("📋 MOVE PLAN SUMMARY")
    logger.info("=" * 60)

    for move in plan["moves"][:5]:  # Show first 5
        logger.info(f"📂 {move['source'].name}")
        logger.info(f"   From: {move['source'].parent}")
        logger.info(f"   To:   {move['target'].parent}")
        logger.info(f"   Pattern: {move['pattern']}")
        if move["exists_at_target"]:
            logger.info(f"   ⚠️  Target exists (will be moved to /invalid)")
        logger.info("")

    if len(plan["moves"]) > 5:
        logger.info(f"... and {len(plan['moves']) - 5} more moves")

    # Ask for confirmation if not dry run
    if not args.dry_run:
        logger.info("\n⚠️  This will actually move files! Are you sure you want to continue?")
        response = input("Type 'yes' to continue, anything else to abort: ")
        if response.lower() != "yes":
            logger.info("❌ Operation aborted by user")
            return

    # Execute the plan
    logger.info("\n🚀 Executing move plan...")
    stats = execute_moves(plan, args.dry_run, logger)

    # Print final statistics
    logger.info("\n" + "=" * 60)
    logger.info("📊 OPERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Files moved successfully: {stats['moves_completed']}")
    logger.info(f"🗂️  Files moved to /invalid: {stats['files_moved_to_invalid']}")
    logger.info(f"⚠️  Files skipped: {stats['skipped']}")
    logger.info(f"❌ Errors encountered: {stats['errors']}")

    if args.dry_run:
        logger.info("\n🧪 This was a dry run - no files were actually moved")
        logger.info("Remove --dry-run flag to perform actual moves")
    else:
        logger.info(f"\n📝 Detailed log saved to: {log_file}")
        logger.info("💡 Run the SLEAP analysis again to verify the fixes")


if __name__ == "__main__":
    main()
