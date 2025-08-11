#!/usr/bin/env python3
"""
Diagnostic script to check SLEAP file video path consistency using SleapUtils with multiprocessing.

This script uses multiprocessing instead of threading to bypass the GIL and achieve
true parallelism for CPU-intensive H5 file processing.

Usage:
    python check_sleap_video_paths_yaml_multiprocess.py [--yaml-file PATH] [--specific-directory PATH]
"""

import argparse
import yaml
from pathlib import Path
import sys
import os
import csv
import glob
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Pool, Manager
import functools
import time

# Import the SLEAP utilities
try:
    from utils_behavior import Sleap_utils
except ImportError as e:
    print(f"❌ Error importing utils_behavior: {e}")
    print("Make sure utils_behavior package is installed and accessible.")
    sys.exit(1)


def extract_arena_corridor_from_path(file_path):
    """Extract arena and corridor numbers from the file path."""
    path_parts = Path(file_path).parts
    arena_num = None
    corridor_num = None

    for part in path_parts:
        if part.startswith("arena") and part[5:].isdigit():
            arena_num = int(part[5:])
        elif part.startswith("corridor") and part[8:].isdigit():
            corridor_num = int(part[8:])

    return arena_num, corridor_num


def extract_arena_corridor_from_video_path(video_path):
    """Extract arena and corridor numbers from video path."""
    if not video_path:
        return None, None

    path_parts = Path(video_path).parts
    arena_num = None
    corridor_num = None

    for part in path_parts:
        if part.startswith("arena") and part[5:].isdigit():
            arena_num = int(part[5:])
        elif part.startswith("corridor") and part[8:].isdigit():
            corridor_num = int(part[8:])

    return arena_num, corridor_num


def process_single_h5_file(args):
    """Process a single H5 file - designed for multiprocessing."""
    h5_file_path, file_pattern = args

    try:
        # Load the SLEAP H5 file using SleapUtils
        print(f"Loading SLEAP H5 file with SleapUtils: {h5_file_path}")
        sleap_tracks = Sleap_utils.Sleap_Tracks(h5_file_path)

        # Get video path from the H5 file
        video_path = sleap_tracks.video

        # Extract arena/corridor from H5 file path
        h5_arena, h5_corridor = extract_arena_corridor_from_path(h5_file_path)

        # Extract arena/corridor from video path
        video_arena, video_corridor = extract_arena_corridor_from_video_path(video_path)

        # Check if video file exists
        video_exists = video_path and Path(video_path).exists()

        # Check for arena/corridor mismatch
        arena_corridor_match = h5_arena == video_arena and h5_corridor == video_corridor

        result = {
            "h5_file_path": h5_file_path,
            "h5_arena": h5_arena,
            "h5_corridor": h5_corridor,
            "video_path": video_path,
            "video_arena": video_arena,
            "video_corridor": video_corridor,
            "video_exists": video_exists,
            "arena_corridor_match": arena_corridor_match,
            "file_pattern": file_pattern,
            "error": None,
        }

        return result

    except Exception as e:
        error_msg = f"Error loading SLEAP H5 file with SleapUtils: {str(e)}"
        print(f"❌ {error_msg}")

        # Extract basic info even if file is corrupted
        h5_arena, h5_corridor = extract_arena_corridor_from_path(h5_file_path)

        result = {
            "h5_file_path": h5_file_path,
            "h5_arena": h5_arena,
            "h5_corridor": h5_corridor,
            "video_path": None,
            "video_arena": None,
            "video_corridor": None,
            "video_exists": False,
            "arena_corridor_match": False,
            "file_pattern": file_pattern,
            "error": error_msg,
        }

        return result


def find_sleap_files_in_directory(directory_path, file_pattern):
    """Find all SLEAP H5 files matching the pattern in a directory.
    Excludes files in '/invalid' subdirectories."""
    directory = Path(directory_path)
    if not directory.exists():
        print(f"❌ Directory does not exist: {directory_path}")
        return []

    # Use glob to find files matching the pattern
    pattern_path = directory / "**" / file_pattern
    all_h5_files = glob.glob(str(pattern_path), recursive=True)

    # Filter out files in '/invalid' directories
    valid_files = []
    for file_path in all_h5_files:
        if "/invalid" not in str(file_path):
            valid_files.append(Path(file_path))
        else:
            print(f"🗑️  Excluding file in invalid directory: {file_path}")

    return valid_files


def analyze_results_by_type(results, file_type):
    """Analyze results for a specific file type."""
    type_results = [r for r in results if file_type in r["file_pattern"]]

    if not type_results:
        return None

    total_files = len(type_results)
    files_with_issues = []
    files_without_issues = []

    for result in type_results:
        if result["error"] or not result["arena_corridor_match"]:
            files_with_issues.append(result)
        else:
            files_without_issues.append(result)

    return {
        "total_files": total_files,
        "files_with_issues": files_with_issues,
        "files_without_issues": files_without_issues,
        "issue_count": len(files_with_issues),
        "success_count": len(files_without_issues),
    }


def save_mismatches_to_csv(mismatches, output_dir, identifier, file_type):
    """Save mismatch data to CSV file."""
    if not mismatches:
        return None

    # Create outputs directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # For TNT screen analysis, use dedicated subdirectory
    if "TNT_screen" in identifier or "multiprocess" in identifier:
        tnt_dir = output_path / "tnt_screen_analysis"
        tnt_dir.mkdir(exist_ok=True)
        output_path = tnt_dir

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sleap_mismatches_{file_type}_{identifier}_{timestamp}.csv"
    csv_filepath = output_path / csv_filename

    # Define CSV headers
    headers = [
        "file_number",
        "h5_filename",
        "h5_full_path",
        "h5_directory",
        "h5_arena",
        "h5_corridor",
        "video_path_in_h5",
        "video_arena",
        "video_corridor",
        "video_exists",
        "arena_corridor_match",
        "error_type",
        "error_message",
        "experiment_date",
        "experiment_session",
        "suggested_correct_h5_path",
        "file_type",
    ]

    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for i, mismatch in enumerate(mismatches, 1):
            h5_path = Path(mismatch["h5_file_path"])
            h5_filename = h5_path.name
            h5_directory = str(h5_path.parent)

            # Extract experiment info from path
            path_parts = h5_path.parts
            experiment_date = "unknown"
            experiment_session = "unknown"

            for part in path_parts:
                if any(char.isdigit() for char in part) and "_" in part:
                    date_parts = part.split("_")
                    if len(date_parts) >= 2 and date_parts[0].isdigit():
                        experiment_date = date_parts[0]
                        experiment_session = "_".join(date_parts[1:])
                        break

            # Determine error type
            if mismatch["error"]:
                error_type = "file_corruption"
                error_message = mismatch["error"]
            elif not mismatch["arena_corridor_match"]:
                error_type = "arena_corridor_mismatch"
                error_message = f"Arena/Corridor mismatch: H5 is in arena{mismatch['h5_arena']}/corridor{mismatch['h5_corridor']}, but video points to arena{mismatch['video_arena']}/corridor{mismatch['video_corridor']}"
            else:
                error_type = "other"
                error_message = "Unknown issue"

            # Suggest correct path based on video path
            suggested_path = "N/A"
            if mismatch["video_path"] and mismatch["video_arena"] and mismatch["video_corridor"]:
                video_dir = Path(mismatch["video_path"]).parent
                suggested_path = str(video_dir / h5_filename)

            row = [
                i,
                h5_filename,
                str(h5_path),
                h5_directory,
                mismatch["h5_arena"],
                mismatch["h5_corridor"],
                mismatch["video_path"],
                mismatch["video_arena"],
                mismatch["video_corridor"],
                mismatch["video_exists"],
                mismatch["arena_corridor_match"],
                error_type,
                error_message,
                experiment_date,
                experiment_session,
                suggested_path,
                file_type,
            ]
            writer.writerow(row)

    return csv_filepath


def check_multiple_file_types_multiprocess(directory_path, max_processes=None):
    """Check multiple H5 file types using multiprocessing."""
    if max_processes is None:
        max_processes = mp.cpu_count()

    print(f"🚀 Using {max_processes} processes for parallel processing")

    file_patterns = {"full_body": "*full_body*.h5", "ball": "*ball*.h5", "fly": "*fly*.h5"}

    all_results = []

    for file_type, pattern in file_patterns.items():
        print(f"\n{'='*80}")
        print(f"🔍 CHECKING {file_type.upper()} FILES")
        print(f"{'='*80}")

        # Find all files of this type
        h5_files = find_sleap_files_in_directory(directory_path, pattern)

        if not h5_files:
            print(f"📁 No {file_type} files found in {directory_path}")
            continue

        print(f"📁 {directory_path}: {len(h5_files)} {file_type} files")
        print(f"📋 Total *{pattern}* files found: {len(h5_files)}")
        print(f"🚀 Processing {len(h5_files)} files using {max_processes} processes...")

        # Prepare arguments for multiprocessing
        process_args = [(str(h5_file), pattern) for h5_file in h5_files]

        # Process files using multiprocessing
        start_time = time.time()
        with Pool(processes=max_processes) as pool:
            type_results = pool.map(process_single_h5_file, process_args)
        end_time = time.time()

        print(f"⏱️ Processing completed in {end_time - start_time:.2f} seconds")

        # Analyze results
        analysis = analyze_results_by_type(type_results, file_type)
        if analysis:
            print(f"\n📊 {file_type.upper()} SUMMARY:")
            print(f"Total *{pattern}* files checked: {analysis['total_files']}")
            print(f"Files with issues: {analysis['issue_count']}")
            print(f"Files without issues: {analysis['success_count']}")

            if analysis["issue_count"] > 0:
                print(f"\n⚠️  {analysis['issue_count']} {file_type} files have issues!")

                # Save detailed results to CSV
                identifier = f"multiprocess_{Path(directory_path).name}"
                csv_path = save_mismatches_to_csv(analysis["files_with_issues"], "outputs", identifier, file_type)
                if csv_path:
                    print(f"💾 Detailed {file_type} mismatch data saved to: {csv_path}")
                    print(
                        f"📊 CSV contains {analysis['issue_count']} problematic {file_type} files with full paths and analysis"
                    )

                # Show first few mismatches
                print(f"\n📝 MISPLACED {file_type.upper()} FILES (First 5):")
                print("=" * 50)
                for i, mismatch in enumerate(analysis["files_with_issues"][:5], 1):
                    h5_filename = Path(mismatch["h5_file_path"]).name
                    print(f"{i}. {h5_filename}")
                    print(f"   Full path: {mismatch['h5_file_path']}")
                    print(f"   File location: ({mismatch['h5_arena']}, {mismatch['h5_corridor']})")

                    if mismatch["error"]:
                        print(f"   Error: {mismatch['error']}")
                    else:
                        print(f"   Video points to: ({mismatch['video_arena']}, {mismatch['video_corridor']})")
                        print(
                            f"   Error: Arena/Corridor mismatch: H5 is in arena{mismatch['h5_arena']}/corridor{mismatch['h5_corridor']}, but video points to arena{mismatch['video_arena']}/corridor{mismatch['video_corridor']}"
                        )
                    print()
            else:
                print(f"\n✅ All {file_type} files have consistent arena/corridor numbers!")

        all_results.extend(type_results)

    return all_results


def check_directories_from_yaml_multiprocess(yaml_file, max_processes=None):
    """Process directories from YAML file using multiprocessing."""
    if max_processes is None:
        max_processes = mp.cpu_count()

    try:
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print(f"❌ Error reading YAML file: {e}")
        return []

    all_results = []
    directory_paths = []

    # Collect all directory paths
    if isinstance(data, dict):
        for key, paths in data.items():
            if isinstance(paths, list):
                directory_paths.extend(paths)
            elif isinstance(paths, str):
                directory_paths.append(paths)
    elif isinstance(data, list):
        directory_paths = data

    print(f"🎯 Found {len(directory_paths)} directories in YAML file")
    print(f"🚀 Using {max_processes} processes for parallel processing")

    # Process each directory
    for i, directory_path in enumerate(directory_paths, 1):
        print(f"\n{'='*80}")
        print(f"🏁 Processing directory {i}/{len(directory_paths)}: {Path(directory_path).name}")
        print(f"{'='*80}")

        if not Path(directory_path).exists():
            print(f"⚠️ Directory does not exist: {directory_path}")
            continue

        # Process this directory with multiprocessing
        directory_results = check_multiple_file_types_multiprocess(directory_path, max_processes)
        all_results.extend(directory_results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Check SLEAP H5 file video path consistency using multiprocessing")
    parser.add_argument(
        "--yaml-file", default="experiments_yaml/TNT_screen.yaml", help="YAML file containing directory paths"
    )
    parser.add_argument("--specific-directory", help="Check a specific directory instead of using YAML file")
    parser.add_argument("--test-cases", action="store_true", help="Run with known test cases for validation")
    parser.add_argument(
        "--max-processes", type=int, default=None, help="Maximum number of processes to use (default: CPU count)"
    )

    args = parser.parse_args()

    if args.max_processes is None:
        args.max_processes = mp.cpu_count()

    if args.test_cases:
        print("🧪 Running test cases with known examples...")
        test_directories = [
            "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231129_TNT_Fine_1_Videos_Tracked",  # Tommy's known mismatch
            "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231204_TNT_Fine_1_Videos_Tracked",  # Another example
        ]

        all_results = []
        for test_dir in test_directories:
            if Path(test_dir).exists():
                print(f"\n🔍 Testing directory: {test_dir}")
                results = check_multiple_file_types_multiprocess(test_dir, args.max_processes)
                all_results.extend(results)
            else:
                print(f"⚠️ Test directory not found: {test_dir}")

        return all_results

    elif args.specific_directory:
        print(f"🔍 Checking specific directory: {args.specific_directory}")
        return check_multiple_file_types_multiprocess(args.specific_directory, args.max_processes)

    else:
        yaml_path = Path(args.yaml_file)
        if not yaml_path.exists():
            print(f"❌ YAML file not found: {args.yaml_file}")
            return []

        print(f"📄 Loading directories from: {args.yaml_file}")
        return check_directories_from_yaml_multiprocess(args.yaml_file, args.max_processes)


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎉 Analysis complete! Processed {len(results)} H5 files total.")
    else:
        print("\n❌ No results to report.")
