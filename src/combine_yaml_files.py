#!/usr/bin/env python3
"""
Script to combine multiple YAML files containing directory lists into a single YAML file.

Each input YAML file should have the structure:
directories:
- /path/to/directory1
- /path/to/directory2
- ...

The script will combine all directory lists into a single YAML file with the same structure.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set
import yaml


def load_yaml_file(file_path: Path) -> List[str]:
    """
    Load a YAML file and extract the directories list.

    Args:
        file_path: Path to the YAML file

    Returns:
        List of directory paths

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
        KeyError: If the file doesn't contain a 'directories' key
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict in {file_path}, got {type(data)}")

        if "directories" not in data:
            raise KeyError(f"'directories' key not found in {file_path}")

        directories = data["directories"]
        if not isinstance(directories, list):
            raise ValueError(f"Expected list for 'directories' in {file_path}, got {type(directories)}")

        return directories

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in {file_path}: {e}")
        raise
    except (KeyError, ValueError) as e:
        print(f"Error: {e}")
        raise


def combine_yaml_files(input_files: List[Path], output_file: Path, remove_duplicates: bool = True) -> None:
    """
    Combine multiple YAML files with directory lists into a single file.

    Args:
        input_files: List of paths to input YAML files
        output_file: Path to output YAML file
        remove_duplicates: Whether to remove duplicate directory paths
    """
    all_directories = []

    print(f"Processing {len(input_files)} input files...")

    for input_file in input_files:
        print(f"  Reading {input_file}...")
        try:
            directories = load_yaml_file(input_file)
            all_directories.extend(directories)
            print(f"    Found {len(directories)} directories")
        except Exception as e:
            print(f"    Skipping {input_file} due to error: {e}")
            continue

    # Remove duplicates if requested
    if remove_duplicates:
        original_count = len(all_directories)
        all_directories = list(dict.fromkeys(all_directories))  # Preserves order
        duplicates_removed = original_count - len(all_directories)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate directories")

    # Create output data structure
    output_data = {"directories": all_directories}

    # Write to output file
    print(f"Writing {len(all_directories)} directories to {output_file}...")
    try:
        # Create parent directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

        print(f"Successfully created {output_file}")

    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        raise


def main():
    """Main function to parse arguments and combine YAML files."""
    parser = argparse.ArgumentParser(
        description="Combine multiple YAML files containing directory lists into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python combine_yaml_files.py -i file1.yaml file2.yaml file3.yaml -o combined.yaml
  python combine_yaml_files.py -i outputs/*.yaml -o combined_all.yaml
  python combine_yaml_files.py -i outputs/Empty_flies.yaml outputs/ExR1_flies.yaml -o combined.yaml --keep-duplicates
        """,
    )

    parser.add_argument("-i", "--input", nargs="+", required=True, help="Input YAML files to combine")

    parser.add_argument("-o", "--output", required=True, help="Output YAML file path")

    parser.add_argument(
        "--keep-duplicates", action="store_true", help="Keep duplicate directory paths (default: remove duplicates)"
    )

    args = parser.parse_args()

    # Convert string paths to Path objects
    input_files = [Path(f) for f in args.input]
    output_file = Path(args.output)

    # Validate input files exist
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        print("Error: The following input files do not exist:")
        for f in missing_files:
            print(f"  {f}")
        sys.exit(1)

    # Check if output file already exists
    if output_file.exists():
        response = input(f"Output file {output_file} already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        combine_yaml_files(input_files=input_files, output_file=output_file, remove_duplicates=not args.keep_duplicates)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
