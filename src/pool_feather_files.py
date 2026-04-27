#!/usr/bin/env python3
"""
Pool individual feather files from the TNT Full dataset summary directory.
This replicates what dataset_builder.py does when creating pooled datasets.
"""

import pandas as pd
from pathlib import Path
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Default base directory (kept for backward compatibility)
BASE_DIR_DEFAULT = Path("/mnt/upramdya_data/MD/F1_Tracks/Datasets/260102_20_F1_coordinates_F1_TNT_Full_Data")


def pool_directory(base_dir, metrics=None, chunk_size=5, overwrite=False):
    """Pool feather files in `base_dir`. Creates pooled_<metric>.feather files in each metric subdirectory."""
    if metrics is None:
        metrics = ["summary"]

    base_dir = Path(base_dir)

    for metric in metrics:
        metric_dir = base_dir / metric

        if not metric_dir.exists():
            logging.warning(f"Directory not found: {metric_dir}")
            continue

        # Find all individual feather files (exclude pooled files)
        individual_files = [f for f in metric_dir.glob("*.feather") if not f.name.startswith("pooled_")]

        if not individual_files:
            logging.warning(f"No individual files found in {metric_dir}")
            continue

        logging.info(f"\n{'='*80}")
        logging.info(f"POOLING {metric.upper()}")
        logging.info(f"{'='*80}")
        logging.info(f"Found {len(individual_files)} individual files")

        # Output pooled file path
        pooled_path = metric_dir / f"pooled_{metric}.feather"

        if pooled_path.exists():
            if not overwrite:
                user_input = input(f"Pooled file already exists: {pooled_path}\nOverwrite? (y/n): ").strip().lower()
                if user_input != "y":
                    logging.info("Skipping...")
                    continue
            else:
                logging.info(f"Overwriting existing pooled file: {pooled_path}")

        try:
            # Use chunking to avoid loading all files at once
            total_chunks = (len(individual_files) + chunk_size - 1) // chunk_size

            first_chunk = True
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(individual_files))
                chunk_files = individual_files[start_idx:end_idx]

                logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_files)} files)")

                chunk_dfs = []
                for f in chunk_files:
                    try:
                        df = pd.read_feather(f)
                        chunk_dfs.append(df)
                        logging.info(f"  ✓ Loaded {f.name} ({len(df)} rows)")
                    except Exception as e:
                        logging.warning(f"  ✗ Failed to read {f.name}: {str(e)}")

                if not chunk_dfs:
                    logging.warning(f"No valid data in chunk {chunk_idx + 1}")
                    continue

                chunk_df = pd.concat(chunk_dfs, ignore_index=True)

                if first_chunk:
                    # First chunk: create the file
                    chunk_df.to_feather(pooled_path)
                    first_chunk = False
                    logging.info(f"  Created pooled file with {len(chunk_df)} rows")
                else:
                    # Subsequent chunks: append to existing
                    existing_df = pd.read_feather(pooled_path)
                    combined_df = pd.concat([existing_df, chunk_df], ignore_index=True)
                    combined_df.to_feather(pooled_path)
                    logging.info(f"  Appended {len(chunk_df)} rows (total: {len(combined_df)} rows)")
                    del existing_df, combined_df

                del chunk_df, chunk_dfs
                gc.collect()

            # Final verification
            if pooled_path.exists():
                final_df = pd.read_feather(pooled_path)
                logging.info(f"\n✓ Successfully created: {pooled_path}")
                logging.info(f"  Total rows: {len(final_df)}")
                logging.info(f"  Columns: {len(final_df.columns)}")

                # Show sample
                if "fly" in final_df.columns:
                    logging.info(f"  Unique flies: {final_df['fly'].nunique()}")
                if "ball_identity" in final_df.columns:
                    logging.info(f"  Ball identities: {final_df['ball_identity'].value_counts().to_dict()}")
            else:
                logging.error(f"✗ Failed to create pooled file: {pooled_path}")

        except Exception as e:
            logging.error(f"Error pooling {metric}: {str(e)}")
            import traceback

            traceback.print_exc()

    logging.info(f"\n{'='*80}")
    logging.info("POOLING COMPLETE")
    logging.info(f"{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pool feather files in a dataset directory.")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=str(BASE_DIR_DEFAULT),
        help="Base dataset directory containing metric subfolders (default from script)",
    )
    parser.add_argument("--chunk-size", type=int, default=5, help="Number of files to process per chunk")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="List of metric subdirectories to pool (default: ['summary'])",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing pooled files without prompting")

    args = parser.parse_args()

    pool_directory(args.base_dir, metrics=args.metrics, chunk_size=args.chunk_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
