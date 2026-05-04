#!/usr/bin/env python3
"""
Script to check for missing nickname videos and rename existing ones using simplified nicknames.
"""

import logging
from pathlib import Path
import pandas as pd
import re
from typing import Dict, Set, List, Tuple
import shutil

from ballpushing_utils.paths import dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "data_path": dataset(
        "Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
    ),
    "output_dir": dataset("TNT_Screen_RawGrids"),
    "mapping_csv": dataset("Region_map_250908.csv"),
    "video_suffix": "_grid.mp4",
    "backup_dir": None,  # Set to a path if you want to backup original files
}

def sanitize_filename(filename: str) -> str:
    """Remove or replace problematic characters in filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the main dataset and mapping CSV."""
    try:
        # Load main dataset
        main_data = pd.read_feather(CONFIG["data_path"])
        logger.info(f"Loaded main dataset with {len(main_data)} rows")
        
        # Load mapping CSV
        mapping_data = pd.read_csv(CONFIG["mapping_csv"])
        logger.info(f"Loaded mapping CSV with {len(mapping_data)} rows")
        
        return main_data, mapping_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def get_existing_videos() -> Dict[str, Path]:
    """Get all existing video files in the output directory."""
    output_path = Path(CONFIG["output_dir"])
    if not output_path.exists():
        logger.error(f"Output directory does not exist: {output_path}")
        return {}
    
    video_files = {}
    for video_file in output_path.glob(f"*{CONFIG['video_suffix']}"):
        # Extract the base name without suffix
        base_name = video_file.stem.replace("_grid", "")
        video_files[base_name] = video_file
    
    logger.info(f"Found {len(video_files)} existing video files")
    return video_files

def create_nickname_mapping(main_data: pd.DataFrame, mapping_data: pd.DataFrame) -> Dict[str, str]:
    """Create mapping from nicknames/genotypes to simplified nicknames."""
    nickname_to_simplified = {}
    
    # Create mapping from Nickname column in mapping CSV (correct column name)
    if 'Nickname' in mapping_data.columns and 'Simplified Nickname' in mapping_data.columns:
        for _, row in mapping_data.iterrows():
            if pd.notna(row['Nickname']) and pd.notna(row['Simplified Nickname']):
                nickname_to_simplified[row['Nickname']] = row['Simplified Nickname']
    
    # Also create mapping from Genotype column
    if 'Genotype' in mapping_data.columns and 'Simplified Nickname' in mapping_data.columns:
        for _, row in mapping_data.iterrows():
            if pd.notna(row['Genotype']) and pd.notna(row['Simplified Nickname']):
                nickname_to_simplified[row['Genotype']] = row['Simplified Nickname']
    
    # Try other potential column name variations
    for col in ['Name', 'Original_Nickname', 'Full_Nickname']:
        if col in mapping_data.columns and 'Simplified Nickname' in mapping_data.columns:
            for _, row in mapping_data.iterrows():
                if pd.notna(row[col]) and pd.notna(row['Simplified Nickname']):
                    nickname_to_simplified[row[col]] = row['Simplified Nickname']
    
    logger.info(f"Created mapping for {len(nickname_to_simplified)} nicknames")
    logger.info(f"CSV columns: {list(mapping_data.columns)}")
    return nickname_to_simplified

def find_missing_videos(main_data: pd.DataFrame, existing_videos: Dict[str, Path], nickname_mapping: Dict[str, str]) -> List[str]:
    """Find nicknames/genotypes that should have videos but don't."""
    missing_identifiers = []
    covered_rows = set()
    
    # For each row, check if there's a video for either Nickname or Genotype
    for idx, row in main_data.iterrows():
        has_video = False
        row_identifiers = []
        
        # Collect possible identifiers for this row
        if 'Nickname' in main_data.columns and pd.notna(row['Nickname']):
            row_identifiers.append(str(row['Nickname']))
        
        if 'Genotype' in main_data.columns and pd.notna(row['Genotype']):
            row_identifiers.append(str(row['Genotype']))
        
        # Check if any of these identifiers has a video (original or simplified name)
        for identifier in row_identifiers:
            sanitized_identifier = sanitize_filename(identifier)
            
            # Check original identifier
            if identifier in existing_videos or sanitized_identifier in existing_videos:
                has_video = True
                covered_rows.add(idx)
                break
            
            # Check simplified nickname if mapping exists
            if identifier in nickname_mapping:
                simplified_name = nickname_mapping[identifier]
                sanitized_simplified = sanitize_filename(simplified_name)
                
                if simplified_name in existing_videos or sanitized_simplified in existing_videos:
                    has_video = True
                    covered_rows.add(idx)
                    break
        
        # If no video found for this row, add the identifiers to missing list
        if not has_video and row_identifiers:
            missing_identifiers.extend(row_identifiers)
    
    # Get unique missing identifiers
    unique_missing = list(set(missing_identifiers))
    
    logger.info(f"Found {len(covered_rows)} rows with existing videos")
    logger.info(f"Found {len(main_data) - len(covered_rows)} rows without videos")
    logger.info(f"Found {len(unique_missing)} unique missing identifiers")
    
    return unique_missing

def rename_videos(existing_videos: Dict[str, Path], nickname_mapping: Dict[str, str]) -> Dict[str, str]:
    """Rename existing videos using simplified nicknames."""
    renamed_count = 0
    rename_log = {}
    
    if CONFIG["backup_dir"]:
        backup_path = Path(CONFIG["backup_dir"])
        backup_path.mkdir(parents=True, exist_ok=True)
    
    for original_name, video_path in existing_videos.items():
        # Try to find a mapping for this video
        simplified_name = None
        
        # First try direct mapping
        if original_name in nickname_mapping:
            simplified_name = nickname_mapping[original_name]
        else:
            # Try to find by checking if any mapping key matches
            for nickname, simplified in nickname_mapping.items():
                if sanitize_filename(nickname) == original_name:
                    simplified_name = simplified
                    break
        
        if simplified_name:
            # Create new filename
            new_filename = f"{sanitize_filename(simplified_name)}{CONFIG['video_suffix']}"
            new_path = video_path.parent / new_filename
            
            # Skip if already has the correct name
            if video_path.name == new_filename:
                logger.info(f"Video already has correct name: {video_path.name}")
                continue
            
            # Skip if target file already exists
            if new_path.exists():
                logger.warning(f"Target file already exists, skipping: {new_path}")
                continue
            
            try:
                # Backup original if requested
                if CONFIG["backup_dir"]:
                    backup_file = Path(CONFIG["backup_dir"]) / video_path.name
                    shutil.copy2(video_path, backup_file)
                    logger.info(f"Backed up: {video_path.name} -> {backup_file}")
                
                # Rename the file
                video_path.rename(new_path)
                logger.info(f"Renamed: {video_path.name} -> {new_filename}")
                rename_log[original_name] = simplified_name
                renamed_count += 1
                
            except Exception as e:
                logger.error(f"Error renaming {video_path.name}: {e}")
        else:
            logger.warning(f"No mapping found for: {original_name}")
    
    logger.info(f"Successfully renamed {renamed_count} videos")
    return rename_log

def generate_report(missing_videos: List[str], rename_log: Dict[str, str], main_data: pd.DataFrame, existing_videos: Dict[str, Path], nickname_mapping: Dict[str, str]):
    """Generate a summary report."""
    print("\n" + "="*80)
    print("VIDEO PROCESSING REPORT")
    print("="*80)
    
    # Calculate coverage statistics more accurately
    total_rows = len(main_data)
    covered_rows = 0
    
    for idx, row in main_data.iterrows():
        has_video = False
        row_identifiers = []
        
        if 'Nickname' in main_data.columns and pd.notna(row['Nickname']):
            row_identifiers.append(str(row['Nickname']))
        
        if 'Genotype' in main_data.columns and pd.notna(row['Genotype']):
            row_identifiers.append(str(row['Genotype']))
        
        # Check if any identifier has a video (original or simplified)
        for identifier in row_identifiers:
            sanitized_identifier = sanitize_filename(identifier)
            
            # Check original identifier
            if identifier in existing_videos or sanitized_identifier in existing_videos:
                has_video = True
                break
            
            # Check simplified nickname if mapping exists
            if identifier in nickname_mapping:
                simplified_name = nickname_mapping[identifier]
                sanitized_simplified = sanitize_filename(simplified_name)
                
                if simplified_name in existing_videos or sanitized_simplified in existing_videos:
                    has_video = True
                    break
        
        if has_video:
            covered_rows += 1
    
    print(f"\nCOVERAGE STATISTICS:")
    print("-" * 40)
    print(f"  Total data rows: {total_rows}")
    print(f"  Rows with videos: {covered_rows}")
    print(f"  Rows without videos: {total_rows - covered_rows}")
    print(f"  Coverage percentage: {covered_rows/total_rows*100:.1f}%")
    
    print(f"\nMISSING VIDEO IDENTIFIERS ({len(set(missing_videos))}):")
    print("-" * 40)
    if missing_videos:
        for identifier in sorted(set(missing_videos)):
            # Try to find additional info about this identifier
            identifier_data = main_data[
                (main_data.get('Nickname', '') == identifier) | 
                (main_data.get('Genotype', '') == identifier)
            ]
            if not identifier_data.empty:
                sample_count = len(identifier_data)
                # Show simplified name if available
                display_name = nickname_mapping.get(identifier, identifier)
                if display_name != identifier:
                    print(f"  - {identifier} -> {display_name} ({sample_count} samples)")
                else:
                    print(f"  - {identifier} ({sample_count} samples)")
            else:
                print(f"  - {identifier}")
    else:
        print("  🎉 All videos are now available!")
    
    print(f"\nRENAMED VIDEOS ({len(rename_log)}):")
    print("-" * 40)
    if rename_log:
        for original, simplified in sorted(rename_log.items()):
            print(f"  {original} -> {simplified}")
    else:
        print("  No videos were renamed.")
    
    print("\n" + "="*80)

def main():
    """Main execution function."""
    try:
        logger.info("Starting video check and rename process...")
        
        # Load data
        main_data, mapping_data = load_data()
        
        # Get existing videos
        existing_videos = get_existing_videos()
        if not existing_videos:
            logger.error("No existing videos found. Exiting.")
            return
        
        # Create nickname mapping
        nickname_mapping = create_nickname_mapping(main_data, mapping_data)
        if not nickname_mapping:
            logger.warning("No nickname mappings found in CSV file")
        
        # Find missing videos
        missing_videos = find_missing_videos(main_data, existing_videos, nickname_mapping)
        
        # Rename existing videos
        rename_log = rename_videos(existing_videos, nickname_mapping)
        
        # Generate report
        generate_report(missing_videos, rename_log, main_data, existing_videos, nickname_mapping)
        
        # Save missing videos list for reference
        if missing_videos:
            missing_file = Path(CONFIG["output_dir"]) / "missing_videos.txt"
            with open(missing_file, 'w') as f:
                f.write("Missing Videos:\n")
                f.write("================\n\n")
                for nickname in sorted(missing_videos):
                    f.write(f"{nickname}\n")
            logger.info(f"Missing videos list saved to: {missing_file}")
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()