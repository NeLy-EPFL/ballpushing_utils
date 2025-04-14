from utils_behavior import HoloviewsTemplates

# Load a dataset

import pandas as pd

dataset = pd.read_feather("/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250414_summary_test_folder_Data/summary/231130_TNT_Fine_2_Videos_Tracked_summary.feather")

# Configure which are the metrics and what are the metadata

metrics = [
