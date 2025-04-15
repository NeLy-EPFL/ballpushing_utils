from utils_behavior import HoloviewsTemplates

# Load a dataset

import pandas as pd
import holoviews as hv

hv.extension("bokeh")

dataset = pd.read_feather(
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250414_summary_test_folder_Data/summary/231130_TNT_Fine_2_Videos_Tracked_summary.feather"
)

# Configure which are the metrics and what are the metadata

metric_names = [
    "nb_events",
    "max_event",
    "max_event_time",
    "max_distance",
    "final_event",
    "final_event_time",
    "nb_significant_events",
    "significant_ratio",
    "first_significant_event",
    "first_significant_event_time",
    "major_event",
    "major_event_time",
    "major_event_first",
    "insight_effect",
    "insight_effect_log",
    "cumulated_breaks_duration",
    "chamber_time",
    "chamber_ratio",
    "pushed",
    "pulled",
    "pulling_ratio",
    "success_direction",
    "interaction_proportion",
    "interaction_persistence",
    "distance_moved",
    "distance_ratio",
    "exit_time",
    "chamber_exit_time",
    "number_of_pauses",
    "total_pause_duration",
    "learning_slope",
    "learning_slope_r2",
    "logistic_L",
    "logistic_k",
    "logistic_t0",
    "logistic_r2",
    "avg_displacement_after_success",
    "avg_displacement_after_failure",
    "influence_ratio",
    "normalized_velocity",
    "velocity_during_interactions",
    "velocity_trend",
]

# Set the metadata to be whatever columns are not in the metric names

metadata_names = [col for col in dataset.columns if col not in metric_names]

print(f"Metrics: {metric_names}")
print(f"Metadata: {metadata_names}")

# Test with one metric

test_metric = "nb_events"

plot = HoloviewsTemplates.jitter_boxplot(
    data=dataset,
    metric=test_metric,
    kdims="Simplified Nickname",
    control="TNTxPR",
    hline="boxplot",
    groupby="Brain region",
    colorby="Brain region",
    render="grouped",
    metadata=metadata_names,
)

# Save the plot to a file

hv.save(plot, "test_plot.html", fmt="html")
