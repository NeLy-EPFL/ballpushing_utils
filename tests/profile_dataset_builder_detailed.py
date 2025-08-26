#!/usr/bin/env python3

"""
Dataset Builder with Profiling

Modified version of dataset_builder.py with detailed timing and profiling capabilities
to identify performance bottlenecks in ballpushing metrics computation.
"""

import sys
import time
import cProfile
import pstats
import io
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the original dataset_builder module
import dataset_builder
import Ballpushing_utils


class DatasetBuilderProfiler:
    """Add profiling capabilities to dataset builder operations."""

    def __init__(self):
        self.timings = {}
        self.detailed_profiles = {}

    def time_operation(self, operation_name, func, *args, **kwargs):
        """Time an operation and optionally profile it."""
        print(f"⏱️  Starting {operation_name}...")
        start_time = time.time()

        # Run with profiling for detailed analysis
        if operation_name in ["summary_dataset_creation", "ballpushing_metrics_computation"]:
            pr = cProfile.Profile()
            pr.enable()

            result = func(*args, **kwargs)

            pr.disable()

            # Store detailed profile
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
            ps.print_stats(30)
            self.detailed_profiles[operation_name] = s.getvalue()

        else:
            result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        self.timings[operation_name] = elapsed

        print(f"✅ {operation_name} completed in {elapsed:.2f}s")
        return result

    def profile_experiment_processing(self, folder, metrics, output_data):
        """Profile the experiment processing with detailed breakdowns."""
        print(f"\n🔍 PROFILING EXPERIMENT: {folder.name}")
        print("=" * 50)

        # Time experiment creation
        def create_experiment():
            return Ballpushing_utils.Experiment(folder)

        experiment = self.time_operation("experiment_creation", create_experiment)

        # Process each metric separately
        for metric in metrics:
            print(f"\n📊 Processing metric: {metric}")

            # Time dataset creation
            def create_dataset():
                return Ballpushing_utils.Dataset(experiment, dataset_type=metric)

            dataset = self.time_operation(f"{metric}_dataset_creation", create_dataset)

            # Additional profiling for summary (ballpushing) metrics
            if metric == "summary" and hasattr(experiment, "flies") and experiment.flies:
                self.profile_ballpushing_details(experiment.flies[0])

            # Save dataset if data exists
            if dataset.data is not None and not dataset.data.empty:
                dataset_path = output_data / metric / f"{folder.name}_{metric}.feather"
                dataset_path.parent.mkdir(parents=True, exist_ok=True)

                def save_dataset():
                    dataset.data.to_feather(dataset_path)

                self.time_operation(f"{metric}_save", save_dataset)

                print(f"   💾 Saved {len(dataset.data)} rows to {dataset_path}")
            else:
                print(f"   ⚠️  No data generated for {metric}")

        # Clear caches and cleanup
        if hasattr(experiment, "flies"):
            for fly in experiment.flies:
                if hasattr(fly, "clear_caches"):
                    fly.clear_caches()

        print(f"\n📈 TIMING SUMMARY for {folder.name}:")
        for operation, timing in self.timings.items():
            print(f"   {operation}: {timing:.2f}s")

    def profile_ballpushing_details(self, fly):
        """Detailed profiling of ballpushing metrics computation."""
        print(f"\n🎯 DETAILED BALLPUSHING PROFILING: {fly.metadata.name}")

        # Profile tracking data access
        def get_tracking_data():
            return fly.tracking_data

        tracking_data = self.time_operation("tracking_data_access", get_tracking_data)

        # Profile ballpushing metrics object creation
        def create_ballpushing_metrics():
            return Ballpushing_utils.BallPushingMetrics(tracking_data)

        ballpushing_obj = self.time_operation("ballpushing_metrics_computation", create_ballpushing_metrics)

        # Profile individual metric computation steps
        if hasattr(ballpushing_obj, "metrics"):
            print(f"   📊 Generated {len(ballpushing_obj.metrics)} metric entries")

        # Profile accessing cached results
        def get_event_summaries():
            return fly.event_summaries

        event_summaries = self.time_operation("event_summaries_access", get_event_summaries)

        if event_summaries:
            print(f"   📋 Event summaries keys: {list(event_summaries.keys())[:5]}...")

    def print_detailed_profiles(self):
        """Print detailed cProfile results for slow operations."""
        print(f"\n🔬 DETAILED FUNCTION PROFILING")
        print("=" * 50)

        for operation, profile_output in self.detailed_profiles.items():
            print(f"\n📈 {operation.upper()}:")
            print("-" * 30)
            print(profile_output[:1500])  # First 1500 characters

    def save_profiling_report(self, output_path):
        """Save profiling results to a file."""
        report_path = output_path / f"dataset_builder_profile_{int(time.time())}.txt"

        with open(report_path, "w") as f:
            f.write("DATASET BUILDER PROFILING REPORT\n")
            f.write("=" * 40 + "\n\n")

            f.write("TIMING RESULTS:\n")
            f.write("-" * 20 + "\n")
            for operation, timing in self.timings.items():
                f.write(f"{operation}: {timing:.2f}s\n")

            f.write("\nDETAILED PROFILES:\n")
            f.write("-" * 20 + "\n")
            for operation, profile_output in self.detailed_profiles.items():
                f.write(f"\n{operation}:\n")
                f.write(profile_output)

        print(f"\n📋 Full profiling report saved to: {report_path}")


def profile_single_experiment(experiment_path, metrics=None):
    """Profile a single experiment with specified metrics."""
    if metrics is None:
        metrics = ["standardized_contacts", "summary"]

    experiment_path = Path(experiment_path)
    output_data = Path("/tmp/profiling_output")

    profiler = DatasetBuilderProfiler()

    print(f"🚀 PROFILING DATASET BUILDER")
    print(f"📁 Experiment: {experiment_path}")
    print(f"📊 Metrics: {metrics}")

    try:
        profiler.profile_experiment_processing(experiment_path, metrics, output_data)
        profiler.print_detailed_profiles()
        profiler.save_profiling_report(output_data)

        # Performance analysis
        print(f"\n🎯 PERFORMANCE ANALYSIS")
        print("=" * 30)

        standardized_time = profiler.timings.get("standardized_contacts_dataset_creation", 0)
        summary_time = profiler.timings.get("summary_dataset_creation", 0)

        if standardized_time > 0 and summary_time > 0:
            ratio = summary_time / standardized_time
            print(f"📈 Summary vs Standardized: {ratio:.1f}x slower")
            print(f"⏱️  Time difference: {summary_time - standardized_time:.2f}s")

            if ratio > 2:
                print(f"⚠️  SIGNIFICANT SLOWDOWN DETECTED!")
                print(f"   Check the detailed profiling report for bottlenecks.")

        ballpushing_time = profiler.timings.get("ballpushing_metrics_computation", 0)
        tracking_time = profiler.timings.get("tracking_data_access", 0)

        if ballpushing_time > 0:
            print(f"🎯 BallPushing metrics computation: {ballpushing_time:.2f}s")
            if tracking_time > 0:
                print(f"📊 Tracking data access: {tracking_time:.2f}s")
                print(f"🧮 Pure computation time: {ballpushing_time - tracking_time:.2f}s")

    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_dataset_builder_detailed.py <experiment_path> [metrics]")
        print("Example: python profile_dataset_builder_detailed.py /path/to/experiment")
        print("         python profile_dataset_builder_detailed.py /path/to/experiment summary")
        sys.exit(1)

    experiment_path = sys.argv[1]

    # Parse metrics from command line
    if len(sys.argv) > 2:
        metrics = sys.argv[2:]
    else:
        metrics = ["standardized_contacts", "summary"]

    profile_single_experiment(experiment_path, metrics)
