#!/usr/bin/env python3

"""
Dataset Builder Profiler

This script profiles the dataset builder to identify performance bottlenecks,
specifically comparing standardized_contacts vs summary (ballpushing metrics) processing.
"""

import sys
import cProfile
import pstats
import io
import time
import psutil
import os
import gc
from pathlib import Path
from datetime import datetime
import threading
import pandas as pd

# Add the src directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

import Ballpushing_utils
from Ballpushing_utils import config


class PerformanceProfiler:
    """Profile dataset building performance with detailed timing and memory tracking."""

    def __init__(self):
        self.process = psutil.Process()
        self.timings = {}
        self.memory_usage = {}
        self.start_time = None

    def start_timing(self, operation):
        """Start timing an operation."""
        self.start_time = time.time()
        gc.collect()  # Clean up before measurement
        self.memory_usage[f"{operation}_start"] = self.process.memory_info().rss / 1024 / 1024

    def end_timing(self, operation):
        """End timing an operation and record results."""
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        self.timings[operation] = elapsed
        self.memory_usage[f"{operation}_end"] = self.process.memory_info().rss / 1024 / 1024
        self.memory_usage[f"{operation}_delta"] = (
            self.memory_usage[f"{operation}_end"] - self.memory_usage[f"{operation}_start"]
        )

        print(f"✓ {operation}: {elapsed:.2f}s, Memory: {self.memory_usage[f'{operation}_delta']:+.1f}MB")
        self.start_time = None

    def profile_experiment_creation(self, experiment_path):
        """Profile the creation of an Experiment object."""
        print(f"\n=== Profiling Experiment Creation: {experiment_path.name} ===")

        self.start_timing("experiment_creation")
        experiment = Ballpushing_utils.Experiment(experiment_path)
        self.end_timing("experiment_creation")

        print(f"  - Number of flies: {len(experiment.flies) if hasattr(experiment, 'flies') else 'Unknown'}")

        return experiment

    def profile_dataset_creation(self, experiment, dataset_type):
        """Profile the creation of a specific dataset type."""
        print(f"\n=== Profiling Dataset Creation: {dataset_type} ===")

        # Profile dataset object creation
        self.start_timing(f"{dataset_type}_dataset_creation")
        dataset = Ballpushing_utils.Dataset(experiment, dataset_type=dataset_type)
        self.end_timing(f"{dataset_type}_dataset_creation")

        # Check if data was actually generated
        if dataset.data is not None and not dataset.data.empty:
            print(f"  - Dataset rows: {len(dataset.data)}")
            print(f"  - Dataset columns: {len(dataset.data.columns)}")
            print(f"  - Dataset memory: {dataset.data.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
        else:
            print(f"  - No data generated")

        return dataset

    def profile_fly_processing(self, experiment, dataset_type):
        """Profile individual fly processing for deeper analysis."""
        print(f"\n=== Profiling Individual Fly Processing: {dataset_type} ===")

        if not hasattr(experiment, "flies") or not experiment.flies:
            print("  - No flies to process")
            return

        # Test with first fly only
        fly = experiment.flies[0]
        print(f"  - Testing with fly: {fly.metadata.name}")

        # Profile tracking data access
        self.start_timing(f"{dataset_type}_tracking_data")
        tracking_data = fly.tracking_data
        self.end_timing(f"{dataset_type}_tracking_data")

        if dataset_type == "summary":
            # Profile ballpushing metrics computation
            self.start_timing(f"{dataset_type}_ballpushing_metrics")
            event_summaries = fly.event_summaries
            self.end_timing(f"{dataset_type}_ballpushing_metrics")

            print(f"    - Event summaries keys: {list(event_summaries.keys()) if event_summaries else 'None'}")

        elif dataset_type == "standardized_contacts":
            # Profile skeleton metrics computation
            self.start_timing(f"{dataset_type}_skeleton_metrics")
            skeleton_metrics = fly.skeleton_metrics
            self.end_timing(f"{dataset_type}_skeleton_metrics")

            if skeleton_metrics:
                print(f"    - Skeleton metrics available")
            else:
                print(f"    - No skeleton metrics")

    def profile_with_cprofile(self, func, *args, **kwargs):
        """Run a function with cProfile and return the stats."""
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions

        return result, s.getvalue()

    def save_profile_report(self, output_path, additional_info=""):
        """Save the profiling report to a file."""
        report_path = output_path / f"profile_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_path, "w") as f:
            f.write("DATASET BUILDER PERFORMANCE PROFILE REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")

            if additional_info:
                f.write(f"Additional Info:\n{additional_info}\n\n")

            f.write("TIMING RESULTS:\n")
            f.write("-" * 20 + "\n")
            for operation, timing in self.timings.items():
                f.write(f"{operation}: {timing:.2f}s\n")

            f.write("\nMEMORY USAGE:\n")
            f.write("-" * 20 + "\n")
            for operation, memory in self.memory_usage.items():
                if "_delta" in operation:
                    f.write(f"{operation}: {memory:+.1f}MB\n")

        print(f"\n📊 Profile report saved to: {report_path}")
        return report_path


def compare_metrics_performance(experiment_path):
    """Compare performance between standardized_contacts and summary metrics."""

    profiler = PerformanceProfiler()

    print("🔍 DATASET BUILDER PERFORMANCE ANALYSIS")
    print("=" * 50)

    # Test experiment creation
    experiment = profiler.profile_experiment_creation(experiment_path)

    # Test both dataset types
    datasets = {}

    for dataset_type in ["standardized_contacts", "summary"]:
        print(f"\n🎯 TESTING DATASET TYPE: {dataset_type}")
        print("-" * 40)

        # Profile dataset creation
        dataset = profiler.profile_dataset_creation(experiment, dataset_type)
        datasets[dataset_type] = dataset

        # Profile individual fly processing
        profiler.profile_fly_processing(experiment, dataset_type)

        # Clear caches between tests
        if hasattr(experiment, "flies"):
            for fly in experiment.flies:
                if hasattr(fly, "clear_caches"):
                    fly.clear_caches()
        gc.collect()

    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON")
    print("-" * 30)

    standardized_time = profiler.timings.get("standardized_contacts_dataset_creation", 0)
    summary_time = profiler.timings.get("summary_dataset_creation", 0)

    if standardized_time > 0 and summary_time > 0:
        ratio = summary_time / standardized_time
        print(f"Summary vs Standardized ratio: {ratio:.1f}x slower")
        print(f"Time difference: {summary_time - standardized_time:.2f}s")

    # Save detailed report
    output_dir = Path("/tmp")
    profiler.save_profile_report(output_dir, f"Experiment: {experiment_path.name}")

    return profiler


def detailed_ballpushing_profiling(experiment_path):
    """Perform detailed profiling specifically on ballpushing metrics computation."""

    print(f"\n🔬 DETAILED BALLPUSHING METRICS PROFILING")
    print("=" * 50)

    # Create experiment
    experiment = Ballpushing_utils.Experiment(experiment_path)

    if not hasattr(experiment, "flies") or not experiment.flies:
        print("No flies available for detailed profiling")
        return

    fly = experiment.flies[0]
    print(f"Profiling fly: {fly.metadata.name}")

    profiler = PerformanceProfiler()

    # Profile step by step
    print("\n1. Tracking Data Access:")
    profiler.start_timing("tracking_data_access")
    tracking_data = fly.tracking_data
    profiler.end_timing("tracking_data_access")

    print("\n2. BallPushingMetrics Object Creation:")
    profiler.start_timing("ballpushing_object_creation")
    ballpushing_metrics = Ballpushing_utils.BallPushingMetrics(tracking_data)
    profiler.end_timing("ballpushing_object_creation")

    print("\n3. Metrics Computation:")
    profiler.start_timing("metrics_computation")
    metrics = ballpushing_metrics.metrics
    profiler.end_timing("metrics_computation")

    print(f"   - Number of metrics computed: {len(metrics) if metrics else 0}")

    # Profile with cProfile for detailed function analysis
    print("\n4. Detailed Function Profiling:")

    def create_ballpushing_metrics():
        return Ballpushing_utils.BallPushingMetrics(tracking_data)

    _, profile_output = profiler.profile_with_cprofile(create_ballpushing_metrics)
    print("Top time-consuming functions:")
    print(profile_output[:2000])  # First 2000 characters

    return profiler


if __name__ == "__main__":
    print("🚀 Starting Dataset Builder Performance Analysis")

    # Find a test experiment
    test_experiment_paths = [
        Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos"),
        Path("/home/matthias/ballpushing_utils"),
    ]

    experiment_path = None
    for base_path in test_experiment_paths:
        if base_path.exists():
            # Look for experiment directories
            for subdir in base_path.iterdir():
                if subdir.is_dir() and (subdir / "Metadata.json").exists():
                    experiment_path = subdir
                    break
            if experiment_path:
                break

    if not experiment_path:
        print("❌ No test experiment found. Please specify an experiment path.")
        print("Usage: python profile_dataset_builder.py [experiment_path]")
        if len(sys.argv) > 1:
            experiment_path = Path(sys.argv[1])
            if not experiment_path.exists():
                print(f"❌ Specified path does not exist: {experiment_path}")
                sys.exit(1)
        else:
            sys.exit(1)

    print(f"📁 Using test experiment: {experiment_path}")

    try:
        # Run performance comparison
        profiler = compare_metrics_performance(experiment_path)

        # Run detailed ballpushing profiling
        detailed_profiler = detailed_ballpushing_profiling(experiment_path)

        print(f"\n✅ PROFILING COMPLETE!")
        print(f"Check the generated profile reports for detailed analysis.")

    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback

        traceback.print_exc()
