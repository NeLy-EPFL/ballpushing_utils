import numpy as np
import pandas as pd
from typing import Any
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from utils_behavior import Processing
from functools import lru_cache


class BallPushingMetrics:
    def __init__(self, tracking_data, compute_metrics_on_init=True):

        self.tracking_data = tracking_data
        self.fly = tracking_data.fly

        # Cache for expensive computations
        self._cached_computations = {}

        # Cache for skeleton-related data to avoid repeated loading
        self._skeleton_metrics = None
        self._skeleton_data_cache = {}
        self._ball_data_cache = {}

        # Iterate over the indices of the flytrack objects list
        # self.chamber_exit_times = {
        #     fly_idx: self.get_chamber_exit_time(fly_idx, 0)
        #     for fly_idx in range(len(self.tracking_data.flytrack.objects))
        # }

        if self.fly.config.debugging:
            print(f"Chamber exit time: {self.tracking_data.chamber_exit_time}")
            print(f"F1 exit time: {self.tracking_data.f1_exit_time}")

        self.metrics = {}
        if compute_metrics_on_init:
            self.compute_metrics()

    def is_metric_enabled(self, metric_name):
        """
        Check if a metric should be computed based on the enabled_metrics configuration.

        Parameters
        ----------
        metric_name : str
            Name of the metric to check.

        Returns
        -------
        bool
            True if the metric should be computed, False otherwise.
        """
        enabled_metrics = self.fly.config.enabled_metrics

        # If enabled_metrics is None, compute all metrics
        if enabled_metrics is None:
            return True

        # Check if the metric is in the enabled list
        return metric_name in enabled_metrics

    def clear_caches(self):
        """
        Clear all caches to free up memory. This should be called between processing different experiments.
        """
        # Clear instance-level caches
        self._cached_computations.clear()
        self._skeleton_data_cache.clear()
        self._ball_data_cache.clear()

        # Clear median coordinates cache if it exists
        if hasattr(self, "_median_coords_cache"):
            self._median_coords_cache.clear()

        # Reset skeleton metrics to None so it gets recreated for new experiment
        self._skeleton_metrics = None

        # Clear LRU caches for all methods that use them
        lru_cached_methods = [
            "get_max_event",
            "get_significant_events",
            "get_major_event",
            "compute_learning_slope",
            "compute_logistic_features",
        ]

        for method_name in lru_cached_methods:
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                if hasattr(method, "cache_clear"):
                    method.cache_clear()

        # Force garbage collection to free memory
        import gc

        gc.collect()

    def _get_skeleton_metrics(self):
        """
        Get or create SkeletonMetrics object with caching.
        Only creates the object when skeleton-based metrics are needed.
        """
        if self._skeleton_metrics is None:
            try:
                from .skeleton_metrics import SkeletonMetrics

                self._skeleton_metrics = SkeletonMetrics(self.fly)
            except ImportError:
                if self.fly.config.debugging:
                    print("Could not import SkeletonMetrics")
                return None
            except Exception as e:
                if self.fly.config.debugging:
                    print(f"Error creating SkeletonMetrics: {e}")
                return None
        return self._skeleton_metrics

    def _get_skeleton_data(self, fly_idx):
        """Get skeleton data with caching. Returns None if missing."""
        if fly_idx in self._skeleton_data_cache:
            return self._skeleton_data_cache[fly_idx]
        # Check for missing skeletontrack or objects
        skeletontrack = getattr(self.tracking_data, "skeletontrack", None)
        if skeletontrack is None or not hasattr(skeletontrack, "objects") or skeletontrack.objects is None:
            if self.fly.config.debugging:
                print(f"No skeletontrack or objects for fly_idx {fly_idx}")
            self._skeleton_data_cache[fly_idx] = None
            return None
        try:
            self._skeleton_data_cache[fly_idx] = skeletontrack.objects[fly_idx].dataset
        except (IndexError, AttributeError, TypeError) as e:
            if self.fly.config.debugging:
                print(f"Error accessing skeleton data for fly_idx {fly_idx}: {e}")
            self._skeleton_data_cache[fly_idx] = None
        return self._skeleton_data_cache[fly_idx]

    def _get_ball_data(self, ball_idx):
        """Get ball data with caching and pre-computed rolling medians."""
        if ball_idx not in self._ball_data_cache:
            data = self.tracking_data.balltrack.objects[ball_idx].dataset.copy()
            # Pre-compute common rolling medians to speed up coordinate calculations
            for keypoint in ["centre"]:
                data[f"x_{keypoint}_rolling_median"] = data[f"x_{keypoint}"].rolling(window=10, center=True).median()
                data[f"y_{keypoint}_rolling_median"] = data[f"y_{keypoint}"].rolling(window=10, center=True).median()
            self._ball_data_cache[ball_idx] = data
        return self._ball_data_cache[ball_idx]

    def _needs_skeleton_metrics(self):
        """Check if any skeleton-based metrics are enabled."""
        skeleton_metrics = [
            "head_pushing_ratio",
            "leg_visibility_ratio",
            "flailing",
            "fraction_not_facing_ball",
            "median_head_ball_distance",
            "mean_head_ball_distance",
        ]
        return any(self.is_metric_enabled(metric) for metric in skeleton_metrics)

    def _compute_skeleton_metrics_batch(self, fly_idx, ball_idx):
        """
        Compute all skeleton-based metrics in a batch to optimize data loading.
        This loads skeleton data and skeleton metrics objects once and computes all needed metrics.
        """
        results = {}

        def safe_call(method, *args, default=None, **kwargs):
            """Local safe_call helper for batch processing."""
            try:
                return method(*args, **kwargs)
            except Exception as e:
                if self.fly.config.debugging:
                    print(f"Error in {method.__name__}: {e}")
                return default

        # Pre-load skeleton data once
        skeleton_data = self._get_skeleton_data(fly_idx)
        ball_data = self._get_ball_data(ball_idx)

        # Get skeleton metrics once
        skeleton_metrics = self._get_skeleton_metrics()

        # If skeleton data is missing, return default values for all skeleton-based metrics
        if skeleton_data is None or skeleton_metrics is None:
            if self.is_metric_enabled("fraction_not_facing_ball"):
                results["fraction_not_facing_ball"] = np.nan
            if self.is_metric_enabled("flailing"):
                results["flailing"] = np.nan
            if self.is_metric_enabled("head_pushing_ratio"):
                results["head_pushing_ratio"] = np.nan
            if self.is_metric_enabled("leg_visibility_ratio"):
                results["leg_visibility_ratio"] = np.nan
            return results

        # Compute individual metrics only if enabled
        if self.is_metric_enabled("fraction_not_facing_ball"):
            results["fraction_not_facing_ball"] = safe_call(
                self.compute_fraction_not_facing_ball, fly_idx, ball_idx, default=0.0
            )
        else:
            results["fraction_not_facing_ball"] = 0.0

        if self.is_metric_enabled("flailing"):
            results["flailing"] = safe_call(self.compute_flailing, fly_idx, ball_idx, default=0.0)
        else:
            results["flailing"] = 0.0

        if self.is_metric_enabled("head_pushing_ratio"):
            results["head_pushing_ratio"] = safe_call(self.compute_head_pushing_ratio, fly_idx, ball_idx, default=0.5)
        else:
            results["head_pushing_ratio"] = 0.5

        if self.is_metric_enabled("leg_visibility_ratio"):
            results["leg_visibility_ratio"] = safe_call(
                self.compute_leg_visibility_ratio, fly_idx, ball_idx, default=0.0
            )
        else:
            results["leg_visibility_ratio"] = 0.0

        return results

    def compute_metrics(self):
        """
        Compute and store various metrics for each pair of fly and ball.
        """

        def safe_call(method, *args, default: Any = np.nan, **kwargs) -> Any:
            """
            Helper function to safely call a method and handle exceptions.

            Parameters
            ----------
            method : callable
                The method to call.
            *args : tuple
                Positional arguments to pass to the method.
            default : Any, optional
                Default value to return in case of an exception (default is np.nan).
            **kwargs : dict
                Keyword arguments to pass to the method.

            Returns
            -------
            Any
                The result of the method call, or the default value if an exception occurs.
            """
            try:
                return method(*args, **kwargs)
            except Exception as e:
                if self.fly.config.debugging:
                    print(f"Error in {method.__name__}: {e}")
                return default

        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                # Generate key with ball identity if available
                ball_identity = self.tracking_data.get_ball_identity(ball_idx)
                if ball_identity:
                    key = f"fly_{fly_idx}_ball_{ball_identity}"
                else:
                    key = f"fly_{fly_idx}_ball_{ball_idx}"

                # Define metrics and their corresponding methods
                metrics = {
                    "nb_events": lambda: safe_call(self.get_adjusted_nb_events, fly_idx, ball_idx, signif=False),
                    "max_event": lambda: safe_call(self.get_max_event, fly_idx, ball_idx),
                    "max_distance": lambda: safe_call(self.get_max_distance, fly_idx, ball_idx),
                    "significant_events": lambda: safe_call(self.get_significant_events, fly_idx, ball_idx, default=[]),
                    "nb_significant_events": lambda: (
                        safe_call(self.get_adjusted_nb_events, fly_idx, ball_idx, signif=True)
                        if self.fly.config.experiment_type == "F1"
                        else len(safe_call(self.get_significant_events, fly_idx, ball_idx, default=[]))
                    ),
                    "first_significant_event": lambda: safe_call(self.get_first_significant_event, fly_idx, ball_idx),
                    "first_major_event": lambda: safe_call(self.get_major_event, fly_idx, ball_idx),
                    "events_direction": lambda: safe_call(
                        self.find_events_direction, fly_idx, ball_idx, default=([], [])
                    ),
                    "final_event": lambda: safe_call(self.get_final_event, fly_idx, ball_idx),
                    "success_direction": lambda: safe_call(self.get_success_direction, fly_idx, ball_idx),
                    "cumulated_breaks_duration": lambda: safe_call(
                        self.get_cumulated_breaks_duration, fly_idx, ball_idx
                    ),
                    "chamber_time": lambda: safe_call(self.get_chamber_time, fly_idx),
                    "chamber_ratio": lambda: safe_call(self.chamber_ratio, fly_idx),
                    "distance_moved": lambda: safe_call(self.get_distance_moved, fly_idx, ball_idx),
                    "distance_ratio": lambda: safe_call(self.get_distance_ratio, fly_idx, ball_idx),
                    "has_major": lambda: safe_call(self.get_has_major, fly_idx, ball_idx, default=0),
                    "has_significant": lambda: safe_call(self.get_has_significant, fly_idx, ball_idx, default=0),
                    "insight_effect": lambda: (
                        safe_call(self.get_insight_effect, fly_idx, ball_idx)
                        if safe_call(self.get_major_event, fly_idx, ball_idx)
                        else {
                            "raw_effect": np.nan,
                            "log_effect": np.nan,
                            "classification": "none",
                            "first_event": False,
                            "post_aha_count": 0,
                        }
                    ),
                }

                # Compute basic metrics only if needed
                nb_events = None
                if self.is_metric_enabled("nb_events") or self.is_metric_enabled("significant_ratio"):
                    nb_events = metrics["nb_events"]()

                max_event = None
                if self.is_metric_enabled("max_event") or self.is_metric_enabled("max_event_time"):
                    max_event = metrics["max_event"]()

                # Handle final event - only compute if needed
                final_event = None
                final_event_idx = np.nan
                final_event_time = np.nan
                final_event_end_time = np.nan  # Initialize here for all code paths
                raw_final_event_time = np.nan  # Initialize here for all code paths
                if (
                    self.is_metric_enabled("final_event")
                    or self.is_metric_enabled("final_event_time")
                    or self.is_metric_enabled("cumulated_breaks_duration")
                    or self.is_metric_enabled("chamber_time")
                    or self.is_metric_enabled("chamber_ratio")
                    or self.is_metric_enabled("distance_moved")
                    or self.is_metric_enabled("distance_ratio")
                    or self.is_metric_enabled("interaction_proportion")
                    or self.is_metric_enabled("interaction_persistence")
                    or self.is_metric_enabled("nb_pauses")
                    or self.is_metric_enabled("total_pause_duration")
                    or self.is_metric_enabled("nb_long_pauses")
                    or self.is_metric_enabled("median_long_pause_duration")
                    or self.is_metric_enabled("raw_pauses")
                ):
                    final_event = metrics["final_event"]()
                    if final_event is None:
                        final_event_idx = np.nan
                        final_event_time = np.nan
                        final_event_end_time = np.nan
                        raw_final_event_time = np.nan
                    else:
                        final_event_idx, final_event_time, final_event_end_time = final_event

                        # For F1 test balls, we need both adjusted and raw final_event_time
                        # - final_event_time: adjusted time (for time-based metrics)
                        # - final_event_end_time: adjusted end time (for interaction_proportion denominator)
                        # - raw_final_event_time: unadjusted time (for other rate calculations)
                        if self._should_use_adjusted_time(fly_idx, ball_idx):
                            # final_event_time and final_event_end_time are already adjusted, calculate raw version
                            raw_final_event_time = final_event_time + self.tracking_data.f1_exit_time
                        else:
                            # final_event_time is raw, use as-is for both
                            raw_final_event_time = final_event_time

                significant_events = None
                if (
                    self.is_metric_enabled("significant_events")
                    or self.is_metric_enabled("nb_significant_events")
                    or self.is_metric_enabled("significant_ratio")
                    or self.is_metric_enabled("first_significant_event")
                    or self.is_metric_enabled("first_significant_event_time")
                    or self.is_metric_enabled("pushed")
                    or self.is_metric_enabled("pulled")
                    or self.is_metric_enabled("pulling_ratio")
                ):
                    significant_events = metrics["significant_events"]()

                # Filter events and significant events up to and including the final event
                filtered_events = events
                filtered_significant_events = significant_events if significant_events is not None else []

                if not pd.isna(final_event_idx) and final_event_idx >= 0:
                    filtered_events = events[: final_event_idx + 1]
                    if significant_events is not None:
                        filtered_significant_events = [e for e in significant_events if e[1] <= final_event_idx]

                # Filtered event directions for pulling/pushing ratio - only compute if needed
                filtered_pushed = []
                filtered_pulled = []
                if (
                    self.is_metric_enabled("pushed")
                    or self.is_metric_enabled("pulled")
                    or self.is_metric_enabled("pulling_ratio")
                ):
                    filtered_events_direction = self.find_events_direction(fly_idx, ball_idx)
                    filtered_pushed = [
                        e for e in filtered_events_direction[0] if e in [ev[0] for ev in filtered_significant_events]
                    ]
                    filtered_pulled = [
                        e for e in filtered_events_direction[1] if e in [ev[0] for ev in filtered_significant_events]
                    ]

                # Conditional computation of expensive metrics
                major_event = None
                if (
                    self.is_metric_enabled("first_major_event")
                    or self.is_metric_enabled("first_major_event_time")
                    or self.is_metric_enabled("major_event_first")
                ):
                    major_event = metrics["first_major_event"]()

                events_direction = ([], [])
                if (
                    self.is_metric_enabled("pushed")
                    or self.is_metric_enabled("pulled")
                    or self.is_metric_enabled("pulling_ratio")
                ):
                    events_direction = metrics["events_direction"]()

                insight_effect = {
                    "raw_effect": np.nan,
                    "log_effect": np.nan,
                    "classification": "none",
                    "first_event": False,
                    "post_aha_count": 0,
                }
                if self.is_metric_enabled("major_event_first"):
                    val = metrics["insight_effect"]()
                    if isinstance(val, dict):
                        insight_effect = val

                pause_metrics = {"nb_pauses": 0, "median_pause_duration": np.nan, "total_pause_duration": 0.0}
                if (
                    self.is_metric_enabled("nb_pauses")
                    or self.is_metric_enabled("median_pause_duration")
                    or self.is_metric_enabled("total_pause_duration")
                ):
                    pause_metrics = safe_call(
                        self.compute_pause_metrics,
                        fly_idx,
                        default=pause_metrics,
                    )

                long_pause_metrics = {
                    "nb_long_pauses": 0,
                    "median_long_pause_duration": np.nan,
                    "total_long_pause_duration": 0.0,
                }
                if (
                    self.is_metric_enabled("nb_long_pauses")
                    or self.is_metric_enabled("median_long_pause_duration")
                    or self.is_metric_enabled("total_long_pause_duration")
                ):
                    long_pause_metrics = safe_call(
                        self.compute_long_pause_metrics,
                        fly_idx,
                        default={
                            "nb_long_pauses": 0,
                            "median_long_pause_duration": np.nan,
                            "total_long_pause_duration": 0.0,
                        },
                    )

                # Raw pauses - the unfiltered list of pauses detected
                raw_pauses = []
                if self.is_metric_enabled("raw_pauses"):
                    raw_pauses = safe_call(self.get_raw_pauses, fly_idx, default=[])

                interaction_persistence = np.nan
                if self.is_metric_enabled("interaction_persistence"):
                    interaction_persistence = safe_call(self.compute_interaction_persistence, fly_idx, ball_idx)

                # Expensive computations - only compute if needed
                learning_slope = {"slope": np.nan, "r2": np.nan}
                if self.is_metric_enabled("learning_slope") or self.is_metric_enabled("learning_slope_r2"):
                    learning_slope = safe_call(
                        self.compute_learning_slope, fly_idx, ball_idx, default={"slope": np.nan, "r2": np.nan}
                    )

                logistic_features = {"L": np.nan, "k": np.nan, "t0": np.nan, "r2": np.nan}
                if (
                    self.is_metric_enabled("logistic_L")
                    or self.is_metric_enabled("logistic_k")
                    or self.is_metric_enabled("logistic_t0")
                    or self.is_metric_enabled("logistic_r2")
                ):
                    logistic_features = safe_call(
                        self.compute_logistic_features,
                        fly_idx,
                        ball_idx,
                        default={"L": np.nan, "k": np.nan, "t0": np.nan, "r2": np.nan},
                    )

                event_influence = np.nan
                if self.is_metric_enabled("event_influence"):
                    event_influence = safe_call(self.compute_event_influence, fly_idx, ball_idx)

                # Velocity metrics - conditional computation
                normalized_velocity = np.nan
                if self.is_metric_enabled("normalized_velocity"):
                    normalized_velocity = safe_call(self.compute_normalized_velocity, fly_idx, ball_idx)

                velocity_during_interactions = np.nan
                if self.is_metric_enabled("velocity_during_interactions"):
                    velocity_during_interactions = safe_call(
                        self.compute_velocity_during_interactions, fly_idx, ball_idx
                    )

                velocity_trend = np.nan
                if self.is_metric_enabled("velocity_trend"):
                    velocity_trend = safe_call(self.compute_velocity_trend, fly_idx)

                # New metrics - conditional computation

                has_finished = 0
                if self.is_metric_enabled("has_finished"):
                    has_finished = safe_call(self.get_has_finished, fly_idx, ball_idx, default=0)

                has_major = 0
                if self.is_metric_enabled("has_major"):
                    has_major = safe_call(self.get_has_major, fly_idx, ball_idx, default=0)

                has_significant = 0
                if self.is_metric_enabled("has_significant"):
                    has_significant = safe_call(self.get_has_significant, fly_idx, ball_idx, default=0)

                has_long_pauses_flag = 0
                if self.is_metric_enabled("has_long_pauses"):
                    has_long_pauses_flag = safe_call(self.has_long_pauses, fly_idx, default=0)

                persistence_at_end = np.nan
                if self.is_metric_enabled("persistence_at_end"):
                    persistence_at_end = safe_call(self.compute_persistence_at_end, fly_idx, default=np.nan)

                fly_distance_moved = np.nan
                if self.is_metric_enabled("fly_distance_moved"):
                    fly_distance_moved = safe_call(self.compute_fly_distance_moved, fly_idx, default=np.nan)

                time_chamber_beginning = np.nan
                if self.is_metric_enabled("time_chamber_beginning"):
                    time_chamber_beginning = safe_call(self.get_time_chamber_beginning, fly_idx, default=np.nan)

                median_stop_duration = np.nan
                if self.is_metric_enabled("median_stop_duration"):
                    median_stop_duration = safe_call(self.compute_median_stop_duration, fly_idx, default=np.nan)

                # Stop metrics
                stop_metrics = {"nb_stops": 0, "median_stop_duration": np.nan, "total_stop_duration": 0.0}
                if (
                    self.is_metric_enabled("nb_stops")
                    or self.is_metric_enabled("median_stop_duration")
                    or self.is_metric_enabled("total_stop_duration")
                ):
                    stop_metrics = safe_call(self.compute_stop_metrics, fly_idx, default=stop_metrics)

                # Skeleton metrics - batch computation optimization
                skeleton_metrics_results = {}
                if self._needs_skeleton_metrics():
                    skeleton_metrics_results = self._compute_skeleton_metrics_batch(fly_idx, ball_idx)

                fraction_not_facing_ball = skeleton_metrics_results.get("fraction_not_facing_ball", 0.0)
                flailing = skeleton_metrics_results.get("flailing", 0.0)
                head_pushing_ratio = skeleton_metrics_results.get("head_pushing_ratio", 0.5)
                leg_visibility_ratio = skeleton_metrics_results.get("leg_visibility_ratio", 0.0)

                # Binned metrics - expensive, only compute if any binned metric is enabled
                binned_slope = []
                binned_slope_dict = {}
                if any(self.is_metric_enabled(f"binned_slope_{i}") for i in range(12)):
                    binned_slope = safe_call(self.compute_binned_slope, fly_idx, ball_idx, default=[])
                    binned_slope_dict = {f"binned_slope_{i}": val for i, val in enumerate(binned_slope or [])}

                interaction_rate_by_bin = []
                interaction_rate_by_bin_dict = {}
                if any(self.is_metric_enabled(f"interaction_rate_bin_{i}") for i in range(12)):
                    interaction_rate_by_bin = safe_call(
                        self.compute_interaction_rate_by_bin, fly_idx, ball_idx, default=[]
                    )
                    interaction_rate_by_bin_dict = {
                        f"interaction_rate_bin_{i}": val for i, val in enumerate(interaction_rate_by_bin or [])
                    }

                overall_slope = np.nan
                if self.is_metric_enabled("overall_slope"):
                    overall_slope = safe_call(self.compute_overall_slope, fly_idx, ball_idx)

                overall_interaction_rate = np.nan
                if self.is_metric_enabled("overall_interaction_rate"):
                    overall_interaction_rate = safe_call(self.compute_overall_interaction_rate, fly_idx, ball_idx)

                auc = np.nan
                if self.is_metric_enabled("auc"):
                    auc = safe_call(self.compute_auc, fly_idx, ball_idx)

                binned_auc = []
                binned_auc_dict = {}
                if any(self.is_metric_enabled(f"binned_auc_{i}") for i in range(12)):
                    binned_auc = safe_call(self.compute_binned_auc, fly_idx, ball_idx, default=[])
                    binned_auc_dict = {f"binned_auc_{i}": val for i, val in enumerate(binned_auc or [])}

                # Store metrics in the dictionary - only compute enabled metrics
                metrics_dict = {}

                # Add identification information
                metrics_dict["fly_idx"] = fly_idx
                metrics_dict["ball_idx"] = ball_idx
                if ball_identity:
                    metrics_dict["ball_identity"] = ball_identity
                else:
                    # For backward compatibility, set to None/unknown for regular experiments
                    metrics_dict["ball_identity"] = None

                # Basic event metrics
                if self.is_metric_enabled("nb_events"):
                    metrics_dict["nb_events"] = nb_events
                if self.is_metric_enabled("max_event") and max_event is not None:
                    # max_event is a tuple (max_event_idx, max_event_time)
                    # Check if max_event_idx is valid (not None or NaN)
                    if max_event[0] is not None and not (
                        isinstance(max_event[0], (int, float)) and np.isnan(max_event[0])
                    ):
                        metrics_dict["max_event"] = max_event[0]
                    else:
                        metrics_dict["max_event"] = np.nan
                if self.is_metric_enabled("max_event_time") and max_event is not None:
                    # Check if max_event_time is valid (not None or NaN)
                    if max_event[1] is not None and not (
                        isinstance(max_event[1], (int, float)) and np.isnan(max_event[1])
                    ):
                        metrics_dict["max_event_time"] = max_event[1]
                    else:
                        metrics_dict["max_event_time"] = np.nan
                if self.is_metric_enabled("max_distance"):
                    metrics_dict["max_distance"] = metrics["max_distance"]()
                if self.is_metric_enabled("final_event"):
                    metrics_dict["final_event"] = final_event_idx
                if self.is_metric_enabled("final_event_time"):
                    metrics_dict["final_event_time"] = final_event_time
                if self.is_metric_enabled("nb_significant_events"):
                    metrics_dict["nb_significant_events"] = len(filtered_significant_events)
                if self.is_metric_enabled("significant_ratio"):
                    metrics_dict["significant_ratio"] = (
                        len(filtered_significant_events) / len(filtered_events) if len(filtered_events) > 0 else np.nan
                    )

                first_significant_event_result = None
                if self.is_metric_enabled("first_significant_event") or self.is_metric_enabled(
                    "first_significant_event_time"
                ):
                    first_significant_event_result = metrics["first_significant_event"]()

                if self.is_metric_enabled("first_significant_event") and first_significant_event_result is not None:
                    metrics_dict["first_significant_event"] = first_significant_event_result[0]
                if (
                    self.is_metric_enabled("first_significant_event_time")
                    and first_significant_event_result is not None
                ):
                    metrics_dict["first_significant_event_time"] = first_significant_event_result[1]
                if self.is_metric_enabled("first_major_event"):
                    metrics_dict["first_major_event"] = major_event[0] if major_event else np.nan
                if self.is_metric_enabled("first_major_event_time"):
                    metrics_dict["first_major_event_time"] = major_event[1] if major_event else np.nan
                if self.is_metric_enabled("major_event_first"):
                    metrics_dict["major_event_first"] = insight_effect["first_event"]

                # Movement and interaction metrics
                if self.is_metric_enabled("cumulated_breaks_duration"):
                    metrics_dict["cumulated_breaks_duration"] = safe_call(
                        self.get_cumulated_breaks_duration, fly_idx, ball_idx, subset=filtered_events
                    )
                if self.is_metric_enabled("chamber_time"):
                    metrics_dict["chamber_time"] = safe_call(
                        self.get_chamber_time, fly_idx, end_time=raw_final_event_time
                    )
                if self.is_metric_enabled("chamber_ratio"):
                    metrics_dict["chamber_ratio"] = safe_call(
                        self.chamber_ratio, fly_idx, end_time=raw_final_event_time
                    )
                if self.is_metric_enabled("pushed"):
                    metrics_dict["pushed"] = len(filtered_pushed)
                if self.is_metric_enabled("pulled"):
                    metrics_dict["pulled"] = len(filtered_pulled)
                if self.is_metric_enabled("pulling_ratio"):
                    metrics_dict["pulling_ratio"] = (
                        len(filtered_pulled) / (len(filtered_pushed) + len(filtered_pulled))
                        if (len(filtered_pushed) + len(filtered_pulled)) > 0
                        else np.nan
                    )
                if self.is_metric_enabled("success_direction"):
                    metrics_dict["success_direction"] = metrics["success_direction"]()
                if self.is_metric_enabled("interaction_proportion"):
                    # For interaction proportion, use the end time of the final event to include
                    # the full duration of the final event in the accessible time period
                    # If no final event detected, use full duration and all events
                    if final_event_end_time is not None:
                        proportion_denominator = final_event_end_time
                        events_for_proportion = filtered_events
                    else:
                        proportion_denominator = self.tracking_data.duration
                        events_for_proportion = events  # Use all events when no final event

                    # Convert event durations from frames to seconds (event[2] is in frames)
                    total_interaction_duration_seconds = sum(
                        [event[2] / self.fly.experiment.fps for event in events_for_proportion]
                    )
                    metrics_dict["interaction_proportion"] = (
                        total_interaction_duration_seconds / proportion_denominator
                        if proportion_denominator > 0
                        else np.nan
                    )
                if self.is_metric_enabled("interaction_persistence"):
                    metrics_dict["interaction_persistence"] = safe_call(
                        self.compute_interaction_persistence, fly_idx, ball_idx, subset=filtered_events
                    )
                if self.is_metric_enabled("distance_moved"):
                    metrics_dict["distance_moved"] = safe_call(
                        self.get_distance_moved, fly_idx, ball_idx, subset=filtered_events
                    )
                if self.is_metric_enabled("distance_ratio"):
                    metrics_dict["distance_ratio"] = safe_call(
                        self.get_distance_ratio, fly_idx, ball_idx, subset=filtered_events
                    )
                if self.is_metric_enabled("f1_exit_time"):
                    metrics_dict["f1_exit_time"] = self.tracking_data.f1_exit_time
                if self.is_metric_enabled("chamber_exit_time"):
                    metrics_dict["chamber_exit_time"] = self.tracking_data.chamber_exit_time

                # Pause metrics
                pause_metrics = {"nb_pauses": 0, "median_pause_duration": np.nan, "total_pause_duration": 0.0}
                if (
                    self.is_metric_enabled("nb_pauses")
                    or self.is_metric_enabled("median_pause_duration")
                    or self.is_metric_enabled("total_pause_duration")
                ):
                    pause_metrics = safe_call(
                        self.compute_pause_metrics,
                        fly_idx,
                        subset=filtered_events,
                        default=pause_metrics,
                    )

                if self.is_metric_enabled("nb_pauses"):
                    metrics_dict["nb_pauses"] = pause_metrics["nb_pauses"]
                if self.is_metric_enabled("median_pause_duration"):
                    metrics_dict["median_pause_duration"] = pause_metrics["median_pause_duration"]
                if self.is_metric_enabled("total_pause_duration"):
                    metrics_dict["total_pause_duration"] = pause_metrics["total_pause_duration"]

                # Long pause metrics
                if self.is_metric_enabled("nb_long_pauses"):
                    metrics_dict["nb_long_pauses"] = safe_call(
                        self.compute_long_pause_metrics,
                        fly_idx,
                        subset=filtered_events,
                        default={
                            "nb_long_pauses": 0,
                            "median_long_pause_duration": np.nan,
                            "total_long_pause_duration": 0.0,
                        },
                    )["nb_long_pauses"]
                if self.is_metric_enabled("median_long_pause_duration"):
                    metrics_dict["median_long_pause_duration"] = safe_call(
                        self.compute_long_pause_metrics,
                        fly_idx,
                        subset=filtered_events,
                        default={
                            "nb_long_pauses": 0,
                            "median_long_pause_duration": np.nan,
                            "total_long_pause_duration": 0.0,
                        },
                    )["median_long_pause_duration"]
                if self.is_metric_enabled("total_long_pause_duration"):
                    metrics_dict["total_long_pause_duration"] = safe_call(
                        self.compute_long_pause_metrics,
                        fly_idx,
                        subset=filtered_events,
                        default={
                            "nb_long_pauses": 0,
                            "median_long_pause_duration": np.nan,
                            "total_long_pause_duration": 0.0,
                        },
                    )["total_long_pause_duration"]

                # Learning and logistic metrics (expensive)
                if self.is_metric_enabled("learning_slope"):
                    metrics_dict["learning_slope"] = learning_slope["slope"]
                if self.is_metric_enabled("learning_slope_r2"):
                    metrics_dict["learning_slope_r2"] = learning_slope["r2"]
                if self.is_metric_enabled("logistic_L"):
                    metrics_dict["logistic_L"] = logistic_features["L"]
                if self.is_metric_enabled("logistic_k"):
                    metrics_dict["logistic_k"] = logistic_features["k"]
                if self.is_metric_enabled("logistic_t0"):
                    metrics_dict["logistic_t0"] = logistic_features["t0"]
                if self.is_metric_enabled("logistic_r2"):
                    metrics_dict["logistic_r2"] = logistic_features["r2"]

                # Velocity metrics
                if self.is_metric_enabled("normalized_velocity"):
                    metrics_dict["normalized_velocity"] = normalized_velocity
                if self.is_metric_enabled("velocity_during_interactions"):
                    metrics_dict["velocity_during_interactions"] = velocity_during_interactions
                if self.is_metric_enabled("velocity_trend"):
                    metrics_dict["velocity_trend"] = velocity_trend
                if self.is_metric_enabled("overall_slope"):
                    metrics_dict["overall_slope"] = overall_slope
                if self.is_metric_enabled("overall_interaction_rate"):
                    metrics_dict["overall_interaction_rate"] = overall_interaction_rate

                # Binned metrics (expensive)
                for i, val in enumerate(binned_slope_dict or {}):
                    metric_name = f"binned_slope_{i}"
                    if self.is_metric_enabled(metric_name):
                        metrics_dict[metric_name] = binned_slope_dict[metric_name]

                for i, val in enumerate(interaction_rate_by_bin_dict or {}):
                    metric_name = f"interaction_rate_bin_{i}"
                    if self.is_metric_enabled(metric_name):
                        metrics_dict[metric_name] = interaction_rate_by_bin_dict[metric_name]

                # AUC metrics
                if self.is_metric_enabled("auc"):
                    metrics_dict["auc"] = auc

                for i, val in enumerate(binned_auc_dict or {}):
                    metric_name = f"binned_auc_{i}"
                    if self.is_metric_enabled(metric_name):
                        metrics_dict[metric_name] = binned_auc_dict[metric_name]

                # Additional metrics
                if self.is_metric_enabled("has_finished"):
                    metrics_dict["has_finished"] = has_finished
                if self.is_metric_enabled("has_major"):
                    metrics_dict["has_major"] = has_major
                if self.is_metric_enabled("has_significant"):
                    metrics_dict["has_significant"] = has_significant
                if self.is_metric_enabled("has_long_pauses"):
                    metrics_dict["has_long_pauses"] = has_long_pauses_flag
                if self.is_metric_enabled("persistence_at_end"):
                    metrics_dict["persistence_at_end"] = persistence_at_end
                if self.is_metric_enabled("fly_distance_moved"):
                    metrics_dict["fly_distance_moved"] = fly_distance_moved
                if self.is_metric_enabled("time_chamber_beginning"):
                    metrics_dict["time_chamber_beginning"] = time_chamber_beginning
                if self.is_metric_enabled("nb_stops"):
                    metrics_dict["nb_stops"] = stop_metrics["nb_stops"]
                if self.is_metric_enabled("median_stop_duration"):
                    metrics_dict["median_stop_duration"] = stop_metrics["median_stop_duration"]
                if self.is_metric_enabled("total_stop_duration"):
                    metrics_dict["total_stop_duration"] = stop_metrics["total_stop_duration"]
                if self.is_metric_enabled("nb_long_pauses"):
                    metrics_dict["nb_long_pauses"] = long_pause_metrics["nb_long_pauses"]
                if self.is_metric_enabled("median_long_pause_duration"):
                    metrics_dict["median_long_pause_duration"] = long_pause_metrics["median_long_pause_duration"]
                if self.is_metric_enabled("total_long_pause_duration"):
                    metrics_dict["total_long_pause_duration"] = long_pause_metrics["total_long_pause_duration"]

                if self.is_metric_enabled("raw_pauses"):
                    metrics_dict["raw_pauses"] = raw_pauses
                if self.is_metric_enabled("fraction_not_facing_ball"):
                    metrics_dict["fraction_not_facing_ball"] = fraction_not_facing_ball
                if self.is_metric_enabled("flailing"):
                    metrics_dict["flailing"] = flailing
                if self.is_metric_enabled("head_pushing_ratio"):
                    metrics_dict["head_pushing_ratio"] = head_pushing_ratio
                if self.is_metric_enabled("leg_visibility_ratio"):
                    metrics_dict["leg_visibility_ratio"] = leg_visibility_ratio

                self.metrics[key] = metrics_dict

    def get_metrics_by_identity(self, fly_idx, ball_identity):
        """
        Get metrics for a specific fly and ball identity (e.g., 'training', 'test').

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_identity : str
            Ball identity ('training', 'test', etc.)

        Returns
        -------
        dict or None
            Metrics dictionary if found, None otherwise.
        """
        # First try with identity-based key
        key = f"fly_{fly_idx}_ball_{ball_identity}"
        if key in self.metrics:
            return self.metrics[key]

        # Fallback: search through all metrics to find matching ball_identity
        for metric_key, metrics_dict in self.metrics.items():
            if metrics_dict.get("fly_idx") == fly_idx and metrics_dict.get("ball_identity") == ball_identity:
                return metrics_dict

        return None

    def get_training_ball_metrics(self, fly_idx):
        """
        Get metrics for the training ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        dict or None
            Training ball metrics if found, None otherwise.
        """
        return self.get_metrics_by_identity(fly_idx, "training")

    def get_test_ball_metrics(self, fly_idx):
        """
        Get metrics for the test ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        dict or None
            Test ball metrics if found, None otherwise.
        """
        return self.get_metrics_by_identity(fly_idx, "test")

    def get_all_ball_identities(self, fly_idx):
        """
        Get all available ball identities for a given fly.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        list
            List of available ball identities for the fly.
        """
        identities = []
        for metric_key, metrics_dict in self.metrics.items():
            if metrics_dict.get("fly_idx") == fly_idx:
                identity = metrics_dict.get("ball_identity")
                if identity and identity not in identities:
                    identities.append(identity)
        return identities

    def has_ball_identity(self, fly_idx, ball_identity):
        """
        Check if metrics exist for a specific fly and ball identity.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_identity : str
            Ball identity to check for.

        Returns
        -------
        bool
            True if metrics exist for the specified identity, False otherwise.
        """
        return self.get_metrics_by_identity(fly_idx, ball_identity) is not None

    def _get_appropriate_final_event_threshold(self, fly_idx, ball_idx):
        """
        Determine the appropriate final event threshold based on ball identity for F1 experiments.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            The appropriate final event threshold.
        """
        # Check if we have ball identity information (F1 experiment)
        if (
            hasattr(self.tracking_data, "ball_identities")
            and self.tracking_data.ball_identities is not None
            and ball_idx in self.tracking_data.ball_identities
        ):

            ball_identity = self.tracking_data.ball_identities[ball_idx]

            if ball_identity == "test":
                # Test ball uses F1 threshold (shorter corridor)
                return self.fly.config.final_event_F1_threshold
            elif ball_identity == "training":
                # Training ball uses regular threshold (longer corridor)
                return self.fly.config.final_event_threshold

        # Fallback: use the existing position-based logic for backwards compatibility
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            return self.fly.config.final_event_threshold
        else:
            return self.fly.config.final_event_F1_threshold

    def _should_use_adjusted_time(self, fly_idx, ball_idx):
        """
        Determine if adjusted time should be used for F1 experiments with test ball.

        For F1 experiments, we want to use adjusted times for the test ball to measure
        performance relative to when the fly exits the first corridor (training corridor).

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        bool
            True if adjusted time should be used, False otherwise.
        """
        # Check if this is an F1 experiment
        if not hasattr(self.fly.config, "experiment_type") or self.fly.config.experiment_type != "F1":
            return False

        # Check if we have ball identity information
        if not hasattr(self.tracking_data, "ball_identities") or self.tracking_data.ball_identities is None:
            return False

        # Check if this ball is the test ball
        ball_identity = self.tracking_data.ball_identities.get(ball_idx)
        if ball_identity != "test":
            return False

        # Check if we have F1 exit time available
        if not hasattr(self.tracking_data, "f1_exit_time") or self.tracking_data.f1_exit_time is None:
            return False

        return True

    def _adjust_time_for_f1_test_ball(self, raw_time, fly_idx, ball_idx):
        """
        Adjust time for F1 test ball metrics by subtracting the corridor exit time.

        For F1 experiments with test ball, we want to measure time relative to when the fly
        exits the first (training) corridor, not from the absolute start of the experiment.

        Parameters
        ----------
        raw_time : float
            Raw time value (seconds from experiment start).
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Adjusted time (seconds from corridor exit) if adjustment should be applied,
            otherwise returns the raw time unchanged.
        """
        if not self._should_use_adjusted_time(fly_idx, ball_idx):
            return raw_time

        if np.isnan(raw_time) or raw_time is None:
            return raw_time

        # Apply the adjustment: subtract F1 corridor exit time to get time relative to corridor exit
        adjusted_time = raw_time - self.tracking_data.f1_exit_time

        # Ensure adjusted time is not negative (shouldn't happen for test ball events)
        if adjusted_time < 0:
            if self.fly.config.debugging:
                print(f"Warning: Adjusted time is negative ({adjusted_time:.2f}s) for test ball event")
            return np.nan

        return adjusted_time

    def get_adjusted_nb_events(self, fly_idx, ball_idx, signif=False):
        """
        Calculate the adjusted number of events for a given fly and ball. adjustment is based on the duration of the experiment.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        signif : bool, optional
            Whether to use significant events only (default is False).

        Returns
        -------
        float
            Adjusted number of events.
        """
        if signif:
            events = self.get_significant_events(fly_idx, ball_idx)
        else:
            events = self.tracking_data.interaction_events[fly_idx][ball_idx]

        if self.fly.config.debugging:
            print(f"Events for fly {fly_idx} ball {ball_idx}, signif:{signif}: {len(events)}; {events}")

        adjusted_nb_events = 0  # Initialize to 0 in case there are no events

        if hasattr(self.fly.metadata, "F1_condition"):

            if self.fly.metadata.F1_condition == "control":
                if ball_idx == 0 and self.tracking_data.chamber_exit_time is not None:
                    chamber_exit_time = self.tracking_data.chamber_exit_time
                    adjusted_nb_events = (
                        len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / (self.tracking_data.duration - chamber_exit_time)
                        if self.tracking_data.duration - chamber_exit_time > 0
                        else 0
                    )
            else:
                if ball_idx == 1 and self.tracking_data.f1_exit_time is not None:
                    adjusted_nb_events = (
                        len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / (self.tracking_data.duration - self.tracking_data.f1_exit_time)
                        if self.tracking_data.duration - self.tracking_data.f1_exit_time > 0
                        else 0
                    )
                elif ball_idx == 0:
                    # For F1 training ball (ball_idx == 0), calculate events per normalized time unit
                    # This gives a rate: events * 1000 / training_duration_seconds
                    # High values (e.g., 925) are expected for short training durations (e.g., ~1 second)
                    # since it represents events per 1000 seconds of normalized time
                    chamber_exit_time = None
                    if hasattr(self.tracking_data, "chamber_exit_times") and self.tracking_data.chamber_exit_times:
                        chamber_exit_time = self.tracking_data.chamber_exit_times.get(fly_idx)
                    elif hasattr(self.tracking_data, "chamber_exit_time"):
                        chamber_exit_time = self.tracking_data.chamber_exit_time

                    if chamber_exit_time and chamber_exit_time > 0:
                        adjusted_nb_events = (
                            len(events) * self.fly.config.adjusted_events_normalisation / chamber_exit_time
                        )
                    else:
                        # Fallback: use total duration for training ball
                        adjusted_nb_events = (
                            len(events) * self.fly.config.adjusted_events_normalisation / self.tracking_data.duration
                            if self.tracking_data.duration > 0
                            else 0
                        )

        else:
            adjusted_nb_events = (
                len(events)
                * self.fly.config.adjusted_events_normalisation
                / (self.tracking_data.duration - self.tracking_data.chamber_exit_times[fly_idx])
                if self.tracking_data.duration > 0
                else 0
            )

        return adjusted_nb_events

    def _calculate_median_coordinates(self, data, start_idx=None, end_idx=None, window=10, keypoint="centre"):
        """
        Calculate the median coordinates for the start and/or end of a given range.
        Optimized with internal result caching for frequently called combinations.

        Parameters
        ----------
        data : pandas.DataFrame
            Data containing the x and y coordinate columns.
        start_idx : int, optional
            Start index of the range. If None, the start median is not calculated.
        end_idx : int, optional
            End index of the range. If None, the end median is not calculated.
        window : int, optional
            Number of frames to consider for the median calculation (default is 10).
        keypoint : str, optional
            Keypoint prefix for the x and y columns (e.g., "centre" for "x_centre" and "y_centre",
            or "thorax" for "x_thorax" and "y_thorax"). Default is "centre".

        Returns
        -------
        tuple
            Median coordinates for the start and/or end of the range. If `start_idx` or `end_idx`
            is None, the corresponding value in the tuple will be None.
        """
        # Create cache key for this specific calculation
        cache_key = (id(data), start_idx, end_idx, window, keypoint)

        # Check if we have this calculation cached
        if not hasattr(self, "_median_coords_cache"):
            self._median_coords_cache = {}

        if cache_key in self._median_coords_cache:
            return self._median_coords_cache[cache_key]

        x_col = f"x_{keypoint}"
        y_col = f"y_{keypoint}"

        start_x, start_y, end_x, end_y = None, None, None, None

        # Calculate the median position of the first `window` frames if start_idx is provided
        if start_idx is not None:
            start_x = data[x_col].iloc[start_idx : start_idx + window].median()
            start_y = data[y_col].iloc[start_idx : start_idx + window].median()

        # Calculate the median position of the last `window` frames if end_idx is provided
        if end_idx is not None:
            end_x = data[x_col].iloc[end_idx - window : end_idx].median()
            end_y = data[y_col].iloc[end_idx - window : end_idx].median()

        result = (start_x, start_y, end_x, end_y)

        # Cache the result (limit cache size to prevent memory issues)
        if len(self._median_coords_cache) < 1000:
            self._median_coords_cache[cache_key] = result

        return result

    def find_event_by_distance(self, fly_idx, ball_idx, threshold, distance_type="max"):
        """
        Find the event at which the ball has been moved a given amount of pixels for a given fly and ball.
        Uses median smoothed coordinates at event start/end to determine if ball crossed threshold,
        which is more robust against tracking errors.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float
            Distance threshold.
        distance_type : str, optional
            Type of distance to check ("max" or "threshold", default is "max").

        Returns
        -------
        tuple
            Event and event index.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Get ball initial position using median coordinates for robustness
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, window=10, keypoint="centre"
        )

        if distance_type == "max":
            # For max distance type, first calculate the actual max distance reached
            # using median coordinates at the end of all events
            max_distance_reached = 0
            for event in self.tracking_data.interaction_events[fly_idx][ball_idx]:
                start_idx, end_idx = event[0], event[1]
                _, _, end_x, end_y = self._calculate_median_coordinates(
                    ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
                )
                if end_x is not None and end_y is not None:
                    distance_from_initial = Processing.calculate_euclidian_distance(end_x, end_y, initial_x, initial_y)
                    max_distance_reached = max(max_distance_reached, distance_from_initial)

            target_distance = max_distance_reached - threshold

            def distance_check(event):
                start_idx, end_idx = event[0], event[1]
                _, _, end_x, end_y = self._calculate_median_coordinates(
                    ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
                )
                if end_x is None or end_y is None:
                    return False
                distance_from_initial = Processing.calculate_euclidian_distance(end_x, end_y, initial_x, initial_y)
                return distance_from_initial >= target_distance

        elif distance_type == "threshold":

            def distance_check(event):
                start_idx, end_idx = event[0], event[1]
                # Get median coordinates at event start and end
                start_x, start_y, end_x, end_y = self._calculate_median_coordinates(
                    ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
                )
                if start_x is None or start_y is None or end_x is None or end_y is None:
                    return False

                # Check if ball moved at least threshold distance from initial position by event end
                distance_from_initial = Processing.calculate_euclidian_distance(end_x, end_y, initial_x, initial_y)
                return distance_from_initial >= threshold

        else:
            raise ValueError("Invalid distance_type. Use 'max' or 'threshold'.")

        try:
            event, event_index = next(
                (event, i)
                for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
                if distance_check(event)
            )
        except StopIteration:
            event, event_index = None, None

        return event, event_index

    # TODO: Implement adjustment by computing time to exit chamber and subtract it from the time of the event (perhaps keep both?)
    def get_chamber_exit_time(self, fly_idx, ball_idx):
        """
        Get the chamber exit time from the tracking data.
        Since we have only one fly in one chamber, this delegates to the tracking_data method.

        Args:
            fly_idx (int): Index of the fly (ignored, kept for API compatibility).
            ball_idx (int): Index of the ball (ignored, kept for API compatibility).

        Returns:
            float or None: The exit time in seconds, or None if the fly never left the chamber.
        """
        return self.tracking_data.chamber_exit_time

    @lru_cache(maxsize=128)
    def get_max_event(self, fly_idx, ball_idx, threshold=None):
        """
        Get the event at which the ball was moved at its maximum distance for a given fly and ball.
        Maximum here doesn't mean the ball has reached the end of the corridor.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float, optional
            Distance threshold (default is None).

        Returns
        -------
        tuple
            Maximum event index and maximum event time.
        """
        if threshold is None:
            threshold = self.fly.config.max_event_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Find the event with the maximum distance
        max_event, max_event_idx = self.find_event_by_distance(fly_idx, ball_idx, threshold, distance_type="max")

        # Get the chamber exit time for the current fly
        chamber_exit_time = self.tracking_data.chamber_exit_time

        # Calculate the time of the maximum event
        if max_event:
            # Get raw time first
            raw_max_event_time = max_event[0] / self.fly.experiment.fps
            # Apply appropriate time adjustment based on ball identity
            max_event_time = self._adjust_time_for_f1_test_ball(raw_max_event_time, fly_idx, ball_idx)
        else:
            max_event_time = np.nan
        # max_event_time = self._adjust_time_for_f1_test_ball(max_event_time, fly_idx, ball_idx)

        return max_event_idx if max_event_idx is not None else np.nan, max_event_time

    def get_max_distance(self, fly_idx, ball_idx):
        """
        Get the maximum distance moved by the ball for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Maximum distance moved by the ball.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Calculate the median initial position
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(ball_data, start_idx=0, window=10)

        # Calculate the Euclidean distance from the initial position
        distances = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
        )

        # Return the maximum distance
        return distances.max()

    def get_final_event(self, fly_idx, ball_idx, threshold=None, init=False):
        """
        Get the final event for a given fly and ball based on a distance threshold.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float, optional
            Distance threshold for the final event (default is None).
        init : bool, optional
            Whether to use initial conditions (default is False).

        Returns
        -------
        tuple or None
            A tuple (final_event_idx, final_event_time, final_event_end) if a final event is found,
            or None if no final event is found.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Check the ball data time boundaries
        ball_data_start = ball_data["time"].iloc[0]
        ball_data_end = ball_data["time"].iloc[-1]

        if self.fly.config.debugging:
            print(f"Ball data time range: {ball_data_start} to {ball_data_end}")

        # # Apply time range filtering
        # time_range = self.tracking_data.fly.config.time_range
        # if time_range:
        #     start = int(time_range[0] * self.tracking_data.fly.experiment.fps) if time_range[0] is not None else 0
        #     end = int(time_range[1] * self.tracking_data.fly.experiment.fps) if time_range[1] is not None else None
        #     ball_data = ball_data.iloc[start:end]

        # Get the chamber exit time for the current fly
        chamber_exit_time = self.tracking_data.chamber_exit_time

        # Determine the appropriate threshold
        if threshold is None:
            threshold = self._get_appropriate_final_event_threshold(fly_idx, ball_idx)
        if self.fly.config.debugging:
            print(f"Threshold for fly {fly_idx}, ball {ball_idx}: {threshold}")

        # Find the final event based on the threshold
        final_event, final_event_idx = self.find_event_by_distance(
            fly_idx, ball_idx, threshold, distance_type="threshold"
        )
        if self.fly.config.debugging:
            print(f"Final event for fly {fly_idx}, ball {ball_idx}: {final_event}")

        # If no final event is found, return None
        if not final_event:
            return None

        # Calculate the raw time and end time of the final event (no adjustments yet)
        raw_final_event_start_time = final_event[0] / self.fly.experiment.fps
        raw_final_event_time = final_event[0] / self.fly.experiment.fps  # Use START time for final event
        raw_final_event_end = final_event[1] / self.fly.experiment.fps

        # Apply appropriate time adjustment based on ball identity
        final_event_start_time = self._adjust_time_for_f1_test_ball(raw_final_event_start_time, fly_idx, ball_idx)
        final_event_time = self._adjust_time_for_f1_test_ball(raw_final_event_time, fly_idx, ball_idx)
        final_event_end = self._adjust_time_for_f1_test_ball(raw_final_event_end, fly_idx, ball_idx)

        return final_event_idx, final_event_time, final_event_end

    @lru_cache(maxsize=128)
    def get_significant_events(self, fly_idx, ball_idx, distance=5):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = [
            (event, i)
            for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        return significant_events

    def get_first_significant_event(self, fly_idx, ball_idx, distance=5):
        """
        Get the first significant event for a given fly and ball based on a distance threshold.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        distance : float, optional
            Distance threshold for significant events (default is 5).

        Returns
        -------
        tuple
            First significant event index and first significant event time.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Get the chamber exit time for the current fly
        chamber_exit_time = self.tracking_data.chamber_exit_time

        # Get significant events
        significant_events = self.get_significant_events(fly_idx, ball_idx, distance=distance)

        if significant_events:
            first_significant_event = significant_events[0]
            first_significant_event_idx = first_significant_event[1]

            # Calculate the raw time of the first significant event (no adjustments yet)
            raw_first_significant_event_time = first_significant_event[0][0] / self.fly.experiment.fps

            # Apply appropriate time adjustment based on ball identity
            first_significant_event_time = self._adjust_time_for_f1_test_ball(
                raw_first_significant_event_time, fly_idx, ball_idx
            )

            return first_significant_event_idx, first_significant_event_time
        else:
            return np.nan, np.nan

    def check_yball_variation(self, event, ball_data, threshold=None):

        if threshold is None:
            threshold = self.fly.config.significant_threshold

        yball_event = ball_data.loc[event[0] : event[1], "y_centre"]
        variation = yball_event.max() - yball_event.min()
        return variation > threshold

    def find_breaks(self, fly_idx, ball_idx):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        breaks = []
        if not self.tracking_data.interaction_events[fly_idx][ball_idx]:
            breaks.append((0, len(ball_data), len(ball_data)))
            return breaks

        if self.tracking_data.interaction_events[fly_idx][ball_idx][0][0] > 0:
            breaks.append(
                (
                    0,
                    self.tracking_data.interaction_events[fly_idx][ball_idx][0][0],
                    self.tracking_data.interaction_events[fly_idx][ball_idx][0][0],
                )
            )

        for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx][:-1]):
            start = event[1]
            end = self.tracking_data.interaction_events[fly_idx][ball_idx][i + 1][0]
            duration = end - start
            breaks.append((start, end, duration))

        if self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1] < len(ball_data):
            breaks.append(
                (
                    self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1],
                    len(ball_data),
                    len(ball_data) - self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1],
                )
            )

        return breaks

    def get_cumulated_breaks_duration(self, fly_idx, ball_idx, subset=None):
        breaks = self.find_breaks(fly_idx, ball_idx)
        if subset is not None:
            # Only include breaks that overlap with any event in subset
            valid_ranges = [(event[0], event[1]) for event in subset]
            filtered_breaks = []
            for br in breaks:
                br_start, br_end, _ = br
                if any((br_start < end and br_end > start) for start, end in valid_ranges):
                    filtered_breaks.append(br)
            breaks = filtered_breaks
        cumulated_breaks_duration = sum([break_[2] for break_ in breaks])
        return cumulated_breaks_duration

    def get_chamber_time(self, fly_idx, end_time=None):
        """
        Compute the time spent by the fly in the chamber. The chamber is defined as the area within a certain radius of the fly start position.

        Args:
            fly_idx (int): Index of the fly.

        Returns:
            float: Time spent in the chamber in seconds.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Get the fly start position using the median of the first 10 frames
        start_x, start_y, _, _ = self._calculate_median_coordinates(fly_data, start_idx=0, window=10, keypoint="thorax")

        # Calculate the distance from the fly start position for each frame
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"], fly_data["y_thorax"], start_x, start_y
        )

        # Determine the frames where the fly is within a certain radius of the start position
        in_chamber = distances <= self.fly.config.chamber_radius

        if end_time is not None:
            in_chamber = in_chamber[: int(end_time * self.fly.experiment.fps)]

        # Calculate the time spent in the chamber
        time_in_chamber = np.sum(in_chamber) / self.fly.experiment.fps

        return time_in_chamber

    # TODO: Maybe use the non smoothed data to compute this kind of thing as it looks like there's a slight lag in the smoothed data.

    def chamber_ratio(self, fly_idx, end_time=None):
        """Compute the ratio of time spent in the chamber to the total time of the experiment.
        Args:
            fly_idx (int): Index of the fly.
        """

        time_in_chamber = self.get_chamber_time(fly_idx, end_time=end_time)
        total_time = end_time if end_time is not None else self.tracking_data.duration

        # Calculate the ratio
        if total_time > 0:
            ratio = time_in_chamber / total_time
        else:
            ratio = np.nan

        return ratio

    def find_events_direction(self, fly_idx, ball_idx):
        """
        Categorize significant events as pushing or pulling based on the change in distance
        between the ball and the fly during each event.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.

        Returns:
            tuple: Two lists - pushing_events and pulling_events.
        """
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = self.get_significant_events(fly_idx, ball_idx)

        pushing_events = []
        pulling_events = []

        for event in significant_events:
            event = event[0]

            start_roi = event[0]
            end_roi = event[1]

            # Calculate the median positions for the start and end of the event
            start_ball_x, start_ball_y, end_ball_x, end_ball_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_roi, end_idx=end_roi, window=10, keypoint="centre"
            )

            start_fly_x, start_fly_y, _, _ = self._calculate_median_coordinates(
                fly_data, start_idx=start_roi, window=10, keypoint="thorax"
            )

            # Calculate the distances using the median positions
            start_distance = Processing.calculate_euclidian_distance(
                start_ball_x, start_ball_y, start_fly_x, start_fly_y
            )
            end_distance = Processing.calculate_euclidian_distance(end_ball_x, end_ball_y, start_fly_x, start_fly_y)

            # Categorize the event based on the change in distance
            if end_distance > start_distance:
                pushing_events.append(event)
            else:
                pulling_events.append(event)

        return pushing_events, pulling_events

    def get_distance_moved(self, fly_idx, ball_idx, subset=None):
        """
        Calculate the total distance moved by the ball for a given fly and ball,
        using the median position of the first and last 10 frames of each event
        to reduce the impact of tracking aberrations.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.
            subset (list, optional): List of events to consider. Defaults to all interaction events.

        Returns:
            float: Total distance moved by the ball.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if subset is None:
            subset = self.tracking_data.interaction_events[fly_idx][ball_idx]

        total_distance = 0

        for event in subset:
            start_idx, end_idx = event[0], event[1]

            # Ensure the event duration is long enough to calculate medians
            if end_idx - start_idx < 10:
                continue

            # Use the helper method to calculate median coordinates
            start_x, start_y, end_x, end_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
            )

            # Calculate the Euclidean distance between the start and end positions
            distance = Processing.calculate_euclidian_distance(start_x, start_y, end_x, end_y)

            total_distance += distance

        return total_distance

    def get_distance_ratio(self, fly_idx, ball_idx, subset=None):
        """
        Calculate the ratio between the maximum distance moved from the start position
        and the total distance moved by the ball. This ratio highlights discrepancies
        caused by behaviors like pulling and pushing the ball repeatedly.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.

        Returns:
            float: The distance ratio. A value closer to 1 indicates consistent movement
                in one direction, while a lower value indicates more back-and-forth movement.
        """
        # Get the maximum distance moved from the start position
        max_distance = self.get_max_distance(fly_idx, ball_idx)

        # Get the total distance moved
        total_distance = self.get_distance_moved(fly_idx, ball_idx, subset=subset)

        # Calculate the ratio
        if total_distance > 0:
            distance_ratio = total_distance / max_distance
        else:
            distance_ratio = np.nan  # Handle cases where total distance is zero

        return distance_ratio

    @lru_cache(maxsize=128)
    def get_major_event(self, fly_idx, ball_idx, distance=None):
        """
        Identify the aha moment for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        distance : float, optional
            Distance threshold for the aha moment (default is None).

        Returns
        -------
        tuple
            Aha moment index and aha moment time.
        """
        if distance is None:
            distance = self.fly.config.major_event_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        major_event = [
            (event, i)
            for i, event in enumerate(self.tracking_data.interaction_events[fly_idx][ball_idx])
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        if major_event:

            major_event_instance, major_event_idx = major_event[0]

            # Calculate raw time first (no adjustments yet)
            raw_first_major_event_time = major_event_instance[0] / self.fly.experiment.fps

            # Apply appropriate time adjustment based on ball identity
            first_major_event_time = self._adjust_time_for_f1_test_ball(raw_first_major_event_time, fly_idx, ball_idx)

            return major_event_idx, first_major_event_time
        else:
            return np.nan, np.nan

    def get_insight_effect(self, fly_idx, ball_idx, epsilon=1e-6, strength_threshold=2, subset=None):
        """
        Calculate enhanced insight effect with performance analytics, optionally on a subset of events.
        """
        if subset is not None:
            significant_events = [event[0] for event in subset]
        else:
            significant_events = [event[0] for event in self.get_significant_events(fly_idx, ball_idx)]
        major_event_index, _ = self.get_major_event(fly_idx, ball_idx)

        if not significant_events:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        if major_event_index is None:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        before_aha = significant_events[: major_event_index + 1]
        after_aha = significant_events[major_event_index + 1 :]

        avg_before = self._calculate_avg_distance(fly_idx, ball_idx, before_aha)
        avg_after = self._calculate_avg_distance(fly_idx, ball_idx, after_aha)

        if major_event_index == 0:
            insight_effect = 1.0
        elif avg_before == 0:
            insight_effect = np.nan
        else:
            insight_effect = avg_after / avg_before

        log_effect = np.log(insight_effect + 1) if insight_effect > 0 else np.nan
        classification = "strong" if insight_effect > strength_threshold else "weak"

        return {
            "raw_effect": insight_effect,
            "log_effect": log_effect,
            "classification": classification,
            "first_event": major_event_index == 0,
            "post_aha_count": len(after_aha),
        }

    def _calculate_avg_distance(self, fly_idx, ball_idx, events):
        """Helper method to safely calculate average distances"""
        if not events:
            return np.nan

        try:
            distances = self.get_distance_moved(fly_idx, ball_idx, subset=events)
            return np.mean(distances) / len(events)
        except (ValueError, ZeroDivisionError):
            return np.nan

    def get_success_direction(self, fly_idx, ball_idx, threshold=None):
        """
        Determine the success direction (push, pull, or both) based on the ball's movement.

        Args:
            fly_idx (int): Index of the fly.
            ball_idx (int): Index of the ball.
            threshold (float, optional): Distance threshold for success. Defaults to the configured threshold.

        Returns:
            str: "push", "pull", "both", or None if no success direction is determined.
        """
        if threshold is None:
            threshold = self.fly.config.success_direction_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Get the median initial position of the ball
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, window=10, keypoint="centre"
        )

        # Calculate the Euclidean distance from the initial position
        ball_data["euclidean_distance"] = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
        )

        # Filter the data for frames where the ball has moved beyond the threshold
        moved_data = ball_data[ball_data["euclidean_distance"] >= threshold]

        if moved_data.empty:
            return None

        # Determine the success direction based on the y-coordinate movement
        if abs(initial_x - self.tracking_data.start_x) < 100:
            pushed = any(moved_data["y_centre"] < initial_y)
            pulled = any(moved_data["y_centre"] > initial_y)
        else:
            pushed = any(moved_data["y_centre"] > initial_y)
            pulled = any(moved_data["y_centre"] < initial_y)

        if pushed and pulled:
            return "both"
        elif pushed:
            return "push"
        elif pulled:
            return "pull"
        else:
            return None

    def detect_pauses(self, fly_idx, threshold=None, window=None, minimum_duration=None):
        """
        Detect pauses in the fly's movement based on skeleton keypoints, excluding pauses
        that occur when the fly is in the chamber.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        threshold : float, optional
            Movement threshold in pixels to consider as a pause (default is 5).
        window : int, optional
            Number of frames to use for calculating movement (default is 5).
        minimum_duration : float, optional
            Minimum duration (in seconds) for a pause to be considered valid (default is 2).

        Returns
        -------
        list of tuple
            List of pauses, where each pause is represented as a tuple (start_time, end_time, duration).
        """
        if threshold is None:
            threshold = self.fly.config.pause_threshold
        if window is None:
            window = self.fly.config.pause_window
        if minimum_duration is None:
            minimum_duration = self.fly.config.pause_min_duration

        # Get the skeleton data for the fly
        skeleton_data = self.tracking_data.skeletontrack.objects[fly_idx].dataset

        # Extract all keypoints (assuming columns are named like "x_<keypoint>" and "y_<keypoint>")
        keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]

        # Initialize a boolean array to track movement
        is_static = np.ones(len(skeleton_data), dtype=bool)

        for keypoint in keypoints:
            # Calculate the velocity (magnitude of movement) for each keypoint over a rolling window
            x_velocity = skeleton_data[f"x_{keypoint}"].diff().rolling(window=window).mean().abs()
            y_velocity = skeleton_data[f"y_{keypoint}"].diff().rolling(window=window).mean().abs()

            # Check if the velocity is below the threshold
            keypoint_static = (x_velocity <= threshold) & (y_velocity <= threshold)

            # Combine with the overall static status
            is_static &= keypoint_static

        # Smooth the static status using a rolling window to avoid noise
        is_static_smoothed = pd.Series(is_static).rolling(window=window, min_periods=1).mean() == 1

        # Identify chunks of static frames
        pauses = []
        start_frame = None

        for i, static in enumerate(is_static_smoothed):
            if static and start_frame is None:
                start_frame = i
            elif not static and start_frame is not None:
                end_frame = i
                duration = (end_frame - start_frame) / self.fly.experiment.fps
                if duration >= minimum_duration:
                    # Check if the fly is in the chamber during the pause
                    fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
                    start_x, start_y, _, _ = self._calculate_median_coordinates(
                        fly_data, start_idx=0, window=10, keypoint="thorax"
                    )
                    distances = Processing.calculate_euclidian_distance(
                        fly_data["x_thorax"].iloc[start_frame:end_frame],
                        fly_data["y_thorax"].iloc[start_frame:end_frame],
                        start_x,
                        start_y,
                    )
                    in_chamber = np.all(distances <= self.fly.config.chamber_radius)

                    if not in_chamber:  # Exclude pauses in the chamber
                        pauses.append((start_frame, end_frame, duration))
                start_frame = None

        # Handle the case where a pause extends to the end of the recording
        if start_frame is not None:
            end_frame = len(is_static_smoothed)
            duration = (end_frame - start_frame) / self.fly.experiment.fps
            if duration >= minimum_duration:
                fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
                start_x, start_y, _, _ = self._calculate_median_coordinates(
                    fly_data, start_idx=0, window=10, keypoint="thorax"
                )
                distances = Processing.calculate_euclidian_distance(
                    fly_data["x_thorax"].iloc[start_frame:end_frame],
                    fly_data["y_thorax"].iloc[start_frame:end_frame],
                    start_x,
                    start_y,
                )
                in_chamber = np.all(distances <= self.fly.config.chamber_radius)

                if not in_chamber:  # Exclude pauses in the chamber
                    pauses.append((start_frame, end_frame, duration))

        # Convert frame indices to timestamps
        pauses_timestamps = [
            (start_frame / self.fly.experiment.fps, end_frame / self.fly.experiment.fps, duration)
            for start_frame, end_frame, duration in pauses
        ]

        if self.fly.config.debugging:
            print(f"Detected pauses for fly {fly_idx}: {pauses_timestamps}")

        return pauses_timestamps

    def get_raw_pauses(self, fly_idx, threshold=None, window=None, minimum_duration=None):
        """
        Get the raw list of pauses detected by detect_pauses without any time threshold filtering.
        This returns the complete list of pauses as detected by the algorithm, allowing for
        post-processing analysis of pause distributions, durations, and filtering strategies.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        threshold : float, optional
            Movement threshold in pixels to consider as a pause. If None, uses default from config.
        window : int, optional
            Number of frames to use for calculating movement. If None, uses default from config.
        minimum_duration : float, optional
            Minimum duration (in seconds) for a pause to be considered valid.
            If None, uses default from config. Set to 0 to get all pauses regardless of duration.

        Returns
        -------
        list of tuple
            List of pauses, where each pause is represented as a tuple (start_time, end_time, duration).
            Times are in seconds from the start of the experiment.
        """
        return self.detect_pauses(fly_idx, threshold=threshold, window=window, minimum_duration=1)

    def compute_pause_metrics(
        self,
        fly_idx,
        subset=None,
        threshold=5,
        window=5,
        minimum_duration=2.0,
        pause_threshold=5.0,
        pause_max_threshold=10.0,
    ):
        """
        Compute the number of pauses and total duration of pauses for a given fly.
        Uses consistent detection parameters across all pause-related metrics.

        Pauses represent medium-term movement cessation events between 5s and 10s duration.
        For comparison with stops (2-5s) and long pauses (≥10s).
        """
        pauses = self.detect_pauses(fly_idx, threshold=threshold, window=window, minimum_duration=minimum_duration)

        if subset is not None:
            # Flatten all intervals in subset to a set of valid time ranges (in seconds)
            valid_ranges = []
            for event in subset:
                start_time = event[0] / self.fly.experiment.fps
                end_time = event[1] / self.fly.experiment.fps
                valid_ranges.append((start_time, end_time))
            # Filter pauses to only those that overlap with any valid range
            filtered_pauses = []
            for pause in pauses:
                pause_start, pause_end, duration = pause
                if any((pause_start < end and pause_end > start) for start, end in valid_ranges):
                    filtered_pauses.append(pause)
            pauses = filtered_pauses

        # Filter for pauses between pause_threshold and pause_max_threshold (exclusive upper bound)
        pause_events = [pause for pause in pauses if pause_threshold <= pause[2] < pause_max_threshold]

        if not pause_events:
            return {
                "nb_pauses": 0,
                "median_pause_duration": np.nan,
                "total_pause_duration": 0.0,
            }

        nb_pauses = len(pause_events)
        pause_durations = [pause[2] for pause in pause_events]
        median_pause_duration = np.median(pause_durations)
        total_pause_duration = sum(pause_durations)

        return {
            "nb_pauses": nb_pauses,
            "median_pause_duration": median_pause_duration,
            "total_pause_duration": total_pause_duration,
        }

    def compute_long_pause_metrics(
        self, fly_idx, subset=None, threshold=5, window=5, minimum_duration=2.0, long_pause_threshold=10.0
    ):
        """
        Compute the number of long pauses using a fixed duration threshold and median long pause duration.
        Long pauses are defined as pauses that are >= long_pause_threshold seconds (default 10s).

        Uses consistent detection parameters with other pause metrics for fair comparison.
        The 10s threshold captures phenotypes where flies show extended freezing behavior.
        For comparison with stops (2-5s) and pauses (5-10s).

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        subset : list, optional
            List of events to consider. If None, all pauses are considered.
        threshold : float, optional
            Movement threshold in pixels to consider as a pause (default is 5).
        window : int, optional
            Number of frames to use for calculating movement (default is 5).
        minimum_duration : float, optional
            Minimum duration (in seconds) for a pause to be considered valid (default is 2).
        long_pause_threshold : float, optional
            Duration threshold (in seconds) for defining long pauses (default is 10.0).
            Pauses >= this threshold are considered "long".

        Returns
        -------
        dict
            Dictionary containing 'nb_long_pauses', 'median_long_pause_duration', and 'total_long_pause_duration'.
        """
        pauses = self.detect_pauses(fly_idx, threshold=threshold, window=window, minimum_duration=minimum_duration)

        if subset is not None:
            # Flatten all intervals in subset to a set of valid time ranges (in seconds)
            valid_ranges = []
            for event in subset:
                start_time = event[0] / self.fly.experiment.fps
                end_time = event[1] / self.fly.experiment.fps
                valid_ranges.append((start_time, end_time))
            # Filter pauses to only those that overlap with any valid range
            filtered_pauses = []
            for pause in pauses:
                pause_start, pause_end, duration = pause
                if any((pause_start < end and pause_end > start) for start, end in valid_ranges):
                    filtered_pauses.append(pause)
            pauses = filtered_pauses

        if not pauses:
            return {
                "nb_long_pauses": 0,
                "median_long_pause_duration": np.nan,
                "total_long_pause_duration": 0.0,
            }

        # Filter for long pauses using fixed threshold
        long_pauses = [pause for pause in pauses if pause[2] >= long_pause_threshold]

        if not long_pauses:
            return {
                "nb_long_pauses": 0,
                "median_long_pause_duration": np.nan,
                "total_long_pause_duration": 0.0,
            }

        nb_long_pauses = len(long_pauses)
        long_pause_durations = [pause[2] for pause in long_pauses]
        median_long_pause_duration = np.median(long_pause_durations)
        total_time_in_long_pauses = sum(long_pause_durations)

        return {
            "nb_long_pauses": nb_long_pauses,
            "median_long_pause_duration": median_long_pause_duration,
            "total_long_pause_duration": total_time_in_long_pauses,
        }

    def has_long_pauses(self, fly_idx, threshold=5, window=5, minimum_duration=2.0, long_pause_threshold=10.0):
        """
        Check if a fly has at least one long pause (>= 10s).

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        threshold : float, optional
            Movement threshold in pixels to consider as a pause (default is 5).
        window : int, optional
            Number of frames to use for calculating movement (default is 5).
        minimum_duration : float, optional
            Minimum duration (in seconds) for a pause to be detected (default is 2.0).
        long_pause_threshold : float, optional
            Duration threshold (in seconds) for defining long pauses (default is 10.0).

        Returns
        -------
        int
            1 if the fly has at least one long pause, 0 otherwise.
        """
        pauses = self.detect_pauses(fly_idx, threshold=threshold, window=window, minimum_duration=minimum_duration)

        # Check if any pause is >= long_pause_threshold
        has_long = any(pause[2] >= long_pause_threshold for pause in pauses)

        return 1 if has_long else 0

    def compute_stop_metrics(
        self, fly_idx, stop_threshold=2.0, stop_max_threshold=5.0, threshold=5, window=5, minimum_duration=2.0
    ):
        """
        Compute comprehensive stop metrics for a given fly.
        Uses the same base detection parameters as other pause metrics for fair comparison.
        Stops represent short-term movement cessation events between 2s and 5s duration.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        stop_threshold : float, optional
            Minimum duration (in seconds) for a pause to be considered a stop (default is 2.0).
        stop_max_threshold : float, optional
            Maximum duration (in seconds) for a pause to be considered a stop (default is 5.0).
        threshold : float, optional
            Movement threshold in pixels to consider as a pause (default is 5).
        window : int, optional
            Number of frames to use for calculating movement (default is 5).
        minimum_duration : float, optional
            Minimum duration (in seconds) for any pause to be detected (default is 2.0).
            For consistency with other pause metrics.

        Returns
        -------
        dict
            Dictionary containing 'nb_stops', 'median_stop_duration', and 'total_stop_duration'.
        """
        pauses = self.detect_pauses(fly_idx, threshold=threshold, window=window, minimum_duration=minimum_duration)

        # Filter pauses that are between stop_threshold and stop_max_threshold (exclusive upper bound)
        stop_events = [pause for pause in pauses if stop_threshold <= pause[2] < stop_max_threshold]

        if self.fly.config.debugging:
            print(
                f"Stop events for fly {fly_idx} (threshold={stop_threshold}s): {len(stop_events)} out of {len(pauses)} total pauses"
            )

        if not stop_events:
            return {
                "nb_stops": 0,
                "median_stop_duration": np.nan,
                "total_stop_duration": 0.0,
            }

        nb_stops = len(stop_events)
        stop_durations = [pause[2] for pause in stop_events]
        median_stop_duration = np.median(stop_durations)
        total_stop_duration = sum(stop_durations)

        return {
            "nb_stops": nb_stops,
            "median_stop_duration": median_stop_duration,
            "total_stop_duration": total_stop_duration,
        }

    def compute_nb_stops(
        self, fly_idx, stop_threshold=2.0, stop_max_threshold=5.0, threshold=5, window=5, minimum_duration=2.0
    ):
        """
        Backward compatibility method that returns only the number of stops.
        Use compute_stop_metrics() for comprehensive metrics.
        """
        stop_metrics = self.compute_stop_metrics(
            fly_idx, stop_threshold, stop_max_threshold, threshold, window, minimum_duration
        )
        return stop_metrics["nb_stops"]

    def compute_interaction_persistence(self, fly_idx, ball_idx, subset=None):
        """
        Compute the interaction persistence for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            The average duration of interaction events in seconds, or np.nan if no events exist.
        """
        # Get the interaction events for the fly and ball
        events = subset if subset is not None else self.tracking_data.interaction_events[fly_idx][ball_idx]
        if not events:
            # Return NaN if there are no interaction events
            return np.nan

        # Calculate the duration of each event
        event_durations = [(event[1] - event[0]) / self.fly.experiment.fps for event in events]

        # Compute the average duration
        average_duration = np.mean(event_durations)

        return average_duration

    @lru_cache(maxsize=128)
    def compute_learning_slope(self, fly_idx, ball_idx):
        """
        Compute the learning slope and R² for a given fly and ball based on ball position over time.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        dict
            Dictionary containing:
            - "slope": float, the slope of the linear regression.
            - "r2": float, the R² of the linear regression.
        """
        # Get the ball position data
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Extract time (in seconds) and position (e.g., y_centre)
        time = np.arange(len(ball_data)) / self.fly.experiment.fps  # Time in seconds
        position = ball_data["y_centre"].values  # Replace with "x_centre" if needed

        # Apply F1 test ball time adjustment if applicable
        if self._should_use_adjusted_time(fly_idx, ball_idx):
            # Adjust time array to be relative to corridor exit
            time = time - self.tracking_data.f1_exit_time
            # Filter out negative times (data before corridor exit)
            valid_mask = time >= 0
            time = time[valid_mask]
            position = position[valid_mask]

        # Check if there is enough data to fit a regression
        if len(time) < 2 or np.all(position == position[0]):  # No movement
            return {"slope": np.nan, "r2": np.nan}

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(time.reshape(-1, 1), position)

        # Calculate R²
        r2 = model.score(time.reshape(-1, 1), position)

        # Return the slope and R²
        return {"slope": model.coef_[0], "r2": r2}

    @lru_cache(maxsize=128)
    def compute_logistic_features(self, fly_idx, ball_idx):
        """
        Compute logistic features and R² for a given fly and ball based on ball position over time.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        dict
            Dictionary containing:
            - "L": float, maximum value (plateau).
            - "k": float, growth rate (steepness of the curve).
            - "t0": float, midpoint (time at which the curve reaches half of L).
            - "r2": float, R² of the logistic regression.
        """
        # Get the ball position data
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Extract time (in seconds) and position (e.g., y_centre)
        time = np.arange(len(ball_data)) / self.fly.experiment.fps  # Time in seconds
        position = ball_data["y_centre"].values  # Replace with "x_centre" if needed

        # Apply F1 test ball time adjustment if applicable
        if self._should_use_adjusted_time(fly_idx, ball_idx):
            # Adjust time array to be relative to corridor exit
            time = time - self.tracking_data.f1_exit_time
            # Filter out negative times (data before corridor exit)
            valid_mask = time >= 0
            time = time[valid_mask]
            position = position[valid_mask]

        # Check if there is enough data to fit a logistic function
        if len(time) < 3 or np.all(position == position[0]):  # No movement
            return {"L": np.nan, "k": np.nan, "t0": np.nan, "r2": np.nan}

        # Initial guesses for logistic parameters
        initial_guess = [position.max(), 1, time.mean()]  # L, k, t0

        try:
            # Fit the logistic function to the data
            params, _ = curve_fit(Processing.logistic_function, time, position, p0=initial_guess, maxfev=10000)
            L, k, t0 = params

            # Calculate predicted values
            predicted = Processing.logistic_function(time, L, k, t0)

            # Calculate R²
            ss_res = np.sum((position - predicted) ** 2)  # Residual sum of squares
            ss_tot = np.sum((position - np.mean(position)) ** 2)  # Total sum of squares
            r2 = 1 - (ss_res / ss_tot)

        except RuntimeError:
            # If the fit fails, return NaN
            return {"L": np.nan, "k": np.nan, "t0": np.nan, "r2": np.nan}

        return {"L": L, "k": k, "t0": t0, "r2": r2}

    def compute_event_influence(self, fly_idx, ball_idx, success_threshold=None):
        """
        Compute the influence of event n-1 on event n for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        success_threshold : float, optional
            Threshold for significant ball displacement to classify an event as successful (default is 5).

        Returns
        -------
        dict
            Dictionary containing:
            - "avg_displacement_after_success": float, average displacement of events following successful events.
            - "avg_displacement_after_failure": float, average displacement of events following unsuccessful events.
            - "influence_ratio": float, ratio of the two averages (success/failure).
        """
        if success_threshold is None:
            success_threshold = self.fly.config.success_threshold

        # Get the interaction events for the fly and ball
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if len(events) < 2:
            # Not enough events to compute influence
            return {
                "avg_displacement_after_success": np.nan,
                "avg_displacement_after_failure": np.nan,
                "influence_ratio": np.nan,
            }

        # Helper function to calculate displacement for an event
        def calculate_displacement(event):
            start_idx, end_idx = event[0], event[1]
            start_x, start_y, end_x, end_y = self._calculate_median_coordinates(
                ball_data, start_idx=start_idx, end_idx=end_idx, window=10, keypoint="centre"
            )
            return Processing.calculate_euclidian_distance(start_x, start_y, end_x, end_y)

        # Classify events as successful or unsuccessful
        event_success = []
        for event in events:
            displacement = calculate_displacement(event)
            event_success.append(displacement >= success_threshold)

        # Analyze the influence of event n-1 on event n
        displacements_after_success = []
        displacements_after_failure = []

        for i in range(1, len(events)):
            displacement_n = calculate_displacement(events[i])
            if event_success[i - 1]:  # If event n-1 was successful
                displacements_after_success.append(displacement_n)
            else:  # If event n-1 was unsuccessful
                displacements_after_failure.append(displacement_n)

        # Compute averages
        avg_displacement_after_success = np.mean(displacements_after_success) if displacements_after_success else 0
        avg_displacement_after_failure = np.mean(displacements_after_failure) if displacements_after_failure else 0

        # Compute influence ratio
        if avg_displacement_after_failure > 0:
            influence_ratio = avg_displacement_after_success / avg_displacement_after_failure
        else:
            influence_ratio = 0

        return {
            "avg_displacement_after_success": avg_displacement_after_success,
            "avg_displacement_after_failure": avg_displacement_after_failure,
            "influence_ratio": influence_ratio,
        }

    def compute_normalized_velocity(self, fly_idx, ball_idx):
        """
        Compute the fly's velocity normalized by the available space.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            The average velocity normalized by the available space.
        """
        # Get the fly and ball data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Calculate the available space as the distance between the ball and the fly's starting position
        initial_ball_x, initial_ball_y, _, _ = self._calculate_median_coordinates(ball_data, start_idx=0, window=10)
        ball_distances = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_ball_x, initial_ball_y
        )

        # Calculate the fly's velocity
        fly_velocity = (
            np.sqrt(fly_data["x_thorax"].diff() ** 2 + fly_data["y_thorax"].diff() ** 2) * self.fly.experiment.fps
        )  # Convert to velocity in pixels/second

        # Normalize velocity by the available space
        normalized_velocity = fly_velocity / (ball_distances + 1e-6)  # Add epsilon to avoid division by zero

        # Return the average normalized velocity
        return np.nanmean(normalized_velocity)

    def compute_velocity_during_interactions(self, fly_idx, ball_idx):
        """
        Compute the fly's average velocity during interaction events.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            The average velocity during interaction events.
        """
        # Get the fly data and interaction events
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]

        if not events:
            return np.nan

        # Calculate the fly's velocity
        fly_velocity = (
            np.sqrt(fly_data["x_thorax"].diff() ** 2 + fly_data["y_thorax"].diff() ** 2) * self.fly.experiment.fps
        )  # Convert to velocity in pixels/second

        # Extract velocity during interaction events
        velocities_during_events = []
        for event in events:
            start_idx, end_idx = event[0], event[1]
            velocities_during_events.extend(fly_velocity[start_idx:end_idx])

        # Return the average velocity during interaction events
        return np.nanmean(velocities_during_events)

    def compute_velocity_trend(self, fly_idx):
        """
        Compute the trend (slope) of the fly's velocity over time.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            The slope of the velocity trend over time.
        """
        # Get the fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate the fly's velocity
        fly_velocity = (
            np.sqrt(fly_data["x_thorax"].diff() ** 2 + fly_data["y_thorax"].diff() ** 2) * self.fly.experiment.fps
        )  # Convert to velocity in pixels/second

        # Remove NaN values
        valid_indices = ~np.isnan(fly_velocity)
        time = np.arange(len(fly_velocity))[valid_indices] / self.fly.experiment.fps  # Time in seconds
        velocity = fly_velocity[valid_indices]

        if len(time) < 2:
            return np.nan

        # Fit a linear regression to the velocity trend
        model = LinearRegression()
        model.fit(time.reshape(-1, 1), velocity)

        # Return the slope of the trend
        return model.coef_[0]

    def compute_binned_slope(self, fly_idx, ball_idx, n_bins=12):
        """
        Compute the slope of the ball position curve in time bins.

        Returns
        -------
        list of float: Slope for each bin.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        position = ball_data["y_centre"].values

        # Apply F1 test ball time adjustment if applicable
        if self._should_use_adjusted_time(fly_idx, ball_idx):
            # Adjust time array to be relative to corridor exit
            time = time - self.tracking_data.f1_exit_time
            # Filter out negative times (data before corridor exit)
            valid_mask = time >= 0
            time = time[valid_mask]
            position = position[valid_mask]

        if len(time) == 0:
            return [np.nan] * n_bins

        bins = np.linspace(time[0], time[-1], n_bins + 1)
        slopes = []

        for i in range(n_bins):
            mask = (time >= bins[i]) & (time < bins[i + 1])
            t_bin = time[mask]
            p_bin = position[mask]
            if len(t_bin) > 1 and not np.all(p_bin == p_bin[0]):
                model = LinearRegression()
                model.fit(t_bin.reshape(-1, 1), p_bin)
                slopes.append(model.coef_[0])
            else:
                slopes.append(np.nan)
        return slopes

    def compute_overall_slope(self, fly_idx, ball_idx):
        """
        Compute the overall slope of the ball trajectory.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        position = ball_data["y_centre"].values

        # Apply F1 test ball time adjustment if applicable
        if self._should_use_adjusted_time(fly_idx, ball_idx):
            # Adjust time array to be relative to corridor exit
            time = time - self.tracking_data.f1_exit_time
            # Filter out negative times (data before corridor exit)
            valid_mask = time >= 0
            time = time[valid_mask]
            position = position[valid_mask]

        if len(time) < 2 or np.all(position == position[0]):
            return np.nan
        model = LinearRegression()
        model.fit(time.reshape(-1, 1), position)
        return model.coef_[0]

    def compute_interaction_rate_by_bin(self, fly_idx, ball_idx, n_bins=12):
        """
        Compute the rate of interactions per time bin.
        Returns a list of rates (events per second) for each bin.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset
        time = np.arange(len(ball_data)) / self.fly.experiment.fps
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]

        # Apply F1 test ball time adjustment if applicable
        if self._should_use_adjusted_time(fly_idx, ball_idx):
            # Adjust time array to be relative to corridor exit
            time = time - self.tracking_data.f1_exit_time
            # Filter out negative times (data before corridor exit)
            valid_mask = time >= 0
            time = time[valid_mask]

            # Adjust event times and filter events that occur after corridor exit
            adjusted_events = []
            for event in events:
                event_time = event[0] / self.fly.experiment.fps - self.tracking_data.f1_exit_time
                if event_time >= 0:  # Only include events after corridor exit
                    adjusted_events.append((event_time * self.fly.experiment.fps, event[1], event[2]))
            events = adjusted_events

        if len(time) == 0:
            return [np.nan] * n_bins

        bins = np.linspace(time[0], time[-1], n_bins + 1)
        rates = []
        for i in range(n_bins):
            bin_start, bin_end = bins[i], bins[i + 1]
            # Count events whose start falls in this bin
            if self._should_use_adjusted_time(fly_idx, ball_idx):
                # Events are already adjusted
                count = sum(
                    (event[0] / self.fly.experiment.fps >= bin_start) and (event[0] / self.fly.experiment.fps < bin_end)
                    for event in events
                )
            else:
                # Use original logic for non-F1 or non-test-ball scenarios
                count = sum(
                    (event[0] / self.fly.experiment.fps >= bin_start) and (event[0] / self.fly.experiment.fps < bin_end)
                    for event in events
                )
            duration = bin_end - bin_start
            rates.append(count / duration if duration > 0 else np.nan)
        return rates

    def compute_overall_interaction_rate(self, fly_idx, ball_idx):
        """
        Compute the overall rate of interactions (events per second).
        """
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]
        duration = self.tracking_data.duration
        return len(events) / duration if duration > 0 else np.nan

    def compute_auc(self, fly_idx, ball_idx):
        """
        Compute the area under the euclidean distance vs. time curve for the ball.
        This represents the cumulative progress made by the fly in moving the ball
        away from its initial position over time.

        Returns
        -------
        float: The total AUC based on distance from initial position.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Get the median initial position of the ball
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, window=10, keypoint="centre"
        )

        # Calculate euclidean distance from initial position for each frame
        euclidean_distances = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
        )

        # Create time array
        time = np.arange(len(ball_data)) / self.fly.experiment.fps

        # Apply F1 test ball time adjustment if applicable
        if self._should_use_adjusted_time(fly_idx, ball_idx):
            # Adjust time array to be relative to corridor exit
            time = time - self.tracking_data.f1_exit_time
            # Filter out negative times (data before corridor exit)
            valid_mask = time >= 0
            time = time[valid_mask]
            euclidean_distances = euclidean_distances[valid_mask]

        if len(time) == 0:
            return np.nan

        # Compute AUC using euclidean distance (progress)
        auc = np.trapz(euclidean_distances, time)
        return auc

    def compute_binned_auc(self, fly_idx, ball_idx, n_bins=12):
        """
        Compute the AUC in each time bin using euclidean distance from initial position.

        Returns
        -------
        list of float: AUC for each bin.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Get the median initial position of the ball
        initial_x, initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, window=10, keypoint="centre"
        )

        # Calculate euclidean distance from initial position for each frame
        euclidean_distances = Processing.calculate_euclidian_distance(
            ball_data["x_centre"], ball_data["y_centre"], initial_x, initial_y
        )

        # Create time array
        time = np.arange(len(ball_data)) / self.fly.experiment.fps

        # Apply F1 test ball time adjustment if applicable
        if self._should_use_adjusted_time(fly_idx, ball_idx):
            # Adjust time array to be relative to corridor exit
            time = time - self.tracking_data.f1_exit_time
            # Filter out negative times (data before corridor exit)
            valid_mask = time >= 0
            time = time[valid_mask]
            euclidean_distances = euclidean_distances[valid_mask]

        if len(time) == 0:
            return [np.nan] * n_bins

        # Create time bins
        bins = np.linspace(time[0], time[-1], n_bins + 1)
        aucs = []

        for i in range(n_bins):
            mask = (time >= bins[i]) & (time < bins[i + 1])
            t_bin = time[mask]
            distance_bin = euclidean_distances[mask]
            if len(t_bin) > 1:
                aucs.append(np.trapz(distance_bin, t_bin))
            else:
                aucs.append(np.nan)
        return aucs

    def get_has_finished(self, fly_idx, ball_idx):
        """
        Check if a final event exists for this fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        int
            1 if a final event exists, 0 otherwise.
        """
        final_event = self.get_final_event(fly_idx, ball_idx)
        return 1 if final_event is not None else 0

    def get_has_major(self, fly_idx, ball_idx):
        """
        Check if a major event exists for this fly and ball.

        Returns
        -------
        int
            1 if a major event exists, 0 otherwise.
        """
        major_event_idx, _ = self.get_major_event(fly_idx, ball_idx)
        return 1 if not (np.isnan(major_event_idx)) else 0

    def get_has_significant(self, fly_idx, ball_idx):
        """
        Check if a significant event exists for this fly and ball.

        Returns
        -------
        int
            1 if a significant event exists, 0 otherwise.
        """
        significant_events = self.get_significant_events(fly_idx, ball_idx)
        return 1 if significant_events and len(significant_events) > 0 else 0

    def compute_persistence_at_end(self, fly_idx):
        """
        Compute the fraction of time the fly spent at a certain distance from start.
        This distance is defined by the corridor_end config parameter.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Fraction of time spent at corridor_end distance from start.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Get the fly start position using median of first 10 frames
        start_x, start_y, _, _ = self._calculate_median_coordinates(fly_data, start_idx=0, window=10, keypoint="thorax")

        # Calculate distance from start position for each frame
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"], fly_data["y_thorax"], start_x, start_y
        )

        # Get corridor_end distance from config (assuming it exists)
        corridor_end_distance = getattr(self.fly.config, "corridor_end_threshold", 170)  # Default to 170 if not set

        # Count frames where fly is at or beyond corridor_end distance
        at_end = distances >= corridor_end_distance

        # Calculate fraction of time
        total_frames = len(distances)
        if total_frames > 0:
            return np.sum(at_end) / total_frames
        else:
            return np.nan

    def compute_fly_distance_moved(self, fly_idx):
        """
        Compute the overall distance moved by the fly over the whole experiment.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Total distance moved by the fly in millimeters.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate frame-to-frame distances
        x_diff = fly_data["x_thorax"].diff()
        y_diff = fly_data["y_thorax"].diff()

        # Calculate Euclidean distance for each frame transition
        frame_distances = np.sqrt(x_diff**2 + y_diff**2)

        # Sum all distances (excluding NaN from first frame)
        total_distance_pixels = np.nansum(frame_distances)

        # Convert from pixels to millimeters
        pixels_per_mm = getattr(self.fly.config, "pixels_per_mm", 500 / 30)  # Default conversion factor
        total_distance_mm = total_distance_pixels / pixels_per_mm

        return total_distance_mm

    def get_time_chamber_beginning(self, fly_idx):
        """
        Compute the time spent by the fly in the chamber during the first 25% of the video.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Time spent in chamber during first 25% of video in seconds.
        """
        # Get fly data
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset

        # Calculate the end frame for first 25% of video
        total_frames = len(fly_data)
        end_frame_25_percent = int(total_frames * 0.25)

        # Get the fly start position using median of first 10 frames
        start_x, start_y, _, _ = self._calculate_median_coordinates(fly_data, start_idx=0, window=10, keypoint="thorax")

        # Calculate distance from start position for first 25% of frames
        distances = Processing.calculate_euclidian_distance(
            fly_data["x_thorax"].iloc[:end_frame_25_percent],
            fly_data["y_thorax"].iloc[:end_frame_25_percent],
            start_x,
            start_y,
        )

        # Determine frames where fly is in chamber
        in_chamber = distances <= self.fly.config.chamber_radius

        # Calculate time spent in chamber
        time_in_chamber = np.sum(in_chamber) / self.fly.experiment.fps

        return time_in_chamber

    def compute_median_stop_duration(self, fly_idx):
        """
        Compute the median duration of stop events for a given fly.
        Uses the same detection parameters as other pause metrics for consistency.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.

        Returns
        -------
        float
            Median duration of stop events in seconds.
        """
        # Get pause metrics which include individual pause durations
        # Use consistent parameters with other metrics (minimum_duration=2.0)
        pauses = self.detect_pauses(fly_idx, minimum_duration=2.0)

        if not pauses:
            return np.nan

        # Extract durations from pause tuples (start_time, end_time, duration)
        durations = [pause[2] for pause in pauses]

        # Calculate median duration
        median_duration = np.median(durations)

        return median_duration

    def compute_fraction_not_facing_ball(self, fly_idx, ball_idx, angle_threshold=30):
        """
        Compute the fraction of time when the fly is not facing the ball direction
        (end of corridor) while outside the chamber.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        angle_threshold : float, optional
            Angle threshold in degrees above which the fly is considered not facing the ball (default is 30).

        Returns
        -------
        float
            Fraction of time (0-1) when the fly is not facing the ball while outside the chamber.
        """
        # Get skeleton and fly tracking data
        skeleton_data = self.tracking_data.skeletontrack.objects[fly_idx].dataset
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # Check that we have required keypoints (case-insensitive check)
        required_keypoints = ["Head", "Thorax", "Abdomen"]  # Use proper case
        available_keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]

        # Create a mapping for case-insensitive matching
        keypoint_mapping = {}
        for available in available_keypoints:
            for required in required_keypoints:
                if available.lower() == required.lower():
                    keypoint_mapping[required] = available
                    break

        if len(keypoint_mapping) < len(required_keypoints):
            if self.fly.config.debugging:
                missing = [kp for kp in required_keypoints if kp not in keypoint_mapping]
                print(f"Missing required keypoints: {missing}. Available: {available_keypoints}")
            return 0.0  # Return 0.0 instead of np.nan - assume always facing if can't calculate

        # Get fly start position for chamber detection (use thorax keypoint)
        thorax_keypoint = keypoint_mapping["Thorax"]
        start_x, start_y, _, _ = self._calculate_median_coordinates(
            fly_data, start_idx=0, window=10, keypoint=thorax_keypoint.lower()
        )

        # Calculate distances from start position to determine when fly is outside chamber
        distances = Processing.calculate_euclidian_distance(
            fly_data[f"x_{thorax_keypoint.lower()}"], fly_data[f"y_{thorax_keypoint.lower()}"], start_x, start_y
        )
        outside_chamber = distances > self.fly.config.chamber_radius

        # Get ball initial position to determine corridor direction
        ball_initial_x, ball_initial_y, _, _ = self._calculate_median_coordinates(
            ball_data, start_idx=0, window=10, keypoint="centre"
        )

        # Check for None values or NaN from median calculation
        if (
            start_x is None
            or start_y is None
            or ball_initial_x is None
            or ball_initial_y is None
            or pd.isna(start_x)
            or pd.isna(start_y)
            or pd.isna(ball_initial_x)
            or pd.isna(ball_initial_y)
        ):
            if self.fly.config.debugging:
                print(
                    f"Could not determine start or ball positions: start=({start_x}, {start_y}), ball=({ball_initial_x}, {ball_initial_y})"
                )
            return np.nan

        # Determine corridor direction based on experimental setup
        # If ball starts close to fly start position, corridor goes in negative y direction
        # Otherwise, corridor direction is determined by ball position relative to start
        if abs(ball_initial_x - start_x) < 100:
            # Standard setup: corridor goes down (negative y)
            corridor_direction = np.array([0, -1])
        else:
            # F1 setup or different orientation
            corridor_direction = np.array([ball_initial_x - start_x, ball_initial_y - start_y])
            corridor_direction = corridor_direction / np.linalg.norm(corridor_direction)

        # Calculate fly body orientation for each frame - OPTIMIZED VERSION
        try:
            # Get body keypoints using the mapped names - vectorized approach
            head_x = skeleton_data[f"x_{keypoint_mapping['Head']}"].values
            head_y = skeleton_data[f"y_{keypoint_mapping['Head']}"].values
            thorax_x = skeleton_data[f"x_{keypoint_mapping['Thorax']}"].values
            thorax_y = skeleton_data[f"y_{keypoint_mapping['Thorax']}"].values
            abdomen_x = skeleton_data[f"x_{keypoint_mapping['Abdomen']}"].values
            abdomen_y = skeleton_data[f"y_{keypoint_mapping['Abdomen']}"].values

            # Create body vectors (from abdomen to head) - vectorized
            body_vectors = np.column_stack([head_x - abdomen_x, head_y - abdomen_y])

            # Calculate norms - vectorized
            body_norms = np.linalg.norm(body_vectors, axis=1)

            # Normalize body vectors - vectorized (avoid division by zero)
            valid_norms = body_norms > 0
            normalized_body_vectors = np.zeros_like(body_vectors)
            normalized_body_vectors[valid_norms] = body_vectors[valid_norms] / body_norms[valid_norms, np.newaxis]

            # Calculate dot products with corridor direction - vectorized
            dot_products = np.dot(normalized_body_vectors, corridor_direction)

            # Clamp dot products and calculate angles - vectorized
            dot_products = np.clip(dot_products, -1.0, 1.0)
            angles_rad = np.arccos(dot_products)
            fly_orientations = np.degrees(angles_rad)

            # Set invalid orientations to NaN
            invalid_mask = (
                np.isnan(head_x)
                | np.isnan(head_y)
                | np.isnan(thorax_x)
                | np.isnan(thorax_y)
                | np.isnan(abdomen_x)
                | np.isnan(abdomen_y)
                | (body_norms == 0)
            )
            fly_orientations[invalid_mask] = np.nan

        except Exception as e:
            if self.fly.config.debugging:
                print(f"Error calculating orientations: {e}")
            return 0.0

        # Determine frames where fly is not facing the ball and is outside chamber
        not_facing_ball = fly_orientations > angle_threshold
        outside_and_not_facing = outside_chamber & not_facing_ball & ~np.isnan(fly_orientations)

        # Calculate fraction of valid frames outside chamber where fly is not facing ball
        valid_outside_frames = outside_chamber & ~np.isnan(fly_orientations)

        if np.sum(valid_outside_frames) == 0:
            # No valid frames outside chamber - assume always facing if we can't measure
            return 0.0

        fraction_not_facing = np.sum(outside_and_not_facing) / np.sum(valid_outside_frames)

        if self.fly.config.debugging:
            print(
                f"Fly {fly_idx}: {np.sum(outside_and_not_facing)}/{np.sum(valid_outside_frames)} frames not facing ball ({fraction_not_facing:.3f})"
            )

        return fraction_not_facing

    def compute_flailing(self, fly_idx, ball_idx):
        """
        Compute the average motion energy of left and right front legs during interaction events.

        This metric measures how much the fly's front legs are moving during interactions,
        which can indicate flailing behavior.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Average motion energy of front legs during interactions, or np.nan if no data.
        """
        # Get skeleton data using cached method
        skeleton_data = self._get_skeleton_data(fly_idx)

        # Use the leg keypoints from config
        config_leg_names = [name for name in self.fly.config.contact_nodes if name not in ["Head"]]

        available_keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]
        available_legs = [leg for leg in config_leg_names if leg in available_keypoints]

        if self.fly.config.debugging:
            print(f"Available keypoints: {available_keypoints}")
            print(f"Found leg keypoints: {available_legs}")

        # If no specific leg keypoints found, use a fallback approach with any available keypoints
        if not available_legs:
            # Use any available keypoints as a fallback (exclude head/thorax for leg-like movement)
            exclude_keypoints = ["Head", "Thorax", "centre", "center"]
            fallback_keypoints = [kp for kp in available_keypoints if kp not in exclude_keypoints]

            if len(fallback_keypoints) >= 2:
                available_legs = fallback_keypoints[:2]  # Take first 2 available
                if self.fly.config.debugging:
                    print(f"Using fallback keypoints for flailing: {available_legs}")
            else:
                if self.fly.config.debugging:
                    print(f"No suitable keypoints found for flailing metric")
                return 0.0  # Return 0.0 instead of NaN to indicate no flailing detected

        # Get interaction events up to final event
        events = self.tracking_data.interaction_events[fly_idx][ball_idx]

        # Get final event to filter interactions
        final_event = self.get_final_event(fly_idx, ball_idx)
        if final_event is not None:
            final_event_idx, _, _ = final_event
            if final_event_idx is not None and final_event_idx >= 0:
                events = events[: final_event_idx + 1]

        if not events:
            if self.fly.config.debugging:
                print(f"No interaction events found for flailing calculation")
            return 0.0  # Return 0.0 instead of NaN when no events

        # Calculate motion energy for each available leg during interactions
        leg_motion_energies = []

        for leg in available_legs:
            x_col = f"x_{leg}"
            y_col = f"y_{leg}"

            # Calculate velocity (frame-to-frame displacement) for this leg
            x_velocity = skeleton_data[x_col].diff().fillna(0)
            y_velocity = skeleton_data[y_col].diff().fillna(0)

            # Motion energy is the magnitude of velocity
            motion_energy = np.sqrt(x_velocity**2 + y_velocity**2)

            # Extract motion energy during interaction events
            event_motion_energies = []
            for event in events:
                start_frame, end_frame = event[0], event[1]
                event_motion = motion_energy.iloc[start_frame : end_frame + 1]

                # Skip events with insufficient data
                if len(event_motion) == 0:
                    continue

                # Calculate mean motion energy for this event
                mean_event_motion = event_motion.mean()
                if not np.isnan(mean_event_motion):
                    event_motion_energies.append(mean_event_motion)

            # Add this leg's average motion energy across all events
            if event_motion_energies:
                leg_motion_energies.append(np.mean(event_motion_energies))

        # Return average motion energy across all available legs
        if leg_motion_energies:
            average_flailing = np.mean(leg_motion_energies)

            if self.fly.config.debugging:
                print(
                    f"Fly {fly_idx}: Flailing computed from {len(available_legs)} legs, {len(events)} events: {average_flailing:.3f}"
                )

            return average_flailing
        else:
            # Return 0.0 instead of NaN to indicate no flailing detected
            if self.fly.config.debugging:
                print(f"Fly {fly_idx}: No leg motion detected during interactions")
            return 0.0

    def compute_head_pushing_ratio(self, fly_idx, ball_idx):
        """
        Compute the ratio of head pushing vs leg pushing based on contact events.

        Uses frame-by-frame analysis within each contact to detect temporal patterns.
        For each contact, analyzes individual frames to distinguish:
        - Head pushing: Head consistently closer to ball throughout contact
        - Leg pushing: Legs become visible and closer to ball, especially at contact end
        - Mixed: Both head and leg pushing detected within same contact

        The behavior when legs are not visible depends on config.exclude_hidden_legs:
        - If exclude_hidden_legs=True: Exclude contacts where legs have <10% visibility
        - If exclude_hidden_legs=False: Assume head pushing when legs are not visible

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Ratio of head pushing contacts (0-1), or np.nan if no contact data or
            all contacts were excluded due to poor leg visibility.
        """
        # Use cached skeleton metrics instead of creating new ones
        skeleton_metrics = self._get_skeleton_metrics()
        if skeleton_metrics is None:
            return np.nan

        contact_events = skeleton_metrics.contacts
        if not contact_events:
            if self.fly.config.debugging:
                print("No contact events found")
            return np.nan

        # Get skeleton and ball data using cached methods
        skeleton_data = self._get_skeleton_data(fly_idx)
        ball_data = self._get_ball_data(ball_idx)

        # Use keypoints from config
        available_keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]

        # Head is required
        if "Head" not in available_keypoints:
            if self.fly.config.debugging:
                print("Head keypoint not found, cannot compute head pushing ratio")
            return np.nan

        # Get leg keypoints from config (exclude Head)
        config_leg_names = [name for name in self.fly.config.contact_nodes if name != "Head"]
        available_legs = [leg for leg in config_leg_names if leg in available_keypoints]

        head_pushing_count = 0
        leg_pushing_count = 0
        mixed_pushing_count = 0  # Track contacts with both head and leg pushing
        excluded_count = 0  # Track how many contacts were excluded due to hidden legs
        min_contact_frames = 10  # Only consider contacts with at least 10 frames
        total_contacts = len(contact_events)
        filtered_contacts = 0

        for contact_event in contact_events:
            start_frame, end_frame = contact_event[0], contact_event[1]

            # Get frame-by-frame analysis for this contact
            contact_frames = list(range(start_frame, min(end_frame + 1, len(skeleton_data))))

            if len(contact_frames) < min_contact_frames:
                if self.fly.config.debugging:
                    print(
                        f"Skipping short contact {contact_event} with {len(contact_frames)} frames (min required: {min_contact_frames})"
                    )
                filtered_contacts += 1
                continue

            # Frame-by-frame analysis
            head_pushing_frames = 0
            leg_pushing_frames = 0
            valid_comparison_frames = 0
            legs_visible_frames = 0

            for frame in contact_frames:
                # Get head position for this frame
                head_x = skeleton_data.loc[frame, "x_Head"]
                head_y = skeleton_data.loc[frame, "y_Head"]

                # Get ball position for this frame
                ball_x = ball_data.loc[frame, "x_centre"]
                ball_y = ball_data.loc[frame, "y_centre"]

                # Skip frames with invalid head or ball data
                if any(pd.isna([head_x, head_y, ball_x, ball_y])):
                    continue

                # Calculate head-to-ball distance
                head_to_ball_distance = np.sqrt((head_x - ball_x) ** 2 + (head_y - ball_y) ** 2)

                # Check leg positions for this frame
                frame_leg_distances = []
                frame_legs_visible = False

                for leg in available_legs:
                    leg_x = skeleton_data.loc[frame, f"x_{leg}"]
                    leg_y = skeleton_data.loc[frame, f"y_{leg}"]

                    # Check if leg is visible: coordinates must not be NaN
                    if not pd.isna(leg_x) and not pd.isna(leg_y):
                        frame_legs_visible = True
                        leg_to_ball_distance = np.sqrt((leg_x - ball_x) ** 2 + (leg_y - ball_y) ** 2)
                        frame_leg_distances.append(leg_to_ball_distance)

                if frame_legs_visible:
                    legs_visible_frames += 1

                # Only make comparison if we have leg data for this frame
                if len(frame_leg_distances) > 0:
                    valid_comparison_frames += 1
                    min_leg_distance = min(frame_leg_distances)

                    if head_to_ball_distance <= min_leg_distance:
                        head_pushing_frames += 1
                    else:
                        leg_pushing_frames += 1

            # Calculate visibility ratio for this contact
            leg_visibility_ratio = legs_visible_frames / len(contact_frames)

            # Determine classification based on frame-by-frame analysis
            if self.fly.config.exclude_hidden_legs:
                # Exclude contacts with insufficient leg visibility
                if leg_visibility_ratio < 0.1 or valid_comparison_frames == 0:
                    excluded_count += 1
                    if self.fly.config.debugging:
                        print(
                            f"Excluding contact {contact_event} - leg visibility: {leg_visibility_ratio:.2f}, "
                            f"valid frames: {valid_comparison_frames}"
                        )
                    continue
            else:
                # If no leg visibility, default to head pushing (original behavior)
                if valid_comparison_frames == 0:
                    head_pushing_count += 1
                    continue

            # Classify contact based on frame-by-frame analysis
            head_ratio = head_pushing_frames / max(valid_comparison_frames, 1)
            leg_ratio = leg_pushing_frames / max(valid_comparison_frames, 1)

            # Use majority voting with temporal pattern consideration
            if head_ratio > self.fly.config.head_pushing_threshold:  # Predominantly head pushing
                head_pushing_count += 1
            elif leg_ratio > self.fly.config.head_pushing_threshold:  # Predominantly leg pushing
                leg_pushing_count += 1
            else:  # Mixed or ambiguous - check temporal patterns
                # Look for leg pushing at the end of contact (configurable window)
                contact_length = len(contact_frames)
                window_start = int((1 - self.fly.config.late_contact_window) * contact_length)
                late_frames = contact_frames[window_start:]

                late_leg_pushing = 0
                late_valid_frames = 0

                for frame in late_frames:
                    head_x = skeleton_data.loc[frame, "x_Head"]
                    head_y = skeleton_data.loc[frame, "y_Head"]
                    ball_x = ball_data.loc[frame, "x_centre"]
                    ball_y = ball_data.loc[frame, "y_centre"]

                    if any(pd.isna([head_x, head_y, ball_x, ball_y])):
                        continue

                    head_to_ball_distance = np.sqrt((head_x - ball_x) ** 2 + (head_y - ball_y) ** 2)

                    frame_leg_distances = []
                    for leg in available_legs:
                        leg_x = skeleton_data.loc[frame, f"x_{leg}"]
                        leg_y = skeleton_data.loc[frame, f"y_{leg}"]

                        # Check if leg is visible: coordinates must not be NaN
                        if not pd.isna(leg_x) and not pd.isna(leg_y):
                            leg_to_ball_distance = np.sqrt((leg_x - ball_x) ** 2 + (leg_y - ball_y) ** 2)
                            frame_leg_distances.append(leg_to_ball_distance)
                            leg_to_ball_distance = np.sqrt((leg_x - ball_x) ** 2 + (leg_y - ball_y) ** 2)
                            frame_leg_distances.append(leg_to_ball_distance)

                    if len(frame_leg_distances) > 0:
                        late_valid_frames += 1
                        if min(frame_leg_distances) < head_to_ball_distance:
                            late_leg_pushing += 1

                # If significant leg pushing detected in final third, classify as leg pushing
                if late_valid_frames > 0 and (late_leg_pushing / late_valid_frames) > 0.5:
                    leg_pushing_count += 1
                    if self.fly.config.debugging:
                        print(
                            f"Contact {contact_event} classified as leg pushing based on late-contact pattern "
                            f"({late_leg_pushing}/{late_valid_frames} late frames)"
                        )
                else:
                    # Default to head pushing for ambiguous cases
                    head_pushing_count += 1

        # Calculate ratio
        total_contacts = head_pushing_count + leg_pushing_count

        if total_contacts == 0:
            if self.fly.config.debugging:
                total_analyzed = len(contact_events)
                print(
                    f"Fly {fly_idx}: No valid contacts for head pushing analysis. "
                    f"Total contacts: {total_analyzed}, Excluded: {excluded_count}, "
                    f"exclude_hidden_legs: {self.fly.config.exclude_hidden_legs}"
                )
            return np.nan

        head_pushing_ratio = head_pushing_count / total_contacts

        if self.fly.config.debugging:
            total_analyzed = len(contact_events)
            used_contacts = total_contacts - filtered_contacts
            print(
                f"Fly {fly_idx}: {head_pushing_count} head, {leg_pushing_count} leg pushing contacts "
                f"({head_pushing_ratio:.3f} ratio). "
                f"Total contacts: {total_analyzed}, Used: {used_contacts}, Filtered short: {filtered_contacts}, Excluded hidden legs: {excluded_count}, "
                f"exclude_hidden_legs: {self.fly.config.exclude_hidden_legs}"
            )
            print(f"  Frame-by-frame temporal analysis used for better leg vs head distinction")

        return head_pushing_ratio

    def compute_leg_visibility_ratio(self, fly_idx, ball_idx):
        """
        Compute the weighted ratio of front leg visibility during contact events.

        This metric measures the weighted fraction of front leg visibility during contact events,
        where each frame is scored based on the number of visible front legs (Lfront, Rfront):
        - 0 legs visible = 0 points
        - 1 leg visible = 1 point
        - 2 legs visible = 2 points

        The final ratio is: total_score / (total_frames * num_available_legs)

        This provides more nuanced insight into leg visibility patterns during interactions
        and helps distinguish between head pushing (low leg visibility) and leg pushing
        (high leg visibility).

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Weighted ratio (0-1) where 1.0 means both legs are always visible,
            0.5 means on average one leg is visible, and 0.0 means no legs are visible.
            Returns np.nan if no contact data available.
        """
        # Import skeleton metrics to get contact events
        try:
            from .skeleton_metrics import SkeletonMetrics

            skeleton_metrics = SkeletonMetrics(self.fly)
            contact_events = skeleton_metrics.contacts
        except ImportError:
            if self.fly.config.debugging:
                print("Could not import SkeletonMetrics for contact detection")
            return np.nan
        except Exception as e:
            if self.fly.config.debugging:
                print(f"Error creating SkeletonMetrics: {e}")
            return np.nan

        if not contact_events:
            if self.fly.config.debugging:
                print("No contact events found for leg visibility analysis")
            return np.nan

        # Get skeleton data
        skeleton_data = self.tracking_data.skeletontrack.objects[fly_idx].dataset
        available_keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]

        # Specifically look for front leg keypoints
        front_leg_names = ["Lfront", "Rfront"]
        available_front_legs = [leg for leg in front_leg_names if leg in available_keypoints]

        if not available_front_legs:
            if self.fly.config.debugging:
                print(f"No front leg keypoints found. Available: {available_keypoints}, Looking for: {front_leg_names}")
            return np.nan

        if self.fly.config.debugging:
            print(f"Found front leg keypoints: {available_front_legs}")

        # Get all left and right keypoints for sideways detection
        left_keypoints = [kp for kp in available_keypoints if kp.startswith("L")]
        right_keypoints = [kp for kp in available_keypoints if kp.startswith("R")]

        if self.fly.config.debugging:
            print(f"Left keypoints: {left_keypoints}")
            print(f"Right keypoints: {right_keypoints}")

        total_contact_frames = 0
        total_visibility_score = 0  # Weighted visibility score
        min_contact_frames = 10  # Only consider contacts with at least 10 frames
        total_contacts = len(contact_events)
        filtered_contacts = 0
        sideways_contacts = 0  # Track contacts filtered due to sideways orientation

        for contact_event in contact_events:
            start_frame, end_frame = contact_event[0], contact_event[1]

            # Get all frames for this contact
            contact_frames = list(range(start_frame, min(end_frame + 1, len(skeleton_data))))

            if len(contact_frames) < min_contact_frames:
                if self.fly.config.debugging:
                    print(
                        f"Skipping short contact {contact_event} with {len(contact_frames)} frames (min required: {min_contact_frames})"
                    )
                filtered_contacts += 1
                continue

            # Check if animal is sideways during this contact
            # Sample some frames to check orientation (check every 5th frame or so)
            sample_frames = contact_frames[:: max(1, len(contact_frames) // 5)]  # Sample ~5 frames
            sideways_frames = 0

            for frame in sample_frames:
                # Count visible left and right keypoints
                visible_left = 0
                visible_right = 0

                for kp in left_keypoints:
                    x_val = skeleton_data.loc[frame, f"x_{kp}"]
                    y_val = skeleton_data.loc[frame, f"y_{kp}"]
                    if not pd.isna(x_val) and not pd.isna(y_val):
                        visible_left += 1

                for kp in right_keypoints:
                    x_val = skeleton_data.loc[frame, f"x_{kp}"]
                    y_val = skeleton_data.loc[frame, f"y_{kp}"]
                    if not pd.isna(x_val) and not pd.isna(y_val):
                        visible_right += 1

                # Animal is sideways if we see ≤2 keypoints on one side
                if visible_left <= 2 or visible_right <= 2:
                    sideways_frames += 1

            # Skip contact if more than half the sampled frames show sideways orientation
            sideways_ratio = sideways_frames / len(sample_frames)
            if sideways_ratio > 0.5:
                if self.fly.config.debugging:
                    print(
                        f"Skipping sideways contact {contact_event}: {sideways_frames}/{len(sample_frames)} frames sideways "
                        f"(ratio: {sideways_ratio:.2f})"
                    )
                sideways_contacts += 1
                continue

            total_contact_frames += len(contact_frames)

            # Check leg visibility for each frame in this contact with weighted scoring
            for frame in contact_frames:
                visible_legs_count = 0

                for leg in available_front_legs:
                    leg_x = skeleton_data.loc[frame, f"x_{leg}"]
                    leg_y = skeleton_data.loc[frame, f"y_{leg}"]

                    # Check if leg is visible: coordinates must not be NaN
                    if not pd.isna(leg_x) and not pd.isna(leg_y):
                        visible_legs_count += 1

                # Add weighted score based on number of visible legs
                # 0 legs visible = 0 points
                # 1 leg visible = 1 point
                # 2 legs visible = 2 points
                total_visibility_score += visible_legs_count

        # Calculate overall visibility ratio
        if total_contact_frames == 0:
            if self.fly.config.debugging:
                print("No valid contact frames found for leg visibility analysis")
            return np.nan

        # Maximum possible score is 2 points per frame (both legs visible)
        max_possible_score = total_contact_frames * len(available_front_legs)
        visibility_ratio = total_visibility_score / max_possible_score

        if self.fly.config.debugging:
            used_contacts = total_contacts - filtered_contacts - sideways_contacts
            avg_legs_visible = total_visibility_score / total_contact_frames if total_contact_frames > 0 else 0
            print(
                f"Fly {fly_idx}: Front leg visibility ratio: {total_visibility_score}/{max_possible_score} weighted score "
                f"({visibility_ratio:.3f}, avg {avg_legs_visible:.2f} legs/frame) across {used_contacts}/{total_contacts} contact events "
                f"(filtered {filtered_contacts} short contacts, {sideways_contacts} sideways contacts) using {available_front_legs}"
            )

        return visibility_ratio

    def compute_median_head_ball_distance(self, fly_idx, ball_idx):
        """
        Compute the median distance between the fly's head and the ball during contact events.

        This metric provides a simple and robust measure to distinguish head-pushing behavior:
        - Lower values indicate flies that keep their head closer to the ball (head pushing)
        - Higher values indicate flies that maintain greater head-ball distance (leg pushing)

        Uses median instead of mean to be more robust against outliers and noise.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Median distance between head and ball during contact events in pixels,
            or np.nan if no contact data available.
        """
        # Import skeleton metrics to get contact events
        try:
            from .skeleton_metrics import SkeletonMetrics

            skeleton_metrics = SkeletonMetrics(self.fly)
            contact_events = skeleton_metrics.contacts
        except ImportError:
            if self.fly.config.debugging:
                print("Could not import SkeletonMetrics for contact detection")
            return np.nan
        except Exception as e:
            if self.fly.config.debugging:
                print(f"Error creating SkeletonMetrics: {e}")
            return np.nan

        if not contact_events:
            if self.fly.config.debugging:
                print("No contact events found for head-ball distance analysis")
            return np.nan

        # Get skeleton and ball data (use raw ball data for better accuracy)
        skeleton_data = self.tracking_data.skeletontrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.raw_balltrack.objects[ball_idx].dataset

        # Check if Head keypoint is available
        available_keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]
        if "Head" not in available_keypoints:
            if self.fly.config.debugging:
                print("Head keypoint not found, cannot compute head-ball distance")
            return np.nan

        all_head_ball_distances = []
        min_contact_frames = 5  # Only consider contacts with at least 5 frames
        total_contacts = len(contact_events)
        filtered_contacts = 0

        for contact_event in contact_events:
            start_frame, end_frame = contact_event[0], contact_event[1]

            # Get all frames for this contact
            contact_frames = list(range(start_frame, min(end_frame + 1, len(skeleton_data))))

            if len(contact_frames) < min_contact_frames:
                if self.fly.config.debugging:
                    print(
                        f"Skipping short contact {contact_event} with {len(contact_frames)} frames (min required: {min_contact_frames})"
                    )
                filtered_contacts += 1
                continue

            # Calculate head-ball distance for each frame in this contact
            contact_distances = []

            for frame in contact_frames:
                # Get head position for this frame
                head_x = skeleton_data.loc[frame, "x_Head"]
                head_y = skeleton_data.loc[frame, "y_Head"]

                # Get ball position for this frame
                ball_x = ball_data.loc[frame, "x_centre"]
                ball_y = ball_data.loc[frame, "y_centre"]

                # Skip frames with invalid head or ball data
                if any(pd.isna([head_x, head_y, ball_x, ball_y])):
                    continue

                # Calculate Euclidean distance between head and ball
                distance = np.sqrt((head_x - ball_x) ** 2 + (head_y - ball_y) ** 2)
                contact_distances.append(distance)

            # Add all distances from this contact to the overall list
            all_head_ball_distances.extend(contact_distances)

        # Calculate median distance across all contact events
        if not all_head_ball_distances:
            if self.fly.config.debugging:
                print("No valid head-ball distance measurements found")
            return np.nan

        median_distance = np.median(all_head_ball_distances)

        if self.fly.config.debugging:
            used_contacts = total_contacts - filtered_contacts
            print(
                f"Fly {fly_idx}: Median head-ball distance: {median_distance:.2f} pixels "
                f"from {len(all_head_ball_distances)} measurements across {used_contacts}/{total_contacts} contact events "
                f"(filtered {filtered_contacts} short contacts)"
            )

        return median_distance

    def compute_mean_head_ball_distance(self, fly_idx, ball_idx):
        """
        Compute the mean distance between the fly's head and the ball during contact events.

        This complements the median distance metric and can provide additional insights:
        - Mean is more sensitive to extreme values (very close or very far contacts)
        - Useful for detecting flies that occasionally get very close to the ball
        - Can be compared with median to understand the distribution shape

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Mean distance between head and ball during contact events in pixels,
            or np.nan if no contact data available.
        """
        # Import skeleton metrics to get contact events
        try:
            from .skeleton_metrics import SkeletonMetrics

            skeleton_metrics = SkeletonMetrics(self.fly)
            contact_events = skeleton_metrics.contacts
        except ImportError:
            if self.fly.config.debugging:
                print("Could not import SkeletonMetrics for contact detection")
            return np.nan
        except Exception as e:
            if self.fly.config.debugging:
                print(f"Error creating SkeletonMetrics: {e}")
            return np.nan

        if not contact_events:
            if self.fly.config.debugging:
                print("No contact events found for head-ball distance analysis")
            return np.nan

        # Get skeleton and ball data (use raw ball data for better accuracy)
        skeleton_data = self.tracking_data.skeletontrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.raw_balltrack.objects[ball_idx].dataset

        # Check if Head keypoint is available
        available_keypoints = [col.split("_")[1] for col in skeleton_data.columns if col.startswith("x_")]
        if "Head" not in available_keypoints:
            if self.fly.config.debugging:
                print("Head keypoint not found, cannot compute head-ball distance")
            return np.nan

        all_head_ball_distances = []
        min_contact_frames = 5  # Only consider contacts with at least 5 frames
        total_contacts = len(contact_events)
        filtered_contacts = 0

        for contact_event in contact_events:
            start_frame, end_frame = contact_event[0], contact_event[1]

            # Get all frames for this contact
            contact_frames = list(range(start_frame, min(end_frame + 1, len(skeleton_data))))

            if len(contact_frames) < min_contact_frames:
                filtered_contacts += 1
                continue

            # Calculate head-ball distance for each frame in this contact
            for frame in contact_frames:
                # Get head position for this frame
                head_x = skeleton_data.loc[frame, "x_Head"]
                head_y = skeleton_data.loc[frame, "y_Head"]

                # Get ball position for this frame
                ball_x = ball_data.loc[frame, "x_centre"]
                ball_y = ball_data.loc[frame, "y_centre"]

                # Skip frames with invalid head or ball data
                if any(pd.isna([head_x, head_y, ball_x, ball_y])):
                    continue

                # Calculate Euclidean distance between head and ball
                distance = np.sqrt((head_x - ball_x) ** 2 + (head_y - ball_y) ** 2)
                all_head_ball_distances.append(distance)

        # Calculate mean distance across all contact events
        if not all_head_ball_distances:
            if self.fly.config.debugging:
                print("No valid head-ball distance measurements found")
            return np.nan

        mean_distance = np.mean(all_head_ball_distances)

        if self.fly.config.debugging:
            used_contacts = total_contacts - filtered_contacts
            median_distance = np.median(all_head_ball_distances)
            std_distance = np.std(all_head_ball_distances)
            print(
                f"Fly {fly_idx}: Mean head-ball distance: {mean_distance:.2f} ± {std_distance:.2f} pixels "
                f"(median: {median_distance:.2f}) from {len(all_head_ball_distances)} measurements "
                f"across {used_contacts}/{total_contacts} contact events"
            )

        return mean_distance
