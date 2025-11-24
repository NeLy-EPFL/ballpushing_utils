# Ball Pushing Metrics Documentation

This document provides a comprehensive explanation of all metrics computed by the `BallPushingMetrics` class for analyzing fly-ball interaction behavior.

## Overview

The Ball Pushing Metrics system analyzes behavioral data from experiments where flies interact with balls in controlled environments. Each metric captures different aspects of the fly's learning and motor behavior during ball manipulation tasks.

**Pixel-to-millimeter conversion**: All thresholds are given in both pixels and millimeters. The conversion factor is 1 pixel = 0.06 mm (based on 30 mm = 500 pixels).

## Core Event Definitions

### Basic Event Types

These are metrics that are used as tools to generate the summary metrics.

- **Interaction Event**: A period where the fly is in close proximity to the ball and potentially manipulating it
  - *Close proximity threshold*: ≤ 45 pixels (2.7 mm) distance between fly and ball
- **Significant Event**: An interaction event that results in significant ball displacement
  - *Threshold*: > 5 pixels (0.3 mm) ball displacement
- **First Major Event**: The first significant event that crosses a higher threshold, often considered the "aha moment"
  - *Threshold*: ≥ 20 pixels (1.2 mm) ball displacement
- **Final Event**: The last interaction event that achieves the final distance threshold before task completion
  - *Threshold*: 170 pixels (10.2 mm) for standard experiments, 100 pixels (6.0 mm) for F1 experiments
  - *Note*: This represents the last significant interaction that moves the ball toward the task goal

## Temporal Metrics

These metrics capture how behavior changes over time:

### Event Counts and Timing

#### `fly_distance_moved`

**Description**: Total distance traveled by fly throughout experiment
**Calculation**: Sum of frame-to-frame Euclidean distances for fly position
**Units**: millimeters
# Ball Pushing Metrics

This document provides a comprehensive explanation of all metrics computed by the `BallPushingMetrics` class for analyzing fly-ball interaction behavior.

## Overview

The Ball Pushing Metrics system analyzes behavioral data from experiments where flies interact with balls in controlled environments. Each metric captures different aspects of the fly's learning and motor behavior during ball manipulation tasks.

**Pixel-to-millimeter conversion**: All thresholds are given in both pixels and millimeters. The conversion factor is 1 pixel = 0.06 mm (based on 30 mm = 500 pixels).

## Core Event Definitions

### Basic Event Types

These are metrics that are used as tools to generate the summary metrics.

- **Interaction Event**: A period where the fly is in close proximity to the ball and potentially manipulating it

  - *Close proximity threshold*: ≤ 45 pixels (2.7 mm) distance between fly and ball

- **Significant Event**: An interaction event that results in significant ball displacement

  - *Threshold*: > 5 pixels (0.3 mm) ball displacement

- **First Major Event**: The first significant event that crosses a higher threshold, often considered the "aha moment"

  - *Threshold*: ≥ 20 pixels (1.2 mm) ball displacement

- **Final Event**: The last interaction event that achieves the final distance threshold before task completion
# Ball Pushing Metrics (current)

This README documents the metrics produced by `BallPushingMetrics` as implemented in `ballpushing_metrics.py`. It removes duplicates and reflects the current code paths, units, and configurable thresholds.

Note on configuration and gating
- Each metric is computed only if enabled via `fly.config.enabled_metrics` (None means compute all).
- Many thresholds are configurable in `fly.config` (values below are typical defaults; use your config for ground truth).
- Pixel-to-millimeter conversion uses `fly.config.pixels_per_mm` (default ≈ 16.67 px/mm → 1 px ≈ 0.06 mm).

Core event concepts (as used by the code)
- Interaction event: a contiguous contact/proximity bout between fly and ball (precomputed in `tracking_data`).
- Significant event: event with ball movement exceeding `config.significant_threshold` (in pixels) along y.
- Major event ("aha"): first event exceeding `config.major_event_threshold` (pixels).
- Final event: last event where ball distance from start reaches the threshold. Threshold is chosen per ball identity:
  - Standard: `config.final_event_threshold` (e.g., 170 px)
  - F1 test ball: `config.final_event_F1_threshold` (e.g., 100 px)

Identification fields
- fly_idx, ball_idx, ball_identity ("training", "test", or None)

Event counts and timing
- nb_events: Adjusted number of interaction events normalized by available time (float).
- nb_significant_events: Count of significant events up to the final event.
- significant_ratio: nb_significant_events / number of events considered (0–1).
- max_event / max_event_time: Index/time of event closest to maximum displacement (s).
- first_significant_event / first_significant_event_time: Index/time of first significant event (s).
- first_major_event / first_major_event_time: Index/time of first major (aha) event (s).
- final_event / final_event_time: Index/time (s) of last event reaching the final threshold.
- has_significant / has_major / has_finished: 1 if such event exists, else 0.
- exit_time / chamber_exit_time: Raw timing from tracking_data (s).

Spatial manipulation metrics
- max_distance: Max euclidean distance of ball from its start (pixels).
- distance_moved: Sum of euclidean distances across events using median start/end frames (pixels).
- distance_ratio: distance_moved / max_distance (≥ 1 indicates efficient directional movement).
- success_direction: "push", "pull", "both", or None based on movement past `config.success_direction_threshold`.

Directionality counts (significant events only)
- pushed: Number of pushing events.
- pulled: Number of pulling events.
- pulling_ratio: pulled / (pushed + pulled) in [0,1].

Interaction amount and persistence
- interaction_proportion: Fraction of session spent in interaction (0–1).
- interaction_persistence: Mean duration of interaction events (s).
- cumulated_breaks_duration: Total duration of gaps between events that overlap analyzed intervals (frames).

Velocity and trends
- normalized_velocity: Fly velocity normalized by available space (dimensionless).
- velocity_during_interactions: Mean fly speed during interactions (px/s).
- velocity_trend: Slope of velocity vs time (px/s²).
- overall_slope: Slope of ball y-position vs time (px/s).
- overall_interaction_rate: Events per second across the whole session (events/s).

Learning and curve fitting
- learning_slope / learning_slope_r2: Linear fit of ball y-position over time.
- logistic_L / logistic_k / logistic_t0 / logistic_r2: Logistic fit parameters and R² of ball y-position.
- event_influence: Dict with
  - avg_displacement_after_success
  - avg_displacement_after_failure
  - influence_ratio (= success/failure)

Area under curve (AUC) and binning
- auc: ∫ distance_from_start dt using euclidean distance (pixel·seconds).
- binned_slope_[0..11]: Slope per time bin (px/s).
- interaction_rate_bin_[0..11]: Events per second per bin (events/s).
- binned_auc_[0..11]: AUC per bin (pixel·seconds).

Chamber and spatial occupancy
# Ball Pushing Metrics (current)
- chamber_time: Time in chamber radius (s).
- chamber_ratio: chamber_time / total analyzed time (0–1).
- time_chamber_beginning: Time in chamber during first 25% of video (s).
- persistence_at_end: Fraction of frames at/after `config.corridor_end_threshold` (0–1).
- Pauses (medium): 5–10 s
  - nb_pauses, median_pause_duration, total_pause_duration
Skeleton-derived metrics (computed only when skeleton data is available)
- fraction_not_facing_ball: Fraction of time outside chamber the fly is not facing corridor direction (angle > 30°).
- flailing: Mean motion energy of front legs during interactions (higher = more flailing).
- head_pushing_ratio: Fraction of contacts dominated by head pushing (0–1). Honors `config.exclude_hidden_legs`.

Fly movement (global)
- success_direction is particularly informative in F1 where pulling may achieve success.

- pixels_per_mm, enabled_metrics, debugging
- exclude_hidden_legs, contact_nodes, head_pushing_threshold, late_contact_window

Version
**Units**: pixel·seconds


**Description**: Average duration of interaction events

**Units**: seconds

**Description**: Whether major breakthrough occurred in first event

**Interpretation**: Immediate vs. gradual task discovery
## Temporal Structure Metrics
### Activity Patterns
#### `cumulated_breaks_duration`
# Ball Pushing Metrics (current)
# Ball Pushing Metrics (current)
**Description**: Total time between interaction events

**Interpretation**: Rest/planning time between manipulation attempts



**Description**: Weighted ratio of front leg visibility during contact events


**Description**: When fly left experimental chamber
*Note*: Exit time is also used to compute timing based metrics such as `final_event_time`
## Threshold Summary
The following table summarizes all thresholds used in the metrics:
**Description**: Efficiency of ball manipulation


**Interpretation**: 0.5 = balanced, > 0.5 = pulling more often than pushing, < 0.5 = pushing more often than pulling.

*Note*: This metric is particularly relevant for F1 experiments where the task can be achieved by pulling the ball as well as pushing it.
## Movement and Locomotion Metrics
### Velocity Analysis



#### `velocity_during_interactions`


#### `velocity_trend`
**Description**: Total distance traveled by fly throughout experiment

### Spatial Distribution

**Calculation**: Duration within chamber radius during first quarter of video


**Calculation**: Proportion of frames where fly is at or beyond corridor end threshold distance

#### `interaction_proportion`

**Description**: Fraction of time spent interacting with ball
**Calculation**: Total interaction duration / experiment duration
**Range**: 0.0 to 1.0
**Interpretation**: Task engagement level
### Freeze and Pause Behavior

#### `number_of_pauses` / `total_pause_duration`

**Units**: count / seconds


#### `nb_freeze`


**Units**: count


#### `median_freeze_duration`
**Description**: Median duration of locomotor pause events

**Calculation**: Median of all detected pause episode durations
## Learning and Strategy Metrics

### Learning Dynamics

**Units**: pixels/second / R²

- `t0`: Midpoint time (50% achievement)

- `r2`: Model fit quality

**Interpretation**: Captures S-shaped learning curves typical of skill acquisition
**Description**: Global behavioral trends

**Calculation**: Overall ball displacement rate / total interaction frequency

**Units**: pixels/second / events/second

**Interpretation**: Summary measures of performance and activity

#### `auc`

**Description**: Total area under ball position curve

**Calculation**: Integral of ball Y-position over entire experiment

**Units**: pixel·seconds

**Interpretation**: Cumulative manipulation achievement

### Strategy and Persistence

#### `interaction_persistence`

**Description**: Average duration of interaction events

**Calculation**: Mean duration of all interaction episodes

**Units**: seconds

**Interpretation**: Longer values indicate sustained manipulation attempts

#### `major_event_first`

**Description**: Whether major breakthrough occurred in first event

**Values**: True/False

**Calculation**: Boolean indicating if first_major_event index = 0

**Interpretation**: Immediate vs. gradual task discovery

## Temporal Structure Metrics

### Activity Patterns

#### `cumulated_breaks_duration`

**Description**: Total time between interaction events

**Calculation**: Sum of all inter-event intervals

**Units**: seconds

**Interpretation**: Rest/planning time between manipulation attempts

## Behavioral Strategy Metrics

These metrics analyze specific behavioral patterns and strategies used by flies during ball manipulation:

### Body Orientation

#### `fraction_not_facing_ball`

**Description**: Fraction of time when fly is not facing the ball direction while outside chamber

**Calculation**: Proportion of frames where fly's body orientation deviates more than 30° from corridor direction (toward ball) when outside starting chamber

**Range**: 0.0 to 1.0

**Interpretation**: Higher values indicate distraction or lack of directional focus; lower values suggest goal-directed behavior

### Motor Behavior Patterns

#### `flailing`

**Description**: Average motion energy of front legs during interaction events

**Calculation**: Motion energy computed as sum of squared velocity differences for leg keypoints during ball interactions

**Units**: dimensionless motion energy

**Interpretation**: Higher values indicate more energetic leg movement during interactions, potentially reflecting struggle or inefficient manipulation

#### `head_pushing_ratio`

**Description**: Proportion of contacts where head is used for pushing rather than legs

**Calculation**: Frame-by-frame analysis during contact events to determine whether head or legs are closer to ball

**Range**: 0.0 to 1.0

**Interpretation**: 1.0 = pure head pushing strategy, 0.0 = pure leg pushing strategy, 0.5 = mixed strategy

### Contact Analysis

#### `median_head_ball_distance`

**Description**: Median distance between fly head and ball during contact events

**Calculation**: Median Euclidean distance across all contact frames

**Units**: pixels

**Interpretation**: Lower values indicate head-pushing behavior; higher values suggest leg-pushing with head maintained at distance

#### `mean_head_ball_distance`

**Description**: Mean distance between fly head and ball during contact events

**Calculation**: Average Euclidean distance across all contact frames

**Units**: pixels

**Interpretation**: Complements median distance; comparison reveals distribution shape of head-ball distances

#### `leg_visibility_ratio`

**Description**: Weighted ratio of front leg visibility during contact events

**Calculation**: Weighted score based on number of visible front legs per frame during contacts (0-2 legs visible)

**Range**: 0.0 to 1.0

**Interpretation**: Higher values indicate better leg tracking quality and potentially more leg-based manipulation

## Task Completion Metrics

### Achievement Status

#### `has_finished`

**Description**: Binary indicator of task completion

**Calculation**: 1 if final event exists (ball moved to completion threshold), 0 otherwise

**Values**: 0 or 1

**Interpretation**: Simple binary measure of whether fly successfully completed the ball-pushing task

#### `has_major`

**Description**: Binary indicator of major event achievement

**Calculation**: 1 if a major event was detected, 0 otherwise

**Values**: 0 or 1

**Interpretation**: Indicates whether the fly achieved a major breakthrough event during the experiment

#### `has_significant`

**Description**: Binary indicator of significant event achievement

**Calculation**: 1 if at least one significant event (ball displacement > 5 pixels / 0.3 mm) was detected, 0 otherwise

**Values**: 0 or 1

**Interpretation**: Indicates whether the fly achieved any significant ball manipulation event

## Experimental Context Metrics

### Timing References

#### `exit_time` / `chamber_exit_time`

**Description**: When fly left experimental chamber

*Note*: Exit time is also used to compute timing based metrics such as `final_event_time`

**Calculation**: Timestamp when fly moved beyond chamber radius

**Units**: seconds

**Interpretation**: Later exit_time can indicate struggle to exit or lack of motivation to explore.

## Threshold Summary

The following table summarizes all thresholds used in the metrics:

| Threshold Type | Pixels | Millimeters | Degrees | Used For |
|---|---|---|---|---|
| **Interaction proximity** | ≤ 45 | ≤ 2.7 mm | - | Detecting when fly is close enough to ball for interaction events |
| **Significant event** | > 5 | > 0.3 mm | - | Significant ball displacement; used for significant events, push/pull classification |
| **Major event** | ≥ 20 | ≥ 1.2 mm | - | First major breakthrough ("aha moment") |
| **Success direction** | ≥ 25 | ≥ 1.5 mm | - | Determining successful manipulation direction (push/pull/both) |
| **Final event (standard)** | 170 | 10.2 mm | - | Task completion threshold for standard experiments |
| **Final event (F1)** | 100 | 6.0 mm | - | Task completion threshold for F1 experiments (second part) |
| **Body orientation** | - | - | 30° | Angle deviation from corridor direction for `fraction_not_facing_ball` |

## Statistical tests

### Broad exploration approach

For broad exploration of different experiments, we realised a Mann-whitney test for each metric to compare conditions to controls as it doesn't assume normal distribution and thus can be used with our data generally. In conditions where more than two conditions were compared, we applied a FDR multiple comparisons correction.

As an alternative, especially for heavily skewed data, we also generated permutation test of the metrics tested, which is more robust to skewed data but for the mean yields similar results.
