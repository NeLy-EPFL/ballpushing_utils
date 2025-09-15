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

**Interpretation**: Overall locomotor activity and exploration behavior

#### `nb_events`

**Description**: Total number of interaction events

**Calculation**: Count of all detected fly-ball interaction periods, adjusted for experiment duration

**Range**: 0 to ∞

**Interpretation**: Higher values indicate more frequent interaction attempts

#### `time_chamber_beginning`

**Description**: Time spent in starting chamber during first 25% of experiment

**Calculation**: Duration within chamber radius during first quarter of video

**Units**: seconds

**Interpretation**: Early chamber attachment behavior; higher values suggest reluctance to explore

#### `persistence_at_end`

**Description**: Fraction of time spent at corridor end (goal area)

**Calculation**: Proportion of frames where fly is at or beyond corridor end threshold distance

**Range**: 0.0 to 1.0

**Interpretation**: Goal-directed persistence; higher values indicate staying near task completion area

#### `interaction_proportion`

**Description**: Fraction of time spent interacting with ball

**Calculation**: Total interaction duration / experiment duration

**Range**: 0.0 to 1.0

**Interpretation**: Task engagement level

#### `nb_significant_events`

**Description**: Number of significant interaction events

**Calculation**: Count of events where ball displacement exceeds threshold (5 pixels / 0.3 mm)

**Range**: 0 to `nb_events`

#### `number_of_pauses` / `total_pause_duration`

**Description**: Locomotor pause analysis

**Calculation**: Count and duration of periods with minimal movement

**Units**: count / seconds

**Interpretation**: Behavioral arrest episodes, potentially indicating decision-making

**Interpretation**: Measures successful manipulation attempts vs. mere contact

#### `significant_ratio`

**Description**: Proportion of successful interactions

**Calculation**: `nb_significant_events / nb_events`

**Range**: 0.0 to 1.0

**Interpretation**: Higher values indicate better task efficiency

### Key Event Timing

#### `max_event` / `max_event_time`

**Description**: Event/time when maximum ball displacement was achieved

*Note*: max events are fly specific, each fly has its own peak.

**Calculation**: Event index and timestamp of the interaction that moved the ball furthest

**Units**: Index / seconds

**Interpretation**: Indicates when peak performance was reached

#### `first_significant_event` / `first_significant_event_time`

**Description**: First successful ball manipulation

**Calculation**: Index and timestamp of first event exceeding displacement threshold

**Units**: Index / seconds

**Interpretation**: Learning onset time - how quickly the fly discovers the task

#### `first_major_event` / `first_major_event_time`

**Description**: The "aha moment" - first major breakthrough event

**Calculation**: Event preceding the first significant displacement above major threshold (≥ 20 pixels / 1.2 mm)

**Units**: Index / seconds

**Interpretation**: Moment of behavioral insight or strategy change

#### `final_event` / `final_event_time`

**Description**: Last significant interaction before task completion

**Calculation**: Final event achieving distance threshold (170 pixels / 10.2 mm for standard experiments, 100 pixels / 6.0 mm for F1 experiments) before fly exits chamber

**Units**: Index / seconds

**Interpretation**: Task completion time and final achievement level

### Temporal Dynamics

#### `binned_slope_[0-11]`

**Description**: Ball position slope in 12 time bins

**Calculation**: Linear regression slope of ball Y-position within each time bin

**Units**: pixels/second

**Interpretation**: Captures learning dynamics - how manipulation strategy evolves over time

#### `interaction_rate_bin_[0-11]`

**Description**: Interaction frequency in 12 time bins

**Calculation**: Number of interaction events per second within each time bin

**Units**: events/second

**Interpretation**: Shows activity patterns - when the fly is most/least active

#### `binned_auc_[0-11]`

**Description**: Area under ball position curve in 12 time bins

**Calculation**: Integral of ball Y-position over time within each bin

**Units**: pixel·seconds

**Interpretation**: Cumulative displacement progress over time

## Spatial Metrics

These metrics describe the spatial aspects of ball manipulation:

### Distance and Displacement

#### `max_distance`

**Description**: Maximum ball displacement from start position

**Calculation**: Euclidean distance of furthest ball position from initial location

**Units**: pixels

**Interpretation**: Peak manipulation achievement

#### `distance_moved`

**Description**: Total distance ball was moved during all interactions

*Note*: This differs from max_distance as it also counts pulling distances.

**Calculation**: Sum of Euclidean distances between start/end positions of each event

**Units**: pixels

**Interpretation**: Total mechanical work performed

#### `distance_ratio`

**Description**: Efficiency of ball manipulation

**Calculation**: `distance_moved / max_distance`

**Range**: ≥ 1.0

**Interpretation**: Values close to 1.0 indicate efficient, directional movement; higher values suggest back-and-forth manipulation

### Directional Behavior

#### `pushed` / `pulled`

**Description**: Count of significant pushing vs. pulling events

**Calculation**: Events classified by whether ball moves away from or toward fly's starting position, using significant event threshold (5 pixels / 0.3 mm)

**Units**: count

**Interpretation**: Reveals manipulation strategy preferences for significant interactions

#### `pulling_ratio`

**Description**: Preference for significant pulling vs. pushing events

**Calculation**: `pulled / (pushed + pulled)` where both events exceed significant threshold (5 pixels / 0.3 mm)

**Range**: 0.0 to 1.0

**Interpretation**: 0.5 = balanced, > 0.5 = pulling more often than pushing, < 0.5 = pushing more often than pulling.

#### `success_direction`

**Description**: Primary successful manipulation direction

*Note*: This metric is particularly relevant for F1 experiments where the task can be achieved by pulling the ball as well as pushing it.

**Values**: "push", "pull", "both", or None

**Calculation**: Direction that achieved threshold displacement (25 pixels / 1.5 mm)

**Interpretation**: Identifies the fly's successful strategy

## Movement and Locomotion Metrics

### Velocity Analysis

#### `normalized_velocity`

**Description**: Fly velocity normalized by available space

**Calculation**: Average velocity divided by ball-start distance

**Units**: dimensionless

**Interpretation**: Speed relative to workspace size

#### `velocity_during_interactions`

**Description**: Average fly speed during ball contact

**Calculation**: Mean velocity during all interaction events

**Units**: pixels/second

**Interpretation**: Locomotor activity level during manipulation

#### `velocity_trend`

**Description**: Change in fly velocity over time

**Calculation**: Linear regression slope of velocity vs. time

**Units**: pixels/second²

**Interpretation**: Positive = accelerating, negative = decelerating over session

#### `fly_distance_moved`

**Description**: Total distance traveled by fly throughout experiment

**Calculation**: Sum of frame-to-frame Euclidean distances for fly position

**Units**: millimeters

**Interpretation**: Overall locomotor activity and exploration behavior

### Spatial Distribution

#### `chamber_time` / `chamber_ratio`

**Description**: Time spent in starting chamber area

**Calculation**: Duration within defined radius of start position / total time

**Units**: seconds / ratio (0.0-1.0)

**Interpretation**: Higher values indicate more conservative exploration

#### `time_chamber_beginning`

**Description**: Time spent in starting chamber during first 25% of experiment

**Calculation**: Duration within chamber radius during first quarter of video

**Units**: seconds

**Interpretation**: Early chamber attachment behavior; higher values suggest reluctance to explore

#### `persistence_at_end`

**Description**: Fraction of time spent at corridor end (goal area)

**Calculation**: Proportion of frames where fly is at or beyond corridor end threshold distance

**Range**: 0.0 to 1.0

**Interpretation**: Goal-directed persistence; higher values indicate staying near task completion area

#### `interaction_proportion`

**Description**: Fraction of time spent interacting with ball

**Calculation**: Total interaction duration / experiment duration

**Range**: 0.0 to 1.0

**Interpretation**: Task engagement level

### Freeze and Pause Behavior

#### `number_of_pauses` / `total_pause_duration`

**Description**: Locomotor pause analysis

**Calculation**: Count and duration of periods with minimal movement, with some time threshold (default: 5 seconds)

**Units**: count / seconds

**Interpretation**: Behavioral arrest episodes, potentially indicating decision-making

#### `nb_freeze`

**Description**: Number of freezing events, i.e., short pauses (> 2s)

**Calculation**: Compute pauses without minimal threshold and keep all pauses >2s.

**Units**: count

**Interpretation**: Brief stops, can be fearful events, can be similar to pauses.

#### `median_freeze_duration`

**Description**: Median duration of locomotor pause events

**Calculation**: Median of all detected pause episode durations

**Units**: seconds

**Interpretation**: Typical duration of behavioral arrest periods; longer values may indicate decision-making or fatigue

## Learning and Strategy Metrics

### Learning Dynamics

#### `learning_slope` / `learning_slope_r2`

**Description**: Overall learning trend in ball positioning

**Calculation**: Linear regression of ball Y-position vs. time

**Units**: pixels/second / R²

**Interpretation**: Positive slope = successful directional movement; R² = consistency

#### `logistic_L` / `logistic_k` / `logistic_t0` / `logistic_r2`

**Description**: Logistic growth model parameters for ball position

**Calculation**: Fit of sigmoid function to ball position over time

- `L`: Maximum displacement (plateau)

- `k`: Learning rate (steepness)

- `t0`: Midpoint time (50% achievement)

- `r2`: Model fit quality

**Interpretation**: Captures S-shaped learning curves typical of skill acquisition

#### `overall_slope` / `overall_interaction_rate`

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


**Description**: Efficiency of ball manipulation

**Calculation**: `distance_moved / max_distance`

**Range**: ≥ 1.0

**Interpretation**: Values close to 1.0 indicate efficient, directional movement; higher values suggest back-and-forth manipulation

### Directional Behavior

#### `pushed` / `pulled`

**Description**: Count of significant pushing vs. pulling events

**Calculation**: Events classified by whether ball moves away from or toward fly's starting position, using significant event threshold (5 pixels / 0.3 mm)

**Units**: count

**Interpretation**: Reveals manipulation strategy preferences for significant interactions

#### `pulling_ratio`

**Description**: Preference for significant pulling vs. pushing events

**Calculation**: `pulled / (pushed + pulled)` where both events exceed significant threshold (5 pixels / 0.3 mm)

**Range**: 0.0 to 1.0

**Interpretation**: 0.5 = balanced, > 0.5 = pulling more often than pushing, < 0.5 = pushing more often than pulling.

#### `success_direction`

**Description**: Primary successful manipulation direction

*Note*: This metric is particularly relevant for F1 experiments where the task can be achieved by pulling the ball as well as pushing it.

**Values**: "push", "pull", "both", or None

**Calculation**: Direction that achieved threshold displacement (25 pixels / 1.5 mm)

**Interpretation**: Identifies the fly's successful strategy

## Movement and Locomotion Metrics

### Velocity Analysis

#### `normalized_velocity`

**Description**: Fly velocity normalized by available space

**Calculation**: Average velocity divided by ball-start distance

**Units**: dimensionless

**Interpretation**: Speed relative to workspace size

#### `velocity_during_interactions`

**Description**: Average fly speed during ball contact

**Calculation**: Mean velocity during all interaction events

**Units**: pixels/second

**Interpretation**: Locomotor activity level during manipulation

#### `velocity_trend`

**Description**: Change in fly velocity over time

**Calculation**: Linear regression slope of velocity vs. time

**Units**: pixels/second²

**Interpretation**: Positive = accelerating, negative = decelerating over session

#### `fly_distance_moved`

**Description**: Total distance traveled by fly throughout experiment

**Calculation**: Sum of frame-to-frame Euclidean distances for fly position

**Units**: millimeters

**Interpretation**: Overall locomotor activity and exploration behavior

### Spatial Distribution

#### `chamber_time` / `chamber_ratio`

**Description**: Time spent in starting chamber area

**Calculation**: Duration within defined radius of start position / total time

**Units**: seconds / ratio (0.0-1.0)

**Interpretation**: Higher values indicate more conservative exploration

#### `time_chamber_beginning`

**Description**: Time spent in starting chamber during first 25% of experiment

**Calculation**: Duration within chamber radius during first quarter of video

**Units**: seconds

**Interpretation**: Early chamber attachment behavior; higher values suggest reluctance to explore

#### `persistence_at_end`

**Description**: Fraction of time spent at corridor end (goal area)

**Calculation**: Proportion of frames where fly is at or beyond corridor end threshold distance

**Range**: 0.0 to 1.0

**Interpretation**: Goal-directed persistence; higher values indicate staying near task completion area

#### `interaction_proportion`

**Description**: Fraction of time spent interacting with ball

**Calculation**: Total interaction duration / experiment duration

**Range**: 0.0 to 1.0

**Interpretation**: Task engagement level

### Freeze and Pause Behavior

#### `number_of_pauses` / `total_pause_duration`

**Description**: Locomotor pause analysis

**Calculation**: Count and duration of periods with minimal movement, with some time threshold (default: 5 seconds)

**Units**: count / seconds

**Interpretation**: Behavioral arrest episodes, potentially indicating decision-making

#### `nb_freeze`

**Description**: Number of freezing events, i.e., short pauses (> 2s)

**Calculation**: Compute pauses without minimal threshold and keep all pauses >2s.

**Units**: count

**Interpretation**: Brief stops, can be fearful events, can be similar to pauses.

#### `median_freeze_duration`

**Description**: Median duration of locomotor pause events

**Calculation**: Median of all detected pause episode durations

**Units**: seconds

**Interpretation**: Typical duration of behavioral arrest periods; longer values may indicate decision-making or fatigue

## Learning and Strategy Metrics

### Learning Dynamics

#### `learning_slope` / `learning_slope_r2`

**Description**: Overall learning trend in ball positioning

**Calculation**: Linear regression of ball Y-position vs. time

**Units**: pixels/second / R²

**Interpretation**: Positive slope = successful directional movement; R² = consistency

#### `logistic_L` / `logistic_k` / `logistic_t0` / `logistic_r2`

**Description**: Logistic growth model parameters for ball position

**Calculation**: Fit of sigmoid function to ball position over time

- `L`: Maximum displacement (plateau)

- `k`: Learning rate (steepness)

- `t0`: Midpoint time (50% achievement)

- `r2`: Model fit quality

**Interpretation**: Captures S-shaped learning curves typical of skill acquisition

#### `overall_slope` / `overall_interaction_rate`

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
