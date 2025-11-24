# Ball Pushing Metrics Documentation

This document provides a comprehensive explanation of all metrics computed by the `BallPushingMetrics` class for analyzing fly-ball interaction behavior.

## Overview

The Ball Pushing Metrics system analyzes behavioral data from experiments where flies interact with balls in controlled environments. Each metric captures different aspects of the fly's learning and motor behavior during ball manipulation tasks.

**Pixel-to-millimeter conversion**: All thresholds are given in both pixels and millimeters. The conversion factor is 1 pixel = 0.06 mm (based on 30 mm = 500 pixels).

## Core Event Definitions

### Basic Event Types

These are the fundamental building blocks used to generate summary metrics:

- **Interaction Event**: A period where the fly is in close proximity to the ball and potentially manipulating it
  - *Close proximity threshold*: ≤ 45 pixels (2.7 mm) distance between fly and ball
- **Significant Event**: An interaction event that results in significant ball displacement
  - *Threshold*: > 5 pixels (0.3 mm) ball displacement
- **Major Event**: The first significant event that crosses a higher threshold, often considered the "aha moment"
  - *Threshold*: ≥ 20 pixels (1.2 mm) ball displacement
- **Final Event**: The last interaction event that achieves the final distance threshold before task completion
  - *Threshold*: 170 pixels (10.2 mm) for standard experiments, 100 pixels (6.0 mm) for F1 experiments

## 1. Task Performance Metrics

### Success and Completion

#### `has_finished`
**Description**: Binary indicator of task completion
**Calculation**: 1 if a final event exists (ball reached final distance threshold), 0 otherwise
**Range**: 0 or 1
**Interpretation**: Simple success indicator independent of timing or quality metrics

#### `max_distance`
**Description**: Maximum ball displacement from start position
**Calculation**: Euclidean distance of furthest ball position from initial location
**Units**: pixels
**Interpretation**: Peak distance achievement

### Learning and Breakthrough Events

#### `has_major`
**Description**: Binary indicator of breakthrough event
**Calculation**: 1 if a major event exists (first event ≥ 20 pixels displacement), 0 otherwise
**Range**: 0 or 1
**Interpretation**: Indicates whether fly performed a major push.

#### `first_major_event` / `first_major_event_time`
**Description**: The "aha moment" - first major breakthrough event
**Calculation**: Event preceding the first significant displacement above major threshold (≥ 20 pixels / 1.2 mm)
**Units**: Index / seconds
**Interpretation**: Moment of behavioral insight or strategy change

#### `max_event` / `max_event_time`
**Description**: Event/time when maximum ball displacement was achieved
**Calculation**: Event index and timestamp of the interaction that moved the ball furthest
**Units**: Index / seconds
**Interpretation**: Indicates when peak performance was reached

#### `final_event` / `final_event_time`
**Description**: The last interaction event that achieves the final distance threshold
**Calculation**: Event index and timestamp when ball first crosses the final distance threshold (170 pixels / 10.2 mm for standard experiments, 100 pixels / 6.0 mm for F1 experiments)
**Units**: Index / seconds
**Interpretation**: Moment of task completion; indicates when the fly successfully completed the ball-pushing task

## 2. Interaction Behavior Metrics

### Event Frequency and Success Rate

#### `nb_events`
**Description**: Total number of interaction events
**Calculation**: Count of all detected fly-ball interaction periods, adjusted for experiment duration and time to exit the chamber.
**Range**: 0 to ∞
**Interpretation**: Higher values indicate more frequent interaction attempts

#### `has_significant`
**Description**: Binary indicator of meaningful interaction
**Calculation**: 1 if any significant events exist (> 5 pixels displacement), 0 otherwise
**Range**: 0 or 1
**Interpretation**: Distinguishes flies that achieved meaningful ball manipulation from those with only minor contacts

#### `nb_significant_events`
**Description**: Number of significant interaction events
**Calculation**: Count of events where ball displacement exceeds threshold (5 pixels / 0.3 mm)
**Range**: 0 to `nb_events`

#### `significant_ratio`
**Description**: Proportion of successful interactions
**Calculation**: `nb_significant_events / total_nb_events`
**Range**: 0.0 to 1.0
**Interpretation**: Higher values indicate better task efficiency

#### `first_significant_event` / `first_significant_event_time`
**Description**: First event with significant ball displacement
**Calculation**: Event index and timestamp of the first interaction event that moves the ball more than the significant threshold (> 5 pixels / 0.3 mm)
**Units**: Index / seconds
**Interpretation**: Indicates when the fly first achieved meaningful ball manipulation; earlier times may suggest faster learning or engagement

#### `overall_interaction_rate`

**Description**: Global rate of interaction with the ball

**Calculation**: total interaction frequency

**Units**: events/second

**Interpretation**: Summary measures of activity level

### Ball Manipulation Patterns

#### `distance_moved`
**Description**: Total distance ball was moved during all interactions
**Calculation**: Sum of Euclidean distances between start/end positions of each event
**Units**: pixels
**Interpretation**: Total movements, including pulling distance

#### `distance_ratio`
**Description**: Efficiency of ball manipulation
**Calculation**: `distance_moved / max_distance`
**Range**: ≥ 1.0
**Interpretation**: Values close to 1.0 indicate efficient, directional movement; higher values suggest back-and-forth manipulation

### Directional Strategy

#### `pushed` / `pulled`
**Description**: Count of significant pushing vs. pulling events
**Calculation**: Events classified by whether ball moves away from or toward fly's starting position, using significant event threshold (5 pixels / 0.3 mm)
**Units**: count
**Interpretation**: Reveals manipulation strategy preferences for significant interactions

#### `pulling_ratio`
**Description**: Preference for significant pulling vs. pushing events
**Calculation**: `pulled / (pushed + pulled)` where both events exceed significant threshold (5 pixels / 0.3 mm)
**Range**: 0.0 to 1.0
**Interpretation**: 0.5 = balanced, > 0.5 = pulling more often than pushing, < 0.5 = pushing more often than pulling

### Behavioral patterns

#### `flailing`

**Description**: Average motion energy of front legs during interaction events

**Calculation**: Motion energy computed as sum of squared velocity differences for leg keypoints during ball interactions

**Units**: dimensionless motion energy

**Interpretation**: Higher values indicate more energetic leg movement during interactions, potentially reflecting flailing often associated with inefficient manipulation

#### `head_pushing_ratio`

**Description**: Proportion of contacts where head is closer to the ball than the legs

**Calculation**: Frame-by-frame analysis during contact events to determine whether head or legs are closer to ball

**Range**: 0.0 to 1.0

**Interpretation**: Closer to 1 indicates closer head to ball distances, potential head pushing. Closer to 0 means more leg contacts.

#### `leg_visibility_ratio`

**Description**: Weighted ratio of front leg visibility during contact events
**Calculation**: For each frame during contact events, score is calculated based on visible front legs (Lfront, Rfront): 0 legs = 0 points, 1 leg = 1 point, 2 legs = 2 points. Final ratio = total_score / (total_frames × 2)
**Range**: 0.0 to 1.0
**Interpretation**: 1.0 = both legs always visible, 0.5 = on average one leg visible, 0.0 = no legs visible. Distinguishes between head pushing (low visibility) and leg pushing (high visibility)

## 3. Locomotor Activity Metrics

### Velocity and Movement

#### `normalized_velocity`
**Description**: Fly velocity normalized by available space
**Calculation**: Average velocity divided by ball distance
**Units**: dimensionless
**Interpretation**: Speed relative to available space.

#### `velocity_during_interactions`
**Description**: Average fly speed during ball contact
**Calculation**: Mean velocity during all interaction events
**Units**: pixels/second
**Interpretation**: Velocity during interactions

#### `velocity_trend`
**Description**: Change in fly velocity over time
**Calculation**: Linear regression slope of velocity vs. time
**Units**: pixels/second²
**Interpretation**: Positive = accelerating, negative = decelerating over session

### Pause Behavior

#### `has_long_pauses`
**Description**: Binary indicator of extended freezing behavior
**Calculation**: 1 if any pauses ≥ 10 seconds duration exist, 0 otherwise
**Range**: 0 or 1
**Interpretation**: Identifies flies exhibiting extended behavioral arrest episodes

#### Short-term Pauses (Stops: 2-5 seconds)
- `nb_stops`: Count of brief pauses
- `total_stop_duration`: Total time in brief pauses
- `median_stop_duration`: Median duration of brief pauses

#### Medium-term Pauses (5-10 seconds)
- `nb_pauses`: Count of medium pauses
- `total_pause_duration`: Total time in medium pauses
- `median_pause_duration`: Median duration of medium pauses

#### Long-term Pauses (>10 seconds)
- `nb_long_pauses`: Count of extended pauses
- `total_long_pause_duration`: Total time in extended pauses
- `median_long_pause_duration`: Median duration of extended pauses

**Interpretation**: Different pause durations may reflect different behavioral states - stops (fearful reactions), medium pauses (decision-making), long pauses (behavioral arrest)

## 4. Spatial Behavior Metrics

### Chamber and Corridor Usage

#### `time_chamber_beginning`
**Description**: Time spent in starting chamber during first 25% of experiment
**Calculation**: Duration within chamber radius during first quarter of video
**Units**: seconds
**Interpretation**: Early chamber attachment behavior; higher values suggest reluctance to explore

#### `persistence_at_end`
**Description**: Fraction of time spent at corridor end (goal area)
**Calculation**: Proportion of frames where fly is at or beyond corridor end threshold distance (170 px / 10.2mm from fly start)
**Range**: 0.0 to 1.0
**Interpretation**: Goal-directed persistence; higher values indicate flies that do not stop the interaction even after the trial is over.

#### `chamber_time`
**Description**: Total time spent in starting chamber throughout entire experiment
**Calculation**: Sum of all frames where fly is within chamber radius (measured from starting position)
**Units**: seconds
**Interpretation**: Total chamber attachment; higher values may indicate less exploration or greater chamber preference

#### `chamber_ratio`
**Description**: Proportion of experiment time spent in chamber
**Calculation**: `chamber_time / total_experiment_duration`
**Range**: 0.0 to 1.0
**Interpretation**: Normalized measure of chamber attachment independent of experiment duration; 1.0 = never left chamber, 0.0 = never in chamber

#### `chamber_exit_time`
**Description**: Time when fly permanently exits the starting chamber
**Calculation**: First timepoint when fly moves beyond chamber radius and does not return
**Units**: seconds
**Interpretation**: Measures latency to begin exploration; earlier times indicate faster engagement with the task

#### `interaction_persistence`
**Description**: Average duration of individual interaction events
**Calculation**: Mean duration across all interaction events
**Units**: seconds
**Interpretation**: Indicates sustained attention during interactions; higher values suggest longer continuous manipulation attempts

#### `interaction_proportion`
**Description**: Fraction of accessible time spent interacting with ball
**Calculation**: `total_interaction_duration / time_until_final_event` (or total duration if no completion)
**Range**: 0.0 to 1.0
**Interpretation**: Measure of task engagement; higher values indicate more time actively manipulating the ball vs. other behaviors

#### `cumulated_breaks_duration`
**Description**: Total time spent in breaks between interaction events
**Calculation**: Sum of all periods between interaction events where fly is not in close proximity to ball
**Units**: seconds
**Interpretation**: Measures disengagement periods; higher values may indicate less sustained interest or more exploratory behavior

#### `fly_distance_moved`
**Description**: Total distance traveled by the fly during the experiment
**Calculation**: Sum of frame-to-frame Euclidean distances of thorax position throughout experiment
**Units**: millimeters
**Interpretation**: Overall locomotor activity; higher values indicate more movement, which may correlate with exploration or general activity levels

#### `fraction_not_facing_ball`
**Description**: Proportion of time fly is not oriented toward the ball
**Calculation**: Fraction of frames where body orientation deviates > 30° from direct line to ball during available time
**Range**: 0.0 to 1.0
**Interpretation**: Higher values suggest less goal-directed orientation; may indicate distraction, exploration, or pulling strategy

## Threshold Summary

| Threshold Type | Pixels | Millimeters | Degrees | Used For |
|---|---|---|---|---|
| **Interaction proximity** | ≤ 45 | ≤ 2.7 mm | - | Detecting when fly is close enough to ball for interaction events |
| **Significant event** | > 5 | > 0.3 mm | - | Significant ball displacement; used for significant events, push/pull classification |
| **Major event** | ≥ 20 | ≥ 1.2 mm | - | First major breakthrough ("aha moment") |
| **Success direction** | ≥ 25 | ≥ 1.5 mm | - | Determining successful manipulation direction (push/pull/both) |
| **Final event** | 170 | 10.2 mm | - | Task completion threshold for standard experiments |
| **Body orientation** | - | - | 30° | Angle deviation from corridor direction for `fraction_not_facing_ball` |
