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

## Threshold Summary

| Threshold Type | Pixels | Millimeters | Degrees | Used For |
|---|---|---|---|---|
| **Interaction proximity** | ≤ 45 | ≤ 2.7 mm | - | Detecting when fly is close enough to ball for interaction events |
| **Significant event** | > 5 | > 0.3 mm | - | Significant ball displacement; used for significant events, push/pull classification |
| **Major event** | ≥ 20 | ≥ 1.2 mm | - | First major breakthrough ("aha moment") |
| **Success direction** | ≥ 25 | ≥ 1.5 mm | - | Determining successful manipulation direction (push/pull/both) |
| **Final event** | 170 | 10.2 mm | - | Task completion threshold for standard experiments |
| **Body orientation** | - | - | 30° | Angle deviation from corridor direction for `fraction_not_facing_ball` |