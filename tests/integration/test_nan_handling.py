#!/usr/bin/env python3

import sys
import math
import numpy as np

# Test the NaN handling functions


def resize_coordinates(x, y, original_width, original_height, new_width, new_height):
    """Resize the coordinates according to the new frame size."""
    # Check for NaN values and return None for both coordinates if either is NaN
    if math.isnan(x) or math.isnan(y):
        return None, None

    x_scale = new_width / original_width
    y_scale = new_height / original_height
    return int(x * x_scale), int(y * y_scale)


def apply_arena_mask_to_labels(x, y, mask_padding, crop_top, crop_bottom, new_height):
    """Adjust the coordinates according to the cropping and padding applied to the frame."""
    # Check if coordinates are None (from NaN handling)
    if x is None or y is None:
        return None, None

    # Crop from top and bottom
    if crop_top <= y < (new_height - crop_bottom):
        y -= crop_top
    else:
        return None, None

    # Add padding to the left and right
    x += mask_padding

    return x, y


def resize_and_transform_coordinate(
    x, y, original_width, original_height, new_width, new_height, mask_padding, crop_top, crop_bottom
):
    """Resize and transform the coordinate to match the preprocessed frame."""
    # Resize the coordinate
    x, y = resize_coordinates(x, y, original_width, original_height, new_width, new_height)

    # Apply cropping offset and padding
    x, y = apply_arena_mask_to_labels(x, y, mask_padding, crop_top, crop_bottom, new_height)

    return x, y


# Test cases
print("Testing NaN handling:")

# Test 1: Normal coordinates
result = resize_and_transform_coordinate(100, 200, 640, 480, 320, 240, 10, 20, 30)
print(f"Normal coordinates (100, 200): {result}")

# Test 2: NaN in x
result = resize_and_transform_coordinate(float("nan"), 200, 640, 480, 320, 240, 10, 20, 30)
print(f"NaN in x: {result}")

# Test 3: NaN in y
result = resize_and_transform_coordinate(100, float("nan"), 640, 480, 320, 240, 10, 20, 30)
print(f"NaN in y: {result}")

# Test 4: Both NaN
result = resize_and_transform_coordinate(float("nan"), float("nan"), 640, 480, 320, 240, 10, 20, 30)
print(f"Both NaN: {result}")

# Test 5: Create arrays with NaN handling like in the actual code
ball_coords = [(100, 200), (float("nan"), 300), (400, float("nan")), (float("nan"), float("nan"))]

ball_coords_processed = [resize_and_transform_coordinate(x, y, 640, 480, 320, 240, 10, 20, 30) for x, y in ball_coords]

print(f"Processed coordinates: {ball_coords_processed}")

# Convert to arrays like in the actual code
ball_coords_x = []
ball_coords_y = []
for x, y in ball_coords_processed:
    if x is not None and y is not None:
        ball_coords_x.append(x)
        ball_coords_y.append(y)
    else:
        ball_coords_x.append(np.nan)
        ball_coords_y.append(np.nan)

print(f"Final x coordinates: {ball_coords_x}")
print(f"Final y coordinates: {ball_coords_y}")

print("Test completed successfully!")
