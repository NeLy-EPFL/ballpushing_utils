DATA_DIR = "/mnt/upramdya_data/TL/ball_pushing_low_heat"
"""Folder containing a subfolder for each line, each of which contain videos."""

FRAMES_PER_SECOND = 80
"""Frame rate of the videos"""

PIX_PER_MM = 188 / 1.5
"""1.5 mm diameter ball is 188 px"""

ARENA_WIDTH = 2048
"""Arena video width is 2048 px"""

CORRIDOR_CROP_Y_OFFSETS = (0, 330, 690, 1040, 1390, 1680)
"""Vertical position to start cropping each corridor in the arena from."""

CORRIDOR_CROP_HEIGHT = 368
"""How high to crop each corridor."""

LINES_TO_DISCARD = {
    "ExR1",  # not a hit
    "GH146",  # VNC expression
    "LH272",  #
}
"""Lines that we won't analyse for some reason"""

VIDEOS_TO_DISCARD = {
    # empty-split
    "20250813-181107",  # bad visibility arena
    "20250821-182936",  # bad visibility arena
    "20250822-172413",  # bad visibility arena
    "20250826-165602",  # bad visibility arena
    "20250827-150248",  # bad visibility arena
    # IR8a
    "20250825-143921",  # bad visibility arena
    "20250825-170354",  # bad visibility arena
    "20250828-174041",  # bad visibility arena
    "20250903-190324",  # bad visibility arena
    "20250908-183002",  # bad visibility arena
    # IR25a
    "20250828-150317",  # bad visibility arena
    "20250829-162011",  # bad visibility arena
    "20250830-183312",  # bad visibility arena
    "20250903-164458",  # bad visibility arena
    "20250908-161236",  # bad visibility arena
    # LC10-2
    "20250910-171203",  # bad visibility arena
    "20250910-171203",  # bad visibility arena
    # MB247
    "20250815-162205",  # bad visibility arena
    "20250818-155123",  # bad visibility arena
}
"""If a whole arena video is bad for some reason"""

FLIES_TO_DISCARD = {
    # empty-Gal4
    ("20250918-191702", 1),  # ball moves to block chamber
    # empty-split
    ("20250813-181107", 3),  # turns around
    ("20250821-182936", 1),  # tries to turn around
    ("20250821-182936", 3),  # tries to turn around
    ("20250826-165602", 3),  # tries to turn around
    # IR25a
    ("20250827-161828", 2),  # no fly
    ("20250828-185324", 3),  # fly turns around
    # GR63a
    ("20250924-174342", 2),  # fly turns around
    # OR67d
    ("20250924-152733", 2),  # fly turns around
    # LC16-1
    ("20250917-141757", 4),  # no fly
    # MB247
    ("20250821-171016", 6),  # no fly
    # PR
    ("20260126-140010", 3),  # fly moves very slowly
    ("20260127-125609", 6),  # fly enters corridor backwards
    ("20260123-120123", 5),  # fly enters corridor backwards ~4:00
    ("20260123-100602", 2),  # fly enters corridor backwards ~27:20
}
"""If a particular corridor or fly is bad for some reason"""

CORRIDOR_END = 14
"""End of the corridor (in mm)"""

MAJOR_EVENT_DISTANCE = 1.2
"""Distance the ball needs to travel in one contact event to be a major event."""

SIGNIFICANT_EVENT_DISTANCE = 0.3
"""Distance the ball needs to travel in one contact event to be a significant event."""
