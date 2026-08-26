"""HSV color ranges and other tunable thresholds, kept in one place.

OpenCV's HSV hue axis runs 0-180 (not the usual 0-360), and red sits at *both*
ends of that range, so red needs two hue bands stitched together; yellow and
green each need only one.

Every range also shares one brightness (Value) floor: a lit lamp is bright
regardless of color, while the dark, unlit housing around it is not. Applying
the same floor everywhere is what separates "lamp" from "background" before
hue is even considered.

These are reasonable starting values, not final ones — B4 tunes them against
real cropped traffic-light images.
"""

# Brightness floor shared by every color (0-255, OpenCV's Value scale). Pixels
# dimmer than this are treated as unlit housing/background, never a lit lamp.
VALUE_FLOOR = 150

# Minimum count of matching pixels required before a color counts as
# "detected" in a crop. Below this, classify_color() returns "unknown" rather
# than guessing off a couple of stray, possibly noisy pixels.
MIN_ACTIVE_PIXELS = 30

# Hue/Saturation bands, as (hue_min, hue_max, sat_min) on OpenCV's H:0-180,
# S:0-255 scales. The saturation floor excludes washed-out/near-white or
# near-gray pixels that happen to fall in the right hue range.
RED_HUE_RANGES = [
    (0, 10, 100),  # low end of the hue wheel
    (170, 180, 100),  # high end of the hue wheel (red wraps around)
]
YELLOW_HUE_RANGE = (15, 35, 100)
GREEN_HUE_RANGE = (40, 90, 100)
