"""HSV-based color classification for cropped traffic-light boxes.

classify_color(crop) -> "red" | "yellow" | "green" | "unknown"
"""

import cv2
import numpy as np

from src.config import (
    GREEN_HUE_RANGE,
    MIN_ACTIVE_PIXELS,
    RED_HUE_RANGES,
    VALUE_FLOOR,
    YELLOW_HUE_RANGE,
)


def classify_color(crop):
    """Classify the dominant lit color of a cropped traffic-light image.

    Converts the crop to HSV, then for each color counts pixels whose hue (and
    saturation) fall in that color's band *and* whose brightness clears the
    shared Value floor. Whichever color has the most matching pixels wins; if
    even the winner falls short of MIN_ACTIVE_PIXELS, the crop is too dim/small
    to trust and "unknown" is returned instead of guessing.
    """
    if crop is None or crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    counts = {
        "red": _count_matching_pixels(hsv, RED_HUE_RANGES),
        "yellow": _count_matching_pixels(hsv, [YELLOW_HUE_RANGE]),
        "green": _count_matching_pixels(hsv, [GREEN_HUE_RANGE]),
    }

    best_color = max(counts, key=counts.get)
    if counts[best_color] < MIN_ACTIVE_PIXELS:
        return "unknown"
    return best_color


def _count_matching_pixels(hsv, hue_ranges):
    """Count pixels in `hsv` covered by any of `hue_ranges` (each a
    (hue_min, hue_max, sat_min) band), bright enough per VALUE_FLOOR."""
    mask = None
    for hue_min, hue_max, sat_min in hue_ranges:
        lower = np.array([hue_min, sat_min, VALUE_FLOOR], dtype=np.uint8)
        upper = np.array([hue_max, 255, 255], dtype=np.uint8)
        band_mask = cv2.inRange(hsv, lower, upper)
        mask = band_mask if mask is None else cv2.bitwise_or(mask, band_mask)
    return int(cv2.countNonZero(mask))
