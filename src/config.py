"""HSV color ranges and other tunable thresholds, kept in one place.

OpenCV's HSV hue axis runs 0-180 (not the usual 0-360), and red sits at *both*
ends of that range, so red needs two hue bands stitched together; yellow and
green each need only one.

Every range also shares one brightness (Value) floor: a lit lamp is bright
regardless of color, while the dark, unlit housing around it is not. Applying
the same floor everywhere is what separates "lamp" from "background" before
hue is even considered.

Values below were tuned (B4) against ~45 real ground-truth crops sampled
across 5 LISA clips, not just guessed:

- VALUE_FLOOR came down from 150 to 100. LISA's lights are often small
  (many boxes are only ~15-30px per side), so the "lit lamp" region is a
  small fraction of an already-small crop; 150 discarded too much of it.
- MIN_ACTIVE_PIXELS came down from 30 to 10 for the same reason: 30 pixels
  is a lot to demand from a 20x15 box.
- GREEN_HUE_RANGE's upper bound moved from 90 to 100, and its saturation
  floor rose to 130: real LED-green in this dataset renders more cyan than
  a first guess assumes, and needed a higher saturation floor to avoid
  picking up desaturated background pixels at the wider hue.

One confusion this tuning can't fix: overexposed yellow ("warning") lights
often have their brightest pixels measured at hue 0-11 - indistinguishable
from red at the pixel level, a camera/exposure effect, not a bug here.
Loosening RED_HUE_RANGES to compensate just breaks real red crops the other
way. Documented as a known limitation (see README / plan Phase I) rather
than papered over with thresholds.
"""

# Brightness floor shared by every color (0-255, OpenCV's Value scale). Pixels
# dimmer than this are treated as unlit housing/background, never a lit lamp.
VALUE_FLOOR = 100

# Minimum count of matching pixels required before a color counts as
# "detected" in a crop. Below this, classify_color() returns "unknown" rather
# than guessing off a couple of stray, possibly noisy pixels.
MIN_ACTIVE_PIXELS = 10

# Hue/Saturation bands, as (hue_min, hue_max, sat_min) on OpenCV's H:0-180,
# S:0-255 scales. The saturation floor excludes washed-out/near-white or
# near-gray pixels that happen to fall in the right hue range.
RED_HUE_RANGES = [
    (0, 10, 100),  # low end of the hue wheel
    (170, 180, 100),  # high end of the hue wheel (red wraps around)
]
YELLOW_HUE_RANGE = (15, 35, 100)
GREEN_HUE_RANGE = (40, 100, 130)
