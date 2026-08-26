"""Run detect + color over the full image set and write results.csv."""

from pathlib import Path

import cv2

from src.color import classify_color
from src.detect import detect_traffic_lights

DATASET_ROOT = Path("data/lisa")

# Clip/sequence frame folders that have matching ground-truth annotations.
# The two sample-* folders are excluded: they carry no Annotations/ labels,
# so Phase F could never score predictions made on them.
LABELED_CLIP_DIRS = (
    [DATASET_ROOT / "dayTrain/dayTrain" / f"dayClip{i}" / "frames" for i in range(1, 14)]
    + [DATASET_ROOT / "nightTrain/nightTrain" / f"nightClip{i}" / "frames" for i in range(1, 6)]
    + [
        DATASET_ROOT / "daySequence1/daySequence1/frames",
        DATASET_ROOT / "daySequence2/daySequence2/frames",
        DATASET_ROOT / "nightSequence1/nightSequence1/frames",
        DATASET_ROOT / "nightSequence2/nightSequence2/frames",
    ]
)


def iter_dataset_images(clip_dirs=LABELED_CLIP_DIRS):
    """Yield every .jpg path under the given clip frame directories, in
    stable order (LISA's zero-padded frame numbers sort correctly as text)."""
    for clip_dir in clip_dirs:
        for image_path in sorted(clip_dir.glob("*.jpg")):
            yield image_path


def process_image(image_path, model):
    """Detect + classify every traffic light in one image.

    Returns a list of dicts, one per detected box - the exact row shape D2
    writes to results.csv:
    {image_name, x1, y1, x2, y2, predicted_color, confidence}
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    rows = []
    for x1, y1, x2, y2, confidence in detect_traffic_lights(image_path, model=model):
        color = classify_color(image[y1:y2, x1:x2])
        rows.append(
            {
                "image_name": image_path.name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "predicted_color": color,
                "confidence": confidence,
            }
        )
    return rows
