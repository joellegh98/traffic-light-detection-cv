"""Quick manual validation of classify_color() against three real LISA crops,
one per color, pulled from data/lisa/dayTrain/dayTrain/dayClip5 (chosen because
that clip has all three states: go/stop/warning).

Run from the project root:
    python -m tools.validate_color
"""

import csv
from pathlib import Path

import cv2

from src.color import classify_color

CLIP_DIR = Path("data/lisa/dayTrain/dayTrain/dayClip5")
ANNOTATIONS_CSV = Path(
    "data/lisa/Annotations/Annotations/dayTrain/dayClip5/frameAnnotationsBOX.csv"
)
OUTPUT_DIR = Path("outputs")

# LISA annotation tag -> expected color (see README's Data section).
TAG_TO_COLOR = {
    "go": "green",
    "stop": "red",
    "stopLeft": "red",
    "warning": "yellow",
    "warningLeft": "yellow",
}


def first_row_per_color():
    """Return the first annotation row seen for each of red/yellow/green."""
    found = {}
    with open(ANNOTATIONS_CSV, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            color = TAG_TO_COLOR.get(row["Annotation tag"])
            if color and color not in found:
                found[color] = row
            if len(found) == 3:
                break
    return found


def load_crop(row):
    """Load the frame an annotation row refers to and crop its box.

    The CSV's Filename column carries a stale folder prefix (e.g.
    "dayTraining/dayClip5--00000.jpg"); only the basename matches the actual
    extracted layout, which nests frames under dayTrain/dayTrain/<clip>/frames/.
    """
    frame_name = Path(row["Filename"]).name
    image_path = CLIP_DIR / "frames" / frame_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    x1, y1, x2, y2 = (
        int(row[key])
        for key in (
            "Upper left corner X",
            "Upper left corner Y",
            "Lower right corner X",
            "Lower right corner Y",
        )
    )
    return image[y1:y2, x1:x2]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = first_row_per_color()

    print(f"{'expected':<10} {'predicted':<10} {'result':<6} source frame")
    all_correct = True
    for expected_color, row in rows.items():
        crop = load_crop(row)
        predicted = classify_color(crop)
        ok = predicted == expected_color
        all_correct &= ok

        frame_name = Path(row["Filename"]).name
        cv2.imwrite(str(OUTPUT_DIR / f"b3_{expected_color}.png"), crop)
        print(f"{expected_color:<10} {predicted:<10} {'PASS' if ok else 'FAIL':<6} {frame_name}")

    print()
    if all_correct:
        print("All three correct.")
    else:
        print("Some crops misclassified - see Phase B4 (threshold tuning).")


if __name__ == "__main__":
    main()
