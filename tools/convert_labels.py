"""Convert LISA annotation CSVs to YOLO-format label .txt files.

Single class throughout - `traffic_light` (class index 0). Phase B's HSV
classifier handles color separately, so the detector only ever needs to learn
"is this a traffic light", keeping training simpler and the class count at 1.

Label files are written to data/lisa_labels/<clip_name>/, mirroring the frame
filenames from data/lisa/.../<clip_name>/frames/ but with a .txt extension -
kept separate from the raw dataset so nothing generated is mixed into
data/lisa/. Phase E2 assembles the actual images/labels training tree by
picking which clips go to train vs. val and copying from here.

Run from the project root, e.g.:
    python -m tools.convert_labels
"""

import csv
from collections import defaultdict
from pathlib import Path

import cv2

CLASS_ID = 0  # single class: traffic_light

LABELS_ROOT = Path("data/lisa_labels")


def convert_clip(frames_dir, annotations_csv, labels_dir=None):
    """Convert one clip's annotation CSV into YOLO .txt label files.

    Every image under `frames_dir` gets a label file, even one with zero
    boxes (an empty .txt is the standard YOLO way to say "no objects here" -
    without it, images with no traffic light would silently vanish from
    training instead of teaching the model what "nothing here" looks like).

    Returns (images_labeled, boxes_written).
    """
    frames_dir = Path(frames_dir)
    labels_dir = Path(labels_dir) if labels_dir else LABELS_ROOT / frames_dir.parent.name
    labels_dir.mkdir(parents=True, exist_ok=True)

    boxes_by_image = defaultdict(list)
    with open(annotations_csv, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            image_name = Path(row["Filename"]).name
            box = tuple(
                int(row[key])
                for key in (
                    "Upper left corner X",
                    "Upper left corner Y",
                    "Lower right corner X",
                    "Lower right corner Y",
                )
            )
            boxes_by_image[image_name].append(box)

    images_labeled = 0
    boxes_written = 0
    for image_path in sorted(frames_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        lines = []
        for x1, y1, x2, y2 in boxes_by_image.get(image_path.name, []):
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path = labels_dir / (image_path.stem + ".txt")
        label_path.write_text("\n".join(lines))
        images_labeled += 1
        boxes_written += len(lines)

    return images_labeled, boxes_written


if __name__ == "__main__":
    # Smoke-test conversion on one known clip; Phase E2 decides the real
    # train/val clip set and runs this for real over that set.
    images_labeled, boxes_written = convert_clip(
        frames_dir="data/lisa/dayTrain/dayTrain/dayClip5/frames",
        annotations_csv="data/lisa/Annotations/Annotations/dayTrain/dayClip5/frameAnnotationsBOX.csv",
    )
    print(f"dayClip5: {images_labeled} images labeled, {boxes_written} boxes written")
