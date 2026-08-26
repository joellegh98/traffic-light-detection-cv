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
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

CLASS_ID = 0  # single class: traffic_light

DATASET_ROOT = Path("data/lisa")
LABELS_ROOT = Path("data/lisa_labels")

# Where Phase E2's assembled training tree lives - Ultralytics auto-finds
# labels for an image by replacing "/images/" with "/labels/" in its path,
# so both need this exact "images/<split>" and "labels/<split>" layout
# (unlike the raw dataset's own "frames" folders, which don't match that
# convention).
YOLO_DATASET_ROOT = Path("data/yolo_dataset")


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


# --- Phase E2: train/val split, by whole clip -------------------------------
#
# LISA frames come from video, so neighboring frames are near-duplicates. A
# random frame-level split would leak almost-identical images into both train
# and val and inflate the reported accuracy. Splitting by whole clip instead
# means every frame from one clip stays entirely on one side.
#
# Each entry is (split_name, clip_name, stride). `stride` samples every Nth
# frame of that clip so the small, CPU-sized training set still spans each
# clip's full duration (and color-state range) instead of just its start -
# same reasoning as Phase D3's capped run.
TRAIN_CLIPS = [
    ("dayTrain", "dayClip1", 14),
    ("dayTrain", "dayClip5", 14),
    ("nightTrain", "nightClip2", 14),
]
VAL_CLIPS = [
    ("dayTrain", "dayClip7", 22),
    ("nightTrain", "nightClip1", 22),
]


def _clip_paths(split_name, clip_name):
    frames_dir = DATASET_ROOT / split_name / split_name / clip_name / "frames"
    annotations_csv = (
        DATASET_ROOT / "Annotations/Annotations" / split_name / clip_name / "frameAnnotationsBOX.csv"
    )
    return frames_dir, annotations_csv


def assemble_split(clips, split):
    """Convert (if needed) and copy a stride-sampled subset of `clips` into
    data/yolo_dataset/{images,labels}/<split>/.

    Returns (images_copied, boxes_copied).
    """
    images_dir = YOLO_DATASET_ROOT / "images" / split
    labels_dir = YOLO_DATASET_ROOT / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_copied = 0
    boxes_copied = 0
    for split_name, clip_name, stride in clips:
        frames_dir, annotations_csv = _clip_paths(split_name, clip_name)
        convert_clip(frames_dir, annotations_csv)
        clip_labels_dir = LABELS_ROOT / clip_name

        for image_path in sorted(frames_dir.glob("*.jpg"))[::stride]:
            label_path = clip_labels_dir / (image_path.stem + ".txt")
            shutil.copy(image_path, images_dir / image_path.name)
            shutil.copy(label_path, labels_dir / label_path.name)
            images_copied += 1
            with open(label_path) as f:
                boxes_copied += sum(1 for _ in f)

    return images_copied, boxes_copied


if __name__ == "__main__":
    train_images, train_boxes = assemble_split(TRAIN_CLIPS, "train")
    print(f"train: {train_images} images, {train_boxes} boxes ({[c[1] for c in TRAIN_CLIPS]})")

    val_images, val_boxes = assemble_split(VAL_CLIPS, "val")
    print(f"val:   {val_images} images, {val_boxes} boxes ({[c[1] for c in VAL_CLIPS]})")
