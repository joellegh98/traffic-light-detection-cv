"""Score predictions (results.csv) against LISA ground-truth labels.

Two separate things get scored:
- Detection: did we find the real lights? (precision/recall via IoU matching)
- Color: over the lights we *did* find, did classify_color name the state
  right? (accuracy + a confusion matrix)

Color is only meaningful for a box that was actually matched to a real light
- a missed light is a recall failure, not a color mistake, so it isn't folded
into the confusion matrix.
"""

import csv
from collections import defaultdict
from pathlib import Path

from src.pipeline import iter_dataset_images

DATASET_ROOT = Path("data/lisa")

# The same 5 clips + stride Phase D3 used to build outputs/results.csv - keep
# this in sync with pipeline.py's run_pipeline() call if that scope changes.
# Both forms are needed: (split_name, clip_name) to find each clip's
# annotations CSV, and the matching frame directory to reconstruct exactly
# which images D3 actually processed (stride=5 skips 4 of every 5 frames -
# those were never attempted, so they must not count as misses).
RESULT_CLIPS = [
    ("dayTrain", "dayClip1"),
    ("dayTrain", "dayClip5"),
    ("dayTrain", "dayClip7"),
    ("nightTrain", "nightClip1"),
    ("nightTrain", "nightClip2"),
]
RESULT_STRIDE = 5


def evaluated_image_names(clips=RESULT_CLIPS, stride=RESULT_STRIDE):
    """The exact set of image filenames Phase D3 actually processed - i.e.
    what iter_dataset_images(..., stride=stride) yields for these clips."""
    clip_dirs = [DATASET_ROOT / split / split / clip / "frames" for split, clip in clips]
    return {p.name for p in iter_dataset_images(clip_dirs, stride=stride)}

# LISA annotation tag -> ground-truth color (see README's Data section).
TAG_TO_COLOR = {
    "go": "green",
    "stop": "red",
    "stopLeft": "red",
    "warning": "yellow",
    "warningLeft": "yellow",
}

IOU_THRESHOLD = 0.5


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def load_ground_truth(clips=RESULT_CLIPS):
    """Return {image_name: [(x1, y1, x2, y2, color), ...]} for the given
    (split_name, clip_name) clips, skipping any tag with no color mapping."""
    gt_by_image = defaultdict(list)
    for split_name, clip_name in clips:
        csv_path = DATASET_ROOT / "Annotations/Annotations" / split_name / clip_name / "frameAnnotationsBOX.csv"
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                color = TAG_TO_COLOR.get(row["Annotation tag"])
                if color is None:
                    continue
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
                gt_by_image[image_name].append((*box, color))
    return gt_by_image


def load_predictions(results_csv):
    """Return {image_name: [(x1, y1, x2, y2, predicted_color), ...]}."""
    pred_by_image = defaultdict(list)
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            pred_by_image[row["image_name"]].append((*box, row["predicted_color"]))
    return pred_by_image


def evaluate(results_csv, clips=RESULT_CLIPS, stride=RESULT_STRIDE, iou_threshold=IOU_THRESHOLD):
    """Match predictions to ground truth by IoU and score detection +
    color. Returns a dict of counts, precision/recall, color accuracy, and a
    {true_color: {predicted_color: count}} confusion matrix.

    Scoring is restricted to the images Phase D3 actually processed
    (evaluated_image_names) - ground truth includes every frame in a clip,
    but a strided run only attempts a fraction of them, and an image with
    zero predicted boxes doesn't appear in results.csv at all. Without this
    restriction, both cases would be wrongly counted as missed detections.
    """
    images = evaluated_image_names(clips, stride)
    gt_by_image = load_ground_truth(clips)
    pred_by_image = load_predictions(results_csv)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    color_correct = 0
    color_total = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for image_name in images:
        preds = pred_by_image.get(image_name, [])
        gts = gt_by_image.get(image_name, [])

        used_gt = set()
        for x1, y1, x2, y2, pred_color in preds:
            best_iou, best_idx = 0.0, None
            for i, (gx1, gy1, gx2, gy2, _true_color) in enumerate(gts):
                if i in used_gt:
                    continue
                score = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                if score > best_iou:
                    best_iou, best_idx = score, i

            if best_iou >= iou_threshold:
                true_positives += 1
                used_gt.add(best_idx)
                true_color = gts[best_idx][4]
                confusion[true_color][pred_color] += 1
                color_total += 1
                if pred_color == true_color:
                    color_correct += 1
            else:
                false_positives += 1

        false_negatives += len(gts) - len(used_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    color_accuracy = color_correct / color_total if color_total else 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "color_total": color_total,
        "color_correct": color_correct,
        "color_accuracy": color_accuracy,
        "confusion": {true: dict(preds) for true, preds in confusion.items()},
    }


def print_confusion_matrix(confusion, colors=("red", "yellow", "green", "unknown")):
    header = "true\\pred".ljust(10) + "".join(c.ljust(10) for c in colors)
    print(header)
    for true_color in colors:
        row = confusion.get(true_color, {})
        print(true_color.ljust(10) + "".join(str(row.get(c, 0)).ljust(10) for c in colors))


if __name__ == "__main__":
    stats = evaluate("outputs/results.csv")

    print(f"Detection (IoU >= {IOU_THRESHOLD}):")
    print(f"  true positives:  {stats['true_positives']}")
    print(f"  false positives: {stats['false_positives']}")
    print(f"  false negatives: {stats['false_negatives']}")
    print(f"  precision: {stats['precision']:.3f}")
    print(f"  recall:    {stats['recall']:.3f}")
    print()
    print(f"Color accuracy (over {stats['color_total']} matched detections):")
    print(f"  {stats['color_correct']}/{stats['color_total']} = {stats['color_accuracy']:.3f}")
    print()
    print("Confusion matrix (rows = true color, columns = predicted):")
    print_confusion_matrix(stats["confusion"])
