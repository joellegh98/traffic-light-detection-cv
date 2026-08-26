"""Run detect + color over the full image set and write results.csv."""

import csv
import time
from pathlib import Path

import cv2

from src.color import classify_color
from src.detect import detect_traffic_lights, load_model

DATASET_ROOT = Path("data/lisa")

CSV_FIELDNAMES = ["image_name", "x1", "y1", "x2", "y2", "predicted_color", "confidence"]

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


def iter_dataset_images(clip_dirs=LABELED_CLIP_DIRS, stride=1):
    """Yield .jpg paths under the given clip frame directories, in stable
    order (LISA's zero-padded frame numbers sort correctly as text).

    `stride` > 1 samples every Nth frame *within each clip* rather than every
    frame - e.g. for a capped run that still spans each clip's full duration
    (and therefore its full range of light colors/states) instead of only
    covering its first few seconds.
    """
    for clip_dir in clip_dirs:
        for image_path in sorted(clip_dir.glob("*.jpg"))[::stride]:
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


def write_results_csv(rows, output_path):
    """Write detection+color rows (as produced by process_image) to a flat
    CSV file, one row per detected box."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    clip_dirs=LABELED_CLIP_DIRS,
    output_path="outputs/results.csv",
    model=None,
    progress_every=500,
    stride=1,
):
    """Run detect+color over images under `clip_dirs` (every `stride`-th
    frame per clip), writing results to `output_path`. Prints periodic
    progress and a final summary.

    Returns (images_processed, boxes_found).
    """
    if model is None:
        model = load_model()

    all_rows = []
    images_processed = 0
    start = time.time()

    for image_path in iter_dataset_images(clip_dirs, stride=stride):
        all_rows.extend(process_image(image_path, model))
        images_processed += 1
        if images_processed % progress_every == 0:
            elapsed = time.time() - start
            rate = images_processed / elapsed
            print(
                f"  {images_processed} images, {len(all_rows)} boxes so far "
                f"({rate:.1f} img/s, {elapsed:.0f}s elapsed)"
            )

    write_results_csv(all_rows, output_path)

    elapsed = time.time() - start
    print(
        f"Done: {images_processed} images processed, {len(all_rows)} boxes found "
        f"in {elapsed:.0f}s -> {output_path}"
    )
    return images_processed, len(all_rows)
