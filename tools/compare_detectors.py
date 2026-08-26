"""Compare pretrained vs. custom-trained YOLO detection recall on Phase E2's
held-out validation clips (dayClip7, nightClip1) - never seen during
training, so this is a fair before/after read on whether fine-tuning
actually helped.

Uses a simple IoU>=0.3 match to ground truth as a quick detection-only signal
(no color). Phase F builds the full precision/recall + color confusion-matrix
evaluation with a stricter threshold over the whole results.csv - this script
is E5's narrower "did training help" comparison, not a replacement for that.

Run from the project root:
    python -m tools.compare_detectors
"""

from pathlib import Path

import cv2

from src.detect import detect_traffic_lights, load_model

VAL_IMAGES_DIR = Path("data/yolo_dataset/images/val")
VAL_LABELS_DIR = Path("data/yolo_dataset/labels/val")

PRETRAINED_WEIGHTS = "yolov8n.pt"
CUSTOM_WEIGHTS = "outputs/train_runs/lisa_traffic_light/weights/best.pt"

IOU_THRESHOLD = 0.3


def load_ground_truth_boxes(label_path, img_w, img_h):
    """Decode a YOLO label .txt back into pixel-space (x1, y1, x2, y2) boxes."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = (float(v) for v in line.split())
        cx, w = cx * img_w, w * img_w
        cy, h = cy * img_h, h * img_h
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


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


def evaluate_model(weights, iou_threshold=IOU_THRESHOLD):
    """Run detection over every val image, greedily matching predictions to
    ground truth by IoU. Returns detection-only precision/recall stats."""
    model = load_model(weights)
    total_gt = 0
    total_pred = 0
    matched_gt = 0

    for image_path in sorted(VAL_IMAGES_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        img_h, img_w = image.shape[:2]
        gt_boxes = load_ground_truth_boxes(VAL_LABELS_DIR / (image_path.stem + ".txt"), img_w, img_h)
        pred_boxes = [box[:4] for box in detect_traffic_lights(image_path, model=model)]

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        used_preds = set()
        for gt in gt_boxes:
            best_iou, best_idx = 0.0, None
            for i, pred in enumerate(pred_boxes):
                if i in used_preds:
                    continue
                score = iou(gt, pred)
                if score > best_iou:
                    best_iou, best_idx = score, i
            if best_iou >= iou_threshold:
                matched_gt += 1
                used_preds.add(best_idx)

    return {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "matched": matched_gt,
        "recall": matched_gt / total_gt if total_gt else 0.0,
        "precision": matched_gt / total_pred if total_pred else 0.0,
    }


if __name__ == "__main__":
    print("Evaluating on Phase E2 held-out val clips (dayClip7, nightClip1) - not seen during training.\n")

    pretrained = evaluate_model(PRETRAINED_WEIGHTS)
    print(f"Pretrained ({PRETRAINED_WEIGHTS}):")
    print(f"  {pretrained}\n")

    custom = evaluate_model(CUSTOM_WEIGHTS)
    print(f"Custom ({CUSTOM_WEIGHTS}):")
    print(f"  {custom}\n")

    print(
        f"Recall:    pretrained {pretrained['recall']:.3f} -> custom {custom['recall']:.3f}\n"
        f"Precision: pretrained {pretrained['precision']:.3f} -> custom {custom['precision']:.3f}"
    )
