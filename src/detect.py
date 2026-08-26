"""YOLO detection of traffic lights in single images (no tracking).

Pretrained YOLO already knows "traffic light" as COCO class 9 (in the 80-class
list YOLOv8 uses), so detection needs zero training here - just the class
filter, applied at predict time so the model never wastes a pass considering
cars/people/etc.
"""

import cv2
from ultralytics import YOLO

from src.color import classify_color

# COCO class 9 = "traffic light" in YOLOv8's 80-class list.
TRAFFIC_LIGHT_CLASS_ID = 9

DEFAULT_WEIGHTS = "yolov8n.pt"

# BGR draw colors per classified color, for the annotate helper below.
_BOX_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "unknown": (200, 200, 200),
}


def load_model(weights=DEFAULT_WEIGHTS):
    """Load a YOLO model. Ultralytics downloads `weights` automatically on
    first use if it isn't already cached locally."""
    return YOLO(weights)


def detect_traffic_lights(image_path, model=None, conf=0.25):
    """Run YOLO on a single image, returning only traffic-light boxes.

    `model` should be passed in (loaded once via load_model()) when calling
    this repeatedly over many images - e.g. Phase D's pipeline loop - so each
    call doesn't reload the weights from disk.

    Returns a list of (x1, y1, x2, y2, confidence) tuples in integer pixel
    coordinates, one per detected traffic light.
    """
    if model is None:
        model = load_model()

    results = model.predict(
        source=str(image_path),
        classes=[TRAFFIC_LIGHT_CLASS_ID],
        conf=conf,
        verbose=False,
    )

    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        boxes.append((int(x1), int(y1), int(x2), int(y2), confidence))
    return boxes


def annotate_image(image_path, output_path, model=None, conf=0.25):
    """Detect + classify every traffic light in an image, draw a box and
    color label for each, and save the result - for eyeballing a handful of
    sample images rather than trusting numbers blindly.

    Returns the list of (x1, y1, x2, y2, confidence, color) actually drawn.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    boxes = detect_traffic_lights(image_path, model=model, conf=conf)

    annotated = image.copy()
    results = []
    for x1, y1, x2, y2, confidence in boxes:
        color = classify_color(image[y1:y2, x1:x2])
        results.append((x1, y1, x2, y2, confidence, color))

        draw_color = _BOX_COLORS[color]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_color, 2)
        label = f"{color} {confidence:.2f}"
        label_y = y1 - 6 if y1 - 6 > 10 else y1 + 14
        cv2.putText(
            annotated, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), annotated)
    return results
