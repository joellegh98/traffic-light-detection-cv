"""YOLO detection of traffic lights in single images (no tracking).

Pretrained YOLO already knows "traffic light" as COCO class 9 (in the 80-class
list YOLOv8 uses), so detection needs zero training here - just the class
filter, applied at predict time so the model never wastes a pass considering
cars/people/etc.
"""

from ultralytics import YOLO

# COCO class 9 = "traffic light" in YOLOv8's 80-class list.
TRAFFIC_LIGHT_CLASS_ID = 9

DEFAULT_WEIGHTS = "yolov8n.pt"


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
