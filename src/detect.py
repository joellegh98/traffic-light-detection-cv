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
