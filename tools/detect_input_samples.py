"""Run detection + color classification over every image in input/ and save
annotated copies (boxes + color labels) to input/detected/.

A quick, self-contained demo separate from the main dataset pipeline (Phase
D/H) - no LISA download required, just the 10 sample images committed in
input/. Uses annotate_image() from src/detect.py (Phase C3), so it's the same
already-verified detection + drawing code the rest of the project relies on.

Run from the project root:
    python -m tools.detect_input_samples
"""

from pathlib import Path

from src.detect import annotate_image, load_model

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("input/detected")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()  # pretrained yolov8n.pt - the default per Phase E6

    image_paths = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"No images found in {INPUT_DIR}/")
        return

    total_boxes = 0
    for image_path in image_paths:
        output_path = OUTPUT_DIR / image_path.name
        detections = annotate_image(image_path, output_path, model=model)
        total_boxes += len(detections)
        colors = [color for *_box, _conf, color in detections]
        print(f"{image_path.name}: {len(detections)} light(s) -> {colors} -> {output_path}")

    print(f"\nDone: {len(image_paths)} images, {total_boxes} lights detected -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
