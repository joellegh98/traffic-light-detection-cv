"""End-to-end entrypoint: images -> results.csv -> accuracy evaluation -> plots.

Run from the project root:
    python -m src.main                    # pretrained (default; see Phase E6)
    python -m src.main --weights custom   # Phase E's fine-tuned best.pt
    python -m src.main --weights path/to/other.pt
"""

import argparse
import sys
import time
from pathlib import Path

from src.detect import load_model
from src.evaluate import DATASET_ROOT, RESULT_CLIPS, RESULT_STRIDE, evaluate
from src.pipeline import run_pipeline
from src.report import plot_color_counts, plot_confusion_matrix, save_example_grid

RESULTS_CSV = "outputs/results.csv"
COLOR_COUNTS_PNG = "outputs/color_counts.png"
CONFUSION_MATRIX_PNG = "outputs/confusion_matrix.png"
EXAMPLE_GRID_PNG = "outputs/example_grid.png"

# Default scope matches Phases D3/E5/F/G's established subset (RESULT_CLIPS,
# defined once in evaluate.py) - NOT run_pipeline()'s own default, which
# covers the full ~43k-image dataset and takes about an hour on this CPU.
# A silent hour-long default here would break "one fast command" for
# everything this project has built so far.
DEFAULT_CLIP_DIRS = [DATASET_ROOT / split / split / clip / "frames" for split, clip in RESULT_CLIPS]

# --weights accepts these short aliases, or any literal weights path.
CUSTOM_WEIGHTS_PATH = "outputs/train_runs/lisa_traffic_light/weights/best.pt"
WEIGHTS_ALIASES = {
    "pretrained": None,  # None -> load_model()'s own default (yolov8n.pt)
    "custom": CUSTOM_WEIGHTS_PATH,
}


def resolve_weights(weights_arg):
    """Turn a --weights value into what load_model() expects: one of the
    named aliases above, or any other string is treated as a literal path
    (so a future retrained model doesn't need code changes to use)."""
    if weights_arg in WEIGHTS_ALIASES:
        return WEIGHTS_ALIASES[weights_arg]
    return weights_arg


def check_dataset_available(clip_dirs=DEFAULT_CLIP_DIRS, clips=RESULT_CLIPS):
    """Fail with one clear, actionable message rather than a raw traceback
    partway through processing if the LISA dataset isn't where expected."""
    missing = [d for d in clip_dirs if not d.exists()]
    missing += [
        p
        for split, clip in clips
        if not (p := DATASET_ROOT / "Annotations/Annotations" / split / clip / "frameAnnotationsBOX.csv").exists()
    ]
    if missing:
        listed = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(
            f"LISA dataset not found - missing:\n{listed}\n\n"
            "Download the LISA Traffic Light Dataset into data/lisa/ - see the "
            "README's Data section for the expected folder layout."
        )


def check_weights_available(resolved):
    """Same idea for model weights: catch a missing custom-trained model
    before Ultralytics raises something less clear. A bare model name with
    no path separator (e.g. 'yolov8s.pt') is left alone - Ultralytics
    downloads those itself, so a missing local file isn't an error there.
    """
    if resolved and ("/" in resolved or "\\" in resolved) and not Path(resolved).exists():
        raise FileNotFoundError(
            f"Weights file not found: {resolved}\n"
            "If this is the Phase E custom model, train it first: python -m tools.train"
        )


def main(weights="pretrained"):
    start = time.time()
    resolved = resolve_weights(weights)

    check_dataset_available()
    check_weights_available(resolved)

    print(f"[1/4] Loading model (weights={weights!r})...")
    model = load_model(resolved) if resolved else load_model()

    print(f"[2/4] Running detection + color over {len(DEFAULT_CLIP_DIRS)} clips (stride={RESULT_STRIDE})...")
    images_processed, boxes_found = run_pipeline(
        clip_dirs=DEFAULT_CLIP_DIRS, output_path=RESULTS_CSV, stride=RESULT_STRIDE, model=model
    )

    print("[3/4] Evaluating against ground truth...")
    stats = evaluate(RESULTS_CSV, collect_examples=True)
    print(f"       precision={stats['precision']:.3f}  recall={stats['recall']:.3f}")
    print(f"       color accuracy={stats['color_correct']}/{stats['color_total']} = {stats['color_accuracy']:.3f}")

    print("[4/4] Generating plots...")
    plot_color_counts(RESULTS_CSV, COLOR_COUNTS_PNG)
    plot_confusion_matrix(stats["confusion"], CONFUSION_MATRIX_PNG)
    save_example_grid(stats["examples"], EXAMPLE_GRID_PNG)

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s. Outputs:")
    print(f"  {RESULTS_CSV}          {images_processed} images, {boxes_found} boxes")
    print(f"  {COLOR_COUNTS_PNG}")
    print(f"  {CONFUSION_MATRIX_PNG}")
    print(f"  {EXAMPLE_GRID_PNG}")
    print(
        f"Detection: precision={stats['precision']:.3f} recall={stats['recall']:.3f} | "
        f"color accuracy={stats['color_accuracy']:.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="pretrained",
        help="'pretrained' (default, see Phase E6), 'custom' (Phase E's best.pt), or a literal weights path.",
    )
    args = parser.parse_args()

    try:
        main(weights=args.weights)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
