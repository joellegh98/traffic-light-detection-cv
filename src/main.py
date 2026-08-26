"""End-to-end entrypoint: images -> results.csv -> accuracy evaluation -> plots.

Run from the project root:
    python -m src.main
"""

from pathlib import Path

from src.evaluate import DATASET_ROOT, RESULT_CLIPS, RESULT_STRIDE, evaluate
from src.pipeline import run_pipeline
from src.report import plot_color_counts, plot_confusion_matrix, save_example_grid

RESULTS_CSV = "outputs/results.csv"

# Default scope matches Phases D3/E5/F/G's established subset (RESULT_CLIPS,
# defined once in evaluate.py) - NOT run_pipeline()'s own default, which
# covers the full ~43k-image dataset and takes about an hour on this CPU.
# A silent hour-long default here would break "one fast command" for
# everything this project has built so far.
DEFAULT_CLIP_DIRS = [DATASET_ROOT / split / split / clip / "frames" for split, clip in RESULT_CLIPS]


def main():
    images_processed, boxes_found = run_pipeline(
        clip_dirs=DEFAULT_CLIP_DIRS, output_path=RESULTS_CSV, stride=RESULT_STRIDE
    )

    stats = evaluate(RESULTS_CSV, collect_examples=True)
    print(f"Detection: precision={stats['precision']:.3f} recall={stats['recall']:.3f}")
    print(f"Color accuracy: {stats['color_correct']}/{stats['color_total']} = {stats['color_accuracy']:.3f}")

    plot_color_counts(RESULTS_CSV, "outputs/color_counts.png")
    plot_confusion_matrix(stats["confusion"], "outputs/confusion_matrix.png")
    save_example_grid(stats["examples"], "outputs/example_grid.png")


if __name__ == "__main__":
    main()
