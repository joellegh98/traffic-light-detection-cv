"""Turn evaluation results into stats and plots.

G1: bar chart of predicted color counts.
G2: confusion-matrix heatmap.
G3: a grid of annotated example detections, some correct, some wrong -
Phase F found exactly one real color failure mode (yellow read as red under
overexposure), so the "wrong" examples are drawn from that.
"""

import csv
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate import DATASET_ROOT, RESULT_CLIPS, evaluate

COLOR_ORDER = ("red", "yellow", "green", "unknown")
COLOR_HEX = {"red": "#d62728", "yellow": "#bcbd22", "green": "#2ca02c", "unknown": "#7f7f7f"}

# Reuse Phase F's clip list to resolve an image_name back to its file path -
# results.csv only stores the bare filename, not which clip/split it's from.
CLIP_TO_SPLIT = {clip: split for split, clip in RESULT_CLIPS}


def resolve_image_path(image_name):
    """Map a results.csv image_name (e.g. 'dayClip5--00126.jpg') back to its
    file under data/lisa/."""
    clip_name = image_name.split("--")[0]
    split_name = CLIP_TO_SPLIT[clip_name]
    return DATASET_ROOT / split_name / split_name / clip_name / "frames" / image_name


def plot_color_counts(results_csv, output_path):
    """Bar chart of predicted color counts across results.csv."""
    with open(results_csv, newline="") as f:
        counts = Counter(row["predicted_color"] for row in csv.DictReader(f))

    colors = [c for c in COLOR_ORDER if c in counts] or list(counts)
    values = [counts[c] for c in colors]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(colors, values, color=[COLOR_HEX.get(c, "#888888") for c in colors])
    ax.set_xlabel("Predicted color")
    ax.set_ylabel("Count")
    ax.set_title("Predicted traffic-light color counts")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(confusion, output_path, colors=COLOR_ORDER):
    """Confusion-matrix heatmap: rows = true color, columns = predicted."""
    matrix = [[confusion.get(true, {}).get(pred, 0) for pred in colors] for true in colors]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(colors)))
    ax.set_xticklabels(colors)
    ax.set_yticks(range(len(colors)))
    ax.set_yticklabels(colors)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Color confusion matrix")

    max_val = max((v for row in matrix for v in row), default=0)
    for i in range(len(colors)):
        for j in range(len(colors)):
            value = matrix[i][j]
            text_color = "white" if max_val and value > max_val / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_example_grid(examples, output_path, n_correct=2, n_incorrect=2, pad=25):
    """Grid of a few annotated example crops, some correct, some wrong - the
    failure examples are the most useful part of a write-up.
    """
    correct = [e for e in examples if e["correct"]][:n_correct]
    incorrect = [e for e in examples if not e["correct"]][:n_incorrect]
    chosen = correct + incorrect
    if not chosen:
        return

    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4))
    if len(chosen) == 1:
        axes = [axes]

    for ax, example in zip(axes, chosen):
        image = cv2.imread(str(resolve_image_path(example["image_name"])))
        x1, y1, x2, y2 = example["box"]
        h, w = image.shape[:2]
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = image[cy1:cy2, cx1:cx2].copy()

        box_color = (0, 255, 0) if example["correct"] else (0, 0, 255)
        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), box_color, 2)

        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        status = "correct" if example["correct"] else "WRONG"
        ax.set_title(f"{status}: true={example['true_color']}, pred={example['predicted_color']}", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    results_csv = "outputs/results.csv"
    stats = evaluate(results_csv, collect_examples=True)

    plot_color_counts(results_csv, "outputs/color_counts.png")
    print("Saved outputs/color_counts.png")

    plot_confusion_matrix(stats["confusion"], "outputs/confusion_matrix.png")
    print("Saved outputs/confusion_matrix.png")

    save_example_grid(stats["examples"], "outputs/example_grid.png")
    print("Saved outputs/example_grid.png")
