# Traffic Light Detection & Color Classification

Detects traffic lights in road **images** with a YOLO detector, classifies each
light's state (red/yellow/green) with HSV color analysis, and evaluates accuracy
against the dataset's ground-truth labels. Results are logged to CSV and
summarized with Matplotlib.

**Stack:** Python, OpenCV, Ultralytics YOLO, PyTorch, NumPy, CSV, Matplotlib

> **Status:** Under construction, built phase by phase — see [plan.md](plan.md) for
> the full build plan and current progress. Sections below marked *(coming soon)*
> aren't implemented yet.

## Setup

1. Create and activate a virtual environment (Python 3.13):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Confirm the environment works:
   ```
   python -c "import cv2, ultralytics, numpy, matplotlib"
   ```

## Project layout

```
src/
  config.py     HSV ranges + tunable thresholds
  color.py      color classification (Phase B)
  detect.py     YOLO detection on images (Phase C)
  pipeline.py   run detect+color over the image set, write results.csv (Phase D)
  evaluate.py   accuracy vs. labels: precision/recall, confusion matrix (Phase F)
  report.py     stats + plots (Phase G)
  main.py       end-to-end entrypoint (Phase H)
data/       dataset images + labels (git-ignored, not committed)
outputs/    results.csv, plots (git-ignored, not committed)
```

## Running the pipeline *(coming soon)*

Once Phase H is complete, the full pipeline will run as:
```
python -m src.main --data data/<dataset_folder>
```

## Expected outputs

Running the pipeline produces:
- `outputs/results.csv` — one row per detected box (`image_name, x1, y1, x2, y2,
  predicted_color, confidence`)
- printed detection precision/recall and a color confusion matrix *(coming soon,
  Phase F)*
- `outputs/color_counts.png` — bar chart of predicted count per color *(coming
  soon, Phase G)*
- `outputs/confusion_matrix.png` — confusion-matrix heatmap *(coming soon, Phase G)*
- a saved grid of a few annotated example images (correct and incorrect)
  *(coming soon, Phase G)*

**Current `results.csv` scope:** to keep iteration fast on a CPU-only machine (the
full ~43k-image dataset takes roughly an hour), this run covers a capped subset —
`dayClip1`, `dayClip5`, `dayClip7`, `nightClip1`, `nightClip2` — sampled every 5th
frame per clip (`stride=5` in `run_pipeline`) so each clip's full color-state range
is still represented, not just its first few seconds. 2,090 images processed, 6,795
boxes found. A full-dataset run can be done later (e.g. before final submission) by
calling `run_pipeline()` with its defaults.

## Custom-trained detector (Phase E)

A YOLOv8n model was fine-tuned on a LISA subset, entirely on CPU (`python -m
tools.train`, see `configs/data.yaml` for the dataset and `tools/train.py` for the
profile):

- **Train:** `dayClip1`, `dayClip5`, `nightClip2` (511 images, 1594 boxes)
- **Val:** `dayClip7`, `nightClip1` — different clips than train, per Phase E2
  (151 images, 451 boxes)
- **Profile:** 15 epochs, imgsz 416, batch 8 — 15.3 minutes wall time
- **Result (`best.pt`, on val):** Precision 0.722, Recall 0.702, mAP50 0.651,
  mAP50-95 0.328
- Weights at `outputs/train_runs/lisa_traffic_light/weights/best.pt` (git-ignored
  like all other outputs - retrain locally with the command above to reproduce)

**Pretrained vs. custom comparison (Phase E5)** — `python -m tools.compare_detectors`,
run on the held-out val clips only (`dayClip7`, `nightClip1` - never seen during
training), matching predictions to ground truth by IoU>=0.3:

| Model | Recall | Precision |
|---|---|---|
| Pretrained (`yolov8n.pt`) | 0.783 | 0.774 |
| Custom (`best.pt`) | 0.585 | 0.923 |

The custom model is far more precise (fewer false positives) but misses more real
lights than pretrained - with only 511 training images and 15 epochs, it learned to
be conservative rather than to generalize broadly. The improved-or-not conclusion
(Phase E6) is still to come.

## Data

The **[LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)**
(Kaggle, ~5GB extracted) goes in `data/lisa/` (git-ignored, not committed).

Extracted layout (each top-level folder has one redundant nested copy of itself,
e.g. `dayTrain/dayTrain/...` — a quirk of this Kaggle mirror, not a mistake):

```
data/lisa/
  dayTrain/dayTrain/dayClip<N>/frames/dayClip<N>--<frame>.jpg      training clips, day
  nightTrain/nightTrain/nightClip<N>/frames/...                    training clips, night
  daySequence1/daySequence1/frames/...                             day test sequence
  daySequence2/daySequence2/frames/...
  nightSequence1/nightSequence1/frames/...                         night test sequence
  nightSequence2/nightSequence2/frames/...
  sample-dayClip6/, sample-nightClip1/                             small samples
  Annotations/Annotations/<same clip/sequence names>/
    frameAnnotationsBOX.csv     one row per annotated light box (ground truth)
    frameAnnotationsBULB.csv    per-bulb annotations (finer-grained, not used here)
```

`frameAnnotationsBOX.csv` is `;`-separated with a header row; the columns used by
this project are `Filename`, `Annotation tag`, and the four box corners
(`Upper left corner X/Y`, `Lower right corner X/Y`). `Annotation tag` values map to
our 3 colors:

| Tag(s) | Color |
|---|---|
| `go` | green |
| `stop`, `stopLeft` | red |
| `warning`, `warningLeft` | yellow |

(`*Left` variants are directional arrow lights — still scored by color, direction is
ignored.)

## Limitations & future work

*(To be documented in Phase I, after the pipeline is built and evaluated.)*
