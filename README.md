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

## Expected outputs *(coming soon)*

Running the pipeline will produce:
- `outputs/results.csv` — one row per detected box (`image_name, x1, y1, x2, y2,
  predicted_color, confidence`)
- printed detection precision/recall and a color confusion matrix
- `outputs/color_counts.png` — bar chart of predicted count per color
- `outputs/confusion_matrix.png` — confusion-matrix heatmap
- a saved grid of a few annotated example images (correct and incorrect)

## Data

The **LISA Traffic Light Dataset** goes in `data/` (git-ignored). Folder layout:
*(to be documented in Phase A4 once the dataset is downloaded).*

## Limitations & future work

*(To be documented in Phase I, after the pipeline is built and evaluated.)*
