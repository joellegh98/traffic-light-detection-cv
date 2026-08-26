# Traffic Light Detection & Color Classification

Detects traffic lights in road **images** with a YOLO detector, classifies each
light's state (red/yellow/green) with HSV color analysis, and evaluates accuracy
against the dataset's ground-truth labels. Results are logged to CSV and
summarized with Matplotlib.

**Stack:** Python, OpenCV, Ultralytics YOLO, PyTorch, NumPy, CSV, Matplotlib

> **Status:** Core pipeline complete (Phases A-H) and documented (Phase I) - see
> [plan.md](plan.md) for the full build plan and phase-by-phase progress. The
> optional `tests/` suite (noted as skippable in the plan) was not built.

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
tools/
  convert_labels.py    LISA CSV -> YOLO label .txt files + train/val split (Phase E1-E2)
  train.py             fine-tune YOLO on the split (Phase E3-E4)
  compare_detectors.py pretrained vs. custom recall/precision on held-out clips (Phase E5)
  validate_color.py    manual classify_color() check against 3 real crops (Phase B3)
configs/
  data.yaml     YOLO dataset config used by tools/train.py (Phase E3)
data/       dataset images + labels (git-ignored, not committed)
outputs/    results.csv, plots, trained weights (git-ignored, not committed)
input/      10 sample images + detected/ annotated output (committed - see below)
```

## Quick demo (no dataset download needed)

`input/` has 10 sample images committed to the repo (a curated pick from LISA -
day and night, all three colors, chosen using Phase F's own evaluation data so
they're frames the detector is known to read correctly, not a random gamble).
Unlike `data/lisa/`, these don't require downloading anything.

```
python -m tools.detect_input_samples
```

Runs the pretrained detector + color classifier (the same `annotate_image()`
Phase C3 uses) over every image in `input/` and saves annotated copies - boxes
and color labels drawn on - to `input/detected/`.

*(These 10 frames are sourced from the LISA Traffic Light Dataset - see
[Data](#data) below for the full dataset and its citation.)*

## Running the pipeline

```
python -m src.main                    # pretrained detector (default, see Phase E6)
python -m src.main --weights custom   # Phase E's fine-tuned best.pt instead
python -m src.main --weights path/to/other.pt
```

One command: runs detection + color over the dataset subset described below,
scores it against ground truth, and generates all three plots. Takes ~2.5-3
minutes on a CPU-only machine. Prints a final summary of every file written.

## Expected outputs

`python -m src.main` (above) produces all of these in one run; each is also runnable
standalone during development:
- `outputs/results.csv` — one row per detected box (`image_name, x1, y1, x2, y2,
  predicted_color, confidence`) - `run_pipeline()` in `src/pipeline.py`
- printed detection precision/recall and a color confusion matrix -
  `python -m src.evaluate`
- `outputs/color_counts.png` — bar chart of predicted count per color -
  `python -m src.report`
- `outputs/confusion_matrix.png` — confusion-matrix heatmap - `python -m src.report`
- `outputs/example_grid.png` — a few annotated example detections, correct and
  wrong - `python -m src.report`

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
be conservative rather than to generalize broadly.

**Conclusion (Phase E6): pretrained is the better default for this project.**
Custom training did not improve overall detection - it's a genuine trade-off, not
a clean win. For a monitoring system, missing a real light (recall) is a more
serious failure than an extra false positive (precision): a light that's never
detected can never be tracked or timed, while a spurious box is comparatively
harmless downstream. On that basis, **pretrained `yolov8n.pt` is used as the
default detector** for the rest of this project (Phases F-H), with the
custom-trained option kept available via `--weights` (Phase H2) for anyone who
wants to reproduce or extend the comparison.

This isn't a verdict against fine-tuning in general - it's specific to a
compute-constrained, CPU-only, 511-image/15-epoch run. The custom model's much
higher precision (0.923 vs 0.774) shows it did learn something real about LISA's
lights; more training data and epochs would plausibly close, or reverse, the
recall gap. That's noted as future work (Phase I) rather than pursued further
here, given the project's CPU-only budget.

## Accuracy evaluation (Phase F)

`python -m src.evaluate` scores the pretrained detector's `outputs/results.csv`
(the D3 subset: `dayClip1/5/7`, `nightClip1/2`) against LISA's ground-truth labels -
IoU matching for detection, then color accuracy over what was actually detected.

**Detection (IoU >= 0.5):** precision 0.361, recall 0.388 (2455 true positives, 4340
false positives, 3875 false negatives). Lower than it might look at first: restricting
the same evaluation to just `dayClip7`+`nightClip1` (Phase E5's exact scope, IoU>=0.3)
reproduces E5's numbers almost exactly (recall 0.792 vs. E5's 0.783) - confirming the
evaluator is correct - so the weaker aggregate score is a real effect of the other 3
clips, not a bug. `dayClip5` in particular is the dusk/short-shutter clip flagged back
in Phase C for backlit, hard-to-detect lights; it's dragging the average down.

**Color accuracy (over the 2455 correctly-detected lights): 96.9%** (2379/2455) -
validates Phase B/B4's HSV tuning holds up at scale, not just on the handful of
crops used to tune it. The confusion matrix shows exactly one real failure mode:
19 true `yellow` lights read as `red` (13 correctly read as yellow) - the same
overexposed-yellow-measures-as-red-hue effect documented in `config.py` back in
B4, now confirmed at dataset scale rather than on a handful of samples. Red and
green both have effectively zero cross-confusion.

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

**Small/distant lights are the dominant failure mode.** Detection recall (0.388 at
IoU>=0.5, aggregate) is the weakest number in this project, and it's a detection
problem, not a color one - Phase C's eyeballing (`outputs/c3_sample*.png`) first
showed YOLO missing lights a human spots instantly at a glance, and Phase F
confirmed it numerically at dataset scale. Pretrained YOLOv8n simply wasn't
trained with tiny, distant traffic lights as a priority class.

**Backlit/dusk conditions break color classification, not just detection.**
`dayClip5` (dusk, short camera shutter) produces dark, backlit light housings
silhouetted against a bright sky - `classify_color` correctly falls back to
`unknown` rather than guess (see `outputs/example_grid.png`'s red-read-as-unknown
example), but this means real lights are being scored as color-misses even when
they're detected. This is the housing/background suppression from Phase B4
working as designed, not a bug - it's just a harder condition than daylight.

**Red/yellow confusion under overexposure.** The one real cross-color confusion
found (Phase F's confusion matrix, 19 cases): an overexposed "warning" (yellow)
light's brightest pixels measure at hue 0-11, statistically indistinguishable from
red at the pixel level - a camera/exposure effect, not a classifier bug. Widening
`RED_HUE_RANGES` to compensate would just break real red crops the other way (see
`config.py`'s B4 comments) - this is a genuine limit of HSV-only classification
under some lighting, not a tuning gap.

**HSV thresholds are tuned on LISA (US) lights, not verified elsewhere.** All of
`config.py`'s thresholds were tuned against LISA crops. LISA is US-standard traffic
lights; lights elsewhere (e.g. Israeli intersections) can differ in lamp
color/brightness, housing style, and typical camera exposure, so these exact
thresholds aren't guaranteed to transfer without re-validating against local
footage the way Phase B4 did here.

**Custom training was compute-constrained, not exhausted.** Phase E6 found
pretrained is the better default *for this run* (511 images, 15 epochs, CPU-only) -
the custom model's higher precision suggests real learning happened; more data and
epochs would plausibly close or reverse the recall gap. Worth revisiting with a
larger training set or a GPU/cloud notebook, per the original Phase E concept note.

**This project also only ever ran on a subset (`dayClip1/5/7`, `nightClip1/2`,
2,090 of ~43k images)**, chosen throughout for CPU iteration speed. `run_pipeline()`
and `main.py`'s `DEFAULT_CLIP_DIRS` both support running the full dataset - the
reported numbers would likely shift somewhat (probably down slightly, since more
clips means more of LISA's harder day/night variety) with a full run.

**Future work:** the most natural extension is what this project deliberately
scoped out - video + tracking for duration analysis (how long was a given light
red?), which was the original plan before pivoting to images specifically to get
measurable accuracy against labels. Also worth trying: a longer/larger Phase E
training run now that the pipeline and evaluation are proven correct, and testing
against non-US footage to see how far the HSV thresholds actually transfer.
