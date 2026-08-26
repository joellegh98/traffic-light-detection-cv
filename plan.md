# Traffic Light Detection & Color Classification — Build Plan

Detect traffic lights in road **images** with a YOLO detector, classify each light's
state (red / yellow / green) with HSV color analysis, and evaluate accuracy against
the dataset's labels. Results are logged to CSV and summarized with Matplotlib.

**Stack:** Python, OpenCV, Ultralytics YOLO, PyTorch, NumPy, CSV, Matplotlib,
Git/GitHub

*Image-based by design: no video, no tracking, no durations — so there are no
footage-sourcing headaches and you can measure real accuracy against labeled data.*

---

## Scope choices (decide before you start)

Two parts are *rigor upgrades* — nice for a portfolio, but extra work that isn't
computer vision. Pick now so scope doesn't balloon silently:

- **Layout.** Simple: flat modules in `src/` (easier to read and explain). Rigorous:
  a `src/traffic_light_detector/` package with `__init__.py` (cleaner imports, more
  packaging friction). This plan lists the flat layout; add the package folder only
  if you want the "real project" signal.
- **Tests.** A `tests/` suite is **optional**. Strengthens the project if you have
  time; skip it in favor of the CV and error analysis if not.

Everything else is core.

---

## The mental model (read once)

For each image: **YOLO detects light boxes → crop each box → HSV decides its color →
log `(image, box, color)`.** Then, because the dataset is labeled, compare your
predictions to the ground truth and report accuracy. Detection, color, logging, and
evaluation are separate problems, built one at a time.

## Two different "data" things — don't confuse them

- **Training dataset (LISA):** labeled images used to *teach* YOLO (Phase E) and to
  *score* your accuracy (Phase F). Input.
- **Results CSV:** a file your program writes holding *your* predictions. Output,
  used in Phases D, F, G.

## Why CSV, not a database?

The database in the old (video) plan existed to store thousands of per-frame rows so
you could re-run analysis without re-processing the video. With images there's no
per-frame explosion and no "slow video to avoid re-running" — a flat results table
(`image, box, predicted_color`) is all you need, and a **CSV** covers it with far
less machinery. You can swap in SQLite later if you specifically want it on your CV,
but CSV is the right default here.

---

## Relevant files

```
root: README.md, requirements.txt, .gitignore, plan.md
src/
  config.py        HSV ranges + tunable thresholds in one place
  color.py         Phase B — classify_color(crop)
  detect.py        Phase C — YOLO detection on images
  pipeline.py      Phase D — run detect+color over the image set, write results CSV
  evaluate.py      Phase F — accuracy vs labels
  report.py        Phase G — stats + plots
  main.py          Phase H — one-command entry point
tools/convert_labels.py, configs/data.yaml   (Phase E, training)
tests/test_color.py, test_evaluate.py         (optional rigor)
data/       dataset images + labels (git-ignored)
outputs/    results.csv, plots (git-ignored)
```

---

## Phase A — Project foundation
*Blocks all later phases.*

**Concept.** A clean repo and an isolated environment so packages don't clash and Git
tracks each phase.

**Tasks.**
- [x] A1. Initialize repo + virtual environment; install `ultralytics`,
  `opencv-python`, `numpy`, `matplotlib`; confirm imports.
- [x] A2. Create the skeleton (see Relevant files) and a `.gitignore` for `venv/`,
  `data/`, `outputs/`, `__pycache__/`, `*.pt`, `*.csv`.
- [x] A3. README bootstrap: setup, run-command placeholders, expected outputs.
- [x] A4. Download the **LISA Traffic Light Dataset** (search "LISA traffic light
  Kaggle") into `data/`; note the folder layout in the README. It arrives as labeled
  image frames — exactly what you want.

**Verify.** venv activates; `python -c "import cv2, ultralytics, numpy, matplotlib"`
runs clean. Dataset images are present in `data/`. Commit.

---

## Phase B — Color classification
*Depends on A.*

**Concept.** Raw images mix color and brightness (BGR), making "is this red?" hard.
**HSV** splits Hue (which color), Saturation (how vivid), Value (how bright), so you
can ask "hue in the red range *and* bright enough to be a lit lamp?" The brightness
test excludes the dark housing. OpenCV hue is 0–180 and red sits at *both* ends, so
red needs two hue bands; yellow and green need one each. Build and test this alone,
before detection, so you trust it later.

**Tasks.**
- [x] B1. Put HSV thresholds in `config.py` (two red bands, one each for yellow/green;
  a Value floor; a minimum-active-pixel count).
- [x] B2. `classify_color(crop)` → `"red"|"yellow"|"green"|"unknown"`; guard empty /
  `None` crops. Convert to HSV, mask hue-in-range **and** bright pixels per color,
  count, highest wins; below the minimum count → `"unknown"`.
- [x] B3. A quick manual-validation script for three representative crops.
- [x] B4. Tune the Value floor and minimum pixels to suppress housing/background.

**Verify.** Three hand-cropped test images (one per color) are each labeled
correctly. Commit.

---

## Phase C — Detection on images
*Depends on A; uses B.*

**Concept.** Pretrained YOLO already knows "traffic light" (COCO class 9), so you can
detect with zero training. Because inputs are single images, there's no tracking to
worry about — each image is independent. Get boxes landing correctly on pretrained
first; Phase E improves them by training.

**Tasks.**
- [x] C1. In `detect.py`, load pretrained YOLO (`yolov8n.pt`), constrained to the
  traffic-light class.
- [x] C2. A function that takes an image path, returns the detected boxes.
- [x] C3. An annotate helper that draws boxes (and, using Phase B on each crop, the
  color label) on an image and saves it, for eyeballing a few samples.

**Verify.** Boxes are drawn correctly around lights on a handful of sample images,
with sensible color labels. Commit.

---

## Phase D — Run the pipeline → results CSV
*Depends on C.*

**Concept.** Run detect + color over the whole image set once and record every
prediction to a flat file. Everything downstream (accuracy, stats, plots) reads this
CSV, so the heavy detection pass and the light analysis passes stay cleanly separated.

**Tasks.**
- [x] D1. In `pipeline.py`, loop over the dataset images; for each, detect boxes and
  classify each crop's color.
- [x] D2. Write one CSV row per detected box: `image_name, x1, y1, x2, y2,
  predicted_color, confidence`. Use Python's built-in `csv` module.
- [x] D3. Save to `outputs/results.csv`; print a count of images processed and boxes
  found.

**Verify.** `results.csv` exists with a plausible row count and readable rows.
Commit.

---

## Phase E — Custom training *(optional but recommended)*
*Improves C.*

**Concept.** Fine-tuning starts from the pretrained model and keeps training it on
LISA's labeled images, improving the hard cases (small/distant lights). Keep a
**single** class, "traffic light"; let Phase B handle color — fewer classes = simpler
training and a cleaner story. Slowest phase, mostly waiting; far faster on a GPU or a
free cloud notebook. **Skippable** — "I evaluated pretrained and it was sufficient"
is a valid, reportable result.

**Tasks.**
- [x] E1. Convert LISA's labels to YOLO format with `tools/convert_labels.py`, single
  class `traffic_light` (one `.txt` per image, `class cx cy w h` normalized 0–1).
- [x] E2. Split train/val **by clip/sequence, not by random frame.** LISA frames come
  from video, so neighboring frames are near-duplicates; a random split leaks almost
  identical images into both sets and inflates your accuracy. Split so all frames from
  one sequence stay on one side.
- [x] E3. `configs/data.yaml` (train/val paths, one class) + a train profile
  (epochs / imgsz 640 / device).
- [x] E4. Train; capture the `best.pt` path and metrics.
- [x] E5. Re-run Phase D with pretrained **and** `best.pt`; compare.
- [x] E6. Record the conclusion: custom improved results, or pretrained was enough.

**Verify.** Detection is clearly better than pretrained, or you've documented it
wasn't needed. Commit.

---

## Phase F — Accuracy evaluation
*Depends on D; uses the dataset labels.*

**Concept.** This is the payoff that images give you that video didn't: the dataset is
labeled, so you can measure *real* accuracy instead of "it looked right." Two things
to score — did detection find the lights (boxes), and did HSV name the color right.

**Tasks.**
- [x] F1. In `evaluate.py`, match each predicted box to a ground-truth box by overlap
  (IoU above a threshold, e.g. 0.5).
- [x] F2. Detection metrics: precision and recall (how many real lights you found, how
  many predictions were correct).
- [x] F3. Color accuracy: over the correctly-detected lights, how often did
  `classify_color` match the labeled state.
- [x] F4. Build a **confusion matrix** for color (rows = true color, columns =
  predicted) — this shows exactly where HSV fails, e.g. red↔yellow.

**Verify.** You can print detection precision/recall and a color confusion matrix
with real numbers. Commit.

---

## Phase G — Reporting visuals
*Depends on D and F.*

**Concept.** Turn the numbers into pictures for the README.

**Tasks.**
- [x] G1. Bar chart of predicted count per color → `outputs/color_counts.png`.
- [x] G2. Confusion-matrix heatmap → `outputs/confusion_matrix.png`.
- [x] G3. Save a grid of a few annotated example images (some correct, some wrong) —
  the failure examples make the strongest part of the write-up.

**Verify.** The plot files are generated and open correctly. Commit.

---

## Phase H — End-to-end entry point
*Depends on C, D, F, G.*

**Concept.** One command runs everything, so the project is reproducible.

**Tasks.**
- [x] H1. `main.py` runs: pipeline over images → `results.csv` → evaluation → plots.
- [x] H2. Model-selection option (pretrained vs `best.pt` path).
- [x] H3. Progress logging + a final summary of where outputs were written.
- [x] H4. Graceful messages for missing dataset / model / file issues.

**Verify.** One command produces `results.csv`, printed accuracy, and the plots.
Commit.

---

## Phase I — Documentation and delivery
*Depends on all implementation phases.*

**Concept.** Show you understand the system, not just that it runs.

**Tasks.**
- [ ] I1. README: install, run examples, outputs, and your **accuracy numbers**.
- [ ] I2. Limitations & future work — name the hard cases *and why*: small/distant
  lights; overexposed lights washing toward white and breaking color detection;
  red/yellow confusion under some lighting; and that HSV is lighting-sensitive so
  results would differ on non-LISA (e.g. Israeli) images. A natural "future work"
  line: extend to video + tracking for duration analysis.
- [ ] I3. Final QA pass; commit by phase milestones; push to GitHub.

---

## Optional: tests (rigor upgrade)
*Do only if time allows.*

- [ ] `tests/test_color.py` — `classify_color` on fixed crops, incl. empty-crop guard.
- [ ] `tests/test_evaluate.py` — IoU matching and the confusion-matrix build on a tiny
  synthetic set with known answers.

---

## Verification summary (whole project)

- Environment: import smoke test passes; dataset present.
- Color: classifier correct on ≥3 labeled crops.
- Detection: boxes correct on sample images.
- Pipeline: `results.csv` has a plausible row count.
- Evaluation: detection precision/recall and a color confusion matrix print with real
  numbers.
- Plots: count chart + confusion matrix generated and readable.
- End-to-end: one command → CSV + accuracy + plots.
- Regression (if tests built): color/evaluate suites pass.

---

## Time budget

| Phase | Result | Rough time |
|------|--------|------------|
| A | Environment + repo + dataset | 45 min |
| B | Color classifier you trust | 1–2 hrs |
| C | Detection on images | 1–2 hrs |
| D | Results CSV over the image set | 1 hr |
| E | Custom-trained detector *(optional)* | 3–5 hrs |
| F | Accuracy + confusion matrix | 2–3 hrs |
| G | Stats + plots | 1–2 hrs |
| H | One-command pipeline | 1 hr |
| I | README + write-up | 1–2 hrs |
| Tests | Regression suite *(optional)* | 1 hr |

Demoable result after **Phase D** — you have detections and colors logged. Everything
after is training, evaluation, or polish, so you're never left with nothing.

---

## Working with the AI without losing the learning

- Feed it **one phase at a time**, with that phase's Concept — not the whole doc.
- Read the Concept first so you can judge whether the generated code is right.
- Check each result against the Tasks and the Verify step before ticking boxes.
- When code breaks, understand the fix, don't just paste it — that's where the CV
  learning happens.

---

## What you'll have learned

- HSV color segmentation (B)
- Object detection and pretrained models (C)
- Building a batch inference pipeline and logging results (D)
- Fine-tuning a neural network on a custom dataset, incl. correct train/val splitting
  (E)
- Evaluating a model properly: IoU matching, precision/recall, confusion matrices (F)
- Turning results into clear statistics and visuals (G)
- Assembling and documenting a real CV project (H–I)
