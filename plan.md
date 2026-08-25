# Traffic Light Detection & Monitoring System — Build Plan

Detect, track, and monitor traffic-light states (red/yellow/green) in road videos
using a YOLO detector, HSV color analysis, SQLite storage, and Matplotlib reporting.

**Stack:** Python, OpenCV, Ultralytics YOLO, PyTorch, NumPy, SQLite, Matplotlib,
Git/GitHub


## Scope choices (decide before you start)

Two parts of this plan are *rigor upgrades* — great for a portfolio, but extra work
that isn't computer vision. Pick now so scope doesn't balloon silently:

- **Layout.** Simple: flat modules in `src/` (easier to read and explain, fine for a
  course). Rigorous: a `src/traffic_light_monitor/` package with `__init__.py`
  (cleaner imports, more "real project", more packaging friction). This plan lists
  the package layout; drop the package folder and keep the files flat if you chose
  simple.
- **Tests.** A `tests/` suite is included as **optional**. If you have time, it
  strengthens the project a lot. If time is tight, skip it and put the hours into the
  CV and the error analysis instead.

Everything else is core.

---

## The mental model (read once)

One loop, plus a store, plus a report:

1. Take a frame → 2. YOLO detects light boxes → 3. tracker gives each a stable ID →
4. crop each box → 5. decide its color → 6. save `(id, frame, color)` to the DB →
7. repeat every frame → 8. collapse per-frame rows into durations → 9. stats + plots.

Detection, tracking, color, storage, analysis are separate problems, built one at a
time.

## Two different "databases" — don't confuse them

- **Training dataset (LISA / Bosch):** labeled images used *only* in Phase E to teach
  YOLO. Input to training.
- **SQLite database:** a file your program writes at runtime holding *your* results.
  Output store, used in Phases D, F, G.

You use both; they do unrelated jobs.

---

## Relevant files

```
root: README.md, requirements.txt, .gitignore, plan.md
src/traffic_light_monitor/   (or flat src/ if you chose simple layout)
  __init__.py        (package layout only)
  config.py          HSV ranges + tunable thresholds in one place
  color.py           Phase B
  detect_track.py    Phase C
  db.py              Phase D
  analyze.py         Phases F–G
  main.py            Phase H
tools/convert_labels.py, configs/data.yaml   (Phase E, training)
tests/test_color.py, test_db.py, test_analyze.py   (optional rigor)
data/       input videos (git-ignored)
outputs/    annotated video, plots (git-ignored)
```

---

## Phase A — Project foundation
*Blocks all later phases.*

**Concept.** A clean repo and an isolated environment so packages don't clash and Git
tracks each phase. SQLite ships with Python — nothing to install for the database.

**Tasks.**
- [x] A1. Initialize repo + virtual environment; install `ultralytics`,
  `opencv-python`, `numpy`, `matplotlib`; confirm imports.
- [x] A2. Create the skeleton (see Relevant files) and a `.gitignore` for `venv/`,
  `data/`, `outputs/`, `__pycache__/`, `*.pt`, `*.db`.
- [x] A3. README bootstrap: setup, run-command placeholders, expected outputs.
- [ ] A4. Put one road video in `data/`; note the filename convention in the README.

**Verify.** venv activates; `python -c "import cv2, ultralytics, numpy, matplotlib"`
runs clean. First commit.

---

## Phase B — Color classification
*Depends on A.*

**Concept.** Raw images mix color and brightness (BGR), making "is this red?" hard.
**HSV** splits Hue (which color), Saturation (how vivid), Value (how bright), so you
can ask "hue in the red range *and* bright enough to be a lit lamp?" The brightness
test excludes the dark housing. OpenCV hue is 0–180 and red sits at *both* ends, so
red needs two hue bands; yellow and green need one each. Build and test this alone,
before video, so you trust it later.

**Tasks.**
- [ ] B1. Put HSV thresholds in `config.py` (two red bands, one each for yellow/green;
  a Value floor; a minimum-active-pixel count).
- [ ] B2. `classify_color(crop)` → `"red"|"yellow"|"green"|"unknown"`; guard empty /
  `None` crops. Convert to HSV, mask hue-in-range **and** bright pixels per color,
  count, highest wins; below the minimum count → `"unknown"`.
- [ ] B3. A quick manual-validation script for three representative crops.
- [ ] B4. Tune the Value floor and minimum pixels to suppress housing/background.

**Verify.** Three hand-cropped test images (one per color) are each labeled
correctly. Commit.

---

## Phase C — Detection and tracking
*Depends on A; uses B.*

**Concept.** Pretrained YOLO already knows "traffic light" (COCO class 9), so you can
detect with zero training. Detection alone treats every frame independently — a
**tracker** links detections across frames and assigns a stable ID, which is what
makes "how long was *this* light red?" answerable later. The library has tracking
built in; enable it with ID persistence rather than building one. Do frame-level
detection first, confirm boxes land right, *then* add tracking.

**Tasks.**
- [ ] C1. Load pretrained YOLO (`yolov8n.pt`), constrained to the traffic-light class.
- [ ] C2. Frame-level annotate utility for sampled-frame sanity checks.
- [ ] C3. Enable tracking mode with persisted IDs across frames.
- [ ] C4. Per tracked box: crop, call `classify_color`, draw ID + color label.
- [ ] C5. Collect in-memory `(track_id, frame_index, color)` tuples for the DB path.
- [ ] C6. Save the annotated video to `outputs/`.

**Verify.** Annotated video shows boxes with stable IDs and color labels following
each light across frames. Commit.

---

## Phase D — Persistence layer
*Depends on C.*

**Concept.** Instead of holding results in memory, store one row per light per frame
in SQLite: persistence (process the slow video once, analyze many times), easy
querying, and a clean split — the pipeline only *writes*, analysis only *reads*.

**Tasks.**
- [ ] D1. SQLite init with an `observations` table (`id` PK, `track_id`,
  `frame_index`, `color`) and indexes on `track_id`, `frame_index`.
- [ ] D2. `insert_observation(...)`; use a transaction/batch for efficient writes.
- [ ] D3. Persist video metadata needed later — at minimum **FPS** (a small `meta`
  table works).
- [ ] D4. Wire the Phase C loop to insert one row per tracked light per frame.
- [ ] D5. `fetch_all` returning rows ordered by `track_id` then `frame_index`.

**Verify.** After processing a clip, the `.db` exists with a non-zero row count;
sample ordered rows look sane. Commit.

---

## Phase E — Custom training *(optional; research it in parallel with D)*
*Improves C; compared against pretrained.*

**Concept.** Fine-tuning starts from the pretrained model and keeps training it on
traffic-light images, improving the hard cases (small/distant lights, your camera's
look). Keep a **single** class, "traffic light"; let Phase B handle color — fewer
classes = simpler training and a cleaner report. Slowest phase, mostly waiting; far
faster on a GPU or free cloud notebook. **Skippable** — "I evaluated pretrained and
it was sufficient" is a valid, reportable result. Your title says custom-trained,
though, so you'll likely want it.

**Tasks.**
- [ ] E1. Choose data (LISA = US lights, common; Bosch = harder/distant; or label
  your own frames) and document why. *Note: LISA is US lights; if your footage is
  Israeli roads, test transfer and mention it in the write-up.*
- [ ] E2. `tools/convert_labels.py` → YOLO format, single class `traffic_light`.
- [ ] E3. `configs/data.yaml` (train/val paths, one class) + a train profile
  (epochs/imgsz/device).
- [ ] E4. Train; capture the `best.pt` path and metrics.
- [ ] E5. Re-run C with pretrained **and** custom weights for comparison.
- [ ] E6. Record the conclusion: custom improved results, or pretrained was enough.

**Verify.** Detection is clearly better than pretrained, or you've documented that it
wasn't needed. Commit.

---

## Phase F — Run collapsing and metrics
*Depends on D; works on C-only or E outputs.*

**Concept.** The DB has one row per light per frame. Durations come from grouping by
light ID, ordering by frame, and merging runs of the same color into intervals.
Convert frames to seconds with the video's **real** FPS — read it from the video,
never assume a fixed value, or every duration is wrong. Detections flicker (a stray
misread frame), so drop color runs only a frame or two long to avoid splitting one
real interval into several.

**Tasks.**
- [ ] F1. `collapse_runs(rows, fps, min_run)`: group by `track_id`, sort by frame,
  merge contiguous same-color runs, drop sub-`min_run` runs.
- [ ] F2. Frames → seconds using the stored real FPS.
- [ ] F3. Per-light interval table: color, start, end, duration.
- [ ] F4. Aggregate totals and averages per color across the video.
- [ ] F5. (Optional) transition counts and estimated cycle lengths.

**Verify.** Per-light interval table prints with plausible second durations. Commit.

---

## Phase G — Reporting visuals
*Depends on F.*

**Concept.** The "monitoring" payoff — durations become numbers and pictures for the
report.

**Tasks.**
- [ ] G1. Save a bar chart of total time per color → `outputs/color_totals.png`.
- [ ] G2. (Optional) timeline plot: time on X, light IDs on Y, colored bar per
  interval → `outputs/timeline.png`.
- [ ] G3. Deterministic output naming + overwrite policy.

**Verify.** Required plot files are generated and open correctly. Commit.

---

## Phase H — End-to-end entrypoint
*Depends on C, D, F, G.*

**Concept.** One command runs everything, so the project is reproducible.

**Tasks.**
- [ ] H1. `main.py` runs the full pipeline: video → annotated video + filled DB →
  durations → stats + plots.
- [ ] H2. Model-selection option (pretrained vs custom weights path).
- [ ] H3. Progress logging + a final summary of artifact paths.
- [ ] H4. Graceful messages for missing video / model / DB problems.

**Verify.** One command produces the annotated video, DB, printed stats, and plots.
Commit.

---

## Phase I — Documentation and delivery
*Depends on all implementation phases.*

**Concept.** Show you understand the system, not just that it runs.

**Tasks.**
- [ ] I1. README: install, run examples, outputs, interpretation notes.
- [ ] I2. Limitations & future work — name the hard cases *and why*: small/distant
  lights; night vs. day (lit bulb vs. dark reflective housing); overexposed lights
  washing toward white and breaking color detection; red/yellow confusion under some
  lighting.
- [ ] I3. Final QA pass; commit by phase milestones; push to GitHub.

---

## Optional: tests (rigor upgrade)
*Do only if time allows — skip in favor of CV work and error analysis if tight.*

- [ ] `tests/test_color.py` — `classify_color` on fixed crops, incl. empty-crop guard.
- [ ] `tests/test_db.py` — insert then `fetch_all`, check count and ordering.
- [ ] `tests/test_analyze.py` — `collapse_runs` on a tiny synthetic log, check
  intervals and that flicker filtering works.

---

## Verification summary (whole project)

- Environment: import smoke test passes.
- Color: classifier correct on ≥3 labeled crops.
- Detection: boxes correct on sample frames.
- Tracking: IDs stable across consecutive frames in the annotated output.
- DB: non-zero row count; ordered rows inspected.
- Analysis: collapsed intervals have plausible second durations.
- Plots: required files generated and readable.
- End-to-end: one command → annotated video + DB + stats + plots.
- Regression (if tests built): color/db/analyze suites pass.

---

## Time budget

| Phase | Result | Rough time |
|------|--------|------------|
| A | Environment + repo | 30 min |
| B | Color classifier you trust | 1–2 hrs |
| C | Detection + tracking + color on video | 3–4 hrs |
| D | Observation database | 1–2 hrs |
| E | Custom-trained detector *(optional)* | 3–5 hrs |
| F | Clean per-light durations | 2 hrs |
| G | Stats + plots | 1–2 hrs |
| H | One-command pipeline | 1 hr |
| I | README + write-up | 1–2 hrs |
| Tests | Regression suite *(optional)* | 1–2 hrs |

Demoable result after **Phase C** — everything after is storage, analysis, or rigor,
so you're never left with nothing.

---

## Working with the AI without losing the learning

- Feed it **one phase at a time**, with that phase's Concept — not the whole doc.
- Read the Concept first so you can judge whether the generated code is right.
- Check each result against the Tasks and the Verify step before ticking boxes.
- When code breaks, understand the fix, don't just paste it — that's where the CV
  learning happens.
- Graded project: check your course's rules on AI-assisted code, and be ready to
  explain any line you submit.

---

## What you'll have learned

- HSV color segmentation (B)
- Object detection and pretrained models (C)
- Multi-object tracking and persistent IDs (C)
- Database design and separating collection from analysis (D)
- Fine-tuning a neural network on a custom dataset (E)
- Turning noisy per-frame data into clean statistics (F–G)
- Assembling, testing, and documenting a real CV project (H–I)
