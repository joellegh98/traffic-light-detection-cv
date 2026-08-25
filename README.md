# Traffic Light Detection & Monitoring System

Detects, tracks, and monitors traffic-light states (red/yellow/green) in road
videos using a YOLO detector, HSV color analysis, SQLite storage, and Matplotlib
reporting.

**Stack:** Python, OpenCV, Ultralytics YOLO, PyTorch, NumPy, SQLite, Matplotlib

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
src/traffic_light_monitor/
  config.py          HSV ranges + tunable thresholds
  color.py           color classification (Phase B)
  detect_track.py    detection + tracking (Phase C)
  db.py              SQLite persistence (Phase D)
  analyze.py         run collapsing, stats, plots (Phases F-G)
  main.py            end-to-end entrypoint (Phase H)
data/       input videos (git-ignored, not committed)
outputs/    annotated video, plots (git-ignored, not committed)
```

## Running the pipeline *(coming soon)*

Once Phase H is complete, the full pipeline will run as:
```
python -m traffic_light_monitor.main --video data/<your_video>.mp4
```

## Expected outputs *(coming soon)*

Running the pipeline on a video will produce:
- `outputs/annotated_<video>.mp4` — input video with tracked boxes, IDs, and
  detected color labels drawn per frame
- a SQLite `.db` file with one row per tracked light per frame (`track_id`,
  `frame_index`, `color`)
- printed per-light interval stats (color, start/end time, duration)
- `outputs/color_totals.png` — bar chart of total time per color
- `outputs/timeline.png` *(optional)* — per-light color timeline over the video

## Data

Put input road videos in `data/` (git-ignored). Filename convention: *(to be
documented in Phase A4 once a sample video is added).*

## Limitations & future work

*(To be documented in Phase I, after the pipeline is built and evaluated.)*
