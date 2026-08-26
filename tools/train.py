"""Fine-tune pretrained YOLO on the LISA subset assembled by Phase E2.

Train profile chosen for a CPU-only machine (no GPU available here - see
plan.md Phase E). Sized from an empirical timing probe (~130ms/image/epoch on
this CPU): 511 train / 151 val images x 15 epochs x imgsz 416 comes out to
roughly 15-20 minutes, well inside the "small/fast profile" budget agreed on
before starting Phase E.

Run from the project root:
    python -m tools.train
"""

from pathlib import Path

from ultralytics import YOLO

DATA_CONFIG = "configs/data.yaml"
BASE_WEIGHTS = "yolov8n.pt"

# `project` must be absolute: Ultralytics' global settings.json sets its own
# runs_dir, and a *relative* project path gets silently prepended with
# "<runs_dir>/detect/" instead of being used as-is (confirmed empirically -
# a first run landed at runs/detect/outputs/train_runs/... instead of
# outputs/train_runs/...). An absolute path is used verbatim.
TRAIN_PROFILE = dict(
    epochs=15,
    imgsz=416,
    batch=8,
    device="cpu",
    project=str(Path("outputs/train_runs").resolve()),
    name="lisa_traffic_light",
)


def train():
    model = YOLO(BASE_WEIGHTS)
    results = model.train(data=DATA_CONFIG, **TRAIN_PROFILE)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training done. best.pt -> {best_weights}")
    return best_weights


if __name__ == "__main__":
    train()
