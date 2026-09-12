"""Reproduces the RT-DETR-L training run used for this project's weights/best.pt.

Trains against the project-relative yolo_dataset/data.yaml, using the exact
config the final checkpoint was produced with. See README.md's "Model &
training" section for the corresponding verified training environment
(NVIDIA T4, Python 3.12.13, PyTorch 2.10.0+cu128, Ultralytics 8.4.146).

Usage:
    python scripts/train_rtdetr.py
"""

from pathlib import Path

from ultralytics import RTDETR

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "yolo_dataset" / "data.yaml"


def main() -> None:
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        seed=0,
        deterministic=False,
        save_period=1,
        project="rap_runs",
        name="train",
    )


if __name__ == "__main__":
    main()
