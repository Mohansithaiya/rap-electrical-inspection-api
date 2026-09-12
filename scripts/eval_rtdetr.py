"""Reproduces the final held-out test evaluation of weights/best.pt.

Usage:
    python scripts/eval_rtdetr.py
"""

from pathlib import Path

from ultralytics import RTDETR

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = REPO_ROOT / "yolo_dataset" / "data.yaml"
WEIGHTS = REPO_ROOT / "weights" / "best.pt"


def main() -> None:
    model = RTDETR(str(WEIGHTS))
    results = model.val(data=str(DATA_YAML), split="test")

    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall:    {results.box.mr:.3f}")
    print(f"mAP50:     {results.box.map50:.3f}")
    print(f"mAP50-95:  {results.box.map:.3f}")


if __name__ == "__main__":
    main()
