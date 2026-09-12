"""RT-DETR inference for /detect."""

import io
from pathlib import Path
from typing import List, Dict, Tuple

from PIL import Image, UnidentifiedImageError

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "best.pt"

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import RTDETR
        _model = RTDETR(str(WEIGHTS_PATH))
    return _model


def run_inference(image_bytes: bytes) -> Tuple[List[Dict], int, int]:
    """Returns (detections, image_width, image_height).

    Raises ValueError if image_bytes cannot be decoded as an image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Uploaded file is not a valid image") from e

    results = _get_model().predict(image, verbose=False)[0]

    height, width = results.orig_shape
    names = results.names

    detections = [
        {
            "class": names[int(cls_idx)],
            "confidence": float(conf),
            "bbox": [float(x) for x in box],
        }
        for box, conf, cls_idx in zip(
            results.boxes.xyxy, results.boxes.conf, results.boxes.cls
        )
    ]

    return detections, width, height
