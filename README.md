# Constrained Object Detection & Reasoning API

A FastAPI service that fine-tunes RT-DETR on a Roboflow-sourced electrical-equipment
dataset (Part A) and layers a deterministic, hand-written natural-language
reasoning module on top of its detections (Part B) — with no agentic
frameworks anywhere in the pipeline.

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Requires `weights/best.pt` to be present (see **Model weights** below). Visit
`http://localhost:8000` for a minimal browser UI (`app/static/index.html`), or
call the endpoints directly.

## Dataset provenance

- Source: Roboflow project **"Station" v2** (workspace `ppe-detection-7kco8`,
  slug `station-raaga-ogkgf`), License **CC BY 4.0**.
- 9,718 images in the exported v2 COCO dataset, including augmentation-generated images (2025-09-23 export).
- Root COCO category `helmet-insulator` (id 0) is the project's unannotated
  supercategory and is intentionally excluded from training.
- **8 trained classes**: `arc`, `disconnector`, `disconnector_open`,
  `insulator`, `spark`, `switch`, `switchgear`, `transformer`.
- Pipeline: raw `train/`/`valid/`/`test/` (COCO) → `convert_coco.py` →
  `yolo_dataset/` (YOLO format + `data.yaml`), the format actually used for
  training. `yolo_dataset/data.yaml` uses a relative dataset root
  (`path: yolo_dataset`) and is portable across machines/clones.
- Validation scripts (run against the raw export before trusting a
  conversion):
  - `integrity.py` — cross-checks images on disk vs. COCO references,
    MD5-based duplicate/cross-split-leak detection.
  - `audit.py` — per-class annotation counts per split, flags invalid boxes.
  - `dataset_stats.py` — image dimension histograms, bbox size stats,
    unannotated-image counts.

## Model & training

- **RT-DETR-L**, fine-tuned via Ultralytics.
- Training config: `epochs=30, imgsz=640, batch=8, device=0, seed=0,
  deterministic=False, save_period=1`, run on Kaggle (after an earlier Colab
  run was interrupted by a GPU usage-limit disconnect at epoch 28/50).
- Training environment: **NVIDIA T4**, Python **3.12.13**, PyTorch
  **2.10.0+cu128**, Ultralytics **8.4.146**.
- Exact training wall-clock time was not recorded.

## Evaluation metrics

Full held-out test set (568 images, 1,934 instances), `model.val()` on
`weights/best.pt`:

| Metric | Value |
|---|---|
| Precision | 0.916 |
| Recall | 0.896 |
| mAP50 | 0.933 |
| mAP50-95 | 0.669 |

Per-class breakdown and supporting plots (PR curves, confusion matrices) are
in `runs/detect/val/`. Note: this evaluation ran in the local development
environment (Ultralytics 8.4.147, Python 3.12.2, torch 2.8.0+cpu, CPU-only) —
not the Kaggle training environment.

### Failure analysis

Five failure cases were mined from real inference on held-out test images and
their root causes documented, including at least one case where the root
cause is **explicitly left unconfirmed** rather than overstated, and one
false-positive case in dense clutter (a spurious detection wedged between
correctly-detected instances, not a hallucinated class).

## API endpoints

### `GET /health`

```
200 OK
{"status": "ok"}
```

### `POST /detect`

Multipart file upload; runs RT-DETR-L inference.

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@example.jpg"
```

```json
200 OK
{
  "image_id": "example.jpg",
  "detections": [
    {"class": "insulator", "confidence": 0.91, "bbox": [120.4, 88.2, 210.6, 340.7]}
  ],
  "image_width": 640,
  "image_height": 480
}
```

Error responses:
- `400` if the upload isn't a valid image:
  `{"detail": "Uploaded file is not a valid image"}`
- `500` on an unexpected internal error:
  `{"detail": "Internal server error"}`

### `POST /ask`

JSON body: a question plus a `DetectionResult` (typically the output of a
prior `/detect` call).

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "How many switches are in this image?",
        "detections": {
          "image_id": "example.jpg",
          "image_width": 640,
          "image_height": 480,
          "detections": [
            {"class": "switch", "confidence": 0.88, "bbox": [40.1, 60.3, 120.5, 180.2]},
            {"class": "switch", "confidence": 0.79, "bbox": [200.0, 55.0, 280.4, 175.6]},
            {"class": "switch", "confidence": 0.73, "bbox": [360.2, 70.8, 440.9, 190.1]}
          ]
        }
      }'
```

```json
200 OK
{
  "question": "How many switches are in this image?",
  "intent": "COUNT",
  "answer_state": "ANSWERED",
  "answer": "There are 3 switches detected in this image."
}
```

## Reasoning behavior (Part B)

Hand-written control flow only — no LangChain/LangGraph/CrewAI/AutoGen or any
agentic framework, per the assignment's constraint.

- **Intents**: `PRESENCE`, `COUNT`, `LOCATION`, `RISK_ASSESSMENT`,
  `LIST_ALL`, `OUT_OF_SCOPE`.
- **Confidence handling**: detections below `CONF_THRESH=0.5` are excluded;
  detections between 0.5 and `AMBIGUITY_BAND_HIGH=0.6` are still reported but
  hedged in phrasing rather than stated flatly.
- **Risk classes**: `HIGH_RISK_CLASSES={arc, spark}`,
  `MEDIUM_RISK_CLASSES={disconnector_open}`.
- When detections don't support a confident answer, the API explicitly
  returns `answer_state: "INSUFFICIENT_INFO"` rather than guessing.
- Questions about out-of-taxonomy objects (e.g. "person", "helmet") are
  routed to `OUT_OF_SCOPE` rather than misapplied to the 8 known classes.

## Tests

27 tests total, all passing:
- `app/tests/` — **13 tests** (FastAPI endpoint tests, RT-DETR inference
  mocked out; covers `/health`, `/detect`, `/ask`, validation errors, the
  invalid-image 400 path, and the unhandled-error 500 path).
- `part_b/tests/` — **14 tests** (reasoning-layer unit tests covering every
  intent, confidence-band behavior, and out-of-scope handling).

```bash
pytest app/tests/ part_b/tests/
```

## Docker

A `Dockerfile` is provided (CPU-only PyTorch wheels via
`--index-url https://download.pytorch.org/whl/cpu`, avoiding an unnecessary
CUDA-runtime install):

```bash
docker build -t rap-electrical-inspection-api .
docker run -p 8000:8000 rap-electrical-inspection-api
```

**Note**: this Dockerfile has been reviewed but not built or run in this
development environment (no local Docker installation was available) — it
has not been locally verified end-to-end.

## Model weights

`weights/best.pt` (RT-DETR-L, ~66 MB) is the only weight file required to run
the API — `app/detector.py` loads it via a project-relative path. This size
is under GitHub's 100 MB hard limit but above its 50 MB soft-warning
threshold; Git LFS or external hosting is worth considering if repository
size becomes a concern.

## Reproducibility

1. Raw Roboflow export (`train/`, `valid/`, `test/`) → `convert_coco.py` →
   `yolo_dataset/` (regenerate rather than hand-edit `data.yaml` if the
   dataset changes).
2. Run `integrity.py` and `audit.py` against the raw export before trusting a
   conversion.
3. Train RT-DETR-L via Ultralytics using the config in **Model & training**
   above.
4. Evaluate with `model.val(data='yolo_dataset/data.yaml', split='test')`.
5. Run `app/detector.py` (loads `weights/best.pt`) via the FastAPI app for
   inference and reasoning.
