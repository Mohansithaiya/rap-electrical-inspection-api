# Quick Start

A FastAPI service that fine-tunes RT-DETR on an electrical-equipment dataset
for object detection, then answers natural-language questions about the
detections through a deterministic reasoning layer — no agentic frameworks.

## Get the code

```bash
git clone https://github.com/Mohansithaiya/rap-electrical-inspection-api.git
cd rap-electrical-inspection-api
```

## Prerequisites

- Python 3.12
- The model weight file at `weights/best.pt` must be present

## Installation

```bash
pip install -r requirements.txt
```

## Start

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Open

```
http://127.0.0.1:8000
```

## Demo (3 steps)

1. Upload image
2. Run Detection
3. Ask a question

### Example questions

- How many switches are in this image?
- Is there any hazard?
- Is there a person?

## API endpoints

- `GET /health`
- `POST /detect`
- `POST /ask`

## Note on reasoning

`/ask` uses deterministic reasoning over the RT-DETR detections (no LLM, no
agentic framework) and returns `INSUFFICIENT_INFO` when the evidence isn't
sufficient to answer confidently, rather than guessing.
