"""FastAPI app exposing the Constrained Object Detection & Reasoning API.

Endpoints: /health (liveness), /detect (Part A - runs the fine-tuned RT-DETR
model on an uploaded image and returns boxes/classes/confidences), and /ask
(Part B - a deterministic, hand-written reasoning layer that routes intent
and answers questions over a DetectionResult, explicitly returning
insufficient-info rather than guessing).
"""

import logging
import os
import sys

_PART_B_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part_b")
if _PART_B_DIR not in sys.path:
    sys.path.insert(0, _PART_B_DIR)

from typing import List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from schema import DetectionResult  # noqa: E402
from pipeline import answer_question  # noqa: E402
from app.detector import run_inference

app = FastAPI(
    title="Constrained Object Detection & Reasoning API",
    version="0.1.0",
    description="RT-DETR object detection (/detect) plus a deterministic Part B reasoning layer (/ask) over the detections.",
)


class DetectionIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cls: str = Field(alias="class", min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4)


class DetectionResultIn(BaseModel):
    image_id: str = ""
    detections: List[DetectionIn] = Field(default_factory=list)
    image_width: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    detections: DetectionResultIn


class AskResponse(BaseModel):
    question: str
    intent: str
    answer_state: str
    answer: str


class DetectResponse(BaseModel):
    image_id: str
    detections: List[DetectionIn]
    image_width: int
    image_height: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(file: UploadFile = File(...)) -> DetectResponse:
    image_bytes = file.file.read()
    try:
        detections, width, height = run_inference(image_bytes)
    except ValueError as e:
        logger.warning("Rejected /detect upload %r: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    logger.info(
        "/detect processed %r: %d detection(s), %dx%d",
        file.filename, len(detections), width, height,
    )
    return DetectResponse(
        image_id=file.filename or "",
        detections=detections,
        image_width=width,
        image_height=height,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    detection_result = DetectionResult.from_dict(request.detections.model_dump(by_alias=True))
    result = answer_question(request.question, detection_result)
    logger.info(
        "/ask %r -> intent=%s answer_state=%s",
        request.question, result["intent"], result["answer_state"],
    )
    return AskResponse(**result)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
