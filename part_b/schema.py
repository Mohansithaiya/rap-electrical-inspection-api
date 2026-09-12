"""Data structures shared across the Part B reasoning layer."""

from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    PRESENCE = "PRESENCE"
    COUNT = "COUNT"
    LOCATION = "LOCATION"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    LIST_ALL = "LIST_ALL"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


# The 8 classes the Part A model was trained on (must stay in sync with
# convert_coco.py CLASS_NAMES).
KNOWN_CLASSES = [
    "arc",
    "disconnector",
    "disconnector_open",
    "insulator",
    "spark",
    "switch",
    "switchgear",
    "transformer",
]

HIGH_RISK_CLASSES = {"arc", "spark"}
MEDIUM_RISK_CLASSES = {"disconnector_open"}

CONF_THRESH = 0.5
# Detections with confidence in [CONF_THRESH, AMBIGUITY_BAND_HIGH) still count as
# "present" but are hedged in the phrased answer rather than stated flatly.
AMBIGUITY_BAND_HIGH = 0.6

# Common real-world nouns that are NOT in KNOWN_CLASSES, used to distinguish
# "you asked about something outside this model's taxonomy" from "I couldn't
# figure out what class you meant".
UNSUPPORTED_CLASS_HINTS = [
    "person", "people", "human", "worker",
    "car", "vehicle", "truck",
    "dog", "cat", "animal",
    "helmet", "hard hat",
    "fire", "smoke",
    "pole", "wire", "cable", "meter", "fence", "ladder",
]


@dataclass
class Detection:
    cls: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2]


@dataclass
class DetectionResult:
    """Raw model output for one image, before any confidence filtering."""

    image_id: str
    detections: list  # list[Detection]
    image_width: int
    image_height: int

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionResult":
        dets = [
            Detection(d["class"], float(d["confidence"]), list(d["bbox"]))
            for d in data.get("detections", [])
        ]
        return cls(
            image_id=data.get("image_id", ""),
            detections=dets,
            image_width=data.get("image_width", 0),
            image_height=data.get("image_height", 0),
        )


@dataclass
class RoutedQuestion:
    intent: Intent
    target_classes: list = field(default_factory=list)  # known classes mentioned, in question order
    raw_question: str = ""
    unsupported_hints: list = field(default_factory=list)  # recognized but out-of-taxonomy nouns


@dataclass
class ReasoningResult:
    """Structured output of the reasoning layer, before phrasing."""

    answer_state: str  # "ANSWERED" | "INSUFFICIENT_INFO"
    reason: str = ""  # populated when INSUFFICIENT_INFO
    data: dict = field(default_factory=dict)  # intent-specific payload for the formatter
