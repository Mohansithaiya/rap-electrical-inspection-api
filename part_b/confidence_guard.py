"""Confidence-based evidence guardrail.

Keeps raw and confidence-filtered detections separate so callers can tell
"not detected" apart from "detected only below the confidence threshold".
"""

from schema import AMBIGUITY_BAND_HIGH, CONF_THRESH, DetectionResult


class Evidence:
    PRESENT = "PRESENT"                # in filtered set -> confident
    LOW_CONFIDENCE_ONLY = "LOW_CONFIDENCE_ONLY"  # in raw set, not filtered
    ABSENT = "ABSENT"                  # in neither set


def filter_detections(raw_detections: list, conf_thresh: float = CONF_THRESH) -> list:
    """Return only detections meeting the confidence threshold."""
    return [d for d in raw_detections if d.confidence >= conf_thresh]


def evidence_for_class(raw_detections: list, filtered_detections: list, cls: str):
    """Return (Evidence state, matching filtered dets, matching raw dets)."""
    raw_matches = [d for d in raw_detections if d.cls == cls]
    filtered_matches = [d for d in filtered_detections if d.cls == cls]

    if filtered_matches:
        return Evidence.PRESENT, filtered_matches, raw_matches
    if raw_matches:
        return Evidence.LOW_CONFIDENCE_ONLY, filtered_matches, raw_matches
    return Evidence.ABSENT, filtered_matches, raw_matches


def split(detection_result: DetectionResult, conf_thresh: float = CONF_THRESH):
    """Convenience: return (raw_detections, filtered_detections) lists."""
    raw = detection_result.detections
    filtered = filter_detections(raw, conf_thresh)
    return raw, filtered


def is_ambiguous_confidence(confidence: float, low: float = CONF_THRESH,
                             high: float = AMBIGUITY_BAND_HIGH) -> bool:
    """True for confidences that clear the filter but are still borderline

    (i.e. in [CONF_THRESH, AMBIGUITY_BAND_HIGH)) - these should still be
    answered, but hedged in the phrased response rather than stated flatly.
    """
    return low <= confidence < high
