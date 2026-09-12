"""Unit tests for the Part B reasoning layer, using mock detections only."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema import Detection, DetectionResult  # noqa: E402
from pipeline import answer_question  # noqa: E402
from intent_router import route  # noqa: E402
from schema import Intent  # noqa: E402


def _detection_result(dets):
    return DetectionResult(image_id="img1", detections=dets, image_width=640, image_height=480)


def test_confident_presence():
    dets = [Detection("insulator", 0.91, [10, 10, 50, 50])]
    result = answer_question("Is there an insulator in this image?", _detection_result(dets))
    assert result["answer_state"] == "ANSWERED"
    assert "insulator" in result["answer"].lower()
    assert "yes" in result["answer"].lower()


def test_low_confidence_only_presence():
    dets = [Detection("switchgear", 0.32, [10, 10, 50, 50])]
    result = answer_question("Is there a switchgear present?", _detection_result(dets))
    assert result["answer_state"] == "INSUFFICIENT_INFO"
    assert "low confidence" in result["answer"].lower()


def test_absent_class():
    dets = [Detection("transformer", 0.88, [10, 10, 50, 50])]
    result = answer_question("Is there a spark visible?", _detection_result(dets))
    assert result["answer_state"] == "INSUFFICIENT_INFO"
    assert "not detected" in result["answer"].lower()


def test_arc_spark_risk_unsafe():
    dets = [
        Detection("arc", 0.77, [0, 0, 20, 20]),
        Detection("transformer", 0.9, [30, 30, 60, 60]),
    ]
    result = answer_question("Is this scene safe?", _detection_result(dets))
    assert result["answer_state"] == "ANSWERED"
    assert "unsafe" in result["answer"].lower()
    assert "arc" in result["answer"].lower()


def test_no_detections_insufficient_info():
    result = answer_question("Is this scene safe?", _detection_result([]))
    assert result["answer_state"] == "INSUFFICIENT_INFO"
    assert "insufficient information" in result["answer"].lower()


def test_out_of_scope_question():
    dets = [Detection("insulator", 0.9, [10, 10, 50, 50])]
    result = answer_question("What color is the insulator?", _detection_result(dets))
    assert result["intent"] == "OUT_OF_SCOPE"
    assert result["answer_state"] == "INSUFFICIENT_INFO"


def test_count_with_low_confidence_extra():
    dets = [
        Detection("disconnector", 0.8, [0, 0, 10, 10]),
        Detection("disconnector", 0.9, [10, 10, 20, 20]),
        Detection("disconnector", 0.3, [20, 20, 30, 30]),
    ]
    result = answer_question("How many disconnectors are visible?", _detection_result(dets))
    assert result["answer_state"] == "ANSWERED"
    assert "2 instance" in result["answer"]
    assert "+1 more" in result["answer"]


def test_word_boundary_safe_class_extraction():
    # "arc" must not be extracted from substrings like "search"/"march".
    routed = route("Please search the image for defects")
    assert "arc" not in routed.target_classes

    # "switch" must not be extracted from "switchgear".
    routed = route("Is there a switchgear in the image?")
    assert routed.target_classes == ["switchgear"]


def test_multi_class_presence_question():
    dets = [Detection("arc", 0.9, [0, 0, 10, 10])]
    result = answer_question("Is there a spark or an arc?", _detection_result(dets))
    assert result["answer_state"] == "ANSWERED"
    assert "arc" in result["answer"].lower()
    assert "spark" in result["answer"].lower()
    assert "not detected" in result["answer"].lower()  # spark absent, reported per-class


def test_unsupported_class_question():
    dets = [Detection("insulator", 0.9, [10, 10, 50, 50])]
    result = answer_question("Is there a person in this image?", _detection_result(dets))
    assert result["answer_state"] == "INSUFFICIENT_INFO"
    assert "not one of the classes" in result["answer"].lower()


def test_risk_low_confidence_informational_only_vs_true_no_detections():
    # Only a sub-threshold, non-hazard detection: must NOT claim "no objects detected".
    dets = [Detection("insulator", 0.2, [0, 0, 10, 10])]
    result = answer_question("Is this scene safe?", _detection_result(dets))
    assert result["answer_state"] == "INSUFFICIENT_INFO"
    assert "no objects were detected" not in result["answer"].lower()
    assert "below the" in result["answer"].lower()

    # Truly empty detections: the original "no objects detected" message applies.
    result_empty = answer_question("Is this scene safe?", _detection_result([]))
    assert "no objects were detected" in result_empty["answer"].lower()


def test_list_all_natural_phrasing():
    dets = [Detection("transformer", 0.9, [0, 0, 10, 10])]
    routed = route("What's in this image?")
    assert routed.intent == Intent.LIST_ALL


def test_hazardous_and_risk_vs_location_priority():
    # "hazardous" (not just "hazard") must route to RISK_ASSESSMENT.
    routed = route("Is the switchgear hazardous?")
    assert routed.intent == Intent.RISK_ASSESSMENT

    # RISK_ASSESSMENT keywords must win over the generic "where" LOCATION match.
    routed = route("Where is the danger in this image?")
    assert routed.intent == Intent.RISK_ASSESSMENT


def test_ambiguity_band_hedge():
    # Confidence just above threshold (0.5-0.6) should still answer, but hedge.
    dets = [Detection("insulator", 0.55, [0, 0, 10, 10])]
    result = answer_question("Is there an insulator?", _detection_result(dets))
    assert result["answer_state"] == "ANSWERED"
    assert "verify manually" in result["answer"].lower()

    # Confidence well above the band should not be hedged.
    dets_high = [Detection("insulator", 0.95, [0, 0, 10, 10])]
    result_high = answer_question("Is there an insulator?", _detection_result(dets_high))
    assert "verify manually" not in result_high["answer"].lower()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
