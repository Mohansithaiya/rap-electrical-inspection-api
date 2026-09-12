"""Part B orchestrator: question + detections -> plain-language answer.

question -> intent_router.route -> confidence_guard.split -> reasoning.reason
-> answer_formatter.format_answer
"""

from schema import DetectionResult
from intent_router import route
from confidence_guard import split
from reasoning import reason
from answer_formatter import format_answer


def answer_question(question: str, detection_result: DetectionResult) -> dict:
    """Run the full Part B pipeline for one (question, image detections) pair.

    Returns a dict suitable for a FastAPI JSON response.
    """
    routed = route(question)
    raw, filtered = split(detection_result)
    result = reason(routed, raw, filtered)
    text = format_answer(routed, result)

    return {
        "question": question,
        "intent": routed.intent.value,
        "answer_state": result.answer_state,
        "answer": text,
    }
