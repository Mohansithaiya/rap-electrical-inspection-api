"""Turns a ReasoningResult into a plain-language response string.

Pure string templating — no model calls. Kept separate from reasoning.py so
phrasing can change without touching decision logic.
"""

from schema import CONF_THRESH, Intent, ReasoningResult

_HEDGE_NOTE = " (low-confidence detection, {:.2f}-{:.2f} range — verify manually)"

_INSUFFICIENT_TEMPLATES = {
    "out_of_scope": "I can't answer that from object detection alone.",
    "unsupported_class": "'{cls}' is not one of the classes this model detects.",
    "no_class_identified": "I couldn't identify which object class the question refers to.",
    "not_detected": "Insufficient information: '{cls}' was not detected in this image.",
    "low_confidence_only": (
        "Insufficient information: '{cls}' was detected only at low confidence "
        "(max {max_confidence:.2f}, below the {thresh:.2f} threshold)."
    ),
    "multi_class_none_present": "Insufficient information for the requested classes ({summary}).",
    "no_detections": "Insufficient information: no objects were detected in this image.",
    "low_confidence_only_all": (
        "Insufficient information: objects were detected only at low confidence "
        "({low_confidence_classes})."
    ),
    "low_confidence_hazard": (
        "Insufficient information to assess risk: possible {hazard_classes} detected "
        "at low confidence (max {max_confidence:.2f}) — recommend manual inspection."
    ),
    "low_confidence_informational_only": (
        "Insufficient information to assess risk: objects were detected only below the "
        "{thresh:.2f} confidence threshold — cannot confirm scene safety."
    ),
}

_RISK_TEMPLATES = {
    "UNSAFE": "Unsafe / immediate hazard: detected {hazard_classes} ({count} instance(s)).",
    "CAUTION": "Caution — inspect before proceeding: detected {hazard_classes} ({count} instance(s)).",
    "NO_IMMEDIATE_HAZARD": "No immediate hazard detected among the recognized object classes.",
}


def _hedge_suffix(hedge: bool) -> str:
    return _HEDGE_NOTE.format(CONF_THRESH, 0.6) if hedge else ""


def _format_presence_entry(e: dict) -> str:
    if e["status"] == "present":
        return f"'{e['cls']}': yes, {e['count']} instance(s), max confidence {e['max_confidence']:.2f}{_hedge_suffix(e.get('hedge', False))}"
    if e["status"] == "low_confidence":
        return f"'{e['cls']}': insufficient information (low confidence only, max {e['max_confidence']:.2f})"
    return f"'{e['cls']}': not detected"


def _format_count_entry(e: dict) -> str:
    if e["status"] == "present":
        extra = f" (+{e['low_confidence_extra']} more at low confidence)" if e.get("low_confidence_extra") else ""
        return f"'{e['cls']}': {e['count']}{extra}{_hedge_suffix(e.get('hedge', False))}"
    if e["status"] == "low_confidence":
        return f"'{e['cls']}': insufficient information (low confidence only, max {e['max_confidence']:.2f})"
    return f"'{e['cls']}': 0 (not detected)"


def _format_location_entry(e: dict) -> str:
    if e["status"] == "present":
        boxes = ", ".join(str(b) for b in e["boxes"])
        return f"'{e['cls']}': {boxes}{_hedge_suffix(e.get('hedge', False))}"
    if e["status"] == "low_confidence":
        return f"'{e['cls']}': insufficient information (low confidence only, max {e['max_confidence']:.2f})"
    return f"'{e['cls']}': not detected"


def format_answer(routed, result: ReasoningResult) -> str:
    if result.answer_state == "INSUFFICIENT_INFO":
        template = _INSUFFICIENT_TEMPLATES[result.reason]
        fields = dict(result.data)
        fields.setdefault("thresh", CONF_THRESH)
        return template.format(**fields)

    data = result.data
    intent = routed.intent

    if intent == Intent.PRESENCE:
        entries = data["entries"]
        if len(entries) == 1:
            e = entries[0]
            return (
                f"Yes, '{e['cls']}' is present ({e['count']} instance(s), "
                f"max confidence {e['max_confidence']:.2f})"
                f"{_hedge_suffix(e.get('hedge', False))}."
            )
        return "; ".join(_format_presence_entry(e) for e in entries) + "."

    if intent == Intent.COUNT:
        entries = data["entries"]
        if len(entries) == 1:
            e = entries[0]
            extra = f" (+{e['low_confidence_extra']} more at low confidence)" if e.get("low_confidence_extra") else ""
            return f"{e['count']} instance(s) of '{e['cls']}' detected{extra}{_hedge_suffix(e.get('hedge', False))}."
        return "; ".join(_format_count_entry(e) for e in entries) + "."

    if intent == Intent.LOCATION:
        entries = data["entries"]
        if len(entries) == 1:
            e = entries[0]
            boxes = ", ".join(str(b) for b in e["boxes"])
            return f"'{e['cls']}' located at: {boxes}{_hedge_suffix(e.get('hedge', False))}."
        return "; ".join(_format_location_entry(e) for e in entries) + "."

    if intent == Intent.LIST_ALL:
        counts_str = ", ".join(f"{cls} ({n})" for cls, n in sorted(data["counts"].items()))
        low_conf = data.get("low_confidence_classes") or []
        suffix = f" Additional low-confidence candidates: {', '.join(low_conf)}." if low_conf else ""
        return f"Detected objects: {counts_str}.{suffix}"

    if intent == Intent.RISK_ASSESSMENT:
        text = _RISK_TEMPLATES[data["level"]].format(**data)
        if data.get("hedge"):
            text += _hedge_suffix(True).strip() + "."
        return text

    raise ValueError(f"No formatter for intent {intent}")
