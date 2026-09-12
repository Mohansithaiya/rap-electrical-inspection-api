"""Deterministic per-intent reasoning handlers.

Each handler consumes raw + filtered detections (never re-derives filtering
itself) and returns a ReasoningResult. No LLM, no agentic framework —
plain control flow only.
"""

from schema import (
    CONF_THRESH,
    HIGH_RISK_CLASSES,
    Intent,
    MEDIUM_RISK_CLASSES,
    ReasoningResult,
    RoutedQuestion,
)
from confidence_guard import Evidence, evidence_for_class, is_ambiguous_confidence


def _insufficient(reason: str, **data) -> ReasoningResult:
    return ReasoningResult(answer_state="INSUFFICIENT_INFO", reason=reason, data=data)


def _answered(**data) -> ReasoningResult:
    return ReasoningResult(answer_state="ANSWERED", data=data)


def _unsupported_class_result(hint: str) -> ReasoningResult:
    return _insufficient("unsupported_class", cls=hint)


def _evidence_list(target_classes: list, raw: list, filtered: list) -> list:
    results = []
    for cls in target_classes:
        state, filt_matches, raw_matches = evidence_for_class(raw, filtered, cls)
        results.append({"cls": cls, "state": state, "filtered": filt_matches, "raw": raw_matches})
    return results


def _no_target_class_result(routed: RoutedQuestion):
    """Shared not-found handling for PRESENCE/COUNT/LOCATION when the router
    extracted no known class at all."""
    if routed.unsupported_hints:
        return _unsupported_class_result(routed.unsupported_hints[0])
    return _insufficient("no_class_identified")


def handle_presence(routed: RoutedQuestion, raw: list, filtered: list) -> ReasoningResult:
    if not routed.target_classes:
        return _no_target_class_result(routed)

    per_class = _evidence_list(routed.target_classes, raw, filtered)
    entries = []
    any_present = False

    for pc in per_class:
        cls = pc["cls"]
        if pc["state"] == Evidence.PRESENT:
            any_present = True
            conf = max(d.confidence for d in pc["filtered"])
            entries.append({
                "cls": cls, "status": "present", "count": len(pc["filtered"]),
                "max_confidence": conf, "hedge": is_ambiguous_confidence(conf),
            })
        elif pc["state"] == Evidence.LOW_CONFIDENCE_ONLY:
            entries.append({
                "cls": cls, "status": "low_confidence",
                "max_confidence": max(d.confidence for d in pc["raw"]),
            })
        else:
            entries.append({"cls": cls, "status": "absent"})

    # Single-class-not-present keeps the original, more specific insufficient
    # reasons/templates instead of the generic multi-class one.
    if len(entries) == 1 and entries[0]["status"] != "present":
        e = entries[0]
        if e["status"] == "low_confidence":
            return _insufficient("low_confidence_only", cls=e["cls"], max_confidence=e["max_confidence"])
        return _insufficient("not_detected", cls=e["cls"])

    if not any_present:
        summary = "; ".join(f"{e['cls']}: {e['status']}" for e in entries)
        return _insufficient("multi_class_none_present", summary=summary)

    return _answered(entries=entries)


def handle_count(routed: RoutedQuestion, raw: list, filtered: list) -> ReasoningResult:
    if not routed.target_classes:
        return _no_target_class_result(routed)

    per_class = _evidence_list(routed.target_classes, raw, filtered)
    entries = []
    any_present = False

    for pc in per_class:
        cls = pc["cls"]
        if pc["state"] == Evidence.PRESENT:
            any_present = True
            conf = max(d.confidence for d in pc["filtered"])
            entries.append({
                "cls": cls, "status": "present", "count": len(pc["filtered"]),
                "low_confidence_extra": len(pc["raw"]) - len(pc["filtered"]),
                "hedge": is_ambiguous_confidence(conf),
            })
        elif pc["state"] == Evidence.LOW_CONFIDENCE_ONLY:
            entries.append({
                "cls": cls, "status": "low_confidence",
                "max_confidence": max(d.confidence for d in pc["raw"]),
            })
        else:
            entries.append({"cls": cls, "status": "absent"})

    if len(entries) == 1 and entries[0]["status"] != "present":
        e = entries[0]
        if e["status"] == "low_confidence":
            return _insufficient("low_confidence_only", cls=e["cls"], max_confidence=e["max_confidence"])
        return _insufficient("not_detected", cls=e["cls"])

    if not any_present:
        summary = "; ".join(f"{e['cls']}: {e['status']}" for e in entries)
        return _insufficient("multi_class_none_present", summary=summary)

    return _answered(entries=entries)


def handle_location(routed: RoutedQuestion, raw: list, filtered: list) -> ReasoningResult:
    if not routed.target_classes:
        return _no_target_class_result(routed)

    per_class = _evidence_list(routed.target_classes, raw, filtered)
    entries = []
    any_present = False

    for pc in per_class:
        cls = pc["cls"]
        if pc["state"] == Evidence.PRESENT:
            any_present = True
            conf = max(d.confidence for d in pc["filtered"])
            entries.append({
                "cls": cls, "status": "present", "boxes": [d.bbox for d in pc["filtered"]],
                "hedge": is_ambiguous_confidence(conf),
            })
        elif pc["state"] == Evidence.LOW_CONFIDENCE_ONLY:
            entries.append({
                "cls": cls, "status": "low_confidence",
                "max_confidence": max(d.confidence for d in pc["raw"]),
            })
        else:
            entries.append({"cls": cls, "status": "absent"})

    if len(entries) == 1 and entries[0]["status"] != "present":
        e = entries[0]
        if e["status"] == "low_confidence":
            return _insufficient("low_confidence_only", cls=e["cls"], max_confidence=e["max_confidence"])
        return _insufficient("not_detected", cls=e["cls"])

    if not any_present:
        summary = "; ".join(f"{e['cls']}: {e['status']}" for e in entries)
        return _insufficient("multi_class_none_present", summary=summary)

    return _answered(entries=entries)


def handle_list_all(routed: RoutedQuestion, raw: list, filtered: list) -> ReasoningResult:
    if not filtered and not raw:
        return _insufficient("no_detections")

    by_class = {}
    for d in filtered:
        by_class[d.cls] = by_class.get(d.cls, 0) + 1

    low_conf_classes = sorted({d.cls for d in raw} - set(by_class.keys()))

    if not by_class:
        return _insufficient("low_confidence_only_all", low_confidence_classes=low_conf_classes)

    return _answered(counts=by_class, low_confidence_classes=low_conf_classes)


def handle_risk_assessment(routed: RoutedQuestion, raw: list, filtered: list) -> ReasoningResult:
    if not raw:
        return _insufficient("no_detections")

    high = [d for d in filtered if d.cls in HIGH_RISK_CLASSES]
    if high:
        hedge = any(is_ambiguous_confidence(d.confidence) for d in high)
        return _answered(level="UNSAFE", hazard_classes=sorted({d.cls for d in high}),
                          count=len(high), hedge=hedge)

    medium = [d for d in filtered if d.cls in MEDIUM_RISK_CLASSES]
    if medium:
        hedge = any(is_ambiguous_confidence(d.confidence) for d in medium)
        return _answered(level="CAUTION", hazard_classes=sorted({d.cls for d in medium}),
                          count=len(medium), hedge=hedge)

    # Sub-threshold hazard-class detections: never silently treat as "safe".
    low_conf_hazard = [
        d for d in raw
        if d.confidence < CONF_THRESH and d.cls in (HIGH_RISK_CLASSES | MEDIUM_RISK_CLASSES)
    ]
    if low_conf_hazard:
        return _insufficient(
            "low_confidence_hazard",
            hazard_classes=sorted({d.cls for d in low_conf_hazard}),
            max_confidence=max(d.confidence for d in low_conf_hazard),
        )

    if filtered:
        return _answered(level="NO_IMMEDIATE_HAZARD", hazard_classes=[], count=0)

    # raw is non-empty but every detection is a sub-threshold, non-hazard
    # class: distinct from "no_detections" - something WAS seen, just not
    # confidently and not anything hazardous.
    return _insufficient("low_confidence_informational_only")


_HANDLERS = {
    Intent.PRESENCE: handle_presence,
    Intent.COUNT: handle_count,
    Intent.LOCATION: handle_location,
    Intent.RISK_ASSESSMENT: handle_risk_assessment,
    Intent.LIST_ALL: handle_list_all,
}


def reason(routed: RoutedQuestion, raw: list, filtered: list) -> ReasoningResult:
    if routed.intent == Intent.OUT_OF_SCOPE:
        return _insufficient("out_of_scope")
    return _HANDLERS[routed.intent](routed, raw, filtered)
