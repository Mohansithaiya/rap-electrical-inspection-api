"""Hand-written keyword/regex intent router. No ML, no agentic framework."""

import re

from schema import Intent, RoutedQuestion, UNSUPPORTED_CLASS_HINTS

# Word-boundary-safe patterns per known class. Order in the dict doesn't
# determine extraction order — match position in the question does (see
# _extract_target_classes). disconnector's pattern excludes "open
# disconnector"/"opened disconnector" so that phrase is attributed only to
# disconnector_open, not double-counted as plain disconnector too.
_CLASS_PATTERNS = {
    "disconnector_open": [r"\bdisconnector_open\b", r"\bopen(?:ed)? disconnectors?\b"],
    "disconnector": [r"(?<!open )(?<!opened )\bdisconnectors?\b"],
    "switchgear": [r"\bswitchgears?\b"],
    "switch": [r"\bswitch(?:es)?\b"],
    "transformer": [r"\btransformers?\b"],
    "insulator": [r"\binsulators?\b"],
    "arc": [r"\barc(?:s|ing)?\b"],
    "spark": [r"\bspark(?:s|ing)?\b"],
}

_INTENT_KEYWORDS = {
    Intent.COUNT: [r"\bhow many\b", r"\bcount\b", r"\bnumber of\b"],
    Intent.RISK_ASSESSMENT: [
        r"\bsafe\b", r"\bsafety\b", r"\bhazards?\b", r"\bhazardous\b", r"\brisk\b",
        r"\bdanger(?:ous)?\b", r"\bunsafe\b",
    ],
    Intent.LOCATION: [r"\bwhere\b", r"\blocated\b", r"\blocation\b", r"\bposition\b"],
    Intent.PRESENCE: [
        r"\bis there\b", r"\bare there\b", r"\bdo(?:es)? .* (?:have|contain)\b",
        r"\bpresent\b", r"\bvisible\b", r"\bany\b",
    ],
}

# Checked as plain substrings (order-independent phrasing), not a single
# backtracking regex - avoids missing short natural phrasings like
# "what's in this image".
_LIST_ALL_PHRASES = [
    "what's in", "whats in", "what is in", "what are in",
    "what objects", "what items", "what things",
    "list the objects", "list objects", "list all",
]

_OUT_OF_SCOPE_KEYWORDS = [
    r"\bcolor\b", r"\bcolour\b", r"\bbrand\b", r"\bmanufacturer\b",
    r"\bweather\b", r"\btime of day\b", r"\bwhen\b", r"\bwho\b",
    r"\bprice\b", r"\bcost\b", r"\bmodel number\b",
]


def _extract_target_classes(q: str) -> list:
    """Known classes mentioned in the question, ordered by first appearance."""
    matches = []
    for cls_name, patterns in _CLASS_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, q)
            if m:
                matches.append((m.start(), cls_name))
                break
    matches.sort(key=lambda x: x[0])

    ordered = []
    for _, cls_name in matches:
        if cls_name not in ordered:
            ordered.append(cls_name)
    return ordered


def _extract_unsupported_hints(q: str) -> list:
    hints = []
    for hint in UNSUPPORTED_CLASS_HINTS:
        if re.search(r"\b" + re.escape(hint) + r"\b", q):
            hints.append(hint)
    return hints


def route(question: str) -> RoutedQuestion:
    """Classify a natural-language question into an Intent + target classes.

    Deterministic rule order:
      1. out-of-scope keyword match -> OUT_OF_SCOPE (short-circuits, no detection call)
      2. known-class extraction (word-boundary-safe, all matches kept, in question order)
      3. if no known class was found, check for recognized-but-unsupported nouns
      4. intent keyword match, in fixed priority: COUNT, RISK_ASSESSMENT, LOCATION,
         LIST_ALL, PRESENCE (risk/count/location checked ahead of the generic
         PRESENCE catch-alls so e.g. "hazardous"/"where is the danger" route correctly)
      5. nothing matched -> OUT_OF_SCOPE (unsupported question shape)
    """
    q = question.lower().strip()

    for pattern in _OUT_OF_SCOPE_KEYWORDS:
        if re.search(pattern, q):
            return RoutedQuestion(Intent.OUT_OF_SCOPE, [], question, [])

    target_classes = _extract_target_classes(q)
    unsupported_hints = [] if target_classes else _extract_unsupported_hints(q)

    for pattern in _INTENT_KEYWORDS[Intent.COUNT]:
        if re.search(pattern, q):
            return RoutedQuestion(Intent.COUNT, target_classes, question, unsupported_hints)

    for pattern in _INTENT_KEYWORDS[Intent.RISK_ASSESSMENT]:
        if re.search(pattern, q):
            return RoutedQuestion(Intent.RISK_ASSESSMENT, target_classes, question, unsupported_hints)

    for pattern in _INTENT_KEYWORDS[Intent.LOCATION]:
        if re.search(pattern, q):
            return RoutedQuestion(Intent.LOCATION, target_classes, question, unsupported_hints)

    if any(phrase in q for phrase in _LIST_ALL_PHRASES):
        return RoutedQuestion(Intent.LIST_ALL, target_classes, question, unsupported_hints)

    for pattern in _INTENT_KEYWORDS[Intent.PRESENCE]:
        if re.search(pattern, q):
            return RoutedQuestion(Intent.PRESENCE, target_classes, question, unsupported_hints)

    return RoutedQuestion(Intent.OUT_OF_SCOPE, target_classes, question, unsupported_hints)
