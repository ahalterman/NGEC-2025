"""JSON schema and response parsing for the attribute model's output.

The Attributes TypedDict (attribute_model.py) is the canonical output shape,
but ATTRIBUTE_SCHEMA is hand-written rather than derived from it: importing
Attributes here would create a circular import, since attribute_model.py
imports parse_response from this module. Keep the two in sync by hand.

Note the shape difference: Attributes declares actor/recipient/date/location
as list[str] -- that is the *post-split* shape process() returns. The model
itself emits semicolon-joined strings (see the OUTPUT FORMAT block in
_make_system_content_short / _make_system_content_v5), so ATTRIBUTE_SCHEMA
describes strings, not lists. parse_response() does the splitting.
"""
import json
import re

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

ATTRIBUTE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string"},
            "anchor_quote": {"type": "string"},
            "actor": {"type": "string"},
            "recipient": {"type": "string"},
            "date": {"type": "string"},
            "location": {"type": "string"},
        },
        "required": ["event_type", "anchor_quote", "actor", "recipient",
                      "date", "location"],
    },
}


def _extract_first_bracketed_array(text: str) -> str | None:
    """Return the first balanced [...] substring, or None if none closes.

    A plain character scan rather than regex, since matching balanced
    brackets isn't regular -- and this has to skip brackets that appear
    inside string literals (e.g. a quote containing "[sic]").
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def clean_response(text: str) -> str:
    """Best-effort cleanup of raw model output before JSON parsing.

    Strips <think>...</think> blocks and a stray leading ':' (both observed
    failure shapes). If the result still won't parse as JSON, falls back to
    extracting the first balanced [...] block as a salvage attempt.
    """
    text = THINK_TAG_RE.sub("", text).strip()
    if text.startswith(":"):
        text = text[1:].strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    salvaged = _extract_first_bracketed_array(text)
    return salvaged if salvaged is not None else text


def parse_response(raw: str) -> tuple[list[dict], str | None]:
    """clean -> parse -> coerce shape -> split semicolon spans into lists.

    Returns (events, failure_reason). On failure, events is always [] (the
    caller can append it directly to a per-record results list). A bare dict
    is wrapped into a one-element list rather than treated as a failure; a
    scalar or a list containing non-dict items is a failure.
    """
    cleaned = clean_response(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return [], "json_decode_error"

    if isinstance(parsed, dict):
        events = [parsed]
    elif isinstance(parsed, list):
        if parsed and not all(isinstance(item, dict) for item in parsed):
            return [], "not_a_dict_list"
        events = parsed
    else:
        return [], "scalar"

    for event in events:
        for key in ("actor", "recipient", "date", "location"):
            value = event.get(key)
            if isinstance(value, str):
                event[key] = [v.strip() for v in value.split(";")]
            elif isinstance(value, list):
                event[key] = [v.strip() for v in value]

    return events, None