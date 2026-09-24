"""JSON schema and response parsing for the attribute model's output.

The Attributes TypedDict (attribute_model.py) is the canonical output shape,
but ATTRIBUTE_SCHEMA is hand-written rather than derived from it: importing
Attributes here would create a circular import, since attribute_model.py
imports parse_response from this module. Keep the two in sync by hand.

Note the shape difference: Attributes declares actor/recipient/date/location
as list[str] -- that is the shape process() returns. The legacy and v5 models
emit semicolon-joined strings (see the OUTPUT FORMAT block in
_make_system_content_short / _make_system_content_v5), so ATTRIBUTE_SCHEMA
describes strings, and normalize_spans() splits them. The v6 model already
emits JSON lists, which normalize_spans() keeps as they are: a semicolon inside
a v6 list element is part of the span, not a separator. No schema is sent for
v6 at all; the model was evaluated unconstrained.
"""
import json
import re

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Every attribute whose value is a list of spans. The last three come only from
# v6 models (killed and injured for ASSAULT, PROTEST and COERCE).
SPAN_KEYS = ("actor", "recipient", "date", "location", "killed", "injured",
             "reporter")


def normalize_spans(event: dict) -> dict:
    """Make every span attribute a list of stripped strings, in place.

    Accepts both output shapes: a semicolon-joined string from the legacy and
    v5 models ("the police; protesters") is split, and a JSON list from the v6
    model (["the police", "protesters"]) is kept, element by element. Empty
    strings and nulls inside a list are dropped, so an empty role is always [].
    """
    for key in SPAN_KEYS:
        value = event.get(key)
        if isinstance(value, str):
            event[key] = [v.strip() for v in value.split(";") if v.strip()]
        elif isinstance(value, list):
            event[key] = [str(v).strip() for v in value
                          if v is not None and str(v).strip()]
    return event

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


def _salvage_truncated_array(text: str) -> list | None:
    """The complete records at the start of a JSON array that was cut off.

    A response that hits the output-token limit ends mid-record. Every record
    before the cut is intact, so they are decoded one at a time and the broken
    tail is dropped. None if there is no array or not even one whole record.
    """
    start = text.find("[")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    records = []
    i = start + 1
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        try:
            record, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        records.append(record)
    return records or None


def _drop_duplicate_records(events: list[dict]) -> list[dict]:
    """Remove records identical to an earlier one, keeping the first.

    Under greedy decoding the v6 model occasionally repeats the same few records
    until it runs out of tokens (3 of the 500 gold500_a documents). Two
    identical records describe the same event, so only the first is kept.
    """
    seen, kept = set(), []
    for event in events:
        key = json.dumps(event, sort_keys=True)
        if key not in seen:
            seen.add(key)
            kept.append(event)
    return kept


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
    """clean -> parse -> coerce shape -> normalize spans into lists.

    Returns (events, failure_reason). On failure, events is [] (the caller can
    append it directly to a per-record results list) -- except for "truncated",
    a response cut off mid-array, where the records completed before the cut
    are returned along with the reason. Exact duplicate records are dropped. A bare dict
    is wrapped into a one-element list rather than treated as a failure; a
    scalar or a list containing non-dict items is a failure.
    """
    cleaned = clean_response(raw)
    failure = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Most often a response cut off at the token limit: keep the records
        # that were finished, and report it.
        parsed = _salvage_truncated_array(cleaned)
        if parsed is None:
            return [], "json_decode_error"
        failure = "truncated"

    if isinstance(parsed, dict):
        events = [parsed]
    elif isinstance(parsed, list):
        if parsed and not all(isinstance(item, dict) for item in parsed):
            return [], "not_a_dict_list"
        events = parsed
    else:
        return [], "scalar"

    for event in events:
        normalize_spans(event)

    return _drop_duplicate_records(events), failure