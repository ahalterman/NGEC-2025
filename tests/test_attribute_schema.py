"""Unit tests for parsing the attribute model's raw output.

No model is loaded here: every input is a string that a model actually produced
(or a shape it is known to produce), so these run in milliseconds and cover the
failure modes that otherwise only show up as quietly dropped events.
"""
import pytest

from ngec.attributes.schema import (
    _extract_first_bracketed_array,
    clean_response,
    parse_response,
)


def test_well_formed_array_splits_semicolon_spans():
    raw = """[
      {"event_type": "PROTEST",
       "anchor_quote": "A group of Hindu nationalists rioted in Dehli",
       "actor": "a group of Hindu nationalists; the VHP",
       "recipient": "Muslim shops",
       "date": "last week",
       "location": "Dehli"}
    ]"""

    events, failure = parse_response(raw)

    assert failure is None
    assert len(events) == 1
    assert events[0]["actor"] == ["a group of Hindu nationalists", "the VHP"]
    assert events[0]["recipient"] == ["Muslim shops"]
    # Non-span fields are left as the model emitted them.
    assert events[0]["event_type"] == "PROTEST"
    assert events[0]["anchor_quote"].startswith("A group of Hindu nationalists")


def test_bare_dict_is_wrapped_not_a_failure():
    events, failure = parse_response('{"event_type": "ASSAULT", "actor": "Turkish forces"}')

    assert failure is None
    assert events == [{"event_type": "ASSAULT", "actor": ["Turkish forces"]}]


def test_empty_array_is_a_success_with_no_events():
    """The model saying "no events of this type" is a result, not a failure."""
    events, failure = parse_response("[]")

    assert failure is None
    assert events == []


def test_leading_colon_is_stripped():
    events, failure = parse_response(': [{"event_type": "PROTEST", "actor": "students"}]')

    assert failure is None
    assert events == [{"event_type": "PROTEST", "actor": ["students"]}]


def test_think_block_is_stripped():
    raw = ('<think>The document mentions a protest, so the actor is the '
           'students.</think>\n[{"event_type": "PROTEST", "actor": "students"}]')

    events, failure = parse_response(raw)

    assert failure is None
    assert events == [{"event_type": "PROTEST", "actor": ["students"]}]


def test_prose_around_the_array_is_salvaged():
    raw = ('Here are the extracted attributes:\n'
           '[{"event_type": "PROTEST", "actor": "students"}]\n'
           'Let me know if you need anything else.')

    events, failure = parse_response(raw)

    assert failure is None
    assert events == [{"event_type": "PROTEST", "actor": ["students"]}]


def test_brackets_inside_a_quoted_span_do_not_end_the_array():
    """The salvage scan has to skip brackets inside string literals."""
    raw = ('Extracted:\n[{"event_type": "ACCUSE", '
           '"anchor_quote": "the minister [sic] denied the charge", '
           '"actor": "the minister"}]')

    events, failure = parse_response(raw)

    assert failure is None
    assert events[0]["anchor_quote"] == "the minister [sic] denied the charge"


def test_escaped_quote_inside_a_span_does_not_end_the_string():
    raw = r'[{"event_type": "ACCUSE", "anchor_quote": "he said \"no\" twice", "actor": "he"}]'

    events, failure = parse_response(raw)

    assert failure is None
    assert events[0]["anchor_quote"] == 'he said "no" twice'


def test_list_values_are_stripped_but_not_resplit():
    """Some responses come back with lists already; don't mangle them."""
    events, failure = parse_response('[{"event_type": "PROTEST", "actor": [" students ", "workers"]}]')

    assert failure is None
    assert events[0]["actor"] == ["students", "workers"]


@pytest.mark.parametrize("raw, reason", [
    ("", "json_decode_error"),
    ("   ", "json_decode_error"),
    ("actor: the students\nlocation: Dehli", "json_decode_error"),
    # A truncated response -- the max_tokens ceiling being hit mid-object.
    ('[{"event_type": "PROTEST", "actor": "stud', "json_decode_error"),
    ('"PROTEST"', "scalar"),
    ("42", "scalar"),
    ('["PROTEST", "ASSAULT"]', "not_a_dict_list"),
])
def test_failure_shapes_report_a_reason_and_no_events(raw, reason):
    events, failure = parse_response(raw)

    assert events == []
    assert failure == reason


def test_extract_first_bracketed_array_returns_none_when_unclosed():
    assert _extract_first_bracketed_array('[{"actor": "students"') is None
    assert _extract_first_bracketed_array("no brackets at all") is None


def test_clean_response_returns_text_unchanged_when_nothing_salvageable():
    """Failure detail is the parser's job; clean_response passes the blob on."""
    assert clean_response("actor: the students") == "actor: the students"
