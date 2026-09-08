

from ngec.attribute_model import AttributeModelInput, AttributeModelOutput



def test_attribute_model_minimal_input(attribute_model):
    am = attribute_model

    input = [
        AttributeModelInput(
            event_text="A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.",
            event_type="PROTEST"
        )
    ]

    # process() explodes multi-event records and drops empty ones, so it returns
    # a NEW list rather than mutating the input in place. This document yields a
    # single event, whose 'attributes' is a single dict.
    output = am.process(input)

    assert len(output) == 1
    attributes = dict(output[0])["attributes"]

    # Asserted as shape plus a few unambiguous spans, not as one exact dict.
    # The fixture seeds the sampler, so this is stable run to run, but the exact
    # span the model picks still shifts across machines and torch versions --
    # 'A group of Hindu nationalists' vs 'Hindu nationalists' -- and an exact
    # dict here just gets hand-edited to whatever came out that day. Span
    # quality is the eval harness's job; this test covers the plumbing.
    assert set(attributes) == {"event_type", "anchor_quote", "actor",
                               "recipient", "date", "location"}
    assert attributes["event_type"] == "PROTEST"
    # Every span is supposed to be copied verbatim from the document.
    assert attributes["anchor_quote"] in input[0]["event_text"]
    for key in ("actor", "recipient", "date", "location"):
        # The semicolon-separated string the model emits has been split.
        assert isinstance(attributes[key], list)
        assert attributes[key]
        assert all(isinstance(span, str) and span for span in attributes[key])
    assert attributes["date"] == ["last week"]
    assert attributes["location"] == ["Dehli"]
    assert "Hindu nationalists" in attributes["actor"][0]
