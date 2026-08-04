

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

    expected = {'event_type': 'PROTEST',
                'anchor_quote': 'A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.',
                'actor': ['A group of Hindu nationalists'],
                'recipient': ['Muslim shops'],
                'date': ['last week'],
                'location': ['Dehli']}

    assert len(output) == 1
    assert dict(output[0])["attributes"] == expected
