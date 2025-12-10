
import logging

import ngec
from ngec.attribute_model import AttributeModelInput, AttributeModelOutput



def test_attribute_model_minimal_input(attribute_model):
    am = attribute_model

    input = [
        AttributeModelInput(
            event_text="A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.",
            event_type="PROTEST"
        )
    ]

    _ = am.process(input)

    expected = dict(input[0]) 
    expected['attributes'] = [{'event_type': 'PROTEST', 
                               'anchor_quote': 'A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.', 
                               'actor': ['A group of Hindu nationalists'], 
                               'recipient': ['Muslim shops'], 
                               'date': ['last week'], 
                               'location': ['Dehli']}]

    assert input[0] == expected
