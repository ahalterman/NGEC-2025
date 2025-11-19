
import logging

import ngec
from ngec.attribute_model import EventDict
from ngec.logging import setup_logging

setup_logging(level=logging.WARNING, quiet_third_party=True)


def smoke_test_attribute_model():
    am = ngec.AttributeModel(silent=True, gpu=False, backend="transformers")

    input = [
        EventDict(
            event_text="A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.",
            event_type="PROTEST",
            event_mode="riot"
        )
    ]

    _ = am.process(input)
