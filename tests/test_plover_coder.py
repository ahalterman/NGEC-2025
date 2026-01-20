
import pytest

from ngec.plover_coder import PloverCoder


def test_plover_coder(es_client_local):
    # TODO: adjust this once event and context coders are in the plover coder
    story_list = [
        {"id": "story1", 
         "event_text": "President Macron and Chancellor Angel Merkel met in Brussels today to discuss EU debt relief plans", 
         "event_type": ["CONSULT"], 
         "pub_date": "2016-05-01"}
    ]

    pc = PloverCoder(es_client=es_client_local)
    event_list = pc.process(story_list)
    assert event_list is not None