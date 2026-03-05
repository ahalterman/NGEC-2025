
import pytest

from ngec.plover_coder import PloverCoder


def test_plover_coder(es_client_local):
    # TODO: adjust this once event and context coders are in the plover coder
    story_list = [
        {"id": "story1", "event_text": "Protesters were in the streets in Paris again today to protest against the government's austerity measures.", "pub_date": "2016-05-01"}
    ]

    pc = PloverCoder(es_client=es_client_local)
    
    event_list = pc.process(story_list)
    
    assert event_list is not None