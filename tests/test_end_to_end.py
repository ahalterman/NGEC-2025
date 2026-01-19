"""
Test the full pipeline from story to output events

Requires ES with wiki and geonames data
"""

import pytest

from ngec import EventClass
from ngec import AttributeModel
from ngec import ActorResolver
from ngec import GeolocationModel
from ngec import Formatter
from ngec import utilities
from ngec import load_nlp, setup_logging


def test_end_to_end_with_one_story(es_client_local):
    setup_logging()

    nlp = load_nlp()    

    # Instantiate components
    # TODO: doesn't work, Andy has not synced
    #event_model = EventClass()
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)
    attribute_model = AttributeModel(silent=True, gpu=False, backend="transformers")
    actor_resolution_model = ActorResolver(spacy_model=nlp, es_client=es_client_local)
    formatter = Formatter()

    # Prepare input data
    story_list = [
        {"id": "story1", "event_text": "President Macron and Chancellor Angel Merkel met in Brussels today to discuss EU debt relief plans", "event_type": ["CONSULT"], "pub_date": "2016-05-01"}
    ]
    
    just_text = [i['event_text'] for i in story_list]
    doc_list = [doc for doc in nlp.pipe(just_text)]

    # Pipeline
    # TODO event class not working
    #story_list = event_model.process(story_list)
    story_list = geolocation_model.process(story_list, doc_list)
    event_list = utilities.stories_to_events(story_list, doc_list)
    event_list = attribute_model.process(event_list)
    event_list = actor_resolution_model.process(event_list)
    cleaned_events = formatter.process(event_list, return_raw=True)
    # TODO these aren't really cleaned up events yet, big mess of all outputs
    assert cleaned_events is not None


# Just want to make sure it doesn't accidentally only work with a single member
# input list or single event story
def test_end_to_end_with_two_stories(es_client_local):
    pytest.skip("Haven't finished implementation")
    setup_logging()

    nlp = load_nlp()    

    # Instantiate components
    # TODO: doesn't work, Andy has not synced
    #event_model = EventClass()
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)
    attribute_model = AttributeModel(silent=True, gpu=False, backend="transformers")
    actor_resolution_model = ActorResolver(spacy_model=nlp, es_client=es_client_local)
    formatter = Formatter()

    # Prepare input data
    story_list = [
        {"id": "story1", "event_text": "President Macron and Chancellor Angel Merkel met in Brussels today to discuss EU debt relief plans", "event_type": ["CONSULT"], "pub_date": "2016-05-01"},
        {"id": "story2", "event_text": "Georgian and South Ossetian forces clashed on several border villages on Thursday, with gunfire reported. Georgian president Mikheil Saakashvili reportedly spoke via phone to the head of the Russian peacekeeping force in South Ossetia in order to calm tensions.", "event_type": ["ASSAULT", "CONSULT"], "pub_date": "2007-07-01"},
    ]
    
    just_text = [i['event_text'] for i in story_list]
    doc_list = [doc for doc in nlp.pipe(just_text)]

    # Pipeline
    # TODO event class not working
    #story_list = event_model.process(story_list)
    story_list = geolocation_model.process(story_list, doc_list)
    event_list = utilities.stories_to_events(story_list, doc_list)
    event_list = attribute_model.process(event_list)
    event_list = actor_resolution_model.process(event_list)
    cleaned_events = formatter.process(event_list)
