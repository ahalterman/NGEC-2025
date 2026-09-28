import logging
from importlib.util import find_spec

import pytest

import ngec.plover_coder
from ngec.plover_coder import PloverCoder


def test_plover_coder(es_client_local):
    # TODO: adjust this once event and context coders are in the plover coder
    story_list = [
        {"id": "story1", "event_text": "Protesters were in the streets in Paris again today to protest against the government's austerity measures.", "pub_date": "2016-05-01"}
    ]

    # A CPU backend, named rather than left to "auto": on a machine with vllm
    # and a GPU, "auto" would start vLLM, which reserves most of a (possibly
    # shared) GPU for the length of the test run.
    backend = "llamacpp" if find_spec("llama_cpp") else "transformers"
    pc = PloverCoder(es_client=es_client_local, attribute_backend=backend)
    
    event_list = pc.process(story_list)
    
    assert event_list is not None


# The tests below check that PloverCoder's customization arguments reach the
# component that uses them. The components are replaced with stand-ins that
# only record the keyword arguments they were built with, so no model, spaCy
# pipeline or Elasticsearch is needed.

def _stand_in(name, built):
    """A class that records its keyword arguments in `built[name]`."""
    class StandIn:
        def __init__(self, *args, **kwargs):
            built[name] = kwargs
    return StandIn


@pytest.fixture
def built(monkeypatch):
    built = {}
    for name in ["PloverSklearnClassifier", "GeolocationModel", "AttributeModel",
                 "ActorResolver", "Formatter"]:
        monkeypatch.setattr(ngec.plover_coder, name, _stand_in(name, built))
    monkeypatch.setattr(ngec.plover_coder, "load_nlp", lambda: "nlp")
    return built


def test_defaults_are_unchanged(built):
    PloverCoder(es_client="es")
    assert built["PloverSklearnClassifier"] == {"threshold": None}
    assert built["AttributeModel"]["event_definitions_file"] is None
    assert built["AttributeModel"]["model_name"] is None
    assert built["AttributeModel"]["backend"] == "auto"
    assert built["ActorResolver"]["agents_file"] is None
    assert built["ActorResolver"]["priorities_file"] is None


def test_customization_arguments_reach_components(built):
    pc = PloverCoder(es_client="es",
                     attribute_model_name="my/model",
                     event_definitions_file="my_codebook.csv",
                     agents_file="my_agents.txt",
                     priorities_file="my_priorities.csv")
    assert built["AttributeModel"]["model_name"] == "my/model"
    assert built["AttributeModel"]["event_definitions_file"] == "my_codebook.csv"
    assert built["ActorResolver"]["agents_file"] == "my_agents.txt"
    assert built["ActorResolver"]["priorities_file"] == "my_priorities.csv"
    assert built["ActorResolver"]["es_client"] == "es"
    assert pc.actor_resolution_model is not None


def test_custom_event_classifier_is_used(built):
    class MyClassifier:
        def process(self, story_list):
            for story in story_list:
                story["event_type"] = ["PROTEST"]
                story["event_type_confidence"] = {"PROTEST": 1.0}
                story["event_mode"] = []
            return story_list

    mine = MyClassifier()
    pc = PloverCoder(es_client="es", event_classifier=mine)
    assert pc.event_model is mine
    # The default classifier is not built at all.
    assert "PloverSklearnClassifier" not in built


def test_event_threshold_with_custom_classifier_raises(built):
    with pytest.raises(ValueError, match="event_threshold"):
        PloverCoder(es_client="es", event_classifier=object(), event_threshold=0.5)


def test_agents_file_without_priorities_warns(built, caplog):
    with caplog.at_level(logging.WARNING, logger="ngec.plover_coder"):
        PloverCoder(es_client="es", agents_file="my_agents.txt")
    assert "priorities_file" in caplog.text
