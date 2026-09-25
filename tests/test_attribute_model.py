import json
import logging
from pathlib import Path

import pytest

from ngec.attribute_model import (AttributeModel, AttributeModelInput,
                                  AttributeModelOutput, DEFAULT_MODEL,
                                  resolve_prompt_format, _load_v6_definitions)

V6_PROMPTS = Path(__file__).parent / "data" / "v6_prompts.json"


def _v6_prompt_builder():
    """An AttributeModel that can build v6 prompts, without loading a model.

    Prompt building needs only the format and the definitions, so this skips
    __init__ (which would load weights) and sets those two by hand.
    """
    am = AttributeModel.__new__(AttributeModel)
    am.prompt_format = "v6"
    am.v6_definitions = _load_v6_definitions()
    return am


def test_default_model_is_v6():
    assert resolve_prompt_format(DEFAULT_MODEL) == "v6"


def test_v6_prompt_matches_training_prompt():
    """The prompt NGEC sends is byte for byte the one the model was trained on.

    tests/data/v6_prompts.json was rendered by `student.prompting.render_messages`
    in train_NGEC_2026 from training examples: twelve event types, including
    the three whose prompt adds `killed` and `injured`, and five sub-events.
    """
    am = _v6_prompt_builder()
    for case in json.loads(V6_PROMPTS.read_text()):
        record = {"event_text": case["event_text"], "event_type": case["event_type"],
                  "event_mode": case["event_mode"]}
        system, user = am._build_conversation(record)
        assert system["content"] == case["system"], case["event_type"]
        assert user["content"] == case["user"], (case["event_type"], case["event_mode"])


def test_v6_definitions_cover_every_classifier_mode():
    """Every (event type, mode) the event classifier can emit has a trained definition."""
    from importlib import resources

    definitions = _load_v6_definitions()
    with resources.files("ngec").joinpath(
            "assets", "event_models_v2", "metadata.json").open() as f:
        metadata = json.load(f)
    for key in metadata["mode_metrics"]:
        event_type, mode = key.split("-", 1)
        assert (event_type, mode) in definitions, key
        assert (event_type, "") in definitions, event_type


def test_v6_custom_definition_is_used_as_given():
    am = _v6_prompt_builder()
    record = {"event_text": "Hackers took down the ministry's website.",
              "event_type": "CYBERATTACK", "event_mode": "",
              "event_def": "An attack on computer systems or networks."}
    system, user = am._build_conversation(record)
    assert "## Event: **CYBERATTACK**: An attack on computer systems or networks." in user["content"]
    assert '"killed"' not in system["content"]


def test_v6_unknown_type_without_definition_raises():
    am = _v6_prompt_builder()
    with pytest.raises(KeyError, match="event_def"):
        am._build_conversation({"event_text": "x", "event_type": "CYBERATTACK"})


def test_prompt_format_from_ngec_json(tmp_path):
    """A renamed local copy is still prompted in the format it declares."""
    model_dir = tmp_path / "my-renamed-model"
    model_dir.mkdir()
    (model_dir / "ngec.json").write_text('{"prompt_format": "v6"}')
    assert resolve_prompt_format(str(model_dir)) == "v6"


def test_prompt_format_by_name_and_fallback(tmp_path):
    assert resolve_prompt_format("ahalt/qwen3-event-extraction-exp5.1") == "v5"
    assert resolve_prompt_format("/models/qwen3-event-extraction-exp5.1/") == "v5"
    assert resolve_prompt_format("ahalt/event-attribute-extractor") == "legacy"
    assert resolve_prompt_format(str(tmp_path)) == "legacy"


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
    # Decoding is greedy, so this is stable run to run, but the exact span can
    # still shift across machines and torch versions. Span quality is the eval
    # harness's job; this test covers the plumbing.
    #
    # The default (v6) model adds `mode`, and for PROTEST `killed` and `injured`.
    assert set(attributes) == {"event_type", "mode", "anchor_quote", "actor",
                               "recipient", "date", "location", "killed", "injured"}
    assert attributes["event_type"] == "PROTEST"
    # Every span is supposed to be copied verbatim from the document.
    assert attributes["anchor_quote"] in input[0]["event_text"]
    for key in ("actor", "recipient", "date", "location", "killed", "injured"):
        # Always a list of strings; an empty role is [].
        assert isinstance(attributes[key], list)
        assert all(isinstance(span, str) and span for span in attributes[key])
    for key in ("actor", "date", "location"):
        assert attributes[key]
        assert all(span in input[0]["event_text"] for span in attributes[key])
    assert "Hindu nationalists" in attributes["actor"][0]
    assert "last week" in attributes["date"][0]
    assert "Dehli" in attributes["location"][0]


def test_v6_warns_that_definitions_file_is_unused(monkeypatch, caplog):
    # The v6 format reads the trained definitions, not a codebook CSV, so a
    # definitions file passed to it would otherwise be silently ignored. The
    # model itself is not loaded: the engine is replaced by a stand-in.
    import ngec.llm.transformers

    class StandInEngine:
        def __init__(self, **kwargs):
            self.tokenizer = None
    monkeypatch.setattr(ngec.llm.transformers, "TransformersEngine", StandInEngine)
    monkeypatch.delenv("NGEC_ATTRIBUTE_MODEL", raising=False)

    with caplog.at_level(logging.WARNING, logger="ngec.attribute_model"):
        am = AttributeModel(backend="transformers", silent=True,
                            event_definitions_file="PLOVER_structured_codebook_updated.csv")
    assert am.prompt_format == "v6"
    assert "is not used" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ngec.attribute_model"):
        AttributeModel(backend="transformers", silent=True)
    assert "is not used" not in caplog.text
