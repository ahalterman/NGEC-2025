"""`ngec download-models`: which models it asks for, with nothing downloaded.

The downloaders (spaCy's, huggingface_hub's, sentence-transformers') are
replaced with fakes that record what they were asked for, so these tests check
the choices this code makes -- which model names, which flags -- and not the
downloads themselves.
"""

import importlib.util

import huggingface_hub
import pytest
import sentence_transformers

import ngec.actors.common
import ngec.cli
import ngec.models
from ngec.actors.common import DEFAULT_ENCODER
from ngec.attribute_model import DEFAULT_MODEL
from ngec.models import (REQUIRED_SPACY_MODELS, classifier_encoder_name,
                         download_attribute_model, download_encoders,
                         download_spacy_models)


@pytest.fixture
def gguf_calls(monkeypatch):
    """Record the calls download_attribute_model makes to hf_hub_download,
    which is how it fetches a GGUF file."""
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kwargs: calls.append(kwargs))
    return calls


@pytest.fixture
def snapshot_calls(monkeypatch, gguf_calls):
    """Record the calls download_attribute_model makes to snapshot_download."""
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda **kwargs: calls.append(kwargs))
    monkeypatch.delenv("NGEC_ATTRIBUTE_MODEL", raising=False)
    return calls


def test_attribute_model_default(snapshot_calls):
    download_attribute_model(gguf=False)
    assert snapshot_calls == [{"repo_id": DEFAULT_MODEL, "force_download": False}]


def test_attribute_model_gguf(snapshot_calls, gguf_calls):
    download_attribute_model(gguf=True)
    assert gguf_calls == [{"repo_id": "ahalt/qwen3.5-event-extraction-0.8b-GGUF",
                           "filename": "qwen3.5-event-extraction-0.8b-Q8_0.gguf",
                           "force_download": False}]


def test_attribute_model_gguf_follows_llama_cpp_install(snapshot_calls, gguf_calls,
                                                        monkeypatch):
    # By default the GGUF comes along exactly when llama-cpp-python is installed.
    real_find_spec = importlib.util.find_spec
    for installed in (False, True):
        gguf_calls.clear()
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name, *a: (object() if installed else None) if name == "llama_cpp"
            else real_find_spec(name, *a))
        download_attribute_model()
        assert len(gguf_calls) == int(installed)


def test_attribute_model_without_published_gguf(snapshot_calls, gguf_calls):
    # Nothing to fetch, and not an error: the user has to supply their own.
    download_attribute_model("ahalt/event-attribute-extractor", gguf=True)
    assert gguf_calls == []


def test_attribute_model_env_var(snapshot_calls, monkeypatch):
    monkeypatch.setenv("NGEC_ATTRIBUTE_MODEL", "ahalt/event-attribute-extractor")
    download_attribute_model()
    assert snapshot_calls[0]["repo_id"] == "ahalt/event-attribute-extractor"


def test_attribute_model_argument_beats_env_var(snapshot_calls, monkeypatch):
    monkeypatch.setenv("NGEC_ATTRIBUTE_MODEL", "ahalt/event-attribute-extractor")
    download_attribute_model("someone/other-model", force=True, gguf=False)
    assert snapshot_calls == [{"repo_id": "someone/other-model", "force_download": True}]


def test_attribute_model_local_directory_is_skipped(snapshot_calls, tmp_path):
    download_attribute_model(str(tmp_path), gguf=False)
    assert snapshot_calls == []


def test_spacy_models_present_need_no_pip(monkeypatch):
    # A uv venv has no pip; that only matters if something needs installing.
    monkeypatch.setattr(ngec.models, "installed_spacy_models",
                        lambda: set(REQUIRED_SPACY_MODELS))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "pip" else real_find_spec(name, *a))
    download_spacy_models()


def test_spacy_models_missing_without_pip_raises(monkeypatch):
    monkeypatch.setattr(ngec.models, "installed_spacy_models", lambda: set())
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "pip" else real_find_spec(name, *a))
    with pytest.raises(RuntimeError, match="needs pip"):
        download_spacy_models()


def test_classifier_encoder_comes_from_metadata():
    # The shipped event_models_v2 were trained on all-mpnet-base-v2.
    assert classifier_encoder_name() == "sentence-transformers/all-mpnet-base-v2"


def test_encoders(monkeypatch):
    loaded = []

    def fake_sentence_transformer(name, device=None, **kwargs):
        loaded.append((name, device))

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer",
                        fake_sentence_transformer)
    monkeypatch.setattr(ngec.actors.common, "SentenceTransformer",
                        fake_sentence_transformer)
    monkeypatch.delenv("NGEC_WIKI_ENCODER", raising=False)
    monkeypatch.setenv("NGEC_AGENT_ENCODER", "someone/agent-encoder")

    download_encoders()

    assert loaded == [(classifier_encoder_name(), "cpu"),
                      (DEFAULT_ENCODER, "cpu"),
                      ("someone/agent-encoder", "cpu")]


@pytest.mark.parametrize("argv, expected", [
    ([], {"force": False, "attribute_model": None, "include_attribute_model": True,
          "gguf": None}),
    (["--force", "--attribute-model", "x/y"],
     {"force": True, "attribute_model": "x/y", "include_attribute_model": True,
      "gguf": None}),
    (["--no-attribute-model"],
     {"force": False, "attribute_model": None, "include_attribute_model": False,
      "gguf": None}),
    (["--gguf"],
     {"force": False, "attribute_model": None, "include_attribute_model": True,
      "gguf": True}),
])
def test_cli(monkeypatch, argv, expected):
    received = {}
    monkeypatch.setattr(ngec.cli, "download_models", lambda **kwargs: received.update(kwargs))
    assert ngec.cli.main(["download-models", *argv]) == 0
    assert received == expected
