"""The llamacpp backend and the automatic backend choice, with no model loaded.

`llama_cpp.Llama` and the Hugging Face tokenizer are replaced with stand-ins
that record what they were given, so these tests check the choices NGEC makes
-- which engine, which decoding settings, which GGUF file, how many threads --
and not the model. tests/backend_and_device/ runs the real GGUF.
"""

import logging
import sys
import types

import huggingface_hub
import pytest

import ngec.llm
import ngec.llm.llamacpp
from ngec.attribute_model import DEFAULT_MODEL, AttributeModel
from ngec.llm import GenerationConfig, choose_backend, llamacpp_server_url
from ngec.llm.llamacpp import (KNOWN_GGUF_FILES, MAX_DEFAULT_THREADS,
                               LlamaCppLocalEngine, default_threads, find_gguf)


class FakeTokenizer:
    """Renders a conversation as text and records how it was asked to."""

    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, conversation, **kwargs):
        self.template_kwargs.append(kwargs)
        return "".join(f"<{m['role']}>{m['content']}" for m in conversation) + "<assistant>"


class FakeLlama:
    """Stands in for llama_cpp.Llama; returns `reply` for every prompt."""

    reply = ' [{"event_type": "PROTEST", "actor": ["students"]}] '

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        self.calls = []
        self.closed = False

    def create_completion(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if "too long" in prompt:
            raise ValueError("Requested tokens (9000) exceed context window of 8192")
        return {"choices": [{"text": self.reply}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7}}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_llama(monkeypatch, tmp_path):
    """Install the stand-ins, and a GGUF 'file' for them to be pointed at."""
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr(ngec.llm.llamacpp.AutoTokenizer, "from_pretrained",
                        lambda name: FakeTokenizer())
    for var in ("NGEC_LLAMACPP_URL", "NGEC_LLAMACPP_THREADS",
                "NGEC_ATTRIBUTE_GGUF", "NGEC_ATTRIBUTE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    return str(gguf)


CONVERSATION = [{"role": "system", "content": "Extract."},
                {"role": "user", "content": "## Document: Students marched."}]


# ---------------------------------------------------------------- the engine

def test_local_engine_prompt_and_greedy_decoding(fake_llama):
    greedy = GenerationConfig(temperature=0.0, top_p=1.0, top_k=1, min_p=0.0,
                              presence_penalty=0.0, max_tokens=1024)
    engine = LlamaCppLocalEngine(DEFAULT_MODEL, gguf_path=fake_llama,
                                 n_threads=3, config=greedy, silent=True)

    assert engine.generate([CONVERSATION]) == [FakeLlama.reply.strip()]

    # The prompt is rendered the way every other backend renders it...
    assert engine.tokenizer.template_kwargs == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}]
    prompt, settings = engine.llm.calls[0]
    assert prompt == FakeTokenizer().apply_chat_template(CONVERSATION)
    # ...and decoded greedily, to the same length, stopping at the end of turn.
    assert settings["temperature"] == 0.0 and settings["top_k"] == 1
    assert settings["repeat_penalty"] == 1.0 and settings["presence_penalty"] == 0.0
    assert settings["max_tokens"] == 1024
    assert settings["stop"] == ["<|im_end|>"]
    assert "seed" not in settings
    # The model runs on the requested threads, with llama-server's context.
    assert engine.llm.model_path == fake_llama
    assert engine.llm.kwargs["n_threads"] == 3
    assert engine.llm.kwargs["n_threads_batch"] == 3
    assert engine.llm.kwargs["n_ctx"] == 8192
    assert engine.last_timings == [{"prompt_n": 12, "predicted_n": 7}]


def test_local_engine_passes_a_seed(fake_llama):
    engine = LlamaCppLocalEngine(DEFAULT_MODEL, gguf_path=fake_llama, n_threads=1,
                                 config=GenerationConfig(seed=5), silent=True)
    engine.generate([CONVERSATION])
    assert engine.llm.calls[0][1]["seed"] == 5


def test_local_engine_prompt_too_long_is_an_empty_response(fake_llama, caplog):
    # Like a failed request to llama-server: one empty response, which the
    # attribute model then reports as a dropped event, and the rest go on.
    engine = LlamaCppLocalEngine(DEFAULT_MODEL, gguf_path=fake_llama, n_threads=1,
                                 silent=True)
    too_long = [{"role": "user", "content": "too long"}]
    with caplog.at_level(logging.ERROR, logger="ngec.llm.llamacpp"):
        responses = engine.generate([too_long, CONVERSATION])
    assert responses == ["", FakeLlama.reply.strip()]
    assert "context window" in caplog.text


def test_local_engine_close(fake_llama):
    engine = LlamaCppLocalEngine(DEFAULT_MODEL, gguf_path=fake_llama, n_threads=1,
                                 silent=True)
    llm = engine.llm
    engine.close()
    engine.close()
    assert llm.closed and engine.llm is None


def test_local_engine_without_llama_cpp_says_what_to_install(fake_llama, monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)   # makes the import fail
    with pytest.raises(ImportError, match=r'pip install "ngec\[llamacpp\]"'):
        LlamaCppLocalEngine(DEFAULT_MODEL, gguf_path=fake_llama, silent=True)


# ---------------------------------------------------------- finding the GGUF

def test_find_gguf_prefers_the_given_path(fake_llama, monkeypatch, tmp_path):
    other = tmp_path / "other.gguf"
    other.write_bytes(b"GGUF")
    monkeypatch.setenv("NGEC_ATTRIBUTE_GGUF", str(other))
    assert find_gguf(DEFAULT_MODEL, fake_llama) == fake_llama
    assert find_gguf(DEFAULT_MODEL) == str(other)


def test_find_gguf_missing_file(fake_llama, tmp_path):
    with pytest.raises(FileNotFoundError):
        find_gguf(DEFAULT_MODEL, str(tmp_path / "nowhere.gguf"))


def test_find_gguf_downloads_the_published_file(fake_llama, monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kwargs: calls.append(kwargs) or "/cache/x.gguf")
    assert find_gguf(DEFAULT_MODEL) == "/cache/x.gguf"
    repo_id, filename = KNOWN_GGUF_FILES[DEFAULT_MODEL]
    assert calls == [{"repo_id": repo_id, "filename": filename}]


def test_find_gguf_for_a_model_without_one(fake_llama):
    with pytest.raises(ValueError, match="no published GGUF"):
        find_gguf("ahalt/event-attribute-extractor")


# ------------------------------------------------------------------ threads

def test_threads_from_environment(monkeypatch):
    monkeypatch.setenv("NGEC_LLAMACPP_THREADS", "12")
    assert default_threads() == 12


def test_threads_default_is_capped(monkeypatch):
    monkeypatch.delenv("NGEC_LLAMACPP_THREADS", raising=False)
    monkeypatch.setattr(ngec.llm.llamacpp, "_linux_core_count", lambda: 64)
    monkeypatch.setattr(ngec.llm.llamacpp, "_mac_core_count", lambda: 64)
    assert default_threads() == MAX_DEFAULT_THREADS


def test_threads_fallback_is_half_the_logical_cpus(monkeypatch):
    monkeypatch.delenv("NGEC_LLAMACPP_THREADS", raising=False)
    monkeypatch.setattr(ngec.llm.llamacpp, "_linux_core_count", lambda: None)
    monkeypatch.setattr(ngec.llm.llamacpp, "_mac_core_count", lambda: None)
    monkeypatch.setattr(ngec.llm.llamacpp.os, "cpu_count", lambda: 4)
    assert default_threads() == 2


def test_threads_default_on_this_machine(monkeypatch):
    monkeypatch.delenv("NGEC_LLAMACPP_THREADS", raising=False)
    assert 1 <= default_threads() <= MAX_DEFAULT_THREADS


# ------------------------------------------- which engine AttributeModel uses

def test_llamacpp_backend_runs_in_process_without_a_url(fake_llama, monkeypatch):
    am = AttributeModel(backend="llamacpp", gguf_path=fake_llama,
                        llamacpp_threads=2, silent=True)
    assert isinstance(am.engine, LlamaCppLocalEngine)
    assert am.engine.n_threads == 2
    # Greedy, as the v6 model was evaluated.
    assert am.engine.config.temperature == 0.0


@pytest.mark.parametrize("how", ["argument", "environment"])
def test_llamacpp_backend_uses_a_server_when_given_a_url(fake_llama, monkeypatch, how):
    # The demo deployment passes llamacpp_url; NGEC_LLAMACPP_URL does the same
    # for everyone else. Either way nothing is loaded in this process.
    from ngec.llm.llamacpp import LlamaCppServerEngine
    kwargs = {}
    if how == "argument":
        kwargs["llamacpp_url"] = "http://127.0.0.1:9999"
    else:
        monkeypatch.setenv("NGEC_LLAMACPP_URL", "http://127.0.0.1:9999")
    am = AttributeModel(backend="llamacpp", silent=True, **kwargs)
    assert isinstance(am.engine, LlamaCppServerEngine)
    assert am.engine.url == "http://127.0.0.1:9999"


def test_process_through_the_local_engine(fake_llama):
    am = AttributeModel(backend="llamacpp", gguf_path=fake_llama, silent=True)
    out = am.process([{"id": "s1", "event_text": "Students marched.",
                       "event_type": "PROTEST", "event_mode": ""}])
    assert len(out) == 1
    assert out[0]["attributes"]["actor"] == ["students"]


def test_server_url_resolution(monkeypatch):
    monkeypatch.delenv("NGEC_LLAMACPP_URL", raising=False)
    assert llamacpp_server_url() is None
    assert llamacpp_server_url("http://a") == "http://a"
    monkeypatch.setenv("NGEC_LLAMACPP_URL", "http://b")
    assert llamacpp_server_url() == "http://b"
    assert llamacpp_server_url("http://a") == "http://a"


# ------------------------------------------------------------ backend="auto"

def _pretend(monkeypatch, *, installed=(), cuda=False, platform="linux", machine="x86_64"):
    import torch
    real_find_spec = ngec.llm.find_spec
    known = {"vllm", "mlx_lm"}
    monkeypatch.setattr(
        ngec.llm, "find_spec",
        lambda name: (object() if name in installed else None) if name in known
        else real_find_spec(name))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(ngec.llm.sys, "platform", platform)
    monkeypatch.setattr(ngec.llm.platform, "machine", lambda: machine)


@pytest.mark.parametrize("situation, expected", [
    ({"installed": ["vllm"], "cuda": True}, "vllm"),
    ({"installed": ["vllm"], "cuda": False}, "llamacpp"),   # vllm but no GPU
    ({"installed": [], "cuda": True}, "llamacpp"),          # GPU but no vllm
    ({"installed": ["mlx_lm"], "platform": "darwin", "machine": "arm64"}, "mlx"),
    ({"installed": [], "platform": "darwin", "machine": "arm64"}, "llamacpp"),
    ({"installed": ["mlx_lm"], "platform": "darwin", "machine": "x86_64"}, "llamacpp"),
    ({}, "llamacpp"),
])
def test_choose_backend(monkeypatch, situation, expected):
    _pretend(monkeypatch, **situation)
    assert choose_backend() == expected


def test_auto_backend_is_resolved_and_logged(fake_llama, monkeypatch, caplog):
    _pretend(monkeypatch)
    with caplog.at_level(logging.INFO, logger="ngec.attribute_model"):
        am = AttributeModel(gguf_path=fake_llama, silent=True)   # backend="auto"
    assert am.backend == "llamacpp"
    assert isinstance(am.engine, LlamaCppLocalEngine)
    assert "chosen automatically" in caplog.text


def test_auto_backend_without_llama_cpp_says_what_to_install(fake_llama, monkeypatch):
    _pretend(monkeypatch)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(ImportError, match=r"ngec\[llamacpp\]"):
        AttributeModel(silent=True)


def test_vllm_missing_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ImportError, match=r"ngec\[vllm\]"):
        AttributeModel(backend="vllm", silent=True)


# ------------------------------------------------ transformers is deprecated

def test_transformers_backend_logs_deprecation(monkeypatch, caplog):
    import ngec.llm.transformers

    class StandInEngine:
        def __init__(self, **kwargs):
            self.tokenizer = None
    monkeypatch.setattr(ngec.llm.transformers, "TransformersEngine", StandInEngine)
    with caplog.at_level(logging.WARNING, logger="ngec.attribute_model"):
        # Shown even with silent=True, which is what PloverCoder passes.
        AttributeModel(backend="transformers", silent=True)
    assert "deprecated" in caplog.text
    assert "llamacpp" in caplog.text
