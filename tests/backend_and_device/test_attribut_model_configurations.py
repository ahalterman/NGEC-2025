"""
Test the differnet possible backend and GPU configurations for AttributeModel.

| Platform | GPU possible | transformers | vLLM | mlx | llamacpp |
|----------|--------------|--------------|------|-----|----------|
| Linux    |            x |            x |    x |     |        x |
| MacOS    |              |            x |      |   x |        x |
| Windows  |            x |            x |    x |     |        x |

"""

import pytest
import platform
from importlib.util import find_spec

from ngec.attribute_model import AttributeModel


def has_package(package_name):
    """Check if a package is installed."""
    return find_spec(package_name) is not None


def is_mac():
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def has_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_mps():
    """Check if MPS (Metal Performance Shaders) is available."""
    if not has_package("torch"):
        return False
    import torch
    return torch.backends.mps.is_available()


def llamacpp_server_available(url="http://127.0.0.1:8080"):
    """Check if a llama-server is reachable at the configured URL."""
    import os
    import urllib.request

    url = os.environ.get("NGEC_LLAMACPP_URL", url)
    try:
        urllib.request.urlopen(f"{url}/health", timeout=2)
        return True
    except Exception:
        return False


VLLM_AVAILABLE = has_package("vllm")
MLX_AVAILABLE = has_package("mlx")
TRANSFORMERS_AVAILABLE = has_package("transformers")


@pytest.fixture
def sample_attribute_model_input():
    from ngec.attribute_model import AttributeModelInput
    return [
        AttributeModelInput(
            event_text="A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.",
            event_type="PROTEST",
        )
    ]


# This should work on all platforms
def test_transformers_cpu(sample_attribute_model_input):
    am = AttributeModel(silent=True, gpu=False, backend="transformers")
    output = am.process(sample_attribute_model_input)
    assert output is not None


def test_mlx(sample_attribute_model_input):
    from ngec.attribute_model import AttributeModel
    
    if not is_mac():
        pytest.skip("MLX backend only supported on macOS with mlx package installed.")
    if not MLX_AVAILABLE:
        pytest.skip("MLX package not installed.")

    am = AttributeModel(silent=True, gpu=False, backend="mlx")
    output = am.process(sample_attribute_model_input)
    assert output is not None


def test_vllm_cpu(sample_attribute_model_input):
    from ngec.attribute_model import AttributeModel

    if is_mac():
        pytest.skip("vLLM backend not supported on macOS.")
    if not VLLM_AVAILABLE:
        pytest.skip("vLLM package not installed.")

    am = AttributeModel(silent=True, gpu=False, backend="vllm")
    output = am.process(sample_attribute_model_input)
    assert output is not None


def test_llamacpp_server(sample_attribute_model_input):
    import os
    from ngec.attribute_model import AttributeModel

    if not llamacpp_server_available():
        pytest.skip("No llama-server reachable; set NGEC_LLAMACPP_URL or start one "
                    "(see DEVELOPING.md).")

    # The URL is passed explicitly: without one, the llamacpp backend runs the
    # model in this process instead (test_llamacpp_in_process).
    url = os.environ.get("NGEC_LLAMACPP_URL", "http://127.0.0.1:8080")
    am = AttributeModel(silent=True, gpu=False, backend="llamacpp", llamacpp_url=url)
    output = am.process(sample_attribute_model_input)

    assert len(output) == 1
    attributes = output[0]["attributes"]
    assert attributes["event_type"]
    assert isinstance(attributes["actor"], list)


def published_gguf_is_downloaded():
    """Whether the default model's GGUF file is already in the Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache
    from ngec.attribute_model import DEFAULT_MODEL
    from ngec.llm.llamacpp import KNOWN_GGUF_FILES

    repo_id, filename = KNOWN_GGUF_FILES[DEFAULT_MODEL]
    return isinstance(try_to_load_from_cache(repo_id, filename), str)


def test_llamacpp_in_process(sample_attribute_model_input, monkeypatch):
    """The real GGUF, run in this process. Opt-in in the sense that it only runs
    where llama-cpp-python is installed and the GGUF is already downloaded
    (`ngec download-models --gguf`); it never downloads the 834 MB file itself."""
    from ngec.attribute_model import AttributeModel
    from ngec.llm.llamacpp import LlamaCppLocalEngine

    if not has_package("llama_cpp"):
        pytest.skip("llama-cpp-python not installed (the llamacpp extra).")
    if not published_gguf_is_downloaded():
        pytest.skip("GGUF not downloaded; run `ngec download-models --gguf`.")
    monkeypatch.delenv("NGEC_LLAMACPP_URL", raising=False)
    monkeypatch.delenv("NGEC_ATTRIBUTE_GGUF", raising=False)
    monkeypatch.delenv("NGEC_ATTRIBUTE_MODEL", raising=False)

    am = AttributeModel(silent=True, backend="llamacpp")
    assert isinstance(am.engine, LlamaCppLocalEngine)
    output = am.process(sample_attribute_model_input)
    am.engine.close()

    assert len(output) == 1
    attributes = output[0]["attributes"]
    assert attributes["event_type"] == "PROTEST"
    assert any("Hindu nationalists" in actor for actor in attributes["actor"])
    assert any("Dehli" in place for place in attributes["location"])