"""
Test the differnet possible backend and GPU configurations for AttributeModel.

| Platform | GPU possible | transformers | vLLM | mlx | 
|----------|--------------|--------------|------|-----|
| Linux    |            x |            x |    x |     |
| MacOS    |              |            x |      |   x |
| Windows  |            x |            x |    x |     |

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


VLLM_AVAILABLE = has_package("vllm")
MLX_AVAILABLE = has_package("mlx")
TRANSFORMERS_AVAILABLE = has_package("transformers")


@pytest.fixture
def sample_attribute_model_input():
    from ngec.attribute_model import EventDict
    return [
        EventDict(
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