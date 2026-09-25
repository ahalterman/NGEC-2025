
# TODO: AttributeModel does not use load_engine() yet; it picks its engine
#       itself. vllm and mlx are not engines yet either.

import logging
import os
import platform
import sys
from importlib.util import find_spec

from .base import EngineCapabilities, GenerationConfig, GenerationEngine

logger = logging.getLogger(__name__)


def choose_backend() -> str:
    """The backend that `backend="auto"` stands for on this machine.

    - "vllm" if vllm is installed and PyTorch can see a CUDA GPU;
    - "mlx" on a Mac with Apple Silicon, if mlx-lm is installed;
    - "llamacpp" otherwise, which runs on any CPU.

    This only looks at what is installed. If it chooses llamacpp and
    llama-cpp-python is not installed, loading the model then says what to
    install. It never chooses "transformers", which is deprecated and much
    slower than llamacpp on a CPU.
    """
    if find_spec("vllm") is not None:
        import torch
        if torch.cuda.is_available():
            return "vllm"
    if (sys.platform == "darwin" and platform.machine() == "arm64"
            and find_spec("mlx_lm") is not None):
        return "mlx"
    return "llamacpp"


def llamacpp_server_url(url: str | None = None) -> str | None:
    """The llama-server to use, if any: `url`, then NGEC_LLAMACPP_URL.

    None means there is no server, and the llamacpp backend runs the model in
    this process instead.
    """
    return url or os.environ.get("NGEC_LLAMACPP_URL") or None


def load_engine(spec: str = "auto", **kw) -> GenerationEngine:
    """Resolve a backend spec to an engine.

    spec: "auto" | "transformers" | "llamacpp" | "llamacpp:<url>" | "mlx" | "vllm"
          | "openai:http://localhost:8080/v1"

    "auto" is `choose_backend()`. "llamacpp" runs the model in this process
    unless a server URL is given (as "llamacpp:<url>", url=, or
    NGEC_LLAMACPP_URL). Logs the choice at INFO: a user reporting "it's slow"
    should be able to tell you which engine they got without instrumenting
    anything.
    """
    if spec == "auto":
        spec = choose_backend()
        logger.info(f"Backend 'auto' chose '{spec}'")

    if spec == "llamacpp" or spec.startswith("llamacpp:"):
        if ":" in spec:
            kw.setdefault("url", spec.split(":", 1)[1])
        url = llamacpp_server_url(kw.pop("url", None))
        if url:
            from .llamacpp import LlamaCppServerEngine
            return LlamaCppServerEngine(url=url, **kw)
        from .llamacpp import LlamaCppLocalEngine
        return LlamaCppLocalEngine(**kw)

    if spec != "transformers":
        logger.warning(f"Backend '{spec}' is not wired up yet; "
                       "only transformers and llamacpp are implemented. "
                       "Using transformers.")

    # Default to transformers
    from .transformers import TransformersEngine
    return TransformersEngine(**kw)
