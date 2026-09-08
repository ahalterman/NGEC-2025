
# TODO: implement load_engine(), right now it just defaults to transformers

import logging

from .base import EngineCapabilities, GenerationConfig, GenerationEngine

logger = logging.getLogger(__name__)


def load_engine(spec: str = "auto", **kw) -> GenerationEngine:
    """Resolve a backend spec to an engine.

    spec: "auto" | "transformers" | "llamacpp" | "llamacpp:<url>" | "mlx" | "vllm"
          | "openai:http://localhost:8080/v1"

    "auto" probes what is importable, prefers fast over portable, and falls
    back to transformers -- which is a hard dependency, so it always works.
    Log the choice at INFO: a user reporting "it's slow" should be able to
    tell you which engine they got without instrumenting anything.
    """
    if spec == "llamacpp" or spec.startswith("llamacpp:"):
        from .llamacpp import LlamaCppServerEngine
        if ":" in spec:
            kw.setdefault("url", spec.split(":", 1)[1])
        return LlamaCppServerEngine(**kw)

    if spec != "transformers":
        logger.warning(f"Backend '{spec}' is not wired up yet; "
                       "only transformers and llamacpp are implemented. "
                       "Using transformers.")

    # Default to transformers
    from .transformers import TransformersEngine
    return TransformersEngine(**kw)
