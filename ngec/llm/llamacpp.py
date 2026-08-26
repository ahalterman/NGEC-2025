"""HTTP client for a running `llama-server` instance.

This module only talks to an already-running server -- it never builds,
quantizes, or launches one. See DEVELOPING.md for local dev setup and
demo/deploy/README.md for the systemd deployment.
"""
import json
import logging
import os
import urllib.error
import urllib.request

from transformers import AutoTokenizer

from .base import Conversation, EngineCapabilities, GenerationConfig

logger = logging.getLogger(__name__)


class LlamaCppServerEngine:
    """Talks to `llama-server`'s native `/completion` endpoint over HTTP.

    Deliberately not `/v1/chat/completions`: the demo's ~12s/doc figure
    depends on `cache_prompt: True` reusing the server's KV cache across the
    several event types extracted from one document, and that was only ever
    measured against `/completion`. Revisit only after confirming
    `/v1/chat/completions` caches the shared prefix the same way on this
    llama.cpp build -- see the "one thing to measure" note in
    TASK-llm-interface.md.

    `/completion` takes a pre-rendered prompt string, not a messages list, so
    this engine renders the chat template itself -- which is why it loads an
    AutoTokenizer despite loading no weights. That tokenizer only has to match
    the *prompt format* the served GGUF was trained on; the actual weights are
    whatever `llama-server` was started with, and nothing enforces that the
    two agree (see AttributeModel's model_name docstring, and
    demo/ngec_demo/resources.py::_llamacpp_model_health, which checks this for
    the demo specifically -- a direct AttributeModel caller gets no such
    check).

    No shared base class with a hypothetical in-process LlamaCppLocalEngine:
    they diverge on close() (no-op here, must free the model there), on
    failure mode (a retryable connection error here, a fatal load error
    there), and on whether batching could ever be True (continuous batching on
    concurrent requests is plausible here; never there).
    """

    capabilities = EngineCapabilities(schema=True, batching=False)

    def __init__(self, model_name: str, *, url: str | None = None,
                 config: GenerationConfig | None = None, silent: bool = False):
        self.config = config or GenerationConfig()
        self.url = (url or os.environ.get("NGEC_LLAMACPP_URL")
                   or "http://127.0.0.1:8080")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not silent:
            logger.info(f"Using llama-server at {self.url}")

    def generate(self, conversations: list[Conversation], *,
                 schema: dict | None = None) -> list[str]:
        responses = []
        for conversation in conversations:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            body = {
                "prompt": prompt,
                "n_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "min_p": self.config.min_p,
                "presence_penalty": self.config.presence_penalty,
                "cache_prompt": True,
            }
            if self.config.seed is not None:
                body["seed"] = self.config.seed
            if schema is not None:
                body["json_schema"] = schema
            request = urllib.request.Request(
                f"{self.url}/completion",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as resp:
                    payload = json.loads(resp.read())
                responses.append(payload.get("content", "").strip())
            except urllib.error.HTTPError as e:
                # The server responded but rejected the request -- e.g. this
                # llama-server build may not support `json_schema`. Distinct
                # from "not running", so it gets its own message.
                logger.error(
                    f"llama-server at {self.url} rejected the request "
                    f"({e.code}): {e.read().decode(errors='replace')[:200]}"
                )
                responses.append("")
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                logger.error(
                    f"llama-server at {self.url} did not respond: {e}. "
                    "Is it running? See DEVELOPING.md."
                )
                responses.append("")
        return responses

    def close(self) -> None:
        self.tokenizer = None