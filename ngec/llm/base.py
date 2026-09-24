from dataclasses import dataclass
from typing import Protocol, runtime_checkable


Message = dict[str, str]              # {"role": ..., "content": ...}
Conversation = list[Message]


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling settings, defined once and translated per engine.

    Defaults follow Qwen3 non-thinking guidance, for the Qwen3 attribute models
    (greedy decoding sent those into repetition loops). The Qwen3.5 v6 model is
    decoded greedily instead, which is `temperature=0.0`; see
    `AttributeModel._generation_config`.

    Not every engine honors every field: an engine translates what its backend
    supports and ignores the rest, so check here before assuming a knob took
    effect. `presence_penalty` in particular has no transformers equivalent
    (`repetition_penalty` is a different thing, not a rename), so it is vLLM-only
    for now rather than universal.
    """
    temperature: float = 0.5
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5     # vLLM only; ignored by TransformersEngine
    max_tokens: int = 1024
    seed: int | None = None


@dataclass(frozen=True)
class EngineCapabilities:
    """What a backend can do, so the caller can adapt."""
    schema: bool = False              # can constrain output to a JSON schema
    batching: bool = False            # meaningfully faster given a full list


@runtime_checkable
class GenerationEngine(Protocol):
    """Protocol that a backend engine must implement to be used by the attribute
    model.

    Engines own chat templating: local engines apply the tokenizer's template,
    HTTP engines let the server do it. Callers pass structured conversations
    and never touch a tokenizer.
    """
    capabilities: EngineCapabilities

    def generate(self, conversations: list[Conversation], *,
                 schema: dict | None = None) -> list[str]:
        """One raw completion per conversation, in order.

        `schema` is advisory: engines without `capabilities.schema` ignore it,
        and the caller falls back to prompt-instructed JSON plus salvage
        parsing rather than erroring.
        """
        ...

    def close(self) -> None:
        ...
