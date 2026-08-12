
# TODO: take another look at batching. Is it worth adding?
# TODO: presence_penalty isn't implemented here -- transformers has no
#       equivalent sampler (repetition_penalty is a different thing), so a
#       caller who sets it gets vLLM behaviour only. Documented in base.py.
# TODO: close() doesn't release GPU memory? Do we want torch.cuda.empty_cache()?

import logging

import torch
from transformers import AutoTokenizer, pipeline, set_seed

from .base import Conversation, EngineCapabilities, GenerationConfig

logger = logging.getLogger(__name__)

# Don't subclass on GenerationEngine so that we can have an easier FakeEngine 
# for testing. 
class TransformersEngine:
    """A local engine using the HuggingFace Transformers library.

    This is the fallback engine, so it is always importable. It is slower than
    vLLM and LLaMA.cpp, but it is portable and can run on CPU-only machines.

    Generates one conversation at a time rather than batching. Batching looks
    like it should help -- decode is memory-bandwidth-bound, so a batch reads
    the weights once for several sequences -- and on uniform-length synthetic
    prompts it measured 1.7x faster. On real documents it was 1.8x *slower*
    (50s/doc -> 90s/doc): a batch runs until its longest member finishes, and
    these outputs vary from 1 to 345 tokens, so short sequences sit padded
    while the longest one decodes. Hence capabilities.batching=False.
    """

    capabilities = EngineCapabilities(schema=False, batching=False)

    def __init__(self, model_name: str, *, device: str = "cpu",
                 config: GenerationConfig | None = None, silent: bool = False):
        self.config = config or GenerationConfig()
        device_id = 0 if device == "cuda" else -1
        if not silent:
            logger.debug("Loading transformers model")
        # The checkpoint is bfloat16. On a GPU that is what we want, but on a
        # CPU without AVX512-BF16 or AMX every bf16 matmul is emulated, which
        # measured ~25% slower than float32 on an AVX2 machine.
        self.model = pipeline(
            "text-generation",
            model=model_name,
            device=device_id,
            torch_dtype="auto" if device == "cuda" else torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Kept even though generate() runs one conversation at a time: left
        # padding is REQUIRED for batched decoder-only generation, and without
        # it a future batch would silently generate from pad tokens rather than
        # from the end of the prompt (~17% malformed JSON when measured). Set on
        # BOTH objects: the pipeline builds its collator from
        # self.model.tokenizer, so setting only self.tokenizer is a no-op.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.tokenizer.pad_token = self.tokenizer.pad_token
        self.model.tokenizer.padding_side = "left"

    def generate(self, conversations: list[Conversation], *,
                 schema: dict | None = None) -> list[str]:
        # schema is ignored -- capabilities.schema is False, so the caller
        # falls back to prompt-instructed JSON plus salvage parsing.
        if self.config.seed is not None:
            # Seeded once per call, not per conversation: that makes a whole
            # process() run reproducible without making every conversation in it
            # sample from the same starting state.
            set_seed(self.config.seed)
        responses = []
        for conversation in conversations:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            output = self.model(
                prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                min_p=self.config.min_p,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            responses.append(output[0]["generated_text"].strip())
        return responses

    def close(self) -> None:
        # Assigned rather than deleted so a second close() -- or an attribute
        # check afterwards -- doesn't raise AttributeError.
        self.model = None
        self.tokenizer = None
