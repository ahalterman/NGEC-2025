import os
import pandas as pd
import re
import json
import logging

from collections import Counter
from importlib import resources
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any, cast, Literal, TypedDict, NotRequired

from .attributes.schema import ATTRIBUTE_SCHEMA, normalize_spans, parse_response
from .llm.base import Conversation, GenerationEngine
from .utilities import explode_events, write_intermediate

logger = logging.getLogger(__name__)

# The line below is only useful for debugging/timing
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Address vLLM multiprocessing method error
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


#
#   Types and classes for type annotations
#   ================================
#
#   These don't do anything at runtime, but make it easier to see what kind
#   of input the AttributeModel expects and what kind of output it produces.
#

BackendType = Literal["auto", "vllm", "llamacpp", "mlx", "transformers"]

# What to install for each backend, for the error raised when it is missing.
BACKEND_INSTALL_HINTS = {
    "vllm": 'pip install "ngec[vllm]" (Linux with an NVIDIA GPU)',
    "mlx": 'pip install "ngec[mlx]" (a Mac with Apple Silicon)',
    "llamacpp": ('pip install "ngec[llamacpp]" --extra-index-url '
                 'https://abetlen.github.io/llama-cpp-python/whl/cpu'),
}

# Which model to extract attributes with, and the prompt format it was trained
# on. The two are not independent: a model produces markedly worse spans when
# prompted in a format it never saw, and the difference is quiet — valid JSON
# with worse contents, not an error. So they are chosen together here rather
# than being two knobs a caller can mismatch.
#
#   legacy  the format `ahalt/event-attribute-extractor` was trained on:
#           a terse system prompt, and a user message of
#           `### Document: … ### Event: **TYPE**: …` with the sub-event and
#           special instructions on their own `###` lines, then a closing
#           "Extract the attributes…" instruction.
#   v5      the format of the 2026 retraining (exp5.1): a system prompt that
#           states the output format and the extraction rules, and a user
#           message of `## Document: … ## Event Type: <whole definition>` with
#           the sub-event and instructions inline in that definition, and no
#           closing instruction. Reproduced from `eval_unified.py` in
#           train_NGEC_2026, which is what produced that model's reported
#           numbers.
#   v6      the format of the Qwen3.5-0.8B student
#           (ahalt/qwen3.5-event-extraction-0.8b). The model returns roles as
#           JSON lists rather than semicolon-joined strings, adds a `mode` key,
#           and for ASSAULT, PROTEST and COERCE also returns `killed` and
#           `injured`. The system prompt depends on the event type (those three
#           get the extra attributes), and the definitions have to be the exact
#           strings the model was trained on, which ship as
#           assets/event_definitions_v6.json. Decoding is greedy. Reproduced from
#           `student/prompting.py` in train_NGEC_2026, and checked against it on
#           every training example.
PromptFormat = Literal["legacy", "v5", "v6"]

# ahalt/qwen3.5-event-extraction-0.8b is the model of the revised paper: 70.0
# mean F1 on the 500-document VOA test set against 54.9 for
# ahalt/qwen3-event-extraction-exp5.1, the model of the submitted paper. (That
# 70.0 was measured with definitions from an older codebook; with the trained
# definitions this module sends, the same model scores 71.0 on the same set,
# through NGEC's vllm backend.) See its model card in setup/hf_release/. The two
# older models stay published and selectable by name, for comparison.
DEFAULT_MODEL = "ahalt/qwen3.5-event-extraction-0.8b"


def resolve_model_name(model_name: str | None = None) -> str:
    """The attribute model to use: `model_name` if given, then the
    NGEC_ATTRIBUTE_MODEL environment variable, then DEFAULT_MODEL.

    AttributeModel and `ngec download-models` both go through this, so the
    model that gets downloaded is the one the pipeline then loads.
    """
    return (model_name
            or os.environ.get("NGEC_ATTRIBUTE_MODEL")
            or DEFAULT_MODEL)

# Models whose prompt format is known. A path or name that is not listed falls
# back to "legacy" with a warning, because guessing silently is how a model ends
# up being evaluated in a format it was never trained on.
#
# A Hugging Face id ("namespace/name") also resolves via the basename branch of
# resolve_prompt_format below, since os.path.basename("ahalt/foo") == "foo" — so
# "ahalt/qwen3-event-extraction-exp5.1" matches the local-directory-style key
# "qwen3-event-extraction-exp5.1" without a separate entry. It is listed
# explicitly anyway, so the mapping this actually depends on is visible here
# rather than relying on that basename coincidence.
KNOWN_PROMPT_FORMATS: dict[str, PromptFormat] = {
    "ahalt/event-attribute-extractor": "legacy",
    "ahalt/qwen3-event-extraction-exp5.1": "v5",
    "qwen3-event-extraction-exp5.1": "v5",
    "qwen3-event-extraction-exp5.2": "v5",
    "ahalt/qwen3.5-event-extraction-0.8b": "v6",
    "qwen3.5-event-extraction-0.8b": "v6",
}

# A model can also say which format it was trained on, in a small `ngec.json`
# file next to its weights: {"prompt_format": "v6"}. The v6 model's Hugging Face
# repository has one. This is what lets a copy that has been renamed, or
# downloaded to a directory with a different name, still be prompted correctly.
MODEL_INFO_FILE = "ngec.json"


def _declared_prompt_format(model_name: str) -> PromptFormat | None:
    """The format named in the model's own `ngec.json`, if it has one.

    Looks in a local directory first, then in a Hugging Face repository (the
    file is downloaded once and cached like the weights). Anything that goes
    wrong -- no such file, no network, an unreadable file -- returns None and
    the caller falls back to the name lookup.
    """
    local = os.path.join(str(model_name), MODEL_INFO_FILE)
    path = local if os.path.isfile(local) else None
    if path is None and not os.path.isdir(str(model_name)):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(str(model_name), MODEL_INFO_FILE)
        except Exception:  # noqa: BLE001 - missing file, offline, not a repo id
            return None
    if path is None:
        return None
    try:
        with open(path) as f:
            declared = json.load(f).get("prompt_format")
    except (OSError, ValueError, AttributeError):
        return None
    if declared in ("legacy", "v5", "v6"):
        return cast(PromptFormat, declared)
    logger.warning(f"{path} names an unknown prompt format {declared!r}; ignoring it.")
    return None


def resolve_prompt_format(model_name: str) -> PromptFormat:
    """The prompt format a model was trained on.

    Checked in this order: the model's name in KNOWN_PROMPT_FORMATS, the
    format the model declares in its own `ngec.json`, and the name of the
    directory it was loaded from.
    """
    if model_name in KNOWN_PROMPT_FORMATS:
        return KNOWN_PROMPT_FORMATS[model_name]
    declared = _declared_prompt_format(model_name)
    if declared is not None:
        return declared
    # Local models are given as paths; match on the directory name.
    basename = os.path.basename(str(model_name).rstrip("/"))
    if basename in KNOWN_PROMPT_FORMATS:
        return KNOWN_PROMPT_FORMATS[basename]
    logger.warning(
        f"Unknown attribute model '{model_name}'; assuming the 'legacy' prompt "
        "format. If this model was trained on a different format, pass "
        "prompt_format= explicitly — a mismatch degrades extraction quietly "
        "rather than raising. Add it to KNOWN_PROMPT_FORMATS in attribute_model.py."
    )
    return "legacy"

class Attributes(TypedDict):
    """
    Dictionary representing the extracted attributes of a single event.

    The model may extract more than one event from a document; each becomes its
    own event record (via ``explode_events``) with one of these as its
    ``attributes`` value.
    """
    event_type: str
    anchor_quote: str
    actor: list[str]
    recipient: list[str]
    date: list[str]
    location: list[str]
    # Only from v6 models: the sub-event the model assigned, and, for ASSAULT,
    # PROTEST and COERCE, the people reported killed and injured.
    mode: NotRequired[str]
    killed: NotRequired[list[str]]
    injured: NotRequired[list[str]]

class AttributeModelInput(TypedDict):
    """
    Dictionary representing minimal input for AttributeModel processing.
    
    Required keys:
        event_text: The text describing the event
        event_type: The type/category of the event
    
    Optional keys:
        event_mode: The mode of the event (optional)
        
    Additional keys are allowed and will be preserved.
    """
    event_text: str  # Required
    event_type: str  # Required
    event_mode: NotRequired[str]  # Optional
    attributes: NotRequired[Attributes]  # A single extracted event (after exploding)
    # Any other keys are allowed


# Each output record carries a single 'attributes' dict. Note the output list is
# NOT the same object as the input list: process() explodes multi-event records
# and drops empty ones, so callers must use the returned list.
class AttributeModelOutput(AttributeModelInput):
    """
    Dictionary representing output from AttributeModel processing.

    Each record has an 'attributes' key holding a single extracted event.

    """
    pass


def _load_event_definitions(def_file="PLOVER_structured_codebook_updated.csv",
                            base_path=None):
    """
    Load a CSV of event definitions (including special instructions for the model.)
    """
    if base_path is None:
        # The copy shipped in ngec/assets/
        with resources.files("ngec").joinpath("assets", def_file).open() as f:
            event_definitions = pd.read_csv(f)
    else:
        # Use provided base_path
        file_path = os.path.join(base_path, def_file)
        event_definitions = pd.read_csv(file_path)

    if 'event' not in event_definitions.columns:
        raise ValueError("During loading of the event definitions file, 'event' column was not found.")
    if 'event_def' not in event_definitions.columns:
        raise ValueError("During loading of the event definitions file, 'event_def' column was not found.")
    if 'extraction_notes' not in event_definitions.columns:
        # raise a warning instead of an error
        logger.warning(f"No 'extraction_notes' column was found in {def_file}. Are you sure you don't want to add it?")
    if 'mode' not in event_definitions.columns:
        # raise a warning instead of an error
        logger.warning(f"No 'mode' column was found in {def_file}. Are you sure you don't want to add it?")
    return event_definitions


def _make_system_content_short():
    system_content_short = """Extract political events as JSON.

OUTPUT FORMAT:
[
  {
    "event_type": "EVENT_TYPE",
    "anchor_quote": "quote from text",
    "actor": "who performed action OR N/A",
    "recipient": "who was targeted OR N/A",
    "date": "when occurred OR N/A",
    "location": "where occurred OR N/A"
  }
]

Return valid JSON only. Empty array [] if no events."""
    return system_content_short


def _make_system_content_v5():
    """The system prompt the 2026 models were evaluated with.

    Copied verbatim from `eval_unified.py::_make_prompt` in train_NGEC_2026 —
    that script produced the model's reported numbers, so this string is part of
    the measurement and should not be edited for style. It is longer than the
    legacy prompt because the rules moved out of the training data and into the
    prompt.
    """
    return """Given the event type definition below, find all instances of that event in the document and extract their attributes as JSON.

OUTPUT FORMAT:
[
  {
    "event_type": "EVENT_TYPE",
    "anchor_quote": "exact 5-15 word quote from text",
    "actor": "who performed action OR N/A",
    "recipient": "who was targeted OR N/A",
    "date": "when occurred OR N/A",
    "location": "where occurred OR N/A"
  }
]

RULES:
- All values must be exact spans copied from the text. Do not rephrase.
- ACTOR: The person, group, or entity who performed the action. Use N/A only if truly unknown/unstated. Descriptions like "gunman" or "suicide bomber" ARE valid actors.
- LOCATION: Use the most specific named place (city > region > country).
- Use short, concise spans. Omit articles (a/an/the) and unnecessary context.
- Multiple values: separate with semicolons.
- Return [] if no events of the specified type are present.
- Follow any Special Instructions provided with the event type definition."""


# --- the v6 prompt ------------------------------------------------------------
#
# Copied from `student/prompts/student_t0.md` and `reannotate/attributes.py` in
# train_NGEC_2026. Like the v5 prompt, these strings are part of the
# measurement: the model was trained on exactly this text, so edit nothing here
# for style. tests/test_attribute_model.py checks the rendered prompt.

V6_SYSTEM = (
    "Extract every instance of the given political event type from the document "
    "as a JSON list, or [] if there is none.\n\n"
    "Each record:\n"
    '{{"event_type": "<type>", "mode": "<sub-event name or empty string>",\n'
    ' "anchor_quote": "<short verbatim passage identifying this instance>",\n'
    ' "actor": [...], "recipient": [...], "date": [...], "location": [...]{extra_keys}}}\n\n'
    "Every string in actor, recipient, date, and location must be an exact, "
    "verbatim substring of the document.{extra_verbatim}\n"
    "{extra_block}"
    "Return the JSON list only."
)

# The attributes a v6 model extracts beyond the four core roles, with the
# instructions it was trained on. Which of them apply depends on the event type
# (V6_ATTRIBUTE_EVENT_TYPES).
V6_ATTRIBUTE_TEXT = {
    "killed": (
        "Every person or group the document reports as killed in this event, whether "
        "or not they were its target. Fill it whenever deaths are reported, even when "
        "the dead are also the recipient. One list element per distinct party, counts "
        "kept ('three officers'). Empty when no death is reported. Written like a "
        "recipient: the verbatim noun phrase, counts and quantifiers kept, leading "
        "articles dropped, one list element per distinct party."),
    "injured": (
        "Every person or group the document reports as injured in this event, whether "
        "or not they were its target. Fill it whenever injuries are reported, even when "
        "the injured are also the recipient. One list element per distinct party, "
        "counts kept ('three officers'). Empty when no injury is reported. Written like "
        "a recipient: the verbatim noun phrase, counts and quantifiers kept, leading "
        "articles dropped, one list element per distinct party."),
}
V6_ATTRIBUTE_EVENT_TYPES = {"ASSAULT", "PROTEST", "COERCE"}


def _make_system_content_v6(event_type: str) -> str:
    """The v6 system prompt for one event type.

    ASSAULT, PROTEST and COERCE add `killed` and `injured` to the record
    schema, with a sentence saying they are verbatim too and a block describing
    each. Every other type gets the plain prompt.
    """
    attributes = (["killed", "injured"] if event_type in V6_ATTRIBUTE_EVENT_TYPES
                  else [])
    if not attributes:
        return V6_SYSTEM.format(extra_keys="", extra_verbatim="", extra_block="")
    extra_keys = "".join(f', "{name}": [...]' for name in attributes)
    names = " and ".join(attributes)
    extra_block = ("ADDITIONAL ATTRIBUTES FOR THIS EVENT TYPE\n"
                   + "".join(f"- {name}: {V6_ATTRIBUTE_TEXT[name]}\n"
                             for name in attributes))
    return V6_SYSTEM.format(extra_keys=extra_keys,
                            extra_verbatim=f" The same holds for {names}.",
                            extra_block=extra_block)


def _load_v6_definitions(def_file=None) -> dict[tuple[str, str], str]:
    """Event definitions in the v6 format, keyed by (event type, mode).

    A mode of "" is the definition of the whole event type. With no `def_file`,
    these are the exact definitions the v6 model was trained on, from
    assets/event_definitions_v6.json. They are not rendered from the codebook
    CSV at run time, because the codebook has been edited since the model was
    trained; a definition that differs from the training one is a prompt the
    model never saw.

    `def_file` is the path to a user's own file in the same format: a JSON list
    of {"event_type": ..., "mode": ..., "definition": ...} entries.
    """
    if def_file is None:
        with resources.files("ngec").joinpath("assets", "event_definitions_v6.json").open() as f:
            entries = json.load(f)
    else:
        with open(def_file, encoding="utf-8") as f:
            entries = json.load(f)
    return {(e["event_type"], e.get("mode") or ""): e["definition"] for e in entries}


def _load_vllm_sampling_params(max_tokens=1024, greedy=False):
    """
    Load the sampling parameters for the vLLM model.

    The v6 model is decoded greedily, which is how it was evaluated. The older
    Qwen3 models are sampled, because greedy decoding sent them into repetition
    loops.
    """
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("vLLM is not installed. " + _install_message("vllm"))

    if greedy:
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)
    sampling_params = SamplingParams(
        temperature=0.5,       # Greedy decoding breaks Qwen
        top_p=0.8,             # Qwen3 non-thinking recommendation
        top_k=20,              # Qwen3 recommendation
        presence_penalty=1.5,  # Recommended for quantized models
        min_p=0.0,
        #guided_decoding=guided_decoding_params, # Optionally, set a JSON schema for contrained decoding
        max_tokens=max_tokens,
    )
    return sampling_params




def _install_message(backend: str) -> str:
    """What to install for `backend`, and the alternatives."""
    return (f"Install it with: {BACKEND_INSTALL_HINTS[backend]}. "
            f"On a CPU, backend='llamacpp' needs only: "
            f"{BACKEND_INSTALL_HINTS['llamacpp']}")


class AttributeModel:
    def __init__(self,
                 event_definitions_file=None,
                 silent=False, # whether to silence progress bars and logs
                 batch_size=8,
                 save_intermediate=False,
                 gpu=False,
                 base_path=None,
                 max_gpu_memory=0.8,
                 vllm_model=None,
                 backend: BackendType="auto",
                 llamacpp_url: str | None = None,
                 llamacpp_threads: int | None = None,
                 gguf_path: str | None = None,
                 model_name: str | None = None,
                 prompt_format: PromptFormat | None = None,
                 seed: int | None = None,
                 intermediate_dir: str | None = None,
                 ):
        """
        Initialize the attribute model

        Parameters
        ----------
        event_definitions_file : str, optional
            Your own event definitions. Which file format is expected depends on
            the model's prompt format:

            - "v6" (the default model): a JSON file in the format of
              assets/event_definitions_v6.json, a list of {"event_type",
              "mode", "definition"} entries. Its entries are added to the
              definitions the model was trained on, replacing any with the
              same event type and mode, so the file only needs the event types
              you are adding or rewording. A CSV is ignored under v6, with a
              warning.
            - "legacy" and "v5": a CSV in the format of
              assets/PLOVER_structured_codebook_updated.csv (the default).

            Under any format, a record that carries its own 'event_def' key is
            prompted with that instead.
        silent : bool, default=False
            Whether to silence progress bars and logs
        batch_size : int, default=8
            Batch size for processing
        save_intermediate : bool, default=False
            Write this step's output to a timestamped "*_attribute_output.jsonl"
            file, and any events with no extraction to "*_dropped_events.jsonl".
        gpu : bool, default=False
            Whether to use GPU
        base_path : str, optional
            Base path for loading files
        max_gpu_memory : float, default=0.8
            GPU memory utilization for vLLM
        vllm_model : vllm.LLM, optional
            Pre-initialized vLLM model to use
        backend : {"auto", "vllm", "llamacpp", "mlx", "transformers"}, default="auto"
            Which backend runs the model:

            - "vllm": Linux with an NVIDIA GPU (the `vllm` extra). Fastest.
            - "llamacpp": any CPU (the `llamacpp` extra). Runs the model's
              published GGUF file in this process, downloading it the first
              time, unless `llamacpp_url` (or NGEC_LLAMACPP_URL) points it at a
              running `llama-server`.
            - "mlx": a Mac with Apple Silicon (the `mlx` extra).
            - "transformers": deprecated. Still works, but on a CPU it took
              about three times as long per prompt as "llamacpp" (15 s against
              4.6 s on an i9-12900K) and twice the memory.
            - "auto": vllm if it is installed and there is a CUDA GPU, mlx on a
              Mac with Apple Silicon if it is installed, and llamacpp
              otherwise (see ngec.llm.choose_backend). The choice is logged.
        llamacpp_url : str, optional
            The URL of a running `llama-server`, for the llamacpp backend.
            Defaults to the NGEC_LLAMACPP_URL environment variable. With
            neither, the llamacpp backend runs the model in this process.
        llamacpp_threads : int, optional
            CPU threads for the in-process llamacpp backend. Defaults to the
            NGEC_LLAMACPP_THREADS environment variable, or else the number of
            performance cores, at most 8 (see ngec.llm.llamacpp.default_threads).
            Using every logical CPU is usually much slower.
        gguf_path : str, optional
            A local GGUF file for the in-process llamacpp backend to load.
            Defaults to the NGEC_ATTRIBUTE_GGUF environment variable, or else
            the published GGUF of `model_name`, downloaded from Hugging Face
            (see ngec.llm.llamacpp.KNOWN_GGUF_FILES). It must be a conversion
            of `model_name`, which still supplies the prompt format and the
            chat template.
        model_name : str, optional
            A Hugging Face model name or a path to a local model directory.
            Defaults to DEFAULT_MODEL, or to the NGEC_ATTRIBUTE_MODEL
            environment variable if that is set. Note that with a
            `llama-server`, the weights are whatever the server was started
            with -- this only selects the tokenizer there, so the two have to
            be kept in step by hand.
        prompt_format : {"legacy", "v5", "v6"}, optional
            The prompt format the model was trained on. Defaults to looking
            `model_name` up in KNOWN_PROMPT_FORMATS, then in the model's own
            `ngec.json` (see resolve_prompt_format). Only pass this for a model
            that has neither; a mismatch does not raise, it just makes the
            extractions worse.
        seed : int, optional
            Seed the sampler, making a run repeatable on one machine. The
            legacy and v5 models are sampled rather than decoded greedily
            (greedy decoding sent those Qwen3 models into repetition loops), so
            an unseeded run can return a different span -- or N/A instead of a
            span -- for the same document. The default v6 model is decoded
            greedily and needs no seed. Useful for tests
            and for reproducing a reported extraction; leave it unset otherwise.
            Currently honoured only by backends that go through an engine.
        intermediate_dir : str, optional
            The directory the ``save_intermediate`` files go in. Defaults to the
            current working directory.
        """
        self.silent=silent
        if backend == "auto":
            from .llm import choose_backend
            backend = choose_backend()
            logger.info(f"Attribute model backend: {backend} (chosen automatically)")
        self.backend = backend
        self.model_name = resolve_model_name(model_name)
        self.prompt_format: PromptFormat = (prompt_format
                                            or resolve_prompt_format(self.model_name))
        # The v5 models were evaluated with a 2048-token ceiling; the legacy one
        # has always run at 1024. A document with many events can hit the lower
        # limit, and a truncated response is dropped as unparseable JSON. The v6
        # model was evaluated at 768; 1024 leaves room without changing what
        # greedy decoding returns for any response that fit in 768.
        self.max_output_tokens = 2048 if self.prompt_format == "v5" else 1024
        # v6 is decoded greedily (that is how its numbers were measured); the
        # older Qwen3 models are sampled.
        self.greedy = self.prompt_format == "v6"
        generation_config = self._generation_config(seed)

        if gpu:
            self.device="cuda"
        else:
            self.device="cpu"
        if not self.silent:
            logger.info(f"Device: {self.device}")
            logger.info(f"Backend: {self.backend}")
            logger.info(f"Model: {self.model_name} (prompt format: {self.prompt_format})")

        # None until a backend has been ported to the engine interface; the
        # others still generate through call_llm_batch(). process() branches on
        # this, so it has to be set for every backend, not just the ported ones.
        self.engine: GenerationEngine | None = None

        # Load model based on backend
        if self.backend == "vllm":
            try:
                from vllm import LLM
            except ImportError:
                raise ImportError("The vllm backend needs vLLM, which is not "
                                  "installed. " + _install_message("vllm")) from None
            
            if not self.silent: 
                logger.debug("Loading vLLM model")
            if vllm_model:
                self.model = vllm_model
            else:
                # The v6 checkpoint is Qwen3.5's multimodal layout, because that
                # is the only Qwen3.5 architecture vLLM registers. NGEC only
                # sends text, so language_model_only tells vLLM not to build or
                # load the vision tower and not to profile its image encoder.
                text_only = ({"language_model_only": True}
                             if self.prompt_format == "v6" else {})
                self.model = LLM(model=self.model_name,
                                 enable_prefix_caching=True,
                                 max_model_len=8000,
                                 gpu_memory_utilization=max_gpu_memory,
                                 **text_only)
            self.sampling_params = _load_vllm_sampling_params(self.max_output_tokens,
                                                              greedy=self.greedy)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif self.backend == "transformers":
            # Logged whatever `silent` says: PloverCoder always passes
            # silent=True, and this is the one message a user of this backend
            # needs to see.
            logger.warning(
                "The transformers backend is deprecated. "
                "It still works, but on a CPU it is about three times slower "
                "than backend='llamacpp' and uses twice the memory. "
                "Use backend='llamacpp' on a CPU, 'vllm' on Linux with an "
                "NVIDIA GPU, or 'mlx' on a Mac with Apple Silicon -- or "
                "backend='auto' to pick among them.")
            from .llm.transformers import TransformersEngine
            self.engine = TransformersEngine(
                model_name=self.model_name,
                device=self.device,
                config=generation_config,
                silent=self.silent,
            )
            # Keep the attribute alive for make_prompt() and the demo; delete when
            # the last backend becomes an engine.
            self.tokenizer = self.engine.tokenizer

        elif self.backend == "mlx":
            try:
                from mlx_lm import load, generate
                from mlx_lm.sample_utils import make_sampler
            except ImportError:
                raise ImportError("The mlx backend needs mlx-lm, which is not "
                                  "installed. " + _install_message("mlx")) from None
            
            if not self.silent: 
                logger.debug("Loading MLX model")
            # MLX doesn't use device parameter the same way as PyTorch
            self.model, self.tokenizer = load(self.model_name)
            # Store the generate function and create sampler
            self.mlx_generate = generate
            self.sampler = make_sampler(
                temp=0.0 if self.greedy else 0.5,  # temperature; 0 is greedy
                top_p=0.8,          # nucleus sampling
                top_k=20,           # top-k sampling
                min_p=0.0,          # minimum probability
                min_tokens_to_keep=1,
            )
        elif self.backend == "llamacpp":
            # The fast path on a CPU: the model runs as an 8-bit GGUF file,
            # which cuts the weight bytes that dominate decoding. Runs in this
            # process through llama-cpp-python, unless a llama-server URL is
            # given, in which case it talks to that server over HTTP (how the
            # demo deployment runs; the server's prompt cache also reuses the
            # shared start of the prompts). See DEVELOPING.md.
            from .llm import llamacpp_server_url
            url = llamacpp_server_url(llamacpp_url)
            if url:
                from .llm.llamacpp import LlamaCppServerEngine
                self.engine = LlamaCppServerEngine(
                    model_name=self.model_name,
                    url=url,
                    config=generation_config,
                    silent=self.silent,
                )
            else:
                from .llm.llamacpp import LlamaCppLocalEngine
                self.engine = LlamaCppLocalEngine(
                    model_name=self.model_name,
                    gguf_path=gguf_path,
                    n_threads=llamacpp_threads,
                    config=generation_config,
                    silent=self.silent,
                )
            # Keep the attribute alive for make_prompt() and the demo; delete when
            # the last backend becomes an engine.
            self.tokenizer = self.engine.tokenizer
        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                "Must be 'auto', 'vllm', 'llamacpp', 'mlx', or 'transformers'"
            )

        self.batch_size=batch_size
        self.save_intermediate=save_intermediate
        self.intermediate_dir=intermediate_dir
        # The v6 system prompt depends on the event type, so it is built per
        # record in _build_conversation; the older formats use one for all.
        self.system_prompt = (_make_system_content_v5()
                              if self.prompt_format == "v5"
                              else _make_system_content_short())
        # A v6 model reads its definitions from JSON (see _load_v6_definitions);
        # the older formats read the codebook CSV.
        v6_json = (self.prompt_format == "v6" and event_definitions_file is not None
                   and str(event_definitions_file).lower().endswith(".json"))
        csv_file = event_definitions_file
        if csv_file is None or v6_json:
            csv_file = "PLOVER_structured_codebook_updated.csv"
        self.event_definitions = _load_event_definitions(csv_file, base_path)

        self.v6_definitions = {}
        if self.prompt_format == "v6":
            self.v6_definitions = _load_v6_definitions()
            if v6_json:
                custom = _load_v6_definitions(event_definitions_file)
                self.v6_definitions.update(custom)
                logger.info(f"Read {len(custom)} event definitions from {event_definitions_file}")
            elif event_definitions_file is not None:
                logger.warning(
                    f"event_definitions_file={event_definitions_file!r} is not used by "
                    f"{self.model_name}: its prompt format (v6) reads definitions from "
                    f"a JSON file in the format of assets/event_definitions_v6.json, "
                    f"not a CSV.")

    def _generation_config(self, seed=None):
        """Decoding settings for the engine backends, matched to the prompt format."""
        from .llm import GenerationConfig
        if self.greedy:
            return GenerationConfig(temperature=0.0, top_p=1.0, top_k=1, min_p=0.0,
                                    presence_penalty=0.0,
                                    max_tokens=self.max_output_tokens, seed=seed)
        return GenerationConfig(max_tokens=self.max_output_tokens, seed=seed)


    # TODO (customization): add an informative error when a *mode* is missing
    # from the definitions file, as is now done below for the event type.
    def _get_event_info(self, event):
        """
        Convert an event dict to a message for the model.

        The definition normally comes from the event definitions file, looked up
        by event type. A record may instead carry its own ``event_def`` (and
        optionally ``mode_def`` and ``extraction_notes``), in which case no
        lookup happens. That is the path for an event type outside the codebook:
        the model reads a definition rather than recognising a fixed list of
        labels, so a new event type needs a definition written for it, not a
        retrained model. See the "event types the model has never seen" section
        of the demo's attribute-extraction page.
        """
        mode_def = None
        extraction_notes = None
        doc = event['event_text']
        event_type = event['event_type']

        if event.get('event_def'):
            return (doc, event_type, event['event_def'],
                    event.get('mode_def') or None,
                    event.get('extraction_notes') or None)

        event_rows = self.event_definitions.loc[self.event_definitions['event'] == event_type]
        if len(event_rows) == 0:
            known = ", ".join(sorted(self.event_definitions['event'].unique()))
            raise KeyError(
                f"No definition for event type '{event_type}'. The definitions file "
                f"loaded by this AttributeModel contains: {known}. Either point "
                f"`event_definitions_file=` at a codebook that defines it, or give "
                f"the record its own 'event_def' key (with optional 'mode_def' and "
                f"'extraction_notes') and it will be used as-is."
            )
        event_def = event_rows['event_def'].values[0]
        # Get mode definition and extraction notes if they exist
        if 'event_mode' in event:
            if event['event_mode'] != "":
                if 'mode' in self.event_definitions.columns and 'mode_def' in self.event_definitions.columns:
                    mode_def = event_rows.loc[event_rows['mode'] == event['event_mode'], 'mode_def'].values[0]
                if 'extraction_notes' in self.event_definitions.columns:
                    extraction_notes = event_rows.loc[event_rows['mode'] == event['event_mode'], 'extraction_notes'].values[0]

        return doc, event_type, event_def, mode_def, extraction_notes
    
    def _make_user_message(self,
                           doc,
                           event,
                           event_def,
                           mode_def=None,
                           extraction_notes=None):
        """
        Logic to get the event/mode definitions for a given event type.

        # Example format:
        '## Event: **REQUEST**: All requests, demands, and orders. Requests, demands, and orders are less forceful than threats and potentially carry less serious repercussions
         
        ## Specific Sub-Event: Make a request for changes in policy, government, or institutions
         
        ## Special Instructions: NOTE: Protests (including protests making requests) are coded under a separate PROTEST category. Protest DO NOT fall under this category.'
        """

        if self.prompt_format == "v5":
            return self._make_user_message_v5(doc, event, event_def, mode_def,
                                              extraction_notes)

        user_message = f"### Document:\n\n{doc}\n\n"
        user_message += f"### Event: **{event}**: {event_def}\n"
        if mode_def:
            user_message += f"### Specific Sub-Event: **{mode_def}**\n"
        if extraction_notes:
            if not pd.isna(extraction_notes):
                user_message += f"### Special Instructions: {extraction_notes}\n"
        user_message += "Extract the attributes of the given event in JSON format."
        return user_message

    def _make_user_message_v5(self,
                              doc,
                              event,
                              event_def,
                              mode_def=None,
                              extraction_notes=None):
        """The user message the 2026 models were trained and evaluated with.

        Two differences from the legacy format matter, and both are easy to
        miss. The whole event definition — type, sub-event and special
        instructions — is a single inline string after `## Event Type:`, not
        three separate `###` sections; and there is no closing "Extract the
        attributes" instruction, because the system prompt carries it.

        The definition string is assembled to match the `event_def` field of the
        v5 training data:

            ## Event: **ACCUSE**: <definition> ## Specific Sub-Event: <mode>
            ## Special Instructions: <notes>
        """
        definition = f"## Event: **{event}**: {event_def}"
        if mode_def:
            definition += f" ## Specific Sub-Event: {mode_def}"
        if extraction_notes and not pd.isna(extraction_notes):
            definition += f" ## Special Instructions: {extraction_notes}"
        return f"## Document: {doc}\n\n## Event Type: {definition}"

    def _v6_definition(self, event) -> str:
        """The `## Event Type:` text for one record, in the v6 format.

        A record that carries its own ``event_def`` (an event type outside the
        codebook, or a rewritten definition) is rendered the way the training
        definitions are laid out: one ``##`` section per line. Otherwise the
        definition is the trained one for the record's event type and mode.
        """
        event_type = event['event_type']
        mode = event.get('event_mode') or ""
        if event.get('event_def'):
            definition = f"## Event: **{event_type}**: {event['event_def']}"
            if event.get('mode_def'):
                definition += f"\n## Specific Sub-Event: **{mode}**: {event['mode_def']}"
            notes = event.get('extraction_notes')
            if notes and not pd.isna(notes):
                definition += f"\n## Special Instructions: {notes}"
            return definition
        if (event_type, mode) in self.v6_definitions:
            return self.v6_definitions[(event_type, mode)]
        if (event_type, "") in self.v6_definitions:
            logger.warning(f"No trained definition for mode '{mode}' of {event_type}; "
                           f"using the definition of {event_type} as a whole.")
            return self.v6_definitions[(event_type, "")]
        known = ", ".join(sorted({t for t, _ in self.v6_definitions}))
        raise KeyError(
            f"No definition for event type '{event_type}'. This model was trained "
            f"on definitions of: {known}. For any other event type, give the record "
            f"its own 'event_def' key (with optional 'mode_def' and "
            f"'extraction_notes') and it will be used as-is.")

    def _build_conversation(self, event) -> Conversation:
        if self.prompt_format == "v6":
            # Runs of whitespace are collapsed, as they were in every training
            # document.
            doc = re.sub(r"\s+", " ", str(event['event_text'])).strip()
            user = (f"## Document: {doc}\n\n"
                    f"## Event Type: {self._v6_definition(event)}\n\n"
                    "Return the JSON list.")
            return [
                {"role": "system", "content": _make_system_content_v6(event['event_type'])},
                {"role": "user", "content": user},
            ]
        doc, event_type, event_def, mode_def, notes = self._get_event_info(event)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._make_user_message(
                doc, event_type, event_def, mode_def, notes)},
        ]

    def make_prompt(self, event):
        """Templated prompt string. Legacy backends consume this; engines take
        _build_conversation() and template internally."""
        return self.tokenizer.apply_chat_template(
            self._build_conversation(event),
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    def call_llm_batch(self, prompts):
        if type(prompts) is not list:
            prompts = [prompts]

        if self.backend == "vllm":
            # vLLM backend
            outputs = self.model.generate(prompts, sampling_params=self.sampling_params)
            responses = [i.outputs[0].text.strip() for i in outputs]
        elif self.backend == "transformers":
            # This backend generates through TransformersEngine now, so there is
            # no self.model here to call. process() never reaches this branch;
            # it exists to give a direct caller a real message instead of an
            # AttributeError on a half-migrated object.
            raise RuntimeError(
                "The transformers backend generates through TransformersEngine, "
                "not call_llm_batch(). Use process(), or "
                "self.engine.generate([self._build_conversation(event)])."
            )
        elif self.backend == "llamacpp":
            # This backend generates through an engine (LlamaCppLocalEngine
            # or LlamaCppServerEngine) now. process() never reaches this
            # branch; it exists to give a direct caller a real message instead
            # of an AttributeError on a half-migrated object.
            raise RuntimeError(
                "The llamacpp backend generates through self.engine, "
                "not call_llm_batch(). Use process(), or "
                "self.engine.generate([self._build_conversation(event)])."
            )
        elif self.backend == "mlx":
            # MLX backend
            responses = []
            for prompt in prompts:
                # Generate with MLX
                output = self.mlx_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.max_output_tokens,
                    sampler=self.sampler,
                    verbose=False,
                )
                # The output from mlx_lm.generate is a string
                responses.append(output.strip())
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # The same parsing as the engine backends: strip <think> blocks,
        # salvage the finished records of a truncated response, drop exact
        # duplicates, and turn spans into lists.
        json_responses = []
        failures = []
        for response in responses:
            events, failure = parse_response(response)
            json_responses.append(events)
            if failure:
                failures.append(failure)
                logger.debug(f"Parse failure ({failure}): {response!r}")
        if failures:
            reasons = ", ".join(f"{reason}: {count}" for reason, count
                                in Counter(failures).most_common())
            logger.info(f"Number of parse failures: {len(failures)} of "
                        f"{len(responses)} ({reasons})")
        return json_responses
                

    def process(self,
                event_list: list[AttributeModelInput] | list[dict[str, Any]]
                ) -> list[AttributeModelOutput] | list[dict[str, Any]]:
        """
        Given event records from the previous steps in the NGEC pipeline,
        run the QA model to identify the spans of text corresponding with
        each of the event attributes (e.g. ACTOR, RECIP, LOC, DATE.)

        Parameters
        --------
        event_list: list of event dicts, each with at least:
            - event_text: the document
            - event_type: the event type to extract, e.g. "PROTEST"
          and optionally:
            - event_mode: a mode of that type, or "" (the default) for the
              type as a whole
            - id: kept, with a "_<n>" suffix per extracted event
            - event_def (and mode_def, extraction_notes): a definition to
              prompt with instead of the one for event_type
          Other keys are passed through unchanged.

        Returns
        -----
        event_list: list of dicts (a NEW list, not the input)
          The model may extract zero, one, or several events from a single
          document. Each extracted event becomes its own record (via
          ``explode_events``) with a single 'attributes' dict:
            {'event_type': 'PROTEST',
             'anchor_quote': '...',
             'actor': ['a group of Hindu nationalists'],
             'recipient': ['Muslim shops'],
             'date': ['last week'],
             'location': ['Dehli']}
          Records for which the model extracted no event are dropped from the
          returned list (reported via a warning and written to a separate file),
          so the output never contains empty-attribute junk. Because records are
          exploded and dropped, the returned list is not the input list -- use
          the return value.
        """
        # Step 1: further lengthen the data to generate separate elements
        # for each attribute/question, so we have unique (ID, event_cat, attribute) 
        logger.debug("Starting attribute process")

        # Create a list of prompts
        if not self.silent: 
            print("Making prompts...")
        if self.engine is not None:
            conversations = [self._build_conversation(e)
                            for e in tqdm(event_list, desc="Making prompts", disable=self.silent)]
            # The string schema describes the legacy/v5 output. The v6 model
            # writes lists and was evaluated without a schema, so it gets none.
            schema = (ATTRIBUTE_SCHEMA
                      if self.engine.capabilities.schema and self.prompt_format != "v6"
                      else None)
            raw = self.engine.generate(conversations, schema=schema)
            final_attributes = []
            failures = []
            for text in raw:
                events, failure = parse_response(text)
                if failure:
                    failures.append(failure)
                    logger.debug(f"Parse failure ({failure}): {text!r}")
                final_attributes.append(events)
            # Reported in aggregate at INFO, matching what call_llm_batch logs
            # below: an unparseable response becomes a dropped event rather than
            # an error, so the rate is the only sign that anything is wrong.
            if failures:
                reasons = ", ".join(f"{reason}: {count}" for reason, count
                                    in Counter(failures).most_common())
                logger.info(f"Number of parse failures: {len(failures)} of "
                            f"{len(raw)} ({reasons})")
        else:
            prompts = [self.make_prompt(event) for event in tqdm(event_list, desc="Making prompts", disable=self.silent)]
            final_attributes = self.call_llm_batch(prompts)

        # Post-processing: every span attribute becomes a list of strings,
        # whether the model wrote "a; b" (legacy, v5) or ["a", "b"] (v6).
        # Redundant on the engine path -- parse_response has already done it --
        # but harmless, since normalizing a list of stripped strings changes
        # nothing. Delete it once the last backend generates through an engine.

        # Now, at the very end, put the results back into the event list.
        for n, i in enumerate(event_list):
            # split each attribute into a list (semicolon separated)
            attributes = final_attributes[n]
            # [{'actor': 'a group of Hindu nationalists; the VHP',
            #      'anchor_quote': 'A group of Hindu nationalists and the VHP rioted in '
            #                      'Dehli last week, burning Muslim shops.',
            #      'date': 'last week',
            #      'event_type': 'PROTEST:Violent riot',
            #      'location': 'Dehli',
            #      'recipient': 'Muslim shops'}]
            #i['attributes'] = final_attributes[n]
            if isinstance(attributes, dict):
                attributes = [attributes]
            attributes = [normalize_spans(sub_event) for sub_event in attributes
                          if isinstance(sub_event, dict)]
            # Temporarily store the full list of extracted sub-events; explode_events
            # (below) turns each into its own record with a single 'attributes' dict.
            event_list[n]['attributes'] = attributes

        # Lengthen the data so each extracted event is its own record, and set
        # aside records where the model found no event (attributes == []).
        event_list, dropped = explode_events(event_list)
        if dropped:
            self._report_dropped(dropped)

        if self.save_intermediate:
            write_intermediate(event_list, "attribute_output", self.intermediate_dir)

        return cast(list[AttributeModelOutput], event_list)

    def _report_dropped(self, dropped):
        """
        Report events the model produced no extraction for. These are kept OUT of
        the main output (people are bad at filtering downstream, so we don't emit
        empty-attribute junk), but we warn loudly about how many were dropped and
        their event-type distribution.

        The dropped records are also written to a JSONL file, but only under
        ``save_intermediate`` -- the same switch the other components use for
        their per-step debugging dumps. It used to be unconditional, which is
        fine for a one-off corpus run and wrong for anything long-lived: an
        interactive app coding a document per visitor accumulated one timestamped
        file per interaction in its working directory.
        """
        distribution = Counter(event.get('event_type') for event in dropped)
        dist_str = ", ".join(f"{event_type}: {count}"
                             for event_type, count in distribution.most_common())
        message = (f"Dropped {len(dropped)} event(s) with no extracted attributes and "
                   f"excluded them from the main output. By event type: {dist_str}.")

        if self.save_intermediate:
            path = write_intermediate(dropped, "dropped_events", self.intermediate_dir)
            message += f" The dropped events were written to {path}."
        else:
            message += (" Pass save_intermediate=True to write them to a "
                        "*_dropped_events.jsonl file for inspection.")

        logger.warning(message)


if __name__ == "__main__":
    # add debug logging
    logging.basicConfig(level=logging.DEBUG)

    data = [
        {"event_text": "A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.",
        "id": 123,
        "_doc_position": 0,
        "event_type": "PROTEST",
        "event_mode": "riot"},
        {"event_text": "Turkish forces battled with YPG militants in Syria.",
        "id": 456,
        "_doc_position": 1,
        "event_type": "ASSAULT",
        "event_mode": ""},
        {"event_text": "Turkish forces and Turkish-backed militias battled with YPG militants in Syria.",
        "id": 789,
        "_doc_position": 2,
        "event_type": "ASSAULT",
        "event_mode": ""}
    ]

    # Example: Use vLLM backend (default)
    am = AttributeModel(silent=False, gpu=True, backend="vllm")
    # Or use transformers backend:
    # am = AttributeModel(silent=False, gpu=True, backend="transformers")

    prompt = am.make_prompt(data[0])
    print(prompt)
    output = am.call_llm_batch(prompt)

    all_prompts = [am.make_prompt(event) for event in data]
    all_attributes = am.call_llm_batch(all_prompts)

    all_outputs = am.process(data)

    # clear the cuda cache
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()


    # all_outputs[0]  (one record per extracted event; 'attributes' is a dict,
    # and the id has an appended sub-event index)
    #{'event_text': 'A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.',
    # 'id': '123_0',
    # '_doc_position': 0,
    # 'event_type': 'PROTEST',
    # 'event_mode': 'riot',
    # 'attributes': {'event_type': 'PROTEST: Violent riot',
    #                'anchor_quote': 'A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.',
    #                'actor': ['a group of Hindu nationalists'],
    #                'recipient': ['Muslim shops'],
    #                'date': ['last week'],
    #                'location': ['Dehli']}}
