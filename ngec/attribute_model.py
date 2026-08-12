import os
import pandas as pd
import time
import jsonlines
import re
import json
import logging

from collections import Counter
from importlib import resources
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any, cast, Literal, TypedDict, NotRequired

from .attributes.schema import parse_response
from .llm.base import Conversation, GenerationEngine
from .utilities import explode_events

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

BackendType = Literal["vllm", "transformers", "mlx", "llamacpp"]

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
PromptFormat = Literal["legacy", "v5"]

# ahalt/qwen3-event-extraction-exp5.1 is the 2026 retraining (see
# setup/hf_release/) — ~18pp better than the original ahalt/event-attribute-extractor
# on actor/location exact match. It replaced the original as the default once
# uploaded; the original stays published under its own name since it is still a
# valid (if worse) model and may be referenced elsewhere by that name.
DEFAULT_MODEL = "ahalt/qwen3-event-extraction-exp5.1"

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
}


def resolve_prompt_format(model_name: str) -> PromptFormat:
    """The prompt format a model was trained on, by name or directory name."""
    if model_name in KNOWN_PROMPT_FORMATS:
        return KNOWN_PROMPT_FORMATS[model_name]
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
        # Use importlib.resources to access package data
        try:
            with resources.files("NGEC").joinpath("assets", def_file).open() as f:
                event_definitions = pd.read_csv(f)
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback to file-based approach for development
            current_dir = os.path.dirname(__file__)
            file_path = os.path.join(current_dir, "assets", def_file)
            event_definitions = pd.read_csv(file_path)
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

def _load_vllm_sampling_params(max_tokens=1024):
    """
    Load the sampling parameters for the vLLM model.
    """
    try: 
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("vLLM is not installed. Please install it or use backend='transformers'")
    
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
                 backend: BackendType="vllm",
                 llamacpp_url: str | None = None,
                 model_name: str | None = None,
                 prompt_format: PromptFormat | None = None,
                 seed: int | None = None
                 ):
        """
        Initialize the attribute model

        Parameters
        ----------
        event_definitions_file : str, optional
            Path to event definitions CSV file
        silent : bool, default=False
            Whether to silence progress bars and logs
        batch_size : int, default=8
            Batch size for processing
        save_intermediate : bool, default=False
            Whether to save intermediate results
        gpu : bool, default=False
            Whether to use GPU
        base_path : str, optional
            Base path for loading files
        max_gpu_memory : float, default=0.8
            GPU memory utilization for vLLM
        vllm_model : vllm.LLM, optional
            Pre-initialized vLLM model to use
        backend: BackendType="vllm"
            Which backend to use: "vllm", "transformers", "mlx", or "llamacpp"
        model_name : str, optional
            A Hugging Face model name or a path to a local model directory.
            Defaults to DEFAULT_MODEL, or to the NGEC_ATTRIBUTE_MODEL
            environment variable if that is set. Note that the llamacpp backend
            loads its weights from whatever `llama-server` was started with —
            this only selects the tokenizer there, so the two have to be kept in
            step by hand.
        prompt_format : {"legacy", "v5"}, optional
            The prompt format the model was trained on. Defaults to looking
            `model_name` up in KNOWN_PROMPT_FORMATS. Only pass this for a model
            that is not listed there; a mismatch does not raise, it just makes
            the extractions worse.
        seed : int, optional
            Seed the sampler, making a run repeatable on one machine. Decoding
            samples rather than being greedy (greedy decoding sends Qwen into
            repetition loops), so an unseeded run can return a different span --
            or N/A instead of a span -- for the same document. Useful for tests
            and for reproducing a reported extraction; leave it unset otherwise.
            Currently honoured only by backends that go through an engine.
        """
        self.silent=silent
        self.backend = backend
        self.model_name = (model_name
                           or os.environ.get("NGEC_ATTRIBUTE_MODEL")
                           or DEFAULT_MODEL)
        self.prompt_format: PromptFormat = (prompt_format
                                            or resolve_prompt_format(self.model_name))
        # The v5 models were evaluated with a 2048-token ceiling; the legacy one
        # has always run at 1024. A document with many events can hit the lower
        # limit, and a truncated response is dropped as unparseable JSON.
        self.max_output_tokens = 2048 if self.prompt_format == "v5" else 1024

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
                if not self.silent: 
                    logger.error("vLLM not available. Use another backend.")
                raise ImportError("vLLM is not installed. Please install it or use backend='transformers'")
            
            if not self.silent: 
                logger.debug("Loading vLLM model")
            if vllm_model:
                self.model = vllm_model
            else:
                self.model = LLM(model=self.model_name,
                                 enable_prefix_caching=True,
                                 max_model_len=8000,
                                 gpu_memory_utilization=max_gpu_memory)
            self.sampling_params = _load_vllm_sampling_params(self.max_output_tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif self.backend == "transformers":
            from .llm import GenerationConfig
            from .llm.transformers import TransformersEngine
            self.engine = TransformersEngine(
                model_name=self.model_name,
                device=self.device,
                config=GenerationConfig(max_tokens=self.max_output_tokens,
                                        seed=seed),
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
                raise ImportError("mlx_lm is not installed. Please install it or use another backend.")
            
            if not self.silent: 
                logger.debug("Loading MLX model")
            # MLX doesn't use device parameter the same way as PyTorch
            self.model, self.tokenizer = load(self.model_name)
            # Store the generate function and create sampler
            self.mlx_generate = generate
            self.sampler = make_sampler(
                temp=0.5,           # temperature
                top_p=0.8,          # nucleus sampling
                top_k=20,           # top-k sampling
                min_p=0.0,          # minimum probability
                min_tokens_to_keep=1,
            )
        elif self.backend == "llamacpp":
            # Talks to a running `llama-server` over HTTP rather than loading a
            # model in-process. This is the fast path on CPU: the model is
            # served quantized, which cuts the weight bytes that dominate
            # decode. On an AVX2 desktop, Q8_0 measured ~4x faster per call than
            # the transformers backend in float32, and the server's prompt cache
            # also reuses the shared document prefix across the several event
            # types extracted from one document.
            #
            # Start the server separately, e.g.
            #   llama-server -m attr-q8.gguf --port 8080 -c 8192
            # and point NGEC_LLAMACPP_URL at it. See DEVELOPING.md.
            self.llamacpp_url = (llamacpp_url
                                 or os.environ.get("NGEC_LLAMACPP_URL",
                                                   "http://127.0.0.1:8080"))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if not self.silent:
                logger.info(f"Using llama-server at {self.llamacpp_url}")
        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                "Must be 'vllm', 'transformers', 'mlx', or 'llamacpp'"
            )

        self.batch_size=batch_size
        self.save_intermediate=save_intermediate
        self.system_prompt = (_make_system_content_v5()
                              if self.prompt_format == "v5"
                              else _make_system_content_short())
        if event_definitions_file is None:
            event_definitions_file = "PLOVER_structured_codebook_updated.csv"
        self.event_definitions = _load_event_definitions(event_definitions_file, base_path)


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

    def _build_conversation(self, event) -> Conversation:
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
            # One request per prompt to llama-server. `cache_prompt` lets the
            # server reuse the KV cache for the shared document prefix, which is
            # most of the prompt when several event types are extracted from the
            # same document.
            import urllib.error
            import urllib.request

            responses = []
            for prompt in prompts:
                body = json.dumps({
                    "prompt": prompt,
                    "n_predict": self.max_output_tokens,
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "top_k": 20,
                    "cache_prompt": True,
                }).encode()
                request = urllib.request.Request(
                    f"{self.llamacpp_url}/completion",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=300) as resp:
                        payload = json.loads(resp.read())
                    responses.append(payload.get("content", "").strip())
                except (urllib.error.URLError, OSError, TimeoutError) as e:
                    logger.error(
                        f"llama-server at {self.llamacpp_url} did not respond: {e}. "
                        "Is it running?"
                    )
                    responses.append("")
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

        json_responses = []
        error_responses = []
        for response in responses:
            response = re.sub("<think>.*?</think>", "", response, flags=re.DOTALL)  # Remove <think> tags and content
            try:
                json_responses.append(json.loads(response))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                json_responses.append([])  # Append empty list on error
                error_responses.append(response)
        logger.info(f"Number of JSON decode errors: {len(error_responses)}")
        logger.debug(f"Error responses: {error_responses}")
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
        event_list: list of event dicts. 
          At a minimum, it should entries the following keys:
            - event_text
            - id (id for the event)
            - _doc_position (needed to link back to the nlped list)
            - event_type
            - mode
        doc_list: list of spaCy NLP docs
        expand: bool
          Expand the QA-returned answer to include appositives or compound words?
        show_progress: bool
            If True, show a tqdm progress bar.

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
            raw = self.engine.generate(conversations)
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

        # Post-processing (split the ; separated attributes into lists).
        # Redundant on the engine path -- parse_response has already split these
        # -- but harmless, since the loop below re-strips a list unchanged.
        # Delete it once the last backend generates through an engine.

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
            for sub_event in attributes:
                for key, value in sub_event.items():
                    if key in ['actor', 'date', 'recipient', 'location']:
                        # If the value is a string, split it by semicolon and strip whitespace
                        if isinstance(value, str):
                            value = [v.strip() for v in value.split(';')]
                        # If the value is a list, ensure all items are stripped of whitespace
                        elif isinstance(value, list):
                            value = [v.strip() for v in value]
                        else:
                            continue
                        # Update the sub-event with the cleaned value
                        sub_event[key] = value
            # Temporarily store the full list of extracted sub-events; explode_events
            # (below) turns each into its own record with a single 'attributes' dict.
            event_list[n]['attributes'] = attributes

        # Lengthen the data so each extracted event is its own record, and set
        # aside records where the model found no event (attributes == []).
        event_list, dropped = explode_events(event_list)
        if dropped:
            self._report_dropped(dropped)

        if self.save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_attribute_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)

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
            fn = time.strftime("%Y_%m_%d-%H%M%S") + "_dropped_events.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(dropped)
            message += f" The dropped events were written to {os.path.abspath(fn)}."
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
