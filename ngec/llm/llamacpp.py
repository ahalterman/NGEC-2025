"""The llama.cpp backend, in two forms.

- `LlamaCppLocalEngine` runs the model inside this Python process through the
  `llama-cpp-python` package (the `llamacpp` extra). This is what
  `backend="llamacpp"` uses unless a server URL is given, and it is the
  recommended way to run NGEC on a CPU: it needs no separate program, and it
  downloads the published GGUF file of the attribute model from Hugging Face
  the first time it is used.
- `LlamaCppServerEngine` talks to an already-running `llama-server` over HTTP.
  It is used when a URL is given (`llamacpp_url=` or NGEC_LLAMACPP_URL), which
  is how the demo deployment runs. This module never builds, quantizes, or
  launches a server. See DEVELOPING.md for local dev setup and
  demo/deploy/README.md for the systemd deployment.

Both render the prompt in Python with the attribute model's own chat template
(through its Hugging Face tokenizer, with thinking turned off), exactly as the
transformers and vllm backends do, so all four send the model the same text.
"""
import json
import logging
import os
import sys
import urllib.error
import urllib.request
import weakref

from transformers import AutoTokenizer

from .base import Conversation, EngineCapabilities, GenerationConfig

logger = logging.getLogger(__name__)


# The GGUF files published for attribute models, by the Hugging Face model they
# were converted from: (repository, file name). The in-process engine downloads
# the file from here when it is not given a path. The Q8_0 file of the default
# model scored 74.3 mean F1 on a 100-document VOA subset against 73.6 for the
# unquantized model; see setup/hf_release/gguf_model_card.md.
KNOWN_GGUF_FILES = {
    "ahalt/qwen3.5-event-extraction-0.8b": (
        "ahalt/qwen3.5-event-extraction-0.8b-GGUF",
        "qwen3.5-event-extraction-0.8b-Q8_0.gguf",
    ),
}

# The most threads the in-process engine uses unless told otherwise. Writing
# out each token reads every weight once, so past a handful of cores decoding
# waits on memory rather than on arithmetic, and more threads only add
# coordination. On an i9-12900K (8 performance cores, 16 cores in all), a
# prompt took 4.6 s on 8 threads, 7.2 s on 4 and 6.4 s on 16.
MAX_DEFAULT_THREADS = 8

LLAMACPP_INSTALL_HINT = (
    'Install it with:\n\n'
    '    pip install "ngec[llamacpp]" '
    '--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n\n'
    '(or, in a checkout of the NGEC repository, `uv sync --extra llamacpp`). '
    'The extra index has ready-built CPU packages; without it pip compiles '
    'llama.cpp from source, which needs a C++ compiler and CMake and takes '
    'several minutes.'
)


def gguf_for_model(model_name: str) -> tuple[str, str] | None:
    """The (repository, file name) of the published GGUF for `model_name`, or
    None if there is none."""
    return KNOWN_GGUF_FILES.get(model_name)


def find_gguf(model_name: str, gguf_path: str | None = None) -> str:
    """The local path of the GGUF file to run `model_name` with.

    In order: `gguf_path` if given, then the NGEC_ATTRIBUTE_GGUF environment
    variable, then the published GGUF of `model_name` (see KNOWN_GGUF_FILES),
    downloaded into the Hugging Face cache if it is not there yet.
    """
    gguf_path = gguf_path or os.environ.get("NGEC_ATTRIBUTE_GGUF")
    if gguf_path:
        path = os.path.expanduser(gguf_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"GGUF file not found: {path}")
        return path

    published = gguf_for_model(model_name)
    if published is None:
        known = ", ".join(KNOWN_GGUF_FILES)
        raise ValueError(
            f"There is no published GGUF file for {model_name}, which the "
            f"in-process llama.cpp backend needs (it has one for: {known}). "
            "Convert the model with llama.cpp's convert_hf_to_gguf.py and pass "
            "the file as gguf_path= (or set NGEC_ATTRIBUTE_GGUF), or use "
            "another backend.")

    from huggingface_hub import hf_hub_download

    repo_id, filename = published
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _linux_core_count() -> int | None:
    """Performance cores this process may run on, from Linux's sysfs.

    Counts physical cores, not hardware threads (two threads share one core's
    arithmetic units). On a CPU with both performance and efficiency cores
    (Intel since 12th generation), only the performance cores are counted:
    spreading the work onto the slower cores made every step wait for them.
    """
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None
    try:
        # Present only on Intel hybrid CPUs; lists the performance cores.
        with open("/sys/devices/cpu_core/cpus") as f:
            performance = _parse_cpu_list(f.read())
        allowed = (allowed & performance) or allowed
    except (OSError, ValueError):
        pass

    cores = set()
    for cpu in allowed:
        topology = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        try:
            with open(f"{topology}/physical_package_id") as f:
                package = f.read().strip()
            with open(f"{topology}/core_id") as f:
                core = f.read().strip()
        except OSError:
            return None
        cores.add((package, core))
    return len(cores) or None


def _parse_cpu_list(text: str) -> set[int]:
    """"0-3,8" -> {0, 1, 2, 3, 8}"""
    cpus = set()
    for part in text.strip().split(","):
        if "-" in part:
            start, end = part.split("-")
            cpus.update(range(int(start), int(end) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def _mac_core_count() -> int | None:
    """Performance cores on a Mac (all physical cores on an Intel Mac)."""
    import subprocess
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                 text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return int(out.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def default_threads() -> int:
    """How many threads the in-process engine uses unless told otherwise.

    The NGEC_LLAMACPP_THREADS environment variable if set; otherwise the number
    of performance cores, capped at MAX_DEFAULT_THREADS. Where the cores cannot
    be counted, half the logical CPUs, since most CPUs run two threads per
    core.
    """
    setting = os.environ.get("NGEC_LLAMACPP_THREADS", "").strip()
    if setting:
        return max(1, int(setting))
    if sys.platform.startswith("linux"):
        cores = _linux_core_count()
    elif sys.platform == "darwin":
        cores = _mac_core_count()
    else:
        cores = None
    if cores is None:
        cores = (os.cpu_count() or 2) // 2
    return max(1, min(MAX_DEFAULT_THREADS, cores))


class LlamaCppLocalEngine:
    """Runs a GGUF model in this process with `llama-cpp-python`.

    The prompt is rendered exactly as the other backends render it (the model's
    chat template through its Hugging Face tokenizer, thinking off) and sent as
    text, as LlamaCppServerEngine sends it to `llama-server`. Decoding follows
    the GenerationConfig: the default model is decoded greedily.

    Unlike llama-server, this engine does not reuse the computation for a
    prompt's shared beginning from one call to the next. The Qwen3.5 model is
    part recurrent, and llama-cpp-python cannot rewind a recurrent state to a
    shared prefix, so it processes every prompt from the start. That is always
    correct. Side by side on ten VOA stories with 8 threads, llama-server took
    4.2 s per prompt when nothing was cached and this engine 4.7 s.

    One conversation at a time: capabilities.batching is False.
    """

    # schema=False: the v6 model is run without a JSON schema anyway, and a
    # grammar would only matter for an older model given as a GGUF of its own.
    capabilities = EngineCapabilities(schema=False, batching=False)

    def __init__(self, model_name: str, *, gguf_path: str | None = None,
                 n_threads: int | None = None, n_ctx: int = 8192,
                 config: GenerationConfig | None = None, silent: bool = False):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "The llamacpp backend runs the model with the llama-cpp-python "
                "package, which is not installed. " + LLAMACPP_INSTALL_HINT
                + " To use a running llama-server instead, pass llamacpp_url= "
                "or set NGEC_LLAMACPP_URL.") from None

        self.config = config or GenerationConfig()
        self.gguf_path = find_gguf(model_name, gguf_path)
        self.n_threads = n_threads or default_threads()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # The same context length llama-server is started with (-c 8192).
        # verbose=False keeps llama.cpp's loading report off the terminal.
        self.llm = Llama(model_path=self.gguf_path,
                         n_ctx=n_ctx,
                         n_threads=self.n_threads,
                         n_threads_batch=self.n_threads,
                         verbose=False)
        # Free the model when this engine is garbage-collected or, at the
        # latest, when Python exits. Left to llama-cpp-python's own __del__, the
        # freeing can run after Python has begun tearing down the modules it
        # needs, and prints a TypeError traceback on every exit.
        self._free_model = weakref.finalize(self, self.llm.close)
        # Per-call timings, like LlamaCppServerEngine's: prompt tokens and
        # generated tokens for each response in the last generate() call.
        self.last_timings: list[dict] = []
        if not silent:
            logger.info(f"Running {self.gguf_path} with llama.cpp in this "
                        f"process, {self.n_threads} threads")

    def generate(self, conversations: list[Conversation], *,
                 schema: dict | None = None) -> list[str]:
        # schema is ignored -- capabilities.schema is False, so the caller
        # falls back to prompt-instructed JSON plus salvage parsing.
        self.last_timings = []
        responses = []
        for conversation in conversations:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            sampling = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "min_p": self.config.min_p,
                "presence_penalty": self.config.presence_penalty,
                # llama-cpp-python's default is 1.0 today but was 1.1 in older
                # releases; set so that no version penalizes repeated tokens.
                "repeat_penalty": 1.0,
            }
            if self.config.seed is not None:
                sampling["seed"] = self.config.seed
            try:
                output = self.llm.create_completion(
                    prompt,
                    max_tokens=self.config.max_tokens,
                    # The end-of-turn token. Generation also stops at the
                    # model's end-of-sequence tokens without being told.
                    stop=["<|im_end|>"],
                    **sampling,
                )
            except ValueError as e:
                # llama-cpp-python raises ValueError for a prompt longer than
                # the context. Like a failed server request, this becomes an
                # empty response, and the event is dropped and reported.
                logger.error(f"llama.cpp could not process a prompt: {e}")
                responses.append("")
                continue
            usage = output.get("usage") or {}
            self.last_timings.append({"prompt_n": usage.get("prompt_tokens"),
                                      "predicted_n": usage.get("completion_tokens")})
            responses.append(output["choices"][0]["text"].strip())
        return responses

    def close(self) -> None:
        # Frees the model's memory now rather than whenever the object is
        # garbage-collected. Safe to call twice.
        if getattr(self, "_free_model", None) is not None:
            self._free_model()
        self.llm = None
        self.tokenizer = None


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

    No shared base class with the in-process LlamaCppLocalEngine:
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
        # The server's per-request timings from the last generate() call; see
        # generate(). Empty until the first call.
        self.last_timings: list[dict] = []
        if not silent:
            logger.info(f"Using llama-server at {self.url}")

    def generate(self, conversations: list[Conversation], *,
                 schema: dict | None = None) -> list[str]:
        """Generate one response per conversation.

        Also leaves `self.last_timings` holding the server's own `timings`
        block for each response in this call (`prompt_n`, `prompt_ms`,
        `predicted_n`, `predicted_ms`, plus `tokens_cached`), which is the only
        way to see the prefill/decode split from the client side. Reset on
        every call; a failed request contributes nothing.
        """
        self.last_timings = []
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
                timings = dict(payload.get("timings") or {})
                timings["tokens_cached"] = payload.get("tokens_cached")
                self.last_timings.append(timings)
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