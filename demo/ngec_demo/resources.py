"""Loading and caching the pipeline's models and services.

Everything here is cached with `st.cache_resource`, so the models are loaded
once per server process rather than once per page view. Loading the attribute
model, the two spaCy models and the sentence-transformers encoder takes roughly
ten seconds; doing it per interaction would make the demo unusable.

Each loader returns `None` rather than raising when its dependency is missing,
and `health()` reports what is up. That way a page whose Elasticsearch index is
down degrades to a message instead of a stack trace — a visiting reviewer
hitting a traceback is worse than a page that says what is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass

import streamlit as st

logger = logging.getLogger(__name__)

# `llamacpp` is the default because the demo is destined for a CPU-only server.
# It talks to a running llama-server holding a quantized copy of the attribute
# model: ~5x faster than transformers on CPU, at the cost of a second service to
# supervise and some quantization drift. Developing on a GPU box, set
# NGEC_DEMO_BACKEND=vllm — it is roughly 40x faster again, and it is what
# `build_example_cache.py` should be run with.
#
# The demo says which of the two it is running, on the setup-cost page and in
# the health panel. A visitor should not have to guess whether "ten seconds a
# document" is a property of the pipeline or of the machine it is on.
BACKEND = os.environ.get("NGEC_DEMO_BACKEND", "llamacpp")
LLAMACPP_URL = os.environ.get("NGEC_LLAMACPP_URL", "http://127.0.0.1:8080")

# vllm has no usable CPU path here (the installed build is a CUDA one), so the
# backend choice implies the device unless NGEC_DEMO_GPU says otherwise.
_gpu_env = os.environ.get("NGEC_DEMO_GPU", "").lower()
USE_GPU = _gpu_env in ("1", "true", "yes") or (_gpu_env == "" and BACKEND == "vllm")

# vllm reserves this fraction of the card up front, for weights plus KV cache.
# The package default is 0.8, which suits a corpus run where a big KV cache buys
# throughput. Here it is wrong twice over: the model is 0.6B (~1.2 GB in fp16)
# and the demo codes one document at a time, so almost all of that reservation
# is idle — and it starves the sentence encoder that steps 1 and 4 put on the
# same card, which then fails with an out-of-memory error that does not name
# vllm as its cause. 0.2 of a 24 GB card is ~4.7 GB, ample for this model.
GPU_MEMORY = float(os.environ.get("NGEC_DEMO_GPU_MEMORY", "0.2"))

# The attribute model. `AttributeModel`'s own default is the published exp5.1
# retraining (`ahalt/qwen3-event-extraction-exp5.1`, see setup/hf_release/), so
# leaving ATTRIBUTE_MODEL unset (None, below) already gets the current model. On
# this training box, prefer the local checkpoint directory when it is present so
# the demo doesn't re-download weights already on disk — it is the same weights,
# not a different model. Set NGEC_ATTRIBUTE_MODEL to override either.
#
# Note this only selects weights for the vllm and transformers backends. Under
# llamacpp the weights are whatever `llama-server` was started with; the name
# here just picks the tokenizer and, through it, the prompt format. Point the
# server at the matching GGUF (see demo/deploy/README.md).
LOCAL_ATTRIBUTE_MODEL = os.path.expanduser(
    "~/projects/train_NGEC_2026/qwen3-event-extraction-exp5.1")
ATTRIBUTE_MODEL = os.environ.get("NGEC_ATTRIBUTE_MODEL") or (
    LOCAL_ATTRIBUTE_MODEL if os.path.isdir(LOCAL_ATTRIBUTE_MODEL) else None)

# Set when a loader fails, so the health panel can report a model that did not
# load without having to load it a second time to find out.
_load_errors: dict[str, str] = {}


@dataclass
class Health:
    """One dependency's status.

    `blocks` names what stops working when this dependency is down, and marks it
    as one the app should complain about at startup rather than only in the
    sidebar. `fix` is the command that brings it back, for whoever is running
    the app. Leaving `blocks` empty means the demo works without it.
    """

    ok: bool
    detail: str
    blocks: str = ""
    fix: str = ""


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@st.cache_resource(show_spinner=False)
def get_es():
    """Elasticsearch client for the wiki and geonames indices, or None."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    try:
        from ngec.es_client import setup_es_client

        host = _env("ES_HOST", "localhost")
        port = int(_env("ES_PORT", "9200"))
        user = _env("ES_USER")
        password = _env("ES_PASSWORD")

        kwargs = {}
        if user and password:
            kwargs["http_auth"] = (user, password)

        client = setup_es_client(hosts=[host], port=port, **kwargs)
        client.info()  # force a real connection so a dead ES fails here, not later
        return client
    except Exception as exc:  # noqa: BLE001 - any failure means "no ES"
        logger.warning("Elasticsearch unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_nlp():
    """The shared spaCy pipeline (en_core_web_trf plus the token_tensors pipe)."""
    from ngec.utilities import load_nlp

    return load_nlp()


@st.cache_resource(show_spinner=False)
def get_classifier():
    """Event type and mode classifiers, plus the DemoModelWarning they emit.

    The warning is captured rather than silenced: that these are demonstration
    models and not the ones behind POLECAT is something the demo should say out
    loud, and the classifier itself is the authority on that text.
    """
    from ngec.classifiers.plover_sklearn import PloverSklearnClassifier

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PloverSklearnClassifier()
        messages = [str(w.message) for w in caught]

    return model, messages


@st.cache_resource(show_spinner=False)
def get_geolocation():
    from ngec.geolocation import GeolocationModel

    es = get_es()
    if es is None:
        return None
    try:
        return GeolocationModel(nlp=get_nlp(), es_client=es, quiet=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Geolocation model unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_attribute_model():
    from ngec import AttributeModel

    kwargs = {"silent": True, "gpu": USE_GPU, "backend": BACKEND}
    if BACKEND == "llamacpp":
        kwargs["llamacpp_url"] = LLAMACPP_URL
    if BACKEND == "vllm":
        kwargs["max_gpu_memory"] = GPU_MEMORY
    if ATTRIBUTE_MODEL:
        kwargs["model_name"] = ATTRIBUTE_MODEL
    try:
        model = AttributeModel(**kwargs)
        _load_errors.pop("attribute_model", None)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("Attribute model unavailable: %s", exc)
        _load_errors["attribute_model"] = f"{type(exc).__name__}: {exc}"
        return None


@st.cache_resource(show_spinner=False)
def get_actor_resolver():
    from ngec import ActorResolver

    es = get_es()
    if es is None:
        return None
    try:
        return ActorResolver(spacy_model=get_nlp(), es_client=es, gpu=USE_GPU)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Actor resolver unavailable: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_formatter():
    from ngec import Formatter

    return Formatter(quiet=True)


ES_FIX = (
    "Elasticsearch runs in a Docker container here. Find it with `docker ps -a` "
    "and start it with `docker start <container>`; it needs a minute or so before "
    "it answers on port 9200."
)
# attr-exp5.1-q8.gguf, not attr-q8.gguf: the latter was converted from the
# original `ahalt/event-attribute-extractor` checkpoint and takes the legacy
# prompt format, so serving it while the app prompts for exp5.1 is the silent
# mismatch `_llamacpp_model_health` exists to catch. Keep this in step with
# LOCAL_ATTRIBUTE_MODEL above and with demo/deploy/ngec-llama-server.service.
LLAMACPP_FIX = (
    "cd ~/ngec-llamacpp && LD_LIBRARY_PATH=$HOME/ngec-llamacpp/bin ./bin/llama-server "
    "-m attr-exp5.1-q8.gguf --port 8080 -c 8192 -t 16 --host 127.0.0.1 &"
)


def _gpu_detail() -> str:
    """"vllm on <GPU name>", or a plain description if torch cannot see one."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"vllm on {name} ({total:.0f} GB)"
        return "vllm configured, but no GPU is visible to torch"
    except Exception:  # noqa: BLE001
        return "vllm" + (" (GPU)" if USE_GPU else " (CPU)")


def attribute_model_label() -> str:
    """Which attribute model is in use, in a form worth showing a visitor.

    A reviewer comparing the demo's output against the paper's numbers needs to
    know which checkpoint produced what they are looking at — this mostly
    matters if NGEC_ATTRIBUTE_MODEL has been pointed at something other than the
    current default, e.g. the old `event-attribute-extractor` for a baseline
    comparison.
    """
    if not ATTRIBUTE_MODEL:
        # Imported here rather than at module scope: ngec.attribute_model pulls
        # in torch and transformers, and this module is imported to draw a page.
        from ngec.attribute_model import DEFAULT_MODEL

        return f"{DEFAULT_MODEL} (published)"
    name = os.path.basename(ATTRIBUTE_MODEL.rstrip("/"))
    if ATTRIBUTE_MODEL == LOCAL_ATTRIBUTE_MODEL:
        return f"{name} — local checkpoint (same weights as the published model)"
    return name


def _llamacpp_model_health(blocks: str) -> Health:
    """Check that llama-server is serving the model the Python side expects.

    Under this backend the weights come from whatever GGUF `llama-server` was
    started with, while the prompt format is chosen from the model name the
    Python side was configured with. Nothing forces those to agree, and a
    mismatch is silent: the pipeline runs, returns valid JSON, and the
    extractions are quietly worse. So compare the two names and say so when they
    diverge. The comparison is deliberately loose — the GGUF file is named by
    hand — and an unrecognisable name is reported rather than treated as a
    failure.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{LLAMACPP_URL}/v1/models", timeout=3) as resp:
            models = json.loads(resp.read()).get("models") or []
        served = str(models[0].get("name", "")) if models else ""
    except Exception as exc:  # noqa: BLE001
        return Health(True, f"llama.cpp on CPU, at {LLAMACPP_URL} "
                            f"(could not read the served model: {str(exc)[:40]})")

    if not served:
        return Health(True, f"llama.cpp on CPU, at {LLAMACPP_URL}")

    expected = os.path.basename((ATTRIBUTE_MODEL or "").rstrip("/"))
    # "qwen3-event-extraction-exp5.1" against "attr-exp5.1-q8.gguf": match on the
    # experiment tag, which is the part that distinguishes the two models.
    tag = expected.split("-")[-1] if expected else ""
    if tag and tag not in served:
        return Health(
            False,
            f"llama-server is serving '{served}', but the pipeline is prompting "
            f"for '{expected}'",
            blocks + " — extractions would be silently worse, so this is treated "
                     "as unavailable rather than run anyway",
            f"Point llama-server at the GGUF built from {expected}, or set "
            f"NGEC_ATTRIBUTE_MODEL to the model that '{served}' was built from. "
            f"See demo/deploy/README.md.")

    return Health(True, f"llama.cpp on CPU, serving {served}")


def _index_health(es, name: str, blocks: str) -> Health:
    try:
        if not es.indices.exists(index=name):
            return Health(False, f"index '{name}' not found", blocks,
                          "The index has to be built — see setup/ in the repository.")
        count = es.count(index=name).get("count", 0)
        return Health(True, f"{count:,} documents")
    except Exception as exc:  # noqa: BLE001
        return Health(False, str(exc)[:80], blocks, ES_FIX)


@st.cache_data(ttl=60, show_spinner=False)
def health() -> dict[str, Health]:
    """Status of each dependency, for the sidebar. Cached briefly so that
    re-rendering a page does not re-ping Elasticsearch on every widget change."""
    out: dict[str, Health] = {}

    es_blocks = ("steps 3–5: entity linking, actor categorisation, and the "
                 "place half of dates and places")
    host_port = f"{_env('ES_HOST', 'localhost')}:{_env('ES_PORT', '9200')}"

    es = get_es()
    if es is None:
        out["Elasticsearch"] = Health(False, f"not reachable at {host_port}",
                                      es_blocks, ES_FIX)
        # Only the root failure carries `blocks`; reporting the two indices as
        # separate startup problems would say the same thing three times.
        out["Wikipedia index"] = Health(False, "needs Elasticsearch")
        out["Geonames index"] = Health(False, "needs Elasticsearch")
    else:
        try:
            info = es.info()
            out["Elasticsearch"] = Health(True, f"v{info['version']['number']}")
        except Exception as exc:  # noqa: BLE001
            out["Elasticsearch"] = Health(False, str(exc)[:80], es_blocks, ES_FIX)
        out["Wikipedia index"] = _index_health(
            es, "wiki", "steps 3 and 4: entity linking and actor categorisation")
        out["Geonames index"] = _index_health(
            es, "geonames", "step 5: resolving place names to coordinates")

    try:
        import spacy.util

        for model in ("en_core_web_lg", "en_core_web_trf"):
            ok = model in spacy.util.get_installed_models()
            out[f"spaCy {model}"] = Health(
                ok, "installed" if ok else "missing",
                "" if ok else "every step — the pipeline cannot parse documents without it",
                "" if ok else f"uv run python -m spacy download {model}")
    except Exception as exc:  # noqa: BLE001
        out["spaCy"] = Health(False, str(exc)[:80],
                              "every step — the pipeline cannot parse documents",
                              "uv run python -m spacy download en_core_web_trf")

    out["Attribute model"] = Health(True, attribute_model_label())

    if BACKEND == "llamacpp":
        attr_blocks = ("step 2 onwards: attribute extraction, and so every panel "
                       "that shows a coded record")
        try:
            import urllib.request

            with urllib.request.urlopen(f"{LLAMACPP_URL}/health", timeout=3) as resp:
                ok = json.loads(resp.read()).get("status") == "ok"
            if not ok:
                out["Attribute model backend"] = Health(
                    False, "llama-server not ready", attr_blocks, LLAMACPP_FIX)
            else:
                out["Attribute model backend"] = _llamacpp_model_health(attr_blocks)
        except Exception as exc:  # noqa: BLE001
            out["Attribute model backend"] = Health(
                False, f"llama-server unreachable at {LLAMACPP_URL}: {str(exc)[:50]}",
                attr_blocks, LLAMACPP_FIX)
    elif BACKEND == "vllm":
        # vllm loads the model into this process and holds GPU memory for as long
        # as the app runs, so "is it up" is really "did it load". Checking the
        # recorded load error rather than calling the loader keeps this cheap:
        # the first page view should not block for a minute on a model load just
        # to draw the sidebar.
        error = _load_errors.get("attribute_model")
        if error:
            out["Attribute model backend"] = Health(
                False, f"vllm failed to load: {error[:80]}",
                "step 2 onwards: attribute extraction, and so every panel that "
                "shows a coded record",
                "Usually the GPU is out of memory — check `nvidia-smi` for another "
                "process holding it, or fall back with NGEC_DEMO_BACKEND=llamacpp.")
        else:
            out["Attribute model backend"] = Health(True, _gpu_detail())
    else:
        out["Attribute model backend"] = Health(
            True, BACKEND + (" (GPU)" if USE_GPU else " (CPU)"))
    out["LLM comparison"] = Health(
        bool(_env("ANTHROPIC_API_KEY") or _env("OPENAI_API_KEY")),
        "API key configured" if (_env("ANTHROPIC_API_KEY") or _env("OPENAI_API_KEY"))
        else "no API key — showing precomputed results",
    )
    return out


def degraded() -> list[tuple[str, Health]]:
    """The dependencies that are down *and* stop part of the demo working.

    `ui.setup()` calls this on every page so that a missing service produces a
    banner naming it, rather than pages that quietly show less than they should.
    A dependency whose `blocks` is empty (the LLM comparison key, say) is not
    reported here: the demo is designed to work without it.
    """
    return [(name, h) for name, h in health().items() if not h.ok and h.blocks]


def recheck() -> None:
    """Forget everything that was cached while a service was down.

    The loaders return None when their dependency is missing, and `st.cache_resource`
    remembers that None for the life of the process — so after Elasticsearch or
    llama-server is restarted the app would go on reporting them as down. This
    drops the cached failures (and the one-minute health cache) so the next run
    re-tries the connections. The models themselves are not touched: reloading
    spaCy and the classifiers costs ten seconds and has nothing to do with it.
    """
    health.clear()
    for loader in (get_es, get_geolocation, get_actor_resolver, get_attribute_model):
        loader.clear()


def warm_up(steps=("nlp", "classifier")) -> None:
    """Load the cheap models eagerly so the first interaction is not the slowest."""
    if "nlp" in steps:
        get_nlp()
    if "classifier" in steps:
        get_classifier()
