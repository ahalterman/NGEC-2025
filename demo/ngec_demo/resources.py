"""Loading and caching the pipeline's models and services.

Loading the two spaCy models, the attribute model and the sentence encoder
takes about a minute; doing it per interaction would make the demo unusable, so
everything here is cached for the life of the process. Under Streamlit that is
`st.cache_resource`; `check_demo.py` imports the same functions with no
Streamlit runtime, so the decorator falls back to `functools.cache`.

Loaders return None rather than raising when a dependency is missing, and
`health()` reports what is up, so a page whose Elasticsearch index is down can
say so instead of showing a traceback.
"""

from __future__ import annotations

import functools
import logging
import os
import warnings

logger = logging.getLogger(__name__)


# --- caching -----------------------------------------------------------------

def _streamlit_running() -> bool:
    try:
        import streamlit.runtime

        return streamlit.runtime.exists()
    except Exception:  # noqa: BLE001 - no streamlit, or a version without runtime
        return False


def cache_resource(fn):
    """`st.cache_resource` in the app, `functools.cache` in a plain script."""
    if _streamlit_running():
        import streamlit as st

        return st.cache_resource(show_spinner=False)(fn)
    return functools.cache(fn)


# --- configuration -----------------------------------------------------------

def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken CUDA build counts as no GPU
        return False


@functools.cache
def backend() -> str:
    """Which attribute-model backend to use: "vllm" or "transformers".

    vllm is ~40x faster but is a CUDA-only build here, so it is chosen only when
    it is importable *and* torch can see a card. NGEC_DEMO_BACKEND overrides.
    """
    env = os.environ.get("NGEC_DEMO_BACKEND", "").strip()
    if env:
        return env
    import importlib.util

    if importlib.util.find_spec("vllm") is not None and _has_cuda():
        return "vllm"
    return "transformers"


@functools.cache
def use_gpu() -> bool:
    """NGEC_DEMO_GPU as 1/0, else whether torch can see a card."""
    env = os.environ.get("NGEC_DEMO_GPU", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return _has_cuda()


# vllm reserves this fraction of the card up front, for weights plus KV cache.
# The package default (0.8) suits a corpus run; here the model is 0.6B and the
# demo codes one document at a time, so most of that would sit idle -- and it
# would starve the sentence encoder that steps 1 and 4 put on the same card.
GPU_MEMORY = float(os.environ.get("NGEC_DEMO_GPU_MEMORY", "0.25"))

_errors: dict[str, str] = {}


def _fail(name: str, exc: Exception):
    logger.warning("%s unavailable: %s", name, exc)
    _errors[name] = f"{type(exc).__name__}: {exc}"
    return None


# --- loaders -----------------------------------------------------------------

@cache_resource
def get_es():
    """Elasticsearch client for the wiki and geonames indices, or None."""
    try:
        from ngec.es_client import setup_es_client

        host = os.environ.get("ES_HOST", "localhost")
        port = int(os.environ.get("ES_PORT", "9200"))
        user = os.environ.get("ES_USER")
        password = os.environ.get("ES_PASSWORD")
        kwargs = {"http_auth": (user, password)} if user and password else {}

        client = setup_es_client(hosts=[host], port=port, **kwargs)
        client.info()  # force a real connection, so a dead ES fails here
        _errors.pop("elasticsearch", None)
        return client
    except Exception as exc:  # noqa: BLE001
        return _fail("elasticsearch", exc)


@cache_resource
def get_nlp_trf():
    """en_core_web_trf plus the token_tensors pipe, as the geoparser needs it."""
    from ngec.utilities import load_nlp

    return load_nlp()


@cache_resource
def get_nlp_lg():
    """en_core_web_lg, which is what ActorResolver expects (trf tokenises differently)."""
    import spacy

    return spacy.load("en_core_web_lg")


@cache_resource
def get_classifier():
    """(classifier, warnings). The DemoModelWarning is captured, not silenced.

    That these are demonstration classifiers and not the ones behind POLECAT is
    something the demo should say out loud, and the classifier itself is the
    authority on the wording -- so the warning is caught here and shown once in
    the sidebar rather than printed on every rerun.
    """
    from ngec.classifiers.plover_sklearn import PloverSklearnClassifier

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PloverSklearnClassifier()
    return model, [str(w.message) for w in caught]


@cache_resource
def get_attribute_model():
    """The fine-tuned Qwen3 span extractor, or None if it could not be loaded."""
    from ngec import AttributeModel

    kwargs = {"silent": True, "gpu": use_gpu(), "backend": backend(),
              "save_intermediate": False}
    if backend() == "vllm":
        kwargs["max_gpu_memory"] = GPU_MEMORY
    if os.environ.get("NGEC_ATTRIBUTE_MODEL"):
        kwargs["model_name"] = os.environ["NGEC_ATTRIBUTE_MODEL"]
    try:
        model = AttributeModel(**kwargs)
        _errors.pop("attribute model", None)
        return model
    except Exception as exc:  # noqa: BLE001
        return _fail("attribute model", exc)


@cache_resource
def get_actor_resolver(agents_file: str | None = None):
    """Wikipedia + agent-dictionary actor coder, or None without Elasticsearch.

    `agents_file` swaps in a custom actor dictionary (step 4's "bring your own
    codebook"); each distinct file costs one encode of its patterns and is then
    cached separately.
    """
    from ngec import ActorResolver

    es = get_es()
    if es is None:
        return None
    try:
        return ActorResolver(spacy_model=get_nlp_lg(), es_client=es, gpu=use_gpu(),
                             agents_file=agents_file, save_intermediate=False)
    except Exception as exc:  # noqa: BLE001
        return _fail("actor resolver", exc)


@cache_resource
def get_geolocation():
    """The mordecai3 geoparser, or None without Elasticsearch."""
    from ngec import GeolocationModel

    es = get_es()
    if es is None:
        return None
    try:
        return GeolocationModel(nlp=get_nlp_trf(), es_client=es, quiet=True,
                                save_intermediate=False)
    except Exception as exc:  # noqa: BLE001
        return _fail("geolocation", exc)


@cache_resource
def get_formatter():
    """Final assembly: picks the event location and resolves the date."""
    from ngec import Formatter

    return Formatter(quiet=True)


# --- health ------------------------------------------------------------------

def _index_health(es, name: str) -> dict:
    if es is None:
        return {"ok": False, "detail": "no Elasticsearch"}
    try:
        count = es.count(index=name)["count"]
        return {"ok": count > 0, "detail": f"{count:,} docs"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)[:60]}


def health() -> dict[str, dict]:
    """One row per dependency: {"ok": bool, "detail": str}.

    Cheap enough to call on every rerun: it touches Elasticsearch but never
    loads a model, and reports a model that failed to load from `_errors`.
    """
    es = get_es()
    host = os.environ.get("ES_HOST", "localhost")
    port = os.environ.get("ES_PORT", "9200")

    out = {
        "Elasticsearch": {"ok": es is not None,
                          "detail": f"{host}:{port}" if es is not None
                          else _errors.get("elasticsearch", "not reachable")[:60]},
        "wiki index": _index_health(es, "wiki"),
        "geonames index": _index_health(es, "geonames"),
        "backend": {"ok": True, "detail": f"{backend()} on {'GPU' if use_gpu() else 'CPU'}"},
    }
    if "attribute model" in _errors:
        out["attribute model"] = {"ok": False, "detail": _errors["attribute model"][:60]}
    return out
