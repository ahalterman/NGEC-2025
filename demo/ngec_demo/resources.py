"""Loading and caching the pipeline's models and services.

Loading the two spaCy models, the attribute model and the sentence encoder
takes about a minute; doing it per interaction would make the demo unusable, so
everything here is cached for the life of the process. Under Streamlit that is
`st.cache_resource`; `check_demo.py` imports the same functions with no
Streamlit runtime, so the decorator falls back to `functools.cache`.

Loaders return None rather than raising when a dependency is missing, and
`health()` reports what is up, so a page whose Elasticsearch index is down can
say so instead of showing a traceback.

## Modes

The demo can run the same pipeline two ways in one process, and a visitor
switches between them live:

- **gpu** -- vllm for the attribute model, sentence encoders on CUDA.
- **cpu** -- llama-server (a quantized GGUF over HTTP) for the attribute model,
  sentence encoders on the CPU, and torch limited to `NGEC_DEMO_CPU_THREADS`
  cores so the numbers match the deployment box rather than this workstation.

Both model sets stay loaded: every loader takes `mode` and is cached per mode.
CPU mode has to be *faithful* to be worth showing, which is why the loaders
pass an explicit `device="cpu"` -- `gpu=False` alone leaves the choice to
sentence-transformers, which takes the card when it can see one.

spaCy and the mordecai geoparser are on the CPU in **both** modes: spaCy's
device is process-global (`spacy.require_gpu()` cannot be undone per call) and
mordecai's Geoparser defaults to the CPU. `component_devices()` says so, and
the pages show it, so nobody reads the CPU column as a full CPU-only pipeline.

Each loader also instruments its model (see `timing.py`) and runs one tiny
input through it. Warming up matters more than it sounds: vllm captures CUDA
graphs on its first generate (~60 s) and the encoders initialise lazily, and
without this that cost would land on whichever step a visitor clicked first.
The load and warm-up costs are kept in `load_report()` instead. `load_all()`
does the whole set in one go, which is what the sidebar's "Load models" button
and the first click of any page run.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import warnings
from contextlib import contextmanager

from . import timing

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
def available_modes() -> tuple[str, ...]:
    """The modes this machine can actually run, GPU first."""
    return ("gpu", "cpu") if _has_cuda() else ("cpu",)


def current_mode() -> str:
    """The mode a step should run in when it was not told one.

    The sidebar radio writes `st.session_state["mode"]`; outside Streamlit
    (check_demo.py, a notebook) NGEC_DEMO_MODE serves the same purpose.
    """
    if _streamlit_running():
        try:
            import streamlit as st

            mode = st.session_state.get("mode")
            if mode in available_modes():
                return mode
        except Exception:  # noqa: BLE001 - no session (e.g. a cached function)
            pass
    env = os.environ.get("NGEC_DEMO_MODE", "").strip().lower()
    if env in available_modes():
        return env
    return available_modes()[0]


# The deployment box has four cores. Letting torch use all sixteen of this
# workstation's would make the CPU column a measurement of the wrong machine.
CPU_THREADS = int(os.environ.get("NGEC_DEMO_CPU_THREADS", "4"))

# vllm reserves this fraction of the card up front, for weights plus KV cache.
# The package default (0.8) suits a corpus run; here the model is 0.6B and the
# demo codes one document at a time, so most of that would sit idle -- and it
# would starve the sentence encoder that steps 1 and 4 put on the same card.
GPU_MEMORY = float(os.environ.get("NGEC_DEMO_GPU_MEMORY", "0.25"))

LLAMACPP_URL = os.environ.get("NGEC_LLAMACPP_URL", "http://127.0.0.1:8080")


def cpu_backend() -> str:
    """The attribute-model backend for CPU mode: "llamacpp" or "transformers".

    llama-server is the deployment path and roughly 4x faster than
    transformers in float32, but it is a separate service. Set
    NGEC_DEMO_CPU_BACKEND=transformers to run CPU mode without it.
    """
    return os.environ.get("NGEC_DEMO_CPU_BACKEND", "").strip() or "llamacpp"


def backend(mode: str | None = None) -> str:
    """Which attribute-model backend a mode uses."""
    mode = mode or current_mode()
    return "vllm" if mode == "gpu" else cpu_backend()


def device(mode: str | None = None) -> str:
    """The torch device the sentence encoders run on in a mode."""
    return "cuda" if (mode or current_mode()) == "gpu" else "cpu"


@contextmanager
def compute(mode: str):
    """Hold the process to `mode`'s compute settings for the duration.

    Only CPU mode has anything to do: torch's thread count is process-global,
    so it is set on the way in and restored on the way out rather than once at
    import, which would also throttle GPU mode's data loading.
    """
    if mode != "cpu":
        yield
        return
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


_errors: dict[str, str] = {}

# {(component, mode): {"load": seconds, "warm_up": seconds}}, filled by the
# loaders and shown on the Timing page. A visitor who sees "0.3 s a document"
# should also be able to see the minute of loading that made it possible.
_load_times: dict[tuple[str, str], dict[str, float]] = {}


def _fail(name: str, exc: Exception):
    logger.warning("%s unavailable: %s", name, exc)
    _errors[name] = f"{type(exc).__name__}: {exc}"
    return None


@contextmanager
def _record_load(name: str, mode: str):
    """Time a loader and give it back a `warm_up(fn)` to time the warm-up with.

    The warm-up runs inside a throwaway `timing.collect()` so that loading
    triggered lazily by the first step is not charged to that step.
    """
    start = time.time()
    entry = {"load": 0.0, "warm_up": 0.0}

    def warm_up(fn):
        warm_start = time.time()
        try:
            with compute(mode), timing.collect():
                fn()
        except Exception as exc:  # noqa: BLE001 - a cold model is still usable
            logger.warning("warm-up of %s (%s) failed: %s", name, mode, exc)
        entry["warm_up"] = time.time() - warm_start

    yield warm_up
    entry["load"] = time.time() - start - entry["warm_up"]
    _load_times[(name, mode)] = entry


def load_report() -> list[dict]:
    """What each model cost to load and to warm up, one row per (name, mode)."""
    return [{"component": name, "mode": mode,
             "load": round(times["load"], 2),
             "warm_up": round(times["warm_up"], 2)}
            for (name, mode), times in _load_times.items()]


# --- loaders (unkeyed: CPU in both modes, so one instance is enough) ---------

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

    with _record_load("spaCy trf", "shared") as warm_up:
        nlp = load_nlp()
        warm_up(lambda: list(nlp.pipe(["Police in Paris arrested a protester."])))
    return nlp


@cache_resource
def get_nlp_lg():
    """en_core_web_lg, which is what ActorResolver expects (trf tokenises differently)."""
    import spacy

    with _record_load("spaCy lg", "shared") as warm_up:
        nlp = spacy.load("en_core_web_lg")
        warm_up(lambda: nlp("Police in Paris arrested a protester."))
    return nlp


@cache_resource
def get_geolocation():
    """The mordecai3 geoparser, or None without Elasticsearch.

    On the CPU in both modes: mordecai's Geoparser takes device='cpu' by
    default and the demo does not override it.
    """
    from ngec import GeolocationModel

    es = get_es()
    if es is None:
        return None
    try:
        with _record_load("geoparser", "shared") as warm_up:
            geo = GeolocationModel(nlp=get_nlp_trf(), es_client=es, quiet=True,
                                   save_intermediate=False)
            timing.instrument(geo, {"geo.geoparse_doc": "geoparse doc",
                                    "geo.lookup_city": "city lookup"})
            text = "Police in Paris arrested a protester."
            warm_up(lambda: geo.process([{"id": "warmup", "event_text": text}],
                                        list(get_nlp_trf().pipe([text]))))
        return geo
    except Exception as exc:  # noqa: BLE001
        return _fail("geolocation", exc)


@cache_resource
def get_formatter():
    """Final assembly: picks the event location and resolves the date.

    Pure Python and milliseconds fast, so it is neither instrumented nor
    warmed up.
    """
    from ngec import Formatter

    return Formatter(quiet=True)


# --- loaders (one instance per mode) -----------------------------------------

@cache_resource
def get_classifier(mode: str):
    """(classifier, warnings). The DemoModelWarning is captured, not silenced.

    That these are demonstration classifiers and not the ones behind POLECAT is
    something the demo should say out loud, and the classifier itself is the
    authority on the wording -- so the warning is caught here and shown once in
    the sidebar rather than printed on every rerun.
    """
    from ngec.classifiers.plover_sklearn import PloverSklearnClassifier

    with _record_load("event classifier", mode) as warm_up:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PloverSklearnClassifier(device=device(mode))
        labels = {"encoder.encode": "sentence encoder"}
        if model.vectorizer is not None:
            labels["vectorizer.transform"] = "tf-idf features"
        timing.instrument(model, labels)
        warm_up(lambda: model.process([{"id": "warmup",
                                        "event_text": "Protesters blocked the road."}]))
    return model, [str(w.message) for w in caught]


@cache_resource
def get_attribute_model(mode: str):
    """The fine-tuned Qwen3 span extractor, or None if it could not be loaded.

    In CPU mode with the llamacpp backend this constructs even when
    llama-server is down -- the engine only fails when it tries to generate --
    so `health(mode)` is where a dead server shows up, not here.
    """
    from ngec import AttributeModel

    kwargs = {"silent": True, "gpu": mode == "gpu", "backend": backend(mode),
              "save_intermediate": False}
    if backend(mode) == "vllm":
        kwargs["max_gpu_memory"] = GPU_MEMORY
    if backend(mode) == "llamacpp":
        kwargs["llamacpp_url"] = LLAMACPP_URL
    if os.environ.get("NGEC_ATTRIBUTE_MODEL"):
        kwargs["model_name"] = os.environ["NGEC_ATTRIBUTE_MODEL"]
    try:
        with _record_load("attribute model", mode) as warm_up:
            model = AttributeModel(**kwargs)
            _instrument_attribute_model(model)
            warm_up(lambda: model.process([
                {"id": "warmup", "orig_id": "warmup", "event_type": "PROTEST",
                 "event_mode": "",
                 "event_text": "Protesters blocked the road in Paris on Tuesday."}]))
        _errors.pop("attribute model", None)
        return model
    except Exception as exc:  # noqa: BLE001
        return _fail("attribute model", exc)


def _instrument_attribute_model(model) -> None:
    """Time prompt building and generation, whichever path this backend takes.

    vllm still generates through `call_llm_batch(make_prompt(...))`; the
    transformers and llamacpp backends go through an engine, which builds a
    conversation instead of a prompt string. Same two labels either way.
    """
    if model.engine is None:  # vllm
        timing.instrument(model, {"make_prompt": "build prompt",
                                  "call_llm_batch": "generate"})
        return

    timing.instrument(model, {"_build_conversation": "build prompt"})
    engine = model.engine
    original = engine.generate
    if getattr(original, "_ngec_timed", None):
        return

    def generate(*args, **kwargs):
        with timing.timed("generate"):
            out = original(*args, **kwargs)
            # llama-server reports where its own time went. Prefill and decode
            # are limited by different things (arithmetic and memory
            # bandwidth), so the split is the interesting part of a CPU run;
            # see docs/PERFORMANCE.md. Other engines report nothing.
            reported = getattr(engine, "last_timings", None) or []
            if reported:
                timing.record("prefill",
                              sum(t.get("prompt_ms") or 0 for t in reported) / 1000,
                              len(reported))
                timing.record("decode",
                              sum(t.get("predicted_ms") or 0 for t in reported) / 1000,
                              len(reported))
            return out

    generate._ngec_timed = "generate"
    engine.generate = generate


def get_actor_resolver(mode: str, agents_file: str | None = None):
    """Wikipedia + agent-dictionary actor coder, or None without Elasticsearch.

    `agents_file` swaps in a custom actor dictionary (step 4's "bring your own
    codebook"); each distinct file costs one encode of its patterns and is then
    cached separately.

    The cache keys on the arguments exactly as given, and `f(mode)` is not the
    same key as `f(mode, None)` -- which would quietly load the resolver twice.
    So the caching happens one level down, on a call that always passes both.
    """
    return _load_actor_resolver(mode, agents_file)


@cache_resource
def _load_actor_resolver(mode: str, agents_file: str | None):
    from ngec import ActorResolver

    es = get_es()
    if es is None:
        return None
    name = "actor resolver" if agents_file is None else "actor resolver (custom agents)"
    try:
        with _record_load(name, mode) as warm_up:
            resolver = ActorResolver(spacy_model=get_nlp_lg(), es_client=es,
                                     gpu=mode == "gpu", device=device(mode),
                                     agents_file=agents_file,
                                     save_intermediate=False)
            _instrument_actor_resolver(resolver)

            def warm():
                resolver.actor_to_code("the French police",
                                       context="Police in Paris arrested a protester.")
                resolver.cache_manager.clear()

            warm_up(warm)
        return resolver
    except Exception as exc:  # noqa: BLE001
        return _fail("actor resolver", exc)


def _instrument_actor_resolver(resolver) -> None:
    """The call order in `ActorResolver.actor_to_code`, as a nested breakdown.

    The two encoders are separate objects but the wiki encoder may be shared
    with the agent matcher; `instrument` is idempotent, so wrapping it once
    here covers both callers.
    """
    timing.instrument(resolver, {
        "actor_to_code": "actor mention",
        "country_detector.search_nat": "nationality strip",
        "agent_matcher.trf_agent_match": "agent matcher",
        "wiki_matcher.query_wiki": "wikipedia",
        "wiki_matcher.wiki_searcher.search_wiki": "ES search",
        "wiki_matcher._create_scoring_dataframe": "candidate features",
        "wiki_matcher.trf.encode": "wiki encoder",
        "wiki_matcher.actor_sim.encode": "actor-sim encoder",
        "wiki_matcher._call_ranker": "ranker",
        "wiki_parser.wiki_to_code": "article to code",
        "code_selector.pick_best_code": "pick code",
    })


# --- loading everything at once ----------------------------------------------

# Which modes have been loaded end to end in this process. Loading is
# per-process, not per-visitor -- `st.cache_resource` caches for the life of the
# server -- so a module-level set has exactly the same lifetime as the cache it
# is describing, and a second browser tab correctly sees the models the first
# one loaded.
_loaded_modes: set[str] = set()


def is_loaded(mode: str) -> bool:
    """Has everything this mode needs already been loaded in this process?"""
    return mode in _loaded_modes


def load_all(mode: str, on_step=None) -> float:
    """Load and warm up every model the pipeline uses in `mode`; seconds spent.

    Each loader is cached, so the pages could leave this to happen on the first
    click -- but then that click sits for a minute with nothing to show for it.
    Calling this from the sidebar, or as a first labelled phase of a run, turns
    the wait into something with a name on it. `on_step(name)` is called before
    each component so the caller can say which one is loading.

    The order is the pipeline's own (see `steps.run_pipeline`), so the load
    report reads in the order a document is coded. A component whose service is
    down loads as None and is not retried here; `health()` is where that shows.
    """
    start = time.time()
    stages = [
        ("Elasticsearch", get_es),
        ("spaCy", lambda: (get_nlp_trf(), get_nlp_lg())),
        ("classifier", lambda: get_classifier(mode)),
        ("geoparser", get_geolocation),
        ("extractor", lambda: get_attribute_model(mode)),
        ("encoders", lambda: get_actor_resolver(mode)),
    ]
    for name, load in stages:
        if on_step is not None:
            on_step(name)
        load()
    get_formatter()  # pure Python and instant; not worth announcing
    _loaded_modes.add(mode)
    return time.time() - start


def load_seconds(mode: str) -> float:
    """What loading and warming up this mode cost, from `load_report()`.

    The shared models (spaCy, the geoparser) are counted in whichever mode
    asks, since they were loaded once for both.
    """
    return sum(row["load"] + row["warm_up"] for row in load_report()
               if row["mode"] in (mode, "shared"))


# --- what runs where ---------------------------------------------------------

def component_devices(mode: str | None = None) -> dict[str, str]:
    """One line per component: where it runs in this mode.

    Worth showing next to any CPU-vs-GPU table, because the CPU column is not
    a CPU-only pipeline -- spaCy and the geoparser are on the CPU either way,
    so the parts that move are the encoders and the LLM.
    """
    mode = mode or current_mode()
    if mode == "gpu":
        attribute = "vllm on cuda"
    elif cpu_backend() == "llamacpp":
        attribute = f"llama.cpp server ({LLAMACPP_URL})"
    else:
        attribute = "transformers on cpu"
    encoders = device(mode)
    return {
        "event classifier encoder": encoders,
        "attribute model": attribute,
        "actor encoders": encoders,
        "spaCy": "cpu",
        "geoparser": "cpu",
        "Elasticsearch": "service",
        "torch threads": "default" if mode == "gpu" else str(CPU_THREADS),
    }


# --- health ------------------------------------------------------------------

def _index_health(es, name: str) -> dict:
    if es is None:
        return {"ok": False, "detail": "no Elasticsearch"}
    try:
        count = es.count(index=name)["count"]
        return {"ok": count > 0, "detail": f"{count:,} docs"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)[:60]}


def _llamacpp_health() -> dict:
    """Is llama-server up, and is it serving the model we are prompting for?

    Under this backend the weights come from whatever GGUF `llama-server` was
    started with, while the prompt format is chosen from the model name the
    Python side was configured with. Nothing forces those to agree, and a
    mismatch is silent: the pipeline runs, returns valid JSON, and the
    extractions are quietly worse. So compare the two names. The comparison is
    deliberately loose -- the GGUF file is named by hand -- and a name we
    cannot parse is reported rather than treated as a failure.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{LLAMACPP_URL}/v1/models", timeout=3) as resp:
            models = json.loads(resp.read()).get("models") or []
        served = str(models[0].get("name", "")) if models else ""
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "detail": f"not reachable at {LLAMACPP_URL} ({str(exc)[:40]})"}

    expected = os.environ.get("NGEC_ATTRIBUTE_MODEL", "")
    if not expected:
        from ngec.attribute_model import DEFAULT_MODEL

        expected = DEFAULT_MODEL
    expected = os.path.basename(expected.rstrip("/"))
    # "qwen3-event-extraction-exp5.1" against "attr-exp5.1-q8.gguf": match on
    # the experiment tag, the part that distinguishes the two models.
    tag = expected.split("-")[-1] if expected else ""
    # llama-server reports the GGUF by the path it was started with; the
    # sidebar is one column wide, so only the file name is shown.
    name = os.path.basename(served.rstrip("/"))
    if served and tag and tag not in served:
        return {"ok": False,
                "detail": f"serving '{name}', but prompting for '{expected}'"}
    return {"ok": True, "detail": f"{name or 'up'} at {LLAMACPP_URL}"}


def health(mode: str | None = None) -> dict[str, dict]:
    """One row per dependency: {"ok": bool, "detail": str}.

    Cheap enough to call on every rerun: it touches Elasticsearch and, in CPU
    mode, llama-server, but never loads a model, and reports a model that
    failed to load from `_errors`.
    """
    mode = mode or current_mode()
    es = get_es()
    host = os.environ.get("ES_HOST", "localhost")
    port = os.environ.get("ES_PORT", "9200")

    if mode == "gpu":
        compute_detail = "GPU: vllm + cuda encoders"
    else:
        compute_detail = (f"CPU: {cpu_backend()} + {CPU_THREADS} threads")

    out = {
        "Elasticsearch": {"ok": es is not None,
                          "detail": f"{host}:{port}" if es is not None
                          else _errors.get("elasticsearch", "not reachable")[:60]},
        "wiki index": _index_health(es, "wiki"),
        "geonames index": _index_health(es, "geonames"),
        "compute": {"ok": True, "detail": compute_detail},
    }
    if mode == "cpu" and cpu_backend() == "llamacpp":
        out["llama-server"] = _llamacpp_health()
    if "attribute model" in _errors:
        out["attribute model"] = {"ok": False, "detail": _errors["attribute model"][:60]}
    return out


def generation_error(mode: str) -> str | None:
    """Why the attribute model returned nothing, when the cause is a service.

    An empty extraction is a normal answer -- the model declines documents --
    so a step only calls this once it already has nothing, to tell "the model
    said no" apart from "llama-server is down".
    """
    if get_attribute_model(mode) is None:
        return _errors.get("attribute model", "the attribute model did not load")
    if mode == "cpu" and cpu_backend() == "llamacpp":
        row = _llamacpp_health()
        if not row["ok"]:
            return f"llama-server: {row['detail']}"
    return None
