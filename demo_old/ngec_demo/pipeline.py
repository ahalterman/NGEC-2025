"""Running the pipeline for the demo, with a trace of what each step did.

`run()` walks the same six steps as `ngec.plover_coder.PloverCoder.process`, but
snapshots the record list after each one and times it. The step pages read a
single field out of that trace; the end-to-end page reads all of it. Nothing
here reimplements pipeline logic — if a step page shows something, it is
something the pipeline actually produced.

Snapshots are taken by round-tripping through JSON, both because the records
carry objects that do not deep-copy cleanly (spaCy docs, numpy scalars) and
because everything the demo displays has to be JSON-renderable anyway.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import streamlit as st

from . import resources

# The six steps, in order, with the paper section each corresponds to.
STEPS = [
    ("classify", "Event detection", "step1"),
    ("geolocate", "Geolocation", "step5"),
    ("split", "Story → events", "step1"),
    ("attributes", "Attribute extraction", "step2"),
    ("actors", "Actor resolution", "step3"),
    ("format", "Dates & locations", "step5"),
]

STEP_LABELS = {key: label for key, label, _ in STEPS}


@dataclass
class StepTrace:
    key: str
    label: str
    seconds: float = 0.0
    ok: bool = True
    note: str = ""
    records: list[dict] = field(default_factory=list)
    skipped: bool = False


@dataclass
class Run:
    text: str
    pub_date: str
    steps: list[StepTrace] = field(default_factory=list)
    final: list[dict] = field(default_factory=list)
    error: str | None = None

    def step(self, key: str) -> StepTrace | None:
        for s in self.steps:
            if s.key == key:
                return s
        return None

    def records_after(self, key: str) -> list[dict]:
        s = self.step(key)
        return s.records if s else []

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.steps)


def _snapshot(records) -> list[dict]:
    """A JSON-safe deep copy of the record list."""
    try:
        return json.loads(json.dumps(records, default=str))
    except (TypeError, ValueError):
        return [{"_unrenderable": repr(r)[:2000]} for r in records]


def run(
    text: str,
    pub_date: str,
    stop_after: str | None = None,
    event_type_override: str | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> Run:
    """Execute the pipeline, tracing each step.

    stop_after: last step key to run, so a page that only needs classification
        does not spend ten seconds in the attribute model.
    event_type_override: skip the classifier and force an event type, which is
        how the demo lets a visitor see later steps on a document the
        demonstration classifier misses.
    progress: called as progress(step_key, label) before each step runs.
    """
    result = Run(text=text, pub_date=pub_date)
    wanted = [k for k, _, _ in STEPS]
    if stop_after:
        wanted = wanted[: wanted.index(stop_after) + 1]

    def announce(key):
        if progress:
            progress(key, STEP_LABELS[key])

    try:
        nlp = resources.get_nlp()
        doc_list = [nlp(text)]
        story_list = [{"id": "demo", "event_text": text, "pub_date": pub_date}]

        # --- 1. event detection -------------------------------------------
        if "classify" in wanted:
            announce("classify")
            t0 = time.time()
            if event_type_override:
                story_list[0]["event_type"] = [event_type_override]
                story_list[0]["event_type_confidence"] = {event_type_override: 1.0}
                story_list[0]["event_mode"] = []
                note = f"classifier bypassed; forced to {event_type_override}"
            else:
                clf, _ = resources.get_classifier()
                story_list = clf.process(story_list)
                types = story_list[0].get("event_type") or []
                note = ", ".join(types) if types else "no event type above threshold"
            result.steps.append(
                StepTrace("classify", "Event detection", time.time() - t0,
                          note=note, records=_snapshot(story_list))
            )
            if not story_list[0].get("event_type"):
                result.error = "no_event"
                return result

        # --- 2. geolocation ------------------------------------------------
        if "geolocate" in wanted:
            announce("geolocate")
            t0 = time.time()
            geo = resources.get_geolocation()
            if geo is None:
                result.steps.append(
                    StepTrace("geolocate", "Geolocation", 0.0, ok=False, skipped=True,
                              note="Elasticsearch/geonames unavailable",
                              records=_snapshot(story_list))
                )
            else:
                story_list = geo.process(story_list, doc_list)
                ents = story_list[0].get("geolocated_ents") or []
                result.steps.append(
                    StepTrace("geolocate", "Geolocation", time.time() - t0,
                              note=f"{len(ents)} place mention(s) resolved",
                              records=_snapshot(story_list))
                )

        # --- 3. story -> events --------------------------------------------
        if "split" in wanted:
            announce("split")
            t0 = time.time()
            from ngec.utilities import stories_to_events

            event_list = stories_to_events(story_list, doc_list)
            result.steps.append(
                StepTrace("split", "Story → events", time.time() - t0,
                          note=f"1 story → {len(event_list)} event record(s)",
                          records=_snapshot(event_list))
            )
        else:
            return result

        # --- 4. attribute extraction ---------------------------------------
        if "attributes" in wanted:
            announce("attributes")
            t0 = time.time()
            am = resources.get_attribute_model()
            if am is None:
                result.steps.append(
                    StepTrace("attributes", "Attribute extraction", 0.0, ok=False,
                              skipped=True, note="attribute model unavailable",
                              records=_snapshot(event_list))
                )
                return result
            before = len(event_list)
            event_list = am.process(event_list)
            after = len(event_list)
            note = f"{before} record(s) → {after} extracted event(s)"
            if after == 0:
                note += " — nothing extracted, all records dropped"
            result.steps.append(
                StepTrace("attributes", "Attribute extraction", time.time() - t0,
                          note=note, records=_snapshot(event_list))
            )
            if not event_list:
                result.error = "no_attributes"
                return result

        # --- 5. actor resolution -------------------------------------------
        if "actors" in wanted:
            announce("actors")
            t0 = time.time()
            ar = resources.get_actor_resolver()
            if ar is None:
                result.steps.append(
                    StepTrace("actors", "Actor resolution", 0.0, ok=False, skipped=True,
                              note="Elasticsearch/wiki index unavailable",
                              records=_snapshot(event_list))
                )
            else:
                event_list = ar.process(event_list)
                coded = sum(len(e.get("actor") or []) + len(e.get("recipient") or [])
                            for e in event_list)
                result.steps.append(
                    StepTrace("actors", "Actor resolution", time.time() - t0,
                              note=f"{coded} actor/recipient mention(s) coded",
                              records=_snapshot(event_list))
                )

        # --- 6. formatting --------------------------------------------------
        if "format" in wanted:
            announce("format")
            t0 = time.time()
            fmt = resources.get_formatter()
            event_list = fmt.process(event_list, return_raw=True)
            result.steps.append(
                StepTrace("format", "Dates & locations", time.time() - t0,
                          note="dates and event locations resolved",
                          records=_snapshot(event_list))
            )

        result.final = _snapshot(event_list)

    except Exception as exc:  # noqa: BLE001 - the demo must not show a traceback
        result.error = f"{type(exc).__name__}: {exc}"

    return result


# --------------------------------------------------------------------------
# Serialising runs, so the curated examples can be pre-coded
# --------------------------------------------------------------------------
#
# A document costs roughly a minute on CPU, almost all of it in the attribute
# model, so a visitor clicking between examples cannot wait for a live run.
# `build_example_cache.py` codes every curated example once and writes the
# traces here; the app serves those instantly and only runs live for text a
# visitor typed themselves.


def run_to_dict(result: Run) -> dict:
    return {
        "text": result.text,
        "pub_date": result.pub_date,
        "error": result.error,
        "final": result.final,
        "steps": [
            {
                "key": s.key, "label": s.label, "seconds": s.seconds, "ok": s.ok,
                "note": s.note, "records": s.records, "skipped": s.skipped,
            }
            for s in result.steps
        ],
    }


def run_from_dict(payload: dict) -> Run:
    return Run(
        text=payload["text"],
        pub_date=payload["pub_date"],
        error=payload.get("error"),
        final=payload.get("final") or [],
        steps=[StepTrace(**s) for s in payload.get("steps", [])],
    )


def truncate(result: Run, stop_after: str | None) -> Run:
    """A full cached run, viewed as if it had stopped early.

    Pages that stop early read `records_after(...)`, never `final`, so dropping
    the later steps is enough — and it means one cached full run serves every
    step page rather than one cache entry per depth.
    """
    if stop_after is None:
        return result
    order = [k for k, _, _ in STEPS]
    keep = set(order[: order.index(stop_after) + 1])
    sliced = [s for s in result.steps if s.key in keep]
    return Run(text=result.text, pub_date=result.pub_date, steps=sliced,
               final=[], error=result.error)


# --------------------------------------------------------------------------
# Single-step helpers, for the pages that look inside one component
# --------------------------------------------------------------------------


CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "example_runs.json"


@st.cache_data(show_spinner=False)
def example_cache() -> dict[str, dict]:
    """Pre-coded runs, keyed by "<pub_date>\\x00<text>"."""
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def cache_key(text: str, pub_date: str) -> str:
    return f"{pub_date}\x00{text}"


def cached_example(text: str, pub_date: str, stop_after: str | None = None) -> Run | None:
    payload = example_cache().get(cache_key(text, pub_date))
    if payload is None:
        return None
    return truncate(run_from_dict(payload), stop_after)


@dataclass
class Extraction:
    """One run of the attribute model on one (document, event definition) pair."""

    records: list[dict] = field(default_factory=list)
    prompt: str = ""
    seconds: float = 0.0
    error: str | None = None


def extract_with_definition(text: str, label: str, definition: str,
                            notes: str = "") -> Extraction:
    """Run step 4 alone, with a definition the caller wrote.

    The pipeline normally looks the definition up in the PLOVER codebook by
    event type. Passing `event_def` on the record instead makes the model
    extract for an event type that is not in any codebook, which is what a
    visitor bringing their own ontology is doing — and, because the label never
    reaches a classifier or an index, it is also the whole of what they have to
    supply.

    The prompt is returned along with the records because on this page it is
    half the point: it is the entire briefing the model gets about a new event
    type, and it is short.
    """
    am = resources.get_attribute_model()
    if am is None:
        return Extraction(error="The attribute model is not available.")

    record = {
        "id": "custom",
        "orig_id": "custom",
        "event_text": text,
        "event_type": label,
        "event_mode": "",
        "event_def": definition,
    }
    if notes.strip():
        record["extraction_notes"] = notes.strip()

    try:
        prompt = am.make_prompt(record)
        t0 = time.time()
        # process() explodes and drops, so it returns a new list -- and an empty
        # one when the model finds no such event in the document, which on this
        # page is a result rather than a failure.
        records = am.process([dict(record)])
        return Extraction(records=_snapshot(records), prompt=prompt,
                          seconds=time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - the demo must not show a traceback
        return Extraction(error=f"{type(exc).__name__}: {exc}")


@st.cache_data(show_spinner=False, max_entries=64)
def classify_scores(text: str) -> list[dict]:
    """Every event type's probability and its own decision threshold.

    `PloverSklearnClassifier.process` only reports classes above threshold. The
    step 1 page wants the whole vector, including the near misses, because the
    per-class thresholds are the point.
    """
    clf, _ = resources.get_classifier()
    embeddings = clf._compute_embeddings([text])

    rows = []
    for event_type, model in clf.type_models.items():
        prob = float(model.predict_proba(embeddings)[:, 1][0])
        threshold = clf._threshold_for(event_type)
        rows.append({
            "event_type": event_type,
            "probability": prob,
            "threshold": float(threshold),
            "fired": prob >= threshold,
            "margin": prob - float(threshold),
        })
    return sorted(rows, key=lambda r: -r["probability"])


@st.cache_data(show_spinner=False, max_entries=64)
def mode_scores(text: str, event_type: str) -> list[dict]:
    """Mode probabilities for one event type, with each mode's threshold."""
    clf, _ = resources.get_classifier()
    embeddings = clf._compute_embeddings([text])

    # mode_models is nested: {event_type: {mode: model}}, matching how the models
    # were trained (a mode model only ever sees documents where its parent fired).
    rows = []
    for mode, model in (clf.mode_models.get(event_type) or {}).items():
        full_name = f"{event_type}-{mode}"
        prob = float(model.predict_proba(embeddings)[:, 1][0])
        threshold = clf._threshold_for(full_name, is_mode=True)
        rows.append({
            "mode": mode,
            "full_name": full_name,
            "probability": prob,
            "threshold": float(threshold),
            "fired": prob >= threshold,
        })
    return sorted(rows, key=lambda r: -r["probability"])


@st.cache_data(show_spinner=False, max_entries=256)
def resolve_date_phrase(phrase: str, pub_date: str) -> dict:
    """Run one date span through the resolver cascade."""
    from ngec.formatter import resolve_date

    # resolve_date mutates the event and returns the event, not the resolution.
    event = {"attributes": {"date": [phrase]}, "pub_date": pub_date}
    try:
        return resolve_date(event).get("date_resolved") or {}
    except Exception as exc:  # noqa: BLE001
        return {"reason": f"error: {exc}", "date_type": "error"}


@st.cache_data(show_spinner=False, max_entries=128)
def resolve_actor(mention: str, context: str = "", query_date: str = "today") -> dict | None:
    """Run one actor mention through Wikipedia linking and categorisation."""
    ar = resources.get_actor_resolver()
    if ar is None:
        return None
    try:
        return ar.actor_to_code(mention, context=context, query_date=query_date)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@st.cache_data(show_spinner=False, max_entries=128)
def wiki_candidates(query: str, limit: int = 12) -> list[dict]:
    """The retrieval stage of entity linking: what Elasticsearch returns.

    Linking is retrieve-then-rank. This is the retrieve half — the candidate set
    the XGBoost ranker then chooses from — and seeing it makes clear that a
    linking failure is usually a ranking failure, not a missing page.
    """
    ar = resources.get_actor_resolver()
    if ar is None:
        return []
    try:
        hits = ar.wiki_matcher.wiki_searcher.search_wiki(query, max_results=200)
    except Exception:  # noqa: BLE001
        return []

    out = []
    for hit in (hits or [])[:limit]:
        out.append({
            "title": hit.get("title", ""),
            "short_description": (hit.get("short_description") or [""])[0]
            if isinstance(hit.get("short_description"), list)
            else (hit.get("short_description") or ""),
            "redirects": len(hit.get("redirects") or []),
        })
    return out


@st.cache_data(show_spinner=False, max_entries=128)
def agent_match(mention: str, country: str = "") -> dict | None:
    """Step 4 on its own: a generic mention matched to an ontology category.

    This is the pattern file plus a sentence encoder — no Wikipedia, no
    Elasticsearch, and nothing trained on the categories themselves.
    """
    ar = resources.get_actor_resolver()
    if ar is None:
        return None
    try:
        return ar.agent_matcher.short_text_to_agent(mention)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def agent_patterns() -> list[dict]:
    """The loaded category patterns: [{"pattern", "code_1", "code_2"}, ...].

    These come from `ngec/assets/PLOVER_agents.txt`, a plain text file of
    `pattern [CODE]` lines. It is the file the paper's Table 1 means by "edit a
    category mapping file".
    """
    ar = resources.get_actor_resolver()
    if ar is None:
        return []
    return list(getattr(ar.agent_matcher, "agents", []) or [])


def match_against_patterns(mention: str, extra: list[dict] | None = None,
                           top_k: int = 6) -> list[dict]:
    """Rank category patterns against a mention by embedding cosine similarity.

    `extra` holds patterns the visitor added on the customisation page. Only
    those are encoded here — the built-in patterns use the matrix the matcher
    already has — so adding a category is instant rather than a re-encode of
    several thousand strings.
    """
    ar = resources.get_actor_resolver()
    if ar is None:
        return []

    import numpy as np

    matcher = ar.agent_matcher
    query = matcher.trf.encode(mention, show_progress_bar=False)
    query = query / (np.linalg.norm(query) + 1e-12)

    rows: list[dict] = []

    matrix = getattr(matcher, "trf_matrix", None)
    builtin = list(getattr(matcher, "agents", []) or [])
    if matrix is not None and len(builtin):
        mat = np.asarray(matrix, dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        sims = (mat / (norms + 1e-12)) @ query
        for i in np.argsort(-sims)[: top_k * 3]:
            if i < len(builtin):
                agent = builtin[int(i)]
                rows.append({
                    "pattern": agent.get("pattern", ""),
                    "code_1": agent.get("code_1", ""),
                    "code_2": agent.get("code_2", ""),
                    "similarity": float(sims[int(i)]),
                    "source": "built-in",
                })

    if extra:
        texts = [e["pattern"] for e in extra]
        embeddings = matcher.trf.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=float)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        for agent, sim in zip(extra, embeddings @ query):
            rows.append({
                "pattern": agent["pattern"],
                "code_1": agent.get("code_1", ""),
                "code_2": agent.get("code_2", ""),
                "similarity": float(sim),
                "source": "yours",
            })

    rows.sort(key=lambda r: -r["similarity"])
    return rows[:top_k]


@st.cache_data(show_spinner=False, max_entries=32)
def definition_similarity(definitions: list[tuple[str, str]],
                          texts: list[tuple[str, str]]) -> list[dict]:
    """Cosine similarity between codebook definitions and documents.

    This is *not* a trained classifier — it is the embedding half of
    `setup/train_classifiers/codebook_llm/seed_candidates.py`, which uses
    nearest-neighbours-to-the-definition to find candidate documents worth
    annotating. It runs on CPU in a second and answers a useful question early:
    does this codebook separate this corpus at all, before spending money on
    annotation?

    definitions: [(label, definition_text), ...]
    texts:       [(title, document_text), ...]
    """
    import numpy as np

    clf, _ = resources.get_classifier()
    encoder = clf.encoder

    def_vectors = np.asarray(
        encoder.encode([d for _, d in definitions], show_progress_bar=False), dtype=float
    )
    doc_vectors = np.asarray(
        encoder.encode([t for _, t in texts], show_progress_bar=False), dtype=float
    )
    def_vectors /= np.linalg.norm(def_vectors, axis=1, keepdims=True) + 1e-12
    doc_vectors /= np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-12

    sims = doc_vectors @ def_vectors.T

    out = []
    for i, (title, _) in enumerate(texts):
        row = {"document": title}
        for j, (label, _) in enumerate(definitions):
            row[label] = float(sims[i, j])
        out.append(row)
    return out


def model_metadata() -> dict:
    """The classifier's self-describing metadata.json contents."""
    clf, _ = resources.get_classifier()
    return getattr(clf, "metadata", {}) or {}
