"""One function per pipeline step, each returning a plain JSON-safe dict.

These are the demo's whole interface to NGEC: a page collects inputs, calls one
of these, and draws the result. Nothing here touches Streamlit, so
`check_demo.py` exercises exactly what the pages run.

Every return value carries "seconds", the wall time of the step itself, "mode"
("gpu" or "cpu"), and "timing", a nested breakdown of where those seconds went
(see `timing.py`). Model loading happens in `resources.py`, and every step
fetches the models it needs *before* it starts its clock, so no step's seconds
include the load -- not even the first call in a process, which is the one that
pays for it. `resources.load_report()` reports loading and warm-up separately.

Every step takes an optional `mode`. Left None it follows the sidebar toggle,
which is what the pages want; passing it explicitly is how the Timing page and
`check_demo.py` run the same document through both modes in one process.
"""

from __future__ import annotations

import datetime as _dt
import time

from . import resources as R
from . import timing
from .timing import timed


# --- JSON hygiene ------------------------------------------------------------

def jsonable(obj):
    """Recursively convert numpy scalars, datetimes and spaCy objects to JSON types.

    Pipeline records hold numpy floats (Elasticsearch scores), datetimes (the
    resolved date) and, at the story level, spaCy artefacts. `st.json` and
    `json.dumps` choke on all three, so everything crossing out of this module
    goes through here.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if hasattr(obj, "tolist"):  # numpy scalar or array
        return jsonable(obj.tolist())
    if hasattr(obj, "item"):
        try:
            return jsonable(obj.item())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _wiki_url(title: str | None) -> str | None:
    if not title:
        return None
    return "https://en.wikipedia.org/wiki/" + str(title).replace(" ", "_")


# --- step 1: event classification -------------------------------------------

def classify(text: str, mode: str | None = None) -> dict:
    """Score one text against every PLOVER event type and the fired types' modes.

    Returns
    -------
    {"types": [{event_type, probability, threshold, fired}]   # sorted desc
     "modes": {event_type: [{mode, probability, threshold, fired}]},
     "mode": str, "timing": [rows], "seconds": float}

    Modes are only meaningful under a type that fired -- the mode models were
    trained conditional on their parent -- so they are computed for the fired
    types, or for the single highest-scoring type when nothing fires, so the
    page has something to show.

    Note the two senses of "mode" here: the argument is the compute mode, and
    `modes` in the result is PLOVER's event modes. The names come from
    different places and both are the obvious one in their own context.
    """
    compute_mode = mode or R.current_mode()
    clf, _ = R.get_classifier(compute_mode)  # before the clock: loading is not this step
    start = time.time()
    with R.compute(compute_mode), timing.collect() as collector:
        emb = clf._compute_embeddings([text])

        types = []
        for event_type, model in clf.type_models.items():
            prob = float(model.predict_proba(emb)[:, 1][0])
            threshold = float(clf._threshold_for(event_type))
            types.append({"event_type": event_type, "probability": prob,
                          "threshold": threshold, "fired": prob >= threshold})
        types.sort(key=lambda r: r["probability"], reverse=True)

        fired = [t["event_type"] for t in types if t["fired"]] or [types[0]["event_type"]]
        modes: dict[str, list[dict]] = {}
        for event_type in fired:
            mode_models = clf.mode_models.get(event_type) or {}
            rows = []
            for event_mode, model in mode_models.items():
                prob = float(model.predict_proba(emb)[:, 1][0])
                threshold = float(clf._threshold_for(f"{event_type}-{event_mode}",
                                                     is_mode=True))
                rows.append({"mode": event_mode, "probability": prob,
                             "threshold": threshold, "fired": prob >= threshold})
            rows.sort(key=lambda r: r["probability"], reverse=True)
            if rows:
                modes[event_type] = rows

    return {"types": types, "modes": modes, "mode": compute_mode,
            "timing": collector.rows(), "seconds": time.time() - start}


# --- step 2: attribute extraction -------------------------------------------

def extract_attributes(text: str, event_type: str, event_def: str | None = None,
                       event_mode: str = "", mode_def: str | None = None,
                       mode: str | None = None) -> dict:
    """Run the span extractor over one text, for one event type.

    Passing `event_def` (and optionally `mode_def`) skips the codebook lookup,
    which is how an event type the model has never seen gets extracted: the
    model reads a definition rather than recognising a fixed label set.

    Returns
    -------
    {"records": [{event_type, anchor_quote, actor[], recipient[], date[], location[]}],
     "prompt": str,          # the templated prompt the model was given
     "error": str | None,    # only when the empty list is a service failure
     "mode": str, "timing": [rows], "seconds": float}

    The list is empty when the model declines the document -- one text can yield
    zero, one, or several events. That is a normal answer, so `error` is set
    only when the model could not be reached at all (CPU mode with no
    llama-server), which is the case a page should complain about.
    """
    compute_mode = mode or R.current_mode()
    am = R.get_attribute_model(compute_mode)  # before the clock: loading is not this step
    start = time.time()
    prompt = ""
    records: list[dict] = []
    with R.compute(compute_mode), timing.collect() as collector:
        if am is not None:
            record = {"id": "demo", "orig_id": "demo", "event_text": text,
                      "event_type": event_type, "event_mode": event_mode or ""}
            if event_def:
                record["event_def"] = event_def
            if mode_def:
                record["mode_def"] = mode_def

            prompt = am.make_prompt(record)
            # process() explodes and drops records, so it returns a NEW list.
            out = am.process([dict(record)])
            records = [jsonable(r.get("attributes", {}))
                       for r in out if r.get("attributes")]

    error = R.generation_error(compute_mode) if not records else None
    return {"records": records, "prompt": prompt, "error": error,
            "mode": compute_mode, "timing": collector.rows(),
            "seconds": time.time() - start}


# --- step 3: Wikipedia lookup ------------------------------------------------

def resolve_entity(span: str, context: str = "", query_date: str = "today",
                   mode: str | None = None) -> dict:
    """Search Wikipedia for a span and score every candidate with the ranker.

    Mirrors `WikiMatcher.query_wiki` right up to the ranker -- possessive
    strip, NER expansion of the query against the context, and the second
    search over an alternative surface form -- and then, instead of returning
    only the winner, keeps the whole scored table so the page can show why the
    winner won. Skipping the expansion used to make this page disagree with
    step 4 on any bare surname: "Johnson" searched literally returns 200
    Johnsons, none of them the prime minister.

    Returns
    -------
    {"best": {title, short_desc, intro_para, url, ranker_score, wiki_reason} | None,
     "candidates": [{title, short_desc, url, ranker_score, raw_es_score,
                     exact_title_match, redirect_match, name_coverage,
                     cat_overlap, title_sim, context_sim_intro}],  # sorted desc
     "query": str,               # the term actually searched, after expansion
     "alt_query": str | None,    # the second surface form searched, if any
     "mode": str, "timing": [rows], "seconds": float}

    `query_date` is accepted for symmetry with `categorize_entity` and is not
    used: the Wikipedia ranker has no notion of when the article was true.
    """
    import re

    from ngec.actors.wiki_matcher import merge_ranked_results  # before the clock

    compute_mode = mode or R.current_mode()
    ar = R.get_actor_resolver(compute_mode)  # before the clock: loading is not this step
    wm = ar.wiki_matcher if ar is not None else None
    start = time.time()
    best = None
    candidates: list[dict] = []
    raw_span = re.sub(r"['’]s\s*$", "", span).strip()
    query = raw_span
    alt_term = ""
    with R.compute(compute_mode), timing.collect() as collector:
        good = []
        if wm is not None:
            # NER over the context, exactly as query_wiki does it: "Johnson"
            # in a sentence that also says "Boris Johnson" becomes the latter.
            if context:
                query = wm._expand_query(raw_span, context)
            hits = wm.wiki_searcher.search_wiki(query, max_results=200)
            for article in hits:
                article["from_alt_query"] = 0

            # The second search, over the surface form the pipeline did *not*
            # settle on. Its ES scores come from a different query and are not
            # on the primary's scale, which is what `from_alt_query` tells the
            # ranker. `_create_scoring_dataframe` reads that key off each
            # article with a default of 0, so before this the feature was
            # silently constant -- correct for a single search, but it meant
            # the demo never exercised it.
            alt_term = wm._pick_alt_query_term(query, [raw_span])
            if alt_term:
                alt_hits = wm.wiki_searcher.search_wiki(alt_term, max_results=200)
                for article in alt_hits:
                    article["from_alt_query"] = 1
                hits = merge_ranked_results(hits, alt_hits, 200)

            good = wm.wiki_searcher._trim_results(hits)
        if good:
            df = wm._create_scoring_dataframe(good, query, context=context,
                                              actor_desc="", country="")
            # _call_ranker adds a 'ranker_score' column to df in place and
            # returns the winning row (or None when nothing clears its bar).
            pick = wm._call_ranker(df, context=context)

            cols = ["title", "short_desc", "ranker_score", "raw_es_score",
                    "exact_title_match", "redirect_match", "name_coverage",
                    "cat_overlap", "title_sim", "context_sim_intro"]
            cols = [c for c in cols if c in df.columns]
            ranked = df.sort_values("ranker_score", ascending=False)
            for row in ranked[cols].to_dict("records"):
                row = jsonable(row)
                row["url"] = _wiki_url(row.get("title"))
                candidates.append(row)

            if pick is not None:
                best = jsonable({
                    "title": pick.get("title"),
                    "short_desc": pick.get("short_desc", ""),
                    "intro_para": pick.get("intro_para", ""),
                    "url": _wiki_url(pick.get("title")),
                    "ranker_score": float(pick.get("ranker_score", 0.0)),
                    "wiki_reason":
                        f"XGBoost (score={float(pick.get('ranker_score', 0)):.3f})"})

    return {"best": best, "candidates": candidates, "query": query,
            "alt_query": alt_term or None, "mode": compute_mode,
            "timing": collector.rows(), "seconds": time.time() - start}


# --- step 4: actor codes -----------------------------------------------------

def categorize_entity(span: str, context: str = "", query_date: str = "today",
                      agents_file: str | None = None,
                      mode: str | None = None) -> dict:
    """Resolve an actor mention to PLOVER role and country codes.

    `agents_file` points at a custom actor dictionary; None uses the one
    bundled with the package.

    Returns
    -------
    {code_1, code_2, country, wiki, url | None, source, used_wikipedia: bool,
     best_reason, description, all_code1s: [], all_code2s: [],
     "mode": str, "timing": [rows], "seconds": float}

    `source` says which route produced the codes: "country only", "similarity"
    (the agent dictionary), "Infobox" or "Wiki".
    """
    compute_mode = mode or R.current_mode()
    ar = R.get_actor_resolver(compute_mode, agents_file)  # before the clock
    start = time.time()
    out = {"code_1": "", "code_2": "", "country": "", "wiki": "", "url": None,
           "source": "", "used_wikipedia": False, "best_reason": "",
           "description": "", "all_code1s": [], "all_code2s": []}
    with R.compute(compute_mode), timing.collect() as collector:
        if ar is not None:
            # The cache is keyed on the mention *and* its context, so a changed
            # context no longer returns the previous answer. Clearing it is
            # still needed here: clicking the button again should measure the
            # work, not a cache hit.
            ar.cache_manager.clear()
            res = ar.actor_to_code(span, context=context, query_date=query_date) or {}
            out = {
                "code_1": res.get("code_1", ""),
                "code_2": res.get("code_2", ""),
                "country": res.get("country", ""),
                "wiki": res.get("wiki", ""),
                "url": _wiki_url(res.get("wiki")),
                "source": res.get("source", ""),
                "used_wikipedia": bool(res.get("wiki")),
                "best_reason": res.get("best_reason", ""),
                "description": res.get("description", ""),
                "all_code1s": res.get("all_code1s", []),
                "all_code2s": res.get("all_code2s", []),
            }

    return {**jsonable(out), "mode": compute_mode,
            "timing": collector.rows(), "seconds": time.time() - start}


# --- step 5a: dates ----------------------------------------------------------

def resolve_date(phrase: str, pub_date: str, mode: str | None = None) -> dict:
    """Resolve a date phrase against a publication date.

    Returns
    -------
    {resolved_date: ISO str | None, date_end: ISO str | None, granularity,
     date_type, reason, "mode": str, "timing": [rows], "seconds": float}

    `date_type` is exact / approximate / range / unresolved, and `granularity`
    is the unit the date is known to (day, week, month, quarter, year). Pure
    Python with no model in it, so the mode makes no difference here; it is
    reported for consistency with the other steps.
    """
    compute_mode = mode or R.current_mode()
    from ngec.formatter import _resolve_date  # before the clock: importing is not this step

    start = time.time()
    with R.compute(compute_mode), timing.collect() as collector:
        res = _resolve_date(date_string=phrase, ref_date=pub_date)
    out = {"resolved_date": jsonable(res.resolved_date),
           "date_end": jsonable(res.date_end),
           "granularity": res.granularity,
           "date_type": res.date_type,
           "reason": res.reason}
    return {**out, "mode": compute_mode, "timing": collector.rows(),
            "seconds": time.time() - start}


# --- step 5b: places ---------------------------------------------------------

def geoparse(text: str, mode: str | None = None) -> dict:
    """Find place names in a text and resolve them against geonames.

    Returns
    -------
    {"entities": [{search_name, resolved_placename, country_code3, country_name,
                   admin1_name, lat, lon, feature_code, score,
                   start_char, end_char}],
     "mode": str, "timing": [rows], "seconds": float}

    spaCy and the mordecai ranker are on the CPU in both modes, so the mode
    only changes how many threads torch may use here.
    """
    compute_mode = mode or R.current_mode()
    geo = R.get_geolocation()  # both before the clock: loading is not this step
    nlp = R.get_nlp_trf() if geo is not None else None
    start = time.time()
    entities: list[dict] = []
    with R.compute(compute_mode), timing.collect() as collector:
        if geo is not None:
            with timed("spaCy"):
                doc = nlp(text)
            story = {"id": "demo", "event_text": text}
            geo.process([story], [doc])

            keys = ["search_name", "resolved_placename", "country_code3",
                    "country_name", "admin1_name", "lat", "lon", "feature_code",
                    "score", "start_char", "end_char", "geonameid"]
            entities = [jsonable({k: ent.get(k) for k in keys})
                        for ent in story.get("geolocated_ents", [])]
    return {"entities": entities, "mode": compute_mode,
            "timing": collector.rows(), "seconds": time.time() - start}


def pick_location(location_span: str | None, entities: list[dict],
                  mode: str | None = None) -> dict:
    """Match the extracted location span to one of the geoparsed places.

    Returns
    -------
    {"event_loc": {geoparsed entity} | None, "reason": str,
     "mode": str, "timing": [rows], "seconds": float}

    `reason` says why nothing was picked: no search term, no geo entities, not
    enough word overlap, or not enough confidence in the geoparse.
    """
    compute_mode = mode or R.current_mode()
    from ngec.formatter import pick_event_loc  # before the clock: importing is not this step

    start = time.time()
    with R.compute(compute_mode), timing.collect() as collector:
        res = pick_event_loc(location_span, entities or [])
    return {"event_loc": jsonable(res.get("event_loc")),
            "reason": res.get("reason", ""),
            "mode": compute_mode, "timing": collector.rows(),
            "seconds": time.time() - start}


# --- the whole pipeline ------------------------------------------------------

def run_pipeline(text: str, pub_date: str, mode: str | None = None) -> dict:
    """Code one document end to end, the way `PloverCoder.process` does.

    Returns
    -------
    {"events": [JSON-safe formatter records],
     "story": {event_type, event_type_confidence, event_mode, geolocated_ents},
     "timing": {spacy, classify, geoparse, attributes, actors, format, total},
     "breakdown": [timing rows],   # the same six stages, with their internals
     "error": str | None,          # a service failure behind an empty result
     "mode": str,
     "dropped": ["TYPE-mode", ...]}   # candidates the attribute model declined

    A story with no event type above threshold returns no events; the attribute
    model can also decline a type the classifier fired on, which is what
    `dropped` records.

    "timing" is the flat six-stage summary the home page draws; "breakdown" is
    the nested version, whose depth-0 rows are those same six stages in order.
    """
    from ngec.utilities import stories_to_events

    compute_mode = mode or R.current_mode()

    # Every model this run needs, fetched before the clock starts. The loaders
    # are cached, but the first call in a mode loads and warms the model there,
    # which used to put three seconds of loading inside the "actors" stage.
    # `R.load_report()` is where that cost belongs, and it already reports it.
    nlp = R.get_nlp_trf()
    clf, _ = R.get_classifier(compute_mode)
    geo = R.get_geolocation()
    am = R.get_attribute_model(compute_mode)
    ar = R.get_actor_resolver(compute_mode)
    formatter = R.get_formatter()

    stage: dict[str, float] = {}
    total_start = time.time()

    story = {"id": "demo", "event_text": text, "pub_date": pub_date}

    with R.compute(compute_mode), timing.collect() as collector:
        t = time.time()
        with timed("spacy"):
            docs = list(nlp.pipe([text]))
        stage["spacy"] = time.time() - t

        t = time.time()
        with timed("classify"):
            clf.process([story])
        stage["classify"] = time.time() - t

        t = time.time()
        with timed("geoparse"):
            if geo is not None:
                geo.process([story], docs)
            else:
                story["geolocated_ents"] = []
        stage["geoparse"] = time.time() - t

        events = stories_to_events([story], docs)
        candidates = [f"{e['event_type']}-{e['event_mode']}" if e.get("event_mode")
                      else e["event_type"] for e in events]

        t = time.time()
        with timed("attributes"):
            events = am.process(events) if (am is not None and events) else []
        stage["attributes"] = time.time() - t

        survived = {f"{e['event_type']}-{e.get('event_mode')}" if e.get("event_mode")
                    else e["event_type"] for e in events}
        dropped = [c for c in candidates if c not in survived]

        t = time.time()
        with timed("actors"):
            if ar is not None and events:
                # The demo codes one document at a time, and the Timing page
                # compares modes over repeated runs, so a hit from the previous
                # click would report 0 s for work that really happens.
                ar.cache_manager.clear()
                events = ar.process(events)
        stage["actors"] = time.time() - t

        t = time.time()
        with timed("format"):
            if events:
                events = formatter.process(events, return_raw=True)
        stage["format"] = time.time() - t

    stage["total"] = time.time() - total_start

    story_summary = {
        "event_type": story.get("event_type", []),
        "event_type_confidence": jsonable(story.get("event_type_confidence", {})),
        "event_mode": story.get("event_mode", []),
        "geolocated_ents": jsonable(story.get("geolocated_ents", [])),
    }
    clean_events = [jsonable({k: v for k, v in e.items() if not k.startswith("_")})
                    for e in events]
    error = R.generation_error(compute_mode) if candidates and not events else None

    return {"events": clean_events, "story": story_summary,
            "timing": {k: round(v, 2) for k, v in stage.items()},
            "breakdown": collector.rows(), "mode": compute_mode,
            "error": error, "dropped": dropped}


# --- shared table shaping ----------------------------------------------------

def events_table(events: list[dict]) -> list[dict]:
    """One row per coded event, with the columns the main page shows.

    Not part of the step contract -- it is here rather than in the page so that
    `check_demo.py` fails if the shape of a formatter record changes.
    """
    rows = []
    for event in events:
        attributes = event.get("attributes") or {}

        def coded(key):
            raw = attributes.get(key) or []
            if isinstance(raw, str):  # the model sometimes emits a bare string
                raw = [raw]
            # "N/A" and nothing else: ActorResolver skips exactly those, so
            # filtering any harder would misalign the spans with their codes.
            spans = [s for s in raw if s != "N/A"]
            resolved = event.get(key) or []
            parts = []
            for i, span in enumerate(spans):
                code = resolved[i] if i < len(resolved) else {}
                bits = "".join(filter(None, [code.get("country", ""),
                                             code.get("code_1", "")]))
                parts.append(f"{span} → {bits}" if bits else span)
            return "; ".join(parts)

        loc = (event.get("event_location") or {}).get("event_loc") or {}
        date = event.get("date_resolved") or {}
        resolved_date = jsonable(date.get("resolved_date"))
        rows.append({
            "type": event.get("event_type", ""),
            "mode": event.get("event_mode", ""),
            "actor": coded("actor"),
            "recipient": coded("recipient"),
            "location": ", ".join(filter(None, [loc.get("resolved_placename", ""),
                                                loc.get("country_code3", "")])),
            "date": (resolved_date or "")[:10],
            "quote": attributes.get("anchor_quote", ""),
        })
    return rows
