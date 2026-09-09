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
import re
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

# The six attributes the model was trained to return. Anything else it returns
# is passed through untouched, which is what makes the "extra attributes" box
# on the page work at all.
STANDARD_ATTRIBUTES = ("event_type", "anchor_quote", "actor", "recipient",
                       "date", "location")

# The last line of the OUTPUT FORMAT block, identical in both prompt formats
# (see _make_system_content_short / _make_system_content_v5 in
# ngec/attribute_model.py). Extra fields are inserted after it so they read
# like the six the model already knows.
_FORMAT_ANCHOR = '"location": "where occurred OR N/A"'

# Every message of a rendered chat prompt: <|im_start|>role\n...<|im_end|>.
_CHAT_MESSAGE = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"


def _clean_fields(fields: list[str] | str | None) -> list[str]:
    """Field names from the page's comma-separated box, tidied into a list."""
    import re

    if not fields:
        return []
    if isinstance(fields, str):
        fields = fields.split(",")
    cleaned = []
    for field in fields:
        # A JSON key with a quote or a newline in it would break the prompt's
        # output block; spaces become underscores so "according to whom" works.
        name = re.sub(r"[^A-Za-z0-9_ -]", "", str(field)).strip().replace(" ", "_")
        if name and name not in cleaned:
            cleaned.append(name)
    return cleaned


def _conversation_from_prompt(prompt: str) -> list[dict]:
    """A rendered prompt string back into the [{role, content}] the engines take.

    The llamacpp and transformers backends render the chat template
    themselves, so an edited prompt has to be handed back to them as messages,
    not as text. Round-tripping an unedited prompt reproduces it exactly, and
    edits inside the role blocks survive; an edit to the template markers
    themselves does not. A prompt with no markers at all (some other model's
    template) is sent as a single user message rather than dropped.
    """
    import re

    messages = [{"role": role, "content": content}
                for role, content in re.findall(_CHAT_MESSAGE, prompt, re.DOTALL)]
    return messages or [{"role": "user", "content": prompt}]


def _request_extra_fields(conversation: list[dict], fields: list[str]) -> list[dict]:
    """Add fields to the OUTPUT FORMAT block of whichever message states it."""
    added = ",\n".join(f'    "{f}": "{f.replace("_", " ")} OR N/A"' for f in fields)
    out, done = [], False
    for message in conversation:
        content = message["content"]
        if not done and _FORMAT_ANCHOR in content:
            content = content.replace(_FORMAT_ANCHOR,
                                      _FORMAT_ANCHOR + ",\n" + added, 1)
            done = True
        out.append({**message, "content": content})
    if not done:  # some other model's prompt: ask in a sentence instead
        out[-1] = {**out[-1],
                   "content": out[-1]["content"] + "\nAlso extract, for each "
                   f"event: {', '.join(fields)}."}
    return out


def _schema_with(fields: list[str]) -> dict:
    """ATTRIBUTE_SCHEMA plus the extra fields, for backends that constrain decoding.

    llama-server turns the schema into a decoding grammar, and that grammar
    forbids any key the schema does not list. So asking for a field in the
    prompt without adding it here does not merely lose the field: the model,
    unable to emit it, tends to emit a second, spurious event instead. The two
    have to move together.
    """
    import copy

    from ngec.attributes.schema import ATTRIBUTE_SCHEMA

    schema = copy.deepcopy(ATTRIBUTE_SCHEMA)
    for field in fields:
        schema["items"]["properties"][field] = {"type": "string"}
        schema["items"]["required"].append(field)
    return schema


def _split_spans(event: dict) -> dict:
    """Semicolon-separated spans into lists, as `AttributeModel.process` does.

    Only for the vllm path below, which parses its own JSON. The engine path
    goes through `parse_response`, which has already done this.
    """
    for key in ("actor", "recipient", "date", "location"):
        value = event.get(key)
        if isinstance(value, str):
            event[key] = [v.strip() for v in value.split(";")]
        elif isinstance(value, list):
            event[key] = [str(v).strip() for v in value]
    return event


def _generate_attributes(am, conversation: list[dict], prompt: str,
                         schema: dict | None) -> list[dict]:
    """One document through whichever generation path this backend uses.

    `AttributeModel.process` is the production path and the one the page uses
    by default, but it always sends the stock prompt and the stock schema. A
    customised prompt has to go around it, so this calls the same public
    entry points process() calls -- the engine for llamacpp/transformers,
    `call_llm_batch` for vllm -- and keeps whatever keys come back.

    (The package has no way to hand `process()` a prompt or a schema; that is
    the entry point it should grow if this stops being demo-only.)
    """
    from ngec.attributes.schema import parse_response

    if am.engine is not None:
        raw = am.engine.generate([conversation], schema=schema)[0]
        events, _failure = parse_response(raw)
        return events
    # vllm: generates from the rendered string and parses the JSON itself.
    events = am.call_llm_batch([prompt])[0]
    if isinstance(events, dict):
        events = [events]
    return [_split_spans(e) for e in events if isinstance(e, dict)]


def attribute_prompt(text: str, event_type: str, event_def: str | None = None,
                     event_mode: str = "", mode_def: str | None = None,
                     extra_fields: list[str] | str | None = None,
                     mode: str | None = None) -> str:
    """The prompt `extract_attributes` would send, for the page to show and edit.

    Empty when the model could not be loaded (the tokenizer renders the chat
    template, so there is no prompt without it).
    """
    am = R.get_attribute_model(mode or R.current_mode())
    if am is None:
        return ""
    record = {"id": "demo", "orig_id": "demo", "event_text": text,
              "event_type": event_type, "event_mode": event_mode or ""}
    if event_def:
        record["event_def"] = event_def
    if mode_def:
        record["mode_def"] = mode_def
    prompt = am.make_prompt(record)
    fields = _clean_fields(extra_fields)
    if not fields:
        return prompt
    conversation = _request_extra_fields(_conversation_from_prompt(prompt), fields)
    return am.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)


def extract_attributes(text: str, event_type: str, event_def: str | None = None,
                       event_mode: str = "", mode_def: str | None = None,
                       mode: str | None = None,
                       extra_fields: list[str] | str | None = None,
                       prompt_override: str | None = None) -> dict:
    """Run the span extractor over one text, for one event type.

    Passing `event_def` (and optionally `mode_def`) skips the codebook lookup,
    which is how an event type the model has never seen gets extracted: the
    model reads a definition rather than recognising a fixed label set.

    `extra_fields` asks the model for attributes beyond the six it was trained
    on ("source_attribution", "participant_count"); `prompt_override` replaces
    the whole prompt with an edited one (from `attribute_prompt` above). Either
    one steps outside `AttributeModel.process`, which always sends the stock
    prompt -- see `_generate_attributes`. With an edited prompt the output is
    not constrained to a JSON schema at all, since there is no telling which
    fields the edited text asks for; the parser salvages what it can.

    Returns
    -------
    {"records": [{event_type, anchor_quote, actor[], recipient[], date[], location[]}],
     "prompt": str,          # the templated prompt the model was given
     "error": str | None,    # only when the empty list is a service failure
     "mode": str, "timing": [rows], "seconds": float}

    A record carries any extra keys the model returned, in the order it
    returned them; the page turns those into extra columns.

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
    fields = _clean_fields(extra_fields)
    with R.compute(compute_mode), timing.collect() as collector:
        if am is not None:
            record = {"id": "demo", "orig_id": "demo", "event_text": text,
                      "event_type": event_type, "event_mode": event_mode or ""}
            if event_def:
                record["event_def"] = event_def
            if mode_def:
                record["mode_def"] = mode_def

            prompt = am.make_prompt(record)
            if not fields and not prompt_override:
                # The production path: process() explodes and drops records, so
                # it returns a NEW list.
                out = am.process([dict(record)])
                records = [jsonable(r.get("attributes", {}))
                           for r in out if r.get("attributes")]
            else:
                if prompt_override:
                    prompt = prompt_override
                    conversation = _conversation_from_prompt(prompt)
                    schema = None
                else:
                    conversation = _request_extra_fields(
                        _conversation_from_prompt(prompt), fields)
                    prompt = am.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                    schema = _schema_with(fields)
                records = [jsonable(event) for event
                           in _generate_attributes(am, conversation, prompt, schema)]

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

def parse_pub_date(text: str | None) -> str | None:
    """Read a typed publication date ("June 11, 2024", "last Friday") as an ISO date.

    The pipeline gets each story's publication date from its metadata; in the
    demo someone types it, so it has to accept whatever they type. `_resolve_date`
    already understands all of these forms -- it is what resolves the date
    *phrase* -- so we run it against today rather than write a second parser,
    and read its "unresolved" flag as "this text is not a date". Returns None
    in that case, for the caller to report however it likes.
    """
    from ngec.formatter import _resolve_date

    if text is None or not str(text).strip():
        return None
    res = _resolve_date(date_string=str(text).strip(), ref_date=_dt.date.today())
    if res.resolved_date is None or res.date_type == "unresolved":
        return None
    return res.resolved_date.isoformat()[:10]


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
                # No arrow between the span and its code: readers took it for
                # the direction of the event rather than "resolved to".
                parts.append(f"{span} ({bits})" if bits else span)
            return "; ".join(parts)

        loc = (event.get("event_location") or {}).get("event_loc") or {}
        date = event.get("date_resolved") or {}
        resolved_date = jsonable(date.get("resolved_date"))
        rows.append({
            # The document the event came from. `id` carries a per-event suffix
            # and changes run to run; `orig_id` is the stable key, and it is
            # what a bulk uploader has to match rows back to their file.
            "id": event.get("orig_id", ""),
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


def _quoted(text: str) -> str:
    """A span in typographic quotes, marking it as verbatim from the document.

    The per-event table mixes text the model copied out of the story with codes
    and resolved values it produced. Quoting the copied text is the cheapest way
    to say which is which.
    """
    return f"“{text}”" if text else ""


def event_fields(event: dict) -> list[tuple[str, str]]:
    """One coded event as (field, value) pairs, for the per-event view.

    The span the attribute model found and the code the actor resolver gave it
    are separate fields rather than one "span → code" string: they are the
    output of two different steps, and readers took the arrow between them for
    the direction of the event.

    The standard rows are always shown, empty ones marked "—": a missing
    Location row read as a rendering bug rather than as "nothing was resolved",
    and a reader cannot tell which fields a record is supposed to have from a
    table that only shows the ones that came out filled. The "entity" rows are
    the exception and are dropped when empty -- they only exist when
    Elasticsearch matched a Wikipedia article, and a row of dashes on every
    event with ES down says nothing.

    Event type and mode lead the table. They used to sit in the heading above
    it, which left the heading carrying coded output that the rest of the
    record showed as rows; the heading now only numbers the events. The mode is
    often empty -- an event type can fire with no mode -- and shows as "—"
    like any other empty field.

    The spans the attribute model copied out of the document are quoted and the
    codes and resolved values are not, so it is clear on sight which cells are
    verbatim text and which are the coder's own answer.
    """
    attributes = event.get("attributes") or {}
    fields: list[tuple[str, str]] = [
        ("Event type", event.get("event_type", "")),
        ("Mode", event.get("event_mode", "")),
    ]

    for key, label in [("actor", "Actor"), ("recipient", "Recipient")]:
        raw = attributes.get(key) or []
        if isinstance(raw, str):  # the model sometimes emits a bare string
            raw = [raw]
        # "N/A" and nothing else: ActorResolver skips exactly those, so
        # filtering any harder would misalign the spans with their codes.
        spans = [s for s in raw if s != "N/A"]
        coded = event.get(key) or []
        codes = ["".join(filter(None, [c.get("country", ""), c.get("code_1", "")]))
                 for c in coded]
        wikis = [c.get("wiki", "") for c in coded]
        fields += [(f"{label} span", "; ".join(_quoted(s) for s in spans)),
                   (f"{label} code", "; ".join(c for c in codes if c)),
                   (f"{label} entity", "; ".join(w for w in wikis if w))]

    loc = (event.get("event_location") or {}).get("event_loc") or {}
    fields.append(("Location", ", ".join(filter(None, [
        loc.get("resolved_placename", ""), loc.get("country_code3", "")]))))

    date = event.get("date_resolved") or {}
    resolved = (jsonable(date.get("resolved_date")) or "")[:10]
    # "day" and "exact" are the uninteresting case; say so only when the
    # resolution was fuzzier than that.
    hedges = [h for h in [date.get("granularity"), date.get("date_type")]
              if h and h not in ("day", "exact")]
    fields.append(("Date", f"{resolved} ({', '.join(hedges)})"
                   if resolved and hedges else resolved))

    fields.append(("Anchor quote", _quoted(attributes.get("anchor_quote", ""))))
    return [(name, value or "—") for name, value in fields
            if value or not name.endswith("entity")]


# --- bulk coding -------------------------------------------------------------

# The demo box is shared and 200 documents is already several minutes of GPU
# time; a real corpus run belongs in `PloverCoder`, not in a web page.
BULK_MAX_DOCS = 200

# What a text column may be called. "text" is what the page asks for;
# "event_text" is what the pipeline itself calls the field, so a file exported
# from an earlier NGEC run loads without being renamed first.
TEXT_COLUMNS = ("text", "event_text")

# Likewise for the date. "publication_date" is the documented name; the other
# two are what people's spreadsheets usually call it.
DATE_COLUMNS = ("publication_date", "date", "pub_date")

# The six components, in the order `run_pipeline_bulk` calls them. The progress
# callback gets one of these names after each.
BULK_STAGES = ("spacy", "classify", "geoparse", "attributes", "actors", "format")


def read_documents(data: bytes, filename: str = "") -> tuple[list[dict], list[str]]:
    """Parse an uploaded CSV or JSONL into documents the pipeline can take.

    Returns
    -------
    documents : [{"id": str, "text": str, "pub_date": "YYYY-MM-DD"}, ...]
    problems : ["row 4: no text", ...]

    A row the pipeline cannot use is reported and skipped rather than raised:
    one unparseable date in a spreadsheet should not cost someone the upload.
    The only fatal case is a file with no text or no date column at all, which
    raises ValueError naming the columns that *were* found -- usually a
    misspelled header, and the message is what tells the user that.

    Ids come from an `id` column when there is one and are the row number
    otherwise. They end up in every event's `orig_id`, which is how a coded
    event is traced back to the document it came from.
    """
    import io

    import pandas as pd

    name = (filename or "").lower()
    try:
        if name.endswith((".jsonl", ".ndjson", ".json")):
            frame = pd.read_json(io.BytesIO(data), lines=True)
        else:
            frame = pd.read_csv(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - a malformed file is a user error
        raise ValueError(f"Could not read {filename or 'the file'}: {exc}") from exc

    # Match headers case- and whitespace-insensitively: "Text" and " date "
    # are the same column to a person and should be to us.
    columns = {str(c).strip().lower(): c for c in frame.columns}
    found = ", ".join(str(c) for c in frame.columns) or "no columns"
    text_col = next((columns[c] for c in TEXT_COLUMNS if c in columns), None)
    if text_col is None:
        raise ValueError(f"No 'text' column. The file has: {found}.")
    date_col = next((columns[c] for c in DATE_COLUMNS if c in columns), None)
    if date_col is None:
        raise ValueError("No 'publication_date' column (or 'date' / 'pub_date'). "
                         f"The file has: {found}.")
    id_col = columns.get("id")

    problems: list[str] = []
    if len(frame) > BULK_MAX_DOCS:
        problems.append(f"{len(frame):,} rows uploaded; only the first "
                        f"{BULK_MAX_DOCS} are coded — the demo box is shared.")
        frame = frame.head(BULK_MAX_DOCS)

    # Parsed as a whole column, which is what makes an ambiguous date safe:
    # pandas infers one format and applies it to every row, rather than reading
    # 03/04 as March in one row and April in the next. A row that does not fit
    # that format comes back as NaT and is reported below.
    dates = pd.to_datetime(frame[date_col], errors="coerce")

    documents: list[dict] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        number = position + 1  # the row as a person counts them, header aside
        text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
        if not text:
            problems.append(f"row {number}: no text")
            continue
        date = dates.iloc[position]
        if pd.isna(date):
            problems.append(f"row {number}: could not read the date "
                            f"{str(row[date_col])[:30]!r} — the column's date "
                            "format is taken from the first row")
            continue
        doc_id = str(row[id_col]).strip() if id_col is not None else ""
        documents.append({"id": doc_id or str(number), "text": text,
                          "pub_date": date.strftime("%Y-%m-%d")})

    if not documents and not problems:
        problems.append("The file has no rows.")
    return documents, problems


def run_pipeline_bulk(docs: list[dict], mode: str | None = None,
                      progress=None) -> dict:
    """Code many documents end to end, one component at a time.

    `docs` are what `read_documents` returns: {"id", "text", "pub_date"}.

    This is `run_pipeline` over a list rather than in a loop, because that is
    what the pipeline actually is -- every component takes and returns a list of
    dicts, and vllm batches a hundred prompts far better than it answers a
    hundred separate calls.

    `progress`, if given, is called as `progress(stage, done, total)` after each
    component finishes, where `stage` is one of `BULK_STAGES`. It is the page's
    hook for a progress bar; a bulk run is minutes long and a silent spinner is
    not enough.

    Returns
    -------
    {"events": [JSON-safe formatter records],
     "dropped": [{"id", "event_type", "event_mode"}, ...],  # declined candidates
     "n_docs": int, "seconds": float, "per_doc_seconds": float,
     "timing": {spacy, classify, geoparse, attributes, actors, format, total},
     "breakdown": [timing rows],
     "error": str | None, "mode": str}
    """
    from ngec.utilities import stories_to_events

    compute_mode = mode or R.current_mode()

    # Fetched before the clock starts, exactly as in `run_pipeline`: the first
    # call in a mode loads and warms the models, and that cost belongs to
    # `R.load_report()`, not to whichever stage happened to trigger it.
    nlp = R.get_nlp_trf()
    clf, _ = R.get_classifier(compute_mode)
    geo = R.get_geolocation()
    am = R.get_attribute_model(compute_mode)
    ar = R.get_actor_resolver(compute_mode)
    formatter = R.get_formatter()

    def done(stage: str) -> None:
        if progress is not None:
            progress(stage, BULK_STAGES.index(stage) + 1, len(BULK_STAGES))

    stage: dict[str, float] = {}
    total_start = time.time()

    stories = [{"id": str(doc.get("id") or n + 1), "event_text": doc["text"],
                "pub_date": doc.get("pub_date", "")}
               for n, doc in enumerate(docs)]

    with R.compute(compute_mode), timing.collect() as collector:
        t = time.time()
        with timed("spacy"):
            spacy_docs = list(nlp.pipe([s["event_text"] for s in stories]))
        stage["spacy"] = time.time() - t
        done("spacy")

        t = time.time()
        with timed("classify"):
            clf.process(stories)
        stage["classify"] = time.time() - t
        done("classify")

        t = time.time()
        with timed("geoparse"):
            if geo is not None:
                geo.process(stories, spacy_docs)
            else:
                for story in stories:
                    story["geolocated_ents"] = []
        stage["geoparse"] = time.time() - t
        done("geoparse")

        events = stories_to_events(stories, spacy_docs)
        # A candidate is one (document, type, mode) the classifier fired on.
        # The attribute model may find no event for it, and comparing the two
        # sets afterwards is how we know which ones it declined.
        candidates = [{"id": e["orig_id"], "event_type": e["event_type"],
                       "event_mode": e.get("event_mode", "")} for e in events]

        t = time.time()
        with timed("attributes"):
            # `process` explodes multi-event extractions, so this returns a new
            # and possibly longer list -- one record per extracted event.
            events = am.process(events) if (am is not None and events) else []
        stage["attributes"] = time.time() - t
        done("attributes")

        survived = {(e["orig_id"], e["event_type"], e.get("event_mode", ""))
                    for e in events}
        dropped = [c for c in candidates
                   if (c["id"], c["event_type"], c["event_mode"]) not in survived]

        t = time.time()
        with timed("actors"):
            if ar is not None and events:
                # Cleared once for the run, not per document: within one bulk
                # job the cache is doing its job (the same actor recurs across
                # stories), and clearing it between documents would only make
                # the run slower and the timing a lie.
                ar.cache_manager.clear()
                events = ar.process(events)
        stage["actors"] = time.time() - t
        done("actors")

        t = time.time()
        with timed("format"):
            if events:
                events = formatter.process(events, return_raw=True)
        stage["format"] = time.time() - t
        done("format")

    stage["total"] = time.time() - total_start

    clean_events = [jsonable({k: v for k, v in e.items() if not k.startswith("_")})
                    for e in events]
    error = R.generation_error(compute_mode) if candidates and not events else None

    return {"events": clean_events, "dropped": dropped, "n_docs": len(stories),
            "seconds": stage["total"],
            "per_doc_seconds": stage["total"] / max(len(stories), 1),
            "timing": {k: round(v, 2) for k, v in stage.items()},
            "breakdown": collector.rows(), "error": error, "mode": compute_mode}
