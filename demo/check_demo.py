#!/usr/bin/env python
"""Run every function in `ngec_demo.steps` on real inputs, with no Streamlit.

This is the demo's test: the pages contain no pipeline logic, so if this passes
the pages work. It loads the real models and talks to the real Elasticsearch, so
it takes a couple of minutes per mode.

    cd demo && env -u LD_LIBRARY_PATH uv run --extra cu12 --extra vllm \\
        --group demo-app python check_demo.py --mode all

`--mode gpu|cpu|all` (default: every mode this machine can run). Running both in
one process is the point: the models stay loaded per mode, so the side-by-side
table at the end compares the same document coded two ways.

Exits non-zero on the first failure, and prints a timing per step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngec_demo import resources as R  # noqa: E402
from ngec_demo import steps  # noqa: E402
from ngec_demo.examples import DATE_PHRASES, DOCUMENTS, ENTITY_SPANS, SHORT_TEXTS  # noqa: E402

FAILURES: list[str] = []
TIMINGS: list[tuple[str, float]] = []
# {mode: run_pipeline breakdown rows}, for the side-by-side table.
BREAKDOWNS: dict[str, list[dict]] = {}

# Slack when comparing a step's own wall time with the sum of the calls
# measured inside it: the two clocks are read a few microseconds apart, and a
# GC pause between them is not a bug.
SLACK = 0.05


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        FAILURES.append(f"{name}: {detail or 'failed'}")
        print(f"    FAIL {detail}")


def check_timing(name: str, out: dict) -> None:
    """The timing rows have to nest: no child longer than its parent, and no
    top-level row longer than the step that collected it."""
    rows = out.get("timing")
    if rows is None:
        check(name, False, "no 'timing' rows")
        return
    seconds = out.get("seconds", 0.0)
    by_path = {row["path"]: row["seconds"] for row in rows}
    for row in rows:
        if row["depth"] == 0:
            check(name, row["seconds"] <= seconds + SLACK,
                  f"{row['path']} took {row['seconds']:.2f}s inside a "
                  f"{seconds:.2f}s step")
        else:
            parent = row["path"].rsplit("/", 1)[0]
            if parent in by_path:
                check(name, row["seconds"] <= by_path[parent] + SLACK,
                      f"{row['path']} ({row['seconds']:.2f}s) is longer than "
                      f"{parent} ({by_path[parent]:.2f}s)")


def run(name: str, fn, *args, **kwargs):
    """Call one step, print a line, record its self-reported time."""
    print(f"\n[{name}]")
    start = time.time()
    try:
        out = fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append(f"{name}: raised")
        return None
    wall = time.time() - start
    reported = out.get("seconds", float("nan"))
    TIMINGS.append((name, reported))
    print(f"    {reported:.2f}s reported / {wall:.2f}s wall")
    check(name, "seconds" in out, "no 'seconds' key")
    check(name, "mode" in out, "no 'mode' key")
    check_timing(name, out)
    try:
        json.dumps(out)
    except TypeError as exc:
        check(name, False, f"not JSON-serialisable: {exc}")
    return out


# --- devices -----------------------------------------------------------------

def sentence_transformers(mode: str) -> dict[str, object]:
    """Every SentenceTransformer the demo's models reach, by a readable name.

    This is the check that CPU mode is honest. `gpu=False` does not put the
    encoders on the CPU -- sentence-transformers takes the card whenever it can
    see one -- so the demo passes an explicit device, and this is where that is
    verified rather than assumed.
    """
    found: dict[str, object] = {}
    clf, _ = R.get_classifier(mode)
    found["classifier encoder"] = clf.encoder

    resolver = R.get_actor_resolver(mode)
    if resolver is not None:
        found["resolver trf"] = resolver.trf
        found["agent matcher trf"] = resolver.agent_matcher.trf
        found["wiki encoder"] = resolver.wiki_matcher.trf
        found["actor-sim encoder"] = resolver.wiki_matcher.actor_sim
        found["wiki parser agent trf"] = resolver.wiki_parser.agent_matcher.trf
    return found


def check_devices(mode: str) -> None:
    expected = R.device(mode)
    print(f"\n[devices] expecting {expected}")
    for name, model in sentence_transformers(mode).items():
        actual = str(getattr(model, "device", "?"))
        print(f"    {actual:8} {name}")
        check("devices", actual.split(":")[0] == expected,
              f"{name} is on {actual}, not {expected}")


# --- one mode ----------------------------------------------------------------

def run_mode(mode: str) -> None:
    print("\n" + "=" * 72)
    print(f"MODE: {mode}  (backend {R.backend(mode)}, encoders on {R.device(mode)})")
    print("=" * 72)

    health = R.health(mode)
    for name, row in health.items():
        print(f"  {'ok ' if row['ok'] else 'DOWN'} {name}: {row['detail']}")
    es_up = R.get_es() is not None
    # A dead llama-server is a missing service, not a broken demo: say so once
    # and stop asserting that the model extracted anything.
    llm_up = health.get("llama-server", {"ok": True})["ok"]
    if not llm_up:
        print("\n  llama-server is not reachable, so the extraction checks are "
              "skipped for this mode.\n  Start it (see demo/deploy/README.md) or "
              "set NGEC_DEMO_CPU_BACKEND=transformers.")

    # A step must not time its own model loading. This is the first call into
    # `mode`, so the classifier is still cold: if the loader is inside the
    # stopwatch the first call reports ~10 s and the second ~0.7 s.
    cold = steps.classify(SHORT_TEXTS[0][1], mode=mode)["seconds"]
    warm = steps.classify(SHORT_TEXTS[0][1], mode=mode)["seconds"]
    print(f"\n[cold vs warm] classify reported {cold:.2f}s then {warm:.2f}s")
    check("cold start", abs(cold - warm) < 2.0,
          f"the first classify in {mode} reported {cold:.2f}s and the second "
          f"{warm:.2f}s -- model loading is being timed inside the step")

    check_devices(mode)

    # Load the models before the timed steps rather than inside the first one
    # that happens to need them: vllm's CUDA-graph capture is about a minute,
    # and that belongs in the load report, not in step 2's stopwatch. This is
    # the same call the sidebar's "Load models" button makes, so checking it
    # here costs nothing extra -- but it has to come after the cold-vs-warm
    # check above, which needs one model still cold to mean anything.
    print("\n[load_all]")
    seconds = R.load_all(mode, on_step=lambda name: print(f"    loading {name}"))
    print(f"    {seconds:.1f}s")
    check("load_all", R.is_loaded(mode), "is_loaded() is False after load_all()")

    # --- step 1
    label, text = SHORT_TEXTS[0]
    out = run("classify", steps.classify, text, mode=mode)
    if out:
        top = out["types"][0]
        print(f"    top: {top['event_type']} {top['probability']:.2f} "
              f"(threshold {top['threshold']:.2f}, fired={top['fired']})")
        check("classify", out["types"] and out["types"][0]["probability"] >= 0,
              "no type scores")
        check("classify", isinstance(out["modes"], dict), "modes is not a dict")
        print(f"    modes for {list(out['modes'])}")

    # --- step 2
    event_type = out["types"][0]["event_type"] if out else "PROTEST"
    out = run("extract_attributes", steps.extract_attributes, text, event_type,
              mode=mode)
    if out:
        check("extract_attributes", bool(out["prompt"]), "empty prompt")
        for record in out["records"]:
            print("    " + json.dumps({k: v for k, v in record.items()
                                       if k != "anchor_quote"}))
        if llm_up:
            check("extract_attributes", len(out["records"]) >= 1,
                  f"no events extracted for {event_type}")
        elif out["error"]:
            print(f"    error reported: {out['error']}")

    # An event type outside the codebook, extracted from a written definition.
    out = run("extract_attributes (custom definition)", steps.extract_attributes,
              SHORT_TEXTS[2][1], "DETAIN",
              event_def="One party takes another into custody or arrests them.",
              mode=mode)
    if out:
        print(f"    {len(out['records'])} record(s)")

    # The prompt the page shows in "Advanced: edit the prompt", edited by hand
    # and sent back. The document and definition are the ECAV example from
    # pages/step2.py -- an ontology from another project -- kept in step with
    # that page's EXAMPLES by hand.
    ecav_text = ("Supporters of the opposition Unity Party blocked the main road "
                 "into Kisumu on Thursday, three days after the parliamentary "
                 "election, accusing the electoral commission of tampering with "
                 "the count. Police fired tear gas to disperse the crowd of about "
                 "2,000, a party spokesman said.")
    # TODO(andy): replace with the verbatim definition from the ECAV codebook
    # (Daxecker, Amicarelli & Jung 2019), here and in pages/step2.py. This is a
    # shortened paraphrase of that page's draft definition and has not been
    # checked against the codebook.
    ecav_def = ("Public acts of mobilization, contestation, or coercion by state "
                "or non-state actors used to affect the electoral process, or "
                "arising in the context of electoral competition. The party "
                "carrying out the act is the ACTOR and the party it is directed "
                "against is the RECIPIENT.")
    prompt = steps.attribute_prompt(ecav_text, "ELECTORAL_CONTENTION",
                                    event_def=ecav_def, mode=mode)
    print(f"\n[attribute_prompt] {len(prompt)} characters")
    check("attribute_prompt", "ELECTORAL_CONTENTION" in prompt,
          "the event type is not in the prompt")
    check("attribute_prompt", ecav_text[:40] in prompt,
          "the document is not in the prompt")

    anchor = '"location": "where occurred OR N/A"'
    edited = prompt.replace(
        anchor, anchor + ',\n    "participant_count": "how many took part OR N/A"')
    check("attribute_prompt", edited != prompt,
          "the OUTPUT FORMAT block has moved: the edit changed nothing")
    out = run("extract_attributes (edited prompt)", steps.extract_attributes,
              ecav_text, "ELECTORAL_CONTENTION", event_def=ecav_def,
              prompt_override=edited, mode=mode)
    if out:
        check("extract_attributes (edited prompt)", out["prompt"] == edited,
              "the edited prompt is not the one reported as sent")
        for record in out["records"]:
            print("    " + json.dumps({k: v for k, v in record.items()
                                       if k != "anchor_quote"}))
        if llm_up:
            check("extract_attributes (edited prompt)", len(out["records"]) >= 1,
                  "no events extracted from the ECAV example")
            # Whether the model honours the added field is its own business
            # (it usually does), so this is reported rather than checked.
            extra = sorted({key for record in out["records"] for key in record
                            if key not in steps.STANDARD_ATTRIBUTES})
            print(f"    keys beyond the standard six: {extra or 'none'}")

    # --- step 3
    span, context = ENTITY_SPANS[0]
    out = run("resolve_entity", steps.resolve_entity, span, context=context,
              mode=mode)
    if out:
        if es_up:
            check("resolve_entity", bool(out["candidates"]), "no candidates")
            check("resolve_entity", out["best"] is not None, "no article picked")
        if out["best"]:
            print(f"    best: {out['best']['title']} "
                  f"({out['best']['ranker_score']:.3f}) {out['best']['url']}")
        for cand in out["candidates"][:3]:
            print(f"      {cand['ranker_score']:.3f}  {cand['title']}")
        scores = [c["ranker_score"] for c in out["candidates"]]
        check("resolve_entity", scores == sorted(scores, reverse=True),
              "candidates are not sorted by ranker_score")

    # A bare surname only resolves because the step expands the query against
    # the context, the way WikiMatcher.query_wiki does. Without that this page
    # disagreed with step 4 and returned a novel character.
    surname_context = (
        "Boris Johnson faced questions in the House of Commons over parties held "
        "in Downing Street during lockdown. Johnson, who has been PM since 2019, "
        "has faced criticism.")
    out = run("resolve_entity (bare surname)", steps.resolve_entity, "Johnson",
              context=surname_context, mode=mode)
    if out:
        print(f"    query: {out['query']!r} alt: {out['alt_query']!r}")
        if out["best"]:
            print(f"    best: {out['best']['title']} "
                  f"({out['best']['ranker_score']:.3f})")
        if es_up:
            check("resolve_entity (bare surname)", out["query"] == "Boris Johnson",
                  f"expanded to {out['query']!r}, not 'Boris Johnson'")
            check("resolve_entity (bare surname)",
                  (out["best"] or {}).get("title") == "Boris Johnson",
                  f"picked {(out['best'] or {}).get('title')!r}, not 'Boris Johnson'")

    # --- step 4
    out = run("categorize_entity", steps.categorize_entity, span, context=context,
              query_date="2023-03-15", mode=mode)
    if out:
        print(f"    {out['country']}/{out['code_1']}/{out['code_2']} "
              f"via {out['source']} (wiki={out['wiki'] or '-'})")
        if es_up:
            check("categorize_entity", bool(out["country"] or out["code_1"]),
                  "no codes at all")

    # A custom agents file has to be accepted, even when it is the bundled one.
    from ngec.actors.agent_matcher import AgentMatcher  # noqa: E402
    from importlib import resources as _res  # noqa: E402
    agents_file = str(_res.files("ngec.assets") / "PLOVER_agents.txt")
    out = run("categorize_entity (custom agents file)", steps.categorize_entity,
              ENTITY_SPANS[2][0], context=ENTITY_SPANS[2][1],
              agents_file=agents_file, mode=mode)
    if out:
        print(f"    {out['country']}/{out['code_1']} via {out['source']}")
    check("agents_file kwarg", "agents_file" in AgentMatcher.__init__.__code__.co_varnames)

    # --- step 5a
    # The bounded range is the demo page's fourth example: published on a
    # Friday, so both ends fall earlier in the same week.
    for phrase, pub_date in DATE_PHRASES + [("between Monday and Wednesday",
                                             "2024-06-14")]:
        out = run(f"resolve_date ({phrase!r})", steps.resolve_date, phrase,
                  pub_date, mode=mode)
        if out:
            print(f"    {out['resolved_date']} .. {out['date_end']} "
                  f"[{out['date_type']}/{out['granularity']}]")
            check("resolve_date", out["date_type"] is not None, "no date_type")

    # The page's "Published" box is free text, so the same date can be typed
    # several ways -- and nonsense has to come back as None, not an exception.
    # Not run through `run()`: this one returns a string, not a step dict.
    print("\n[parse_pub_date]")
    for text, expected in [("2024-06-11", "2024-06-11"),
                           ("June 11, 2024", "2024-06-11"),
                           ("last Friday", "a date"),
                           ("banana", None)]:
        got = steps.parse_pub_date(text)
        print(f"    {text!r} -> {got}")
        if expected == "a date":
            check("parse_pub_date", got is not None, f"{text!r} should parse")
        else:
            check("parse_pub_date", got == expected,
                  f"{text!r} gave {got}, expected {expected}")

    # --- step 5b
    doc = DOCUMENTS[0]
    out = run("geoparse", steps.geoparse, doc.text, mode=mode)
    entities = out["entities"] if out else []
    for ent in entities:
        print(f"    {ent['search_name']} -> {ent['resolved_placename']}, "
              f"{ent['country_code3']} ({ent['score']:.2f})")
    if es_up:
        check("geoparse", bool(entities), "no places found in the Paris story")

    out = run("pick_location", steps.pick_location, "Paris", entities, mode=mode)
    if out:
        print(f"    {(out['event_loc'] or {}).get('resolved_placename')} "
              f"({out['reason']})")
    out = run("pick_location (no span)", steps.pick_location, None, entities,
              mode=mode)
    check("pick_location", out and out["event_loc"] is None,
          "an empty span should resolve to no location")

    # --- the whole pipeline
    print(f"\n[run_pipeline] {doc.key}")
    start = time.time()
    result = steps.run_pipeline(doc.text, doc.pub_date, mode=mode)
    TIMINGS.append((f"run_pipeline ({mode})", result["timing"]["total"]))
    BREAKDOWNS[mode] = result["breakdown"]
    print(f"    {time.time() - start:.1f}s wall; timing: {result['timing']}")
    print(f"    story types: {result['story']['event_type']} "
          f"modes: {result['story']['event_mode']}")
    if result["dropped"]:
        print(f"    dropped: {result['dropped']}")
    for row in steps.events_table(result["events"]):
        print("    " + json.dumps(row))
    # The per-event view the main page draws, from the same records.
    for event in result["events"]:
        print("    " + json.dumps(steps.event_fields(event)))
    try:
        json.dumps(result)
    except TypeError as exc:
        check("run_pipeline", False, f"not JSON-serialisable: {exc}")
    if llm_up:
        check("run_pipeline", bool(result["events"]), "no events from the Paris story")
    check("run_pipeline", result["mode"] == mode, "wrong mode reported")
    check("run_pipeline",
          set(result["timing"]) >= {"spacy", "classify", "geoparse", "attributes",
                                    "actors", "format", "total"},
          f"missing timing keys: {sorted(result['timing'])}")
    stages = [row["path"] for row in result["breakdown"] if row["depth"] == 0]
    check("run_pipeline",
          stages == ["spacy", "classify", "geoparse", "attributes", "actors",
                     "format"],
          f"breakdown's top-level rows are {stages}")
    for row in result["breakdown"]:
        stage = row["path"].split("/")[0]
        check("run_pipeline",
              row["seconds"] <= result["timing"].get(stage, 0) + SLACK,
              f"{row['path']} ({row['seconds']:.2f}s) is longer than the "
              f"{stage} stage")
    print_breakdown(result["breakdown"])


def print_breakdown(rows: list[dict]) -> None:
    print("    where the time went:")
    for row in rows:
        indent = "  " * row["depth"]
        print(f"      {row['seconds']:7.3f}s {row['calls']:4d}x  {indent}{row['label']}")


# --- reporting ---------------------------------------------------------------

def print_side_by_side(modes: list[str]) -> None:
    """One row per timing path in run_pipeline, one column per mode."""
    modes = [m for m in modes if m in BREAKDOWNS]
    if not modes:
        return
    paths: list[str] = []
    for mode in modes:
        for row in BREAKDOWNS[mode]:
            if row["path"] not in paths:
                paths.append(row["path"])
    # A path only one mode measured (llama.cpp's prefill/decode) is appended
    # last, so re-sort by each path's ancestors to put it back under its parent.
    rank = {path: i for i, path in enumerate(paths)}
    paths.sort(key=lambda path: tuple(
        rank.get("/".join(path.split("/")[: i + 1]), 0)
        for i in range(path.count("/") + 1)))
    depth = {row["path"]: row["depth"]
             for mode in modes for row in BREAKDOWNS[mode]}
    seconds = {mode: {row["path"]: row["seconds"] for row in BREAKDOWNS[mode]}
               for mode in modes}

    width = max(len(p) + 2 * depth[p] for p in paths) + 2
    header = f"  {'component'.ljust(width)}" + "".join(f"{m:>10}" for m in modes)
    if "cpu" in modes and "gpu" in modes:
        header += f"{'cpu/gpu':>10}"
    print("\n--- run_pipeline, second by second ---")
    print(header)
    for path in paths:
        label = "  " * depth[path] + path.split("/")[-1]
        line = f"  {label.ljust(width)}"
        for mode in modes:
            value = seconds[mode].get(path)
            line += f"{value:>10.3f}" if value is not None else f"{'-':>10}"
        if "cpu" in modes and "gpu" in modes:
            gpu, cpu = seconds["gpu"].get(path), seconds["cpu"].get(path)
            ratio = f"{cpu / gpu:.1f}x" if gpu and cpu else "-"
            line += f"{ratio:>10}"
        print(line)


def print_load_report() -> None:
    rows = R.load_report()
    if not rows:
        return
    print("\n--- loading and warming up (seconds) ---")
    print(f"  {'component'.ljust(32)}{'mode':>8}{'load':>9}{'warm-up':>9}")
    for row in sorted(rows, key=lambda r: (r["mode"], r["component"])):
        print(f"  {row['component'].ljust(32)}{row['mode']:>8}"
              f"{row['load']:>9.2f}{row['warm_up']:>9.2f}")


# --- bulk coding -------------------------------------------------------------

def check_bulk(mode: str) -> None:
    """Three documents through the bulk page's parsing and pipeline.

    The parsing is checked in every mode; the run itself only on the GPU. The
    bulk page is disabled off the GPU, so coding three documents on the CPU
    would measure a path nobody can reach and cost minutes doing it.
    """
    docs = [{"id": doc.key, "text": doc.text, "pub_date": doc.pub_date}
            for doc in DOCUMENTS[:3]]

    # The file parsing is pure Python, so it is checked in every mode: the same
    # three documents written out as a CSV and as a JSONL have to come back the
    # same way the page hands them to the pipeline.
    import pandas as pd  # noqa: PLC0415 - only this section needs it

    frame = pd.DataFrame([{"id": doc["id"], "text": doc["text"],
                           "publication_date": doc["pub_date"]} for doc in docs])
    parsed, problems = steps.read_documents(frame.to_csv(index=False).encode(),
                                            "check.csv")
    check("read_documents", len(parsed) == len(docs),
          f"parsed {len(parsed)} of {len(docs)} CSV rows ({problems})")
    jsonl = "\n".join(json.dumps(row) for row in frame.to_dict("records"))
    from_jsonl, _ = steps.read_documents(jsonl.encode(), "check.jsonl")
    check("read_documents", from_jsonl == parsed,
          "the same documents parsed differently as CSV and as JSONL")

    if mode != "gpu":
        print("\n[run_pipeline_bulk] skipped: bulk coding is GPU-only")
        return
    print(f"\n[run_pipeline_bulk] {len(docs)} documents")
    seen: list[str] = []
    start = time.time()
    result = steps.run_pipeline_bulk(docs, mode=mode,
                                     progress=lambda stage, done, total: seen.append(stage))
    TIMINGS.append((f"run_pipeline_bulk ({mode})", result["seconds"]))
    print(f"    {time.time() - start:.1f}s wall; {len(result['events'])} events "
          f"from {result['n_docs']} documents "
          f"({result['per_doc_seconds']:.1f}s a document)")
    if result["dropped"]:
        print(f"    dropped: {result['dropped']}")

    check("run_pipeline_bulk", result["n_docs"] == len(docs),
          f"reported {result['n_docs']} documents, not {len(docs)}")
    check("run_pipeline_bulk", result["mode"] == mode, "wrong mode reported")
    check("run_pipeline_bulk", list(seen) == list(steps.BULK_STAGES),
          f"progress reported {seen}, not the six components")
    try:
        json.dumps(result)
    except TypeError as exc:
        check("run_pipeline_bulk", False, f"not JSON-serialisable: {exc}")

    # Every document should be traceable back through orig_id, and the records
    # have to be the same shape the single-document page already draws.
    ids = {doc["id"] for doc in docs}
    for event in result["events"]:
        check("run_pipeline_bulk",
              set(event) >= {"id", "orig_id", "event_type", "event_mode",
                             "attributes", "event_location", "date_resolved"},
              f"a record is missing top-level keys: {sorted(event)}")
        check("run_pipeline_bulk", event.get("orig_id") in ids,
              f"orig_id {event.get('orig_id')!r} is not one of the documents")
        check("run_pipeline_bulk", isinstance(event.get("attributes"), dict),
              "attributes is not a single dict")
    if R.get_attribute_model(mode) is not None:
        check("run_pipeline_bulk", bool(result["events"]),
              "no events from three documents")
    for row in steps.events_table(result["events"])[:5]:
        print("    " + json.dumps(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="all", choices=["gpu", "cpu", "all"],
                        help="which compute mode(s) to check (default: all "
                             "available)")
    args = parser.parse_args()

    available = R.available_modes()
    modes = list(available) if args.mode == "all" else [args.mode]
    missing = [m for m in modes if m not in available]
    if missing:
        print(f"{missing[0]} mode is not available on this machine "
              f"(available: {', '.join(available)})")
        return 1
    print(f"modes: {', '.join(modes)}")

    for mode in modes:
        run_mode(mode)
        check_bulk(mode)

    print("\n--- timings (seconds) ---")
    for name, seconds in TIMINGS:
        print(f"  {seconds:8.2f}  {name}")
    print_side_by_side(modes)
    print("\n--- what runs where ---")
    for mode in modes:
        print(f"  {mode}:")
        for name, where in R.component_devices(mode).items():
            print(f"    {name.ljust(28)} {where}")
    print_load_report()

    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
