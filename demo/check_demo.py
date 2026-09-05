#!/usr/bin/env python
"""Run every function in `ngec_demo.steps` on real inputs, with no Streamlit.

This is the demo's test: the pages contain no pipeline logic, so if this passes
the pages work. It loads the real models and talks to the real Elasticsearch, so
it takes a couple of minutes on CPU.

    cd demo && NGEC_DEMO_BACKEND=transformers uv run --group demo-app python check_demo.py

Exits non-zero on the first failure, and prints a timing per step.
"""

from __future__ import annotations

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


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        FAILURES.append(f"{name}: {detail or 'failed'}")
        print(f"    FAIL {detail}")


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
    try:
        json.dumps(out)
    except TypeError as exc:
        check(name, False, f"not JSON-serialisable: {exc}")
    return out


def main() -> int:
    print("backend:", R.backend(), "| gpu:", R.use_gpu())
    for name, row in R.health().items():
        print(f"  {'ok ' if row['ok'] else 'DOWN'} {name}: {row['detail']}")
    es_up = R.get_es() is not None

    # --- step 1
    label, text = SHORT_TEXTS[0]
    out = run("classify", steps.classify, text)
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
    out = run("extract_attributes", steps.extract_attributes, text, event_type)
    if out:
        check("extract_attributes", bool(out["prompt"]), "empty prompt")
        for record in out["records"]:
            print("    " + json.dumps({k: v for k, v in record.items()
                                       if k != "anchor_quote"}))
        check("extract_attributes", len(out["records"]) >= 1,
              f"no events extracted for {event_type}")

    # An event type outside the codebook, extracted from a written definition.
    out = run("extract_attributes (custom definition)", steps.extract_attributes,
              SHORT_TEXTS[2][1], "DETAIN",
              event_def="One party takes another into custody or arrests them.")
    if out:
        print(f"    {len(out['records'])} record(s)")

    # --- step 3
    span, context = ENTITY_SPANS[0]
    out = run("resolve_entity", steps.resolve_entity, span, context=context)
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

    # --- step 4
    out = run("categorize_entity", steps.categorize_entity, span, context=context,
              query_date="2023-03-15")
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
              ENTITY_SPANS[2][0], context=ENTITY_SPANS[2][1], agents_file=agents_file)
    if out:
        print(f"    {out['country']}/{out['code_1']} via {out['source']}")
    check("agents_file kwarg", "agents_file" in AgentMatcher.__init__.__code__.co_varnames)

    # --- step 5a
    for phrase, pub_date in DATE_PHRASES:
        out = run(f"resolve_date ({phrase!r})", steps.resolve_date, phrase, pub_date)
        if out:
            print(f"    {out['resolved_date']} .. {out['date_end']} "
                  f"[{out['date_type']}/{out['granularity']}]")
            check("resolve_date", out["date_type"] is not None, "no date_type")

    # --- step 5b
    doc = DOCUMENTS[0]
    out = run("geoparse", steps.geoparse, doc.text)
    entities = out["entities"] if out else []
    for ent in entities:
        print(f"    {ent['search_name']} -> {ent['resolved_placename']}, "
              f"{ent['country_code3']} ({ent['score']:.2f})")
    if es_up:
        check("geoparse", bool(entities), "no places found in the Paris story")

    out = run("pick_location", steps.pick_location, "Paris", entities)
    if out:
        print(f"    {(out['event_loc'] or {}).get('resolved_placename')} "
              f"({out['reason']})")
    out = run("pick_location (no span)", steps.pick_location, None, entities)
    check("pick_location", out and out["event_loc"] is None,
          "an empty span should resolve to no location")

    # --- the whole pipeline
    print(f"\n[run_pipeline] {doc.key}")
    start = time.time()
    result = steps.run_pipeline(doc.text, doc.pub_date)
    TIMINGS.append(("run_pipeline", result["timing"]["total"]))
    print(f"    {time.time() - start:.1f}s wall; timing: {result['timing']}")
    print(f"    story types: {result['story']['event_type']} "
          f"modes: {result['story']['event_mode']}")
    if result["dropped"]:
        print(f"    dropped: {result['dropped']}")
    for row in steps.events_table(result["events"]):
        print("    " + json.dumps(row))
    try:
        json.dumps(result)
    except TypeError as exc:
        check("run_pipeline", False, f"not JSON-serialisable: {exc}")
    check("run_pipeline", bool(result["events"]), "no events from the Paris story")
    check("run_pipeline",
          set(result["timing"]) >= {"spacy", "classify", "geoparse", "attributes",
                                    "actors", "format", "total"},
          f"missing timing keys: {sorted(result['timing'])}")

    # --- summary
    print("\n--- timings (seconds) ---")
    for name, seconds in TIMINGS:
        print(f"  {seconds:8.2f}  {name}")

    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)}):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
