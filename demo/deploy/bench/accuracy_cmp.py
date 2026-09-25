"""Do two llama.cpp builds produce the same extractions from the same weights?

Runs greedy (temperature 0) so sampling RNG is removed and any difference is
kernel numerics. Compares three ways, loosest to strictest:

  fields   -- the parsed actor/recipient/date/location/event_type values, which
              are the only thing the rest of the pipeline consumes
  n_events -- how many events each response claims
  exact    -- byte-identical response text

Field agreement is the metric that matters; exact match is reported because
DESIGN.md's existing quantization table uses it.
"""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PORTS = {"upstream": 8081, "ik": 8082}
prompts = json.load(open(os.path.join(HERE, "prompts.json")))
SCHEMA = json.load(open(os.path.join(HERE, "schema.json")))

def call(port, prompt):
    body = {"prompt": prompt, "n_predict": 1024, "temperature": 0.0,
            "top_k": 1, "cache_prompt": True, "json_schema": SCHEMA}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        return json.loads(resp.read()).get("content", "").strip()

def fields(text):
    """Normalised (event_type, actor, recipient, date, location) tuples."""
    try:
        evs = json.loads(text)
    except Exception:
        return None
    if not isinstance(evs, list):
        return None
    out = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        out.append(tuple(str(e.get(k, "")).strip().lower()
                         for k in ("event_type", "actor", "recipient",
                                   "date", "location")))
    return sorted(out)

rows = []
for p in prompts:
    a = call(PORTS["upstream"], p["prompt"])
    b = call(PORTS["ik"], p["prompt"])
    rows.append((p["doc"], p["event_type"], a, b))

n = len(rows)
exact = sum(1 for _, _, a, b in rows if a == b)
valid_a = sum(1 for _, _, a, _ in rows if fields(a) is not None)
valid_b = sum(1 for _, _, _, b in rows if fields(b) is not None)
same_fields = sum(1 for _, _, a, b in rows if fields(a) == fields(b))
same_count = sum(1 for _, _, a, b in rows
                 if (fields(a) or []) and (fields(b) or [])
                 and len(fields(a)) == len(fields(b)))

print(f"prompts compared      : {n}")
print(f"valid JSON  upstream  : {valid_a}/{n}")
print(f"valid JSON  ik        : {valid_b}/{n}")
print(f"identical field sets  : {same_fields}/{n}")
print(f"byte-identical text   : {exact}/{n}")
print()
for doc, et, a, b in rows:
    if fields(a) != fields(b):
        print(f"--- DIFFERS  doc{doc} {et}")
        print(f"    upstream: {a[:300]}")
        print(f"    ik      : {b[:300]}")
