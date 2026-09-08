"""Replay real extraction prompts against a llama-server and time them.

Mirrors ngec/llm/llamacpp.py: same body, same cache_prompt, sequential, one
request per record -- so wall time here is the attribute-extraction time the
demo actually pays.

Usage: replay.py PORT LABEL [REPS]
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PORT = sys.argv[1]
LABEL = sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
URL = f"http://127.0.0.1:{PORT}/completion"

prompts = json.load(open(os.path.join(HERE, "prompts.json")))

# Production sends a json_schema (ngec/attributes/schema.py ATTRIBUTE_SCHEMA)
# because LlamaCppServerEngine declares capabilities.schema=True. NGEC_NO_SCHEMA=1
# drops it, to price the grammar constraint itself.
SCHEMA = None if os.environ.get("NGEC_NO_SCHEMA") else json.load(
    open(os.path.join(HERE, "schema.json")))

def call(prompt):
    body = {"prompt": prompt, "n_predict": 1024, "temperature": 0.5,
            "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "cache_prompt": True, "seed": 1234}
    if SCHEMA is not None:
        body["json_schema"] = SCHEMA
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

# Warm: fill the prompt cache the way a real session does.
call(prompts[0]["prompt"])

rows = []
for rep in range(REPS):
    t0 = time.time()
    gen_tok = pp_tok = cached = 0
    pp_ms = gen_ms = 0.0
    per_doc = {}
    for p in prompts:
        t1 = time.time()
        r = call(p["prompt"])
        dt = time.time() - t1
        tm = r.get("timings", {})
        gen_tok += tm.get("predicted_n", 0)
        pp_tok += tm.get("prompt_n", 0)
        pp_ms += tm.get("prompt_ms", 0.0)
        gen_ms += tm.get("predicted_ms", 0.0)
        cached += r.get("tokens_cached", 0)
        per_doc.setdefault(p["doc"], 0.0)
        per_doc[p["doc"]] += dt
    total = time.time() - t0
    rows.append((total, gen_tok, pp_tok))
    other = total - (pp_ms + gen_ms) / 1000.0
    print(f"{LABEL} rep{rep}: total={total:6.2f}s | pp={pp_ms/1000:5.2f}s "
          f"({pp_tok:5d} tok) | gen={gen_ms/1000:6.2f}s ({gen_tok:4d} tok, "
          f"{gen_tok/(gen_ms/1000):5.2f} t/s) | other={other:5.2f}s")

best = min(r[0] for r in rows)
mean = sum(r[0] for r in rows) / len(rows)
print(f"RESULT\t{LABEL}\t{mean:.2f}\t{best:.2f}\t{rows[0][1]}")
