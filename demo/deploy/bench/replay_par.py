"""Replay the extraction prompts with N requests in flight at once.

Concurrency looks like free throughput and is not: `llama-server` auto-selects
several slots, and sequential requests all land on the slot already holding the
shared document prefix while concurrent ones are spread across slots holding
different prefixes. Watch `pp_tok` in the output -- it grows with concurrency,
because the document is re-evaluated once per slot. That extra prefill costs
more than the batching saves.

Usage: replay_par.py PORT LABEL CONCURRENCY [REPS]
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

PORT, LABEL, CONC = sys.argv[1], sys.argv[2], int(sys.argv[3])
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
URL = f"http://127.0.0.1:{PORT}/completion"
prompts = json.load(open(os.path.join(HERE, "prompts.json")))
SCHEMA = json.load(open(os.path.join(HERE, "schema.json")))

def call(p):
    body = {"prompt": p, "n_predict": 1024, "temperature": 0.5, "top_p": 0.8,
            "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "cache_prompt": True, "seed": 1234, "json_schema": SCHEMA}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        return json.loads(resp.read())

call(prompts[0]["prompt"])   # warm

for rep in range(REPS):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        res = list(ex.map(lambda p: call(p["prompt"]), prompts))
    total = time.time() - t0
    gen = sum(r.get("timings", {}).get("predicted_n", 0) for r in res)
    pp = sum(r.get("timings", {}).get("prompt_n", 0) for r in res)
    print(f"{LABEL} conc={CONC} rep{rep}: total={total:6.2f}s  "
          f"gen_tok={gen:4d}  pp_tok={pp:5d}  ({len(prompts)/total:.2f} req/s)")
