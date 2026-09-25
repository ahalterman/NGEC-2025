"""
Step 2: find documents worth annotating for each event type and mode.

A random sample of news gets you plenty of ASSAULT and CONSULT and almost no
PROTEST-hunger or ASSAULT-cleansing. Training a classifier for the rare modes
means going and finding their positives on purpose. Two complementary ways of
looking, because each misses what the other catches:

- **Keyword seeding** matches hand-written regexes (written from the mode
  definitions, see `data/seeds_*.json`). High precision, but only finds phrasings
  somebody thought of in advance.
- **Embedding retrieval** ranks documents by cosine similarity to the mode's
  definition text. Catches paraphrases the regexes miss, but drifts off-topic
  further down the ranking.

Neither is treated as a label. Everything they surface is a *candidate* that the
verification pass then judges, and both stages are recorded in the output so a
seeded positive can be told apart from a randomly-sampled one later. That
distinction matters at evaluation time: seeded documents are enriched for
positives and would inflate any metric computed over them.

Usage:
    python seed_candidates.py --out data/seed_candidates.jsonl --per-mode 120
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from codebook import load_codebook


def load_seeds(data_dir):
    """Merge the per-group seed files into {event_type: {mode_or__type: [regex]}}."""
    seeds = defaultdict(dict)
    for path in sorted(Path(data_dir).glob("seeds_*.json")):
        for event_type, modes in json.load(open(path, encoding="utf-8")).items():
            seeds[event_type].update(modes)
    return dict(seeds)


def compile_seeds(seeds):
    """Compile the regexes once, reporting any that do not compile."""
    compiled = defaultdict(dict)
    bad = []
    for event_type, modes in seeds.items():
        for mode, patterns in modes.items():
            good = []
            for pattern in patterns:
                try:
                    good.append(re.compile(pattern, re.IGNORECASE))
                except re.error as exc:
                    bad.append((event_type, mode, pattern, str(exc)))
            compiled[event_type][mode] = good
    if bad:
        print(f"  WARNING: {len(bad)} seed patterns did not compile:")
        for event_type, mode, pattern, err in bad[:10]:
            print(f"    {event_type}-{mode}: {pattern!r} ({err})")
    return compiled


def keyword_hits(pool, compiled, min_hits=1):
    """
    Documents matching a mode's seeds, ranked by how many distinct seeds matched.

    Every document matching at least `min_hits` patterns is kept, sorted so that
    documents matching several patterns come first. Ranking rather than
    hard-filtering is deliberate: matching two patterns is good evidence (a
    document matching "strike", "walkout" and "union" is far more likely to be a
    labor strike than one matching "strike" alone, which is usually an air
    strike), but for some modes the evidence is inherently a single phrase.

    Requiring two matches dropped every THREATEN, MOBILIZE and REJECT mode to
    zero candidates, because "threatened to arrest" is one construction, not
    three. Since candidates are only candidates -- verification judges them
    afterward -- the cost of a loose seed is one wasted judgment, while the cost
    of a strict one is a mode with no training data at all.
    """
    hits = defaultdict(list)
    for i, rec in enumerate(pool):
        text = rec["event_text"]
        for event_type, modes in compiled.items():
            for mode, patterns in modes.items():
                n = sum(1 for p in patterns if p.search(text))
                if n >= min_hits:
                    hits[(event_type, mode)].append((n, i))
    for key in hits:
        hits[key].sort(reverse=True)
    return hits


def embedding_hits(book, embeddings, encoder_name, top_k):
    """Documents most similar to each mode's definition text."""
    from sentence_transformers import SentenceTransformer

    queries, keys = [], []
    for event_type, entry in book.items():
        # The type query blends the definition with its exemplars/mode defs so it
        # describes the whole category rather than one abstract sentence.
        extra = " ".join(list(entry["exemplars"]) + list(entry["modes"].values()))
        queries.append(f"{entry['definition']} {extra}".strip())
        keys.append((event_type, "_type"))
        for mode, mode_def in entry["modes"].items():
            queries.append(f"{entry['definition']} Specifically: {mode_def}")
            keys.append((event_type, mode))

    model = SentenceTransformer(f"sentence-transformers/{encoder_name}")
    q = model.encode(queries)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    # embeddings are already L2-normalized, so this is cosine similarity.
    sims = q @ embeddings.T
    hits = {}
    for row, key in enumerate(keys):
        order = np.argsort(sims[row])[::-1][:top_k]
        hits[key] = [(float(sims[row][i]), int(i)) for i in order]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/voa_pool.jsonl")
    ap.add_argument("--emb", default="data/pool_emb.npy")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/seed_candidates.jsonl")
    ap.add_argument("--per-mode", type=int, default=120,
                    help="candidates to take per (type, mode) from each method")
    ap.add_argument("--per-type", type=int, default=200,
                    help="candidates to take per event type from each method")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="distinct seed patterns a document must match")
    args = ap.parse_args()

    book = load_codebook()
    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    print(f"{len(pool)} pooled documents")

    # Never seed the evaluation holdout: it has to keep its natural base rates.
    holdout_path = Path(args.pool).parent / "batches" / "holdout_ids.json"
    holdout = set(json.load(open(holdout_path))) if holdout_path.exists() else set()
    print(f"{len(holdout)} holdout ids excluded from seeding")

    seeds = load_seeds(args.data_dir)
    compiled = compile_seeds(seeds)
    n_patterns = sum(len(p) for m in compiled.values() for p in m.values())
    print(f"{n_patterns} seed patterns over {len(compiled)} event types")

    print("Matching keywords ...")
    kw = keyword_hits(pool, compiled, min_hits=args.min_hits)

    emb_path = Path(args.emb)
    if emb_path.exists():
        print("Retrieving by embedding similarity ...")
        embeddings = np.load(emb_path)
        meta = json.load(open(emb_path.with_suffix(".meta.json")))
        emb = embedding_hits(book, embeddings, meta["encoder"], args.per_mode * 3)
    else:
        print(f"  {emb_path} not found, using keyword seeding only")
        emb = {}

    # Union the two methods, remembering which one found each candidate.
    candidates = {}
    for key in set(kw) | set(emb):
        event_type, mode = key
        limit = args.per_type if mode == "_type" else args.per_mode
        for source, ranked in (("keyword", kw.get(key, [])), ("embedding", emb.get(key, []))):
            taken = 0
            for _, idx in ranked:
                if taken >= limit:
                    break
                doc_id = pool[idx]["id"]
                if doc_id in holdout:
                    continue
                taken += 1
                entry = candidates.setdefault(
                    (doc_id, event_type),
                    {"id": doc_id, "event_type": event_type, "source": "seed",
                     "seed_modes": [], "methods": []},
                )
                if mode != "_type" and mode not in entry["seed_modes"]:
                    entry["seed_modes"].append(mode)
                if source not in entry["methods"]:
                    entry["methods"].append(source)

    with open(args.out, "w", encoding="utf-8") as f:
        for entry in candidates.values():
            f.write(json.dumps(entry) + "\n")

    per_type = defaultdict(int)
    for (_, event_type) in candidates:
        per_type[event_type] += 1
    print(f"\n{len(candidates)} (document, type) candidates -> {args.out}")
    for event_type in sorted(per_type):
        print(f"  {event_type:10s} {per_type[event_type]:5d}")

    print("\nDocuments found per (type, mode) by keyword seeding:")
    for (event_type, mode), found in sorted(kw.items()):
        if mode != "_type":
            print(f"  {event_type}-{mode:16s} {len(found):6d}")


if __name__ == "__main__":
    main()
