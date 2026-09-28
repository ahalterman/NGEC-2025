"""
Decide which (document, type) pairs are worth a careful verification judgment.

Verification is the expensive stage -- one focused judgment per pair -- so this
script chooses what to spend it on. Three sources feed it, and they answer
different questions:

- **Screening candidates.** The screen flagged these as possible. Verifying them
  turns a recall-oriented guess into a real positive or a real negative, and is
  where most of the training signal comes from.
- **Seeded candidates.** Keyword and embedding retrieval went looking for rare
  modes on purpose. Without these, a mode like PROTEST-hunger has no positives
  at all, because a random sample of news does not contain one.
- **Audited implied zeros.** Documents the screen did *not* flag. These are used
  as negatives in bulk, so a sample is verified to measure how often the screen
  missed a real positive. Without the audit that error rate is unknown and the
  training negatives are quietly wrong.

Output is ordered by priority, so that stopping the verification stage early
still leaves the most valuable judgments done. Rare modes come first: a type-level
model has thousands of examples either way, while a mode model has whatever
seeding found and nothing else.

Usage:
    python build_verify_candidates.py --out data/candidates.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from codebook import load_codebook
from collect_annotations import collect_screening


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/voa_pool.jsonl")
    ap.add_argument("--screen-dir", default="data/batches/screen")
    ap.add_argument("--seeds", default="data/seed_candidates.jsonl")
    ap.add_argument("--out", default="data/candidates.jsonl")
    ap.add_argument("--per-type-screen", type=int, default=250,
                    help="screening candidates to verify per event type")
    ap.add_argument("--per-type-seed", type=int, default=150,
                    help="seeded candidates to verify per event type")
    ap.add_argument("--per-type-audit", type=int, default=50,
                    help="implied zeros to audit per event type")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    book = load_codebook()
    pool_text = {}
    for line in open(args.pool, encoding="utf-8"):
        rec = json.loads(line)
        pool_text[rec["id"]] = rec["event_text"]

    screen_rows, _ = collect_screening(args.screen_dir, list(book), pool_text)

    candidates = []   # screening flagged it
    implied = defaultdict(list)  # screening did not
    for row in screen_rows:
        if row["source"] == "screen_candidate":
            candidates.append((row["id"], row["event_type"]))
        else:
            implied[row["event_type"]].append(row["id"])

    seeded = defaultdict(list)
    seed_modes = {}
    if Path(args.seeds).exists():
        for line in open(args.seeds, encoding="utf-8"):
            entry = json.loads(line)
            seeded[entry["event_type"]].append(entry["id"])
            seed_modes[(entry["id"], entry["event_type"])] = entry.get("seed_modes", [])

    # How many documents each mode has been seeded for. Modes with the fewest
    # get their candidates verified first.
    mode_supply = Counter()
    for modes in seed_modes.values():
        for mode in modes:
            mode_supply[mode] += 1

    rows = []
    seen = set()

    def add(doc_id, event_type, source, priority, modes=None):
        key = (doc_id, event_type)
        if key in seen:
            return
        seen.add(key)
        rows.append({"id": doc_id, "event_type": event_type, "source": source,
                     "priority": priority, "seed_modes": modes or []})

    # Priority 0: seeded candidates, rarest mode first. These are the only source
    # of positives for modes that a random sample never contains.
    for event_type, doc_ids in seeded.items():
        ranked = sorted(
            doc_ids,
            key=lambda d: min((mode_supply[m] for m in seed_modes.get((d, event_type), [])),
                              default=10 ** 6),
        )
        for doc_id in ranked[:args.per_type_seed]:
            add(doc_id, event_type, "seed", 0, seed_modes.get((doc_id, event_type)))

    # Priority 1: screening candidates, capped per type so that a common type
    # like ACCUSE does not consume the whole budget.
    by_type = defaultdict(list)
    for doc_id, event_type in candidates:
        by_type[event_type].append(doc_id)
    for event_type, doc_ids in by_type.items():
        rng.shuffle(doc_ids)
        for doc_id in doc_ids[:args.per_type_screen]:
            add(doc_id, event_type, "screen", 1)

    # Priority 2: the audit sample of implied zeros.
    for event_type, doc_ids in implied.items():
        rng.shuffle(doc_ids)
        taken = 0
        for doc_id in doc_ids:
            if taken >= args.per_type_audit:
                break
            if (doc_id, event_type) in seen:
                continue
            add(doc_id, event_type, "audit", 2)
            taken += 1

    rows.sort(key=lambda r: r["priority"])
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    counts = Counter((r["source"], r["event_type"]) for r in rows)
    per_source = Counter(r["source"] for r in rows)
    print(f"{len(rows)} (document, type) pairs to verify -> {args.out}")
    for source, n in sorted(per_source.items()):
        print(f"  {source:8s} {n:6d}")
    print("\nper event type:")
    for event_type in sorted(book):
        line = "  ".join(f"{s}={counts[(s, event_type)]}"
                         for s in ("seed", "screen", "audit"))
        print(f"  {event_type:10s} {line}")


if __name__ == "__main__":
    main()
