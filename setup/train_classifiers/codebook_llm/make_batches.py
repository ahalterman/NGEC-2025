"""
Turn pooled documents into ready-to-annotate prompt files.

Annotation is done by LLM subagents rather than by API calls, so the unit of
work is a file on disk: this script writes a fully rendered prompt to
`<outdir>/<batch>.txt`, and an agent reads it, does the judging, and writes its
JSON answer to `<outdir>/<batch>.json`. Keeping the prompt on disk means the
exact text an annotation came from can be re-read later.

Batches are deliberately small (20 documents). A model asked to judge 100
articles in one response gets noticeably sloppier toward the end, and a bad
batch is cheaper to redo when it is small.

Usage:
    # screening: which of the 16 types might each document contain?
    python make_batches.py screen --n 3000 --out data/batches/screen

    # verification: is this one type really present? (needs candidates)
    python make_batches.py verify --candidates data/candidates.jsonl \\
        --out data/batches/verify
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from codebook import load_codebook, render_screening_prompt, render_verification_prompt

BATCH_SIZE = 20


def load_pool(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def write_batches(prompts, outdir, prefix):
    """Write rendered prompts as numbered .txt files and return a manifest."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (prompt, doc_ids, meta) in enumerate(prompts, start=1):
        name = f"{prefix}_{i:04d}"
        (outdir / f"{name}.txt").write_text(prompt, encoding="utf-8")
        manifest.append({"batch": name, "n_docs": len(doc_ids),
                         "doc_ids": doc_ids, **meta})
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def cmd_screen(args):
    """Build screening batches from a random sample of the pool."""
    book = load_codebook()
    pool = load_pool(args.pool)

    rng = random.Random(args.seed)
    sample = pool[:]
    rng.shuffle(sample)

    # Hold out a slice that is never keyword-seeded, so the final evaluation
    # sees documents at their natural base rates rather than at seeded ones.
    holdout = sample[:args.holdout]
    screen = sample[args.holdout:args.holdout + args.n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(args.out).parent / "holdout_ids.json", "w") as f:
        json.dump([d["id"] for d in holdout], f)

    prompts = []
    for i in range(0, len(screen), args.batch_size):
        chunk = screen[i:i + args.batch_size]
        prompts.append((
            render_screening_prompt(book, chunk),
            [d["id"] for d in chunk],
            {"kind": "screen", "arm": "random"},
        ))

    manifest = write_batches(prompts, args.out, "screen")
    print(f"{len(screen)} documents -> {len(manifest)} screening batches in {args.out}")
    print(f"{len(holdout)} documents reserved as never-seeded holdout")


def cmd_verify(args):
    """Build verification batches, one event type per batch."""
    book = load_codebook()
    pool = {rec["id"]: rec for rec in load_pool(args.pool)}
    contrasts = json.load(open(Path(args.pool).parent / "codebook.json"))

    # candidates.jsonl: {"id": ..., "event_type": ..., "source": "screen"|"seed"|"active"}
    by_type = defaultdict(list)
    for line in open(args.candidates, encoding="utf-8"):
        cand = json.loads(line)
        if cand["id"] in pool:
            by_type[cand["event_type"]].append(cand)

    prompts = []
    for event_type in sorted(by_type):
        cands = by_type[event_type]
        for i in range(0, len(cands), args.batch_size):
            chunk = cands[i:i + args.batch_size]
            docs = [pool[c["id"]] for c in chunk]
            prompts.append((
                render_verification_prompt(
                    book, event_type, docs,
                    contrasts.get(event_type, {}).get("contrasts", []),
                ),
                [d["id"] for d in docs],
                {"kind": "verify", "event_type": event_type,
                 "sources": [c.get("source", "") for c in chunk]},
            ))

    manifest = write_batches(prompts, args.out, "verify")
    counts = defaultdict(int)
    for entry in manifest:
        counts[entry["event_type"]] += entry["n_docs"]
    print(f"{len(manifest)} verification batches in {args.out}")
    for event_type in sorted(counts):
        print(f"  {event_type:10s} {counts[event_type]:5d} candidates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/voa_pool.jsonl")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=20260804)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--holdout", type=int, default=1000,
                   help="documents reserved from seeding for unbiased evaluation")
    s.add_argument("--out", default="data/batches/screen")
    s.set_defaults(func=cmd_screen)

    v = sub.add_parser("verify")
    v.add_argument("--candidates", default="data/candidates.jsonl")
    v.add_argument("--out", default="data/batches/verify")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
