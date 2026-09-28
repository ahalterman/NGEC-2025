"""
Embed the annotation pool once, and cache it.

The embeddings are used twice: to retrieve candidate documents for rare modes
(step 2, seeding) and as part of the features the final classifiers are trained
on (step 6). Computing them once and caching keeps those two steps consistent.

Documents are chunked and mean-pooled rather than truncated -- see
`ngec/classifiers/features.py` for why, and note that the same module does the
chunking at serving time, so training and inference cannot drift apart.

This script only needs torch and sentence-transformers, so it can be run from a
separate CUDA-matched environment when the project venv's torch build does not
match the installed driver:

    <gpu-venv>/bin/python embed_pool.py --device cuda
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np


def _load_features_module():
    """
    Load `ngec/classifiers/features.py` directly from its file.

    Importing it as `ngec.classifiers.features` would run `ngec/__init__.py`,
    which pulls in the whole pipeline (dateparser, spacy, elasticsearch...). This
    script is meant to be runnable from a bare CUDA-matched environment that has
    only torch and sentence-transformers, so it takes the one module it needs.
    """
    path = Path(__file__).resolve().parents[3] / "ngec" / "classifiers" / "features.py"
    spec = importlib.util.spec_from_file_location("ngec_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chunk_text = _load_features_module().chunk_text

ENCODER = "all-mpnet-base-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/voa_pool.jsonl")
    ap.add_argument("--out", default="data/pool_emb.npy")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda, cpu, or leave unset")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    texts = [rec["event_text"] for rec in pool]

    # Chunk up front so the whole corpus can be encoded as one flat batch list.
    all_chunks, owners = [], []
    for i, text in enumerate(texts):
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        owners.extend([i] * len(chunks))
    owners = np.asarray(owners)
    print(f"{len(texts)} documents -> {len(all_chunks)} chunks "
          f"({len(all_chunks) / len(texts):.2f} per document)")

    model = SentenceTransformer(f"sentence-transformers/{args.encoder}",
                                device=args.device)
    print(f"encoding on {model.device} with {args.encoder}")

    start = time.time()
    chunk_emb = model.encode(all_chunks, batch_size=args.batch_size,
                             show_progress_bar=True)
    chunk_emb = np.asarray(chunk_emb, dtype=np.float32)
    print(f"  encoded in {time.time() - start:.0f}s")

    # Mean-pool each document's chunks, then L2-normalize so downstream cosine
    # similarity is a plain dot product.
    dim = chunk_emb.shape[1]
    summed = np.zeros((len(texts), dim), dtype=np.float32)
    counts = np.zeros(len(texts), dtype=np.float32)
    np.add.at(summed, owners, chunk_emb)
    np.add.at(counts, owners, 1.0)
    embeddings = summed / counts[:, None]
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    out = Path(args.out)
    np.save(out, embeddings)
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump({"encoder": args.encoder, "n": len(texts),
                   "dim": int(dim), "n_chunks": len(all_chunks),
                   "chunked": True,
                   "ids": [rec["id"] for rec in pool]}, f)
    print(f"Wrote {out} {embeddings.shape}")


if __name__ == "__main__":
    main()
