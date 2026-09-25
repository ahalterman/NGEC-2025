"""
Step 7: score the trained models on the never-seeded holdout.

Why a separate holdout rather than the split inside `train_classifiers.py`: the
training documents are enriched. Keyword seeding and active learning deliberately
go looking for positives, so the proportion of PROTEST documents among the
annotated training set is far higher than the proportion among news articles. A
score computed on that set answers "how well does the model do on documents
selected for being interesting", which is not the question anyone has.

The holdout is drawn at random from the pool before any seeding happens and is
excluded from seeding by `seed_candidates.py`. Precision on it is therefore a
real precision: the negatives are as common, and as boring, as they are in the
corpus.

Numbers this prints that are worth reading carefully:

- **base_rate** -- how often the type actually occurs. For a type at 2%, a model
  can score 98% accuracy by never predicting it, so accuracy is not reported.
- **precision at the shipped threshold** -- what a user of the pipeline gets.
- **n_positive** -- with few holdout positives the interval around F1 is wide,
  so a type with a handful of positives is reported but should not be leaned on.

Usage:
    python evaluate.py --models ../../../ngec/assets/event_models_v2
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import skops.io as sio
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from codebook import load_codebook


def load_models(model_dir, book):
    """Load type models, mode models, the vectorizer, and metadata."""
    model_dir = Path(model_dir)
    meta_path = model_dir / "metadata.json"
    metadata = json.load(open(meta_path)) if meta_path.exists() else {}

    def _load(path):
        return sio.load(path, trusted=sio.get_untrusted_types(file=path))

    types = {}
    for event_type in book:
        path = model_dir / f"{event_type}.skops"
        if path.exists():
            types[event_type] = _load(path)

    modes = defaultdict(dict)
    for event_type in book:
        mode_dir = model_dir / "modes" / event_type
        if not mode_dir.exists():
            continue
        for mode in book[event_type]["modes"]:
            path = mode_dir / f"{mode}.skops"
            if path.exists():
                modes[event_type][mode] = _load(path)

    vec_path = model_dir / "tfidf_vectorizer.skops"
    vectorizer = _load(vec_path) if vec_path.exists() else None
    return types, dict(modes), vectorizer, metadata


def score(y_true, proba, threshold):
    """Precision/recall/F1 at the shipped threshold, plus threshold-free AP."""
    pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    # Average precision summarizes the whole ranking, so a type whose threshold
    # happens to be badly placed can be told apart from one the features simply
    # cannot separate.
    ap = average_precision_score(y_true, proba) if 0 < y_true.sum() < len(y_true) else float("nan")
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "average_precision": float(ap), "threshold": float(threshold),
            "n": int(len(y_true)), "n_positive": int(y_true.sum()),
            "base_rate": float(y_true.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/voa_pool.jsonl")
    ap.add_argument("--emb", default="data/pool_emb.npy")
    ap.add_argument("--annotations", default="data/annotations.jsonl")
    ap.add_argument("--holdout-ids", default="data/batches/holdout_ids.json")
    ap.add_argument("--models", default="../../../ngec/assets/event_models_v2")
    ap.add_argument("--out", default="data/holdout_metrics.json")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ngec.classifiers.features import combine_features

    book = load_codebook()
    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    index = {rec["id"]: i for i, rec in enumerate(pool)}

    holdout = set(json.load(open(args.holdout_ids)))
    labels = {}
    for line in open(args.annotations, encoding="utf-8"):
        row = json.loads(line)
        if row["id"] in holdout and row["label"] != "NA":
            labels[(row["id"], row["event_type"], row["mode"])] = int(row["label"])

    if not labels:
        raise SystemExit(
            "No holdout annotations found. The holdout has to be screened and "
            "verified like the training data before it can be scored against.")

    types, modes, vectorizer, metadata = load_models(args.models, book)

    embeddings = np.load(args.emb)
    if vectorizer is not None:
        texts = [rec["event_text"] for rec in pool]
        X = combine_features(embeddings, vectorizer.transform(texts))
    else:
        X = embeddings

    thresholds = {k: v["threshold"] for k, v in metadata.get("metrics", {}).items()}
    mode_thresholds = {k: v["threshold"]
                       for k, v in metadata.get("mode_metrics", {}).items()}

    results = {"types": {}, "modes": {}}

    print(f"Holdout: {len(holdout)} documents\n")
    print(f"{'type':12s} {'n':>5s} {'pos':>5s} {'base':>6s} "
          f"{'prec':>6s} {'rec':>6s} {'f1':>6s} {'AP':>6s}")
    for event_type in sorted(types):
        cells = [(index[doc_id], lab)
                 for (doc_id, et, mode), lab in labels.items()
                 if et == event_type and mode is None and doc_id in index]
        if not cells:
            continue
        rows = [i for i, _ in cells]
        y = np.array([lab for _, lab in cells])
        if y.sum() == 0:
            continue
        proba = types[event_type].predict_proba(X[rows])[:, 1]
        res = score(y, proba, thresholds.get(event_type, 0.5))
        results["types"][event_type] = res
        print(f"{event_type:12s} {res['n']:5d} {res['n_positive']:5d} "
              f"{res['base_rate']:6.3f} {res['precision']:6.3f} "
              f"{res['recall']:6.3f} {res['f1']:6.3f} {res['average_precision']:6.3f}")

    if modes:
        print(f"\n{'mode':26s} {'n':>5s} {'pos':>5s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
        for event_type in sorted(modes):
            for mode in sorted(modes[event_type]):
                cells = [(index[doc_id], lab)
                         for (doc_id, et, m), lab in labels.items()
                         if et == event_type and m == mode and doc_id in index]
                if not cells:
                    continue
                rows = [i for i, _ in cells]
                y = np.array([lab for _, lab in cells])
                if y.sum() == 0:
                    continue
                proba = modes[event_type][mode].predict_proba(X[rows])[:, 1]
                name = f"{event_type}-{mode}"
                res = score(y, proba, mode_thresholds.get(name, 0.5))
                results["modes"][name] = res
                print(f"{name:26s} {res['n']:5d} {res['n_positive']:5d} "
                      f"{res['precision']:6.3f} {res['recall']:6.3f} {res['f1']:6.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results["types"]:
        macro = np.mean([r["f1"] for r in results["types"].values()])
        print(f"\nMacro F1 over {len(results['types'])} event types: {macro:.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
