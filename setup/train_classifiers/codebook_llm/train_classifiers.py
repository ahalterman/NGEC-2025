"""
Step 6: fit the event type and mode classifiers from the sparse annotations.

Two families of models come out of this:

- **Type models**, one per event type, over all annotated documents.
- **Mode models**, one per (type, mode) pair, fit only on documents where that
  type is present. A mode is a sub-kind of its parent event, so asking "is this
  PROTEST a hunger strike?" only makes sense once the document is a PROTEST at
  all. This mirrors how `PloverSklearnClassifier` applies them at serving time:
  mode models are consulted only for types that fired.

Features are chunked mean-pooled sentence embeddings concatenated with TF-IDF
word features, and the models are L1-penalized logistic regressions. The lasso
penalty is doing real work here: it drives most of the vocabulary to zero and
leaves a short, readable list of terms per class, which `--show-terms` prints so
the selected words can be checked against what a human coder would use.

Cells labeled "NA" and cells that were never annotated are **dropped**, not
treated as negatives. That distinction is the whole point of a sparse annotation
scheme; collapsing it would silently invent labels.

Usage:
    python train_classifiers.py --out ../../../ngec/assets/event_models_v2
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import skops.io as sio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from codebook import load_codebook

# Enough labeled examples, and enough of them positive, for a fitted model to
# mean anything. Classes below the floor are reported and skipped rather than
# shipped as a model trained on a handful of rows.
MIN_LABELED = 40
MIN_POSITIVE = 12

C_GRID = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]


def screen_error_rates(rows):
    """
    How often verification found a positive the screen had implied was a zero.

    Measured per event type from the audit sample. This is the number that
    decides whether a type's implied zeros can be used as training negatives:
    they are numerous and cost nothing, but if the screen misses positives at a
    high rate for some type, those "negatives" are largely mislabeled positives
    and will teach the model that genuine events are not events.

    The rate differs sharply by type. Narrow, concretely-worded categories are
    screened reliably; broad ones like ACCUSE -- which the codebook defines to
    cover "any kind of accusation, disapproval or criticism, investigation, or
    legal accusation" -- are not.
    """
    audited = Counter()
    missed = Counter()
    for row in rows:
        if row.get("screen_source") != "screen_implied" or row["mode"] is not None:
            continue
        audited[row["event_type"]] += 1
        if row["label"] == 1:
            missed[row["event_type"]] += 1
    return {t: (missed[t] / audited[t], audited[t]) for t in audited}


def load_annotations(path):
    """
    Long table -> {(id, event_type, mode): (label, weight)}, dropping NA.

    Implied zeros carry a weight below one, set from the measured screen error
    rate for their event type: a negative that the screen is wrong about 36% of
    the time should not count as much as one a verifier actually looked at.
    Verified labels always weigh 1.

    Down-weighting rather than discarding matters. Discarding the implied zeros
    outright is tempting -- they are the mislabeled ones -- but verification was
    run mostly on candidates, which are enriched for positives, so removing the
    implied zeros removes nearly every negative a broad type has. Doing that gave
    ACCUSE 992 positives against 39 negatives and a model that predicted "yes"
    unconditionally while scoring an F1 of 0.99, and left CONSULT with every
    coefficient at zero. A high F1 there measured the class balance, not the
    classifier.
    """
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    error_rates = screen_error_rates(rows)

    if error_rates:
        print("Screen error rate by event type (from the audit sample):")
        for event_type in sorted(error_rates):
            rate, n = error_rates[event_type]
            print(f"  {event_type:10s} {rate:5.1%} of {n:3d} audited "
                  f"-> implied zeros weighted {max(0.0, 1.0 - rate):.2f}")

    labels = {}
    n_na = 0
    for row in rows:
        if row["label"] == "NA":
            n_na += 1
            continue
        weight = 1.0
        if row["source"] == "screen_implied":
            rate, n = error_rates.get(row["event_type"], (0.0, 0))
            if n >= 20:
                weight = max(0.05, 1.0 - rate)
        key = (row["id"], row["event_type"], row["mode"])
        labels[key] = (int(row["label"]), weight)

    print(f"{len(labels)} usable label cells ({n_na} NA dropped)")
    return labels


def build_features(pool, emb_path, vectorizer=None, max_features=8000):
    """Combine cached embeddings with TF-IDF over the same documents."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ngec.classifiers.features import combine_features

    embeddings = np.load(emb_path)
    meta = json.load(open(Path(emb_path).with_suffix(".meta.json")))
    if meta["ids"] != [rec["id"] for rec in pool]:
        raise SystemExit("Embedding cache does not match the pool; re-run embed_pool.py")
    if not meta.get("chunked"):
        raise SystemExit("Embedding cache is unchunked; re-run embed_pool.py")

    texts = [rec["event_text"] for rec in pool]
    if vectorizer is None:
        # Fit on the whole pool, not just the labeled part. This is unsupervised
        # -- it sets the vocabulary and the IDF weights, and sees no labels -- and
        # a larger fitting sample gives more stable IDF for rare terms, which are
        # exactly the terms the mode models need.
        vectorizer = TfidfVectorizer(
            sublinear_tf=True, min_df=5, max_df=0.5,
            max_features=max_features, ngram_range=(1, 2),
            strip_accents="unicode", lowercase=True,
        )
        vectorizer.fit(texts)
    tfidf = vectorizer.transform(texts)
    print(f"features: {embeddings.shape[1]} embedding + {tfidf.shape[1]} tf-idf")
    return combine_features(embeddings, tfidf), vectorizer


def fit_one(X, y, w, seed=20260804):
    """
    Fit an L1 logistic regression, choosing C by held-out F1.

    `class_weight="balanced"` matters because the negatives dominate: most event
    types appear in a small minority of documents, and the implied zeros from
    screening make that imbalance larger still.

    `w` carries the per-example confidence weights from `load_annotations`, so a
    negative the screen is often wrong about pulls the boundary less hard than one
    a verifier confirmed. The evaluation split is scored unweighted -- the held-out
    F1 should reflect how often the model is right, not how confident we were in
    the labels it was fit on.
    """
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, w, test_size=0.2, random_state=seed, stratify=y)

    best = None
    for C in C_GRID:
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=C,
            class_weight="balanced", max_iter=2000, random_state=seed,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        proba = model.predict_proba(X_test)[:, 1]

        # Pick the threshold with the best F1 rather than assuming 0.5. The
        # shipped default of 0.9 was badly wrong for these models.
        best_at_C = None
        for threshold in np.arange(0.05, 0.96, 0.05):
            pred = (proba >= threshold).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0)
            if best_at_C is None or f1 > best_at_C["f1"]:
                best_at_C = {"f1": f1, "precision": p, "recall": r,
                             "threshold": float(threshold)}
        best_at_C.update(C=C, model=model)
        if best is None or best_at_C["f1"] > best["f1"]:
            best = best_at_C

    # Refit on everything once C is chosen, so the shipped model uses all the
    # annotation effort rather than 80% of it.
    final = LogisticRegression(
        penalty="l1", solver="liblinear", C=best["C"],
        class_weight="balanced", max_iter=2000, random_state=seed,
    )
    final.fit(X, y, sample_weight=w)
    best["model"] = final
    best["n_selected"] = int((final.coef_ != 0).sum())
    return best


def selected_terms(model, vectorizer, emb_dim, top_n=15):
    """The word features the lasso kept, strongest first."""
    coef = model.coef_[0][emb_dim:]
    vocab = vectorizer.get_feature_names_out()
    nonzero = np.flatnonzero(coef)
    order = nonzero[np.argsort(-np.abs(coef[nonzero]))][:top_n]
    return [(vocab[i], round(float(coef[i]), 3)) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/voa_pool.jsonl")
    ap.add_argument("--emb", default="data/pool_emb.npy")
    ap.add_argument("--annotations", default="data/annotations.jsonl")
    ap.add_argument("--out", default="../../../ngec/assets/event_models_v2")
    ap.add_argument("--max-features", type=int, default=8000,
                    help="TF-IDF vocabulary size. The shipped models use 8,000: a "
                         "60,000-word vocabulary scored the same on the holdout "
                         "(0.513 vs 0.521 macro F1, within noise) while making the "
                         "saved vectorizer 41 MB instead of 6 MB. The L1 penalty "
                         "keeps only a few hundred words per class either way.")
    ap.add_argument("--show-terms", action="store_true",
                    help="print the word features the lasso selected per class")
    args = ap.parse_args()

    book = load_codebook()
    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    index = {rec["id"]: i for i, rec in enumerate(pool)}

    labels = load_annotations(args.annotations)
    X, vectorizer = build_features(pool, args.emb, max_features=args.max_features)
    emb_dim = X.shape[1] - len(vectorizer.get_feature_names_out())

    out_dir = Path(args.out)
    (out_dir / "modes").mkdir(parents=True, exist_ok=True)

    # Group the sparse cells by the model each one trains.
    by_target = defaultdict(list)
    for (doc_id, event_type, mode), (label, weight) in labels.items():
        if doc_id in index:
            by_target[(event_type, mode)].append((index[doc_id], label, weight))

    report = {"types": {}, "modes": {}, "skipped": [],
              "screen_error_rates": {t: {"rate": r, "n_audited": n}
                                     for t, (r, n) in
                                     screen_error_rates(
                                         [json.loads(l) for l in
                                          open(args.annotations, encoding="utf-8")]
                                     ).items()}}

    for event_type in sorted(book):
        cells = by_target.get((event_type, None), [])
        rows = [i for i, _, _ in cells]
        y = np.array([lab for _, lab, _ in cells])
        w = np.array([wt for _, _, wt in cells])
        if len(y) < MIN_LABELED or y.sum() < MIN_POSITIVE:
            report["skipped"].append({"target": event_type, "n": len(y),
                                      "n_pos": int(y.sum())})
            print(f"  SKIP {event_type}: {len(y)} labeled, {int(y.sum())} positive")
            continue

        result = fit_one(X[rows], y, w)
        sio.dump(result["model"], out_dir / f"{event_type}.skops")
        report["types"][event_type] = {
            "f1": result["f1"], "precision": result["precision"],
            "recall": result["recall"], "threshold": result["threshold"],
            "C": result["C"], "n_labeled": len(y), "n_positive": int(y.sum()),
            "n_features_selected": result["n_selected"],
        }
        print(f"  {event_type:10s} f1={result['f1']:.3f} "
              f"p={result['precision']:.3f} r={result['recall']:.3f} "
              f"thr={result['threshold']:.2f} n={len(y)} pos={int(y.sum())} "
              f"feats={result['n_selected']}")
        if args.show_terms:
            terms = selected_terms(result["model"], vectorizer, emb_dim)
            print(f"      terms: {', '.join(t for t, _ in terms)}")

        # Mode models, conditional on the parent type being present.
        for mode in book[event_type]["modes"]:
            mode_cells = by_target.get((event_type, mode), [])
            mode_rows = [i for i, _, _ in mode_cells]
            y_mode = np.array([lab for _, lab, _ in mode_cells])
            w_mode = np.array([wt for _, _, wt in mode_cells])
            if len(y_mode) < MIN_LABELED or y_mode.sum() < MIN_POSITIVE:
                report["skipped"].append({
                    "target": f"{event_type}-{mode}", "n": len(y_mode),
                    "n_pos": int(y_mode.sum()) if len(y_mode) else 0})
                continue

            mode_result = fit_one(X[mode_rows], y_mode, w_mode)
            mode_dir = out_dir / "modes" / event_type
            mode_dir.mkdir(parents=True, exist_ok=True)
            sio.dump(mode_result["model"], mode_dir / f"{mode}.skops")
            report["modes"][f"{event_type}-{mode}"] = {
                "f1": mode_result["f1"], "precision": mode_result["precision"],
                "recall": mode_result["recall"],
                "threshold": mode_result["threshold"], "C": mode_result["C"],
                "n_labeled": len(y_mode), "n_positive": int(y_mode.sum()),
            }
            print(f"    {event_type}-{mode:16s} f1={mode_result['f1']:.3f} "
                  f"n={len(y_mode)} pos={int(y_mode.sum())}")
            if args.show_terms:
                terms = selected_terms(mode_result["model"], vectorizer, emb_dim)
                print(f"        terms: {', '.join(t for t, _ in terms)}")

    # The vectorizer is part of the model now: without the exact vocabulary and
    # IDF weights it was fit with, the word features cannot be rebuilt at serving
    # time and the coefficients would line up with the wrong terms.
    sio.dump(vectorizer, out_dir / "tfidf_vectorizer.skops")

    meta = json.load(open(Path(args.emb).with_suffix(".meta.json")))
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "encoder": meta["encoder"],
            "chunked": True,
            "features": "chunked mean-pooled embeddings + L1-penalized tf-idf",
            "annotation": "codebook-LLM (Sonnet) screen-then-verify over VOA articles",
            "event_types": sorted(report["types"]),
            "metrics": report["types"],
            "mode_metrics": report["modes"],
            "skipped": report["skipped"],
            "screen_error_rates": report["screen_error_rates"],
        }, f, indent=2)

    print(f"\n{len(report['types'])} type models, {len(report['modes'])} mode "
          f"models -> {out_dir}")
    if report["skipped"]:
        print(f"{len(report['skipped'])} targets skipped for too few labels; "
              f"see metadata.json")


if __name__ == "__main__":
    main()
