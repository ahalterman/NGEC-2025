# Replacing the demo event classifiers

Written for whoever picks this up next. `RUNNING.md` covers running the pipeline
and `PIPELINE.md` covers the data contracts between steps; this file covers step 1
specifically — where the shipped event and mode classifiers came from, why they
were rebuilt, and what is still wrong with them.

The method and the full numbers live in
`setup/train_classifiers/codebook_llm/README.md`. This is the orientation.

## Why

The demo classifiers were trained on synthetic news: a language model was
prompted with hand-written headlines to *write* articles for each event type, and
the prompt became the label (the approach in Halterman 2023). They scored 0.65–0.91
F1 on held-out synthetic text and noticeably worse on the VOA corpus the pipeline
is actually pointed at, which made the end-to-end evaluation hard to read — you
could not tell a pipeline bug from a classifier that had never seen real news.

Rebuilding them turned up three separate defects, only one of which was the
training data:

1. **The encoder did not match.** The models were trained on `all-mpnet-base-v2`
   embeddings and served `paraphrase-mpnet-base-v2` ones. Both are
   768-dimensional, so nothing ever raised an error; the probabilities were
   simply off the scale the models were fit on. This is also why
   `PloverCoder`'s default `event_threshold=0.9` appeared to work — under the
   correct encoder nothing scored above 0.76, so that threshold would have
   returned almost no events at all.
2. **Mode classification never ran.** `PloverCoder` never passed
   `mode_model_dir`, so `_load_mode_models` returned `{}` and `event_mode` was
   `[]` for every event ever coded. There were no mode models in the repo to
   load. That half of the ontology was missing, not degraded.
3. **Most of each article was discarded.** Sentence-transformers truncate at 384
   word pieces. 76% of VOA articles are longer than that, and on average 31% of
   an article never reached the classifier, so an event described below the fold
   was invisible.

## What replaced them

Labels come from an LLM applying the actual PLOVER codebook to real VOA articles,
following Halterman and Keith (2025): give the model the codebook's definition
text restructured into instruction form rather than the label name alone. Two
passes — a cheap recall-oriented screen over all 16 types at once, then a careful
per-type verification with the full definition and its most-confusable siblings.
Every positive requires a verbatim supporting quote. See the pipeline README.

Scale: 2,000 documents screened, 6,490 verification judgments, 55,527 label cells,
plus a separately annotated 500-document holdout.

Results on that holdout, drawn at random before any keyword seeding and excluded
from it. The old models get the best possible threshold on this very holdout and
unchunked embeddings (how they were trained), so the comparison is generous to
them:

| | old | new |
|---|---|---|
| macro F1 | 0.393 | **0.521** |
| macro average precision | 0.370 | **0.554** |

All 16 types improved. The largest gains are where the old models were worst —
THREATEN 0.197→0.345, SUPPORT 0.245→0.390, REJECT 0.217→0.481 — which is the
signature of training on real news rather than text written to order.

0.52 macro F1 is a usable demonstration model and not a production one. That is
the intended standing: these exist so the end-to-end test has a sane step 1 and so
researchers have a reference implementation to copy, not to produce POLECAT.

## Where things live

**Outside the repo.** The corpus is not redistributable through git:

| what | where | size |
|---|---|---|
| VOA articles (319,248 JSON) | `~/projects/eng_voanews/article/` | 4.0 GB |
| the tarball it came from | `~/projects/eng_voanews/en_voa_articles.tar.gz` | 875 MB |
| VOA video/audio transcripts | `~/projects/eng_voanews/{video,audio}/` | 78 MB |

The video/audio set is the short-form transcript corpus (median 221 characters);
only ~25% clears the 300-character floor. It was **not** used — the article corpus
replaced it. Anything referencing the earlier VOA sample is talking about that
smaller set.

**In the repo, tracked:**

- `ngec/assets/event_models_v2/` — 16 type models, 56 mode models, the fitted
  TF-IDF vectorizer, and `metadata.json` (encoder name, per-class thresholds,
  training metrics, measured screen error rates). 11 MB.
- `ngec/classifiers/features.py` — chunking and feature construction, imported by
  both the trainer and the serving classifier so they cannot drift apart again.
- `setup/train_classifiers/codebook_llm/` — the eight pipeline scripts and their
  README.
- `setup/train_classifiers/codebook_llm/data/batches/**/*.json` — **the 469 raw
  annotation files.** These are the LLM's actual judgments and the only artifact
  here that cannot be regenerated without redoing the entire labeling run. About
  3 MB. `.gitignore` is written to keep these while excluding everything around
  them; if you edit it, preserve that.

**In the repo, deliberately ignored** (regenerable from the scripts plus the
corpus): `data/voa_pool.jsonl` (195 MB), `data/pool_emb.npy`, the rendered
`*.txt` prompt files, and the pilot batches. About 450 MB.

**Gone.** Embedding used a throwaway venv in the session scratch directory with a
CUDA-12-compatible torch build, because the project venv has a CUDA-13 torch on a
CUDA-12 driver (`RUNNING.md` §4). It is not persisted. Rebuild with:

```shell
uv venv --python 3.12 /tmp/gpuenv
VIRTUAL_ENV=/tmp/gpuenv uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
VIRTUAL_ENV=/tmp/gpuenv uv pip install sentence-transformers scikit-learn
```

That takes encoding the 60k pool from ~4.3 hours to about 6 minutes. Fixing the
project venv's torch would also speed up the attribute LLM in step 4, and is
probably worth doing properly at some point.

## What changed in the package

- `ngec/classifiers/features.py` — new. Chunked mean-pooled embeddings, TF-IDF
  concatenation.
- `ngec/classifiers/plover_sklearn.py` — reads the encoder and per-class
  thresholds from the model directory's `metadata.json` instead of hardcoding
  them; loads the TF-IDF vectorizer; defaults `mode_model_dir` to a `modes/`
  subdirectory of the type model directory; scores a whole batch per model rather
  than one document at a time; default model directory is now `event_models_v2`.
- `ngec/plover_coder.py` — `event_threshold` now defaults to `None`, meaning "use
  the per-class thresholds the models were fit with". Passing a number still
  overrides every class at once, which is almost never what you want: the
  F1-maximizing thresholds range from 0.35 to 0.75 across types, and 0.1 to
  0.8 across modes (`metadata.json`).
- `tests/classifiers/test_plover_sklearn.py` — rewritten. The old tests passed
  `threshold=0.9` and only passed *because* of the encoder mismatch. Added guards
  that the encoder matches metadata and that modes are never reported without
  their parent type.
- `tests/test_end_to_end.py` — dropped the two `threshold=0.9` arguments.

`ngec/assets/test_event_models/` is left in place. Nothing points at it now, but
it is what the older metrics in the repo refer to, so deleting it would strand
them.

## Outstanding

**Sixteen of the 72 (type, mode) pairs have no model.** Listed under `skipped` in
`metadata.json`. Seeding found candidates for every one of them; verification did
not confirm enough positives. Worst: `RETREAT-access` and `RETREAT-return` (0
positives), `PROTEST-hunger` (3), `ASSAULT-cleansing` (4), `THREATEN-expel` (1).
The fix is the active-learning round that was scoped but not run — score all 60k
pooled documents with the current models, take the high-scoring unlabeled ones for
each starved mode, and send those to verification. The scripts support this
already; `build_verify_candidates.py` just needs a fourth source alongside
seed/screen/audit.

**Note that mode names repeat across types** — there are 72 (type, mode) pairs but
only 62 distinct mode strings. `arrest`, `ban` and `restrict` appear under both
THREATEN and COERCE with opposite meanings (threatening to do it versus doing it),
and `assist`/`change`/`meet`/`yield` under both REQUEST and REJECT with opposite
polarity. Models are per pair. Anything that keys modes by name alone is wrong.

**CONCEDE is not usable.** F1 0.16 on the holdout, 8 positives at a 1.6% base
rate. Either annotate a lot more of it or document it as unsupported.

**Some mode-level holdout cells are noise.** `RETREAT-resign` reports F1 1.000 on
two positives. Do not put those in a table without their denominators; several
mode rows have fewer than five holdout positives.

**One annotator, no adjudication.** There is no inter-coder agreement figure
because there is only one coder. The verbatim-quote requirement and the audit arm
are what stand in for that, and neither is the same thing. If the article needs a
reliability number, a second independent annotation pass over a few hundred
documents plus your own adjudication of the disagreements would give one.

**Verification saw one type at a time.** That is what makes each judgment careful,
but nothing enforces consistency *across* types within a document, so a story can
come back positive for two types the codebook treats as alternatives.

**The lasso was memorizing entities at small sample sizes.** At 400 documents it
was selecting `trump`, `zuma`, `sinaloa`, `kashmir`. That largely washed out by
2,000 documents, but it is worth re-running `train_classifiers.py --show-terms`
after any future data addition and reading the selected words — they should look
like event language, not a list of proper nouns.

**TF-IDF vocabulary is 8,000 words**, not 60,000. A 60k vocabulary scored the same
on the holdout (0.513 vs 0.521 macro F1, within noise) while making the saved
vectorizer 41 MB instead of 6 MB. If you retrain, keep an eye on whether that
still holds with more data.
