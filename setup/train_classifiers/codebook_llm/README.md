# Codebook-LLM training data for the demo classifiers

This directory builds the demonstration event-type and event-mode classifiers
that ship in `ngec/assets/event_models_v2/`, by having an LLM apply the PLOVER
codebook to real news and using its judgments as training labels.

It replaces the earlier approach in the parent directory, which prompted a
language model to *write* synthetic news for each event type and used the prompt
as a pseudo-label. Those models scored well on held-out synthetic text and poorly
on real articles, because the synthetic text was easier and cleaner than the
thing the pipeline is actually pointed at.

The method here follows Halterman and Keith (2025) on using LLMs as codebook
annotators: give the model the codebook's real definition text, restructured into
instruction form, rather than the label name alone. A model asked "is this
PROTEST?" answers from its own sense of the word; a model given PLOVER's
definition and its coding rules answers from the codebook.

## Running it

Each step writes into `data/`. Steps 3 and 4 are done by LLM subagents reading
prompt files off disk, so they are driven from the agent session rather than the
shell.

```shell
python prepare_corpus.py --n 60000        # 1. corpus -> year-stratified pool
python codebook.py                        # 2. codebook CSV -> prompts
<gpu-venv>/bin/python embed_pool.py --device cuda   # 3. cache chunked embeddings
<gpu-venv>/bin/python seed_candidates.py --min-hits 1  # 4. find rare-mode candidates
python make_batches.py screen --n 2000    # 5. build screening prompts
#    ... subagents annotate data/batches/screen/*.txt -> *.json ...
python collect_annotations.py             # 6. gather labels
python build_verify_candidates.py         # 7. choose what to verify
python make_batches.py --batch-size 25 verify
#    ... subagents annotate data/batches/verify/*.txt -> *.json ...
python collect_annotations.py             # 8. gather again, now with verifications
python train_classifiers.py --show-terms  # 9. fit the models
python evaluate.py                        # 10. score on the never-seeded holdout
```

`embed_pool.py` needs only torch and sentence-transformers, so it can run from a
separate CUDA-matched environment when the project venv's torch build does not
match the driver. On a 4090 the whole pool encodes in about six minutes; on CPU
it takes around four hours.

## How the labels are made

**Screen, then verify.** Annotating 2,000 documents against 16 types is 32,000
judgments if done exhaustively. Instead:

1. **Screening** shows one document against all 16 definitions at once and asks
   which types it *might* contain. Deliberately recall-oriented. One call yields
   two or three candidates plus thirteen implied negatives.
2. **Verification** takes one candidate at a time, shows the full definition for
   that single type plus its modes and its most-confusable siblings, and asks for
   a careful 1/0/NA judgment with a supporting quote.

This is what makes the labels affordable: the negatives come free from screening,
and the expensive careful judgments are spent only where something might be there.

**Every positive needs a verbatim quote.** A quote that is not in the article
means the annotator summarized rather than quoted, so its evidence cannot be
checked and the judgment is discarded. The check allows typographic differences
and `...` elision, since those are ordinary quoting, but not paraphrase. About
1% of screening candidates and 2.5% of verified positives fail it.

**The screen's error rate is measured, not assumed.** Implied negatives are used
in bulk, so a random sample of them is sent to verification anyway. The result is
in `metadata.json` under `screen_error_rates` and it varies a lot:

| event type | screen missed a real positive |
|---|---|
| PROTEST, THREATEN | 2% |
| MOBILIZE, RETREAT | 4–6% |
| AGREE, AID, CONCEDE, COOPERATE, REJECT | 7–10% |
| ASSAULT, COERCE, REQUEST, SUPPORT | 11–14% |
| CONSULT | 17% |
| SANCTION | 26% |
| ACCUSE | 36% |

The pattern is interpretable: narrow, concretely-worded categories get screened
reliably, and broad ones do not. ACCUSE is defined to cover "any kind of
accusation, disapproval or criticism, investigation, or legal accusation", which
is most of political news, and a screen looking at sixteen definitions at once
misses it a third of the time.

Those rates become **per-example weights** on the implied negatives, not a filter.
An earlier version dropped implied negatives for the high-error types outright.
That looked principled and was wrong: verification runs mostly on candidates,
which are enriched for positives, so dropping the implied negatives left ACCUSE
with 992 positives against 39 negatives and a model that predicted "yes"
unconditionally — at an F1 of 0.99. A high score there was measuring the class
balance. Down-weighting keeps the negatives and their known unreliability.

**Rare modes are hunted, not sampled.** A random sample of news contains no
hunger strikes. `seed_candidates.py` finds candidates two ways — regexes written
from each mode definition, and nearest neighbours to the definition text in
embedding space — and verification works through them rarest-mode-first, so
stopping early still leaves the scarce labels done.

An early version of the seeder required two distinct regexes to match before
proposing a document. That silently reduced every THREATEN, MOBILIZE and REJECT
mode to zero candidates, because "threatened to arrest" is one construction, not
three co-occurring terms. Since candidates are only candidates, a loose seed costs
one wasted judgment while a strict one costs a mode its whole training set.

## What the models are

Features (`ngec/classifiers/features.py`, shared with serving so the two cannot
drift):

- **Chunked mean-pooled sentence embeddings.** Sentence-transformers truncate at
  384 word pieces. 76% of these articles are longer than that and on average 31%
  of an article was being discarded, so an event described in the second half of
  a story was invisible. Documents are split into overlapping word windows,
  each encoded, and the results averaged — 2.83 chunks per document here.
- **TF-IDF word features under an L1 penalty.** Dense embeddings smooth away the
  rare decisive terms the mode models need. The lasso keeps a readable handful
  per class; `--show-terms` prints them.

One binary model per event type, and one per (type, mode) pair fit only on
documents where the parent type is present — matching how they are applied.

Note there are **72** (type, mode) pairs, not 62 distinct mode names: `arrest`,
`ban` and `restrict` appear under both THREATEN and COERCE with different
definitions (threatening to do it versus doing it), and `assist`/`change`/
`meet`/`yield` under both REQUEST and REJECT with opposite polarity. The models
are per pair.

## Reading the numbers

`metadata.json` reports F1 from a stratified split of the annotated data. **That
is not a corpus-level estimate.** The annotated set is enriched — seeding went
looking for positives on purpose — so those figures answer "how well does this do
on documents selected for being interesting."

`evaluate.py` scores the models on a holdout of 500 documents drawn at random
before any seeding and excluded from it, then screened and verified the same way.
That is where the numbers below come from, and they are the ones to quote. Note
`base_rate` next to each score: for a type occurring in 2% of documents, a model
that never fires is 98% accurate, which is why accuracy is not reported at all.

### Holdout results

500 randomly-drawn, never-seeded documents, scored at the thresholds committed at
training time.

| event type | base rate | precision | recall | F1 | AP |
|---|---|---|---|---|---|
| ACCUSE | 0.41 | 0.764 | 0.668 | 0.713 | 0.806 |
| AGREE | 0.10 | 0.516 | 0.688 | 0.589 | 0.654 |
| AID | 0.09 | 0.500 | 0.523 | 0.511 | 0.545 |
| ASSAULT | 0.27 | 0.830 | 0.689 | 0.753 | 0.865 |
| COERCE | 0.17 | 0.454 | 0.802 | 0.580 | 0.670 |
| CONCEDE | 0.02 | 0.118 | 0.250 | 0.160 | 0.143 |
| CONSULT | 0.12 | 0.590 | 0.742 | 0.657 | 0.768 |
| COOPERATE | 0.04 | 0.367 | 0.500 | 0.423 | 0.390 |
| MOBILIZE | 0.06 | 0.500 | 0.633 | 0.559 | 0.572 |
| PROTEST | 0.12 | 0.577 | 0.763 | 0.657 | 0.756 |
| REJECT | 0.11 | 0.421 | 0.561 | 0.481 | 0.540 |
| REQUEST | 0.26 | 0.554 | 0.669 | 0.606 | 0.663 |
| RETREAT | 0.07 | 0.350 | 0.389 | 0.368 | 0.349 |
| SANCTION | 0.09 | 0.561 | 0.511 | 0.535 | 0.555 |
| SUPPORT | 0.10 | 0.278 | 0.653 | 0.390 | 0.386 |
| THREATEN | 0.09 | 0.341 | 0.349 | 0.345 | 0.370 |

Macro F1 is 0.521 over 16 types and 0.515 over the 55 modes with holdout positives.

### Against the models this replaces

Both scored on the same holdout. The old models are given the *best possible*
threshold on this holdout, and are fed unchunked embeddings — the way they were
trained — so the comparison is generous to them. The new models use the
thresholds committed at training time.

| event type | base rate | old F1 | new F1 | old AP | new AP |
|---|---|---|---|---|---|
| ACCUSE | 0.41 | 0.612 | **0.694** | 0.627 | **0.801** |
| AGREE | 0.10 | 0.438 | **0.542** | 0.491 | **0.616** |
| AID | 0.09 | 0.400 | **0.476** | 0.443 | **0.542** |
| ASSAULT | 0.27 | 0.624 | **0.782** | 0.663 | **0.860** |
| COERCE | 0.17 | 0.457 | **0.599** | 0.500 | **0.673** |
| CONCEDE | 0.02 | 0.095 | **0.273** | 0.077 | **0.118** |
| CONSULT | 0.12 | 0.488 | **0.667** | 0.505 | **0.772** |
| COOPERATE | 0.04 | 0.408 | **0.409** | 0.307 | **0.456** |
| MOBILIZE | 0.06 | 0.354 | **0.494** | 0.273 | **0.514** |
| PROTEST | 0.12 | 0.603 | **0.724** | 0.633 | **0.746** |
| REJECT | 0.11 | 0.217 | **0.388** | 0.158 | **0.449** |
| REQUEST | 0.26 | 0.430 | **0.577** | 0.316 | **0.654** |
| RETREAT | 0.07 | 0.375 | **0.400** | 0.300 | **0.424** |
| SANCTION | 0.09 | 0.350 | **0.384** | 0.336 | **0.496** |
| SUPPORT | 0.10 | 0.245 | **0.420** | 0.168 | **0.382** |
| THREATEN | 0.09 | 0.197 | **0.380** | 0.131 | **0.355** |
| **macro** | | **0.393** | **0.513** | **0.370** | **0.554** |

Every type improves: +0.12 macro F1 (+30%) and +0.18 macro average precision
(+49%). The gains are largest exactly where the old models were worst — THREATEN,
SUPPORT, REJECT, CONCEDE — which is the signature of training on real news
instead of synthetic text written to order.

Three things contributed and cannot be separated by this comparison alone: real
in-domain labels, chunked rather than truncated documents, and a serving path
that uses the encoder the models were trained on. The last was a live bug: the
shipped models were trained on `all-mpnet-base-v2` and served
`paraphrase-mpnet-base-v2`, which is silent because both are 768-dimensional.

The absolute numbers are still modest, and they should be. A macro F1 of 0.51 at
these base rates is a usable demonstration model, not a production one — these
are 2,000 annotated documents and one annotator, against POLECAT's hand-coded
training data.

## Known gaps

- **16 of the 72 mode pairs have too few positives to fit** and are listed under
  `skipped` in `metadata.json` — among them `PROTEST-hunger` (3 positives),
  `ASSAULT-cleansing` (4), `RETREAT-access` and `RETREAT-return` (0). They are
  genuinely rare in 2,000 VOA articles. Closing them needs an active-learning
  round: score the full 60k pool with the current models and send the
  high-scoring unlabeled documents for verification.
- **The labels are one LLM's judgments**, not adjudicated human coding. No
  second annotator, so there is no inter-coder agreement figure. The quote
  requirement and the audit arm are what stand in for that, and neither is the
  same thing.
- **Verification saw one type at a time.** That is what makes each judgment
  careful, but it means nothing enforces consistency *across* types for a
  document — a story can end up positive for two types whose definitions the
  codebook treats as alternatives.
