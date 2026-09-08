# Making NGEC's Wikipedia actor linking run on a CPU server

Experiment log for the `wiki-cpu-speed` branch of NGEC-2025 (branched from
`RandR`), 2026-09-02 to 2026-09-03. Written to stand alone: it assumes you know
roughly what NGEC does (news text in, PLOVER event records out) but nothing
about this branch.

## 1. The problem and the result

NGEC's fifth pipeline step, actor resolution, turns an actor mention such as
"the interior ministry" into a PLOVER actor code and, where one exists, a
Wikipedia article. When the pipeline was run as a demo on a 4-core CPU server
with no GPU, this step was unusably slow. Profiling put the cost at **about 37
seconds per mention**, and 96% of it on one line: embedding the intro paragraph
and short description of every one of ~190 Elasticsearch candidate articles with
`jinaai/jina-embeddings-v3`, a 572M-parameter model, on every query. A failed
lookup cost twice that, because the code re-ran the identical search as a
"fuzzy fallback".

The branch replaces that design. Measured on four performance cores of an
i9-12900K with `torch.set_num_threads(4)`, which is a generous stand-in for a
4-core cloud box:

| | `RandR` | `wiki-cpu-speed` |
|---|---|---|
| CPU time per actor mention, end to end (retrieval, features, encoders, ranker, code selection) | ~37 s | **0.28 s** |
| Of which the candidate-embedding step | 34 s | 0.03 s |
| Encoder downloads | 3.3 GB (jina, with `trust_remote_code`) | 250 MB (two small models) |
| Wikipedia-linking accuracy, gold set, document-level split, top-1 | 84.7% [78.6, 90.1] | 86.0% [81.6, 90.0] |
| Elasticsearch recall of the gold article within 200 candidates | 92.4% | 95.4% |
| Country-specific institutions ("the interior ministry" in a Ghana story) linked to that country's page, 173 probes | 24% of the 9 that reached the linker; 164 never did | 134 (77%), all reach the linker |
| Generic collectives ("police", "protesters") wrongly given a page, 103 probes | 61% | 1 |
| PLOVER actor categorization on ECAV (the paper's Table 6), gold spans / model spans | 55.2% / 41.0% | 56.5% / 41.7% |
| Substantive test suite (375 hand-written expectations) | 244 pass | 267 pass |

The accuracy numbers are all inside each other's confidence intervals except the
probe rows; the point is that the 130x speedup and 13x smaller download cost
nothing measurable, and that the linking became substantially better on the
one class of actor the gold set does not test. Section 5 says why.

## 2. How the stage works, and where the time went

`ActorResolver.actor_to_code` (`ngec/actors/actor_resolution.py`) does, for one
mention:

1. **Nationality stripping and NER.** `CountryDetector.search_nat` removes a
   nationality ("Kenyan police" to "police", country KEN); spaCy picks a core
   entity out of a noisy span ("Republican Senator Pat Roberts of Kansas" to
   "Pat Roberts").
2. **Agent matching.** The span is embedded and compared by cosine to 2,420
   embedded PLOVER role patterns (`agent_matcher.py`). A confident match used to
   end the process here, with no Wikipedia lookup.
3. **Retrieval.** `WikiSearcher.run_wiki_search` (`wiki_matcher.py`) queries the
   `wiki` Elasticsearch index (7.85M articles, 11 GB) and takes up to 200
   candidates.
4. **Features and ranking.** `_create_scoring_dataframe` computes ~40 features
   per candidate: exact and redirect title matches, edit distances, ES score,
   and four similarity features from a sentence encoder (context vs. intro,
   context vs. short description, actor description vs. each). An XGBoost ranker
   (`ngec/assets/xgb_model.json`) scores them; the top candidate is accepted if
   its probability clears a threshold.
5. **Code selection.** The chosen article's infobox, short description and
   categories are turned into a PLOVER code and country, reconciled with the
   agent match.

On CPU the profile of one lookup was: jina intro embedding 32.2 s, jina short
description embedding 2.9 s, jina context embedding 1.4 s, everything else
(Elasticsearch, spaCy, the MiniLM title-similarity model, edit distance,
XGBoost) under 0.2 s combined.

## 3. How the work was organised

The campaign ran as a PI-plus-subagents research programme: each question below
went to an agent with a written brief, and each returned a numbered report. Seven
research reports preceded any code change; then three implementation agents
worked on disjoint files with a retraining agent behind them. Every claim below
was measured; where a number is quoted it came from one of those runs.

### Evaluation sets

Nothing here would have been decidable without fixing the evaluation first.

- **Gold linking set.** `train_wiki_model/wiki_gold_standard_{1,2}.csv`: 1,966
  (document, mention, correct title) rows over 290 VOA stories; 331 rows (16.8%)
  have no correct article. (The paper text says 102; that counts only literal
  "None" strings and misses the blank cells the training code also treats as
  "no page".)
- **Document-level split.** The deployed ranker had been trained and tested on a
  row-level split. Because the same entity is annotated under several surface
  forms per story, 371 of the 392 held-out rows shared a document with training.
  Splitting by document costs 4 to 6 points of headline accuracy; every number
  in this log uses the document split unless marked leaky. Within-document
  correlation of correctness is about 0.2, so the 372 test rows carry ~167 rows'
  worth of information, and confidence intervals are document-clustered
  bootstraps. They are about 9 points wide, and the gold set **cannot
  distinguish any of the encoder or trim configurations tried**.
- **Policy probe set.** `train_wiki_model/probes_generic_institution.json`: 276
  mentions with context, 103 generic collectives that must get no page and 173
  country-specific institutions that must get their country's page. Built
  because the gold set's "no page" class is 96% obscure named people and tests
  neither policy. Two thirds of the institution probes are synthetic sentences;
  see next steps.
- **ECAV actor categorization** (`eval_ecav_actor_resolution.py` in
  `train_NGEC_2026`), the paper's own measure of PLOVER role coding.
- **Substantive tests** (`tests/substantive/`): 375 hand-written expectations,
  mostly for the agent matcher, none carrying a document. Regression pins, not
  a benchmark; 131 failed on `RandR` before any change.

## 4. What was tried

### 4.1 Options that were measured and rejected

**Pre-computing every article's embedding and storing it.** Feasible but too
big to ship. Elasticsearch 7.10's `dense_vector` cannot be read back out of the
index at all; a `binary` doc-values field would grow the 11 GB download to 48
GB. A memory-mapped int8 side file at full 1024 dimensions costs 12.4 GiB and
2.2 GPU-hours and changes 0 of 60 answers; Matryoshka truncation to 256
dimensions cuts that 4x but flips about 1 answer in 15. The residual per-query
cost would have been 0.23 s. Rejected on download size.

**Two-stage ranking: trim on cheap features, embed only the top k.** A cheap
XGBoost on lexical and ES-score features keeps the gold article in the top 10
of ~188 candidates 91.5% of the time against a 95% retrieval ceiling; raw ES
order manages only 82.7%. But dropping the trim into the existing ranker
collapsed abstention: the ranker's features are normalised by the maximum within
the candidate set, so a trimmed set makes every candidate look strong and
false positives on no-page mentions went from 5 to 30. Keeping full-set
normalisations, retuning the threshold, and retraining stage 2 on trimmed sets
recovered it to a 0.7-point cost at k=20 on the row split, and on the document
split k=20 actually beat the untrimmed model by 1.4 points. Viable, but
superseded: once the encoder cost 26 ms the trim saved 23 ms for a second model
and a normalisation rule.

**Quantising or exporting jina.** int8 dynamic quantisation: 0.99x (memory-bound
on short sequences). ONNX Runtime: 1.6x slower, and the exported graph always
applies a LoRA task adapter the pipeline never uses, so it is not even
equivalent. sentence-transformers' `backend="onnx"` is silently a no-op for
jina by the vendor's own guard. bf16 is unsupported on the CPU used. Batch
size 8 instead of 32 is a free 10% on 4 cores.

**Truncating the intro paragraph** to its first sentence: 2.3x faster with zero
answer changes on 40 queries, and 2.8x with the short description dropped. Not
adopted, on the user's judgement that historical roles in later sentences matter
for people, and because it became irrelevant with a cheap encoder.

**Bigger result windows from Elasticsearch.** 200 to 500 candidates buys 2.6
points of recall for 2.4x the candidates, the worst recall per unit cost of any
lever, since every candidate was embedded. **A low-boost match of the passage
against `intro_para`** as an in-index stand-in for context similarity saturates
every candidate list to 200 and is the slowest clause measured.

**A static embedding model for the agent matcher.** See 4.5.

### 4.2 The encoder does not matter much; the cheap features do

The central finding. With the four similarity features regenerated for each
encoder and the ranker retrained per encoder, held-out gold accuracy on 392 rows
(row split) was:

| Encoder | Parameters | Top-1 | CPU s per query, 188 candidates |
|---|---|---|---|
| bge-small-en-v1.5, with retrieval instruction | 33M | 87.8% | 3.2 |
| jina-embeddings-v3 | 572M | 87.2% | 28.3 |
| **static-retrieval-mrl-en-v1** (a token-to-vector lookup, no transformer) | 31M | **87.2%** | **0.026** |
| all-MiniLM-L6-v2 | 23M | 85.7% | 1.6 |
| no similarity features at all | 0 | 85.5% | 0 |
| bge-base-en-v1.5 | 109M | 85.2% | 11.2 |

The whole embedding stage is worth about 2 points on the gold set, bigger is
not better, and a static model ties the 572M transformer. The signal these
features carry ("does this article's opening paragraph look topically like this
story") is shallow enough that a bag of static vectors captures it.

The probe set told a sharper story. On institution probes, a ranker with **no**
similarity features got **0 of 173** right, picking unrelated pages (a Kenya
story's "supreme court" to the US Supreme Court). Any encoder got 67 to 68. So
the gold set was measuring the wrong thing: the features are near-worthless on
gold and load-bearing on policy.

But four microsecond-cost features matter more than any encoder:

| Feature | What it is | Ambiguous held-out rows recovered |
|---|---|---|
| Document-level country match | country named most often in the story appears in the candidate's intro / title / categories | +3 |
| Lexical context overlap | TF-IDF cosine and shared proper nouns between story and candidate intro, IDF table shipped as a 0.5 MB asset | +2 |
| Importance prior | number of redirects to the page (gold pages average 25, non-gold 3) and number of categories | 0 alone, 8.5% of ranker gain jointly |
| Title flags | "this is a generic concept page", "title names a different country than the story" | policy set only |

Jina's own contribution was +4 rows, concentrated in a 133-row ambiguous
stratum (persons with common surnames, institutions that exist in every
country), and 7 of its 12 wins were abstention confidence rather than
disambiguation. With the four features present, jina became redundant on gold
(same score with or without it), and on the probe set the cheap features took a
ranker from 0 to 92 institutions on their own, with a small encoder adding 12
to 15 more. The old `country_match` feature had been deliberately dropped from
the ranker after an ablation that was run with the country never passed at
inference, so the ablation measured a feature that was always zero.

### 4.3 Retrieval was the binding constraint, and mostly self-inflicted

43% of held-out linking errors were retrieval misses: the gold article was not
among the 200 candidates, and no ranker can fix that. The taxonomy of 124 misses:

| Class | n | Example |
|---|---|---|
| Generic description, country missing from the query | 22 | "Congress" to *United States Congress* (rank 406) |
| Spelling or transliteration | 21 | "Boris Johson" |
| Country-only span never sent to Wikipedia | 15 | "U.N.", "European Union", "DPRK" |
| Gold page renamed since annotation (irreducible) | 15 | *World Bank*, *Twitter* |
| NER or query expansion picked the wrong entity | 14 | "the European Union" sent as "Committee to Protect Journalists" |
| Regex misfires in nationality stripping | 12 | "U.S." matched inside "UWSA"; "Mexico's X" left "'s X" |
| Underspecified, disambiguated titles, acronyms, aliases | 25 | "Charles" to *Charles III*; "PRI" |

Two defects underneath: the three highest-boost clauses in the ES query, exact
`term` matches on title, redirects and alternative names, were dead code
because those fields are analysed text with no keyword sub-field (they fired on
0.4% of queries); and the country regexes were unescaped and unbounded. Fixes,
with recall@200 on the 1,635 gold rows that have a page:

| Change | Recall@200 | Extra candidates per query |
|---|---|---|
| `RandR` | 92.4% | |
| escape and word-bound the country patterns, fix possessives | 93.5% | 2.0 |
| second query on an alternate surface form, unioned | 94.7% | 4.0 |
| country-qualified title clauses ("X (Ghana)", "X of Ghana", "Ghana X") with head-noun inversions ("defence ministry" to "Ministry of Defence") | 95.3% | 4.0 |
| replace the dead `term` clauses with `match_phrase` | 96.3% | 4.0 |

On the institution probes the country-scoped query raised the target's presence
in the candidates from 121 to 139 of 173 and its median rank from 24 to 1.

### 4.4 The gate before the linker

The confident-agent-match early return was silently deciding policy. It sent
201 of the 1,966 gold mentions (177 with a real Wikipedia page) to a role code
with no lookup, mostly bare surnames ("Zuma", "Hollande") because spaCy tags no
entity in a one-word span, and it stopped 164 of 173 institution probes before
the linker ("the defence ministry" is lowercase, multi-word and a good agent
match). The user's policy: generic collectives must not get a page;
generically named institutions should get their country's page when the
context names the country.

The gate is now two named cases. A span never gets a lookup when it is a
generic collective ("police", "protesters", "the displaced"), a pronoun or
indefinite phrase ("they", "two men"), or a single lowercase common word. An
institution mention ("the central bank") goes to the linker whenever a country
is known from the passage or the span. Context-free calls with a merely
confident agent match behave as before, since there is nothing to disambiguate
"Defense Minister" with. Gated gold mentions went from 198 to 14, all of them
correctly. Organisation-like country-only spans ("U.N.", "DPRK") are now looked
up: 13 of 14 recover their page.

### 4.5 The agent matcher can leave jina too

The agent matcher's whole calibrated surface is one cosine threshold (0.6) over
2,420 pattern embeddings; the classifier its docstring promises does not exist.
bge-small's cosine scale coincides with jina's (mean best-match 0.795 vs 0.791),
so at the percentile-matched threshold 0.625 it agrees with jina's code on 72%
of gold and ECAV spans and moves ECAV actor categorization from 55.2% to 56.5%
on gold spans and 41.0% to 41.7% on model spans, both inside the documented
run-to-run drift. The static model was worse (53.7%): PLOVER patterns are
multi-word role phrases ("crowd control police") whose modifiers carry the code,
and static vectors cannot compose them. So the shipped configuration uses two
small encoders: static-retrieval-mrl for Wikipedia candidates and bge-small for
role patterns.

An accident along the way is worth recording. Flipping the wiki encoder default
also switched the agent matcher, because both used one loader, and the agent
matcher's cached pattern matrix did not know which model had built it. A unit
test caught it. The two encoders are now loaded separately and the cache hash
includes an encoder fingerprint.

### 4.6 The acceptance threshold

Retrained rankers calibrate at 0.5 to 0.85, not the shipped 0.1, but tuning a
threshold on the rows the ranker was fit on is unsafe (XGBoost separates its
own training data almost perfectly and the optimum wanders to 0.99). Swept end
to end for the static encoder:

| Threshold | Gold top-1, doc split | No-page rows correctly refused | Institution probes linked |
|---|---|---|---|
| 0.1 | 84.4% | 75.4% | 137 |
| **0.3** | **86.0%** | **89.2%** | 134 |
| 0.5 | 85.8% | 89.2% | 129 |
| 0.8 | 85.5% | 93.8% | 115 |

0.3 is where abstention stops paying for linking. The value is encoder-specific
(bge-small would stay at 0.1), so it lives in the encoder registry next to the
ranker asset.

## 5. Bugs fixed along the way

Each of these changed a published or shipped result, or would have.

1. `query_wiki` re-ran the identical search on a miss, doubling failed lookups.
2. `ActorResolver.process` passed no context to `actor_to_code`, so the
   production path never used the context features the ranker was trained
   with; the demo and the ECAV evaluation did pass it.
3. The resolution cache was keyed on mention and date only, so once context
   varied per document a repeated mention returned the first document's answer.
4. `country_match` was never populated at inference and had been dropped from
   the ranker on an ablation run in that state.
5. Country regexes unescaped and unbounded ("U.S." matched inside "UWSA"; "UN"
   inside "UNITA"; "France" inside "Frances"); possessive rules knew only the
   ASCII apostrophe.
6. The three exact-match ES clauses were dead against the index mapping.
7. The agent-matcher gate discarded 10.8% of linkable mentions.
8. The gold set has 331 no-page rows, not the 102 in the paper text; 15 gold
   titles no longer exist under that name.
9. The wiki-linking accuracy figures in the earlier evaluation notes are
   inflated 4 to 6 points by the row-level split.
10. `tests/substantive/test_actor_resolution.py` passed context and date
    positionally into the wrong parameters in 43 calls, so those tests had
    never exercised context.
11. `_normalize_scores` could emit -inf on a zero column maximum, which XGBoost
    rejects; only reachable with a static encoder, which maps an empty string
    to the zero vector.

## 6. Final configuration

- **Retrieval:** repaired exact-match clauses, country-qualified and head-noun
  variants when a country is known, two-query union over surface forms tagged
  with `from_alt_query`.
- **Features:** the 40 existing plus `cm_doc`, `cm_title`, `cm_cat`,
  `tfidf_ctx_intro`, `pn_overlap`, `pn_overlap_frac`, `n_categories`,
  `title_is_generic_concept`, `title_has_other_country`, `from_alt_query`; IDF
  table `ngec/assets/wiki_idf.json.gz` built by `setup/train_wiki_model/build_idf_table.py`.
- **Encoders:** `WIKI_ENCODERS` registry in `common.py`; wiki encoder
  `sentence-transformers/static-retrieval-mrl-en-v1` (override with
  `NGEC_WIKI_ENCODER`), agent encoder `BAAI/bge-small-en-v1.5` at cosine 0.625
  (override with `NGEC_AGENT_ENCODER`). jina remains selectable and its ranker
  asset is shipped, so the published numbers reproduce.
- **Rankers:** one asset per encoder, `xgb_model_{static-mrl,bge-small,jina}.json`,
  trained on the document split with `country_match` restored; `xgb_model.json`
  is the default's copy. Threshold per encoder (0.3 / 0.1 / 0.1).
- **Gate:** collective, pronoun and single-common-word pre-filters; institutions
  reach the linker when a country is known; country-only organisation spans are
  looked up.
- **Training and evaluation:** `train_NGEC_2026/train_wiki_model/02_generate_features.py`
  (features through the production input path, per encoder), `03_train_ranker.py`
  (document split, fit/calibration threshold check), `04_evaluate.py` (clustered
  CIs, probe set direct and end to end), `05_compare_encoders.py`,
  `06_threshold_sweep.py`; results in `EXPERIMENT_LOG.md` rounds 6 to 8.

## 7. Caveats

- The gold set cannot rank the configurations against each other; 1 held-out
  row is 0.27 points and the clustered CIs are ~9 points wide. The choices here
  rest on the probe set, the profile, and monotone trends, not on gold point
  estimates.
- The retrained rankers are gold-only. The deployed one also trained on an
  augmented feature file worth about 1.5 points; its generating scripts exist
  and were ported, but it was not used.
- The probe set is a smoke test, not a benchmark: two thirds synthetic, country
  coverage concentrated in anglophone Africa, no negative controls for the
  country channel.
- 18 substantive tests newly fail against `RandR` (41 newly pass). Ten are
  context-free calls where a low-scoring page is accepted or a linked page is
  coded differently ("Bank of Mexico" as GOV rather than BUS); eight are
  agent-matcher expectations written for jina. Both sets are listed in
  `train_wiki_model/verify/`.
- The remaining institution failures are one lexical bug: "central bank" and
  "parliament" still lose to the concept page in a handful of cases, and 34
  targets are absent from the candidates because of head-noun forms not yet
  covered ("Bank of Ghana" for "central bank").
- Timings are from fast desktop cores; a 2-core cloud box should be assumed
  2x slower. Elasticsearch, spaCy and the bge-small agent encode now dominate
  the 0.28 s.

## 8. Next steps

1. **A real policy evaluation set.** 400 to 600 mentions over 100+ documents
   from a corpus disjoint from training (ECAV's seven countries are the obvious
   source), with three labels: no page, country-specific page, generic page
   acceptable; plus negative controls with no country named. This is what would
   let the concept-page problem be worked on with a number attached.
2. **Contrastive gold, if any gold is collected.** The learning curve is flat
   from 391 training rows on for ambiguous mentions; undirected rows buy nothing.
   The same surface form ("Interior Ministry", "the ruling party") across 8 to
   12 documents from different countries is the contrast the ranker never sees.
   For confidence intervals the unit is documents: halving the CI width needs
   about four times the test documents.
3. **Retrieval residuals.** Index-side alias and acronym expansion ("PRI",
   "ISIS"), `fuzziness: AUTO` for transliterations (recovers the 24 zero-hit
   queries, loses 2 pages elsewhere), more head-noun inversions for banks and
   ministries, and re-annotating the 15 renamed gold pages.
4. **Retrain with the augmented data** regenerated through the new feature
   code (`02_generate_features.py --augmented`), worth about 1.5 points.
5. **Agent matcher.** Its 0.6/0.625 threshold is loose regardless of encoder (a
   percentile-matched 0.68 for jina scored 274 rather than 265 substantive
   passes); it has no labelled evaluation beyond ECAV; and if a single encoder
   is preferred over two, bge-small for the wiki stage too costs 3 s per
   mention untrimmed, which ONNX or int8 export of a plain BERT would likely
   halve.
6. **Code.** `_expand_query` still parses the whole document with spaCy per
   mention (57 ms; `actor_to_code` accepts a doc but the parse lives in
   `wiki_matcher`); ES scores from the two union queries are on different
   scales; `load_wiki_ranker_model` loads the same file into the "no context"
   slot and `text_is_empty` is a feature the booster never splits on.
7. **Measure on the actual demo hardware** before quoting 0.28 s anywhere.
8. **Paper text.** The no-page count, document-split accuracy with clustered
   CIs in place of the row-split figures, and the renamed gold pages.
9. **Repository.** `train_wiki_model/` is gitignored in `train_NGEC_2026`, so
   the gold CSVs and the old training pipeline that the retraining scripts
   depend on are not in git; `PIPELINE.md` and the other R&R documents in
   NGEC-2025 are untracked.

## 9. Commits on the branch

- `29c8304` Drop duplicated fallback search in query_wiki
- `56f89c8` Pass story text as context in ActorResolver.process
- `7c61e07` Detect country from context for the wiki ranker; key the cache on context
- `04f58d9` Escape and word-bound country patterns; handle curly and orphan possessives
- `3441f64` Add agents_file, priorities_file and override_sources options to ActorResolver
- `29a1d7b` Add PLOVER_priorities.csv, read by load_actor_priorities
- `101c0d6` Fall back to the span when NER's core entity is degenerate
- `4404b45` Let single capitalised spans past the agent-matcher gate
- `3ebb290` Repair the wiki search's exact-match clauses; add country and union queries
- `750d605` Skip the Wikipedia lookup for generic collectives
- `a50f850` Look up organisation-like country-only spans in Wikipedia
- `0aeb129` Give the wiki matcher the mention's other surface forms
- `a080088` Drop a spaCy parse per mention that was thrown away
- `b149725` Add cheap country and word-overlap features to the wiki ranker
- `9abd725` Keep the collective filter safe when the span fails to parse
- `da003e9` Make the wiki matcher's sentence transformer configurable
- `1875b6a` Document the wiki matcher's retrieval and ranking changes
- `92f1c59` Send generically-named institutions to the Wikipedia linker
- `85732c5` Skip the Wikipedia lookup for pronouns and indefinite phrases
- `edf5396` Pass context and date to actor_to_code by keyword in the substantive tests
- `8d1bef2` Retrain the wiki ranker per encoder; default to static-retrieval-mrl
- `7001d87` Keep the agent matcher on its own encoder; pair the wiki ranker with the wiki encoder
- `1fbaa05` Gate generic institutions on having a country; return None when nothing is known
- `3c8325a` Raise the wiki ranker's acceptance threshold to 0.3
- `b2aa356` Move the agent matcher to bge-small; read the ranker threshold from the encoder registry

Training scripts and logs: `train_NGEC_2026` commits `97c80b9`, `8cf94a2`, `b6df158`.
