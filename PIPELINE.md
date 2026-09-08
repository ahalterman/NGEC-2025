# The NGEC Pipeline: Data Contracts, Failure Points, and Notes

This document traces the NGEC event-coding pipeline step by step, checking that
the **output of each step matches the expected input of the next**. It records
where the pipeline works well, where it breaks, and what would make it more
robust. It was written as part of the journal-article revise-and-resubmit, after
a reviewer had trouble running the code.

The reference orchestrator is `PloverCoder.process()` in `ngec/plover_coder.py`,
which is also what the README and `tests/test_end_to_end.py` exercise. (The
top-level `ngec_process.py` is an older CLI entry point that predates
`PloverCoder` and is not the maintained path.)

## The six steps

```
story_list (dicts: event_text, id, pub_date)
  │
  ├─1─ PloverSklearnClassifier.process(story_list)          ngec/classifiers/plover_sklearn.py
  │      + event_type: list[str], event_type_confidence: dict, event_mode: list[str]
  │
  ├─2─ GeolocationModel.process(story_list, doc_list)       ngec/geolocation.py
  │      + geolocated_ents: list[dict]
  │
  ├─3─ stories_to_events(story_list, doc_list)              ngec/utilities.py
  │      splits 1 story → N events (one per event_type)
  │      + story_people/story_organizations/story_places, _doc_position, orig_id
  │      event_type: list[str] → str,  event_mode: list[str] → str
  │
  ├─4─ AttributeModel.process(event_list)                   ngec/attribute_model.py
  │      LLM extraction, then explode_events (ngec/utilities.py):
  │      one record per extracted event, attributes: dict; empty ones dropped
  │      + attributes: dict,  id gains a "_<index>" suffix
  │
  ├─5─ ActorResolver.process(event_list)                    ngec/actors/actor_resolution.py
  │      + actor: list[dict], recipient: list[dict]  (top-level)
  │
  └─6─ Formatter.process(event_list, return_raw=True)       ngec/formatter.py
         + event_location: dict, date_resolved: dict  (top-level)
```

## Step-by-step contract check

### Step 1 → 2: event classifier → geolocation ✅
`PloverSklearnClassifier.process` adds `event_type` (`list[str]`),
`event_type_confidence` (`dict`), and `event_mode` (`list[str]` of
`"EVENT-mode"` strings). Geolocation ignores all of these — it only needs the
spaCy `doc_list` — so the handoff is clean. Geolocation adds `geolocated_ents`.

### Step 2 → 3: geolocation → stories_to_events ✅
`stories_to_events` reads `event_type` (list) and `event_mode` (list), which
step 1 produced. It "lengthens" each story into one event per event type,
copying the dict (`ex.copy()`), and **collapses `event_type` from a list to a
single string** and `event_mode` to a single string. This is the intended shape
change and downstream steps rely on it.

Minor note: if `event_mode` is absent it defaults to `[]` (handled), but
`event_type` is accessed directly (`ex['event_type']`) — a story that reached
this step without going through step 1 would `KeyError`.

### Step 3 → 4: stories_to_events → attribute model ✅ (input side)
`AttributeModel.process` needs `event_text` and `event_type` (str), optionally
`event_mode` (str). All present. `_get_event_info` looks the event type up in the
codebook with `.values[0]`, so an `event_type` **not in the codebook** would
`IndexError` — fine for on-ontology types, brittle for anything off-ontology.

### Step 4 → 5: attribute model → actor resolver ✅ (fixed; see history below)
This *was* the fragile seam — the "list that becomes a dict" that crashed the
pipeline. It was reworked so the attribute step **explodes** each extracted
event into its own record with a single `attributes` **dict**, dropping records
with no extracted event. The original problem and the fix are documented in "The
core issue" below.

### Step 5 → 6: actor resolver → formatter ✅
`ActorResolver.process` reads `event["attributes"].get("actor")` and
`.get("recipient")` (lists of strings), resolves each, and writes **top-level**
`event["actor"]` / `event["recipient"]` as lists of resolved-actor dicts. The
formatter's active code doesn't depend on those top-level keys; they flow
through to the raw output. Clean.

### Step 6: formatter (terminal) ⚠️ partial
The active `process()` sets `event_location` (via `pick_event_loc`) and
`date_resolved` (via `resolve_date`). Two other methods — `find_event_loc` and
`add_meta` — are **commented out** and expect an older attribute format (see
"Format drift" below).

---

## The core issue: `attributes` — list vs. dict, and the crash

> **Status: FIXED.** The attribute step now **explodes** multi-event extractions
> into separate event records, each with a single `attributes` dict, and drops
> records where nothing was extracted. The description below is the original
> problem; the resolution and the new contract are in "The fix" at the end of
> this section.

The attribute LLM is prompted to return a **JSON array of event objects**
(`_make_system_content_short`: *"Empty array `[]` if no events."*). So
`call_llm_batch` yields, per event, a `list` — usually one element, sometimes
zero, occasionally more. On any JSON-decode error it substitutes `[]`
(`attribute_model.py:379`).

Post-processing then does (`attribute_model.py:454`):

```python
event_list[n]['attributes'] = attributes[0]
```

This single line has three distinct consequences:

1. **List → dict (the type change you remembered).** `attributes` is a list; it
   is stored as a single dict (`attributes[0]`). Downstream code *depends* on
   the dict form: `ActorResolver.process` does `event["attributes"]["actor"]`
   (`actor_resolution.py:1092`) and the formatter does
   `event['attributes'].get('location', ...)` (`formatter.py:511`). So the code
   only works *because of* the `[0]` — but the module's own docstrings and
   `__main__` examples show `attributes` as a **list**
   (`attribute_model.py:519`, `actor_resolution.py:1229`). The contract is
   genuinely inconsistent across the codebase; the docstrings are wrong relative
   to the running code.

2. **Silent data loss.** If the model extracts two events from one document (two
   distinct assaults in one paragraph, say), only the first survives. Nothing is
   logged.

3. **`IndexError` crash on empty extraction.** When the model returns `[]` — which
   the prompt *explicitly instructs* for documents with no event, and which also
   happens on every JSON-decode failure — `attributes[0]` raises
   `IndexError: list index out of range` and **the entire batch dies**. A single
   bad or empty LLM response takes down the whole run. This is the most likely
   cause of the reviewer's failure: it doesn't require anything exotic, just one
   document where the model declines to extract.

All three were reproduced in isolation (pure Python, no models):

| LLM returns | `attributes[0]` result |
|---|---|
| one event `[{...}]` | dict — works |
| two events `[{...},{...}]` | dict of the **first only** — second silently dropped |
| no events `[]` | **`IndexError`, pipeline crashes** |

### Two more crashes on the same seam

- **`KeyError` in the actor resolver.** `event["attributes"][attribute_key]` for
  `attribute_key in ["actor", "recipient"]` (`actor_resolution.py:1092`) indexes
  directly. If the model omits `recipient` (or `actor`) from its JSON — common —
  this `KeyError`s. Reproduced.
- **`AttributeError` in the formatter.** `event.get('attributes', ["N/A"]).get('location', ...)`
  (`formatter.py:511`) uses a **list** as the default; if `attributes` is missing
  the fallback `["N/A"]` has no `.get`, giving
  `AttributeError: 'list' object has no attribute 'get'`. The `resolve_date`
  path is guarded by try/except, but this line is not.

## Format drift (lower severity, but confusing)

Two attribute layouts coexist in the repo:

- **Current (lowercase):** `attributes = {'actor': [...], 'recipient': [...], 'date': [...], 'location': [...]}` — produced by the LLM, consumed by the actor resolver and by `pick_event_loc`/`resolve_date`.
- **Legacy (uppercase, span-based):** `attributes = {'ACTOR': [{'text':..., 'qa_start_char':..., 'qa_end_char':...}], 'LOC': [...], 'RECIP': [...]}` — this is what the QA/span attribute model used to emit. The formatter's `find_event_loc` and `add_meta` still assume it (`formatter.py:417,422,479`), as does the big docstring block. They're commented out of `process`, so they're dead, but they mislead a reader about the current contract.

## What works well

- **Steps 1–3 are solid.** Types line up, the story→event lengthening is clean,
  and the shape changes are intentional and consistently consumed.
- **`deepcopy` in the actor resolver** (`actor_resolution.py:1083`) avoids
  mutating the caller's list — good hygiene. (The attribute model, by contrast,
  mutates in place and says so.)
- **The date resolver** (`_resolve_date`) is carefully written: it handles every
  combination of missing date string / missing reference date with explicit
  `match` cases and reasons, and falls back to the publication date rather than
  crashing. (It has since been extended — see "Date resolution" below.)
- **`pick_event_loc`** similarly enumerates all four missing-input combinations
  before doing work.
- **Error handling exists where it was added deliberately** (JSON decode →
  `[]`; date resolution wrapped in try/except). The gaps are the spots that
  weren't reached during earlier single-happy-path testing.

## The fix (what was changed)

The chosen contract **explodes** the multiplicity upstream instead of carrying a
list downstream. The attribute step's output has **one record per extracted
event**, each with a single `attributes` **dict** — the same "lengthening"
pattern `stories_to_events` already uses for event types. This keeps multi-event
extractions, removes every crash on this seam, keeps `attributes` a plain dict
(so downstream code is simple), and drops empty extractions rather than emitting
junk.

**The explode step** (`explode_events` in `ngec/utilities.py`, called at the end
of `AttributeModel.process` so it can't be skipped): for each record whose
`attributes` is a list of *k* extracted events, emit *k* records — each a shallow
copy with `attributes` set to one event dict and `id` set to `<base id>_<index>`
(e.g. `story1_PROTEST__0`, `__1`). `orig_id` remains the stable link back to the
story. Records with **zero** extracted events are dropped (see next paragraph),
not emitted.

**Dropped (empty) events** are kept out of the main output — people are bad at
filtering pipeline output, so we don't put empty-attribute junk in it — but they
are *loudly* accounted for: `AttributeModel._report_dropped` logs a WARNING with
the total count and the per-`event_type` distribution, and writes the dropped
records to a timestamped `*_dropped_events.jsonl` file for inspection.

The **`attributes` contract** is now a single dict per record:

```python
{
  "event_type": "PROTEST",             # the LLM's (possibly paraphrased) echo;
  "anchor_quote": "...",               #   the record's top-level event_type stays authoritative
  "actor":     ["Protesters"],         # raw extraction (str list)
  "recipient": ["the government"],
  "date":      ["today"],
  "location":  ["Paris"],
}
```

and the resolved outputs sit at the **top level** of the record (as the README
first documented):

```python
"actor":          [ {..coded actor..} ],   # added by ActorResolver
"recipient":      [ {..coded actor..} ],   # added by ActorResolver
"event_location": {"event_loc": {...}, "reason": "success"},           # added by Formatter
"date_resolved":  {"resolved_date": ..., "date_end": ..., "granularity": ...,  # Formatter
                   "date_type": ..., "reason": ...},
```

Concretely:

- **`ngec/utilities.py`** gains `explode_events(event_list) -> (exploded, dropped)`,
  a pure, unit-tested function (see `tests/test_utilities.py`).
- **`attribute_model.py`** calls `explode_events` and reports drops. Because it
  now explodes and drops, `process()` **returns a new list** rather than mutating
  the input in place — callers must use the return value. `attributes` is typed
  as a single `Attributes` dict again.
- **`actor_resolution.py`** reads `event.get("attributes", {})` then
  `attributes.get(attribute_key, [])`, and writes top-level `event["actor"]` /
  `event["recipient"]`.
- **`formatter.py`** reads the single `attributes` dict and sets top-level
  `event["event_location"]` and `event["date_resolved"]`; `resolve_date(event)`
  guards the empty/missing-date case.
- Tests updated: `test_utilities.py` (new `explode_events` cases),
  `test_attribute_model.py` (uses the return value; `attributes` is a dict),
  `test_date_resolver.py` (back to `resolve_date(event)`). `examples/plover_app.py`
  renders one row per (now already-exploded) event.

**Event IDs are per-run artifacts.** The attribute model samples at
`temperature=0.5`, so the number and order of extracted events — and thus the
`_<index>` suffixes — are not stable across runs. `orig_id` is the stable join
key back to the source story. If cross-run-stable event IDs are ever needed, a
content hash (e.g. of `anchor_quote`) would be required instead.

## Date resolution

Step 6 originally resolved a date span with a single `dateparser` call, one retry
for "next/later", and otherwise stamped the publication date. That worked for
absolute dates and the handful of relative forms `dateparser` knows natively, but
it silently fell back to the pub date for a large share of the spans the
attribute model actually emits ("late Tuesday night", "since Thursday", "over the
weekend", "March 15-20").

`_resolve_date` now hands the span to **`_resolve_core`**, a cascade of ordered
strategies (direct parse, parentheses, "N units ago", "N-day-old",
"X before/after Y", bounded and open-ended ranges, quarters, future and past
modifiers, weekends, "early/mid/late", "Nth week of <month>", decades, clock-time
and noise stripping). Each step either resolves or falls through; structural
steps recurse on a strictly shorter sub-expression, so nested relatives such as
"a week before last Friday" work and recursion always terminates.

**The design rule is fail-safe, not best-effort.** A step that cannot resolve
returns `None` rather than guessing, and the caller stamps the publication date
flagged `unresolved`. Several steps exist purely to *refuse*: "three days of
clashes" is a duration, not an offset, so it must not become "three days ago"
(step 0c); a bare period word left after stripping a modifier ("beginning of the
year" → "the year") would parse to *today*, so it is anchored to the pub-date
period instead (step 13 / `_anchor_bare_period`). Spans needing world knowledge
("the anniversary of the coup", "last Ramadan", "the 2024 election") stay
unresolved by design, and `tests/test_date_resolver.py::test_correct_rejections`
locks that in.

### The `date_resolved` schema

`ResolvedDate` carries three orthogonal pieces of information, deliberately kept
apart because they answer different questions:

| field | meaning |
|---|---|
| `resolved_date` | the date, or the **start** of a range |
| `date_end` | the **end** of a genuine range; `None` for a point or an open-ended range |
| `granularity` | precision *unit*: `day` / `week` / `month` / `quarter` / `year` |
| `date_type` | `exact` / `approximate` / `range` / `unresolved` |
| `reason` | human-readable trail, chained with `←` through nested resolutions |

`granularity` says how *coarsely* the date is known. It does **not** say whether
the event spanned several days (that's `date_end`) or whether the source was
vague (that's `date_type`). "over the weekend" is `approximate` at `week`
granularity with no `date_end` — ambiguity about *which* day, not a two-day
event. "Tuesday to Thursday" is a `range` at `day` granularity *with* a
`date_end`.

**Schema change:** `date_end` and `date_type` are new fields, and the old
`granularity="uncertain"` sentinel is gone. Unresolved spans now report
`granularity=None` with `date_type="unresolved"`. Anything downstream that keyed
on the string `"uncertain"` must switch to `date_type == "unresolved"`.

### Things worth knowing

- **Range endpoints are resolved independently**, which can put the end before
  the start — either because only the second endpoint states a year ("March to
  April 2024") or because the past preference pushes the end behind the start
  ("Thursday through Tuesday"). `_align_range` repairs both, and drops `date_end`
  rather than emit an inverted range if neither repair works.
- **A bounded connector with a non-date on the right is not a range.** "Monday to
  discuss the deal" resolves to Monday and keeps its own type; only a right-hand
  side that actually resolves produces `date_type="range"`.
- **Publication dates are normalized to naive midnight** before use. They arrive
  with times and timezones, and without normalizing, branches that do arithmetic
  off the reference return tz-aware datetimes while branches that construct a
  date return naive ones — a column holding both is unusable in pandas.
- **A bare `-` is not a range separator** (it would wreck "mid-March", "Covid-19",
  and ISO dates). A hyphen with whitespace on both sides is, and day-of-month
  hyphen ranges ("March 15-20") get their own step that requires a flanking month
  name.
- **`dateparser` version sensitivity.** The cascade leans on `dateparser`'s parse
  behavior much more than the old two-call version did, so the dependency is
  pinned narrowly in `pyproject.toml`. Widen it only with the test suite green.
- **Known soft spot:** "earlier this week" resolves to the pub date typed `exact`
  at `week` granularity. The week is right and the granularity flags the
  imprecision, but the type overstates confidence. Distinguishing it from
  "earlier Tuesday" (genuinely exact) needs more than a word check.

## Actor resolution: matching a mention to a Wikipedia article

Step 5 has two halves. `AgentMatcher` handles generic role mentions ("police",
"protesters") with pattern files; `WikiMatcher` handles named and institutional
mentions by searching an Elasticsearch index of Wikipedia and then *ranking* the
candidates it gets back with a small XGBoost model
(`ngec/assets/xgb_model.json`). Retrieval and ranking fail in different ways and
were fixed separately.

### Retrieval: `ngec/actors/wiki_matcher.py`, `WikiSearcher.run_wiki_search`

**The three highest-boosted clauses in the search were dead.** They were `term`
queries against `title`, `redirects` and `alternative_names`. Those are analyzed
`text` fields with no `.keyword` sub-field, so a `term` query matches only when
the whole query happens to be one already-lowercase token. Measured on the
1,966-row wiki gold set, they fired on **0.4% of queries**. They are now
`match_phrase` (all the query's tokens, in order) and `match ... operator:
"and"` (all the tokens, any order), which is what those boosts were meant to
express.

**Country-qualified title variants.** News stories name institutions the way a
reader would ("the defense ministry"); Wikipedia names them the way a catalogue
would ("Ministry of Defence (Ghana)"). `country_phrase_variants` builds the
strings that bridge the two — the parenthetical, "X of Ghana", "Ghana X", plus
head-noun inversions for "... ministry"/"... department" and central banks, and
both spellings of defence/defense — and `run_wiki_search` adds them as
`match_phrase` clauses when a country is known.

**A union query over surface forms.** `query_wiki` takes `alt_query_terms`, other
surface forms of the same mention (the raw span before country-stripping, the
span before NER expansion). The first one that differs from the primary term is
searched as well and the two ranked lists are interleaved by
`merge_ranked_results`. Whichever surface form the article is actually titled
under is often not the one the pipeline settled on; offline this is worth about
2.6 points of recall@200 overall and 16.6 on polysemous surface forms, for one
extra Elasticsearch round trip. Candidates from the second search carry
`from_alt_query=1`, because their `raw_es_score` comes from a different query
and is not on the same scale.

### Ranking: `WikiMatcher._create_scoring_dataframe`

The ranker sees one row per candidate article. Alongside the string-similarity
and Elasticsearch-score columns that were already there, it now gets features
computed from strings already in hand:

- **`cm_doc`, `cm_title`, `cm_cat`** — does the country the *story* is about
  appear in the candidate's intro/short description, its title, or its
  categories? The story's country is the country named most often in the
  context (`CountryDetector.most_frequent_country`). The older `country_match`
  asks the same question of the `country` argument the caller passes, which in
  the pipeline is usually empty; it is kept.
- **`tfidf_ctx_intro`** — TF-IDF cosine between the story and the candidate's
  intro paragraph, sublinear tf, weighted by `ngec/assets/wiki_idf.json.gz`.
  That table is a 200,000-article sample of the index with a fixed
  `random_score` seed, built by `setup/wiki/build_idf_table.py`, capped at the
  100,000 most widely-seen words. The builder imports its tokenizer from
  `wiki_matcher` so the asset and the runtime cannot drift.
- **`pn_overlap`, `pn_overlap_frac`** — capitalised words shared between story
  and intro, ignoring sentence-initial words and the query's own words. A cheap
  stand-in for "these two texts are about the same people and places".
- **`n_categories`** — how many categories the article has, a crude prominence
  prior alongside the existing `redirect_names_count`.
- **`title_is_generic_concept`, `title_has_other_country`** — Wikipedia has both
  `Interior ministry` (the concept) and `Ministry of the Interior (Ghana)` (the
  institution). The concept page looks like a good match on every string
  feature and is almost never the right answer, so the ranker needs to be able
  to see the difference.
- **`from_alt_query`** — see above.

Offline, these recover more than the sentence-transformer embedding does; on a
probe set of institution mentions they are the difference between 55% correct
and 0%.

### The encoder is now a setting

`WIKI_ENCODERS` in `ngec/actors/common.py` maps a sentence-transformer id to its
load arguments and to the instruction, if any, that model wants on the *query*
side. Pick one with `ModelManager(encoder_name=...)` or the `NGEC_WIKI_ENCODER`
environment variable. Three are registered: `jinaai/jina-embeddings-v3` (the
default), `BAAI/bge-small-en-v1.5` (about nine times cheaper on CPU, slightly
more accurate — but only with its retrieval instruction, which is worth about a
point) and `sentence-transformers/static-retrieval-mrl-en-v1` (no transformer at
all, three orders of magnitude cheaper, same accuracy). `WikiMatcher` applies the
prefix to the story and the actor description only, never to article intros or
short descriptions: those are the passages being searched, and getting it
backwards is silent.

### The ranker asset and the features go together

`_call_ranker` selects columns by `model.feature_names_in_`, so adding a feature
is inert until the model is retrained, and **any change to the feature set or to
the encoder requires retraining `ngec/assets/xgb_model.json`**
(`setup/train_wiki_model/train_wiki_model.py`). The default encoder stays jina
for exactly this reason: the shipped ranker was fit on jina features, and
swapping the encoder underneath it changes the meaning of four of its columns.

## Bugs found by the first full corpus run

The contract analysis above was done by reading the code and exercising the
post-processing paths. Actually running 500 Voice of America news stories
end-to-end (see `RUNNING.md`) surfaced four more problems that only appear with
real models, real Elasticsearch, and text that isn't the README example.

**1. The geoparser call had drifted from mordecai3's API.** `GeolocationModel`
passed `event_geoparse=False`, which the installed `Geoparser` no longer
accepts, so `PloverCoder(...)` raised `TypeError` before processing anything.
`mordecai3` is installed from git, so its signature moves independently of this
repo; `[tool.uv.sources]` now pins a specific rev. The same constructor also
defaulted `base_path` to `"NGEC/assets/"` — a relative path with the wrong case
that only resolves on a case-insensitive filesystem — and is now loaded through
`importlib.resources` like every other asset. It now also accepts the pipeline's
spaCy model and ES client rather than loading a second `en_core_web_trf` and
assuming ES is on localhost.

**2. The wiki ranker asked for a feature the pipeline never built.** Actor
resolution died with `KeyError: "['text_is_empty'] not in index"` partway
through the corpus. The XGBoost model in `assets/xgb_model.json` was retrained
with a `text_is_empty` feature that `setup/train_wiki_model/` adds *after*
`_create_scoring_dataframe` returns, so it existed at training time and not at
inference time. `_create_scoring_dataframe` now builds it the same way.

This is the failure mode to watch whenever the ranker is retrained: the feature
list lives in the model file, and nothing checks it against what inference
produces. Before the fix `tests/substantive/test_actor_resolution.py` was 122
passing / 253 failing; after, 245 passing / 131 failing.

Note this bug was invisible to the README example, because "Protesters" and "the
government" resolve through agent-pattern matching without ever calling the wiki
ranker. It needs a named entity to trigger.

**3. `explode_events` required an `id` the documented minimal input lacks.**
`AttributeModelInput` declares only `event_text` and `event_type` as required,
and `examples/` uses the attribute model standalone, but `explode_events` did
`f"{event['id']}_{idx}"` unconditionally. In the full pipeline
`stories_to_events` always sets an `id`, so this only bit outside the pipeline —
including in two tests that were failing on `main`. The suffix is now only added
when there is an id to suffix.

**4. The commonest date outcome reported a misleading reason.** When the text
states no date the attribute model returns `"N/A"`, which `_NOISE_PHRASES`
already refused to parse — but the fall-through then labelled it
`<dateparser failed to convert relative date, using pub date>`. Nothing had
failed; there was simply no date. That was 11 of 13 events in the corpus run, so
the misleading string was the *typical* case rather than an edge case. It now
reports `<No date given in the text ('N/A'), using pub date>`. The resolved
value and `date_type="unresolved"` are unchanged.

### Environment drift is a pipeline failure mode

Two of the run's blockers had nothing to do with NGEC's own code: `typer 0.27`
(released the day before the run) dropped its `click` dependency, which spaCy
imports but does not declare, so `import spacy` failed on a fresh `uv sync`; and
`vllm`'s unconstrained `xgrammar` requirement resolved to a version with no
wheels for the current Python, so `uv sync --extra vllm` could not install at
all. Both are now bounded in `pyproject.toml`.

Since `uv.lock` is gitignored, those bounds are the only record of a working
environment — a fresh install resolves to whatever is newest that day. Every
dependency now carries an upper bound for that reason.

## What would still make it more robust

1. **Reconcile or delete the legacy uppercase format** (`find_event_loc`,
   `add_meta`, the big docstring in `formatter.py`). Keeping dead code in the old
   `ACTOR`/`LOC`/`RECIP` + `qa_start_char` contract is a reviewer trap.
2. **Finish the two-story / empty-extraction end-to-end test.**
   `test_end_to_end_with_two_stories` is currently `pytest.skip("Haven't
   finished implementation")`. A case where one document yields no event is
   exactly what regressed before; it should be a standing test.
3. **Environment reproducibility.** There is no committed lockfile/venv and the
   full stack (vLLM/torch, spaCy `en_core_web_trf`, mordecai3, a running
   Elasticsearch with wiki+geonames) is heavy. A reviewer-oriented "minimal run"
   path — or a documented way to run steps 1/3/4/6 without ES — would lower the
   barrier that the reviewer hit. `RUNNING.md` now documents the working recipe
   and the CUDA/driver trap, but the upper bounds in `pyproject.toml` are doing
   a lockfile's job.
4. **Check the wiki ranker's feature list at load time.** `_call_ranker`
   indexes the scoring frame with `model.feature_names_in_`, so a retrained
   model that expects a new feature fails deep in a corpus run with a bare
   pandas `KeyError`. Comparing the two sets when the model is loaded would turn
   that into an immediate, legible error.
5. **Assert `anchor_quote` appears in `event_text`.** The attribute model
   sometimes returns the codebook definition of the event type instead of a
   quote from the document (twice in 13 events on the corpus run). It's a cheap
   check and a good signal of prompt leakage.

## How the crash findings were reproduced

The bugs were pure-Python post-processing logic and needed no models. A
standalone reproduction of the exact code paths (from `attribute_model.process`,
`actor_resolution.process`, and `formatter.process`) confirmed, **before** the fix:

- empty extraction `[]` → `IndexError` in the attribute model
- omitted `recipient` key → `KeyError` in the actor resolver
- missing `attributes` → `AttributeError` in the formatter
- two-event extraction → second event silently dropped

and **after** the fix: all four cases pass. A two-event extraction now yields two
separate event records, an empty extraction is dropped (and reported) instead of
crashing, and `explode_events` is covered directly by unit tests in
`tests/test_utilities.py` (single event, multiple events, empty→dropped, missing
`attributes` key, and no-mutation-of-input).
