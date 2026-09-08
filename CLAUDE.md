# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

**NGEC** (Next Generation Event Coder) turns news text into structured political
event records using the **PLOVER** ontology. It's the pipeline behind the POLECAT
dataset. This repo is a pre-release / research codebase; a journal article
describing it is in revise-and-resubmit, so correctness of the described pipeline
and reproducibility for reviewers matter.

## The pipeline (read this first)

The maintained end-to-end path is `PloverCoder.process()` in
`ngec/plover_coder.py`. It runs six steps, each a class/function with a
`.process()` method that **mutates and passes along a list of dicts**:

1. **Event classification** — `ngec/classifiers/plover_sklearn.py`
   `PloverSklearnClassifier` → adds `event_type` (list), `event_type_confidence`
   (dict), `event_mode` (list). Models live in `ngec/assets/event_models_v2/`
   and are **self-describing**: the encoder name and the per-class thresholds come
   from that directory's `metadata.json`, not from defaults. Do not pass
   `threshold=` or `encoder_name=` unless you mean to override every class at
   once. Features are built by `ngec/classifiers/features.py`, which the training
   scripts import too so the two cannot drift. See `CLASSIFIERS.md`.
2. **Geolocation** — `ngec/geolocation.py` `GeolocationModel` (wraps mordecai3) →
   adds `geolocated_ents`.
3. **Story → events split** — `ngec/utilities.py` `stories_to_events` → one event
   per **event type-mode pair** (an event type with no mode gives one record with
   `event_mode=""`; a type detected under three modes gives three records);
   collapses `event_type`/`event_mode` from lists to strings; adds
   `story_people/organizations/places`, `_doc_position`, `orig_id`.
   The per-mode multiplication is easy to mistake for a bug downstream: those
   records go to the attribute model separately and, when the modes describe the
   same underlying event, come back with identical attributes. `event_mode` is
   then the only field distinguishing them, so anything rendering or counting
   records needs to carry it.
4. **Attribute extraction** — `ngec/attribute_model.py` `AttributeModel` (an LLM,
   `ahalt/qwen3-event-extraction-exp5.1` by default, via
   vllm/transformers/mlx/llamacpp). The model is selectable with `model_name=`
   or `NGEC_ATTRIBUTE_MODEL`. The default is the 2026 retraining (trained from
   `~/projects/train_NGEC_2026`; see `setup/hf_release/` for how it was
   published). The original `ahalt/event-attribute-extractor` is older and
   worse but stays resolvable by name for a baseline comparison.
   **A model and its prompt format go together**: see `KNOWN_PROMPT_FORMATS` in
   that file, and "Swapping the attribute model" in `demo/DESIGN.md`. Prompting
   a model in the wrong format does not raise; it quietly extracts worse. The
   model may
   find zero, one, or several events per document; `explode_events`
   (`ngec/utilities.py`) then makes each its own record with a single
   `attributes` dict (dropping records with no extraction). Returns a **new,
   possibly-longer** list — use the return value.
5. **Actor resolution** — `ngec/actors/actor_resolution.py` `ActorResolver`
   (Wikipedia + agent patterns via Elasticsearch) → adds top-level `actor` /
   `recipient` (lists of coded-actor dicts).
6. **Formatting** — `ngec/formatter.py` `Formatter` → adds top-level
   `event_location` and `date_resolved`; can write `events_processed.jsonl`.
   Date resolution runs a fail-safe cascade (`_resolve_core`) and reports
   `resolved_date` / `date_end` / `granularity` / `date_type` / `reason`; see
   the "Date resolution" section of `PIPELINE.md` before changing it.

`ngec_process.py` at the repo root is an **older** CLI entry point that predates
`PloverCoder`; prefer `PloverCoder` / the end-to-end test as the source of truth.

**See `PIPELINE.md` for the full step-by-step data-contract analysis, the known
bugs, and the reasoning behind them.** Keep it in sync when you change any step's
input/output shape.

## The `attributes` contract (important)

**One record = one event.** If the attribute LLM finds several events in a
document, `explode_events` splits them into separate records; if it finds none,
the record is dropped (reported via a WARNING + a `*_dropped_events.jsonl` file).
So by the time actor resolution and formatting run, `event['attributes']` is a
single **dict**:

```python
{
  "event_type": "PROTEST", "anchor_quote": "...",   # event_type here is the LLM's
  "actor": ["Protesters"], "recipient": ["the government"],   # echo; the record's
  "date": ["today"], "location": ["Paris"],                   # top-level one is authoritative
}
```

Resolved outputs live at the **top level** of the record, not inside
`attributes`: `event["actor"]` / `event["recipient"]` (coded, step 5),
`event["event_location"]` / `event["date_resolved"]` (step 6). Read attribute
keys defensively with `.get(...)` — the model may omit a key.

Event `id`s get a `_<index>` suffix per exploded event (e.g. `story1_PROTEST__0`)
and are **per-run artifacts** (temperature 0.5); `orig_id` is the stable key.

This seam used to be the pipeline's main failure point (an `attributes[0]` that
crashed with `IndexError` on empty extractions, silently dropped multi-event
extractions, and turned a list into a dict). The explode-based rework above
replaced it — see `PIPELINE.md` for the full history and rationale.

Still worth cleaning up: `formatter.py`'s `find_event_loc` / `add_meta` are
**commented-out dead code** using an older uppercase span-based format
(`ACTOR`/`LOC`/`RECIP` with `qa_start_char`) — don't take them as the current
contract; reconcile or delete them.

## Environment / running

- **Python ≥ 3.10** (uses `X | Y` types and `match`). The base conda env here is
  3.9 and will not import the package.
- Managed with **`uv`**. There is no committed venv. Installing means choosing
  **exactly one of the `cpu` / `cu12` / `cu13` extras**, which redirect PyTorch
  to the matching [PyTorch index](https://download.pytorch.org/whl) via
  `[tool.uv.sources]`; without one, uv installs the default PyPI (CUDA 13) build
  and an older driver falls back to the CPU silently.
  `uv sync --extra models --extra cu12 --extra vllm --group dev` is the
  reference GPU install; **`python3 install.py --dev`** does the same after
  detecting the driver. `[tool.uv] conflicts` makes the three mutually
  exclusive and forbids `cu13 + vllm`, because the vllm extra is pinned `<0.20`
  (0.20.0+ are CUDA 13 builds) and must run against a CUDA 12 PyTorch. See §4 of
  `RUNNING.md` for the whole story and `DEVELOPING.md` for the macOS/`mlx` path.
- **A stale system CUDA on `LD_LIBRARY_PATH` shadows the PyTorch wheels' own
  libraries** (`undefined symbol: __nvJitLinkGetErrorLogSize_12_9`). On the
  reference machine, run with `env -u LD_LIBRARY_PATH`.
- spaCy models `en_core_web_lg` and `en_core_web_trf` are required (checked on
  import; a warning is emitted if missing).
- Backends for the attribute LLM: `vllm` (default, Linux/CUDA), `transformers`
  (slow, portable, used in tests), `mlx` (macOS).
- Steps 5 (actor resolution) and the end-to-end test require a running
  **Elasticsearch** with wiki + geonames indices. Put ES credentials in a `.env`
  file. `ngec/es_client.py` / `setup_es_client` handle the connection.

## Testing

```shell
uv run pytest                 # default: fast tests, substantive ones skipped
uv run pytest -m substantive  # only the slow "substantive" correctness tests (300+)
uv run pytest -m ""           # everything
```

- `tests/test_end_to_end.py` is the canonical pipeline test (needs ES).
  `test_end_to_end_with_two_stories` is currently **skipped** ("Haven't finished
  implementation") — finishing it would catch multi-event and empty-extraction
  regressions.
- Pure-logic bugs on the `attributes` seam can be reproduced with no models by
  exercising the post-processing code paths directly (see the reproduction
  section in `PIPELINE.md`).

## Conventions

- Each pipeline component exposes `__init__(...)` + `process(list_of_dicts)` and
  returns the list. To add/replace a component (e.g. a custom event classifier),
  copy the reference implementation and keep that interface —
  `plover_sklearn.py`'s module docstring documents the classifier contract.
- Components take `save_intermediate=True` to dump per-step JSONL for debugging.
- Assets (codebook CSV, country list, demo models) live in `ngec/assets/` and are
  loaded via `importlib.resources` (`resources.files("ngec")...`), not hard-coded
  paths.
- This repo accompanies an (under review/R&R) journal article. Because of this,
  code should be easily readable to social scientists (avoid ostenatious CS-type code)
  and replicability is absolutely critical.

## Repo layout

- `ngec/` — package. `actors/` (resolution, wiki_matcher, agent_matcher,
  common), `classifiers/` (reference event classifier + `features.py`, the
  shared feature construction), `assets/` (data/models), `scripts/`.
- `setup/` — training scripts for the classifiers, wiki model, and ES loading.
  `train_classifiers/codebook_llm/` builds the current demo classifiers;
  `train_classifiers/` itself holds the older synthetic-text approach they
  replaced.
- `tests/` — unit + `substantive/` (correctness, opt-in) + `actors/` +
  `classifiers/`.
- `examples/` — small demo apps (attribute model, plover, wiki resolution).
- `PIPELINE.md` — data-contract analysis and bug writeup (R&R deliverable).
- `RUNNING.md` — what happens when you point the pipeline at a corpus.
- `CLASSIFIERS.md` — where the event/mode classifiers came from, what is still
  wrong with them, and where the training data lives (mostly outside the repo).
