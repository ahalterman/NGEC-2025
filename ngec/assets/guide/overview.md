# NGEC guide for coding agents

NGEC turns English news text into political event data: *who did what to
whom, where and when*. It is the research code behind the POLECAT dataset and
the paper "Creating Custom Event Data: A Bag-of-Tricks" (Halterman et al.).

The people using it are usually social scientists with some Python, not
software engineers. Write plain scripts they can read and rerun, explain what a
step will do and roughly how long it will take before starting anything slow,
and keep their results reproducible (see "Reproducibility" below).

More detail, one topic at a time:

| Command | What it covers |
|---|---|
| `ngec guide setup` | installing, the two doctors, Elasticsearch, GPUs |
| `ngec guide run` | coding a whole corpus with the pretrained models, reading the output |
| `ngec guide pieces` | using one step on its own (dates, Wikipedia linking, actor codes, ...) |
| `ngec guide customize` | new event types, your own actor categories, validating the result |

Prefix these with `uv run` in a uv project. Read the topic before writing code
for it: the snippets there match this installed version.

## The pipeline

`ngec.plover_coder.PloverCoder(...).process(stories)` runs six steps. Each is a
class with a `.process(list_of_dicts)` method, so any step can be run alone or
replaced.

| # | Step | Class | Adds to each record | Needs |
|---|---|---|---|---|
| 1 | Event classification | `ngec.classifiers.plover_sklearn.PloverSklearnClassifier` | `event_type`, `event_mode` | models only |
| 2 | Geoparsing | `ngec.GeolocationModel` (mordecai3) | `geolocated_ents` | Elasticsearch `geonames` |
| 3 | Story -> events | `ngec.utilities.stories_to_events` | one record per event type/mode | spaCy |
| 4 | Attribute extraction | `ngec.AttributeModel` (a small LLM) | `attributes`: actor, recipient, date, location spans | models only; GPU helps |
| 5 | Actor resolution | `ngec.ActorResolver` | `actor`, `recipient`: coded (country, code_1, code_2, Wikipedia page) | Elasticsearch `wiki` |
| 6 | Formatting | `ngec.Formatter` | `event_location`, `date_resolved` | nothing extra |

The paper describes the same pipeline as five conceptual steps: event
detection (1), attribute extraction (4), entity linking and entity
categorization (both inside 5), and date/location resolution (2 and 6).

Only steps 2 and 5 need Elasticsearch, which is the slowest part of an install
(a ~10 GB index). Someone who only wants event types, attribute spans, date
resolution or actor categories from short descriptions does not need it.

## Input

A list of dicts, one per news story:

```python
{"id": "story1",                      # unique, any string
 "event_text": "Protesters marched ...",  # the full story text
 "pub_date": "2016-05-01"}            # publication date, YYYY-MM-DD
```

`pub_date` matters more than it looks. Relative dates ("last Wednesday",
"yesterday") are resolved against it, and actors' Wikipedia offices are checked
against it. Without it dates are unresolved, and nothing raises an error. Use
`None` for a missing date, never a string made from a blank spreadsheet cell
(`ngec guide run` shows how to prepare the column).

## Output

`process()` returns one record **per event**, not per story: a story can yield
several event types, a type can be found under several modes, and the
attribute model can find several instances of one type. `orig_id` links a
record back to its story; `id` is a per-run label, not a stable key.

The fields most people want:

- `event_type`, `event_mode` (PLOVER categories, e.g. `PROTEST`, `demo`; the
  classifier on its own writes modes as `PROTEST-demo`)
- `attributes`: the text spans, a dict of lists, e.g.
  `{"actor": ["Protesters"], "recipient": ["the government"], "date": ["today"], "location": ["Paris"], "anchor_quote": "..."}`.
  For ASSAULT, PROTEST and COERCE the default model also returns `killed` and
  `injured` spans. Read keys with `.get()`: any may be missing.
- `actor`, `recipient`: lists of coded actors, each with `country` (ISO3),
  `code_1`, `code_2` (PLOVER actor codes such as `GOV`, `MIL`, `CVL`; the
  table in `ngec guide run` lists them), `wiki` (the Wikipedia title, if
  linked) and `actor_wiki_job` (the office held on the story's date, for a
  linked person).
- `event_location["event_loc"]`: the geocoded place (`lat`, `lon`,
  `country_code3`, `geonameid`, ...), or None with a `reason`.
- `date_resolved`: `resolved_date`, `granularity` (day/week/month/...),
  `date_type` (exact/approximate/range/unresolved) and `reason`.

`ngec.events_to_table(events)` flattens the records into a pandas DataFrame,
one row per event, for CSV, R or Stata.

## Things that go wrong quietly

- **The wrong PyTorch build** runs on the CPU with no error, many times slower.
  `ngec doctor` compares the GPU the driver reports with what PyTorch sees.
- **A missing `pub_date`**: see above.
- **Setting `event_threshold`** replaces the per-event-type thresholds stored
  with the classifier by one number for all types. Leave it unset unless the
  user has a reason.
- **Changing the attribute model** without its prompt format: a model and its
  prompt format go together (`KNOWN_PROMPT_FORMATS` in
  `ngec/attribute_model.py`). A mismatch does not raise; extraction just gets
  worse.
- **Records per event, not per story**: counting rows counts events. When
  modes describe the same event, records can differ only in `event_mode`.

## When something breaks

Run `ngec doctor` first. It checks the install, every setting NGEC reads, the
PyTorch build and Elasticsearch, and prints the command that fixes each
problem. `ngec doctor --smoke` also runs three stories through the whole
pipeline. `--json` is easier for you to read. Fix problems in the order it
lists them.

## Reproducibility

Event data built with NGEC ends up in papers. Help the user record:

- the NGEC version and commit, and the settings in use (`ngec doctor --json`
  prints both),
- the attribute model name (`AttributeModel(model_name=...)`, or the
  `NGEC_ATTRIBUTE_MODEL` setting),
- any custom files (definitions, agents file) alongside the output,
- the date of the Wikipedia and GeoNames indices, since actor linking depends
  on them (`ngec update` reports the installed versions; don't update in the
  middle of a project without telling the user),
- the attribute model's backend: llama.cpp runs a quantized copy of the model,
  and its output differs slightly from vLLM's or transformers' on a few
  documents.

The default attribute model, `ahalt/qwen3.5-event-extraction-0.8b`, decodes
greedily, so reruns on one machine give the same spans. The paper's model is
`ahalt/qwen3-event-extraction-exp5.1`.

## Limits

English news text only. NGEC does not merge duplicate reports of the same event
from different sources. The pretrained event classifiers are demonstration
models trained on a small labeled set; check their output on the user's own
text before trusting counts (see `ngec guide customize`).
