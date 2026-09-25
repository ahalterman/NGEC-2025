# Customizing NGEC

NGEC is built so a researcher can change one step and keep the others. Find
out first what the user actually needs; most projects need only one of these.

| The user wants | Step affected | What it takes |
|---|---|---|
| Different or narrower event definitions | attribute extraction | write definitions; no training |
| New event types | event classification (+ definitions) | labeled examples and a classifier, or their own way of picking documents |
| Their own actor categories | actor resolution | an agents file and a priorities file; no training |
| PLOVER codes mapped onto a coarser scheme | none | a lookup table applied to the output |
| Extra attributes (e.g. number of participants) | attribute extraction | new training data and retraining; see below |
| Non-English text | every step | not supported; translating first is an open question |

Always finish with a validation step (last section). A customized pipeline
that has not been checked against hand-coded examples cannot be trusted.

## Writing event definitions

The attribute model reads the event definition in its prompt, so it can
extract attributes for event types it was never trained on, given a good
definition. The paper's validation on the ECAV election-violence dataset used
this, with no retraining.

The default definitions, in the format the default model was trained on, are
`ngec/assets/event_definitions_v6.json`: a JSON list of entries like

```json
{"event_type": "ELECTORAL_VIOLENCE",
 "mode": "",
 "definition": "## Event: **ELECTORAL_VIOLENCE**: <what the event is>\n## Potential Ambiguities: <what it is easily confused with>\n## Attribute Guidance: ACTOR is <...>. RECIPIENT is <...>."}
```

The `definition` is a series of `##` sections, one per line, in this order.
These are the sections the default model was trained with (they are identical
to the `definitions.json` published with the model on Hugging Face):

| Section | In the shipped definitions | What goes in it |
|---|---|---|
| `## Event: **TYPE**:` | always | what the event is, with examples |
| `## Specific Sub-Event: **mode**:` | entries for a mode | what the sub-type is |
| `## Sub-Events:` | a type's own entry, when it has modes | the list of its modes |
| `## Special Instructions:` | about two thirds | coding rules, e.g. who counts as the actor when a spokesperson speaks |
| `## Potential Ambiguities:` | about two thirds | neighbouring event types it should not be confused with |
| `## Attribute Guidance:` | always, last | who the ACTOR and RECIPIENT are, and when to leave one empty |

`mode` is `""` for the event type as a whole. Read a few of the shipped
entries before writing new ones and copy their shape. To see the file:

```python
from importlib import resources
print(resources.files("ngec").joinpath("assets", "event_definitions_v6.json").read_text()[:3000])
```

Put the new entries in a JSON file of your own and pass it as
`event_definitions_file` (to `PloverCoder` or `AttributeModel`). Its entries
are added to the shipped ones, and replace any with the same `event_type` and
`mode`, so the file needs only the types you add or reword. Every event type the
classifier can emit needs a definition, or the attribute step raises an error
naming the type.

The Attribute Guidance matters most: every shipped definition ends with one.
Say explicitly who the actor and the recipient are, and when the recipient
should be left empty. PROTEST's reads, in part: "ACTOR is the protesting
group. RECIPIENT is the entity the protest is directed against, and ONLY when
the text states it as the target."

Rewording a PLOVER type's definition gives the model a prompt it was not
trained on. That is fine and often the point, but check the output on a
sample afterwards.

## New event types

Event classification is a document classifier per event type: "does this
story report an X event?" A new event type needs a way to answer that question
for each story. Options, from least to most work:

1. **The user already knows the event type** of each document (they selected
   documents by keyword or from a hand-coded dataset). Set `event_type` on each
   story and skip the classifier.
2. **Their own classifier** or labeling method (a fine-tuned model, a
   zero-shot LLM, keyword rules). Any object with a
   `process(list_of_stories)` method that sets `event_type` (a list of type
   names) and `event_mode` (a list of `"TYPE-mode"` strings, possibly empty) on
   each story fits in step 1. The contract is in the module docstring of
   `ngec/classifiers/plover_sklearn.py`.
3. **Train a classifier from labeled stories** (a few hundred per type, with
   both positives and negatives). See "Training a classifier" below.

For options 1 and 2, a small class does it:

```python
class KnownEventTypes:
    """Stories already selected as protests, e.g. by a keyword search."""
    def process(self, story_list):
        for story in story_list:
            story["event_type"] = ["PROTEST"]
            story["event_mode"] = []
            story["event_type_confidence"] = {"PROTEST": 1.0}
        return story_list

coder = PloverCoder(es_client=es_client,
                    event_classifier=KnownEventTypes(),
                    event_definitions_file="my_definitions.json")  # for non-PLOVER types
```

Whatever the classifier, check its output on a sample of the user's own
stories before using counts from it.

### Training a classifier

For one or a few event types from the user's own yes/no labels, fit a model on
the same features the shipped classifiers use and wrap it in a class like the
one above:

```python
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

from ngec.classifiers.features import combine_features, encode_documents

labeled = pd.read_csv("labeled.csv")          # columns: text, land_dispute (1 or 0)
texts = list(labeled["text"])
y = labeled["land_dispute"].to_numpy()

# The same features as the shipped classifiers: sentence embeddings of
# overlapping chunks, averaged, plus TF-IDF word features.
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5,
                             ngram_range=(1, 2), max_features=8000)
vectorizer.fit(texts)
X = combine_features(encode_documents(encoder, texts), vectorizer.transform(texts))

model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                           class_weight="balanced", max_iter=2000)

# Pick the threshold on held-out predictions, not on the training fit.
held_out = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
precision, recall, thresholds = precision_recall_curve(y, held_out)
f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
threshold = float(thresholds[np.argmax(f1[:-1])])
print(f"held-out F1 {f1[:-1].max():.2f} at threshold {threshold:.2f}")

model.fit(X, y)


class LandDisputeClassifier:
    """Step 1 of the pipeline, for one event type."""
    def process(self, story_list):
        texts = [story["event_text"] for story in story_list]
        X = combine_features(encode_documents(encoder, texts), vectorizer.transform(texts))
        probabilities = model.predict_proba(X)[:, 1]
        for story, p in zip(story_list, probabilities):
            story["event_type"] = ["LAND_DISPUTE"] if p >= threshold else []
            story["event_mode"] = []
            story["event_type_confidence"] = {"LAND_DISPUTE": float(p)}
        return story_list
```

Report the held-out F1, not a score on the training data. Save the fitted
objects (`skops.io.dump`, or `joblib`) so the coding run uses exactly the
model that was evaluated.

The shipped classifiers were built by `setup/train_classifiers/codebook_llm/`
in the repository: an LLM applied the PLOVER codebook to news articles, and
`train_classifiers.py` fitted one model per type and mode from those labels.
Adapt it to train many types and modes at once and write a model directory that
`PloverSklearnClassifier(type_model_dir=..., codebook_path=...)` loads. That
directory's `metadata.json` records the encoder (`"encoder"`) and each type's
threshold (`"metrics": {"TYPE": {"threshold": ...}}`); without them the
classifier falls back to `all-mpnet-base-v2` and a threshold of 0.5, without
an error. `CLASSIFIERS.md` in the repository explains how the shipped models
were made and what is still wrong with them.

## Your own actor categories

Actor categories come from a plain-text **agents file**: each line is a
description of a kind of actor and the code it gets. The default is
`ngec/assets/PLOVER_agents.txt` (about 2,400 lines). Categorization is by
meaning, not exact match. Each span (or, for an actor linked to Wikipedia, the
office or short description from its article) is compared to every pattern
with a sentence encoder, and the closest pattern above a threshold wins. So a
few natural phrasings per category go a long way.

The format:

```
# comments start with #
POLICE_OFFICER [~COP]
RIOT_POLICE [~COP]
OPPOSITION_ACTIVIST [~CVLOPP]
UNION_LEADER [~LAB]
```

- Underscores become spaces, and case does not matter.
- **Codes are split by position**: the first three characters are `code_1`
  and the rest `code_2`. `[~CVLOPP]` gives `CVL` + `OPP`. A code must
  therefore start with a three-letter block: `LABOR` would become `LAB` +
  `OR`.
- Countries are assigned separately (from names and adjectives in the span or
  the Wikipedia article), so do not put countries in the codes.

Every agents file needs a matching **priorities file**, in the format of
`ngec/assets/PLOVER_priorities.csv`. An actor often matches more than one
category: a minister who is also a party leader, say, or a person whose
Wikipedia infobox lists both a military and a government office. The
priorities file settles those cases. Each code gets a number, and the higher
number wins. For example:

```
code,priority,special
RUL,150,0
OPP,140,0
SEC,130,0
CVL,10,0
```

Three things to know about it:

- Only `code_1` (the first three characters) is ranked, so `COPLOC` and
  `COPNAT` tie. `code_2` never breaks a tie.
- A code left out of the file gets priority 0, so give every code your agents
  file uses a number above 0.
- `special` is almost always 0. A 1 marks a code that names a polity rather
  than a role (PLOVER's `IGO`, for instance): it is moved into the `country`
  field and `code_1` is left empty. Leave it at 0 unless that is what you
  want.

Pass both to the pipeline:

```python
coder = PloverCoder(es_client=es_client,
                    agents_file="my_agents.txt",
                    priorities_file="my_priorities.csv")
```

or, for the actor step alone, `ActorResolver(agents_file=...,
priorities_file=..., override_sources=())`. There, `override_sources=()` stops
Wikipedia's short description from always winning a disagreement with the
span text, which is usually what a custom scheme wants; `PloverCoder` does not
set it. To try patterns quickly without Elasticsearch,
`AgentMatcher(agents_file="my_agents.txt").short_text_to_agent("...")` codes
one description (`ngec guide pieces`).

**Cover officeholders.** For a person or organization linked to Wikipedia, it
is their infobox office and short description that get matched against your
patterns ("President of Zimbabwe", "military officer"), not the words in the
story. A scheme without patterns for heads of state, ministers and legislators
still codes those people, into whichever category is nearest: in a test with a
four-category election scheme and no pattern for presidents, Zimbabwe's
president came out as a security-forces actor. Add patterns for the offices the
user's actors hold, and check a few well-known names by hand.

Iterate: code a sample, look at the spans that got no code or the wrong code,
add patterns for them, repeat. The paper found this step the weakest when an
existing category scheme was reused as is (56% accuracy on ECAV's categories),
and recommends time spent on project-specific patterns here.

The encoded patterns are cached (keyed by the file's contents), so the first
run with a new or edited file takes longer.

## Mapping PLOVER codes onto a coarser scheme

If the user's categories are unions of PLOVER's (e.g. "state actor" = GOV,
MIL, COP, JUD, LEG), keep the default agents file and map `code_1` afterwards
with a dictionary in pandas. This is how the paper evaluated against ECAV. It
is the least work, but only works when no user category splits a PLOVER one.

## Extra attributes

The attribute model returns actor, recipient, date and location (and, for
ASSAULT, PROTEST and COERCE, killed and injured). A new attribute, such as
the number of participants, means generating new synthetic training data with
it and fine-tuning a new model; the training code is in a separate repository.
Before going down that road, check whether the attribute can be read off the
`anchor_quote` or the story text with a simpler method.

## Validating the result

Before the user reports anything from customized output:

1. Sample 50 to 100 stories at random (more for rare event types).
2. Have the user (or, better, two people independently) code them by hand
   with the same definitions: event types, and for each event its actor,
   recipient, date, location and actor categories.
3. Compare step by step: event-type precision and recall, span agreement
   (exact, and allowing boundary differences such as "students" vs. "four
   students"), and category accuracy. For event types, with the hand codes in
   a CSV with one row per story and a 0/1 column per type:

   ```python
   from sklearn.metrics import classification_report
   gold = pd.read_csv("hand_coded.csv")          # story_id, PROTEST, ASSAULT, ...
   found = table.groupby("story_id")["event_type"].apply(set)
   for event_type in ["PROTEST", "ASSAULT"]:
       predicted = [int(event_type in found.get(s, set())) for s in gold["story_id"]]
       print(event_type)
       print(classification_report(gold[event_type], predicted, digits=2))
   ```
4. Look at the disagreements, not only the scores. They show which step to fix
   (a definition, missing agent patterns, a classifier threshold).

Keep the hand-coded sample and the scores with the output. They are what a
reviewer will ask for.
