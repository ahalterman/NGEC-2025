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
`ngec/assets/event_definitions_v6.json`. Read a few before writing new ones,
and write new ones in the same shape:

```
## Event: **ELECTORAL_VIOLENCE**: <one or two sentences on what the event is>
## Specific Sub-Event: **<mode name>**: <what the sub-type is>      (optional)
## Special Instructions: <who counts as ACTOR and RECIPIENT for this type>
```

The Special Instructions matter most. Say explicitly who the actor and the
recipient are when that is not obvious (e.g. "The recipients are both the
targets and the victims of the attack").

<!-- TODO(coordinator): the per-record `event_def` hook and the PloverCoder
option for it, once the PloverCoder change lands. -->

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
3. **Train classifiers like the shipped ones**: sentence-embedding and TF-IDF
   features with one logistic regression per type. This needs a few hundred
   labeled stories per type. <!-- TODO(coordinator): point at the reference
   training script once it is in the repo. -->

Whatever the classifier, check its output on a sample of the user's own
stories before using counts from it.

<!-- TODO(coordinator): the PloverCoder option for supplying a classifier. -->

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

Every agents file needs a matching **priorities file** that ranks the codes
when several match, in the format of `ngec/assets/PLOVER_priorities.csv`
(`code,priority,special`).

Pass both to `ActorResolver(agents_file=..., priorities_file=...,
override_sources=())`. `override_sources=()` stops Wikipedia's short
description from always winning a disagreement with the span text, which is
usually what a custom scheme wants.

<!-- TODO(coordinator): the PloverCoder options for these. -->

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
   students"), and category accuracy.
4. Look at the disagreements, not only the scores. They show which step to fix
   (a definition, missing agent patterns, a classifier threshold).

Keep the hand-coded sample and the scores with the output. They are what a
reviewer will ask for.
