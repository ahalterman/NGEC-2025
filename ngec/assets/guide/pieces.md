# Using one step on its own

Every step can be used without the rest of the pipeline. `ngec guide setup`
lists what each needs; only Wikipedia linking and geoparsing need
Elasticsearch. Load a model once and reuse it: creating one takes seconds to a
minute, using it takes milliseconds per item.

## Event types for a document

```python
from ngec.classifiers.plover_sklearn import PloverSklearnClassifier

classifier = PloverSklearnClassifier()
event_types, modes = classifier.classify_one(
    "Thousands of teachers marched through Nairobi demanding higher pay.")
# (['PROTEST'], ['PROTEST-demo'])
```

For many documents, `classifier.process(stories)` takes the story dicts
(`event_text` is enough) and adds `event_type` (list), `event_mode` (list of
`"TYPE-mode"`) and `event_type_confidence` (dict) to each.

## Actor, recipient, date and location spans

The attribute model needs the text and the event type to look for. It reads
the event type's definition, so this works for any PLOVER type, or for your
own types given a definition (`ngec guide customize`).

```python
from ngec import AttributeModel

# backend="vllm", gpu=True on Linux with an NVIDIA GPU; "mlx" on a Mac.
attribute_model = AttributeModel(backend="transformers")
records = [{"id": "doc1",
            "event_text": "Police fired tear gas at protesters in Harare on Monday.",
            "event_type": "PROTEST",
            "event_mode": ""}]        # a PLOVER mode, or "" for the type as a whole
events = attribute_model.process(records)
events[0]["attributes"]
# {'event_type': 'PROTEST', 'mode': 'riot',
#  'anchor_quote': 'Police fired tear gas at protesters in Harare on Monday',
#  'actor': ['protesters'], 'recipient': [], 'date': ['on Monday'],
#  'location': ['in Harare'], 'killed': [], 'injured': []}
```

`process` returns a **new** list: one record per event found, so a record can
become several (two protests in one story) or none (the model found no such
event; those are dropped). Ids get a `_0`, `_1`, ... suffix. Spans are copied
verbatim and can keep a leading preposition ("in Harare"); the pipeline's
geocoding and date steps allow for that.

## Resolving a date phrase

```python
from ngec import resolve_date_text

resolve_date_text("last Wednesday", "2025-05-16")
# {'resolved_date': datetime(2025, 5, 14), 'date_end': None,
#  'granularity': 'day', 'date_type': 'exact', 'reason': '...'}
```

The second argument is the publication date the phrase is relative to, as
`YYYY-MM-DD` or a date object. Without it nothing is resolved
(`resolved_date` is None), even for absolute dates. `granularity` says how
precise the result is (`day`, `week`, `month`, ...), and `date_type` whether
it is `exact`, `approximate`, a `range` (then `date_end` is set) or
`unresolved`. No models or Elasticsearch are needed.

## Coding a short actor description (no Elasticsearch)

```python
from ngec.actors.agent_matcher import AgentMatcher

matcher = AgentMatcher(device="cpu")
matcher.short_text_to_agent("Latvian air force")
# {'country': 'LVA', 'code_1': 'MIL', 'code_2': '', 'pattern': 'air force', 'conf': 1.0, ...}
```

Returns None when nothing in the agents file is close enough. Pass
`AgentMatcher(agents_file="my_agents.txt")` to use your own categories
(`ngec guide customize`).

## Linking a name to Wikipedia, and coding an actor

Needs the `wiki` index. `ActorResolver` loads everything the actor step uses.

```python
from ngec import ActorResolver
from ngec.es_client import es_client_from_env

es_client = es_client_from_env()     # reads ES_HOST etc. from .env, else localhost:9200
resolver = ActorResolver(es_client=es_client, device="cpu")

# Which article does a name refer to? The context disambiguates.
article = resolver.wiki_matcher.query_wiki(
    "Blinken", context="US Secretary of State Antony Blinken arrived in Kyiv.")
article["title"]          # 'Antony Blinken'; article is None if nothing matched

# Country and actor codes, from the article or the words themselves.
code = resolver.actor_to_code("Blinken",
                              context="US Secretary of State Antony Blinken arrived in Kyiv.",
                              query_date="2023-06-01")
code["country"], code["code_1"], code["wiki"]     # ('USA', 'GOV', 'Antony Blinken')
```

`query_date` matters: codes for people come from the offices they held on that
date (Rishi Sunak is `GOV` in 2023, not in 2012). Pass the story's publication
date. Give the sentence or story as `context`; a bare surname with no context
("Bush") is often linked to the wrong article. The article dict also has
`short_desc`, `intro_para` and `infobox`, and many ranking features that can be
ignored.

`examples/demo_wiki_resolution.py` in the repository is a longer worked
example, including a failure.

## Finding and geocoding place names

Needs the `geonames` index. NGEC uses the mordecai3 geoparser, which can be
called directly:

```python
from mordecai3 import Geoparser
from ngec.es_client import es_client_from_env

geo = Geoparser(es_client=es_client_from_env())
result = geo.geoparse_doc("Troops were sent to the southern state of Guerrero on Tuesday.")
for place in result["geolocated_ents"]:
    print(place["search_name"], place["country_code3"], place["lat"], place["lon"])
```

Each place has a GeoNames id, feature code (`PPL` a town, `ADM1` a state or
province, ...) and a `score`; low scores are uncertain.

## Several steps, but not all

Each component's `process()` takes and returns the list of dicts shown in
`ngec guide`, so steps chain. `PloverCoder.process` in
`ngec/plover_coder.py` is ten lines and is the reference for the order and for
what each step needs from the ones before it (for example, step 3,
`stories_to_events`, needs the spaCy docs from `ngec.utilities.load_nlp()`).
