# Using one step on its own

Every step can be used without the rest of the pipeline. `ngec guide setup`
lists what each needs; only Wikipedia linking and geoparsing need
Elasticsearch. Load a model once and reuse it: creating one takes seconds to a
minute. After that, classifying a document, resolving a date or coding a short
description is fast (well under a second). Wikipedia linking takes several
seconds per name on a CPU, so 500 names is an hour or more. Attribute
extraction depends on the backend (`ngec guide setup`).

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

# The default backend ("auto") picks vLLM, MLX or llama.cpp for this machine;
# see `ngec guide setup`.
attribute_model = AttributeModel()
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
(`resolved_date` is None), even for absolute dates. Other formats are risky:
`03/04/2024` is read month first, and `20240315` is not understood at all (it
counts as a missing date). `granularity` says how precise the result is
(`day`, `week`, `month`, ...), and `date_type` whether it is `exact`,
`approximate`, a `range` (then `date_end` is set) or `unresolved`. No models
or Elasticsearch are needed.

An empty or unreadable phrase returns the publication date itself, with
`date_type` = `unresolved`. Filter on `date_type`, not on whether
`resolved_date` is None.

## Coding a short actor description (no Elasticsearch)

```python
from ngec.actors.agent_matcher import AgentMatcher

matcher = AgentMatcher(device="cpu")
matcher.short_text_to_agent("Latvian air force")
# {'country': 'LVA', 'code_1': 'MIL', 'code_2': '', 'pattern': 'air force', 'conf': 1.0, ...}
```

`conf` is the cosine similarity between the text and the closest pattern in
the agents file; below 0.625 (the default `threshold=`) the call returns None.
The country comes only from words in the text ("Latvian", "(Kenya)"). A named
group with no country word in it ("Rapid Support Forces") gets a category but
no country; Wikipedia linking (below) is what supplies countries for names.
The table in `ngec guide run` says what the codes mean. Pass
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
date. `code["actor_wiki_job"]` is the office the code came from (e.g.
"Minister of Economics and Finance" for Emmanuel Macron in 2015).
`actor_to_code` does its own Wikipedia search, so there is no need to call
`query_wiki` as well unless the article itself is wanted.

Give the sentence or story as `context`. Without one, build a short one from
what the user has ("Angela Merkel, a politician from Germany, in 2003."), and
pass the country name to `query_wiki(..., country="Germany")`. A bare surname
is often linked to the wrong article ("Bush" to the plant) and can still come
back with a confident code from the words alone; `code["wiki"] == ""` means no
article was linked, and those rows deserve a second look.

The article dict has `title`, `short_desc`, `intro_para` and `infobox`, and
many ranking features that can be ignored. For every office a person held on a
date, not only the one used for the code:

```python
offices = resolver.wiki_parser.parse_offices(article["infobox"])
current, countries = resolver.wiki_parser.get_current_office(offices, "2003-06-30")
[o["office"] for o in current]
# ['Leader of the Christian Democratic Union', 'Leader of the Opposition', ...]
```

Offices come from the infobox's `office` fields, so people whose infobox has
none get no office. Members of the US Congress are the common case: their
infoboxes give a state and district instead, and the code then falls back to
the article's short description.

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
