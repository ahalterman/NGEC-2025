# NGEC

*This is a temporary pre-release version of the code. It's not yet stable and I do not recommend you use it right now.*

## Installation

The recommended installation is with `uv`. 

### Install PyTorch

To make sure the correct version of PyTorch for your system is installed, install it manually first. 

### spacy models

ngec depends on the spacy `en_core_web_lg` and `en_core_web_trf` models, which are delivered as non-standard Python pacakges. 

To attempt to install them alongside the package, use the `models` extra:

```python3
uv add ngec[models]
```

### Inference backend

There are different options for the LLM inference backend. The most basic one, but also slowest is `"transformers"`, which is installed by default. 

For Windows and Linux users, especially with CUDA, install vLLM, which can be done via an extra:

```python3
uv add ngec[models,vllm]
```

macOS users can try to use `"mlx"` by installing the corresponding extra:

```python3
uv add ngec[models,mlx]
```

### Installing with pip

If you are not using `uv` to install `ngec`, the `mordecai3` install will not work correctly. In that case:

1. Manually install `mordecai3` from GitHub into whatever virtual environment you are using.
2. Install `ngec`. 

### Uninstalling - cache

`ngec` caches agent embeddings to improve speed. Those can be easily regenerated if needed. In any case, uninstalling the package will not delete those. They are located at OS-specific cache locations, determing using the [`platformdirs`](https://pypi.org/project/platformdirs/) package. See their documentation for [OS-specific cache folders](https://platformdirs.readthedocs.io/en/latest/platforms.html).


## Usage

NGEC includes a functioning demo PLOVER coder (it does require ES though):

```python
import logging
from pprint import pprint

from ngec.plover_coder import PloverCoder
from ngec.es_client import setup_es_client
from ngec.logging import setup_logging

# Quiet third-party logging
setup_logging(
    level=logging.DEBUG,
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    quiet_third_party=True
)

# Connect to ES
es_client = setup_es_client(hosts=["localhost"], port=9200)

pc = PloverCoder(es_client=es_client)

story_list = [
        {"id": "story1", "event_text": "Protesters were in the streets in Paris again today to protest against the government's austerity measures.", "pub_date": "2016-05-01"}
    ]
    
event_list = pc.process(story_list)

pprint(event_list, sort_dicts=False, width=100)
```

```
[{'id': 'story1_PROTEST_',
  'event_text': 'Protesters were in the streets in Paris again today to protest against the '
                "government's austerity measures.",
  'pub_date': '2016-05-01',
  'event_type': 'PROTEST',
  'event_type_confidence': {'PROTEST': 0.9315939265193662},
  'event_mode': '',
  'geolocated_ents': [{'feature_code': 'PPLC',
                       'feature_class': 'P',
                       'country_code3': 'FRA',
                       'lat': 48.85341,
                       'lon': 2.3488,
                       'admin1_code': '11',
                       'admin1_name': 'Île-de-France',
                       'admin2_code': '75',
                       'admin2_name': 'Paris',
                       'geonameid': '2988507',
                       'score': 1.0,
                       'search_name': 'Paris',
                       'start_char': 34,
                       'end_char': 39,
                       'city_id': '2988507',
                       'city_name': 'Paris',
                       'country_name': 'France',
                       'resolved_placename': 'Paris'}],
  'story_people': [],
  'story_organizations': [],
  'story_places': ['Paris'],
  '_doc_position': 0,
  'orig_id': 'story1',
  'attributes': {'event_type': 'PROTEST',
                 'anchor_quote': 'Protesters were in the streets in Paris again today to protest '
                                 'against the government’s austerity measures.',
                 'actor': ['Protesters'],
                 'recipient': ['the government'],
                 'date': ['today'],
                 'location': ['Paris']},
  'actor': [{'wiki': '',
             'actor_wiki_job': '',
             'all_code1s': [],
             'all_code2s': [],
             'country': '',
             'code_1': 'CVL',
             'code_2': 'OPP',
             'actor_role_query': 'Protesters',
             'actor_resolved_pattern': 'protesters',
             'actor_pattern_conf': 0.9811088938288606,
             'actor_resolution_reason': '',
             'description': 'protesters',
             'source': 'BERT matching full text',
             'best_reason': ''}],
  'recipient': [{'wiki': '',
                 'actor_wiki_job': '',
                 'all_code1s': [],
                 'all_code2s': [],
                 'country': '',
                 'code_1': 'GOV',
                 'code_2': '',
                 'actor_role_query': 'government',
                 'actor_resolved_pattern': 'government',
                 'actor_pattern_conf': 0.9999999999992808,
                 'actor_resolution_reason': '',
                 'description': 'government',
                 'source': 'BERT matching full text',
                 'best_reason': ''}],
  'event_location': {'event_loc': {'feature_code': 'PPLC',
                                   'feature_class': 'P',
                                   'country_code3': 'FRA',
                                   'lat': 48.85341,
                                   'lon': 2.3488,
                                   'admin1_code': '11',
                                   'admin1_name': 'Île-de-France',
                                   'admin2_code': '75',
                                   'admin2_name': 'Paris',
                                   'geonameid': '2988507',
                                   'score': 1.0,
                                   'search_name': 'Paris',
                                   'start_char': 34,
                                   'end_char': 39,
                                   'city_id': '2988507',
                                   'city_name': 'Paris',
                                   'country_name': 'France',
                                   'resolved_placename': 'Paris'},
                     'reason': 'success'},
  'date_resolved': {'resolved_date': datetime.datetime(2016, 5, 1, 0, 0),
                    'granularity': 'day',
                    'reason': '<Resolved relative date with past reference>'}}]
```


### Logging

Some of the third-party dependencies have very verbose loggers by default. To quiet those:

```python
from ngec.logging import quiet_third_party_loggers

quiet_third_party_loggers()
```

There is also a more general helper function included that can do this as well:

```python
import logging
from ngec.logging import setup_logging

setup_logging(
    level=logging.DEBUG,
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    quiet_third_party=True
)
```