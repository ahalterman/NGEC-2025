# NGEC

*This is a pre-release version of the code. Expect instablity and errors when running it.*

See [`docs/RUNNING.md`](docs/RUNNING.md) for notes on running the pipeline over a real corpus. That doc talks about which Elasticsearch indices you need, time estimates, and common errors.

Note that NGEC depends on ElasticSearch indices derived from Wikipedia and GeoNames data for actor resolution and geocoding. The data itself is quite big, 10+GB, and requires running an ElasticSearch instance. See the install instructions below.

## Installation

The recommended installation is with `uv`, and also using `uv` for virtual environment/dependency management.

### (optional) Install PyTorch

Installing ngec if PyTorch is not already installed will install whatever PyTorch version is the default for your platform. For performance reasons, you might want to change that, see https://pytorch.org/get-started/locally/ and [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

If you are on Windows with a NVIDIA GPU, this is something you should probably do, because the default on Windows is for a CPU version that will be slower.

For macOS users, you should be ok with the default.


### Install mordecai3

Geocoding depends on [mordecai3](https://github.com/ahalterman/mordecai3). As both mordecai3 and ngec are in active development right now, we recommend installing it from GitHub:

```shell
uv add "mordecai3 @ git+https://github.com/ahalterman/mordecai3"
```

<details>
<summary>Install with pip</summary>

\`\`\`shell
pip install "mordecai3 @ git+https://github.com/ahalterman/mordecai3"
\`\`\`

</details>

### Setup Elasticsearch and the Wiki/GeoNames indices

NGEC needs two Elasticsearch indices: `wiki` (actor resolution) and `geonames` (geocoding). Both live in a single data directory served by one Elasticsearch node. The easiest setup is to download the prebuilt data directory and run Elasticsearch over it in Docker. Expect >10 GB on disk.

**1. Install Docker.** See https://www.docker.com/get-started/.

**2. Download and unpack the prebuilt index.**

```shell
curl -LO https://andrewhalterman.com/files/geonames_wiki_index_2023-03-02.tar.gz
tar -xzf geonames_wiki_index_2023-03-02.tar.gz
```

This unpacks into a directory called `geonames_index`. It actually holds both the wiki and geo indices. Rename it to make clear it holds both indices:

```shell
mv geonames_index wikigeo_index
```

> **Note:** This prebuilt index predates a later refactor that added extra metadata fields to the `wiki` index, so it will not have those fields. The pipeline should still run, but anything that depends on the newer metadata will not be populated. If you need the current fields, build the wiki index yourself — see [`elasticsearch/README.md`](elasticsearch/README.md).

Note the **absolute** path of the renamed directory for the next step.

**3. Start Elasticsearch against it.**

```shell
docker run -d --name ngec-es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -v /absolute/path/to/wikigeo_index:/usr/share/elasticsearch/data \
  elasticsearch:7.10.1
```

**4. Check that both indices are there.**

```shell
curl -s 'localhost:9200/_cat/indices?v'
```

You should see `wiki` and `geonames`, both with a non-zero `docs.count`. If the list is empty, the volume path in step 3 is wrong — Elasticsearch silently starts with an empty data directory rather than failing. If health is `yellow`, see [Cluster health is yellow](elasticsearch/README.md#cluster-health-is-yellow).

That is all the setup NGEC needs: it connects to `localhost:9200` by default. To use a different host or port, pass them to `ngec.es_client.setup_es_client`.

If you need to build or refresh the indices yourself — a newer Wikipedia dump, a different gazetteer — see [`elasticsearch/README.md`](elasticsearch/README.md). Building the wiki index takes about a day.

### Install ngec

```shell
uv add "ngec @ git+https://github.com/ahalterman/ngec-2025"
```

<details>
<summary>Install with pip</summary>

\`\`\`shell
pip install "ngec @ git+https://github.com/ahalterman/ngec-2025"
\`\`\`

</details>


### spacy models

ngec depends on the spacy `en_core_web_lg` and `en_core_web_trf` models, which are delivered as non-standard Python packages: they are hosted on GitHub rather than PyPI, so installing `ngec` cannot bring them in. Download them once, after installing:

```shell
uv run ngec download-models
```

That is spaCy's own downloader, so `python -m spacy download en_core_web_lg` (and `en_core_web_trf`) installs exactly the same thing if you would rather do it by hand. Together they are about 900 MB. Nothing complains about a missing model until something tries to load it, at which point the error names the model and the command above.

Working from a clone of this repository rather than an install, you get them already: they are the `models` dependency group, which `uv sync` installs by default. See [`DEVELOPING.md`](DEVELOPING.md).

<details>
<summary>If you are using a virtual environment without `uv`</summary>

With the venv active:

\`\`\`shell
ngec download-models
\`\`\`

</details>


### Inference backend

There are different options for the LLM inference backend. The most basic one, but also slowest is `"transformers"`, which is installed by default.

For Windows and Linux users, especially with CUDA, install vLLM, which can be done via an extra:

```python3
uv add ngec[vllm]
```

Note that the currently pinned vLLM is a CUDA 13 build, so it needs a recent NVIDIA driver (roughly 580+). On an older driver, use the `transformers` backend with `gpu=True` instead. See [`RUNNING.md`](RUNNING.md).

macOS users can try to use `"mlx"` by installing the corresponding extra:

```python3
uv add ngec[mlx]
```

<details>
<summary>With pip</summary>

\`\`\`shell
pip install "ngec[extra] @ git+https://github.com/ahalterman/ngec-2025"
\`\`\`

Where `extra` is `vllm`, `mlx`, as needed. 

</details>

### Checking your installation

Installing `ngec` involves enough moving parts -- a PyTorch build, two spaCy models, an inference backend, Elasticsearch -- that several of them can be wrong without anything raising an error. `ngec-doctor` reports what it finds:

```shell
uv run ngec-doctor
```

It prints the installed version and commit, every environment variable ngec and its tooling read (with the effective value and which code reads it), and what the PyTorch build can actually see. Anything it flags is repeated at the bottom with what the problem breaks and the command that fixes it. It exits non-zero only on a real failure, so it is safe to run as a smoke test in CI.

Two flags:

```shell
uv run ngec-doctor --only compute
```

```shell
uv run ngec-doctor --json
```

`--only` takes any comma-separated subset of `install`, `config` and `compute`. `--json` gives the same findings in machine-readable form, which is the more useful thing to paste into a bug report. `python -m ngec.doctor` works too, if you would rather not rely on the console script being on your PATH.

The most common thing it catches is the PyTorch problem described above: on a machine with an NVIDIA GPU, doctor asks the driver directly and compares that against what PyTorch can see, so a torch build that has quietly fallen back to the CPU is reported rather than left to show up as a pipeline that is thirty times slower than expected.

<details>
<summary>If you are using a virtual environment without `uv`</summary>

With the venv active:

\`\`\`shell
ngec-doctor
\`\`\`

Plus any other options as above. 

</details>


### Not using `uv`

Here is a summary of differences if you are not using `uv`, for either install or virtual environment running. 

**Installing packages with `pip` instead of `uv add`:**

1. [Install `mordecai3`](#install-mordecai3) with pip.
2. [Install `ngec`](#install-ngec) with pip.
3. (optional) [Install a backend extra](#inference-backend) like `vllm` or `mlx` with pip.

**Running commands in a manually-activated virtual environment instead of via `uv run`:**

4. [Download the spacy models](#spacy-models) with the venv active.
5. [Run `ngec-doctor`](#checking-your-installation) with the venv active.


### Uninstalling the cache

`ngec` caches some embeddings to improve speed. Those can be easily regenerated if needed. In any case, uninstalling the package will not delete those. They are located at OS-specific cache locations, determing using the [`platformdirs`](https://pypi.org/project/platformdirs/) package. See their documentation for [OS-specific cache folders](https://platformdirs.readthedocs.io/en/latest/platforms.html).


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

# The defaults are shown here; `gpu=True` and/or backend="vllm" are much faster
# on a corpus, and `event_threshold` controls how confident the classifier has
# to be before it assigns an event type.
pc = PloverCoder(es_client=es_client,
                 event_threshold=0.9,
                 attribute_backend="transformers",
                 gpu=False)

story_list = [
        {"id": "story1", "event_text": "Protesters were in the streets in Paris again today to protest against the government's austerity measures.", "pub_date": "2016-05-01"}
    ]
    
event_list = pc.process(story_list)

pprint(event_list, sort_dicts=False, width=100)
```

```
[{'id': 'story1_PROTEST__0',
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
  # Each extracted event is its own record. 'attributes' is a single dict; the
  # resolved actors/recipients, event_location, and date_resolved are top-level.
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
  # 'granularity' is the precision unit (day/week/month/quarter/year);
  # 'date_type' is exact / approximate / range / unresolved; 'date_end' is set
  # only for a genuine range ("Tuesday to Thursday").
  'date_resolved': {'resolved_date': datetime.datetime(2016, 5, 1, 0, 0),
                    'date_end': None,
                    'granularity': 'day',
                    'date_type': 'exact',
                    'reason': '<Resolved day idiom to the publication day>'}}]
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
