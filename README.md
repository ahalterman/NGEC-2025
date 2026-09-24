# NGEC

*This is a pre-release version of the code. Expect instability and errors when running it.*

NGEC depends on Elasticsearch indices built from Wikipedia and GeoNames, for actor resolution and geocoding, so installing it includes downloading those and running Elasticsearch. See the install instructions below.

## Installation

NGEC has four moving parts: the Python package, a PyTorch build that matches your machine, about 3 GB of models (two spaCy models, three sentence encoders and a small LLM), and an Elasticsearch node holding a Wikipedia + GeoNames index. Most of the elapsed time is downloads.

**Before you start:**

| | |
|---|---|
| **Docker** | <https://www.docker.com/get-started/> — for Elasticsearch |
| **Python** | 3.10 or newer |
| **uv** | <https://docs.astral.sh/uv/getting-started/installation/> — or use pip; each step has the pip version |
| **Disk** | ~30 GB free during install, ~18 GB once you delete the archive |
| **Time** | About an hour, most of it downloading |
| **GPU** | Optional. Everything runs on CPU, just slowly |

If a step fails, see [When something is wrong](#when-something-is-wrong) below.

---

### Step 1. Start the index download

This is a ~10 GB download and the longest single step, so start it in a terminal window of its own and carry on with the other steps in another.

> ⚠️ **There is currently no public download URL.** The address this README used
> to give returns HTTP 404. For now, **ask Andy for the archive directly**, or
> build the indices yourself — see [`elasticsearch/SETUP.md`](elasticsearch/SETUP.md).

```shell
mkdir -p ~/ngec-es-data
cd ~/ngec-es-data
curl -LO <URL of the index archive>
```

### Step 2. Install PyTorch

NGEC is installed into a [uv](https://docs.astral.sh/uv/) project. If you don't have one yet, create it in the folder you want to work in (not in `~/ngec-es-data`):

```shell
uv init my-ngec-project
cd my-ngec-project
```

Then install PyTorch **before** NGEC, choosing the build that matches your machine. Otherwise you get whatever the default is for your platform: on Windows with an NVIDIA GPU that is a CPU-only build, and on Linux a CUDA 13 build that falls back to the CPU on an older driver. Both run many times slower, with no error to tell you so.

| Your machine | Command |
|---|---|
| macOS | Nothing to do: the default build is right. Go to step 3 |
| NVIDIA GPU, `nvidia-smi` reports CUDA 12.x | `uv add torch --index pytorch=https://download.pytorch.org/whl/cu129` |
| NVIDIA GPU, `nvidia-smi` reports CUDA 13.x | `uv add torch --index pytorch=https://download.pytorch.org/whl/cu130` |
| NVIDIA GPU on Linux, planning to use the faster vLLM backend | The vLLM command below |
| Everything else | `uv add torch --index pytorch=https://download.pytorch.org/whl/cpu` |

`nvidia-smi` prints the CUDA version in the top right corner. The `--index` is recorded in your project, so installing NGEC in step 3 keeps this build instead of replacing it with the default one.

**For vLLM**, use a CUDA 12 build pinned to the versions vLLM requires, even if `nvidia-smi` reports CUDA 13 ([why](#step-3-install-ngec)):

```shell
uv add "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" --index pytorch=https://download.pytorch.org/whl/cu129
```

[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) shows how to check which build you ended up with.

<details>
<summary>With pip instead of uv</summary>

Install PyTorch first, with the command for your machine from <https://pytorch.org/get-started/locally/>, e.g.:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cu129
```

pip keeps an installed PyTorch when you install NGEC afterwards, as long as it is version 2.6 or newer. For vLLM, install `torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0` from the CUDA 12 index above.

</details>

### Step 3. Install NGEC

In the same project folder, install NGEC and [mordecai3](https://github.com/ahalterman/mordecai3), which geocoding depends on. Both are in active development, so install both from GitHub:

```shell
uv add "mordecai3 @ git+https://github.com/ahalterman/mordecai3"
uv add "ngec @ git+https://github.com/ahalterman/ngec-2025"
```

<details>
<summary>With pip instead of uv</summary>

```shell
pip install "mordecai3 @ git+https://github.com/ahalterman/mordecai3"
pip install "ngec @ git+https://github.com/ahalterman/ngec-2025"
```

</details>

<details>
<summary>A faster inference backend (optional, Linux + NVIDIA)</summary>

The default backend is `transformers`: portable and slow. On CUDA, vLLM is much faster. Instead of the `ngec` line above:

```shell
uv add "ngec[vllm] @ git+https://github.com/ahalterman/ngec-2025"
```

With pip, the same with `pip install`.

The pinned vLLM (`>=0.19,<0.20`) is the last CUDA 12 build, which runs on both CUDA 12 and CUDA 13 drivers. It requires exactly `torch==2.10.0`, and vLLM's compiled code needs the CUDA 12 runtime that a CUDA 12 PyTorch build brings along. That is why step 2 has a separate command for vLLM: a CUDA 13 build of torch 2.10.0 satisfies the version pin, so it is kept, and vLLM then fails when it loads.

vLLM publishes Linux wheels only. On Windows it runs under WSL. Wherever vLLM won't install, the `transformers` backend with `gpu=True` still uses the GPU. macOS users can try `ngec[mlx]` instead.

</details>

### Step 4. Download the models

None of the models NGEC uses come with installing it. One command fetches all of them, about 3 GB together:

```shell
uv run ngec download-models
```

| Model | Size | Used for |
|---|---|---|
| spaCy `en_core_web_trf`, `en_core_web_lg` | ~900 MB | parsing, and word vectors for actor matching |
| `sentence-transformers/all-mpnet-base-v2` | ~440 MB | event classification |
| `sentence-transformers/static-retrieval-mrl-en-v1`, `BAAI/bge-small-en-v1.5` | ~260 MB | actor resolution (Wikipedia and agent matching) |
| `ahalt/qwen3-event-extraction-exp5.1` | ~1.2 GB | attribute extraction |

The spaCy models are installed as Python packages. The rest go into the Hugging Face cache (`~/.cache/huggingface`, or `$HF_HOME` if set), which is where the pipeline looks for them. Models that are already there are skipped, so it is safe to run again.

The spaCy models have to be downloaded this way; the pipeline stops with an error without them. The others would otherwise download the first time the pipeline needs them, which works, but makes the first run slow and fails on a machine without internet access.

A few options:

- `--attribute-model NAME` downloads a different attribute model, e.g. `ahalt/event-attribute-extractor` for the original model. If `NGEC_ATTRIBUTE_MODEL` is set, that model is the default, as it is for the pipeline.
- `--no-attribute-model` skips the LLM, e.g. if you run it through a llama.cpp server, which uses its own GGUF file.
- `--force` reinstalls the spaCy models and re-downloads the LLM, if you suspect a broken download.

<details>
<summary>In a virtual environment without uv</summary>

With the venv active, run `ngec download-models`.

</details>

### Step 5. Start Elasticsearch on the index

Unpack the archive from step 1. The 2023 archive unpacks to a directory named `geonames_index` for historical reasons, but it holds *both* indices — rename it so the next person isn't misled:

```shell
cd ~/ngec-es-data
tar -xzf geonames_wiki_index_2023-03-02.tar.gz
mv geonames_index wikigeo_index
```

Newer archives are named `wikigeo_index.tar.gz` and already unpack to `wikigeo_index`, so there is nothing to rename.

Then start Elasticsearch over it:

```shell
docker run -d --name ngec-es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  --restart unless-stopped \
  -v "$HOME/ngec-es-data/wikigeo_index":/usr/share/elasticsearch/data \
  elasticsearch:7.10.1
```

If you unpacked the archive somewhere else, change the path after `-v`, and write it out in full rather than with `~`. Given a path that does not exist, Docker creates an empty directory, and Elasticsearch starts happily with no indices in it instead of failing.

Elasticsearch takes a minute or so to start. It is ready when this prints a short block of JSON instead of an error:

```shell
curl localhost:9200
```

You can delete the tarball now. NGEC connects to `localhost:9200` by default. To use a different host or port, save [`.env.example`](.env.example) from this repository into your project folder as `.env`, and uncomment the lines you need. It lists every setting NGEC reads, with what each one does. The tests, the demo and `ngec-doctor` read `.env` automatically; your own scripts need to pass the host and port to `ngec.es_client.setup_es_client`.

The Elasticsearch version is pinned: a 7.10 data directory will not open on Elasticsearch 8. [`elasticsearch/SETUP.md`](elasticsearch/SETUP.md) explains every flag, how to tell a wrong volume path from a half-loaded index, and how to build the indices yourself from a newer Wikipedia dump.

### Step 6. Check that it works

```shell
uv run ngec-doctor --smoke
```

This checks the installation, then runs three real news articles all the way through the pipeline and prints the coded events. It takes a few minutes on CPU. If it prints events, the install is good. It never downloads anything: if a model is missing, it says so and tells you to run `ngec download-models`.

---

### When something is wrong

Without `--smoke`, the doctor checks the pieces in a few seconds, without running the pipeline:

```shell
uv run ngec-doctor
```

It prints the installed version and commit, every environment variable NGEC reads, what the PyTorch build can see, and whether Elasticsearch is reachable with both indices in it. It also flags any key in your `.env` that NGEC does not read: a misspelled setting is otherwise ignored without error. Anything it flags is repeated at the bottom with the command that fixes it. `--json` gives the same findings machine-readably, which is the more useful thing to paste into a bug report. `--only` takes any subset of `install`, `config`, `compute`, `elasticsearch`, `smoke`.

The most common thing it catches is the PyTorch problem from step 2: on a machine with an NVIDIA GPU it asks the driver directly and compares that against what PyTorch sees, so a build that has quietly fallen back to the CPU is reported rather than left to show up as a pipeline that is many times slower than expected.

It exits non-zero only on a real failure, so it is also safe to run in CI. An unreachable Elasticsearch counts as one, so on a CI runner without it use `--only install,config,compute`. If the `ngec-doctor` command is not on your PATH, `python -m ngec.doctor` does the same; in a virtual environment without uv, activate it and run `ngec-doctor` directly.

### Working from a clone

Contributors install differently — `uv sync` with one of the `cpu` / `cu12` / `cu13` extras, which redirect PyTorch to the right index automatically and bring in the spaCy models. See [`DEVELOPING.md`](DEVELOPING.md).

A clone also has the **setup doctor**, which checks a machine before anything is installed and prints the exact command to fix whatever is missing. It is a single standard-library file, so it runs on any Python 3.8+:

```shell
python3 setup/doctor/ngec_doctor.py
```

`--serve` opens a local page that runs each command for you and re-checks afterwards. See [`setup/doctor/README.md`](setup/doctor/README.md). Its fixes assume the clone workflow (`uv sync --extra …`), which is why it is here rather than in the steps above. Working with Claude Code, the `ngec-setup` skill drives the same loop conversationally.

### Cached embeddings

NGEC caches some embeddings for speed. Uninstalling the package does not delete them; they live in the OS cache directory reported by [`platformdirs`](https://platformdirs.readthedocs.io/en/latest/platforms.html) and can be regenerated at any time.


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

# `gpu=True` and/or attribute_backend="vllm" are much faster on a corpus.
# `event_threshold` is left unset on purpose: that uses the per-type thresholds
# recorded with the classifier. Setting it applies one threshold to every
# event type instead.
pc = PloverCoder(es_client=es_client,
                 attribute_backend="transformers",
                 gpu=False)

story_list = [
        {"id": "story1", "event_text": "Protesters were in the streets in Paris again today to protest against the government's austerity measures.", "pub_date": "2016-05-01"}
    ]
    
event_list = pc.process(story_list)

pprint(event_list, sort_dicts=False, width=100)
```

```
[{'id': 'story1_PROTEST_demo_0',
  'event_text': 'Protesters were in the streets in Paris again today to protest against the '
                "government's austerity measures.",
  'pub_date': '2016-05-01',
  'event_type': 'PROTEST',
  'event_type_confidence': {'PROTEST': 0.9999998654438295},
  'event_mode': 'demo',
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
                                 "against the government's austerity measures",
                 'actor': ['Protesters'],
                 'recipient': ['government'],
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
             'actor_pattern_conf': 0.9999999999997888,
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
                 'actor_pattern_conf': 0.9999999999996712,
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
