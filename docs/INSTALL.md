# Installing NGEC: the details

The [README](../README.md#quickstart) has the short version: six commands in a
new uv project. This page explains what each of them does, how to choose the
extras for your machine, how to install with pip instead of uv, and what to do
when Elasticsearch runs somewhere other than `localhost:9200`.

- [What gets installed](#what-gets-installed)
- [Do you need all of it?](#do-you-need-all-of-it)
- [Choosing the extras](#choosing-the-extras)
- [Installing with pip](#installing-with-pip)
- [The models](#the-models)
- [The Elasticsearch index](#the-elasticsearch-index)
- [Checking the installation](#checking-the-installation)
- [Keeping it up to date](#keeping-it-up-to-date)
- [Commands](#commands)
- [Working from a clone](#working-from-a-clone)
- [Other notes](#other-notes)

## What gets installed

NGEC has four parts:

| Part | Size | Installed by |
|---|---|---|
| The Python package, with PyTorch | about 2 GB | `uv add "ngec[...] @ git+..."` |
| Models: two spaCy models, three sentence encoders and the attribute-extraction LLM | about 4 GB | `ngec download-models` |
| A Wikipedia + GeoNames index | 11.6 GB to download, about 15 GB unpacked | `ngec download-index` |
| Elasticsearch 7.10.1, which serves the index | a Docker image | `ngec download-index --start` |

You need [Docker](https://www.docker.com/get-started/) for Elasticsearch,
[uv](https://docs.astral.sh/uv/getting-started/installation/) (or pip, see
below), and about 35 GB of free disk during the install, 20 GB once the index
archive has been deleted. Most of the hour or so an install takes is downloads.

NGEC needs Python 3.10 or newer. `uv init --python 3.12` downloads that Python
if you don't have it.

## Do you need all of it?

Only geocoding and Wikipedia-based actor resolution use Elasticsearch. If you
only need some steps of the pipeline, skip the index:

| Goal | `ngec download-models` | Elasticsearch |
|---|---|---|
| The whole pipeline (`PloverCoder`) | yes | `wiki` + `geonames` |
| Event types only | yes | no |
| Attribute spans (actor, recipient, date, location text) only | yes | no |
| Date resolution only | no | no |
| Coding short actor descriptions ("Kenyan police") | yes | no |
| Wikipedia linking | yes | `wiki` |
| Geoparsing | yes | `geonames` |

`ngec guide pieces` shows how to use each step on its own.

## Choosing the extras

The part in square brackets in `ngec[cpu,llamacpp]` chooses two things.

**Which PyTorch build** (pick one):

| Extra | For |
|---|---|
| `cpu` | no NVIDIA GPU |
| `cu12` | an NVIDIA GPU. Works with any driver that reports CUDA 12 or 13 in `nvidia-smi` |
| `cu13` | an NVIDIA GPU with a CUDA 13 driver, if you are not using vLLM |
| none | macOS, where the default build is the right one |

This matters because the PyTorch you get by default depends on the platform, and
two of the defaults are wrong without saying so: on Windows it is a CPU-only
build, and on Linux it is a CUDA 13 build, which falls back to the CPU on an
older driver. Either way the pipeline runs, many times more slowly than it
should. `ngec doctor` checks for this (see
[Checking the installation](#checking-the-installation)).

**Which backend runs the attribute-extraction model** (pick one):

| Extra | For | Speed |
|---|---|---|
| `llamacpp` | any computer without an NVIDIA GPU: laptops, Windows, servers | about 4.6 s per story and event type on a desktop CPU |
| `vllm` | Linux with an NVIDIA GPU | much faster; the one to use for a large corpus |
| `mlx` | a Mac with Apple Silicon | not measured |

Attribute extraction is where almost all of the running time goes. A story in
which three event types are detected is three prompts, so about 14 seconds with
llama.cpp on a desktop CPU, and more on a laptop. Time a batch of 20 stories
before planning a large run. [`PERFORMANCE.md`](PERFORMANCE.md) has
measurements.

NGEC picks the backend itself (`backend="auto"`): vLLM if it is installed and
there is an NVIDIA GPU, MLX on an Apple Silicon Mac if it is installed, and
llama.cpp otherwise. It logs which one it chose.

So the usual combinations are:

| Your computer | Install |
|---|---|
| No NVIDIA GPU (most laptops, Windows or Linux) | `ngec[cpu,llamacpp]` |
| Mac with Apple Silicon | `ngec[mlx]` |
| Linux with an NVIDIA GPU | `ngec[cu12,vllm]` |
| Windows with an NVIDIA GPU | `ngec[cu12,llamacpp]` |

**vLLM needs `cu12`**, even with a CUDA 13 driver. NGEC pins vLLM to
`>=0.19.1,<0.20`: 0.19.1 is the last vLLM release built against CUDA 12, and it
requires exactly `torch==2.10.0` built against CUDA 12, which `cu12` installs.
A CUDA 12 build runs on CUDA 13 drivers too. `cu13` or `cpu` together with
`vllm` installs without an error and then fails when vLLM loads. In a clone of
the repository, `uv sync` refuses those combinations; installed as a package,
nothing checks them. vLLM publishes Linux wheels only; on Windows it runs under
WSL.

The extras choose the right PyTorch index only under uv, which reads NGEC's own
`[tool.uv.sources]`. pip ignores them; see the next section.

## Installing with pip

With pip, install three things in order: PyTorch for your machine, mordecai3,
and then NGEC.

```shell
# 1. PyTorch: the command for your machine from https://pytorch.org/get-started/locally/, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. mordecai3, which does the geocoding. NGEC needs 3.5.0, which is not on PyPI yet
pip install "mordecai3 @ git+https://github.com/ahalterman/mordecai3@release-3.5"

# 3. NGEC, with a backend
pip install "ngec[llamacpp] @ git+https://github.com/ahalterman/ngec-2025" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

pip keeps an installed PyTorch when it installs NGEC, as long as it is version
2.6 or newer. The extra index in step 3 has ready-built llama.cpp wheels.
Without it, pip compiles llama.cpp on your machine, which needs a C++ compiler
and CMake. For vLLM, install `torch==2.10.0 torchvision==0.25.0
torchaudio==2.10.0` from `https://download.pytorch.org/whl/cu129` in step 1,
and `ngec[vllm]` in step 3.

After that, the commands are the same as in the README without `uv run`:
`ngec download-models`, `ngec download-index --start`, `ngec doctor --smoke`.

## The models

`ngec download-models` downloads every model the pipeline loads:

| Model | Size | Used for |
|---|---|---|
| spaCy `en_core_web_trf`, `en_core_web_lg` | ~900 MB | parsing, and word vectors for actor matching |
| `sentence-transformers/all-mpnet-base-v2` | ~440 MB | event classification |
| `sentence-transformers/static-retrieval-mrl-en-v1`, `BAAI/bge-small-en-v1.5` | ~260 MB | actor resolution (Wikipedia and agent matching) |
| `ahalt/qwen3.5-event-extraction-0.8b` | ~1.8 GB | attribute extraction |
| `ahalt/qwen3.5-event-extraction-0.8b-GGUF` (one 834 MB file) | 834 MB | the same model for llama.cpp; only fetched when `llamacpp` is installed |

The spaCy models are installed as Python packages. The rest go into the Hugging
Face cache (`~/.cache/huggingface`, or `$HF_HOME` if set), which is where the
pipeline looks for them. Models that are already there are skipped, so it is
safe to run again.

The spaCy models have to be installed this way; the pipeline stops with an
error without them. The others would otherwise download the first time the
pipeline needs them, which makes the first run slow and fails on a machine
without internet access.

Options:

- `--attribute-model NAME` downloads a different attribute model, e.g.
  `ahalt/qwen3-event-extraction-exp5.1` for the model in the submitted paper. If
  `NGEC_ATTRIBUTE_MODEL` is set, that model is the default, as it is for the
  pipeline.
- `--no-attribute-model` skips the LLM, e.g. if you run it through a separate
  llama.cpp server.
- `--gguf` fetches the llama.cpp file even when llama-cpp-python is not
  installed.
- `--force` reinstalls the spaCy models and re-downloads the LLM, if you suspect
  a broken download.

## The Elasticsearch index

```shell
uv run ngec download-index --start
```

This downloads the index archive (Wikipedia dump of 2026-09-01, GeoNames of
2026-09-23) to `~/ngec-es-data`, picking up where it stopped if a previous
download was interrupted. It then checks the archive against its published
checksum, unpacks it to `~/ngec-es-data/wikigeo_index`, deletes the archive, and
starts Elasticsearch on it in a Docker container called `ngec-es` on port 9200.
Options: `--dest` puts it somewhere other than `~/ngec-es-data`,
`--keep-archive` keeps the `.tar.gz`, and without `--start` it prints the
`docker run` command instead of running it.

Elasticsearch takes a minute or so to open the indices. It is ready when this
prints a short block of JSON instead of an error:

```shell
curl localhost:9200
```

The Elasticsearch version is pinned: a 7.10 data directory will not open on
Elasticsearch 8. [`elasticsearch/SETUP.md`](../elasticsearch/SETUP.md) has the
same steps with plain `curl`, `tar` and `docker run`, and describes how to build
the indices yourself from a newer Wikipedia dump.

Geocoding uses the same Elasticsearch: NGEC hands its connection to
[mordecai3](https://github.com/ahalterman/mordecai3), so mordecai3's own
`mordecai3 index fetch` is not needed.

**Elasticsearch somewhere else.** NGEC looks for Elasticsearch on
`localhost:9200`. To use a different host, port or credentials, save
[`.env.example`](../.env.example) from this repository into your project folder
as `.env` and uncomment the lines you need. Set `NGEC_WIKI_URL` as well as
`ES_HOST` and `ES_PORT`: one part of actor resolution reads only that. In your
own scripts, `ngec.es_client.es_client_from_env()` connects using `.env` (or
`localhost:9200` without one). `ngec doctor` flags any key in `.env` that NGEC
does not read, since a misspelled setting is otherwise ignored.

## Checking the installation

```shell
uv run ngec doctor            # a few seconds
uv run ngec doctor --smoke    # also codes three news stories; a few minutes on a CPU
```

The doctor prints the installed version and commit, every setting NGEC reads,
what the PyTorch build can see, and whether Elasticsearch is reachable with both
indices in it. Anything it flags is repeated at the bottom with the command that
fixes it. It never downloads anything.

The most common problem it catches is the wrong PyTorch build: on a machine with
an NVIDIA GPU it asks the driver directly and compares that with what PyTorch
sees, so a build that has fallen back to the CPU is reported instead of showing
up as a slow pipeline.

- `--json` gives the same findings machine-readably, which is the most useful
  thing to paste into a bug report.
- `--only` takes any subset of `install`, `config`, `compute`, `elasticsearch`,
  `smoke`. It exits non-zero only on a real failure, and an unreachable
  Elasticsearch counts as one, so on a CI runner without it use
  `--only install,config,compute`.
- `ngec doctor` is also installed as `ngec-doctor`, and `python -m ngec.doctor`
  does the same if neither is on your PATH.

## Keeping it up to date

```shell
uv run ngec update            # what is out of date; changes nothing
uv run ngec update --apply    # update it
```

`ngec update` checks the Hugging Face models against the hub, and the
Elasticsearch index against the current published release. Newer models and a
newer index change the coded output, so if you are in the middle of a project
you may want to keep what you have, and record the change if you update.

**`--apply` replaces the running index**: it downloads the new release (about
12 GB) next to the old one, stops the `ngec-es` container that
`ngec download-index --start` created, starts it again on the new index, and
deletes the old index once the new one comes up with the published document
counts. Elasticsearch is down for about a minute, and if the new index does not
come up the old container is put back. Anything that has a model loaded needs a
restart to use an updated model. It can run unattended, e.g. from cron.
`--no-models`, `--no-index` and `--keep-old` limit what it does.

## Commands

| Command | What it does |
|---|---|
| `ngec download-models` | downloads the spaCy, sentence-transformer and attribute models |
| `ngec download-index` | downloads and unpacks the pre-built Elasticsearch index; `--start` also starts Elasticsearch on it |
| `ngec doctor` | checks the installation and prints the fix for anything wrong; `--smoke` also codes three stories |
| `ngec update` | says whether newer models or a newer index have been published; `--apply` updates them |
| `ngec guide` | prints the guide for coding agents; `--init` points your project's `AGENTS.md` at it |

## Working from a clone

Install the package to *use* NGEC: run the pipeline, use a single step, or code
events with your own actor categories, event definitions or classifier. Clone
the repository only to *change* NGEC, retrain its models, rebuild the
Elasticsearch indices, or run its tests.

From a clone, install with `uv sync` and one of the `cpu` / `cu12` / `cu13`
extras, which also brings in the spaCy models. See
[`DEVELOPING.md`](../DEVELOPING.md). A clone also has a setup doctor that checks
a machine before anything is installed and prints the command to fix whatever is
missing. It is a single standard-library file, so it runs on any Python 3.8+:

```shell
python3 setup/doctor/ngec_doctor.py
```

`--serve` opens a local page that runs each command for you and re-checks
afterwards. See [`setup/doctor/README.md`](../setup/doctor/README.md).

## Other notes

**Logging.** Some of NGEC's dependencies log a lot. To quiet them:

```python
from ngec.logging import quiet_third_party_loggers

quiet_third_party_loggers()
```

`ngec.logging.setup_logging(level=..., format_string=..., quiet_third_party=True)`
sets up NGEC's own logging and does the same.

**Cached embeddings.** NGEC caches some embeddings for speed. Uninstalling the
package does not delete them. They live in the OS cache directory reported by
[`platformdirs`](https://platformdirs.readthedocs.io/en/latest/platforms.html)
and can be regenerated at any time.

**A stale system CUDA.** An `undefined symbol: __nvJitLink...` error means a
CUDA installation on `LD_LIBRARY_PATH` is shadowing PyTorch's own libraries. Run
the command with `env -u LD_LIBRARY_PATH` in front of it.
