# Setting up NGEC

## Work from the doctor, not from error messages

```shell
ngec doctor --json          # or: uv run ngec doctor --json
```

It reports the installed version, every setting NGEC reads and where each
value came from, the PyTorch build compared against the GPU driver, and
whether Elasticsearch answers with both indices loaded. Every problem comes with
the command that fixes it. Work through them in the order listed, one at a time:
say what the problem is and why it matters, show the fix, and **ask the user
before running it**. Then rerun the doctor.

`ngec doctor --smoke` then runs three real stories through the whole pipeline.
If it prints events, the install works. It takes a few minutes on a CPU.

Only if `ngec` itself is not installed yet: the NGEC repository has a second
doctor, `setup/doctor/ngec_doctor.py`, that runs on any Python 3.8+ and says
which install command fits this machine. It is not in the pip package.

## Rules

- **Never use `sudo`**, and never install system software (Docker, drivers)
  yourself. Give the user the command and wait.
- **Pick the PyTorch build with an extra**: `cpu` without an NVIDIA GPU,
  `cu12` with one (any driver reporting CUDA 12 or 13), none on macOS. Without
  one, the default build on Linux targets CUDA 13 and, on an older driver, runs
  on the CPU with no error.
- **vLLM** (the fast backend, Linux + NVIDIA only) needs `cu12`, even when the
  driver reports CUDA 13. Installed as a package, nothing stops `cpu,vllm` or
  `cu13,vllm`; both fail when vLLM loads.
- **An `undefined symbol: __nvJitLink...` error** means a system CUDA on
  `LD_LIBRARY_PATH` is shadowing PyTorch's own libraries. Prefix the command
  with `env -u LD_LIBRARY_PATH`.
- **Do not start, stop or remove an Elasticsearch container you did not
  create** unless the user asks. Two Elasticsearch nodes on one data directory
  corrupt it.
- Don't change NGEC's code or the user's `.env` to make a check pass.

## What each goal needs

Install only what the user's goal requires. Elasticsearch is the slow part.

| Goal | PyTorch + `ngec` | `ngec download-models` | Elasticsearch |
|---|---|---|---|
| Whole pipeline (`PloverCoder`) | yes | yes | `wiki` + `geonames` |
| Event types only | yes | yes | no |
| Attribute spans only | yes | yes (or `--attribute-model`) | no |
| Date resolution only | yes | no | no |
| Actor categories from short text | yes | yes | no |
| Wikipedia linking | yes | yes | `wiki` |
| Geoparsing | yes | yes | `geonames` |

## Install, in short

The README's Quickstart has the steps, and `docs/INSTALL.md` in the
repository explains each one. In a new uv project:

```shell
uv init --python 3.12 my-ngec && cd my-ngec    # skip if the user already has a project
uv add "ngec[cpu,llamacpp] @ git+https://github.com/ahalterman/ngec-2025"
uv run ngec download-models                    # ~4 GB
```

The extras choose the PyTorch build and the attribute-model backend, and uv
fetches both from the right index on its own (NGEC's `[tool.uv.sources]`):

| Machine | Extras |
|---|---|
| No NVIDIA GPU | `cpu,llamacpp` |
| Apple Silicon Mac | `mlx` |
| Linux + NVIDIA GPU | `cu12,vllm` |
| Windows + NVIDIA GPU | `cu12,llamacpp` |

With pip the extras do not choose the index: install PyTorch first, then
`mordecai3 @ git+https://github.com/ahalterman/mordecai3@release-3.5`, then
`ngec[llamacpp]` with
`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
`docs/INSTALL.md` has the commands.

Then, if the goal needs it, the pre-built Elasticsearch index (about 11.6 GB)
served by Elasticsearch 7.10.1 in Docker:

```shell
uv run ngec download-index --start
```

This downloads the archive (resuming a partial download), checks its checksum,
unpacks it to `~/ngec-es-data/wikigeo_index` and starts the container on port
9200, published on this machine only (`127.0.0.1`: the node has no password).
If NGEC runs on another machine, use an SSH tunnel rather than opening the port.
Without `--start` it prints the `docker run` command instead. It needs
Docker installed and running; give the user the Docker install to do
themselves. NGEC looks for Elasticsearch on `localhost:9200`. A `.env` file in the working directory
(template: `.env.example` in the repository) changes the host, port and
credentials. In scripts, `ngec.es_client.es_client_from_env()` connects using
it (or `localhost:9200` without one) and fails straight away if Elasticsearch
is not answering.

Elasticsearch runs in Docker on Linux, macOS and Windows. Without Docker, the
steps that need no Elasticsearch still work, on a Mac as elsewhere: event
classification, attribute extraction, date resolution and coding short actor
descriptions.

`ngec doctor` compares the indices' document counts with what a complete
index has. Trust its verdict: an older index has somewhat fewer documents
and works, while one far short of that is a load that stopped partway.

## Keeping it up to date

`ngec update` says whether newer models or a newer Elasticsearch index have
been published, and changes nothing. `ngec update --apply` installs them: it
replaces the running index, which stops Elasticsearch for about a minute.
**Ask the user before running `--apply`.** Newer models and a newer index
change the coded output, so a user in the middle of a project may want to
keep what they have, and should record the change if they update.

## Choosing a backend for the attribute model

Attribute extraction is where almost all the running time goes. By default
(`backend="auto"`) NGEC picks: vLLM if it is installed and there is an NVIDIA
GPU, MLX on an Apple Silicon Mac if it is installed, and llama.cpp otherwise.
It logs which one it chose, and if the package is missing it says what to
install.

| Backend | For | Install | Speed |
|---|---|---|---|
| `vllm` | Linux with an NVIDIA GPU | `ngec[cu12,vllm]` | fastest; for large corpora |
| `llamacpp` | any CPU: laptops, Windows, servers without a GPU | `ngec[cpu,llamacpp]` | about 4.6 s per story and event type on a desktop CPU |
| `mlx` | a Mac with Apple Silicon | `ngec[mlx]` | not measured |
| `transformers` | deprecated | built in | about 3x slower than llamacpp on a CPU, twice the memory |

The time is per story *and event type*: a story with three detected event
types is three prompts, so about 14 seconds with llamacpp on a desktop CPU,
and more on a laptop. Time a batch of 20 stories on the user's machine before
promising anything.

llamacpp runs the model inside Python from a quantized copy (an 834 MB GGUF
file, fetched the first time, or ahead of time by `ngec download-models`).
Quantizing costs nothing measurable in accuracy. It uses one thread per
performance core, up to 8; `NGEC_LLAMACPP_THREADS` changes that, but more
threads are usually slower on CPUs that also have efficiency cores. uv
installs a ready-built llama.cpp; pip does too with the extra index above, and
without it compiles llama.cpp from source, which needs a C++ compiler and
CMake. To use a separately running `llama-server` instead, set
`NGEC_LLAMACPP_URL`.

## pip or git clone?

Install the package (the commands above) to **use** NGEC: run the pipeline,
use single steps, supply your own actor categories, definitions or classifier.
Clone the repository only to **change** NGEC itself, retrain its models,
rebuild the Elasticsearch indices, or run its tests; `DEVELOPING.md` there
covers that setup.
