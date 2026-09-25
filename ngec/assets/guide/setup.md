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
- **Install PyTorch before NGEC, and pick the build for the machine**: CUDA 12
  or 13 to match `nvidia-smi`, or CPU. The default build on Linux targets CUDA
  13 and, on an older driver, runs on the CPU with no error.
- **vLLM** (the fast backend, Linux + NVIDIA only) needs a CUDA 12 PyTorch,
  `torch==2.10.0`, even when the driver reports CUDA 13.
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

The README's "Installation" section has the full steps and explains each one.
In outline, in a uv project:

```shell
uv add torch --index pytorch=https://download.pytorch.org/whl/cu129   # match the machine
uv add "mordecai3 @ git+https://github.com/ahalterman/mordecai3"
uv add "ngec @ git+https://github.com/ahalterman/ngec-2025"         # ngec[vllm] for vLLM
uv run ngec download-models                                          # ~3 GB
```

Then, if the goal needs it, the pre-built Elasticsearch index (about 11.6 GB)
served by Elasticsearch 7.10.1 in Docker:

```shell
uv run ngec download-index --start
```

This downloads the archive (resuming a partial download), checks its checksum,
unpacks it to `~/ngec-es-data/wikigeo_index` and starts the container on port
9200. Without `--start` it prints the `docker run` command instead. It needs
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

## Choosing a backend for the attribute model

| Backend | Where | Speed |
|---|---|---|
| `vllm` | Linux + NVIDIA | fastest; for corpora |
| `transformers` | anywhere | slow on CPU; `gpu=True` uses a GPU |
| `mlx` | Apple Silicon | install `ngec[mlx]` |
| `llamacpp` | CPU machines, via a running `llama-server` | reasonable on CPU |

`PloverCoder(attribute_backend=...)` picks it. For a few stories,
`transformers` on the CPU is fine. For thousands, a GPU with `vllm` saves hours.

## pip or git clone?

Install the package (the commands above) to **use** NGEC: run the pipeline,
use single steps, supply your own actor categories, definitions or classifier.
Clone the repository only to **change** NGEC itself, retrain its models,
rebuild the Elasticsearch indices, or run its tests; `DEVELOPING.md` there
covers that setup.
