# NGEC demo

An eight-page Streamlit app: one end-to-end run, one page per pipeline step, a
bulk uploader and a timing page.

## Run it

```bash
cd demo
env -u LD_LIBRARY_PATH uv run --extra cu12 --extra vllm --group demo-app streamlit run app.py
```

Pass the same extras to every `uv run`: `uv run` syncs the environment to the
extras it is given, so a bare `uv run --group demo-app` swaps the CUDA 12 torch
that vllm needs for the default CUDA 13 build and leaves the two half-installed.
On a CPU-only box use `--extra cpu` instead of the two above.

The `env -u LD_LIBRARY_PATH` prefix is for this box, so torch picks up its own
CUDA libraries.

## Pages

- **NGEC** — a document in, coded event records out.
- **1**–**4** — classification, attribute extraction, entity resolution, actor
  coding, one page each, with the timing broken down by call.
- **5. When and where?** — a date phrase against a free-text publication date,
  then place names against geonames as a table and raw JSON (no map; the
  mapping demo lives elsewhere).
- **Bulk** — a CSV or JSONL of documents in, the coded events out as JSONL or
  CSV. GPU mode only, since on the CPU path a document takes seconds, and capped
  at 200 documents because the box is shared.
- **Timing** — the same document coded in every available mode, component by
  component, with what runs where and what loading cost.

## Models

Loading the models takes a minute and happens once per process, not per
visitor. The sidebar's **Load models** button does it deliberately, naming each
component as it goes, and then says how long it took; otherwise the first click
on any page pays for it, under a status that says so rather than a spinner that
looks like a hang. The sidebar's status block says whether the current mode is
loaded.

## Modes

The sidebar has a **Compute** toggle: GPU (vllm for the attribute model, the
sentence encoders on `cuda`) or CPU (llama-server for the attribute model, the
encoders on `cpu`, torch limited to `NGEC_DEMO_CPU_THREADS` so the numbers match
the deployment box). Both model sets stay loaded, cached per mode, so switching
is a click. spaCy and the mordecai geoparser run on CPU in both modes — spaCy's
device is process-global — and the Timing page says so.

On a box with no card the toggle collapses to a CPU caption. **CPU mode needs a
running `llama-server`** holding the quantized attribute model; without it the
sidebar says the server is down and step 2 onwards returns nothing. The systemd
user unit and the model files are described in `deploy/README.md`.

## Check it

`check_demo.py` runs every function in `ngec_demo/steps.py` on real inputs with
no Streamlit involved. The pages hold no pipeline logic, so this is the test:

```bash
cd demo
env -u LD_LIBRARY_PATH uv run --extra cu12 --extra vllm --group demo-app python check_demo.py --mode all
```

`--mode gpu|cpu|all` (default: every available mode). It loads the real models
and takes a few minutes; the CPU pass needs `llama-server` up.

## Services

- **Elasticsearch** on `localhost:9200` with the `wiki` and `geonames` indices.
  Without it, the entity, actor-code and place steps return nothing and the app
  says so in the sidebar; classification, attribute extraction and date
  resolution still work.
- **llama-server** on `http://127.0.0.1:8080`, for CPU mode only.
- **Models** are downloaded from Hugging Face on first use and cached
  (`ahalt/qwen3-event-extraction-exp5.1` plus the spaCy and sentence-transformer
  models).

## Environment

| Variable | Default | What it does |
| --- | --- | --- |
| `NGEC_DEMO_MODE` | first available (`gpu` when there is a card) | Mode the app opens in, and the mode outside Streamlit |
| `NGEC_DEMO_CPU_THREADS` | `4` | Torch threads in CPU mode — the deployment box's core count |
| `NGEC_DEMO_CPU_BACKEND` | `llamacpp` | Attribute-model backend in CPU mode (`transformers` runs it in-process) |
| `NGEC_LLAMACPP_URL` | `http://127.0.0.1:8080` | Where llama-server is listening |
| `NGEC_DEMO_GPU_MEMORY` | `0.25` | Fraction of the card vllm reserves |
| `NGEC_ATTRIBUTE_MODEL` | published default | A different checkpoint or local path |
| `ES_HOST` / `ES_PORT` | `localhost` / `9200` | Elasticsearch |
| `ES_USER` / `ES_PASSWORD` | unset | Elasticsearch basic auth |

## Layout

- `app.py` — navigation, the compute toggle, the Load models button and the
  status sidebar.
- `ngec_demo/resources.py` — cached model loaders, `load_all()`, `health()` and
  the modes.
- `ngec_demo/steps.py` — one function per step, all returning JSON-safe dicts.
- `ngec_demo/timing.py` — the per-component timers the loaders wrap models in.
- `ngec_demo/examples.py` — the documents and spans the pages open with.
- `ngec_demo/style.py` — the light theme's CSS, the bar chart, the timing
  table, the sidebar block and the widgets the pages share.
- `pages/` — one file per page.
