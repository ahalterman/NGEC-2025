# NGEC demo

A six-page Streamlit app: one end-to-end run, then one page per pipeline step.

## Run it

```bash
cd demo
uv run --group demo-app streamlit run app.py
```

On this box, prefix everything with `env -u LD_LIBRARY_PATH` so torch picks up
its own CUDA libraries.

## Check it

`check_demo.py` runs every function in `ngec_demo/steps.py` on real inputs with
no Streamlit involved. The pages hold no pipeline logic, so this is the test:

```bash
cd demo
NGEC_DEMO_BACKEND=transformers uv run --group demo-app python check_demo.py
```

It loads the real models and takes a few minutes on CPU.

## Services

- **Elasticsearch** on `localhost:9200` with the `wiki` and `geonames` indices.
  Without it, the entity, actor-code and place steps return nothing and the app
  says so in the sidebar; classification, attribute extraction and date
  resolution still work.
- **Models** are downloaded from Hugging Face on first use and cached
  (`ahalt/qwen3-event-extraction-exp5.1` plus the spaCy and sentence-transformer
  models).

## Environment

| Variable | Default | What it does |
| --- | --- | --- |
| `NGEC_DEMO_BACKEND` | `vllm` if importable and a GPU is visible, else `transformers` | Attribute-model backend |
| `NGEC_DEMO_GPU` | auto (`torch.cuda.is_available()`) | `1`/`0` to force the device |
| `NGEC_DEMO_GPU_MEMORY` | `0.25` | Fraction of the card vllm reserves |
| `NGEC_ATTRIBUTE_MODEL` | published default | A different checkpoint or local path |
| `ES_HOST` / `ES_PORT` | `localhost` / `9200` | Elasticsearch |
| `ES_USER` / `ES_PASSWORD` | unset | Elasticsearch basic auth |

## Layout

- `app.py` — navigation and the status sidebar.
- `ngec_demo/resources.py` — cached model loaders and `health()`.
- `ngec_demo/steps.py` — one function per step, all returning JSON-safe dicts.
- `ngec_demo/examples.py` — the documents and spans the pages open with.
- `ngec_demo/style.py` — CSS, the bar chart, the shared widgets.
- `pages/` — one file per page.
