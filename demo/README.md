# The NGEC demo app

An interactive demonstration of the pipeline described in *Creating Custom Event
Data: A Bag-of-Tricks*. Every panel runs the real package on the text the visitor
supplies — nothing is pre-rendered — and the cases where the pipeline does badly
have their own pages.

## Running it

```shell
cd demo
uv run --group demo-app streamlit run app.py
```

Run it **from this directory** so Streamlit picks up `.streamlit/config.toml`,
which carries the theme. Use `uv run`, not a `streamlit` that happens to be on
your `PATH`: the package needs Python 3.10+, and launching it from an older
interpreter is the most common way to get a confusing import error. The app
checks for this at startup and says so.

On the reference machine, a stale system CUDA shadows the PyTorch wheels'
libraries, so prefix with `env -u LD_LIBRARY_PATH`.

## What it needs

| | |
|---|---|
| Elasticsearch | with `wiki` and `geonames` indices. Steps 3–5 degrade to a message without it rather than crashing. |
| spaCy | `en_core_web_lg` and `en_core_web_trf` |
| Models | the sentence encoder and the demo classifiers come from the Hugging Face cache; the attribute model is described below |
| `llama-server` | the default backend. A CPU host is the deployment target, so that is what the demo defaults to. A GPU makes it ~10x faster — see below. |

### Which attribute model

This demo runs the 2026 retraining (`exp5.1`), published as
`ahalt/qwen3-event-extraction-exp5.1` — 18 points better on exact actor match and
19 on location than the original `ahalt/event-attribute-extractor` (see
`setup/hf_release/`). On this training box it is loaded from a local directory
instead of downloaded, to skip re-fetching weights already on disk; if that
directory is missing the demo falls back to `AttributeModel`'s default, which is
the same published model.

The two take **different prompt formats**, and a mismatch is silent — valid JSON,
worse contents. `ngec/attribute_model.py` therefore pairs each model with its
format in `KNOWN_PROMPT_FORMATS` rather than leaving them as two independent
settings. Under the `llamacpp` backend the weights come from whatever GGUF
`llama-server` was started with while the format comes from `NGEC_ATTRIBUTE_MODEL`,
so those two can still disagree; the health check compares them and reports a
mismatch as an unavailability rather than running anyway.

If a service the demo needs is down, every page opens with a banner naming it,
saying which steps are unavailable while it is down, and — behind an expander —
the command that brings it back. `demo/deploy/` has the units that stop this
happening after a reboot.

Configuration is by environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `ES_HOST`, `ES_PORT` | `localhost`, `9200` | Elasticsearch |
| `ES_USER`, `ES_PASSWORD` | unset | passed as `http_auth` when both are set |
| `NGEC_DEMO_BACKEND` | `llamacpp` | `vllm` on a GPU host, `transformers` as a slow fallback |
| `NGEC_LLAMACPP_URL` | `http://127.0.0.1:8080` | where `llama-server` is listening |
| `NGEC_DEMO_GPU` | unset | `1` to force the GPU, `0` to force the CPU; `vllm` implies the GPU |
| `NGEC_DEMO_GPU_MEMORY` | `0.2` | fraction of the card vllm reserves. 0.2 of 24 GB is ample for a 0.6B model coding one document at a time, and leaves room for the sentence encoder on the same card |
| `NGEC_ATTRIBUTE_MODEL` | the local `exp5.1` directory if present, else `ahalt/qwen3-event-extraction-exp5.1` | attribute model name or path |
| `NGEC_DEMO_PAPER_URL` | `app/static/paper.pdf` | where the manuscript is served from |
| `ANTHROPIC_API_KEY` | unset | enables the live comparison on the "Against a frontier LLM" page |
| `NGEC_DEMO_LLM_MODEL` | `claude-opus-5` | model for that comparison |

A `.env` file in the repository root is loaded if present.

## The CPU path (the default)

The deployment target is a CPU-only server, so `llamacpp` is the default
backend. With the plain `transformers` backend the attribute model takes about
**60 seconds per document**, which is too slow to demo; serving a quantized copy
through `llama.cpp` brings that to about **12 seconds**. The numbers and the
reasoning are in `DESIGN.md`; the recipe is:

```shell
# 1. Build llama.cpp, CPU only (~3 minutes; needs cmake and a C++ compiler)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target llama-server llama-quantize

# 2. Convert the attribute model to GGUF and quantize it. Convert the model you
#    actually intend to serve — the local exp5.1 directory here, though it is
#    the same weights as the published ahalt/qwen3-event-extraction-exp5.1.
#    bf16, not f16: the model was trained in bf16, and downcasting bf16 -> f16
#    is an extra, avoidable rounding step before quantization even starts (see
#    setup/hf_release/README.md).
MODEL=~/projects/train_NGEC_2026/qwen3-event-extraction-exp5.1
PYTHONPATH=gguf-py python convert_hf_to_gguf.py "$MODEL" \
    --outfile attr-exp5.1-bf16.gguf --outtype bf16
./build/bin/llama-quantize attr-exp5.1-bf16.gguf attr-exp5.1-q8.gguf Q8_0

# 3. Serve it. The demo defaults to this backend and this URL.
./build/bin/llama-server -m attr-exp5.1-q8.gguf --port 8080 -c 8192 -t 16 &
uv run --group demo-app streamlit run app.py
```

Re-quantize whenever the model changes — the GGUF is a copy, not a reference,
and the health check will tell you when the server is serving a different model
from the one the pipeline is prompting for.

## On a GPU host

```shell
NGEC_DEMO_BACKEND=vllm uv run --group demo-app streamlit run app.py
```

About **1 second per document** against ~12 on CPU, no second service to run,
and the same numbers a corpus run would get. This is also what
`build_example_cache.py` should be run with. vllm reserves 20% of the card by
default here (`NGEC_DEMO_GPU_MEMORY`), which is plenty for a 0.6B model and
leaves room for the sentence encoder the classifier pages use.

**Use Q8_0, not Q4_K_M.** Q4 is faster still, but it diverged from the F16
reference on every one of 12 test prompts where Q8_0 matched 5 — see `DESIGN.md`.
For anything beyond demonstration, validate the quantization against the
held-out annotations before trusting it.

The health panel reports whether `llama-server` is reachable, and a stopped
server produces the startup banner rather than a hang. `demo/deploy/` has a
systemd user unit that keeps it running across reboots and logouts.

## Before deploying

1. **Build the example cache.** Every curated example is pre-coded, so clicking
   between them is instant and every visitor sees the same records from a model
   that samples at temperature 0.5. Build it **on a GPU** — the cached answers
   then come from the unquantized model:

   ```shell
   NGEC_DEMO_BACKEND=vllm uv run python demo/build_example_cache.py
   ```

   Half a minute on a GPU, about three minutes with the llamacpp backend. Re-run
   it whenever the examples or any model change.

2. **Put the paper where the app can serve it.** Copy the compiled PDF to
   `demo/static/paper.pdf`, or point `NGEC_DEMO_PAPER_URL` at a hosted copy.
   Every step page deep-links into it by section, and the page numbers in
   `ngec_demo/paper.py` must match the PDF being served — recompiling the
   manuscript with different pagination silently breaks those links.

3. **Decide about the LLM comparison.** Without an API key that page explains
   itself and runs only the NGEC side. With one, it makes live API calls on
   visitor-supplied text, so rate-limit it if the URL is public.

## Layout

```
app.py                    entry point; st.navigation defines the sections
.streamlit/config.toml    theme
ngec_demo/
  theme.py                palette, typography, record tables, bar rows
  paper.py                section → PDF page map, used for every "in the paper" link
  examples.py             the curated documents every page opens with
  resources.py            cached model loading + dependency health
  pipeline.py             runs the pipeline with a per-step trace; single-step helpers
  llm_baseline.py         the one-prompt frontier-LLM comparison
  ui.py                   page furniture, example picker, record rendering
pages/                    one file per page, grouped by the sections in app.py
data/                     build artefacts (gitignored)
static/                   paper.pdf
```

## Conventions worth keeping

- **No empty text boxes.** Every page opens with a worked example already coded.
  A visiting reviewer will not type one in.
- **The example follows you.** The selected document lives in the URL query
  string, so the step rail carries it from page to page and any panel can be
  linked to directly from a response letter — `.../step5?ex=anniversary`.
- **Degrade, don't crash.** Every dependency is checked and reported in the
  sidebar; a missing index produces a message naming what is unavailable.
  `showErrorDetails` is off in the config so a traceback never reaches a visitor.
- **Ship the failures.** Examples marked `honest=True` in `examples.py` are
  included *because* the pipeline handles them badly, and are labelled as such.

## Look and colour

The theme is the "Pocket Operator" design handoff: white paper, black hairlines,
grey chips, one orange. `handoff/pocket-operator.css` is the handoff as authored
and owns the design tokens; `ngec_demo/theme.py` loads it and adds this app's own
components on top, written against the same custom properties. Change a colour
there, not in a page. See "The visual theme" in `DESIGN.md`.

The four attribute colours are slots 1, 4, 3 and 7 of the reference categorical
palette. They clear the all-pairs colour-vision gates **with the accent included
in the set** (worst CVD ΔE 9.1, worst normal-vision ΔE 16.3) — the accent and an
attribute table appear on the same screen, so validating the four alone is not
enough. Recipient is yellow rather than orange for exactly that reason. The
obvious "muted academic" choice — blue, terracotta, olive, teal — failed
outright: olive and terracotta came out at ΔE 3.9 under protanopia, which is to
say the same colour. If you change them, re-run the validator rather than
trusting your eye. Identity is carried by a saturated 3px rule down the left edge
of the row label plus the label in words, so nothing depends on colour alone.
