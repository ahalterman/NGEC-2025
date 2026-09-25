# Running NGEC end to end

Notes from doing full end-to-end runs on a Linux/CUDA box, written for whoever
does it next. `README.md` covers installation and the API, and `ngec guide run`
covers how to code a corpus (preparing the input, a batch script, counting
events, checking the output). This file is the maintainer's record of what
actually happens when you point the pipeline at a corpus: what you need
running, how fast it goes, what the output looks like, and the errors you are
likely to hit.

If you only want to know what each step puts in the record, read `PIPELINE.md`
instead. For step 1 specifically — where the event and mode classifiers came
from and what is still wrong with them — read `CLASSIFIERS.md`.

**Note on the numbers below.** Each measurement says what it was run with. The
500-story corpus run in §6 and §8 predates both the retrained event classifiers
and the current attribute model, so its event counts and fill rates are not a
baseline for the current pipeline. The CPU timings in §5 were measured with the
current default attribute model.

## 1. What you need before you start

| | requirement | notes |
|---|---|---|
| Python | 3.10+ | the base conda env on many systems is 3.9 and will not import the package |
| Elasticsearch | 7.10.x, with a `geonames` **and** a `wiki` index | see below |
| Disk | ~33 GB free during install, ~18 GB after | 15 GB for the unpacked index, 3–4 GB of models; the 11.6 GB archive is deleted after unpacking |
| GPU | optional | CPU works; see §4 for the CUDA caveats and §5 for CPU speed |

**`PloverCoder` needs Elasticsearch.** Step 2 (geolocation) queries `geonames`
and step 5 (actor resolution) queries `wiki`. Both check their index when they
are constructed, so without them `PloverCoder(...)` fails before processing
anything (the messages are in the §9 table). The single steps that do not
use Elasticsearch (event types, attribute extraction, date resolution, actor
categories from text) can run without it; `ngec guide pieces` shows how.

The current published index is Elasticsearch 7.10.1 in Docker, from the
Wikipedia dump of 2026-09-01 and GeoNames of 2026-09-23:

```
health index    docs.count
green  geonames   13472152
green  wiki        7936742
```

`ngec download-index --start` fetches it and starts it (README step 5).
`elasticsearch/SETUP.md` has the same steps by hand and describes building the
indices yourself. Check what you have with:

```shell
curl -s "localhost:9200/_cat/indices?v&h=health,index,docs.count"
```

Use a 2026-09 or later index. The Wikipedia matcher searches accent-folded
sub-fields by default (`use_folded=True` in `ngec/actors/wiki_matcher.py`),
which older indices do not have; on an older index those clauses match nothing
and ranking changes without an error. `ngec update` compares the running index
with the published release, and the setup doctor in a clone checks its document
counts and age; `ngec doctor` only checks that both indices are there.

The client is pinned to `elasticsearch==7.17.9`, which talks to ES 7.x, and a
7.10 data directory opens only on ES 7.10.x. An ES 8 or 9 server will not work.

## 2. Install

Follow the README's installation steps. In short, for an installed package:
install the PyTorch build for your machine, then NGEC, then

```shell
uv run ngec download-models        # spaCy models, sentence encoders, attribute LLM
uv run ngec download-index --start # the Elasticsearch index, served on port 9200
uv run ngec doctor --smoke         # checks everything, then codes three stories
```

`ngec doctor` without `--smoke` checks the pieces in a few seconds; it reports
what PyTorch can see against what the driver reports, which is how a build that
has fallen back to the CPU gets caught. `ngec update` says whether newer models
or a newer index have been published, and `--apply` updates them.

**From a clone** (to change NGEC, retrain, or run the tests), start with the
setup doctor. It is one standard-library file, runs before anything is
installed, reads the driver, and prints the `uv sync` command for this machine:

```shell
python3 setup/doctor/ngec_doctor.py            # --serve for a page with Run buttons
uv sync --extra cu12 --extra vllm              # the reference GPU install
```

The one decision is which of the `cpu` / `cu12` / `cu13` extras to give, i.e.
which CUDA generation your driver supports (`nvidia-smi` shows it in the
header). **vLLM requires `cu12`**, even on a driver new enough for CUDA 13. §4
is the why. The spaCy models come with `uv sync` in a clone: they are the
`models` dependency group, which is a default group. `DEVELOPING.md` has the
rest of the clone workflow, including macOS/`mlx` and the `llamacpp` extra.

The models `ngec download-models` fetches, into `~/.cache/huggingface` (or
`$HF_HOME`) apart from the spaCy packages:

- `ahalt/qwen3.5-event-extraction-0.8b` — the attribute LLM (about 1.7 GB), a
  fine-tuned Qwen3.5-0.8B. Prompt format v6: decoded greedily, roles returned
  as JSON lists, and `killed` / `injured` added for ASSAULT, PROTEST and
  COERCE. When llama-cpp-python is installed it also fetches the 8-bit GGUF
  (834 MB) that the `llamacpp` backend runs. `ahalt/qwen3-event-extraction-exp5.1`
  is the submitted paper's model and `ahalt/event-attribute-extractor` the
  original; both stay resolvable by name (`--attribute-model NAME` downloads
  one). `AttributeModel` picks the prompt format from the model name via
  `KNOWN_PROMPT_FORMATS` (or the model's own `ngec.json`), so pass a name or
  path it recognises rather than an arbitrary local copy.
- `sentence-transformers/all-mpnet-base-v2` — event classifier encoder, read
  from `ngec/assets/event_models_v2/metadata.json`.
- `sentence-transformers/static-retrieval-mrl-en-v1` — Wikipedia matching in
  actor resolution (`NGEC_WIKI_ENCODER` overrides it).
- `BAAI/bge-small-en-v1.5` — agent-pattern matching in actor resolution
  (`NGEC_AGENT_ENCODER` overrides it).
- spaCy `en_core_web_trf` and `en_core_web_lg`.

Agent-pattern embeddings are computed once and cached in the OS cache
directory (`~/.cache/plover/embeddings/` on Linux, via `platformdirs`).
Uninstalling the package does not remove either cache.

## 3. The smallest thing that runs

```python
import logging
from ngec.plover_coder import PloverCoder
from ngec.es_client import setup_es_client
from ngec.logging import setup_logging

setup_logging(level=logging.INFO, quiet_third_party=True)
es_client = setup_es_client(hosts=["localhost"], port=9200)

pc = PloverCoder(es_client=es_client)
events = pc.process([
    {"id": "story1",
     "event_text": "Protesters were in the streets in Paris again today to "
                   "protest against the government's austerity measures.",
     "pub_date": "2016-05-01"}
])
```

`ngec.es_client.es_client_from_env()` does the same connection from the
`ES_HOST` / `ES_PORT` / `ES_USER` / `ES_PASSWORD` settings in `.env`, and fails
at once if Elasticsearch is not answering.

Input records need `id`, `event_text`, and `pub_date`. `pub_date` matters more
than it looks: it is the reference date for every relative date ("Tuesday",
"last week") and the fallback when the text states no date at all, which is the
common case.

`PloverCoder` takes a few options worth knowing:

```python
PloverCoder(es_client=es_client,
            attribute_backend="auto",     # or "vllm" / "llamacpp" / "mlx"; see §5
            gpu=False,                    # only used by the deprecated "transformers" backend
            max_gpu_memory=0.8,           # vllm only: share of GPU memory to reserve
            save_intermediate=False,      # dump per-step JSONL for debugging
            intermediate_dir=None)        # where those files go (default: working directory)
```

`event_threshold` is left unset on purpose: each event type and mode then uses
the threshold recorded for it in `metadata.json` (§7). The customization
arguments — `event_classifier`, `attribute_model_name`,
`event_definitions_file`, `agents_file`, `priorities_file` — are documented in
the `PloverCoder` docstring and in `ngec guide customize`.

## 4. GPU, CUDA, and the torch build

This is the one piece of setup that is genuinely fiddly. The setup doctor
(`setup/doctor/ngec_doctor.py`, for a clone) and `ngec doctor` (for an
installed package) check it; this section is the why, for when you are
installing by hand or something went wrong.

There are **two** constraints on which PyTorch build you need, and they pull in
different directions. In a clone, the `cpu` / `cu12` / `cu13` extras exist so
that you only have to answer the first one. An installed package gets the same
effect by installing PyTorch from the right index before NGEC (README step 2).

**Constraint 1: the driver.** Without one of those extras, uv installs the
**default PyPI build** of PyTorch. On Linux that build is compiled against
**CUDA 13** and needs an NVIDIA driver of roughly **580 or newer**. On an older
driver it installs fine and then reports:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12060).
torch.cuda.is_available() -> False
```

Everything still runs, silently, on the CPU — the only symptom is that it is
slow.

**Constraint 2: vLLM.** vLLM ships compiled CUDA kernels of its own, and they
have to match the CUDA generation of the PyTorch they run against. **vLLM 0.20.0
is the dividing line**: from that release on, the published wheels link
`libcudart.so.13` and so require CUDA 13 (0.20.0 is also where vLLM moved to
torch 2.11). 0.19.1 is the last CUDA 12 build. `pyproject.toml` pins the extra
to `>=0.19.1,<0.20` deliberately: a CUDA 12 build runs on CUDA 13 drivers too,
so the older wheel is the one that works on the widest set of machines, and
0.19.1 is the first release that allows the transformers 5.5.1+ that Qwen3.5
needs. The cost is that it must be paired with a **CUDA 12** PyTorch, exactly
`torch==2.10.0`, even on a new driver.

If you get this pairing wrong, the failure is delayed and misleading. vLLM
imports lazily, so `import vllm; vllm.__version__` succeeds against the wrong
PyTorch and only the real work fails:

```
ImportError: libcudart.so.13: cannot open shared object file
```

Use `import vllm._C` if you want a smoke test that actually loads the extension.
In a clone, `cu13` + `vllm` and `cpu` + `vllm` are declared as conflicting
extras, so the pairing fails at install time, where the message is about the
extras rather than about a missing `.so` two model loads later. An installed
package has no such check: a CUDA 13 build of torch 2.10.0 satisfies vLLM's
version pin, so it is kept, and vLLM then fails when it loads. That is why the
README's step 2 has a separate command for vLLM.

### How the extras do it

`pyproject.toml` declares the three PyTorch indexes as `explicit` uv indexes and
uses `[tool.uv.sources]` to send `torch` to whichever one matches the extra that
was selected:

```toml
[[tool.uv.index]]
name = "pytorch-cu12"
url = "https://download.pytorch.org/whl/cu129"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu12", extra = "cu12" },
    { index = "pytorch-cu13", extra = "cu13" },
]
```

`cu129` covers every 12.x driver: CUDA's minor version compatibility means a
12.9 build runs on any driver from 525 on. `torchvision` and `torchaudio` go to
the CUDA 12 index too, but only for the `vllm` extra, since nothing else pulls
them in. `[tool.uv] conflicts` declares the three as mutually exclusive, and
`vllm` as incompatible with both `cu13` and `cpu`, so a bad combination is
refused at resolution time instead of being installed and failing later.

Two consequences worth knowing:

- **It survives `uv sync`, as long as you pass the same extra each time.** The
  build is chosen by the extra, not by a one-off reinstall, so it is not undone
  the way the older `uv pip install --torch-backend=...` recipe was. A bare
  `uv sync` without the extra goes back to the default PyPI build.
- **It only works for this project.** uv reads `[tool.uv.sources]` for the
  project it is building, not for dependencies, so `uv add ngec[cu12]` from
  another project gets NGEC without the CUDA-specific PyTorch. Installing that
  way, install PyTorch from the right index first, as the README does.

### The older recipe, and things that don't work

Before the extras existed the advice was to `uv sync` and then reinstall over
the top:

```shell
uv pip install torch --torch-backend=auto --reinstall-package torch
```

`--reinstall-package` is needed there because the `+cu129` suffix is not part of
the version uv compares, so without it uv sees 2.10.0 == 2.10.0 and does
nothing. It works, but it has to be repeated after every `uv sync`, and swapping
builds in place repeatedly left a venv half-broken — `nvidia-*` packages
belonging to the replaced build get removed or kept inconsistently, and torch
stops importing at all:

```
ImportError: libcudnn.so.9: cannot open shared object file
ImportError: libnvshmem_host.so.3: cannot open shared object file
```

If you land in that state, delete `.venv` and reinstall; it is faster than
repairing it package by package.

**Two things that look like they should work and don't.** `torch-backend` is a
persistent uv setting, so

```toml
[tool.uv]
torch-backend = "auto"
```

looks like it should make `uv sync` do the right thing. Tested on uv 0.11.13: uv
accepts the key without complaint, but `uv lock` still resolves torch from
`https://pypi.org/simple` and the sync installs the CUDA 13 build anyway. The
`UV_TORCH_BACKEND` environment variable is ignored by `uv sync` as well, and
neither `uv sync` nor `uv lock` accepts `--torch-backend` as a flag — it exists
only on `uv pip install`. Don't spend time on either a second time.

### A stale system CUDA can shadow the wheels

The PyTorch wheels ship their own CUDA libraries. If `LD_LIBRARY_PATH` points at
an older system-wide CUDA install, that one wins, and you get an error that
looks like a broken PyTorch but is not:

```
ImportError: .../libcusparse.so.12: undefined symbol: __nvJitLinkGetErrorLogSize_12_9,
version libnvJitLink.so.12
```

On the reference machine `LD_LIBRARY_PATH=/usr/local/cuda/lib64` and
`/usr/local/cuda` is **CUDA 12.0**, whose `libnvJitLink.so.12` lacks symbols the
12.9 libraries need. Clearing the variable for the run fixes it:

```shell
env -u LD_LIBRARY_PATH uv run python your_script.py
```

The setup doctor tests for this specifically: it retries its PyTorch probe
without `LD_LIBRARY_PATH` and tells you if that is what made the difference.

### Using the vllm backend

```python
PloverCoder(es_client=es_client, attribute_backend="vllm")
```

`attribute_backend="auto"` chooses vllm by itself when vLLM is installed and
PyTorch can see a CUDA GPU. `gpu=` is ignored: vllm always runs on the GPU.
Two things to know:

**Your script needs a `__main__` guard.** vLLM's v1 engine starts its core in a
spawned subprocess, which re-imports the main module. Pipeline code at module
level therefore fails with "An attempt has been made to start a new process
before the current process has finished its bootstrapping phase". Put the work
in a function called under `if __name__ == "__main__":`.

**vLLM reserves GPU memory up front**, as a fraction of the card's *total*
memory (default 0.8). On a GPU that anything else is using, startup fails
outright:

```
ValueError: Free memory on device cuda:0 (17.92/23.55 GiB) on startup is less
than desired GPU memory utilization (0.8, 18.84 GiB).
```

The attribute model is small (0.8B parameters; NGEC loads it text-only, without
Qwen3.5's vision tower), so lower it: `PloverCoder(..., max_gpu_memory=0.45)`.

## 5. Choosing the attribute backend

Attribute extraction is one LLM prompt per record from `stories_to_events`,
i.e. per story × detected event type × mode, so its cost grows with how often
the classifier fires. `attribute_backend="auto"` (the default) picks:

- `vllm` if vLLM is installed and PyTorch can see a CUDA GPU;
- `mlx` on an Apple Silicon Mac, if `mlx-lm` is installed;
- `llamacpp` otherwise. It never picks `transformers`.

The choice is logged. If the chosen backend's package is missing, loading the
model says what to install.

**`llamacpp`** is the CPU backend. It runs the model's 8-bit GGUF in-process
through llama-cpp-python (the `llamacpp` extra). PyPI has only its source, so
pip compiles llama.cpp unless pointed at the project's ready-built CPU wheels:

```shell
pip install "ngec[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

In a clone, `uv sync --extra cpu --extra llamacpp` goes to that index on its
own. The GGUF is downloaded the first time (or by `ngec download-models`);
`gguf_path=` or `NGEC_ATTRIBUTE_GGUF` points at a local file instead. Only the
default model has a published GGUF, so the older models need another backend
or a GGUF you convert yourself. Threads default to the number of performance
cores, at most 8; `NGEC_LLAMACPP_THREADS` overrides that. With
`NGEC_LLAMACPP_URL` (or `llamacpp_url=`) set, the same backend sends prompts to
a running `llama-server` instead; `DEVELOPING.md` covers starting one.

**`transformers`** is deprecated. It still works and logs a warning.

Measured on an i9-12900K (8 performance cores), with the default model on 10
VOA prompts of about 900–1500 tokens, per (story, event type) prompt:

| backend | time per prompt | memory |
|---|---|---|
| `llamacpp`, in-process, 8 threads | 4.6 s | 2.2 GB |
| `llamacpp`, `llama-server -t 8`, nothing cached | about 4.2 s | |
| `transformers`, CPU, float32 | 15.4 s | 5.2 GB |

On the same CPU the in-process engine took 7.2 s per prompt on 4 threads and
6.4 s on 16, which is why the default stops at the performance-core count. A
story with three detected event types is three prompts. vllm and mlx have not
been timed against the current model in this file. `docs/PERFORMANCE.md` covers
serving the model on a CPU host in more detail.

## 6. Running a corpus

`ngec guide run` has the full recipe: getting a CSV into shape, a script that
codes in batches and writes each batch out, and `events_to_table` for a flat
table. The mechanics worth knowing:

Input is a list of dicts, so a JSONL file is the natural format:

```python
stories = [json.loads(line) for line in open("stories.jsonl")]
events = pc.process(stories)
```

`process()` returns a **new list**, not the input mutated in place. Stories
with no detected event produce no records, and a story can produce several
(one per event type-mode pair, and one per event the attribute model finds).
`orig_id` links each event back to its story.

Everything is batched internally, so hand `process()` a few thousand stories at
a time rather than looping story by story — the spaCy pass and the classifier
embeddings are much faster in bulk. Memory grows with the batch, so for very
large corpora chunk rather than passing everything at once.

### What an early run looked like

The first full corpus run, recorded in September 2026 before the event
classifiers were retrained, used the original attribute model
(`ahalt/event-attribute-extractor`) on the `transformers` backend with
`gpu=True`, on an RTX 4090 against a local ES, over 500 Voice of America news
stories (median 429 characters):

```
init:    8.0s        (model loading, once per process)
process: 96.2s       (0.19 s/story)  ->  13 events from 500 stories
```

With so few events, almost all of that time was the spaCy `en_core_web_trf`
pass and the geoparser queries, which run over every story. The 13 events were
an artifact of the old classifiers (§7), and none of the models or the backend
are the current defaults, so neither the count nor the split of time carries
over. With the current classifiers, time the first 20 stories of your own
corpus and extrapolate, as `ngec guide run` says.

## 7. Event counts and the classifier

There is no current measurement of events per story in this file. The event
counts from the early run were produced by the original demo classifiers,
which were being served the wrong sentence encoder (trained on
`all-mpnet-base-v2`, served `paraphrase-mpnet-base-v2` — same dimensionality,
so nothing errored, but the probabilities were off the scale the models were
fit on). `CLASSIFIERS.md` describes the retrained classifiers and their
holdout scores.

One lesson from the old run still applies as a diagnostic: sweeping a single
threshold from 0.9 down to 0.5 took the old classifiers from 13 of 500 stories
firing to all 500 matching nearly every event type. When a whole ontology's
scores rise and fall in lockstep, suspect the features before the models.

Two things carry over regardless of which models you are running:

**Don't use a single threshold across event types.** They are not calibrated
alike. The F1-maximizing thresholds recorded in
`ngec/assets/event_models_v2/metadata.json` span 0.35 to 0.75 for the event
types and 0.1 to 0.8 for the modes. `PloverCoder(event_threshold=...)`
overrides all of them with one number — leave it unset unless you specifically
want that.

**`event_type_confidence` only contains types that already cleared the
threshold**, so you cannot recover the score distribution from a run's output.
Construct the classifier with a low threshold if you want to see it.

## 8. Reading the output

`ngec guide run` ("What comes out", "Counting events", "Checking the output")
describes the fields and what to check. What the early run showed, which still
describes how the pipeline behaves:

**Dates are usually the publication date.** Most news sentences do not state
when something happened, the attribute model returns no date, and the resolver
falls back to `pub_date` with `date_type="unresolved"`. That is the designed
fail-safe behaviour, not a failure — `date_type` tells you which dates were
actually read out of the text (`exact` / `approximate` / `range`) and which are
just the publication date. Filter on `date_type` before analysing dates. In
the early run, 11 of 13 events fell back this way.

**Locations are often empty even when the story names places.** `event_location`
is only filled when the attribute model's extracted location string overlaps a
place the geoparser resolved. The `reason` field says which step declined; in
the early run:

```
{'no search term': 5, 'no sufficient overlap in search terms': 4, 'success': 4}
```

`no search term` means the attribute model returned no location for the event;
`no sufficient overlap` means it named a place the geoparser did not resolve to
the same string. Both are visible rather than silent.

**An empty `code_1` on an actor is not always a failure.** A bare country name
resolves to a country code with no role code, and says so:

```python
{"actor_role_query": "Taiwan", "country": "TWN", "code_1": "", "code_2": "",
 "source": "country only"}
```

Check `source` before treating a blank code as a miss.

### Eyeballing the coded events

Reading all 13 events from the early run, the coding was broadly reasonable —
"NATO Pledges New Weapons for Ukraine" coded `COOPERATE` with recipient Ukraine
and location Brussels, "US, South Korea Hold Bigger Drills" coded `MOBILIZE`
with both states as actors. These failure modes showed up, and are worth
checking on your own data before trusting aggregate counts:

- **The `anchor_quote` sometimes echoed the codebook definition** instead of
  quoting the document ("COOPERATE: Initiate, resume, improve, or expand mutual
  material cooperation or exchange"). It happened twice in 13, with
  `ahalt/event-attribute-extractor`. It did not reproduce on the demo's eleven
  curated documents with either that model or `exp5.1` (41/41 and 27/27
  verbatim quotes, via `demo/check_extractions.py`), and it has not been
  measured on a corpus with the current default model. A quote that doesn't
  appear in `event_text` is a cheap check; keep making it.

  When you make it, **fold typographic variants first**. A model that returns a
  curly apostrophe where the article has a straight one has still quoted
  verbatim; a naive string comparison calls it a paraphrase. That mistake put a
  wrong claim in three documents in this repo before it was caught.
- **Role coding of organisations was noisy**: "UN agencies in Malawi" coded
  `JRN` (news agency), NATO coded via the pattern "naval". The agent-pattern
  matcher matched on surface strings. The agent encoder has since changed
  (now `BAAI/bge-small-en-v1.5`); these two cases have not been re-checked.
- **One event can produce several near-identical records.** `stories_to_events`
  emits one record per event type-**mode** pair, and each is sent to the
  attribute model separately, so a story classified as both
  `ASSAULT-explosives` and `ASSAULT-heavy-weapons` yields two records that can
  have the same actor, recipient, quote and location, differing only in
  `event_mode`. That is by design, but it means **counting rows is not counting
  events**. Decide explicitly whether your unit of analysis is the record or
  the (`orig_id`, `event_type`) group before reporting any aggregate.

## 9. Troubleshooting

`ngec doctor` (or, in a clone before installing, `python3
setup/doctor/ngec_doctor.py`) catches most of the setup problems below and
prints the fix.

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'click'` on `import spacy` | spaCy imports `click` but only declares `typer`, and a typer release dropped click. Fixed by the explicit `click` dependency; re-install or re-run `uv sync`. |
| `xgrammar ... doesn't have a wheel for the current platform` during `--extra vllm` | xgrammar stopped publishing wheels for newer Pythons. Pinned to `<0.2.4` in `pyproject.toml`. |
| `torch.cuda.is_available()` is `False` on a working GPU | CUDA-13 torch on a CUDA-12 driver. Install the right build (README step 2, or in a clone the extra the setup doctor recommends); see §4. |
| `ImportError: libcudart.so.13` from `vllm._C` | CUDA-13 vLLM (0.20.0+) against a CUDA-12 torch, or vice versa. The extra is pinned `<0.20`, so pair it with a CUDA 12 `torch==2.10.0`. See §4. |
| `ImportError: libcudnn.so.9` / `libnvshmem_host.so.3` on `import torch` | A half-swapped venv from changing torch builds in place. Delete `.venv` and reinstall with the extra you want. |
| `undefined symbol: __nvJitLinkGetErrorLogSize_12_9` | An old system CUDA on `LD_LIBRARY_PATH` shadowing the wheels' libraries. Run with `env -u LD_LIBRARY_PATH`. See §4. |
| `An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | The vllm backend spawns a subprocess. Put your pipeline code under `if __name__ == "__main__":`. |
| `Free memory on device cuda:0 ... is less than desired GPU memory utilization` | vLLM reserves a share of the GPU's total memory at startup and something else is using the card. Lower `PloverCoder(max_gpu_memory=...)`. |
| llama-cpp-python fails to build (CMake / compiler errors) on install | pip found only the source package. Install with `--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`; see §5. |
| `There is no published GGUF file for ...` | The in-process `llamacpp` backend with a model other than the default. Pass `gguf_path=` / set `NGEC_ATTRIBUTE_GGUF`, or use another backend. |
| `The transformers backend is deprecated` warning | You asked for `attribute_backend="transformers"`. Use `"auto"`; see §5. |
| `TypeError: Geoparser.__init__() got an unexpected keyword argument ...` | `mordecai3`'s API moved under `ngec/geolocation.py`. NGEC requires `mordecai3>=3.5.0`, and a clone pins a specific git rev in `[tool.uv.sources]`; install that version. |
| `KeyError: "['<feature>'] not in index"` in `wiki_matcher` | The XGBoost ranker in `ngec/assets/xgb_model.json` was trained with a feature that `_create_scoring_dataframe` does not build. Training-time features live in `setup/train_wiki_model/`; the two must agree. |
| `ElasticsearchConnectionError: Could not connect to Elasticsearch` when constructing `PloverCoder` | mordecai3's check in the geolocation step: nothing is answering on the configured host and port. Start it with `ngec download-index --start` (or `docker start ngec-es` if it exists), then `curl localhost:9200`. |
| `GeonamesIndexError: Connected to Elasticsearch, but the 'geonames' index was not found` | Elasticsearch is running on an empty or wrong data directory. The message suggests `mordecai3 index fetch`, which gets GeoNames only; NGEC also needs `wiki`, so use `ngec download-index` instead. |
| `ValueError: Error checking Wikipedia index: ...` when constructing `PloverCoder` | The actor resolver's check: the `wiki` index is missing, or so old it has no `redirects` field. Check `curl localhost:9200/_cat/indices`. |
| `mordecai3.geoparse - WARNING - Error getting 'next_ent'` | Harmless; the geoparser logs it for some entity positions and continues. |

## 10. Reproducing the early run

The sample was drawn from the public-domain Voice of America corpus, taking
stories between 300 and 6000 characters and mapping VOA's fields onto NGEC's:

```python
{"id": doc["filename"],
 "event_text": f'{doc["title"]}. ' + "\n".join(doc["paragraphs"]),
 "pub_date": doc["time_published"][:10]}
```

That corpus is not in the repo. `data/voa_benchmark/voa_stories.jsonl` is a
separate, smaller VOA set committed for benchmarking, with its own fields
(`title`, `text`, `published`); see its README.

Event `id`s are per-run artifacts. The run above used a model sampled at
temperature 0.5, so counts could shift by an event or two between runs. The
current default model is decoded greedily. `orig_id` is stable either way.
