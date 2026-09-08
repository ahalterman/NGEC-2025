# Improving performance

## Choosing a PyTorch build

`ngec` depends on PyTorch, so installing `ngec` installs whatever PyTorch build
your package manager picks by default. For most people that is the right build
and there is nothing to do here. There are a few situations where it isn't, and
the symptom is almost always the same: everything works, but slowly, with no
error to tell you why.

### What you get by default

The default build depends on your platform:

| Platform | Default PyPI build | Usually right? |
| --- | --- | --- |
| macOS (Apple Silicon) | CPU + MPS | Yes — nothing to change |
| macOS (Intel) | CPU only | Yes — MPS needs Apple Silicon |
| Linux, x86-64 | Bundled with a specific CUDA version | Only if your driver matches |
| Windows | CPU, unless you install a CUDA build explicitly | Only if you have no NVIDIA GPU |

### Check what you actually have

```shell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The three values are the PyTorch version, the CUDA version it was compiled
against (`None` for a CPU-only build), and whether it can actually reach a GPU
right now. On Apple Silicon, check `torch.backends.mps.is_available()` instead.

If you have an NVIDIA GPU and the third value is `False`, that is the problem
this section exists to solve. To see the driver's own view:

```shell
nvidia-smi
```

### Installing a different build

With `uv`, let it detect the right build for your driver:

```shell
uv pip install torch --torch-backend=auto --reinstall-package torch
```

Or ask for a specific one — `cpu` for the no-GPU case, or a CUDA version such
as `cu126` or `cu128`:

```shell
uv pip install torch --torch-backend=cpu --reinstall-package torch
```

With `pip`, select the build through PyTorch's own index:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

### Two things that will trip you up

`--reinstall-package torch` is not optional. The CUDA suffix is not part of the
version number `uv` compares, so without it `uv` sees a version that already
matches and does nothing at all — silently, and with a success message.

**`uv sync` reverts this.** Syncing reinstalls the default build from PyPI, so
re-run your `uv pip install` afterwards. If you have just synced and things
suddenly got slow, this is why.

## Serving the attribute model on a CPU host

Step 4 (attribute extraction) is an LLM, and on a machine with no GPU it is the
most expensive thing in the pipeline by a wide margin — more so since the
Wikipedia linking work, which took step 5 down by two orders of magnitude. The
`llamacpp` backend exists for this case: it serves a quantized GGUF through
`llama-server` instead of loading the checkpoint in-process. `demo/DESIGN.md`
covers why, and `demo/deploy/README.md` covers running it as a service.

This section is about the settings, and its main message is that most of the
obvious ones do not help. The measurements below come from one 8-vCPU AVX2
cloud server; the *shape* of the results should travel, the exact numbers will
not. `demo/deploy/bench/` is a harness for re-running all of it on your own
machine, which is the only way to get a number you should act on.

### Why the settings behave strangely

Serving one extraction splits into two phases that are limited by **different**
resources, and nearly everything below follows from that.

**Prefill** — reading the prompt — is limited by arithmetic. It does the whole
prompt in one pass, so it uses the processor hard and gets faster with more
threads.

**Generation** — writing the answer one token at a time — is limited by memory
bandwidth. Every token requires reading every weight in the model once, and on
the measured box that came to about 10 GB/s while using only a few percent of
the processor's arithmetic capacity. The cores are mostly idle, waiting on
memory.

Generation is the larger share of a request (roughly 60% on the measured box),
so it is generation that sets the overall speed.

### Threads

The consequence is that the two phases want different thread counts, and the
useful number for generation is **much lower than the core count**:

| threads | prefill tok/s | generation tok/s |
|---|---:|---:|
| 2 | 56 | 14 |
| 4 | 108 | 22 |
| 6 | 131 | 22 |
| 8 | 141 | 22 |

Prefill keeps scaling; generation stops improving at about four threads. The
six- and eight-thread generation figures also carry error bars several times
wider than the four-thread one, which is its own signal on a shared cloud vCPU:
past the point where more threads help, they mostly add variance. Handing the
server every core is not wrong so much as pointless, and it takes cores away
from the rest of the pipeline, which runs in the same process while generation
is in flight.

`llama.cpp` has a `--threads-batch` flag that runs prefill on more threads than
generation, which looks like exactly the right tool here. **It made things
slower.** The reason is worth knowing, because it is a trap the standard
benchmark sets: `llama-bench` prefills 512 tokens at a time, but with the
server's prompt cache holding the shared document prefix, real prefill batches
are about 37 tokens. At that size, waking and synchronising eight threads costs
more than the work they do. The scaling in the table above is real and simply
does not apply to the batches this workload produces.

### Send one request at a time

The pipeline calls the attribute model once per event-type record, and sending
those concurrently looks like free throughput. It is not:

| requests in flight | total | prompt tokens evaluated |
|---|---:|---:|
| 1 | 39.9s | 1103 |
| 2 | 43.8s | 1637 |
| 4 | 48.8s | 2232 |

The third column is the explanation. `llama-server` opens several slots, and the
speedup on a document with many event types comes entirely from the **prompt
cache**: every event type for one document shares a long prefix, so only the
event definition at the end has to be evaluated. Sequential requests all land on
the slot already holding that prefix. Concurrent requests are spread across
slots holding different prefixes, so the document is re-read once per slot. The
extra prefill costs more than the batching saves.

This is also why the v5 prompt puts the document *before* the event type: it
makes the shared part a prefix, which is the only part a cache can reuse.

### Two things that sound promising and are not

**Speculative decoding.** Recent `llama.cpp` can draft tokens from n-grams
already in the context (`--spec-type ngram-*`), with no draft model. The
workload looks ideal — the model is told to copy exact spans out of a document
that is sitting in the context window. It was slower in every variant. Drafting
pays off when verifying several tokens at once is cheaper than generating them
one at a time, and on a CPU a verification pass costs nearly a full forward
pass. Every rejected draft is arithmetic spent for nothing, and arithmetic is
what generation is short of.

**Batching generally.** The same reasoning explains the concurrency result
above, and it applies equally to running the server with `--parallel`. Trading
extra arithmetic for fewer sequential steps is the right trade on a GPU, where
generation is starved of memory bandwidth and the arithmetic is free. On a CPU
it runs backwards.

### Quantization

Because generation is bandwidth-bound, its speed is close to proportional to
how many bytes the weights occupy — which is why the quantized GGUF is such a
large win over the fp32 checkpoint in the first place, and why going further
keeps paying.

It also keeps costing. `DESIGN.md` ships Q8_0 rather than the faster Q4_K_M
because the drift was visibly worse, and anything below Q8_0 moves further from
the reference. If you quantize further, validate it: `demo/deploy/bench/`
includes `accuracy_cmp.py`, which compares two servers under greedy decoding on
the **parsed attribute fields** rather than on the response text. That
distinction matters, because greedy decoding is chaotic enough that two
responses can differ in most of their tokens and still say the same thing.
