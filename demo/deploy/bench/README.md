# Measuring the attribute model on a CPU host

Tools for answering "is this setting actually faster on *my* box?" for the
`llamacpp` backend. The CPU section of `docs/PERFORMANCE.md` says what these
measured on one server and which conclusions are likely to travel.

They exist because **`llama-bench` measures the wrong shape for this workload.**
It times a 512-token prefill and a 64-token generation. The real pattern is a
~430-token prompt of which ~90% is already in the server's cache, and a
generation that is usually `[]` and occasionally 60 tokens. Conclusions invert
between the two: `--threads-batch 8` looks free on `llama-bench` and is
measurably slower on real prompts.

Standard library only, except `make_prompts.py` and `torch_threads.py`, which
import the package.

| file | what it does |
|---|---|
| `prompts.json` | 27 extraction prompts, committed so the rest runs without loading a model. See the note at the bottom. |
| `schema.json` | `ATTRIBUTE_SCHEMA`, so replays send the same `json_schema` the pipeline sends. |
| `make_prompts.py` | Rebuilds `prompts.json` through `AttributeModel.make_prompt()`. |
| `replay.py` | Replays the prompts sequentially and splits wall time into prefill / generation / other. The main measurement. |
| `replay_par.py` | The same with N requests in flight, for the concurrency question. |
| `accuracy_cmp.py` | Compares two servers' outputs under greedy decoding, by parsed field set rather than by string. |
| `run_config.sh` | Starts one server with given flags, replays, shuts it down. |
| `ab_test.sh` | Alternates two builds on identical weights. |
| `tb_test.sh` | Three-way thread-allocation comparison. |
| `acc_run.sh` | Brings up two builds and runs `accuracy_cmp.py`. |
| `torch_threads.py` | Thread sweep for the PyTorch encoders (the event classifier and the agent matcher). |

## Running

```shell
cd demo/deploy/bench

python3 replay.py 8080 baseline 2                     # against a running server
./run_config.sh mycfg --ctx-size 8192 --threads 4     # start a server, replay, stop
./acc_run.sh                                          # do two builds agree?

for t in 1 2 4 6 8; do
    uv run --extra models --extra cpu --group demo-app python torch_threads.py $t
done
```

Paths are overridable: `NGEC_LLAMACPP_BIN`, `IK_LLAMACPP_BIN`, `NGEC_GGUF`.

## Validating a quantization

`DESIGN.md` says anyone using this for data production rather than
demonstration should validate their quantization first. `accuracy_cmp.py` is how:
serve the candidate on one port and the reference on another, then run it.

It decodes greedily, so sampling randomness is removed and any difference is the
model, and it compares the **parsed `actor` / `recipient` / `date` / `location`
fields** rather than the response text. That distinction matters. Greedy
decoding is chaotic, so two responses can differ in most of their tokens and
still mean exactly the same thing — which is why `DESIGN.md`'s existing
12-prompt exact-match table cannot separate a real quality change from noise.
Q8_0, the shipped setting, scores 5/12 on it.

27 prompts is enough to catch a badly broken quantization and **not** enough to
certify a subtle one. For that, rebuild `prompts.json` from a few hundred
documents first.

## Two things that will mislead you

**A single run is worth about ±7s on a 45s measurement** — enough to reverse a
ranking. `ab_test.sh` and `tb_test.sh` alternate the arms against servers that
are all loaded at once, so drift hits every arm equally. Do the same for any new
comparison, and do not compare a run today against a number from last week.

**Check who else is on the box.** A second tenant is invisible in the numbers
and will quietly invert them.

```shell
ps -eo pcpu,rss,etime,args --sort=-pcpu | head
```

## A note on `prompts.json`

It is three documents against nine event types each. That is deliberately
denser than reality — the classifier usually fires three or four types for a
document — because it stresses the prompt cache the same way a real run does
while giving a longer and steadier measurement. It is a benchmark input, not a
sample of pipeline output. Rebuild it with `make_prompts.py` for something
closer to your own corpus.
