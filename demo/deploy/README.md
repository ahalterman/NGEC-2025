# Keeping the demo's services up

The demo depends on long-running processes that are not part of the Streamlit
app, and they do not come back on their own after a reboot unless you tell them
to. When one is down the app says so in its "Status" sidebar, which in
CPU mode carries a `llama-server` row of its own, but the point of this
directory is that it should not happen in the first place.

| Service | What it carries | The app without it |
|---|---|---|
| Elasticsearch | the `wiki` and `geonames` indices | steps 3–5 degrade to a message |
| `llama-server` | the quantized attribute model | step 2 onwards degrades to a message |

Both matter on the CPU host the demo is destined for. The demo has a GPU/CPU
toggle; **CPU mode** serves the attribute model through `llama-server`
(the default) unless `NGEC_DEMO_CPU_BACKEND=transformers` sends
it back to loading the checkpoint in-process, which is the slow fallback. In GPU
mode the model is loaded inside the Streamlit process and Elasticsearch is the
only thing here that needs arranging.

## Elasticsearch

It runs as a Docker container here, so Docker's own restart policy is the whole
answer. Containers are created with `restart=no` by default, which is why a
reboot leaves it stopped:

```shell
docker ps -a --filter ancestor=wiki-geonames-container   # find the container
docker update --restart unless-stopped <container>
```

`unless-stopped` rather than `always`: if you stop the container deliberately, it
should stay stopped across a reboot.

## `llama-server`

Needed whenever the demo is in CPU mode, which is what the deployment host
runs. Run it as a systemd **user** unit — it needs no root and the model files
live in `~/ngec-llamacpp/`:

```shell
mkdir -p ~/.config/systemd/user
cp demo/deploy/ngec-llama-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ngec-llama-server
loginctl enable-linger $USER
```

`enable-linger` is the part that is easy to miss: without it the unit stops when
your last session ends and does not start at boot, which leaves you exactly where
you were with a backgrounded shell command.

```shell
systemctl --user status ngec-llama-server
journalctl --user -u ngec-llama-server -f
systemctl --user restart ngec-llama-server     # after rebuilding the GGUF
```

The unit pins the settings the measurements in `demo_old/DESIGN.md` were taken with —
`attr-exp5.1-q8.gguf`, 8192 context, 4 threads, bound to localhost. Rebuilding
the model to a different quantization means editing `ExecStart`, and it means
re-validating: Q4_K_M is faster and drifts further.

**Keep the GGUF and `NGEC_ATTRIBUTE_MODEL` in step.** The server supplies the
weights; the Python side supplies the tokenizer and, through it, the prompt
format the model expects. Serving `attr-q8.gguf` (the original
`ahalt/event-attribute-extractor`, legacy prompt format) while the app prompts
for `exp5.1` produces valid JSON and worse extractions, with no error anywhere.
The app's health check compares the two names and reports a mismatch in the
status sidebar, but it is easier not to create one.

It binds to `127.0.0.1` deliberately. Nothing outside the host should be able to
submit generation jobs to it.

## Checking

```shell
curl -s localhost:9200/_cat/indices?v          # wiki and geonames, with doc counts
curl -s localhost:8080/health                  # {"status":"ok"}
curl -s localhost:8080/v1/models               # which GGUF is actually being served
```

The app's own sidebar reports these under "Status" — Elasticsearch and
its two indices always, and the `llama-server` row whenever the demo is in CPU
mode.
