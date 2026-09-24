# Development Notes

## Sync all dependencies

E.g. after dependencies have been added or removed. 

To sync the local venv using `uv` with all possible dependencies (and assuming the appropriate pytorch has been installed seperately if needed):

```shell
uv sync --extra mlx
```

Or on Linux with GPU and vLLM:

```shell
uv sync --extra vllm
```

The two spaCy models come with these: they are the `models` dependency group,
which is a default group, so a bare `uv sync` installs them and no later sync
prunes them. Note that dependency groups other than the default ones *are*
pruned by the next sync, so keep `--group demo-app` / `--group es-build` on
every sync once you are using them.

## Testing

Substantive tests, which test the correctness of data inputs and which fail at some fraction, are skipped by default. To run the substantive tests:

```shell
# only run substantive tests (like 300+)
uv run pytest -m substantive   

# run ALL tests, including substantive
uv run pytest -m ""            
```


## Tests against external ES instance

Create a `.env` file with the ES credentials. 


## llama.cpp backend

`backend="llamacpp"` talks to a *running* `llama-server` over HTTP — this
repo never builds, quantizes, or launches one itself. For building llama.cpp
and quantizing a checkpoint to GGUF, see the "Run locally on CPU" section of
`demo/README.md`; `demo/deploy/README.md` covers the systemd unit for a
deployed host.

The default attribute model is published as a Q8_0 GGUF, so there is nothing
to convert. Download it, and install llama.cpp (on a Mac, `brew install
llama.cpp`):

```shell
uv run hf download ahalt/qwen3.5-event-extraction-0.8b-GGUF \
    qwen3.5-event-extraction-0.8b-Q8_0.gguf --local-dir models
```

To run the server, with `-t` set to the number of performance cores (letting
llama.cpp use every core made this model much slower on a hybrid CPU):

```shell
llama-server -m models/qwen3.5-event-extraction-0.8b-Q8_0.gguf --port 8080 -c 8192 -t 8
```

To try it out:

```python
from ngec import AttributeModel

am = AttributeModel(backend="llamacpp",
                    model_name="ahalt/qwen3.5-event-extraction-0.8b",
                    llamacpp_url="http://127.0.0.1:8080")  # or set NGEC_LLAMACPP_URL
```

