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

(Andy): to set up the llama.cpp server on my macbook, I had to clone the 
llama.cpp repo to get access to a tool for converting the model from HF.

```shell
uv run hf download ahalt/qwen3-event-extraction-exp5.1 \
    --local-dir models/qwen3-event-extraction-exp5.1
NGEC_DIR=$PWD
cd ~/projects
git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp
LLAMA_DIR=$PWD/llama.cpp
cd $NGEC_DIR
uv run --with gguf --with safetensors --with sentencepiece python "$LLAMA_DIR"/convert_hf_to_gguf.py \
    models/qwen3-event-extraction-exp5.1 \
    --outfile attr-exp5.1-bf16.gguf --outtype bf16

llama-quantize attr-exp5.1-bf16.gguf attr-exp5.1-q8.gguf Q8_0
rm attr-exp5.1-bf16.gguf
```

(I installed llama.cpp with homebrew otherwise, did not build from the cloned
repo.)

To run the server:

```shell
llama-server -m attr-exp5.1-q8.gguf --port 8080 -c 8192 -t 16 
```

To try it out:

```python
from ngec import AttributeModel

am = AttributeModel(backend="llamacpp",
                    model_name="ahalt/qwen3-event-extraction-exp5.1",
                    llamacpp_url="http://127.0.0.1:8080")  # or set NGEC_LLAMACPP_URL
```

