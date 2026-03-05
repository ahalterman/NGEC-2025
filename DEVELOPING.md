# Development Notes

## Sync all dependencies

E.g. after dependencies have been added or removed. 

To sync the local venv using `uv` with all possible dependencies (and assuming the appropriate pytorch has been installed seperately if needed):

```shell
uv sync --extra models --group dev --extra mlx
```

Or on Linux with GPU and vLLM:

```shell
uv sync --extra models --group dev --extra vllm
```

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