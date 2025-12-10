# NGEC

*This is a temporary pre-release version of the code. It's not yet stable and I do not recommend you use it right now.*

## Installation

If you are not using `uv` to install `ngec`, the `mordecai3` install will not work correctly. In that case:

1. Manually install `mordecai3` from GitHub into whatever virtual environment you are using.
2. Install `ngec`. 

### spacy models

ngec depends on the spacy `en_core_web_lg` and `en_core_web_trf` models, which are delivered as non-standard Python pacakges. 

To attempt to install them alongside the package, use the `models` extra:

```python3
uv add ngec[models]
```

### Inference backend

There are different options for the LLM inference backend. The most basic one, but also slowest is `"transformers"`, which is installed by default. 

For Windows and Linux users, especially with CUDA, install vLLM, which can be done via an extra:

```python3
uv add ngec[models,vllm]
```

macOS users can try to use `"mlx"` by installing the corresponding extra:

```python3
uv add ngec[models,mlx]
```


## Misc. notes

To setup a local environment for package development with `uv`, this command 
should work:

```bash
uv venv --python 3.13  # or another version >=3.10
uv sync --extra models --group dev
```

To verify, find the `ngec` entry in `uv.lock`, it should include this key:

```
source = { editable = "." }
```

### Tests against external ES instance

Create a `.env` file with the ES credentials. 


## Logging

Some of the third-party dependencies have very verbose loggers by default. To quiet those:

```python
from ngec.logging import quiet_third_party_loggers

quiet_third_party_loggers()
```

There is also a more general helper function included that can do this as well:

```python
import logging
from ngec.logging import setup_logging

setup_logging(
    level=logging.DEBUG,
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    quiet_third_party=True
)
```