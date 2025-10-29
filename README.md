# NGEC

*This is a temporary pre-release version of the code. It's not yet stable and I do not recommend you use it right now.*

## Installation

If you are not using `uv` to install `ngec`, the `mordecai3` install will not work correctly. In that case:

1. Manually install `mordecai3` from GitHub into whatever virtual environment you are using.
2. Install `ngec`. 

## Misc. notes

To setup a local environment for package development with `uv`, this command 
should work:

```bash
uv venv --python 3.13  # or another version >=3.10
uv sync --extra models
```

To verify, find the `ngec` entry in `uv.lock`, it should include this key:

```
source = { editable = "." }
```