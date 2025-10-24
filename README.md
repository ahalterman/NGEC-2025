# NGEC

*This is a temporary pre-release version of the code. It's not yet stable and I do not recommend you use it right now.*


## Misc. notes

To setup a local environment for package development with `uv`, this command 
should work:

```bash
uv venv --python 3.13  # or another version >=3.10
uv sync --extras models
```

To verify, find the `ngec` entry in `uv.lock`, it should include this key:

```
source = { editable = "." }
```