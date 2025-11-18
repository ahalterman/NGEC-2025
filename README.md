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