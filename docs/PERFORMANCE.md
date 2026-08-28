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
