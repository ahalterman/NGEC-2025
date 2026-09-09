---
name: ngec-setup
description: Install, set up, repair or diagnose an NGEC environment on this machine - the venv and PyTorch extra, Elasticsearch with the wiki and geonames indices, the spaCy and Hugging Face models. Use whenever the user asks to install NGEC, get it running, fix a broken install, or work out why a pipeline step is failing for environmental reasons.
---

# Setting up NGEC

Work from the setup doctor's findings, not from guesses. It is standard-library
only and runs on any Python 3.8+, so it works before anything is installed.

## The loop

1. Run `python3 setup/doctor/ngec_doctor.py --json` from the repo root.
2. Read `checks`. They are already in dependency order (Machine, Python, GPU,
   Elasticsearch, Models, Services). Fix them in that order — a failing check
   often makes later ones meaningless.
3. Summarise for the user: what is in place, what is not, roughly how long the
   rest will take (each `fix.estimate`).
4. For the first unmet check: say what it is, why it matters (`fix.why`), and
   show `fix.command`. Then **ask before running it**. Do not batch fixes.
5. Run it only after the user confirms. If `fix.runnable` is `false`, do not run
   it at all — it needs a human (installing Docker, starting a daemon, a `TODO`
   placeholder, a shell prefix rather than a command). Give it to the user to
   run themselves and wait.
6. Re-run the doctor. Report what changed. Move to the next unmet check.

Stop and report when `summary.failed` is 0, or when the next fix needs something
you cannot do.

## Hard rules

- **Never `sudo`.** Not for Docker, not for anything. Hand those to the user.
- **Never `uv pip …` without `--python .venv/bin/python`.** Bare `uv pip` on
  this machine resolves against the anaconda base, not the project venv.
- **Pass the same extras to every `uv run` and `uv sync` in a session.** `uv run`
  re-syncs the environment, so a call with different extras silently rebuilds
  the venv with a different PyTorch. Use the doctor's `recommended_extra`; on
  this box that is `uv run --extra cu12 --extra vllm …`.
- **Exactly one of `cpu` / `cu12` / `cu13`.** They are declared as conflicting
  in `pyproject.toml`, and `cu13 + vllm` is forbidden (the pinned vllm is a CUDA
  12 build). Without one of them, uv installs the default PyPI CUDA 13 build,
  which falls back to the CPU with no error.
- **`env -u LD_LIBRARY_PATH` whenever the doctor flags `LD_LIBRARY_PATH`.**
  Prefix every command that loads torch. The symptom otherwise is an undefined
  symbol such as `__nvJitLinkGetErrorLogSize_12_9`, which never mentions the
  path.
- **Do not touch a running Elasticsearch container** unless the user asks. Two
  Elasticsearch nodes against one data directory corrupt it.
- Never edit `pyproject.toml`, `.env` or the doctor to make a check pass.

## Elasticsearch is the long pole

`wiki` and `geonames` live in one data directory served by one Elasticsearch
7.10.1 node. Two routes, both in `elasticsearch/SETUP.md`:

- **Download the pre-built data directory** and `docker run` over it. Minutes.
  Prefer this. The download URL is not yet recorded anywhere
  (`PREBUILT_INDEX_URL` in the doctor is `"TODO"`), so this fix is not runnable
  until someone fills it in — say so plainly rather than inventing a URL.
- **Build both indices** from a Wikipedia dump and the GeoNames gazetteer. Over
  30 minutes for geonames, about a day for wiki.

An index that is present but far short of 7,601,204 (wiki) or 13,250,817
(geonames) documents is a load that died part-way, not a working install.

## Afterwards

Once `import ngec` works, `uv run --extra cu12 ngec-doctor` is the other doctor:
it checks configuration and the PyTorch build from inside the environment.
`uv run pytest` is the fast test suite; `tests/test_end_to_end.py` needs
Elasticsearch.
