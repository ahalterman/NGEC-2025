# The NGEC setup doctor

```shell
python3 setup/doctor/ngec_doctor.py            # a checklist for this machine
python3 setup/doctor/ngec_doctor.py --json     # the same report, machine-readable
python3 setup/doctor/ngec_doctor.py --serve    # a local page with Run buttons
```

One file, `ngec_doctor.py`, standard library only, Python 3.8 and up. It runs
*before* NGEC is installed, which is the whole point: it works on a bare conda
base that cannot `import ngec`, and it tells you which of the mutually exclusive
`cpu` / `cu12` / `cu13` extras this machine's driver actually needs — the
decision the README currently asks you to make by hand, and the one that
silently leaves you on the CPU when you get it wrong.

Anything it needs to know about the *installed* package it asks by running
`.venv/bin/python -c ...` as a subprocess, so it never imports it.

## Not the same tool as `ngec-doctor`

|  | `setup/doctor/ngec_doctor.py` | `ngec-doctor` (`ngec/doctor.py`) |
|---|---|---|
| Question | Is the environment **there** yet? | Is the environment **right**? |
| Needs | Python 3.8, nothing else | an installed `ngec` |
| Covers | OS, uv, driver, extras, Docker, Elasticsearch, indices, model cache, disk | version and commit, every env var, the PyTorch build in depth |

Run this one first; run `uv run ngec-doctor` once the package imports.

## What it checks

Six groups, in the order you have to satisfy them:

- **Machine** — OS and architecture, RAM, free disk at the repo and at Docker's
  data root (the index needs about 25 GB).
- **Python** — `python3` on PATH and its version (and whether it is a conda
  environment), `uv`, the project `.venv`, `import ngec`, which PyTorch build is
  installed and whether it can see the GPU, and the two spaCy models.
- **GPU** — the NVIDIA driver, the highest CUDA version it supports, and
  therefore which extra to pass; and whether `LD_LIBRARY_PATH` carries a system
  CUDA that will shadow the PyTorch wheels' own libraries.
- **Elasticsearch** — Docker and its daemon, a node answering on the host from
  `.env`, and `wiki` / `geonames` with document counts compared against the
  expected 7,601,204 and 13,250,817, plus how old each index is. This is the
  group that matters most: it is the longest and most failure-prone part of the
  install. The fixes are the two paths in
  [`elasticsearch/SETUP.md`](../../elasticsearch/SETUP.md).

  The age check warns at `INDEX_MAX_AGE_DAYS = 183` — six months — because both
  indices are snapshots of sources that keep moving and neither updates itself:
  an old `wiki` misses new articles and keeps the former names of renamed
  entities, an old `geonames` misses new gazetteer entries. The date is read
  from the mapping's `_meta` (`build_date`, then `dump_date`), which the loaders
  under `elasticsearch/` stamp when a load finishes and which travels with the
  index; failing that, from `index.creation_date`, which is when the index was
  created *on this machine* — the build date for a data directory unpacked from
  a pre-built tarball, only the restore date for a snapshot restore. The detail
  line says which of the two it used. Indices built before the loaders started
  stamping `_meta` only have the second. The fix points at
  `elasticsearch/SETUP.md` and is deliberately not runnable: a refresh is a
  download or a rebuild, not something a console should launch.
- **Models** — the attribute model and the three sentence encoders in the
  Hugging Face cache. The encoder for the event classifier is read out of
  `ngec/assets/event_models_v2/metadata.json` rather than hard-coded, because
  the models are self-describing.
- **Services** — `llama-server`, which is optional and only used by the
  `llamacpp` backend on a CPU host.

Every failed check carries the exact command for *this* OS, driver and directory
layout, why it is needed, and a time estimate where one can honestly be given.
Download estimates come from a five-second throughput sample against
huggingface.co (`--no-network` skips it; offline it says "not estimated" rather
than guessing). Index-build estimates are the durations quoted in
`elasticsearch/README.md`, which gives durations and not throughput.

## `--serve`

A `http.server` on `127.0.0.1:8765` (`--port` to change it) serving one page:
the same checklist, grouped, with each unmet check's command, a **Run** button
that streams the command's output into the card over server-sent events and
re-runs the checks when it exits, and a **Copy** button.

Three rules, enforced on the server side, not in the page:

1. It runs only commands the doctor itself generated, looked up **by id**. An
   id that is not in the current table is refused.
2. It never runs anything containing `sudo`, and never anything the doctor
   marked as needing a human (installing Docker, starting a system daemon,
   the `env -u LD_LIBRARY_PATH` prefix, a command still containing `TODO`).
3. It strips `LD_LIBRARY_PATH` from the environment of everything it runs.

It binds to the loopback interface and has no authentication, which is fine for
a page that only runs a fixed table of commands as you, and would not be if
either of those changed.

## Known gaps

- **`PREBUILT_INDEX_URL` is `"TODO"`.** Nothing in this repository records where
  the published index tarball lives. The constant is at the top of
  `ngec_doctor.py`; fill it in and the pre-built-index fix becomes runnable. The
  file size used for the download estimate (about 9.4 GB) comes from a local
  copy of the packaged index, not from the published artifact.
- The vllm first-run compile and the PyTorch wheel install are quoted as ranges
  from the reference box. They do not scale with anything the doctor can
  measure, so they are not estimated live.
- There is no final smoke test (code one sentence, compare with a stored
  record). `tests/test_end_to_end.py` is that test today.

## Driving it from Claude Code

`.claude/skills/ngec-setup/SKILL.md` is a skill that reads `--json`, walks the
unmet checks in order, explains each one, and runs fixes only after you confirm.
The doctor is what makes that reliable: the model reads structured findings
instead of guessing from error messages.
