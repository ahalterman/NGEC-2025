#!/usr/bin/env python3
"""NGEC setup doctor: what this machine still needs before NGEC will run.

    python3 setup/doctor/ngec_doctor.py            # a checklist
    python3 setup/doctor/ngec_doctor.py --json     # the same thing, machine-readable
    python3 setup/doctor/ngec_doctor.py --serve    # a local page with Run buttons

This is the *setup* doctor. It runs BEFORE the package is installed, so it
imports nothing outside the Python standard library and works on any Python
from 3.8 up, including a conda base that cannot import `ngec`. Anything it
needs to know about the installed package it asks by running
`.venv/bin/python -c ...` as a subprocess.

It is not the same tool as `ngec-doctor` (`ngec/doctor.py`), which runs *after*
installation, inside the environment, and reports on configuration and the
PyTorch build in much more depth. Rough division of labour:

    setup/doctor/ngec_doctor.py   is the environment there yet?   no imports
    ngec-doctor                   is the environment right?       needs ngec

Every check returns {name, ok, detail, fix, group}. `fix` carries the exact
command for *this* OS, driver and directory layout, plus why it is needed and
a time estimate where one can honestly be given. The console (--serve) runs
only commands that appear in that table, by id, and never anything with sudo.
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# Facts about this project. Everything a future version of NGEC could change
# is gathered here rather than buried in a check.
# --------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: THE DOWNLOAD URL FOR THE PRE-BUILT INDEX IS NOT KNOWN.
# Nothing in the repo or on the reference machine records where the published
# tarball lives, so this is deliberately a placeholder rather than a guess. The
# console refuses to run any command containing "TODO", so the pre-built-index
# fix is shown and copyable but not runnable until someone fills this in.
# README.md currently points at https://andrewhalterman.com/files/... — that URL
# has NOT been verified from here; treat it as a lead, not an answer.
PREBUILT_INDEX_URL = "TODO"

# Size of the packaged index, from the local artifact
# ~/wiki_es_docker/wiki_index_data.tar.gz (10,045,172,774 bytes, July 2025).
# The published tarball may differ; the extracted data directory is ~13 GB.
PREBUILT_INDEX_BYTES = 10045172774
PREBUILT_INDEX_DIRNAME = "geonames_index"   # top level inside the 2023 tarball
ATTRIBUTE_MODEL_BYTES = 1200000000          # ~1.2 GB of safetensors
DISK_NEEDED_GB = 25                         # tarball + extracted index

ES_IMAGE = "elasticsearch:7.10.1"
EXPECTED_DOCS = {"wiki": 7601204, "geonames": 13250817}   # counts on the reference box

# Six months, the point at which the PI wants to be told the data is old. Both
# indices are snapshots of a source that keeps moving, and neither updates
# itself, so age is the only thing that says whether they still resemble it.
INDEX_MAX_AGE_DAYS = 183

# What actually goes stale, per index, said once so the check and the fix agree.
STALE_COST = {
    "wiki": "Actor resolution matches against Wikipedia titles, redirects and infoboxes, so "
            "an old index has none of the articles written since it was built and still "
            "carries the former names of entities that have since been renamed.",
    "geonames": "Geolocation can only return a place that is in the gazetteer, so an old "
                "index misses every geonames entry added since it was built.",
}

ATTRIBUTE_MODEL = "ahalt/qwen3-event-extraction-exp5.1"   # ngec/attribute_model.py
WIKI_ENCODER = "sentence-transformers/static-retrieval-mrl-en-v1"  # actors/common.py
AGENT_ENCODER = "BAAI/bge-small-en-v1.5"                  # actors/common.py
FALLBACK_CLASSIFIER_ENCODER = "sentence-transformers/all-mpnet-base-v2"

# A large public file, used only to measure download throughput for a few
# seconds. Nothing is kept.
SPEED_TEST_URL = ("https://huggingface.co/sentence-transformers/"
                  "all-mpnet-base-v2/resolve/main/model.safetensors")

GROUPS = ["Machine", "Python", "GPU", "Elasticsearch", "Models", "Services"]

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

# Filled in as checks run: {fix_id: {"command":..., "why":..., "runnable":...}}
FIXES = {}
_SPEED = {"sampled": False, "bytes_per_sec": None}
_NO_NETWORK = False


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def run(cmd, timeout=20, env=None):
    """Run a command, return (returncode, combined output). Never raises."""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout, env=env)
        return proc.returncode, proc.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)


def http_get(url, timeout=4):
    """GET a URL, return the decoded body or None."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:                                        # noqa: BLE001
        return None


def fix(fix_id, command, why, estimate="", runnable=True):
    """One remedy: the command for this machine, why, and how long it takes.

    `runnable` is False for anything the console must not run for the user --
    a command needing sudo, a shell prefix rather than a command, or one still
    carrying a placeholder.
    """
    return {"id": fix_id, "command": command, "why": why,
            "estimate": estimate, "runnable": runnable}


def check(name, ok, detail, group, fix=None, level=None):
    """One finding. `level` is 'ok', 'warn', 'fail' or 'info'."""
    if level is None:
        level = "ok" if ok else "fail"
    row = {"name": name, "ok": bool(ok), "detail": detail,
           "group": group, "level": level, "fix": None}
    if fix is not None:
        fix_id = fix["id"]
        FIXES[fix_id] = {"command": fix["command"], "why": fix["why"],
                         "runnable": fix.get("runnable", True)}
        row["fix"] = {"id": fix_id, "command": fix["command"], "why": fix["why"],
                      "runnable": fix.get("runnable", True),
                      "estimate": fix.get("estimate", "")}
    return row


def read_dotenv():
    """The repo-root .env, as a dict. The doctor never exports these."""
    values = {}
    path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        return values
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return values


ENV = read_dotenv()


def es_url():
    """Where Elasticsearch should be, honouring .env and the environment."""
    explicit = os.environ.get("NGEC_ES_URL") or ENV.get("NGEC_ES_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("ES_HOST") or ENV.get("ES_HOST") or "localhost"
    port = os.environ.get("ES_PORT") or ENV.get("ES_PORT") or "9200"
    return "http://" + host + ":" + str(port)


def venv_python():
    """Path to the project venv's interpreter, whether or not it exists."""
    if IS_WINDOWS:
        return os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
    return os.path.join(REPO_ROOT, ".venv", "bin", "python")


def venv_run(code, timeout=120):
    """Run a snippet in the project venv. Returns (rc, output, stripped_ld).

    A stale system CUDA on LD_LIBRARY_PATH shadows the libraries shipped inside
    the PyTorch wheels, and the symptom is an ImportError about an undefined
    symbol rather than anything mentioning the path. So: try with the inherited
    environment first, and if that fails, try again without LD_LIBRARY_PATH. If
    the second attempt works, the path is the problem and we can say so.
    """
    python = venv_python()
    if not os.path.exists(python):
        return 127, "no .venv", False
    rc, out = run([python, "-c", code], timeout=timeout)
    if rc == 0 or "LD_LIBRARY_PATH" not in os.environ:
        return rc, out, False
    clean = dict(os.environ)
    clean.pop("LD_LIBRARY_PATH", None)
    rc2, out2 = run([python, "-c", code], timeout=timeout, env=clean)
    if rc2 == 0:
        return rc2, out2, True
    return rc, out, False


def last_line(text):
    return text.splitlines()[-1] if text else "failed"


def ld_note(stripped):
    """Said of a probe that only succeeded once LD_LIBRARY_PATH was removed."""
    return "  (only with LD_LIBRARY_PATH removed)" if stripped else ""


def gb(num_bytes):
    return round(num_bytes / (1024.0 ** 3), 1)


def human_time(seconds):
    if seconds < 90:
        return "about " + str(int(seconds)) + "s"
    if seconds < 5400:
        return "about " + str(int(round(seconds / 60.0))) + " min"
    return "about " + str(round(seconds / 3600.0, 1)) + " h"


def download_estimate(num_bytes):
    """Time to download `num_bytes`, from a live throughput sample.

    Sampled once per run, and only when something actually needs downloading,
    because it costs five seconds and some bandwidth. Offline, it says so
    rather than inventing a number.
    """
    if _NO_NETWORK:
        return "not estimated (--no-network)"
    if not _SPEED["sampled"]:
        _SPEED["sampled"] = True
        _SPEED["bytes_per_sec"] = sample_throughput()
    rate = _SPEED["bytes_per_sec"]
    if not rate:
        return "not estimated (no connection to huggingface.co)"
    mbps = rate * 8 / 1e6
    return (human_time(num_bytes / rate) + " at the measured "
            + str(round(mbps, 1)) + " Mbit/s")


def sample_throughput(seconds=5.0):
    """Bytes/second, measured against huggingface.co for a few seconds."""
    started = time.time()
    total = 0
    try:
        request = urllib.request.Request(SPEED_TEST_URL,
                                         headers={"User-Agent": "ngec-setup-doctor"})
        with urllib.request.urlopen(request, timeout=6) as resp:
            while time.time() - started < seconds:
                chunk = resp.read(262144)
                if not chunk:
                    break
                total += len(chunk)
    except Exception:                                        # noqa: BLE001
        return None
    elapsed = time.time() - started
    if elapsed <= 0 or total < 100000:
        return None
    return total / elapsed


# --------------------------------------------------------------------------
# Machine
# --------------------------------------------------------------------------

def check_machine():
    rows = []
    detail = (platform.system() + " " + platform.release()
              + " on " + platform.machine())
    rows.append(check("OS and architecture", True, detail, "Machine", level="info"))

    total_ram = None
    if IS_MAC:
        rc, out = run(["sysctl", "-n", "hw.memsize"])
        if rc == 0 and out.isdigit():
            total_ram = int(out)
    elif os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        total_ram = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    if total_ram is None:
        rows.append(check("RAM", True, "could not be read", "Machine", level="info"))
    else:
        enough = total_ram >= 8 * 1024 ** 3
        rows.append(check("RAM", enough, str(gb(total_ram)) + " GB", "Machine",
                          level="ok" if enough else "warn"))

    for label, path in [("Disk free (repo)", REPO_ROOT),
                        ("Disk free (docker)", docker_root())]:
        if not path or not os.path.exists(path):
            continue
        try:
            free = shutil.disk_usage(path).free
        except OSError:
            continue
        enough = gb(free) >= DISK_NEEDED_GB
        rows.append(check(label, enough,
                          str(gb(free)) + " GB at " + path
                          + " (the index needs about " + str(DISK_NEEDED_GB) + " GB)",
                          "Machine", level="ok" if enough else "warn"))
    return rows


def docker_root():
    if not shutil.which("docker"):
        return None
    rc, out = run(["docker", "info", "--format", "{{.DockerRootDir}}"], timeout=10)
    if rc == 0 and out and os.path.exists(out):
        return out
    return "/var/lib/docker" if os.path.exists("/var/lib/docker") else None


# --------------------------------------------------------------------------
# Python and the virtual environment
# --------------------------------------------------------------------------

def check_python():
    rows = []
    python3 = shutil.which("python3") or shutil.which("python")
    py_fix = fix("install-python", "uv python install 3.12",
                 "NGEC requires Python 3.10 or newer (it uses `X | Y` types and `match`). "
                 "uv installs one and uses it for the project venv, without touching the "
                 "system Python.", runnable=bool(shutil.which("uv")))
    if not python3:
        rows.append(check("python3 on PATH", False, "not found", "Python", fix=py_fix))
    else:
        rc, out = run([python3, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"])
        version = out if rc == 0 else "unknown"
        try:
            parts = version.split(".")
            new_enough = (int(parts[0]), int(parts[1])) >= (3, 10)
        except (ValueError, IndexError):
            new_enough = False
        detail = version + " at " + python3
        if "conda" in python3 or os.environ.get("CONDA_PREFIX"):
            detail += " (a conda environment; NGEC runs from .venv, not from here)"
        rows.append(check("python3 on PATH", new_enough, detail, "Python",
                          level="ok" if new_enough else "warn",
                          fix=None if new_enough else py_fix))

    uv = shutil.which("uv")
    if uv:
        rc, out = run([uv, "--version"])
        rows.append(check("uv", True, out if rc == 0 else uv, "Python"))
    else:
        command = ('powershell -c "irm https://astral.sh/uv/install.ps1 | iex"' if IS_WINDOWS
                   else "curl -LsSf https://astral.sh/uv/install.sh | sh")
        rows.append(check("uv", False, "not on PATH", "Python",
                          fix=fix("install-uv", command,
                                  "The project is managed with uv: it owns the virtual "
                                  "environment and picks the right PyTorch build through the "
                                  "cpu/cu12/cu13 extras.", "about 20s")))
    return rows


def check_venv(recommended_extra):
    """The project venv: does it exist, does `import ngec` work, which torch."""
    rows = []
    sync = "uv sync --extra " + recommended_extra
    if recommended_extra == "cu12":
        sync += " --extra vllm"
    sync_fix = fix(
        "uv-sync", sync,
        "Creates .venv and installs ngec with the PyTorch build that matches this "
        "machine. Exactly one of cpu/cu12/cu13 must be passed: without one, uv installs "
        "the default PyPI (CUDA 13) build, which silently falls back to the CPU on an "
        "older driver. The spaCy models and the dev tools come along, since `models` and "
        "`dev` are default dependency groups.",
        "5-20 min, mostly the PyTorch and vllm wheels")

    python = venv_python()
    if not os.path.exists(python):
        rows.append(check(".venv", False, "no virtual environment at " + python,
                          "Python", fix=sync_fix))
        return rows, sync_fix

    rc, out, stripped = venv_run("import ngec; print(ngec.__file__)")
    if rc != 0:
        rows.append(check("import ngec", False, last_line(out), "Python", fix=sync_fix))
        return rows, sync_fix
    rows.append(check("import ngec", True, out + ld_note(stripped), "Python"))

    rc, out, stripped = venv_run("import torch;print(torch.__version__, "
                                 "torch.version.cuda, torch.cuda.is_available())")
    if rc != 0:
        rows.append(check("torch build", False, last_line(out), "Python", fix=sync_fix))
    else:
        version, cuda, available = (out.split() + ["?", "?", "?"])[:3]
        detail = ("torch " + version + ", built for CUDA " + cuda + ", sees a GPU: "
                  + available + ld_note(stripped))
        blind = bool(nvidia_smi_query()) and available != "True"
        rows.append(check("torch build", not blind, detail, "Python",
                          level="ok" if not blind else "warn",
                          fix=None if not blind else fix(
                              "torch-rebuild", sync + " --reinstall-package torch",
                              "There is an NVIDIA GPU here but this torch cannot see it, so "
                              "the whole pipeline is running on the CPU. uv compares version "
                              "numbers only, so without --reinstall-package torch it does "
                              "nothing at all and reports success.", "3-10 min")))

    rc, out, _ = venv_run("import en_core_web_lg, en_core_web_trf; print('both')")
    rows.append(check("spaCy models", rc == 0,
                      "en_core_web_lg and en_core_web_trf are importable" if rc == 0
                      else "en_core_web_lg / en_core_web_trf missing from .venv", "Python",
                      fix=None if rc == 0 else fix(
                          "spacy-models", sync,
                          "The two spaCy models are the `models` dependency group, which a "
                          "bare `uv sync` installs. Nothing complains about a missing model "
                          "until something tries to load it.", "about 900 MB of wheels")))
    return rows, sync_fix


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

def nvidia_smi_query():
    """(driver_version, gpu_name) from nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    rc, out = run(["nvidia-smi", "--query-gpu=driver_version,name",
                   "--format=csv,noheader"], timeout=15)
    if rc != 0 or not out:
        return None
    first = out.splitlines()[0]
    fields = [f.strip() for f in first.split(",")]
    if len(fields) < 2:
        return None
    return fields[0], fields[1]


def nvidia_max_cuda():
    """The highest CUDA version this driver supports, as a float, or None."""
    rc, out = run(["nvidia-smi"], timeout=15)
    if rc != 0:
        return None
    found = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return float(found.group(1)) if found else None


def check_gpu():
    """The extra decision: cpu, cu12 or cu13. Returns (rows, recommended)."""
    rows = []
    smi = nvidia_smi_query()
    if not smi:
        rows.append(check("NVIDIA driver", True,
                          "no NVIDIA GPU (macOS: PyTorch uses MPS, and the attribute model "
                          "can use mlx)" if IS_MAC
                          else "no NVIDIA driver found (nvidia-smi is not on PATH)",
                          "GPU", level="info"))
        recommended = "cpu"
        rows.append(check("Recommended PyTorch extra", True,
                          "--extra cpu. No CUDA GPU here, so the CPU build of PyTorch is "
                          "the right one.", "GPU", level="info"))
    else:
        driver, name = smi
        max_cuda = nvidia_max_cuda()
        rows.append(check("NVIDIA driver", True, name + ", driver " + driver
                          + (", supports CUDA up to " + str(max_cuda) if max_cuda else ""),
                          "GPU"))
        if max_cuda is None:
            recommended, note = "cu12", ("The driver's CUDA version could not be read; "
                                         "cu12 is the safe choice.")
        elif max_cuda >= 13.0:
            # A CUDA 12 build runs on a 13 driver, and the pinned vllm (<0.20) is
            # the last CUDA 12 line; pyproject forbids cu13 + vllm outright.
            recommended, note = "cu12", (
                "The driver supports CUDA " + str(max_cuda) + ", so either cu13 or cu12 "
                "installs. Use cu12 if you want the vllm backend: the pinned vllm (<0.20) "
                "is a CUDA 12 build, and pyproject forbids cu13 + vllm. A CUDA 12 build "
                "runs fine on a CUDA 13 driver.")
        elif max_cuda >= 12.0:
            recommended, note = "cu12", (
                "The driver supports CUDA " + str(max_cuda) + ", so cu12 is the only "
                "working choice (and it is the one vllm needs).")
        else:
            recommended, note = "cpu", (
                "The driver only supports CUDA " + str(max_cuda) + ", older than any "
                "PyTorch build here. Use cpu.")
        rows.append(check("Recommended PyTorch extra", True,
                          "--extra " + recommended + ". " + note, "GPU", level="info"))

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ld and "cuda" in ld.lower():
        rows.append(check("LD_LIBRARY_PATH", False,
                          "contains a system CUDA (" + ld[:80] + ")", "GPU", level="warn",
                          fix=fix("ld-library-path", "env -u LD_LIBRARY_PATH uv run ...",
                                  "A system CUDA on LD_LIBRARY_PATH shadows the libraries "
                                  "inside the PyTorch wheels. The failure is an undefined "
                                  "symbol such as __nvJitLinkGetErrorLogSize_12_9, which "
                                  "says nothing about the path. Prefix every command that "
                                  "loads torch with `env -u LD_LIBRARY_PATH`.",
                                  runnable=False)))
    else:
        rows.append(check("LD_LIBRARY_PATH", True,
                          "set, no CUDA on it" if ld else "no system CUDA on it", "GPU"))
    return rows, recommended


# --------------------------------------------------------------------------
# Elasticsearch: the wiki and geonames indices. The big one.
# --------------------------------------------------------------------------

def existing_es_data_dir():
    """A data directory to mount, if one can be found on this machine.

    In order: NGEC_ES_DATA from .env or the environment; the mount of any
    container running the Elasticsearch image; the repo default.
    """
    for value in (os.environ.get("NGEC_ES_DATA"), ENV.get("NGEC_ES_DATA")):
        if value:
            return value, "NGEC_ES_DATA"
    if shutil.which("docker"):
        rc, out = run(["docker", "ps", "-a", "--filter", "ancestor=" + ES_IMAGE,
                       "--format", "{{.Names}}"], timeout=15)
        for name in (out.splitlines() if rc == 0 else []):
            rc2, mounts = run(["docker", "inspect", name, "--format",
                               "{{range .Mounts}}{{.Source}}::{{.Destination}}\n{{end}}"],
                              timeout=15)
            for line in (mounts.splitlines() if rc2 == 0 else []):
                source, _, dest = line.partition("::")
                if dest.strip() == "/usr/share/elasticsearch/data" and source:
                    return source, "the mount of container '" + name + "'"
    return os.path.join(REPO_ROOT, "elasticsearch", "data", "wikigeo_index"), "the repo default"


def existing_es_container():
    """(name, running) for a container built from the Elasticsearch image."""
    if not shutil.which("docker"):
        return None, False
    rc, out = run(["docker", "ps", "-a", "--filter", "ancestor=" + ES_IMAGE,
                   "--format", "{{.Names}}\t{{.State}}"], timeout=15)
    if rc != 0 or not out:
        return None, False
    name, _, state = out.splitlines()[0].partition("\t")
    return name.strip(), state.strip() == "running"


def docker_run_command(data_dir):
    """The command that launches Elasticsearch over a pre-built data directory.

    Derived from the container running on the reference machine
    (`docker inspect`): image elasticsearch:7.10.1, the default `eswrapper`
    command, discovery.type=single-node as the only environment variable, port
    9200 published, restart policy unless-stopped, and the data directory bind
    mounted at /usr/share/elasticsearch/data. It sets no memory limit and no
    ES_JAVA_OPTS, so Elasticsearch uses the image's default 1 GB heap; that is
    enough to serve these indices.
    """
    return ("docker run -d --name ngec-es \\\n"
            "  -p 9200:9200 \\\n"
            "  -e discovery.type=single-node \\\n"
            "  --restart unless-stopped \\\n"
            "  -v " + shell_quote(data_dir) + ":/usr/share/elasticsearch/data \\\n"
            "  " + ES_IMAGE)


def shell_quote(path):
    if re.match(r"^[A-Za-z0-9_@%+=:,./-]+$", path or ""):
        return path
    return "'" + (path or "").replace("'", "'\\''") + "'"


def download_target_dir():
    """Where a freshly downloaded index should land.

    Deliberately not the directory an existing container already uses: the
    recipe ends in an `mv`, and moving the new index on top of a live one is
    how you lose the old one. NGEC_ES_DATA, if it is set, is the user saying
    where they want it, so that wins.
    """
    for value in (os.environ.get("NGEC_ES_DATA"), ENV.get("NGEC_ES_DATA")):
        if value:
            return value
    return os.path.join(os.path.expanduser("~"), "ngec-es-data", "wikigeo_index")


def download_index_command(target_dir):
    """Download, unpack and rename the pre-built data directory.

    Not runnable while PREBUILT_INDEX_URL is "TODO"; the console refuses it and
    the command is shown for copying, with the placeholder visible.
    """
    parent = os.path.dirname(target_dir.rstrip("/")) or "."
    url = PREBUILT_INDEX_URL
    tarball = os.path.basename(url) if url != "TODO" else "geonames_wiki_index_*.tar.gz"
    lines = []
    if url == "TODO":
        lines.append("# TODO: the download URL is not recorded anywhere yet. Set")
        lines.append("# PREBUILT_INDEX_URL in setup/doctor/ngec_doctor.py once it is known.")
    lines.append("mkdir -p " + shell_quote(parent))
    lines.append("cd " + shell_quote(parent))
    lines.append("curl -LO " + url)
    lines.append("tar -xzf " + tarball)
    lines.append("mv " + PREBUILT_INDEX_DIRNAME + " "
                 + shell_quote(os.path.basename(target_dir.rstrip("/"))))
    return "\n".join(lines)


def iso_date(text):
    """The date part of an ISO 8601 string, or None if it is not one."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.date.fromisoformat(text[:10])
    except ValueError:
        return None


def index_build_date(url, index):
    """When this index's data was built, and where that date came from.

    Returns (date, source) or (None, None), preferring, in order:

    1. the mapping's `_meta`, which the two loaders under `elasticsearch/`
       stamp when a load finishes (`build_date`, and `dump_date` for the date
       of the source dump). This is real provenance and it travels with the
       index, wherever the index is copied to.
    2. `index.creation_date` from the index settings, which is when the index
       was created *on this machine*. For a data directory unpacked from a
       pre-built tarball that is the date the index was built; for a snapshot
       restore it is the date of the restore, and says nothing about the data.

    Indices built before the loaders started stamping `_meta` only have (2).
    """
    body = http_get(url + "/" + index + "/_mapping")
    try:
        meta = json.loads(body or "{}")[index]["mappings"].get("_meta") or {}
    except (ValueError, KeyError, AttributeError, TypeError):
        meta = {}
    for key in ("build_date", "dump_date"):
        built = iso_date(meta.get(key))
        if built:
            return built, "the index's own _meta." + key
    body = http_get(url + "/" + index + "/_settings")
    try:
        created = json.loads(body or "{}")[index]["settings"]["index"]["creation_date"]
        built = datetime.date.fromtimestamp(int(created) / 1000.0)
    except (ValueError, KeyError, AttributeError, TypeError, OSError, OverflowError):
        return None, None
    return built, ("index.creation_date, i.e. when the index was created on this machine "
                   "-- the build date for a data directory unpacked from a pre-built "
                   "tarball, but only the restore date for a snapshot restore")


def months_old(built, today):
    """Whole calendar months between two dates."""
    months = (today.year - built.year) * 12 + (today.month - built.month)
    if today.day < built.day:
        months -= 1
    return max(months, 0)


def check_index_age(url, index):
    """Is this index recent enough to still resemble what it is a copy of?"""
    built, source = index_build_date(url, index)
    if built is None:
        return check(index + " index age", True,
                     "no date available: the index carries no _meta provenance (it predates "
                     "the loaders stamping one) and its creation_date could not be read",
                     "Elasticsearch", level="info")
    today = datetime.date.today()
    age_days = (today - built).days
    months = months_old(built, today)
    age = (built.isoformat() + ", " + str(months)
           + (" month old" if months == 1 else " months old"))
    if age_days < INDEX_MAX_AGE_DAYS:
        return check(index + " index age", True,
                     age + ". Date from " + source + ".", "Elasticsearch")
    return check(
        index + " index age", False,
        age + " -- " + str(INDEX_MAX_AGE_DAYS) + " days or more, so it is out of date. "
        "Date from " + source + ".",
        "Elasticsearch", level="warn",
        fix=fix("es-refresh-" + index,
                "# see elasticsearch/SETUP.md\n"
                "#   path A: download and unpack a newer pre-built index (both indices)\n"
                "#   path B: rebuild this one index from a current source",
                STALE_COST[index] + " Nothing here is broken and the pipeline still runs, "
                "so refresh it when the coverage matters to you rather than before the next "
                "run. There is no command to press: a refresh means either downloading a "
                "newer pre-built index or rebuilding this one, both of which are long jobs "
                "you should start yourself.",
                "a rebuild takes many hours, about a day" if index == "wiki"
                else "a rebuild takes over 30 min",
                runnable=False))


def check_elasticsearch():
    rows = []
    data_dir, provenance = existing_es_data_dir()

    # --- Docker ---------------------------------------------------------
    if not shutil.which("docker"):
        rows.append(check("Docker", False, "not on PATH", "Elasticsearch",
                          fix=fix("install-docker",
                                  "open https://www.docker.com/get-started/" if IS_MAC
                                  else "xdg-open https://www.docker.com/get-started/",
                                  "Elasticsearch runs as a Docker container. Installing "
                                  "Docker itself needs administrator rights, so do it "
                                  "yourself: this console never runs sudo.",
                                  runnable=False)))
    else:
        rc, out = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
        if rc == 0 and out and "error" not in out.lower():
            rows.append(check("Docker", True, "daemon reachable, server " + out,
                              "Elasticsearch"))
        else:
            rows.append(check("Docker", False, "installed, but the daemon is not reachable",
                              "Elasticsearch",
                              fix=fix("start-docker",
                                      "open -a Docker" if IS_MAC
                                      else "sudo systemctl start docker",
                                      "The Docker daemon is not running. On Linux this needs "
                                      "root, so run it yourself in a terminal; the console "
                                      "never runs sudo.", runnable=False)))

    # --- Is Elasticsearch answering? ------------------------------------
    url = es_url()
    body = http_get(url + "/")
    version = None
    if body:
        try:
            version = json.loads(body).get("version", {}).get("number")
        except ValueError:
            version = None

    # An existing but stopped container is much the commonest case, and its fix
    # is one word rather than a mount path someone has to get right.
    container, running = existing_es_container()
    if container and not running:
        launch_fix = fix("es-launch", "docker start " + container,
                         "The container '" + container + "' already exists here with the "
                         "index data directory mounted; it is only stopped. Add `docker "
                         "update --restart unless-stopped " + container + "` afterwards so "
                         "a reboot does not leave it stopped again.", "about 30s to start")
    else:
        launch_fix = fix("es-launch", docker_run_command(data_dir),
                         "Starts Elasticsearch over a pre-built data directory ("
                         + data_dir + ", from " + provenance + "). One node serves both "
                         "indices out of one data directory. If the path is wrong, "
                         "Elasticsearch starts happily with an EMPTY data directory rather "
                         "than failing, so check _cat/indices afterwards. Never run two "
                         "containers against one data directory -- it corrupts it.",
                         "about 30s to start")

    if version:
        rows.append(check("Elasticsearch", True,
                          "answering at " + url + ", version " + version, "Elasticsearch"))
    else:
        rows.append(check("Elasticsearch", False, "nothing answering at " + url,
                          "Elasticsearch", fix=launch_fix))

    # --- The two indices ------------------------------------------------
    # Both live in one data directory, so one download fixes both; show it once.
    counts = {}
    if version:
        cat = http_get(url + "/_cat/indices?format=json")
        try:
            for row in json.loads(cat or "[]"):
                counts[row.get("index")] = row
        except ValueError:
            pass

    download_shown = False
    for index in ("wiki", "geonames"):
        expected = EXPECTED_DOCS[index]
        row = counts.get(index)
        if not version:
            rows.append(check(index + " index", False,
                              "cannot be checked until Elasticsearch answers (see the row "
                              "above)", "Elasticsearch", level="warn"))
        elif row is None and download_shown:
            rows.append(check(index + " index", False,
                              "not present in this cluster -- the same pre-built data "
                              "directory carries both, so the fix above covers this too",
                              "Elasticsearch"))
        elif row is None:
            download_shown = True
            target = download_target_dir()
            rows.append(check(index + " index", False, "not present in this cluster",
                              "Elasticsearch",
                              fix=fix("es-download",
                                      download_index_command(target)
                                      + "\n# then, once it is unpacked:\n"
                                      + docker_run_command(target),
                                      "The " + index + " index is missing. The quickest "
                                      "route is the pre-built data directory, which carries "
                                      "BOTH indices. The download URL is not recorded "
                                      "anywhere in this repo yet (PREBUILT_INDEX_URL is "
                                      "'TODO'), so fill it in before running this. Building "
                                      "the indices yourself instead: elasticsearch/SETUP.md.",
                                      download_estimate(PREBUILT_INDEX_BYTES) + " for the ~"
                                      + str(gb(PREBUILT_INDEX_BYTES)) + " GB tarball",
                                      runnable=False)))
        else:
            try:
                count = int(row.get("docs.count") or 0)
            except (TypeError, ValueError):
                count = 0
            detail = ("{:,}".format(count) + " docs, health " + row.get("health", "?")
                      + " (expected about " + "{:,}".format(expected) + ")")
            if count >= expected * 0.9:
                rows.append(check(index + " index", True, detail, "Elasticsearch"))
            else:
                rows.append(check(
                    index + " index", False,
                    detail + " -- far below the expected count, which usually means a load "
                    "that died part-way", "Elasticsearch", level="warn",
                    fix=fix("es-rebuild-" + index,
                            "# see elasticsearch/SETUP.md, path B\ntools/rebuild_index.sh "
                            + index,
                            "The index is present but short. Either re-download the "
                            "pre-built data directory, or rebuild this one index in place "
                            "-- the other index shares the data directory and is left alone.",
                            "over 30 min (geonames)" if index == "geonames"
                            else "many hours, about a day (wiki)",
                            runnable=False)))
            # Present and full, or present and short: either way it has a date.
            rows.append(check_index_age(url, index))
    return rows


# --------------------------------------------------------------------------
# Models in the Hugging Face cache
# --------------------------------------------------------------------------

def hf_cache_dir():
    home = os.environ.get("HF_HOME") or ENV.get("HF_HOME")
    if home:
        return os.path.join(home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def hf_has(repo_id):
    """Is this repo in the local Hugging Face cache, with a real snapshot?"""
    folder = os.path.join(hf_cache_dir(), "models--" + repo_id.replace("/", "--"))
    snapshots = os.path.join(folder, "snapshots")
    if not os.path.isdir(snapshots):
        return False
    for entry in os.listdir(snapshots):
        if os.listdir(os.path.join(snapshots, entry)):
            return True
    return False


def classifier_encoder():
    """The encoder named in the event models' metadata.json.

    The models are self-describing: which sentence encoder they were trained
    with is recorded there, not defaulted in code.
    """
    path = os.path.join(REPO_ROOT, "ngec", "assets", "event_models_v2", "metadata.json")
    try:
        with open(path) as handle:
            name = json.load(handle).get("encoder")
        if name:
            return name if "/" in name else "sentence-transformers/" + name
    except (OSError, ValueError):
        pass
    return FALLBACK_CLASSIFIER_ENCODER


def check_models():
    rows = []
    cache = hf_cache_dir()
    rows.append(check("Hugging Face cache", os.path.isdir(cache), cache, "Models",
                      level="info"))
    wanted = [(ATTRIBUTE_MODEL, "step 4, attribute extraction", ATTRIBUTE_MODEL_BYTES),
              (classifier_encoder(), "step 1, event classification", 440000000),
              (WIKI_ENCODER, "step 5, Wikipedia actor resolution", 500000000),
              (AGENT_ENCODER, "step 5, agent pattern matching", 130000000)]
    for repo_id, used_by, size in wanted:
        present = hf_has(repo_id)
        rows.append(check(repo_id, present,
                          ("cached" if present else "not in the cache") + " -- " + used_by,
                          "Models", level="ok" if present else "warn",
                          fix=None if present else fix(
                              "hf-" + repo_id.replace("/", "-"),
                              "uv run hf download " + repo_id,
                              "Used by " + used_by + ". It downloads on first use anyway; "
                              "fetching it now means the first pipeline run is not also a "
                              "download.", download_estimate(size))))
    return rows


# --------------------------------------------------------------------------
# Optional services
# --------------------------------------------------------------------------

def check_services():
    """llama-server, which only the llamacpp backend on a CPU host needs."""
    url = (os.environ.get("NGEC_LLAMACPP_URL") or ENV.get("NGEC_LLAMACPP_URL")
           or "http://127.0.0.1:8080")
    if http_get(url + "/health", timeout=2):
        served = ""
        try:
            entries = json.loads(http_get(url + "/v1/models", timeout=2) or "{}")
            models = entries.get("models") or []
            served = os.path.basename(str(models[0].get("name", ""))) if models else ""
        except (ValueError, AttributeError, IndexError):
            served = ""
        return [check("llama-server (optional)", True,
                      "up at " + url + (", serving " + served if served else ""),
                      "Services")]
    return [check("llama-server (optional)", True,
                  "not running at " + url + " -- only needed for the llamacpp backend on a "
                  "CPU host", "Services", level="info",
                  fix=fix("llama-server", "systemctl --user start ngec-llama-server",
                          "Only needed if you run the attribute model through llama.cpp on "
                          "a CPU host. See demo/deploy/README.md; the unit has to be "
                          "installed first.", runnable=False))]


# --------------------------------------------------------------------------
# Running everything
# --------------------------------------------------------------------------

def run_checks():
    FIXES.clear()
    rows = []
    rows += check_machine()
    rows += check_python()
    gpu_rows, recommended = check_gpu()
    rows += gpu_rows
    venv_rows, _ = check_venv(recommended)
    rows += venv_rows
    rows += check_elasticsearch()
    rows += check_models()
    rows += check_services()
    rows.sort(key=lambda row: GROUPS.index(row["group"]) if row["group"] in GROUPS else 99)
    return {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "repo": REPO_ROOT,
            "recommended_extra": recommended,
            "prebuilt_index_url": PREBUILT_INDEX_URL,
            "groups": GROUPS,
            "checks": rows,
            "summary": {"total": len(rows),
                        "failed": sum(1 for r in rows if not r["ok"])}}


MARKS = {"ok": " ok ", "warn": "warn", "fail": "FAIL", "info": "  . "}


def print_checklist(report):
    print("NGEC setup doctor -- " + report["repo"])
    print(report["generated"] + "   recommended PyTorch extra: --extra "
          + report["recommended_extra"])
    current = None
    for row in report["checks"]:
        if row["group"] != current:
            current = row["group"]
            print("")
            print(current.upper())
            print("-" * len(current))
        print("[" + MARKS.get(row["level"], "    ") + "] "
              + row["name"] + ": " + row["detail"])
        fix = row["fix"]
        if fix and not row["ok"]:
            print("        why: " + fix["why"])
            for line in fix["command"].splitlines():
                print("        $ " + line)
            if fix.get("estimate"):
                print("        time: " + fix["estimate"])
    failed = report["summary"]["failed"]
    print("")
    if failed:
        print(str(failed) + " of " + str(report["summary"]["total"])
              + " checks need attention. Run with --serve for a page with "
              "Run buttons, or --json for a machine-readable report.")
    else:
        print("Everything this doctor knows how to check is in place.")
    return 0 if failed == 0 else 1


# --------------------------------------------------------------------------
# --serve: one local page, no frameworks, no CDN
# --------------------------------------------------------------------------

PAGE = """<title>NGEC setup</title>
<style>
:root{--paper:#FFFFFF;--chip:#F2F2F0;--hair:#E6E6E4;--ink:#111111;
--body:#2A2A28;--mute:#6C6C68;--label:#8E8E8A;--accent:#F0521E;--ok:#2E7D53;
--sans:"Helvetica Neue",Helvetica,Arial,sans-serif;--mono:ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--body);font-family:var(--sans);margin:0;
padding:32px 24px 80px;font-size:14px;line-height:1.5}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:1.4rem;color:var(--ink);margin:0 0 4px;letter-spacing:-0.01em}
.sub{color:var(--mute);font-family:var(--mono);font-size:0.72rem;margin-bottom:28px}
h2{font-family:var(--mono);font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;
color:var(--label);border-bottom:1px solid var(--hair);padding-bottom:6px;margin:32px 0 0}
.card{border-bottom:1px solid var(--hair);padding:12px 0}
.line{display:flex;align-items:baseline;gap:10px}
.mark{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.1em;padding:2px 6px;
border-radius:2px;background:var(--chip);color:var(--mute);white-space:nowrap}
.mark.ok{color:var(--ok)} .mark.fail{background:var(--accent);color:#fff}
.mark.warn{color:var(--accent);border:1px solid var(--accent)}
.name{color:var(--ink);font-weight:600}
.detail{color:var(--mute)}
.why{margin:8px 0 0 0;color:var(--body)}
pre.cmd{background:var(--chip);border:1px solid var(--hair);font-family:var(--mono);
font-size:0.78rem;padding:10px 12px;margin:8px 0 0;overflow-x:auto;white-space:pre;color:var(--ink)}
.row{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
button{font-family:var(--mono);font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;
background:var(--paper);color:var(--ink);border:1px solid var(--ink);padding:5px 12px;cursor:pointer}
button:hover{background:var(--accent);border-color:var(--accent);color:#fff}
button[disabled]{border-color:var(--hair);color:var(--label);cursor:default;background:var(--paper)}
.est{font-family:var(--mono);font-size:0.68rem;color:var(--mute)}
pre.out{background:#111;color:#eee;font-family:var(--mono);font-size:0.72rem;
padding:10px 12px;margin:8px 0 0;max-height:320px;overflow:auto;white-space:pre-wrap}
.note{background:var(--chip);border-left:3px solid var(--accent);padding:10px 12px;
margin:16px 0 0;font-size:0.82rem}
</style>
<div class="wrap">
<h1>NGEC setup</h1>
<div class="sub" id="sub">checking...</div>
<div id="body"></div>
<div class="note">This page runs only the commands this doctor generated, by id,
and never anything with <code>sudo</code>. Anything it will not run is still shown
so you can copy it.</div>
</div>
<script>
function el(tag, cls, text){var n=document.createElement(tag);
if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n;}

function render(report){
  document.getElementById('sub').textContent =
    report.repo + '  |  ' + report.generated + '  |  recommended: --extra ' +
    report.recommended_extra + '  |  ' + report.summary.failed + ' of ' +
    report.summary.total + ' need attention';
  var body = document.getElementById('body');
  body.innerHTML = '';
  var current = null;
  report.checks.forEach(function(c){
    if(c.group !== current){ current = c.group; body.appendChild(el('h2', null, current)); }
    var card = el('div','card');
    var line = el('div','line');
    line.appendChild(el('span','mark '+c.level, c.level));
    line.appendChild(el('span','name', c.name));
    line.appendChild(el('span','detail', c.detail));
    card.appendChild(line);
    if(c.fix && !c.ok){
      card.appendChild(el('div','why', c.fix.why));
      card.appendChild(el('pre','cmd', c.fix.command));
      var row = el('div','row');
      var run = el('button', null, c.fix.runnable ? 'Run' : 'Run it yourself');
      if(!c.fix.runnable){ run.disabled = true; }
      var out = el('pre','out'); out.style.display='none';
      run.onclick = function(){ runFix(c.fix.id, run, out); };
      row.appendChild(run);
      var copy = el('button', null, 'Copy');
      copy.onclick = function(){
        navigator.clipboard.writeText(c.fix.command).then(function(){
          copy.textContent='Copied'; setTimeout(function(){copy.textContent='Copy';},1200);});
      };
      row.appendChild(copy);
      if(c.fix.estimate){ row.appendChild(el('span','est', c.fix.estimate)); }
      card.appendChild(row);
      card.appendChild(out);
    }
    body.appendChild(card);
  });
}

function runFix(id, button, out){
  button.disabled = true; button.textContent = 'Running';
  out.style.display = 'block'; out.textContent = '';
  var source = new EventSource('/api/run?id=' + encodeURIComponent(id));
  source.onmessage = function(e){ out.textContent += e.data + '\\n'; out.scrollTop = out.scrollHeight; };
  source.addEventListener('done', function(e){
    out.textContent += '\\n[exit ' + e.data + ']\\n';
    source.close(); button.textContent = 'Re-checking'; refresh();
  });
  source.onerror = function(){ source.close(); button.disabled=false; button.textContent='Run'; };
}

function refresh(){
  fetch('/api/checks').then(function(r){return r.json();}).then(render);
}
refresh();
</script>
"""


class Console(BaseHTTPRequestHandler):
    """One page, one JSON endpoint, one streaming endpoint."""

    server_version = "ngec-setup-doctor"
    lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass                                        # the terminal stays readable

    def _send(self, code, content_type, payload):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):                               # noqa: N802 (http.server API)
        path, _, query = self.path.partition("?")
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/checks":
            report = run_checks()
            self._send(200, "application/json",
                       json.dumps(report, indent=1).encode("utf-8"))
        elif path == "/api/run":
            self.stream_fix(dict(p.split("=", 1) for p in query.split("&") if "=" in p))
        else:
            self._send(404, "text/plain", b"not found")

    def stream_fix(self, params):
        fix_id = urllib.parse.unquote(params.get("id", ""))
        fix = FIXES.get(fix_id)
        # Three refusals, in order: a command we did not generate, one we marked
        # as needing a human, and anything with sudo in it.
        if fix is None:
            self._send(400, "text/plain", b"unknown fix id; re-run the checks")
            return
        if not fix.get("runnable", True) or "sudo" in fix["command"] or "TODO" in fix["command"]:
            self._send(400, "text/plain", b"this command is not run by the console")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        environment = dict(os.environ)
        environment.pop("LD_LIBRARY_PATH", None)    # never let a stale CUDA in
        with Console.lock:
            try:
                proc = subprocess.Popen(fix["command"], shell=True, cwd=REPO_ROOT,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=environment)
                for raw in iter(proc.stdout.readline, b""):
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    self.wfile.write(("data: " + line + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                proc.wait()
                code = proc.returncode
            except Exception as exc:                # noqa: BLE001
                self.wfile.write(("data: " + str(exc) + "\n\n").encode("utf-8"))
                code = 1
        try:
            self.wfile.write(("event: done\ndata: " + str(code) + "\n\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(port):
    run_checks()                                    # populate FIXES before any request
    server = ThreadingHTTPServer(("127.0.0.1", port), Console)
    print("NGEC setup console on http://127.0.0.1:" + str(port) + "  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
    finally:
        server.server_close()
    return 0


def main(argv=None):
    global _NO_NETWORK
    parser = argparse.ArgumentParser(
        description="Check what this machine still needs before NGEC will run.")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable report on stdout")
    parser.add_argument("--serve", action="store_true",
                        help="serve the setup console on 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765,
                        help="port for --serve (default 8765)")
    parser.add_argument("--no-network", action="store_true",
                        help="skip the download-speed sample used for time estimates")
    args = parser.parse_args(argv)
    _NO_NETWORK = args.no_network

    if args.serve:
        return serve(args.port)
    report = run_checks()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["summary"]["failed"] == 0 else 1
    return print_checklist(report)


if __name__ == "__main__":
    sys.exit(main())
