"""NGEC Doctor: a command-line tool to check your environment and diagnose problems.

Run it as `ngec-doctor`, or as `python -m ngec.doctor`.

Implemented here:

- Installation: the ngec version and commit, the Python running it, and where
  the package is being imported from
- Configuration: every environment variable ngec or its tooling reads, the
  effective value, and which code actually reads it (`SETTINGS`); and any key
  in .env that is not one of them, which is usually a misspelling
- Compute: the PyTorch build, whether it can really see the GPU that is
  present, and what `gpu=True` will do on this machine
- Elasticsearch: whether it is reachable, and whether the `wiki` and
  `geonames` indices are in it
- Smoke test (only with `--smoke`): three real news stories run through the
  whole pipeline, after checking that every model it needs is already
  downloaded. Doctor never downloads anything; see `ngec/smoke_test.py`.

Still to come, roughly in this order: packaged assets; whether the index
contents are complete and current, not merely present; the available LLM
backends, including whether `llama-server` is serving the model the Python side
is prompting for.

The `Check` structure, and the idea that a finding has to say what it breaks
and how to fix it, are taken from `demo/ngec_demo/resources.py::Health`, which
does the same job for the demo's sidebar. The demo still has its own copy; the
two are not wired together yet.

Every third-party import happens inside the check that needs it, and failures
come back as findings. A broken install is exactly when this gets run, so
`import torch` raising has to produce a report, not a traceback.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"


@dataclass
class Check:
    """One thing doctor looked at.

    `blocks` names what stops working when this is wrong, and `fix` is the
    command that resolves it. Both are what make a finding worth printing at
    all: a check that cannot say what its failure costs is noise in a report
    someone is reading because something already went wrong.

    `status` is INFO for rows that are reported rather than judged -- the
    Python version, an unset optional variable. They belong in a bug report,
    but there is no such thing as them being wrong, so they never reach the
    problem list or the exit code. `note` is a dim annotation shown beside the
    detail; the settings rows use it to say which code reads each variable.
    """

    name: str
    status: str
    detail: str
    blocks: str = ""
    fix: str = ""
    note: str = ""


# ---------------------------------------------------------------- installation


def _git_describe() -> str | None:
    """Commit, branch and dirty flag, when running from a git checkout.

    The version number moves slowly here (0.1.0 since the beginning), so the
    commit is the only thing that actually identifies what someone is running.
    Returns None for an installed copy with no `.git` beside it.
    """
    if not (REPO_ROOT / ".git").exists():
        return None

    def git(*args: str) -> str:
        result = subprocess.run(("git", "-C", str(REPO_ROOT)) + args,
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else ""

    try:
        commit = git("rev-parse", "--short", "HEAD")
        if not commit:
            return None
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = ", uncommitted changes" if git("status", "--porcelain") else ""
        return f"{commit} on {branch}{dirty}"
    except (OSError, subprocess.SubprocessError):
        return None


def installation() -> list[Check]:
    from importlib.metadata import PackageNotFoundError, version

    checks: list[Check] = []

    try:
        checks.append(Check("ngec", INFO, version("ngec")))
    except PackageNotFoundError:
        checks.append(Check(
            "ngec", WARN,
            "imported, but not installed as a distribution",
            "nothing outright, but the package is coming from a source tree "
            "rather than an install, so the dependency set is whatever "
            "happens to be in this environment",
            "uv sync"))

    commit = _git_describe()
    checks.append(Check("Source", INFO,
                        f"git checkout at {commit}" if commit
                        else "installed copy (no git checkout alongside it)"))
    checks.append(Check("Location", INFO, str(PACKAGE_DIR)))
    checks.append(Check("Python", INFO, platform.python_version(),
                        note=sys.executable))
    checks.append(Check(
        "Platform", INFO,
        f"{platform.system()} {platform.release()} ({platform.machine()})"))

    return checks


# --------------------------------------------------------------- configuration


# Variables that were absent from the process environment and were supplied by
# a .env file, so the settings rows can say where a value came from. Populated
# by `_dotenv()`, which `configuration()` runs first.
_FROM_DOTENV: set[str] = set()


def _find_env_file() -> Path | None:
    """The .env that python-dotenv's own search would find, without importing it.

    `tests/conftest.py` and the demo both call bare `load_dotenv()`, which walks
    up from the working directory. Doing the same walk here means doctor
    reports the file those callers would actually pick up, which is not
    necessarily the one next to the package.
    """
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _dotenv() -> Check:
    """Apply .env the way the tests and the demo do, and report what that did.

    This deliberately mutates the environment of the doctor process: the
    settings below should show the values a caller in this directory would
    actually get, and for most people that means the ones .env supplies.
    """
    env_file = _find_env_file()
    if env_file is None:
        return Check(".env", INFO, "no .env file found from here upwards")

    try:
        from dotenv import load_dotenv
    except ImportError:
        return Check(
            ".env", WARN,
            f"{env_file} exists, but python-dotenv is not installed, so nothing "
            "loads it",
            "any setting that lives only in .env -- the values below are the "
            "process environment alone, and so are the ones the tests would see",
            "uv sync --group dev")

    before = set(os.environ)
    load_dotenv(env_file)
    _FROM_DOTENV.update(set(os.environ) - before)
    # Doctor, the tests and the demo load .env themselves. A user's own script
    # does not unless it calls load_dotenv() or es_client_from_env(), so a
    # value shown below can be one their pipeline never sees.
    return Check(".env", OK,
                 f"{len(_FROM_DOTENV)} setting(s) loaded; your own scripts see "
                 "them only if they load .env too",
                 note=str(env_file))


@dataclass(frozen=True)
class Setting:
    """An environment variable that NGEC, its demo, or its tooling reads.

    A `default` of None means "ask the code that defines it"; only the attribute
    model uses that, because its default lives in `ngec.attribute_model` and a
    second copy of the name here would drift. `secret` values are reported as
    "set" and never printed. Settings with `always_shown=False` matter only to
    the demo, to evaluation, or to building and publishing an index, so their
    rows appear only when the variable is actually set.
    """

    name: str
    default: str | None
    read_by: str
    secret: bool = False
    always_shown: bool = True


# Every environment variable NGEC reads, in the order the report shows them.
# This list is what `.env` is checked against, so it has to be complete:
# tests/test_doctor.py fails if the code reads an NGEC_ name missing from it,
# or if .env.example documents anything other than exactly these.
SETTINGS = [
    Setting("NGEC_ATTRIBUTE_MODEL", None, "ngec.attribute_model"),
    Setting("NGEC_WIKI_ENCODER", "sentence-transformers/static-retrieval-mrl-en-v1",
            "ngec.actors", always_shown=False),
    Setting("NGEC_AGENT_ENCODER", "BAAI/bge-small-en-v1.5", "ngec.actors",
            always_shown=False),
    Setting("NGEC_LLAMACPP_URL", "http://127.0.0.1:8080", "ngec.llm.llamacpp"),
    Setting("ES_HOST", "localhost", "tests, demo, smoke test"),
    Setting("ES_PORT", "9200", "tests, demo, smoke test"),
    Setting("ES_USER", "", "tests, demo, smoke test", secret=True),
    Setting("ES_PASSWORD", "", "tests, demo, smoke test", secret=True),
    # Not the same cluster setting as ES_HOST/ES_PORT, and nothing keeps them in
    # step; see _es_agreement.
    Setting("NGEC_WIKI_URL", "http://localhost:9200/wiki", "ngec.actors (v3 splitter)"),
    Setting("NGEC_ES_URL", "http://localhost:9200/", "tools/, elasticsearch/"),
    Setting("NGEC_ES_DATA", "", "elasticsearch/compose-build.yml"),
    Setting("NGEC_ES_PORT", "9200", "elasticsearch/compose-build.yml", always_shown=False),
    Setting("NGEC_REDIS_HOST", "localhost", "elasticsearch/es_wiki"),
    Setting("NGEC_REDIS_PORT", "6379", "elasticsearch/es_wiki"),
    Setting("NGEC_PUBLISH_DEST", "", "tools/publish_index.sh", always_shown=False),
    Setting("NGEC_INDEX_LATEST_URL",
            "https://andrewhalterman.com/files/wikigeo_index_latest.json",
            "ngec download-index, ngec update", always_shown=False),
    Setting("HF_HOME", "~/.cache/huggingface", "huggingface_hub"),
    Setting("NGEC_DEMO_PASSWORD", "", "demo", secret=True, always_shown=False),
    Setting("NGEC_DEMO_MODE", "gpu if CUDA is available, else cpu", "demo",
            always_shown=False),
    Setting("NGEC_DEMO_CPU_THREADS", "4", "demo", always_shown=False),
    Setting("NGEC_DEMO_GPU_MEMORY", "0.25", "demo", always_shown=False),
    Setting("NGEC_DEMO_CPU_BACKEND", "llamacpp", "demo", always_shown=False),
]

# A line in .env that sets a variable. A commented-out line sets nothing.
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _keys_in(env_file: Path) -> list[str]:
    keys: list[str] = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = _ENV_LINE.match(line)
        if match and match.group(1) not in keys:
            keys.append(match.group(1))
    return keys


def _env_keys(env_file: Path | None) -> list[Check]:
    """Check the keys set in .env against the settings NGEC actually reads.

    A misspelled name is the classic silent failure here. python-dotenv loads
    NGEC_ATTRIBUTE_MODLE without complaint, nothing reads it, and the run quietly
    uses the default model.

    The reference is SETTINGS, not a file on disk, so this works the same for an
    installed copy as for a clone, and a `.env.example` belonging to the user's
    own project cannot be mistaken for NGEC's.

    Only likely typos are warnings: a name close to a known one, or any NGEC_
    name, since nothing else would use that prefix. Other keys (HF_TOKEN,
    CUDA_VISIBLE_DEVICES, a user's own settings) are reported but not judged.
    """
    if env_file is None:
        return []

    known = [setting.name for setting in SETTINGS]
    unknown = [key for key in _keys_in(env_file) if key not in known]
    if not unknown:
        return [Check(".env keys", OK, "every key in .env is a setting NGEC reads")]

    reference = ".env.example in the NGEC repository lists every setting"
    checks: list[Check] = []
    others: list[str] = []
    for key in unknown:
        # Upper-cased so that es_host is caught as well as ES_HOTS. The match is
        # a guess -- ES_URL comes out as ES_USER -- so the fix is worded as one.
        close = difflib.get_close_matches(key.upper(), known, n=1, cutoff=0.75)
        if close:
            checks.append(Check(
                key, WARN, f"set in .env, but nothing reads it; did you mean {close[0]}?",
                "whatever setting it was meant to be: a misspelled name is loaded "
                "without error and then ignored, so that setting keeps its default",
                f"if you meant {close[0]}, rename it in {env_file}; {reference}"))
        elif key.startswith("NGEC_"):
            checks.append(Check(
                key, WARN, "set in .env, but no NGEC code reads it",
                "whatever setting it was meant to be: nothing reads this name, so "
                "that setting keeps its default",
                f"check the name; {reference}"))
        else:
            others.append(key)
    if others:
        checks.append(Check(
            ".env keys", INFO,
            f"also sets {', '.join(others)}, which NGEC does not read",
            note="fine if something else uses them"))
    return checks


def _default_attribute_model() -> str:
    """The model AttributeModel falls back to, asked of the code that defines it."""
    try:
        from .attribute_model import DEFAULT_MODEL

        return DEFAULT_MODEL
    except Exception:  # noqa: BLE001 - a broken import is the compute group's finding
        return "the package default"


def _setting(setting: Setting) -> Check:
    default = setting.default
    if default is None:
        default = _default_attribute_model()
    raw = os.environ.get(setting.name)
    if raw is None:
        detail = f"unset, defaulting to {default}" if default else "unset"
    elif setting.secret:
        # Doctor output is the sort of thing that gets pasted into an issue.
        detail = "set"
    else:
        detail = raw
    if setting.name in _FROM_DOTENV:
        detail += " (from .env)"
    return Check(setting.name, INFO, detail, note=setting.read_by)


def _es_agreement() -> Check:
    """Check the two ways of naming Elasticsearch against each other.

    The pipeline is handed a host and port by its caller -- the tests and the
    demo read ES_HOST and ES_PORT to do it -- while the index build tooling in
    `tools/` and `elasticsearch/` reads a single NGEC_ES_URL. Nothing keeps the
    two in step, so pointing .env at a remote cluster and then rebuilding an
    index quietly rebuilds a local one. Both halves are valid configurations,
    so neither side can complain; only something looking at both can.
    """
    library = f"{os.environ.get('ES_HOST', 'localhost')}:{os.environ.get('ES_PORT', '9200')}"
    parsed = urllib.parse.urlparse(os.environ.get("NGEC_ES_URL", "http://localhost:9200/"))
    tooling = f"{parsed.hostname or 'localhost'}:{parsed.port or 9200}"

    if library == tooling:
        return Check("Elasticsearch target", OK, f"both point at {library}")
    return Check(
        "Elasticsearch target", WARN,
        f"ES_HOST/ES_PORT say {library}, NGEC_ES_URL says {tooling}",
        "index building and the pipeline would use different clusters, with no "
        "error from either side",
        f"if {library} is the cluster you mean, set NGEC_ES_URL=http://{library}/")


def configuration() -> list[Check]:
    checks = [_dotenv()]
    checks += _env_keys(_find_env_file())
    checks += [_setting(setting) for setting in SETTINGS
               if setting.always_shown or setting.name in os.environ]
    checks.append(_es_agreement())
    return checks


# -------------------------------------------------------------------- compute


def _nvidia_smi() -> dict[str, str] | None:
    """What the NVIDIA driver reports, independent of torch.

    Asking the driver directly is the only way to tell "there is no GPU here"
    apart from "there is a GPU here and the installed torch cannot use it".
    The second is the failure README warns about -- a torch built against a
    newer CUDA than the driver supports does not raise, it reports no GPU and
    runs everything on the CPU -- and torch reports both cases identically.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        return None
    name, _, driver = lines[0].partition(",")
    return {"name": name.strip(), "driver": driver.strip(), "count": str(len(lines))}


def compute() -> list[Check]:
    checks: list[Check] = [
        Check("CPU", INFO, f"{os.cpu_count()} logical cores"),
    ]

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - ImportError, but also OSError on a bad build
        checks.append(Check(
            "PyTorch", FAIL, f"cannot import torch: {type(exc).__name__}: {exc}",
            "everything -- spaCy's transformer model, the sentence encoders and "
            "every attribute-model backend go through torch",
            "uv sync, then uv pip install torch --torch-backend=auto "
            "--reinstall-package torch"))
        return checks

    checks.append(Check("PyTorch", INFO, torch.__version__))

    smi = _nvidia_smi()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            checks.append(Check(
                f"GPU {i}", OK,
                f"{properties.name}, {properties.total_memory / 1e9:.0f} GB",
                note=f"driver {smi['driver']}" if smi else ""))
    elif smi:
        checks.append(Check(
            "GPU", WARN,
            f"{smi['name']} is present (driver {smi['driver']}), but torch "
            f"{torch.__version__} does not see it",
            "gpu=True and the vllm backend; the pipeline runs on the CPU "
            "instead, at a fraction of the speed and without saying so",
            "uv pip install torch --torch-backend=auto --reinstall-package torch"
            " (see the PyTorch section of README.md)"))
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            # Not a misconfiguration, so not a warning: ngec has no MPS device
            # path at all. `gpu=True` sets device="cuda" unconditionally
            # (attribute_model.py), which on this machine fails rather than
            # falling back.
            checks.append(Check(
                "GPU", INFO,
                "Apple Silicon GPU available through MPS, but ngec has no MPS "
                "path: gpu=True selects cuda. Use backend='mlx', or "
                "backend='llamacpp' against a local llama-server"))
        else:
            checks.append(Check(
                "GPU", INFO,
                "Apple Silicon, but torch reports MPS unavailable; everything "
                "runs on the CPU"))
    else:
        detail = "none detected; the pipeline will run on the CPU"
        if torch.version.cuda:
            detail += (f" (this torch is a CUDA {torch.version.cuda} build, but "
                       "no NVIDIA GPU is visible)")
        checks.append(Check("GPU", INFO, detail))

    return checks


# -------------------------------------------------------------- elasticsearch


# The indices the pipeline queries: `wiki` for actor resolution
# (ngec/actors/wiki_matcher.py) and `geonames` for mordecai3's geolocation.
ES_INDICES = ("wiki", "geonames")


def _es_target() -> str:
    return f"{os.environ.get('ES_HOST', 'localhost')}:{os.environ.get('ES_PORT', '9200')}"


def elasticsearch() -> list[Check]:
    """Can the pipeline reach Elasticsearch, and are both indices in it?

    Connects the way the tests and the demo do (ES_HOST, ES_PORT, .env), with a
    short timeout so that an unreachable host costs seconds, not minutes.
    """
    try:
        from .es_client import es_client_from_env

        client = es_client_from_env(timeout=5, max_retries=0)
    except Exception as exc:  # noqa: BLE001 - any failure to connect is the finding
        return [Check(
            "Elasticsearch", FAIL,
            f"cannot connect to {_es_target()}: {type(exc).__name__}",
            "geolocation and actor resolution, and so the pipeline as a whole",
            "start Elasticsearch (README, step 5), or point ES_HOST / ES_PORT "
            "at the cluster you mean")]

    version = client.info().get("version", {}).get("number", "unknown")
    checks = [Check("Elasticsearch", OK, f"{_es_target()}, version {version}")]

    for index in ES_INDICES:
        if not client.indices.exists(index=index):
            checks.append(Check(
                f"'{index}' index", FAIL, "missing",
                "actor resolution" if index == "wiki" else "geolocation",
                "`ngec download-index --start` fetches the pre-built index; if "
                "you already have it, a cluster without it is usually the wrong "
                "volume path in `docker run -v`"))
            continue
        count = client.count(index=index)["count"]
        if count == 0:
            checks.append(Check(
                f"'{index}' index", FAIL, "exists but is empty",
                "actor resolution" if index == "wiki" else "geolocation",
                "replace it with `ngec download-index` (see elasticsearch/SETUP.md)"))
        else:
            checks.append(Check(f"'{index}' index", OK, f"{count:,} documents"))

    return checks


# ---------------------------------------------------------------------- smoke


def _attribute_model_is_local(name: str) -> bool:
    """Whether the attribute model can be loaded without downloading anything."""
    if Path(name).expanduser().is_dir():
        return True
    # Ask for the files loading needs, not the whole repository:
    # snapshot_download(local_files_only=True) raises when any file of the repo
    # is missing from the cache, and a model the pipeline fetched itself on
    # first use only has the files it loaded, so that check failed for a model
    # that loads fine.
    try:
        from huggingface_hub import try_to_load_from_cache

        def cached(filename):
            return isinstance(try_to_load_from_cache(name, filename), str)

        return cached("config.json") and (cached("model.safetensors")
                                           or cached("model.safetensors.index.json"))
    except Exception:  # noqa: BLE001 - huggingface_hub is broken
        return False


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


def smoke() -> list[Check]:
    """Run three real news stories through the whole pipeline.

    Opt-in (`--smoke`), because it takes minutes on a CPU. It never downloads a
    model: it first checks that the spaCy models and the attribute model are
    already here, and then runs the pipeline in a separate process with
    Hugging Face's offline mode on, so anything else that is missing fails
    instead of being fetched. See `ngec/smoke_test.py` for why a separate
    process.
    """
    import tempfile
    import time

    from .models import missing_spacy_models

    checks: list[Check] = []

    missing = missing_spacy_models()
    if missing:
        checks.append(Check(
            "spaCy models", FAIL, f"not installed: {', '.join(missing)}",
            "parsing, and so the pipeline as a whole", "ngec download-models"))
    else:
        checks.append(Check("spaCy models", OK, "installed"))

    model = os.environ.get("NGEC_ATTRIBUTE_MODEL") or _default_attribute_model()
    if _attribute_model_is_local(model):
        checks.append(Check("Attribute model", OK, f"{model} is downloaded"))
    else:
        checks.append(Check(
            "Attribute model", FAIL, f"{model} is not downloaded",
            "attribute extraction, and so the pipeline as a whole",
            "ngec download-models"))

    # One row rather than the whole Elasticsearch group, which a plain
    # `--smoke` run also shows; the first failure is enough to act on.
    es_failures = [c for c in elasticsearch() if c.status == FAIL]
    if es_failures:
        first = es_failures[0]
        detail = first.detail if first.name == "Elasticsearch" \
            else f"{first.name}: {first.detail}"
        checks.append(Check("Elasticsearch", FAIL, detail, first.blocks, first.fix))
    else:
        checks.append(Check("Elasticsearch", OK, "reachable, both indices loaded"))

    if any(c.status == FAIL for c in checks):
        # INFO, not FAIL: the failures above already make the exit code
        # non-zero, and this adds nothing to fix.
        checks.append(Check("Pipeline", INFO, "not run: fix the problems above first"))
        return checks

    print("Running three news stories through the pipeline. This takes a few "
          "minutes on a CPU...", file=sys.stderr)

    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="ngec-smoke-") as tmp:
        out_file = Path(tmp) / "result.json"
        start = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ngec.smoke_test", str(out_file)],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            checks.append(Check(
                "Pipeline", FAIL, "still running after an hour; stopped",
                "nothing by itself, but on three short stories this means "
                "something is badly wrong -- check the Compute group above"))
            return checks
        seconds = time.monotonic() - start

        if result.returncode != 0 or not out_file.exists():
            error = _last_line(result.stderr or result.stdout)
            offline = "offline" in error.lower() or "local_files_only" in error
            checks.append(Check(
                "Pipeline", FAIL, error,
                "the pipeline as a whole",
                "ngec download-models (the pipeline needed a model that is not "
                "downloaded)" if offline else
                "run the pipeline directly to see the full error: "
                "python -m ngec.smoke_test out.json"))
            return checks

        summary = json.loads(out_file.read_text(encoding="utf-8"))

    events = summary["events"]
    if not events:
        checks.append(Check(
            "Pipeline", FAIL,
            f"ran in {seconds:.0f}s but coded no events from "
            f"{summary['stories']} stories, which should each produce some",
            "the pipeline's output: something upstream is silently dropping "
            "everything, e.g. the event classifier finding no event types"))
        return checks

    checks.append(Check(
        "Pipeline", OK,
        f"{len(events)} events from {summary['stories']} stories in {seconds:.0f}s"))
    checks += [Check(e["story"], INFO, e["summary"]) for e in events]
    return checks


# --------------------------------------------------------------------- output


GROUPS: dict[str, tuple[str, object]] = {
    "install": ("Installation", installation),
    "config": ("Configuration", configuration),
    "compute": ("Compute", compute),
    "elasticsearch": ("Elasticsearch", elasticsearch),
    "smoke": ("Smoke test", smoke),
}

# Everything but the smoke test, which takes minutes and has to be asked for.
DEFAULT_GROUPS = ["install", "config", "compute", "elasticsearch"]

GLYPHS = {OK: ("check", "green"), INFO: ("dot", "dim"),
          WARN: ("bang", "yellow"), FAIL: ("cross", "red")}
SYMBOLS = {"check": "✓", "dot": "·", "bang": "!", "cross": "✗"}


def render(groups: list[tuple[str, list[Check]]]) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()

    for title, checks in groups:
        console.print(f"\n[bold]{title}[/bold]")
        table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
        table.add_column(width=1)
        table.add_column(style="bold", no_wrap=True)
        table.add_column(overflow="fold")
        # Capped so that a long note -- an interpreter path, usually -- takes
        # room from itself rather than from the detail beside it. Rich hands
        # flexible columns equal width otherwise, and the detail is the part
        # someone is reading.
        table.add_column(style="dim", overflow="fold", max_width=32)
        for check in checks:
            glyph, colour = GLYPHS[check.status]
            table.add_row(f"[{colour}]{SYMBOLS[glyph]}[/{colour}]",
                          escape(check.name), escape(check.detail),
                          escape(check.note))
        console.print(table)

    # The smoke test repeats an Elasticsearch failure so that `--only smoke`
    # still says what is wrong; list it once when both groups ran.
    problems, seen = [], set()
    for _, checks in groups:
        for c in checks:
            if c.status in (WARN, FAIL) and (c.detail, c.fix) not in seen:
                seen.add((c.detail, c.fix))
                problems.append(c)
    if not problems:
        console.print("\n[green]No problems found.[/green]")
        return

    console.print(f"\n[bold]{len(problems)} thing(s) to look at[/bold]")
    for check in problems:
        glyph, colour = GLYPHS[check.status]
        console.print(f"\n[{colour}]{SYMBOLS[glyph]}[/{colour}] "
                      f"[bold]{escape(check.name)}[/bold]: {escape(check.detail)}")
        if check.blocks:
            console.print(f"  [dim]breaks:[/dim] {escape(check.blocks)}")
        if check.fix:
            console.print(f"  [dim]fix:[/dim]    {escape(check.fix)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ngec-doctor",
        description="Check an NGEC environment and report what is wrong with it.")
    parser.add_argument("--only", metavar="GROUP[,GROUP]",
                        help=f"run only these groups ({', '.join(GROUPS)})")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output, for pasting into a bug report")
    parser.add_argument("--smoke", action="store_true",
                        help="also run three news stories through the whole "
                             "pipeline (takes a few minutes on a CPU)")
    args = parser.parse_args(argv)

    selected = DEFAULT_GROUPS + (["smoke"] if args.smoke else [])
    if args.only:
        selected = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = [name for name in selected if name not in GROUPS]
        if unknown:
            parser.error(f"unknown group(s): {', '.join(unknown)}; "
                         f"choose from {', '.join(GROUPS)}")

    groups = [(GROUPS[name][0], GROUPS[name][1]()) for name in selected]

    if args.json:
        json.dump({title: [asdict(c) for c in checks] for title, checks in groups},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        render(groups)

    # Only FAIL is non-zero: a warning is something to look at, not a reason for
    # a CI job that runs doctor as a smoke test to go red.
    return 1 if any(c.status == FAIL for _, checks in groups for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
