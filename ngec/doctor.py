"""NGEC Doctor: a command-line tool to check your environment and diagnose problems.

Run it as `ngec-doctor`, or as `python -m ngec.doctor`.

Implemented here:

- Installation: the ngec version and commit, the Python running it, and where
  the package is being imported from
- Configuration: every environment variable ngec or its tooling reads, the
  effective value, and which code actually reads it
- Compute: the PyTorch build, whether it can really see the GPU that is
  present, and what `gpu=True` will do on this machine

Still to come, roughly in this order: spaCy models and packaged assets;
Elasticsearch reachability and the `wiki` and `geonames` index contents; the
available LLM backends, including whether `llama-server` is serving the model
the Python side is prompting for.

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
import json
import os
import platform
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
    return Check(".env", OK, f"{len(_FROM_DOTENV)} setting(s) loaded",
                 note=str(env_file))


def _default_attribute_model() -> str:
    """The model AttributeModel falls back to, asked of the code that defines it."""
    try:
        from .attribute_model import DEFAULT_MODEL

        return DEFAULT_MODEL
    except Exception:  # noqa: BLE001 - a broken import is the compute group's finding
        return "the package default"


def _setting(name: str, default: str, read_by: str, secret: bool = False) -> Check:
    raw = os.environ.get(name)
    if raw is None:
        detail = f"unset, defaulting to {default}" if default else "unset"
    elif secret:
        # Doctor output is the sort of thing that gets pasted into an issue.
        detail = "set"
    else:
        detail = raw
    if name in _FROM_DOTENV:
        detail += " (from .env)"
    return Check(name, INFO, detail, note=read_by)


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
    # Built here rather than at module scope so the attribute model's own
    # default can be reported instead of a second copy of the name.
    settings = [
        ("NGEC_ATTRIBUTE_MODEL", _default_attribute_model(), "ngec.attribute_model", False),
        ("NGEC_LLAMACPP_URL", "http://127.0.0.1:8080", "ngec.llm.llamacpp", False),
        ("ES_HOST", "localhost", "tests, demo", False),
        ("ES_PORT", "9200", "tests, demo", False),
        ("ES_USER", "", "tests, demo", True),
        ("ES_PASSWORD", "", "tests, demo", True),
        ("NGEC_ES_URL", "http://localhost:9200/", "tools/, elasticsearch/", False),
        ("NGEC_ES_DATA", "", "elasticsearch/compose-build.yml", False),
        ("NGEC_REDIS_HOST", "localhost", "elasticsearch/es_wiki", False),
        ("NGEC_REDIS_PORT", "6379", "elasticsearch/es_wiki", False),
        ("HF_HOME", "~/.cache/huggingface", "huggingface_hub", False),
    ]

    checks = [_dotenv()]
    checks += [_setting(*setting) for setting in settings]
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


# --------------------------------------------------------------------- output


GROUPS: dict[str, tuple[str, object]] = {
    "install": ("Installation", installation),
    "config": ("Configuration", configuration),
    "compute": ("Compute", compute),
}

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

    problems = [c for _, checks in groups for c in checks
                if c.status in (WARN, FAIL)]
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
    args = parser.parse_args(argv)

    selected = list(GROUPS)
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
