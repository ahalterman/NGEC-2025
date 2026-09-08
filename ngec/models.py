"""Getting the spaCy models NGEC needs, and failing usefully when they're missing.

NGEC parses with `en_core_web_trf` and `en_core_web_lg` -- about 900 MB of
weights that installing the package cannot bring in, because spaCy publishes its
models as wheels hosted on GitHub rather than on PyPI. 

Prior to September 2026, the way this was handled was by having an extra called
`models` in `pyproject.toml` for those two models + a `tools.uv.sources` section
pointing to the wheel URLs. This meant that:

- Installing `ngec` only worked with `uv`, not pip.
- Installing the models along with `ngec` required invoking `ngec[models]` in 
  addition to other extras for the backends.
- Would not have worked with PyPI, which does not host the large spaCy model 
  wheels and doesn't allow URL requirements from published packages.

Another issue is that the runtime check for whether the spacy models were
downloaded was by loading them, which is low and unneccessary.   

This is now replaced by an explicit `download_models()` step and a metadata 
check via `installed_spacy_models()`.

`uv` sync for dev purposes still works, because the models were moved to a new
dependency group called `models`, that is installed along with `dev` by default.
"""

import importlib
import importlib.util
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#: The models the pipeline loads. `en_core_web_trf` does the parsing in
#: `load_nlp()`; `en_core_web_lg` supplies the word vectors the actor matcher
#: compares against.
REQUIRED_SPACY_MODELS = ("en_core_web_lg", "en_core_web_trf")


class ModelNotInstalledError(OSError):
    """A required spaCy model is not installed.

    Subclasses `OSError` because that is what `spacy.load()` raises for a model
    it cannot find, so anything already handling that keeps working; the only
    difference is that the message says how to fix it.
    """


def _install_hint(models) -> str:
    models = list(models)
    if len(models) == 1:
        subject = f"The spaCy model {models[0]} is not installed"
    else:
        subject = f"The spaCy models {', '.join(models)} are not installed"
    return (
        f"{subject}. NGEC needs it, and it is not on PyPI, so installing ngec "
        "does not bring it in. Install it with:\n\n"
        "    ngec download-models\n\n"
        "which fetches both models NGEC uses (about 900 MB together). "
        "`python -m spacy download <model>` installs the same thing one at a time."
    )


def installed_spacy_models() -> set[str]:
    """Names of the spaCy models installed in this environment.

    Reads package metadata, so it costs nothing next to the load it guards.
    """
    try:
        from spacy.util import get_installed_models
        return set(get_installed_models())
    except Exception:
        # spaCy itself is unusable, so none of its models are either. Reporting
        # them as missing is both true and more useful than raising from here.
        return set()


def missing_spacy_models(models=REQUIRED_SPACY_MODELS) -> list[str]:
    """Which of `models` are not installed, in the order given."""
    installed = installed_spacy_models()
    return [m for m in models if m not in installed]


def load_spacy(name: str):
    """`spacy.load(name)`, with a missing model reported as an actionable error.

    The one place NGEC loads a spaCy model by name, so the "you never downloaded
    this" message is written once. A load can fail for reasons that have nothing
    to do with the model being absent -- a truncated download, a torch problem --
    so the metadata check only decides which of the two errors to raise; it never
    stands in for the load itself.
    """
    import spacy

    try:
        return spacy.load(name)
    except OSError as exc:
        if name in installed_spacy_models():
            raise
        raise ModelNotInstalledError(_install_hint([name])) from exc


def download_models(models=REQUIRED_SPACY_MODELS, force: bool=False) -> None:
    """Download and install the spaCy models NGEC needs.

    Skips models that are already installed unless `force` is set. This is
    spaCy's own downloader, which pip-installs the model wheel from GitHub --
    exactly what `python -m spacy download` does, so a model installed either
    way is the same package.

    Also reachable as `ngec download-models`.
    """
    # spaCy's downloader shells out to `sys.executable -m pip`, and a uv-created
    # venv has no pip in it. Saying so here beats the "No module named pip"
    # traceback that comes back from three frames down otherwise.
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError(
            "Downloading spaCy models needs pip, which is not installed in this "
            "environment (uv does not install one by default). Either add it "
            "(`uv pip install pip`) and re-run, or install the models with "
            "`uv run --with pip python -m spacy download <model>`.")

    from spacy.cli.download import download

    already = installed_spacy_models()
    for model in models:
        if model in already and not force:
            logger.info("%s is already installed, skipping.", model)
            continue
        logger.info("Downloading %s...", model)
        download(model)

    # The models were installed into an interpreter that has already scanned
    # sys.path, so anything asking about them next (including `load_spacy`)
    # needs the import machinery to look again.
    importlib.invalidate_caches()
