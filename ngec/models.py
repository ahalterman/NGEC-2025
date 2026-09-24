"""Getting the models NGEC needs, and failing usefully when they're missing.

`download_models()` (also `ngec download-models`) fetches everything the
pipeline loads by name, about 3 GB together, so that the first pipeline run is
not also a download:

- the two spaCy models (about 900 MB), which need special handling -- see below;
- the attribute-extraction LLM (about 1.2 GB for the default);
- three sentence encoders (about 700 MB together): the one the event
  classifiers were trained with, and the two the actor matcher uses.

The LLM and the encoders come from the Hugging Face hub and land in its cache
(`~/.cache/huggingface`, or `$HF_HOME`), which is where the pipeline looks for
them. Without this step they would download the first time something loads
them, which works, but means that a first pipeline run spends most of its time
downloading, and that an offline machine fails halfway through a run.

The spaCy models
----------------

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

This is now replaced by an explicit `download_spacy_models()` step and a
metadata check via `installed_spacy_models()`.

`uv` sync for dev purposes still works, because the models were moved to a new
dependency group called `models`, that is installed along with `dev` by default.
"""

import importlib
import importlib.util
import json
import logging
from importlib import resources
from pathlib import Path

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
        "which fetches both spaCy models (about 900 MB together) along with the "
        "other models NGEC uses. `python -m spacy download <model>` installs the "
        "same thing one spaCy model at a time."
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


def download_spacy_models(models=REQUIRED_SPACY_MODELS, force: bool = False) -> None:
    """Download and install the spaCy models NGEC needs.

    Skips models that are already installed unless `force` is set. This is
    spaCy's own downloader, which pip-installs the model wheel from GitHub --
    exactly what `python -m spacy download` does, so a model installed either
    way is the same package.
    """
    already = installed_spacy_models()
    to_install = []
    for model in models:
        if model in already and not force:
            logger.info("%s is already installed, skipping.", model)
        else:
            to_install.append(model)
    if not to_install:
        return

    # spaCy's downloader shells out to `sys.executable -m pip`, and a uv-created
    # venv has no pip in it. Saying so here beats the "No module named pip"
    # traceback that comes back from three frames down otherwise. Checked only
    # once there is something to install, so that a venv without pip whose
    # models are all present is not told to go and fix anything.
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError(
            "Downloading spaCy models needs pip, which is not installed in this "
            "environment (uv does not install one by default). Either add it "
            "(`uv pip install pip`) and re-run, or install the models with "
            "`uv run --with pip python -m spacy download <model>`.")

    from spacy.cli.download import download

    for model in to_install:
        logger.info("Downloading %s...", model)
        download(model)

    # The models were installed into an interpreter that has already scanned
    # sys.path, so anything asking about them next (including `load_spacy`)
    # needs the import machinery to look again.
    importlib.invalidate_caches()


def classifier_encoder_name() -> str:
    """The Hugging Face id of the encoder the shipped event classifiers use.

    The classifier models are self-describing: the encoder they were trained
    with is recorded in their metadata.json. This reads it the same way
    `PloverSklearnClassifier` does for its default model directory.
    """
    from .classifiers.plover_sklearn import DEFAULT_ENCODER

    path = resources.files("ngec").joinpath("assets/event_models_v2/metadata.json")
    with path.open("r", encoding="utf-8") as f:
        name = json.load(f).get("encoder") or DEFAULT_ENCODER
    # PloverSklearnClassifier loads f"sentence-transformers/{encoder_name}".
    return f"sentence-transformers/{name}"


def download_encoders() -> None:
    """Download the sentence encoders the pipeline uses.

    These are the classifier encoder (step 1 of the pipeline) and the actor
    matcher's Wikipedia and agent encoders (step 5). The actor encoders honour
    the NGEC_WIKI_ENCODER and NGEC_AGENT_ENCODER environment variables, as
    `ModelManager` does.

    Each encoder is downloaded by loading it once, on the CPU, rather than by
    downloading its whole repository. sentence-transformers fetches only the
    files it needs, while the repositories also hold ONNX and OpenVINO copies
    of the weights: all of `all-mpnet-base-v2` is 3.8 GB, of which the
    pipeline uses 440 MB. Loading also runs any remote code a model needs
    (jina-embeddings-v3, if selected), so that is downloaded too.

    There is no `force` here: a cached encoder is checked against the hub
    whenever it is loaded online, and a changed file is downloaded again.
    """
    from sentence_transformers import SentenceTransformer

    from .actors.common import ModelManager

    name = classifier_encoder_name()
    logger.info("Downloading %s (event classification)...", name)
    SentenceTransformer(name, device="cpu")

    manager = ModelManager(device="cpu")
    logger.info("Downloading %s (Wikipedia actor matching)...", manager.encoder_name)
    manager.load_wiki_encoder()
    logger.info("Downloading %s (agent pattern matching)...", manager.agent_encoder_name)
    manager.load_trf_model()


def download_attribute_model(model_name: str | None = None, force: bool = False) -> None:
    """Download the attribute-extraction LLM into the Hugging Face cache.

    Which model is resolved exactly as `AttributeModel` resolves it:
    `model_name` if given, then NGEC_ATTRIBUTE_MODEL, then the default. A model
    given as a local directory has nothing to download.

    The vllm, transformers and mlx backends all load from the Hugging Face
    cache, so this covers all three. The llamacpp backend talks to a
    `llama-server` running a GGUF file, which this does not provide.

    Files already in the cache are only downloaded again if they have changed
    on the hub, or if `force` is set.
    """
    from huggingface_hub import snapshot_download

    from .attribute_model import resolve_model_name

    name = resolve_model_name(model_name)
    if Path(name).expanduser().is_dir():
        logger.info("%s is a local directory, nothing to download.", name)
        return
    logger.info("Downloading %s (attribute extraction)...", name)
    snapshot_download(repo_id=name, force_download=force)


def download_models(force: bool = False, attribute_model: str | None = None,
                    include_attribute_model: bool = True) -> None:
    """Download every model the pipeline loads by name.

    That is the spaCy models, the sentence encoders and the attribute LLM;
    see the module docstring for sizes. Models already present are skipped.

    Args:
        force: Reinstall the spaCy models and re-download the attribute model
            even if they are already present. (See `download_encoders` for why
            the encoders need no such option.)
        attribute_model: Which attribute model to download, if not the one
            `AttributeModel` would use by default.
        include_attribute_model: Set to False to skip the attribute model, for
            example when it runs on a llama.cpp server.

    Also reachable as `ngec download-models`.
    """
    # The spaCy models go first so that a missing pip is reported straight
    # away rather than after the other downloads.
    download_spacy_models(force=force)
    download_encoders()
    if include_attribute_model:
        download_attribute_model(attribute_model, force=force)
