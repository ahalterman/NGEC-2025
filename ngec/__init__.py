
import warnings

from huggingface_hub import snapshot_download

from .event_class import EventClass
from .actor_resolution import ActorResolver
from .geolocation import GeolocationModel
from .attribute_model import AttributeModel
from .formatter import Formatter
from .utilities import load_nlp
from .logging import setup_logging

def _is_spacy_model_installed(model):
    try:
        from spacy import load
        load(model)
        return True
    except (OSError):
        return False


def _check_spacy_models(models):
    installed = [_is_spacy_model_installed(m) for m in models]

    if all(installed):
        return

    # Construct warning message
    msg = (
        "\n" + "=" * 70 + "\n" 
        + "WARNING: One or more required spaCy models are not installed:\n\n"
        + "".join([f" - {m}\n" for m, inst in zip(models, installed) if not inst])
        + "\nThese models are required for NGEC to fully function."
        "\n" + "=" * 70 + "\n" 
    )

    warnings.warn(msg, UserWarning, stacklevel=2)


def _is_huggingface_model_installed(model):
    pass


def _check_huggingface_models(models):
    pass


def download_actor_resolver_model() -> None:
    snapshot_download(
        repo_id="jinaai/jina-embeddings-v3",
        revision="main",          # or a commit hash
    )


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False




# Check on import
_ = _check_spacy_models(["en_core_web_lg", "en_core_web_trf"])