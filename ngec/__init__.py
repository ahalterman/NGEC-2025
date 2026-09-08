from huggingface_hub import snapshot_download

from .actors.actor_resolution import ActorResolver
from .geolocation import GeolocationModel
from .attribute_model import AttributeModel
from .formatter import Formatter
from .models import ModelNotInstalledError, download_models
from .utilities import load_nlp
from .logging import setup_logging


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
