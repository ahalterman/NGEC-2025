from .actors.actor_resolution import ActorResolver
from .geolocation import GeolocationModel
from .attribute_model import AttributeModel
from .formatter import Formatter
from .models import ModelNotInstalledError, download_models
from .utilities import load_nlp
from .logging import setup_logging


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
