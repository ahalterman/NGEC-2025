import logging
import pytest

from ngec.attribute_model import AttributeModel
from ngec.logging import setup_logging

setup_logging(level=logging.WARNING, quiet_third_party=True)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def attribute_model():
    am = AttributeModel(silent=True, gpu=False, backend="transformers")
    return am

