import logging

import pytest

from ngec.attribute_model import AttributeModel
from ngec.es_client import setup_es_client
from ngec.logging import setup_logging

setup_logging(level=logging.WARNING, quiet_third_party=True)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def attribute_model():
    am = AttributeModel(silent=True, gpu=False, backend="transformers")
    return am


@pytest.fixture(scope="session")
def es_client_external():
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        ES_HOST = os.getenv("ES_HOST", "localhost")
        ES_PORT = int(os.getenv("ES_PORT", "9200"))
        ES_USER = os.getenv("ES_USER", None)
        ES_PASSWORD = os.getenv("ES_PASSWORD", None)
        client = setup_es_client(hosts=[ES_HOST], port=ES_PORT, http_auth=(ES_USER, ES_PASSWORD))

        # Forece actual (non-lazy) connection so we catch if ES is not running at all
        client.info()

        return client
    except Exception as e:
        logger.error(f"Failed to set up external ES client: {e}")
        pytest.skip("Elasticsearch client (external) setup failed")


@pytest.fixture(scope='session')
def es_client_local():
    try:
        client = setup_es_client(hosts=["localhost"], port=9200)

        # Force actual (non-lazy) connection so we catch if ES is not running at all
        # TODO: this should be more robust, i need a utility to also check data availability
        client.info()

        return client
    except Exception as e:
        logger.error(f"Failed to set up local ES client: {e}")
        pytest.skip("Elasticsearch client (local) setup failed")


