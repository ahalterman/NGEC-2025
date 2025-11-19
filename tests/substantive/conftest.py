import pytest

from ngec import ActorResolver
from ngec.es_client import ESConfig, es_is_available


@pytest.fixture(scope='session', autouse=True)
def ag():
    if not es_is_available(ESConfig())[0]:
        pytest.skip("Elasticsearch not available, skipping actor resolution tests.", allow_module_level=True)
    return ActorResolver()



