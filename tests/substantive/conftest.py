import pytest

from ngec import ActorResolver


@pytest.fixture(scope='session', autouse=True)
def ag(es_client_local):
    return ActorResolver(es_client=es_client_local)
