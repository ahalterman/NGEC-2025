
from ngec.es_client import es_is_available


def test_absense_of_es():
    es_config = {
        "hosts": "nonsense_host",
    }
    is_available, message = es_is_available(es_config)
    assert not is_available
    assert "Ping failed" in message