import pytest

from ngec.actors.wiki_matcher import WikiClient

@pytest.mark.external
def test_WikiClient(es_client_external):
    wiki_client = WikiClient(es_client = es_client_external)

