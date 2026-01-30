import pytest
import torch

from ngec.actors.common import ModelManager
from ngec.actors.wiki_matcher import WikiClient, WikiMatcher

@pytest.mark.external
def test_WikiClient_instantiates(es_client_external):
    wiki_client = WikiClient(es_client = es_client_external)


def test_WikiMatcher_local(es_client_local):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = ModelManager(device=device)
    
    wiki_matcher = WikiMatcher(
        es_client=es_client_local,
        model_manager=model_manager,
        wiki_sort_method="neural",
        device=device
        )

    res = wiki_matcher.query_wiki("Obama",
                            context = "Michelle Obama is the former First Lady of the United States.",
                            method = "rules")