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

def test_merge_ranked_results_keeps_primary_copy_of_shared_article():
    """An article both searches found carries the primary search's copy.

    The alternate copy has that query's ES score and from_alt_query=1; handing
    it to the ranker in place of the primary's made "Colin Powell", ranked
    third by the primary and first by the alternate, look like a weak
    alternate-only candidate.
    """
    from ngec.actors.wiki_matcher import merge_ranked_results

    primary = [{"title": "A", "raw_es_score": 100, "from_alt_query": 0},
               {"title": "B", "raw_es_score": 90, "from_alt_query": 0},
               {"title": "C", "raw_es_score": 80, "from_alt_query": 0}]
    alternate = [{"title": "C", "raw_es_score": 5, "from_alt_query": 1},
                 {"title": "D", "raw_es_score": 4, "from_alt_query": 1}]
    merged = merge_ranked_results(primary, alternate, 10)
    assert [a["title"] for a in merged] == ["A", "C", "B", "D"]  # interleaved, C early
    c = next(a for a in merged if a["title"] == "C")
    assert c["raw_es_score"] == 80 and c["from_alt_query"] == 0  # the primary's copy
    d = next(a for a in merged if a["title"] == "D")
    assert d["from_alt_query"] == 1
