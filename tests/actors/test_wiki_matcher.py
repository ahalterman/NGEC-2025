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


def test_every_registered_ranker_asset_exists():
    """A WIKI_ENCODERS entry naming a ranker that is not in assets/ would fall
    back silently (the context ranker) or fail at load time."""
    from importlib import resources
    from pathlib import Path
    from ngec.actors.common import WIKI_ENCODERS
    assets = Path(str(resources.files("ngec"))) / "assets"
    for name, settings in WIKI_ENCODERS.items():
        for key in ("ranker_asset", "ranker_asset_no_context"):
            if key in settings:
                assert (assets / settings[key]).exists(), f"{name}: {settings[key]} missing"


def test_no_context_ranker_is_its_own_model():
    """The default encoder ships a separate ranker for mentions without a
    story. It was fit without the context features, so it asks for fewer
    columns than the context ranker; loading the context ranker twice (the old
    behaviour) would make the two identical."""
    from importlib import resources
    from pathlib import Path
    from ngec.actors.common import WIKI_ENCODERS, DEFAULT_ENCODER
    from ngec.actors.wiki_matcher import load_wiki_ranker_model
    assets = Path(str(resources.files("ngec"))) / "assets"
    settings = WIKI_ENCODERS[DEFAULT_ENCODER]
    ctx, noctx = load_wiki_ranker_model(assets / settings["ranker_asset"],
                                        assets / settings["ranker_asset_no_context"])
    assert noctx is not ctx
    assert "context_sim_intro" in ctx.feature_names_in_
    assert "context_sim_intro" not in noctx.feature_names_in_
    # without a no-context path, the context ranker stands in
    ctx2, noctx2 = load_wiki_ranker_model(assets / settings["ranker_asset"])
    assert noctx2 is ctx2
