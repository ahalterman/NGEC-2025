import pytest
import torch

from ngec.actors.common import ModelManager, TextPreProcessor
from ngec.actors.wiki_matcher import WikiClient, WikiMatcher, WikiSearcher

@pytest.mark.external
def test_WikiClient(es_client_external):
    wiki_client = WikiClient(es_client = es_client_external)


def test_WikiMatcher_local(es_client_local):
    model_manager = ModelManager(base_path = "ngec/assets", device="gpu" if torch.cuda.is_available() else "cpu")
    nlp = model_manager.load_spacy_lg()
    trf = model_manager.load_trf_model()
    actor_sim = model_manager.load_actor_sim_model()
    text_processor = TextPreProcessor()
    wiki_client = WikiClient(es_client = es_client_local)

    wiki_searcher = WikiSearcher(
            wiki_client, 
            text_processor
        )

    

    wiki_matcher = WikiMatcher(
            base_path = "ngec/assets",
            wiki_searcher=wiki_searcher, 
            text_processor=text_processor, 
            trf_model=trf, 
            actor_sim_model=actor_sim, 
            device=model_manager.device, 
            nlp=nlp,
            wiki_sort_method="neural"
        )

    res = wiki_matcher.query_wiki("Obama",
                            context = "Michelle Obama is the former First Lady of the United States.",
                            method = "rules")