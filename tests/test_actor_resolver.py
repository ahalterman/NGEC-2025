# Just a smoke test for the actor resolver

import os
import pytest 
from dotenv import load_dotenv

from ngec import ActorResolver
from ngec.logging import setup_logging

setup_logging(quiet_third_party=True)

# Load environment variables from .env file
load_dotenv()

@pytest.mark.external
def test_actor_resolver():
    es_config = {
        "es_host": os.getenv("ES_HOST", "localhost"),
        "es_port": int(os.getenv("ES_PORT", "9200")),
        "es_user": os.getenv("ES_USER"),
        "es_password": os.getenv("ES_PASSWORD")
    }
    
    # Skip test if credentials not provided
    if not es_config["es_user"] or not es_config["es_password"]:
        pytest.skip("Elasticsearch credentials not provided")
    
    resolver = ActorResolver(es_config=es_config)
    res = resolver.actor_to_code("Angela Merkel")

    # res["all_code1s"] may be in any order, sort it for consistent testing
    res["all_code1s"].sort()
    
    expected = {'pattern': 'NA', 
                'code_1': 'ELI', 
                'code_2': '', 
                'country': 'DEU', 
                'description': 'previously held a GOV role, so coded as ELI', 
                'source': 'Infobox', 
                'wiki': 'Angela Merkel', 
                'best_reason': 'Only one entry in the info box', 
                'all_code1s': ['ELI', 'GOV'], 
                'all_code2s': []}
    
    assert res == expected
def test_country_detector():
    from ngec.actor_resolution import CountryDetector
    cd = CountryDetector()
    res = cd.search_nat("There were also 5 Americans in the village.")
    assert res == ('USA', 'There were also 5 in the village.')

def test_country_detector_no_country():
    from ngec.actor_resolution import CountryDetector
    cd = CountryDetector()
    res = cd.search_nat("This text has no country mentioned.")
    assert res == (None, 'This text has no country mentioned.')