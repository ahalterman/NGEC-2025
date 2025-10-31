# Just a smoke test for the actor resolver

import os
import pytest 
from dotenv import load_dotenv

from ngec import ActorResolver

# Load environment variables from .env file
load_dotenv()

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