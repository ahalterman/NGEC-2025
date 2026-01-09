# Just a smoke test for the actor resolver

import os

import pytest 
from dotenv import load_dotenv

from ngec import ActorResolver
from ngec.logging import setup_logging

setup_logging(quiet_third_party=True)


@pytest.mark.external
def test_actor_resolver(es_client_external):
    
    resolver = ActorResolver(es_client=es_client_external)
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


@pytest.mark.external
def test_actor_resolver_process(es_client_external):
    """Make sure the process method works; main entry point in production."""
    resolver = ActorResolver(es_client=es_client_external)
    
    input = [
        {'event_text': 'Turkish forces and Turkish-backed militias battled with YPG militants in Syria.', 'id': 789, '_doc_position': 2, 'event_type': 'ASSAULT', 'event_mode': '', 'attributes': {'event_type': 'ASSAULT', 'anchor_quote': 'Turkish forces and Turkish-backed militias battled with YPG militants in Syria.', 'actor': ['Turkish forces', 'Turkish-backed militias'], 'recipient': ['YPG militants'], 'date': ['N/A'], 'location': ['Syria']}}
    ]

    res = resolver.process(input)





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