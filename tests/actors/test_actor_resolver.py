

import pytest 

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
        {"pub_date": "2007-07-01",
         # attribute model output, but only minimal subset needed here
         "attributes": {
             'actor': ['President Macron', 'Chancellor Angel Merkel'],
             'recipient': ['N/A']
         }},
    ]

    res = resolver.process(input)
    assert res[0]['actor'] is not None
    assert res[0]['recipient'] is not None




