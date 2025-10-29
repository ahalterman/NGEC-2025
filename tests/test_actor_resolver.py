# Just a smoke test for the actor resolver

from ngec import ActorResolver
es_config = {
                "es_host": "167.71.184.1",
                "es_port": 9200,
                "es_user": "student",
                "es_password": "TJGrBa668cdKGKtLYF"
            }

resolver = ActorResolver(es_config=es_config)
res = resolver.actor_to_code("Angela Merkel")

print(res)

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