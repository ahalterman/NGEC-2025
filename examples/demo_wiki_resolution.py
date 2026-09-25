"""
Wikipedia linking and actor coding on their own, without the event pipeline.

The full pipeline (`PloverCoder`) finds events and then codes the actor and
recipient of each one. This script shows the actor step by itself, for the
case where you already have names or short descriptions of actors and want
to know (a) which Wikipedia article a name refers to and (b) what PLOVER
country and sector codes it gets.

It shows:

  1. Connecting to Elasticsearch, which holds the offline Wikipedia index.
  2. Linking a name, read in context, to its Wikipedia article.
  3. Coding an actor mention into a country and a PLOVER sector code.
  4. Coding a description ("Latvian air force") with the PLOVER agents file
     alone, which does not need Elasticsearch.
  5. Linking every person and organization in one news story.

Before running it:
  - install NGEC (see README.md at the top of the repo)
  - have Elasticsearch running with the `wiki` index loaded
  - if Elasticsearch is not on localhost:9200, or needs a password, put
    ES_HOST / ES_PORT / ES_USER / ES_PASSWORD in a .env file

Run it from the top of the repo:

    uv run python examples/demo_wiki_resolution.py
"""

import logging
import os

import pandas as pd

from ngec import ActorResolver
from ngec.actors.agent_matcher import AgentMatcher
from ngec.es_client import es_client_from_env

# Elasticsearch and the model loaders log a lot at INFO level. Only show
# warnings.
logging.basicConfig(level=logging.WARNING)


#######################################################
# 1. Connect to Elasticsearch
#######################################################

# es_client_from_env reads ES_HOST, ES_PORT, ES_USER and ES_PASSWORD from a
# .env file if there is one, and otherwise connects to localhost:9200.
es_client = es_client_from_env()
print("Connected to Elasticsearch", es_client.info()["version"]["number"])

# Load the actor resolver. This loads a spaCy model and several small
# sentence-embedding models, so it takes a little while.
# device="cpu" keeps the models on the CPU even on a machine with a GPU.
resolver = ActorResolver(es_client=es_client, device="cpu")


#######################################################
# 2. Link a name to its Wikipedia article
#######################################################

# `query_wiki` searches the Wikipedia index for the name and uses the
# surrounding sentence (`context`) to pick among the candidate articles. It
# returns the whole article as a dict, or None if no article is a good enough
# match.
print("\n=== Linking names to Wikipedia ===\n")

examples = [
    # A surname, with the full name elsewhere in the context.
    ("Blinken", "US Secretary of State Antony Blinken arrived in Kyiv on Wednesday."),
    # The same ambiguous surname, with context pointing to three different people.
    ("Bush", "Former president George H. W. Bush died on Friday. Bush was 94."),
    ("Bush", "President George W. Bush spoke on Tuesday. Bush said the war would continue."),
    ("Bush", "Kate Bush released her first album in ten years."),
    # A surname with no full name in the context. The linker does much better
    # when the full name appears somewhere in the context, as it usually does
    # in a news story. Here it picks the article about the plant.
    ("Bush", "Bush ordered the invasion of Panama in 1989."),
]

for name, context in examples:
    wiki = resolver.wiki_matcher.query_wiki(name, context=context)
    print(context)
    if wiki is None:
        print(f"    {name} ---> no match\n")
    else:
        print(f"    {name} ---> {wiki['title']} ({wiki.get('short_desc', '')})\n")

# The article dict holds the article itself ('title', 'short_desc',
# 'intro_para', 'infobox', 'categories', ...), the ranker's score for it
# ('ranker_score', from 0 to 1), and the features the ranker used to score it.
wiki = resolver.wiki_matcher.query_wiki("Blinken", context=examples[0][1])
print("Title:       ", wiki["title"])
print("Ranker score:", round(wiki["ranker_score"], 3))
print("Intro:       ", wiki["intro_para"][:150], "...")
print("Infobox keys:", list(wiki["infobox"].keys())[:8], "...")


#######################################################
# 3. Code an actor mention: country + PLOVER sector
#######################################################

# `actor_to_code` is what the pipeline itself runs on each actor and
# recipient. It links the mention to Wikipedia where that makes sense, reads
# the country and role from the article (for a person, the office they held
# on `query_date`), and also matches the text against the PLOVER agents file.
# It returns the best code as a dict, or None.
print("\n=== Coding actor mentions ===\n")

mentions = [
    ("Blinken", "US Secretary of State Antony Blinken arrived in Kyiv on Wednesday."),
    ("the Kenyan police", ""),
    ("Hamas", ""),
    ("protesters", "Protesters gathered outside parliament in Tbilisi."),
]

for text, context in mentions:
    code = resolver.actor_to_code(text, context=context, query_date="2023-06-01")
    if code is None:
        print(f"{text:<20} ---> no code")
    else:
        print(f"{text:<20} ---> country={code['country'] or '-':<4} "
              f"code_1={code['code_1'] or '-':<4} code_2={code['code_2'] or '-':<4} "
              f"wiki={code['wiki'] or '-'}")

# A person's code depends on the date: the same article gives a different
# role before and after they took office.
print()
for date in ["2012-01-01", "2023-06-01"]:
    code = resolver.actor_to_code("Rishi Sunak", query_date=date)
    print(f"Rishi Sunak on {date} ---> country={code['country']} code_1={code['code_1']}")


#######################################################
# 4. Code a short description without Elasticsearch
#######################################################

# Short descriptions of kinds of actors ("the air force", "protesters") are
# coded by comparing the text with the patterns in the PLOVER agents file
# (ngec/assets/PLOVER_agents.txt). `AgentMatcher` does this on its own and
# does not need Elasticsearch. It takes a country out of the text first
# ("Latvian" -> LVA) and matches the rest.
#
# (The resolver above already contains one, `resolver.agent_matcher`. We make
# a separate one here to show that this part works without the resolver or
# Elasticsearch.)
print("\n=== Coding descriptions with the agents file ===\n")

agent_matcher = AgentMatcher(device="cpu")

for text in ["Latvian air force", "protesters", "Nigerian opposition lawmakers",
             "Chinese foreign ministry spokesman", "a local farmer"]:
    code = agent_matcher.short_text_to_agent(text)
    if code is None:
        print(f"{text:<35} ---> no match")
    else:
        print(f"{text:<35} ---> country={code['country'] or '-':<4} "
              f"code_1={code['code_1']:<4} (closest pattern: '{code['pattern']}', "
              f"similarity {code['conf']:.2f})")


#######################################################
# 5. Link every person and organization in a news story
#######################################################

# Guardian_SDF_sample.csv.zip, next to this script, has 249 Guardian stories
# that mention the SDF. Take the first one, find the people and organizations
# in it with spaCy, and link each one to Wikipedia, using its sentence as the
# context. The linker makes mistakes, most often on bare acronyms and on names
# that are also ordinary words or the names of other things, so check its
# output before relying on it.
print("\n=== Linking the names in one Guardian story ===\n")

here = os.path.dirname(os.path.abspath(__file__))
stories = pd.read_csv(os.path.join(here, "Guardian_SDF_sample.csv.zip"))
story = stories["text"][0]

doc = resolver.nlp(story)
already_done = set()
for ent in doc.ents:
    if ent.label_ not in ["PERSON", "ORG"] or ent.text in already_done:
        continue
    already_done.add(ent.text)
    wiki = resolver.wiki_matcher.query_wiki(ent.text, context=ent.sent.text)
    if wiki is None:
        print(f"{ent.text:<30} ---> no match")
    else:
        print(f"{ent.text:<30} ---> {wiki['title']}")
