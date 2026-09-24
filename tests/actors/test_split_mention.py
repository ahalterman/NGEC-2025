"""`split_mention` takes a mention apart without touching Elasticsearch.

These run on spaCy and the country list alone, so they are in the default
(fast) test set. The end-to-end behavior of the split -- that searching the
core with the description as evidence finds the article -- needs the wiki
index and lives in the demo's check_demo.py.
"""

import pytest
import spacy

from ngec.actors.actor_resolution import split_mention
from ngec.actors.common import CountryDetector


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_lg")


@pytest.fixture(scope="module")
def detector():
    return CountryDetector()


def test_title_country_name(nlp, detector):
    split = split_mention("former US Secretary of State Colin Powell", nlp, detector)
    assert split["country"] == "USA"
    assert split["country_name"] == "United States"
    assert split["trimmed_text"] == "former Secretary of State Colin Powell"
    assert split["core_query"] == "Colin Powell"
    assert split["actor_desc"] == "former Secretary of State"
    assert split["ner_extracted_specific"] is True
    # Both other surface forms are offered for the second search, raw span first.
    assert split["alt_query_terms"] == ["former US Secretary of State Colin Powell",
                                        "former Secretary of State Colin Powell"]


def test_name_after_title(nlp, detector):
    split = split_mention("the governor of Mexico State, Enrique Pena Nieto", nlp, detector)
    assert split["country"] == "MEX"
    assert split["core_query"] == "Enrique Pena Nieto"
    assert split["actor_desc"] == "the governor of State"


def test_possessive_country_mid_span(nlp, detector):
    split = split_mention("southern Mexico's Zapatista rebel group", nlp, detector)
    assert split["country"] == "MEX"
    assert split["trimmed_text"] == "southern Zapatista rebel group"
    # No PERSON or ORG to cut out, so the whole trimmed span is the query.
    assert split["core_query"] == "southern Zapatista rebel group"
    assert split["actor_desc"] == ""


def test_plain_name_is_left_alone(nlp, detector):
    split = split_mention("Angela Merkel", nlp, detector)
    assert split["country"] is None
    assert split["country_name"] == ""
    assert split["core_query"] == "Angela Merkel"
    assert split["actor_desc"] == ""
    assert split["alt_query_terms"] == []


def test_country_only(nlp, detector):
    split = split_mention("Israeli", nlp, detector)
    assert split["country"] == "ISR"
    assert split["trimmed_text"] == ""
    assert split["doc"] is None


def test_country_from_context_beats_span(nlp, detector):
    # The span names no country; the passage around it does.
    split = split_mention("the defence ministry", nlp, detector,
                          context="Ghana's parliament summoned the defence ministry on Monday.")
    assert split["country"] is None
    assert split["country_name"] == "Ghana"


def test_caller_country_wins(nlp, detector):
    split = split_mention("the defence ministry", nlp, detector,
                          context="Ghana's parliament summoned the defence ministry.",
                          known_country="Togo")
    assert split["country_name"] == "Togo"


def test_degenerate_core_is_rejected(nlp, detector):
    # spaCy tends to tag "House" alone; that is a worse query than the span.
    split = split_mention("House Judiciary", nlp, detector)
    assert split["core_query"] == "House Judiciary"
    assert split["actor_desc"] == ""


# ---------------------------------------------------------------------------
# The v3 splitter (ngec/actors/mention_split_v3.py). It is committed but not
# switched on, so these tests ask for it by name.


def test_default_splitter_is_v1():
    """The pipeline default. Changing it changes every actor the coder resolves."""
    from ngec.actors import actor_resolution
    assert actor_resolution.SPLITTER == "v1"


@pytest.mark.substantive
def test_v3_keeps_the_description_v1_loses(nlp, detector, es_client_local):
    """
    The case the v3 rewrite exists for: v1's NER core swallows part of the
    description ("former economist Colin Powell") and leaves `actor_desc`
    empty, so the ranker never sees what kind of Colin Powell this is.

    v3 embeds the agents file, so it is slow to build and needs Elasticsearch;
    the `es_client_local` fixture is here to skip the test when there is none.
    """
    v1 = split_mention("former British economist Colin Powell", nlp, detector,
                       splitter="v1")
    assert v1["core_query"] == "former economist Colin Powell"
    assert v1["actor_desc"] == ""

    v3 = split_mention("former British economist Colin Powell", nlp, detector,
                       splitter="v3")
    assert v3["core_query"] == "Colin Powell"
    assert v3["actor_desc"] == "former British economist"
    assert v3["country_name"] == "United Kingdom"
    # an ISO-3 code, like v1's, not the name: the agent matcher reads it
    assert v3["country"] == "GBR"


@pytest.mark.substantive
def test_v3_leaves_a_bare_name_alone(nlp, detector, es_client_local):
    v3 = split_mention("Angela Merkel", nlp, detector, splitter="v3")
    assert v3["core_query"] == "Angela Merkel"
    assert v3["actor_desc"] == ""
