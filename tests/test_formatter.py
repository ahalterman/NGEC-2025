from ngec.formatter import _location_search_term, pick_event_loc


def test_location_search_term_drops_a_leading_preposition():
    # The v6 attribute model keeps the word before a place name ("in Haiti").
    assert _location_search_term(["in Haiti"]) == "Haiti"
    assert _location_search_term(["near the Syrian border"]) == "the Syrian border"
    assert _location_search_term(["Inner Mongolia"]) == "Inner Mongolia"
    assert _location_search_term(["Kabul"]) == "Kabul"
    assert _location_search_term([]) is None
    assert _location_search_term(None) is None


def test_prepositioned_location_matches_the_geoparsed_place():
    ents = [{"search_name": "Haiti", "score": 0.99, "name": "Republic of Haiti"}]
    loc = pick_event_loc(_location_search_term(["in Haiti"]), ents)
    assert loc["event_loc"]["name"] == "Republic of Haiti"
