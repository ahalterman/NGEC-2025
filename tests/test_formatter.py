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


def test_process_writes_resolved_dates_to_jsonl(tmp_path):
    # The resolved dates are datetimes, which the json module cannot write.
    # The file gets ISO strings; the returned events keep their datetimes.
    import datetime as dt
    import json
    from ngec.formatter import Formatter

    events = [{"id": "a", "pub_date": "2016-05-01",
               "attributes": {"date": ["yesterday"], "location": []},
               "geolocated_ents": []}]
    out = Formatter(output_dir=str(tmp_path)).process(events)
    assert out[0]["date_resolved"]["resolved_date"] == dt.datetime(2016, 4, 30)

    with open(tmp_path / "events_processed.jsonl") as f:
        written = [json.loads(line) for line in f]
    assert written[0]["date_resolved"]["resolved_date"] == "2016-04-30T00:00:00"


# Place names as the geoparser (mordecai3) returned them for short news
# sentences, trimmed to the fields pick_event_loc reads.
_NAIROBI = [{"search_name": "Nairobi", "name": "Nairobi", "score": 1.0},
            {"search_name": "Kenya", "name": "Republic of Kenya", "score": 1.0}]
_GUERRERO = [{"search_name": "Guerrero", "name": "Estado de Guerrero", "score": 1.0},
             {"search_name": "Mexico City", "name": "Mexico City", "score": 1.0}]
_TBILISI = [{"search_name": "Tbilisi", "name": "Tbilisi", "score": 1.0},
            {"search_name": "Georgia", "name": "Georgia", "score": 1.0}]
_CHAD = [{"search_name": "Chad", "name": "Republic of Chad", "score": 0.999},
         {"search_name": "West Darfur", "name": "West Darfur", "score": 0.878}]


def _picked(span, ents):
    return (pick_event_loc(_location_search_term([span]), ents)["event_loc"] or {}).get("name")


def test_place_named_inside_a_longer_span_is_found():
    # A span with more than one leading preposition used to be compared with
    # each place name as a whole and matched nothing.
    assert _picked("through central Nairobi", _NAIROBI) == "Nairobi"
    assert _picked("in the Mexican state of Guerrero", _GUERRERO) == "Estado de Guerrero"
    assert _picked("outside the parliament in Tbilisi", _TBILISI) == "Tbilisi"
    assert _picked("near the border with Chad", _CHAD) == "Republic of Chad"
    # Unchanged: a bare name, with or without a preposition.
    assert _picked("in Nairobi", _NAIROBI) == "Nairobi"
    assert _picked("Kenya", _NAIROBI) == "Republic of Kenya"


def test_contained_place_prefers_the_longest_name():
    ents = [{"search_name": "Darfur", "name": "Darfur", "score": 0.95},
            {"search_name": "West Darfur", "name": "West Darfur", "score": 0.95}]
    assert _picked("in West Darfur", ents) == "West Darfur"


def test_contained_place_must_be_a_whole_word():
    # "Chad" is not in "Chadian", "Niger" is not in "Nigeria", and the pronoun
    # "us" is not the United States.
    assert _picked("the Chadian border", _CHAD) is None
    assert _picked("northern Nigeria", [{"search_name": "Niger", "name": "Niger", "score": 1.0}]) is None
    assert _picked("near us", [{"search_name": "US", "name": "United States", "score": 1.0}]) is None
    assert _picked("near the US embassy", [{"search_name": "US", "name": "United States", "score": 1.0}]) == "United States"


def test_contained_place_still_needs_confidence():
    ents = [{"search_name": "Nairobi", "name": "Nairobi", "score": 0.4}]
    res = pick_event_loc("through central Nairobi", ents)
    assert res["event_loc"] is None
    assert res["reason"] == "no sufficient confidence in geo entity"
