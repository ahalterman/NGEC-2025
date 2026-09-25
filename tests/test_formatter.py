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
