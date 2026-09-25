

import json
import os

from ngec.utilities import stories_to_events, explode_events, write_intermediate



def test_events_with_modes():
    story_list = [
        {
            "id": "story1",
            "event_type": ["ACCUSE", "CONSULT"],
            "event_mode": ["ACCUSE-disapprove", "ACCUSE-allege", "CONSULT-third-party"]
        },
        {
            "id": "story2",
            "event_type": ["PROTEST"],
            "event_mode": []
        }
    ]
    
    expected_output = [
        {'id': 'story1_ACCUSE_disapprove', 'event_type': 'ACCUSE', 'event_mode': 'disapprove', 'orig_id': 'story1'}, 
        {'id': 'story1_ACCUSE_allege', 'event_type': 'ACCUSE', 'event_mode': 'allege', 'orig_id': 'story1'}, 
        {'id': 'story1_CONSULT_third-party', 'event_type': 'CONSULT', 'event_mode': 'third-party', 'orig_id': 'story1'}, 
        {'id': 'story2_PROTEST_', 'event_type': 'PROTEST', 'event_mode': '', 'orig_id': 'story2'}\
    ]


    event_list = stories_to_events(story_list, doc_list=None)

    assert event_list == expected_output


def test_events_without_modes():
    story_list = [
        {
            "id": "story1",
            "event_type": ["ACCUSE", "CONSULT"],
            "event_mode": []
        }
    ]
    
    expected_output = [
        {'id': 'story1_ACCUSE_', 'event_type': 'ACCUSE', 'event_mode': '', 'orig_id': 'story1'}, 
        {'id': 'story1_CONSULT_', 'event_type': 'CONSULT', 'event_mode': '', 'orig_id': 'story1'}\
    ]

    event_list = stories_to_events(story_list, doc_list=None)

    assert event_list == expected_output


def test_explode_single_event():
    # One extracted sub-event -> one record with a dict 'attributes' and an
    # index-suffixed id.
    event_list = [
        {"id": "story1_PROTEST_", "event_type": "PROTEST",
         "attributes": [{"actor": ["Protesters"], "location": ["Paris"]}]}
    ]
    exploded, dropped = explode_events(event_list)

    assert dropped == []
    assert len(exploded) == 1
    assert exploded[0]["id"] == "story1_PROTEST__0"
    assert exploded[0]["attributes"] == {"actor": ["Protesters"], "location": ["Paris"]}


def test_explode_multiple_events():
    # Two extracted sub-events -> two separate records with unique ids.
    event_list = [
        {"id": "story1_ASSAULT_", "event_type": "ASSAULT",
         "attributes": [
             {"actor": ["Army"], "recipient": ["rebels"]},
             {"actor": ["rebels"], "recipient": ["Army"]},
         ]}
    ]
    exploded, dropped = explode_events(event_list)

    assert dropped == []
    assert [e["id"] for e in exploded] == ["story1_ASSAULT__0", "story1_ASSAULT__1"]
    assert exploded[0]["attributes"] == {"actor": ["Army"], "recipient": ["rebels"]}
    assert exploded[1]["attributes"] == {"actor": ["rebels"], "recipient": ["Army"]}
    # each record keeps the shared story-level fields
    assert all(e["event_type"] == "ASSAULT" for e in exploded)


def test_explode_drops_empty_extraction():
    # No extracted events -> the record is dropped, not emitted as junk.
    event_list = [
        {"id": "story1_PROTEST_", "event_type": "PROTEST", "attributes": []},
        {"id": "story2_AGREE_", "event_type": "AGREE",
         "attributes": [{"actor": ["France"]}]},
    ]
    exploded, dropped = explode_events(event_list)

    assert len(exploded) == 1
    assert exploded[0]["id"] == "story2_AGREE__0"
    assert len(dropped) == 1
    assert dropped[0]["id"] == "story1_PROTEST_"


def test_explode_does_not_mutate_original_ids():
    # The exploded records are copies; the input records are untouched.
    event_list = [
        {"id": "story1_PROTEST_", "event_type": "PROTEST",
         "attributes": [{"actor": ["Protesters"]}]}
    ]
    explode_events(event_list)
    assert event_list[0]["id"] == "story1_PROTEST_"


def test_events_without_mode_key():
    story_list = [
        {
            "id": "story1",
            "event_type": ["ACCUSE", "CONSULT"]
            # no event_mode key
        }
    ]
    
    expected_output = [
        {'id': 'story1_ACCUSE_', 'event_type': 'ACCUSE', 'event_mode': '', 'orig_id': 'story1'}, 
        {'id': 'story1_CONSULT_', 'event_type': 'CONSULT', 'event_mode': '', 'orig_id': 'story1'}\
    ]

    event_list = stories_to_events(story_list, doc_list=None)

    assert event_list == expected_output


def test_write_intermediate_goes_to_given_dir(tmp_path):
    out_dir = tmp_path / "debug" / "run1"   # does not exist yet
    records = [{"id": "a"}, {"id": "b"}]
    path = write_intermediate(records, "geolocation_output", str(out_dir))
    assert os.path.isabs(path)
    assert os.path.dirname(path) == str(out_dir)
    assert path.endswith("_geolocation_output.jsonl")
    with open(path) as f:
        assert [json.loads(line) for line in f] == records


def test_write_intermediate_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_intermediate([{"id": "a"}], "attribute_output")
    assert os.path.samefile(os.path.dirname(path), tmp_path)


def test_events_to_table():
    from datetime import datetime
    from ngec.utilities import events_to_table

    event = {
        "id": "story1_PROTEST_demo_0", "orig_id": "story1", "pub_date": "2016-05-01",
        "event_type": "PROTEST", "event_mode": "demo",
        "attributes": {"anchor_quote": "Protesters marched",
                       # "N/A" is skipped by actor resolution, so the table
                       # must skip it too to keep text and codes aligned
                       "actor": ["N/A", "Protesters", "union leaders"],
                       "recipient": ["government"], "date": ["today"],
                       "location": ["Paris"]},
        "actor": [{"country": "", "code_1": "CVL", "code_2": "OPP", "wiki": ""},
                  {"country": "FRA", "code_1": "LAB", "code_2": "", "wiki": ""}],
        "recipient": [{"country": "FRA", "code_1": "GOV", "code_2": "", "wiki": "Government of France"}],
        "event_location": {"event_loc": {"resolved_placename": "Paris", "country_code3": "FRA",
                                         "lat": 48.85, "lon": 2.35, "geonameid": "2988507"},
                           "reason": "success"},
        "date_resolved": {"resolved_date": datetime(2016, 5, 1), "granularity": "day",
                          "date_type": "exact"},
    }
    no_location = {"id": "story2_AID__0", "orig_id": "story2", "event_type": "AID",
                   "attributes": {}, "event_location": {"event_loc": None, "reason": "no search term"}}

    table = events_to_table([event, no_location])

    row = table.iloc[0]
    assert row["actor_text"] == "Protesters; union leaders"
    assert row["actor_code"] == "CVL OPP; FRA LAB"
    assert row["recipient_code"] == "FRA GOV"
    assert row["recipient_wiki"] == "Government of France"
    assert row["date"] == "2016-05-01"
    assert row["location_name"] == "Paris"
    assert len(table) == 2
    assert table.iloc[1]["location_name"] is None
