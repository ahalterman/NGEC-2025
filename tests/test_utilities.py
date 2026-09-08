

from ngec.utilities import stories_to_events, explode_events



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