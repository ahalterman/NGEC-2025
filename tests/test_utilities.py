

from ngec.utilities import stories_to_events



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