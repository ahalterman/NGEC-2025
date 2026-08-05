

from ngec.classifiers.plover_sklearn import PloverSklearnClassifier



def test_plover_sklearn_classifier_classify_one():

    clf = PloverSklearnClassifier()

    test_text = ("Thousands of protesters marched through the capital on Sunday "
                 "to demand better wages.")
    event_types, event_modes = clf.classify_one(test_text)

    assert "PROTEST" in event_types
    # Mode models ship alongside the type models now, so a demonstration should
    # come back with its mode attached rather than an empty list.
    assert "PROTEST-demo" in event_modes


def test_plover_sklearn_classifier_classify_one_no_event():

    clf = PloverSklearnClassifier()

    test_text = "The cat is on the roof."
    event_types, event_modes = clf.classify_one(test_text)

    assert event_types == []
    assert event_modes == []


def test_plover_sklearn_classifier_process():

    clf = PloverSklearnClassifier()

    test_texts = [
        {"event_text": "Thousands of protesters marched through the capital on "
                       "Sunday to demand better wages.", "id": "1"},
        {"event_text": "The cat is on the roof.", "id": "2"}
    ]

    results = clf.process(test_texts)

    assert "PROTEST" in results[0]["event_type"]
    assert results[0]["event_type_confidence"]["PROTEST"] > 0
    assert results[1]["event_type"] == []
    assert results[1]["event_mode"] == []


def test_modes_are_conditional_on_their_type():
    """A mode may only be reported when its parent type fired."""
    clf = PloverSklearnClassifier()

    results = clf.process([
        {"event_text": "Government troops attacked the rebel base overnight, "
                       "killing at least 30 fighters.", "id": "1"},
    ])

    for mode in results[0]["event_mode"]:
        parent = mode.split("-", 1)[0]
        assert parent in results[0]["event_type"], (
            f"{mode} reported without its parent type {parent}")


def test_encoder_comes_from_model_metadata():
    """
    The encoder must be whatever the models were trained with.

    This is the regression guard for the silent encoder-mismatch bug.
    """
    clf = PloverSklearnClassifier()
    assert clf.metadata.get("encoder"), "model directory must record its encoder"
    assert clf.encoder_name == clf.metadata["encoder"]
