

from ngec.classifiers.plover_sklearn import PloverSklearnClassifier


def test_plover_sklearn_classifier_classify_one():

    clf = PloverSklearnClassifier(
            threshold=0.9
        )

    test_text = "Protesters blocked the highway to demand better wages."
    event_types, event_modes = clf.classify_one(test_text)

    assert event_types == ["PROTEST"]
    assert event_modes == []


def test_plover_sklearn_classifier_classify_one_no_event():

    clf = PloverSklearnClassifier(
            threshold=0.9
        )

    test_text = "The cat is on the roof."
    event_types, event_modes = clf.classify_one(test_text)

    assert event_types == []
    assert event_modes == []


def test_plover_sklearn_classifier_process():

    clf = PloverSklearnClassifier(
            threshold=0.9
        )

    test_texts = [
        {"event_text": "Protesters blocked the highway to demand better wages.", "id": "1"},
        {"event_text": "The cat is on the roof.", "id": "2"}
    ]

    results = clf.process(test_texts)

    assert results[0]["event_type"] == ["PROTEST"]
    assert results[0]["event_mode"] == []
    assert results[1]["event_type"] == []
    assert results[1]["event_mode"] == []