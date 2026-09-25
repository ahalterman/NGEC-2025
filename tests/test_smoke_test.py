"""The smoke test behind `ngec-doctor --smoke`.

The first two tests need nothing. The last runs the real pipeline and so needs
Elasticsearch and the models, like test_end_to_end.py.
"""

from ngec import doctor
from ngec.smoke_test import load_smoke_test_stories


def test_bundled_stories_load():
    stories = load_smoke_test_stories()
    assert len(stories) == 3
    for story in stories:
        assert story["id"].startswith("voa_")
        assert len(story["event_text"]) > 900
        assert len(story["pub_date"]) == 10


def test_smoke_does_not_run_pipeline_without_elasticsearch(monkeypatch):
    # Port 1 is never an Elasticsearch, so this fails fast and never gets as
    # far as loading a model.
    monkeypatch.setenv("ES_HOST", "localhost")
    monkeypatch.setenv("ES_PORT", "1")
    checks = {c.name: c for c in doctor.smoke()}
    assert checks["Elasticsearch"].status == doctor.FAIL
    assert "not run" in checks["Pipeline"].detail


def test_smoke_runs(es_client_local):
    assert doctor.main(["--only", "smoke"]) == 0
