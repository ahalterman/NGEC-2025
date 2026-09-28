"""
Where the package reads its assets from and writes its output to (#39).
"""
import os

from ngec.attribute_model import _load_event_definitions
from ngec.formatter import Formatter


def test_codebook_loads_from_package():
    # Used to look up the package as "NGEC", which always failed and fell
    # through to a __file__-based fallback.
    event_definitions = _load_event_definitions()
    assert "event" in event_definitions.columns
    assert len(event_definitions) > 0


def test_formatter_writes_to_output_dir(tmp_path):
    formatter = Formatter(quiet=True, output_dir=str(tmp_path))
    events = formatter.process([{"id": "a", "attributes": {}}])
    assert [e["id"] for e in events] == ["a"]
    assert os.path.exists(tmp_path / "events_processed.jsonl")


def test_formatter_return_raw_writes_nothing(tmp_path):
    formatter = Formatter(quiet=True, output_dir=str(tmp_path))
    events = formatter.process([{"id": "a", "attributes": {}}], return_raw=True)
    assert len(events) == 1
    assert os.listdir(tmp_path) == []
