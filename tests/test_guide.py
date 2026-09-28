"""`ngec guide` and `ngec doctor`, the two CLI subcommands that only read."""
import ngec.cli
from ngec.guide import END_MARKER, START_MARKER, TOPICS, read_guide, write_agents_file


def test_every_topic_is_shipped():
    # The guide files are package data; a missing one would only show up as a
    # crash on a user's machine.
    for topic in TOPICS:
        assert read_guide(topic).startswith("# ")


def test_guide_prints_the_overview(capsys):
    assert ngec.cli.main(["guide"]) == 0
    assert capsys.readouterr().out.startswith("# NGEC guide")


def test_guide_prints_a_topic(capsys):
    assert ngec.cli.main(["guide", "run"]) == 0
    assert "events_to_table" in capsys.readouterr().out


def test_init_creates_agents_file(tmp_path):
    path, action = write_agents_file(tmp_path)
    assert action == "created"
    text = path.read_text()
    assert START_MARKER in text and END_MARKER in text
    assert "ngec guide" in text


def test_init_appends_once(tmp_path):
    existing = "# My project\n\nSome rules of my own.\n"
    (tmp_path / "AGENTS.md").write_text(existing)

    _, action = write_agents_file(tmp_path)
    assert action == "appended"
    _, action = write_agents_file(tmp_path)
    assert action == "unchanged"

    text = (tmp_path / "AGENTS.md").read_text()
    assert text.startswith(existing)
    assert text.count(START_MARKER) == 1


def test_doctor_subcommand_passes_its_arguments_on(monkeypatch):
    received = {}
    monkeypatch.setattr(ngec.cli.doctor, "main",
                        lambda argv: received.setdefault("argv", argv) and 0)
    ngec.cli.main(["doctor", "--only", "config", "--json"])
    assert received["argv"] == ["--only", "config", "--json"]
