"""ngec-doctor's list of settings, and its check of the keys in .env.

None of these need models or Elasticsearch.
"""

import re
from pathlib import Path

from ngec import doctor

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN = {setting.name for setting in doctor.SETTINGS}


def check_keys(tmp_path, env_text):
    env_file = tmp_path / ".env"
    env_file.write_text(env_text)
    return doctor._env_keys(env_file)


# ----------------------------------------------------------- the .env check


def test_known_keys_pass(tmp_path):
    checks = check_keys(tmp_path, "ES_HOST=es.example.org\n"
                                  "export NGEC_ATTRIBUTE_MODEL=some/model\n")
    assert [c.status for c in checks] == [doctor.OK]


def test_commented_out_lines_in_env_are_ignored(tmp_path):
    checks = check_keys(tmp_path, "#ES_HOTS=es.example.org\n")
    assert [c.status for c in checks] == [doctor.OK]


def test_misspelling_is_a_warning_naming_the_setting(tmp_path):
    checks = check_keys(tmp_path, "NGEC_ATTRIBUTE_MODLE=some/model\n")
    assert len(checks) == 1
    assert checks[0].status == doctor.WARN
    assert checks[0].name == "NGEC_ATTRIBUTE_MODLE"
    assert "NGEC_ATTRIBUTE_MODEL" in checks[0].detail


def test_wrong_case_is_a_warning(tmp_path):
    checks = check_keys(tmp_path, "es_host=es.example.org\n")
    assert checks[0].status == doctor.WARN
    assert "ES_HOST" in checks[0].detail


def test_unknown_ngec_name_is_a_warning(tmp_path):
    checks = check_keys(tmp_path, "NGEC_SOMETHING_ELSE=1\n")
    assert checks[0].status == doctor.WARN


def test_unrelated_keys_are_reported_not_judged(tmp_path):
    checks = check_keys(tmp_path, "HF_TOKEN=x\nCUDA_VISIBLE_DEVICES=0\n")
    assert [c.status for c in checks] == [doctor.INFO]
    assert "HF_TOKEN" in checks[0].detail


def test_a_projects_own_env_example_is_not_consulted(tmp_path):
    # A user's project often has a .env.example for its own settings. It must
    # not stand in for NGEC's list, or every NGEC key would look unread.
    (tmp_path / ".env.example").write_text("MY_APP_SECRET=\n")
    checks = check_keys(tmp_path, "NGEC_ATTRIBUTE_MODEL=some/model\nMY_APP_SECRET=x\n")
    assert all(c.status != doctor.WARN for c in checks)


def test_no_env_file_means_no_check():
    assert doctor._env_keys(None) == []


# -------------------------------------------- SETTINGS against the code


def _names_read(pattern, globs):
    names = set()
    for glob in globs:
        for path in REPO_ROOT.glob(glob):
            names.update(re.findall(pattern, path.read_text(encoding="utf-8")))
    return names


def test_settings_cover_every_ngec_variable_the_code_reads():
    # Quoted names in Python (os.environ.get("NGEC_X")), $NGEC_X / ${NGEC_X} in
    # shell and compose files. Prose mentions in comments don't count.
    in_python = _names_read(r'"(NGEC_[A-Z0-9_]+)"',
                            ["ngec/**/*.py", "demo/app.py", "demo/ngec_demo/**/*.py",
                             "elasticsearch/**/*.py", "tools/**/*.py"])
    in_shell = _names_read(r"\$\{?(NGEC_[A-Z0-9_]+)",
                           ["tools/*.sh", "elasticsearch/**/*.yml"])
    assert (in_python | in_shell) - KNOWN == set()


def test_env_example_documents_exactly_the_settings():
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?([A-Z_][A-Z0-9_]*)=", example, flags=re.MULTILINE))
    assert documented == KNOWN


def test_encoder_defaults_match_the_code():
    # Written out in SETTINGS because importing ngec.actors.common would pull in
    # sentence-transformers just to print a default.
    source = (REPO_ROOT / "ngec/actors/common.py").read_text(encoding="utf-8")
    defaults = {s.name: s.default for s in doctor.SETTINGS}
    assert f'DEFAULT_ENCODER = "{defaults["NGEC_WIKI_ENCODER"]}"' in source
    assert f'AGENT_ENCODER = "{defaults["NGEC_AGENT_ENCODER"]}"' in source
