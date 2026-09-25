"""`ngec download-index`, without downloading anything."""

import os

import pytest

from ngec import index_download


def test_docker_command_runs_as_the_caller_with_group_0(tmp_path):
    """The unpacked files belong to the caller; the image runs as uid 1000
    unless told otherwise, and then cannot write to them."""
    command = index_download.docker_command(tmp_path)
    assert command[:5] == ["docker", "run", "-d", "--name", index_download.CONTAINER_NAME]
    if hasattr(os, "getuid"):
        assert command[command.index("--user") + 1] == f"{os.getuid()}:0"
    assert f"{tmp_path.resolve()}:/usr/share/elasticsearch/data" in command
    assert command[-1] == "elasticsearch:7.10.1"


def test_existing_directory_is_not_overwritten(tmp_path):
    (tmp_path / index_download.UNPACKS_TO).mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        index_download.download_index(dest=tmp_path, url="http://invalid.example/x.tar.gz")


def test_url_constant_matches_the_setup_doctor():
    """The package and the stdlib-only setup doctor each name the archive; they
    must name the same one."""
    from pathlib import Path
    doctor = Path(__file__).resolve().parents[1] / "setup" / "doctor" / "ngec_doctor.py"
    assert f'PREBUILT_INDEX_URL = "{index_download.INDEX_URL}"' in doctor.read_text()
