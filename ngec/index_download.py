"""Download the pre-built Elasticsearch index and start Elasticsearch over it.

Actor resolution (the `wiki` index) and geolocation (the `geonames` index) need
an Elasticsearch node holding both indices. The easiest way to get one is the
published archive: a tar of an Elasticsearch 7.10.1 data directory with both
indices already loaded. `ngec download-index` fetches it, checks it against its
published SHA-256, unpacks it, and prints (or, with `--start`, runs) the one
`docker run` command that serves it on port 9200, where NGEC looks by default.

Building the indices from the source dumps instead takes most of a day; see
`elasticsearch/SETUP.md`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# The published archive. Keep this, `PREBUILT_INDEX_URL` in
# setup/doctor/ngec_doctor.py and elasticsearch/SETUP.md in agreement.
# The checksum is published next to it as <url>.sha256.
INDEX_URL = "https://andrewhalterman.com/files/wikigeo_index_2026-09.tar.gz"
INDEX_BYTES = 11_604_992_023       # the archive
UNPACKED_BYTES = 15_000_000_000    # the data directory it unpacks to
UNPACKS_TO = "wikigeo_index"       # the directory at the top of the archive

ES_IMAGE = "elasticsearch:7.10.1"  # a 7.10 data directory opens only on 7.10.x
CONTAINER_NAME = "ngec-es"
DEFAULT_DEST = Path.home() / "ngec-es-data"


def download(url: str, path: Path) -> None:
    """Download `url` to `path`, resuming a partial download if one is there."""
    have = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        # 416: nothing left after byte `have`, i.e. the file is already
        # complete (e.g. fetched earlier with curl). The checksum decides.
        if e.code == 416 and have:
            logger.info(f"{path.name} is already fully downloaded.")
            return
        raise
    with response:
        # 206 means the server is sending the rest; 200 means it ignored the
        # Range header and is sending the whole file, so start over.
        if have and response.status != 206:
            have = 0
        total = have + int(response.headers.get("Content-Length", 0))
        mode = "ab" if have else "wb"
        with open(path, mode) as out, tqdm(total=total, initial=have, unit="B",
                                           unit_scale=True, desc=path.name) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))


def published_sha256(url: str) -> str:
    """The checksum published next to the archive, as <url>.sha256."""
    with urllib.request.urlopen(url + ".sha256") as response:
        return response.read().decode().split()[0]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f, tqdm(total=path.stat().st_size, unit="B",
                                     unit_scale=True, desc="checking") as bar:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def unpack(archive: Path, dest: Path) -> None:
    """Unpack the archive into `dest`. The system `tar` is much faster than
    Python's tarfile on an 11 GB archive, so use it when there is one."""
    if shutil.which("tar"):
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(dest)], check=True)
    else:
        with tarfile.open(archive) as tar:
            tar.extractall(dest)


def docker_command(data_dir: Path) -> list[str]:
    """The `docker run` that serves the unpacked data directory on port 9200.

    `--user <your uid>:0`: the archive's files belong to whoever unpacked it,
    and the Elasticsearch image otherwise runs as uid 1000 and cannot write to
    them. The image accepts any uid as long as the group is 0. On macOS and
    Windows, Docker Desktop maps file ownership itself and the flag is harmless.
    """
    command = ["docker", "run", "-d", "--name", CONTAINER_NAME]
    if hasattr(os, "getuid"):
        command += ["--user", f"{os.getuid()}:0"]
    return command + ["-p", "9200:9200",
                      "-e", "discovery.type=single-node",
                      "--restart", "unless-stopped",
                      "-v", f"{data_dir.resolve()}:/usr/share/elasticsearch/data",
                      ES_IMAGE]


def start_elasticsearch(data_dir: Path) -> None:
    """Run the `docker run` from `docker_command`."""
    command = docker_command(data_dir)
    if not shutil.which("docker"):
        raise RuntimeError("Docker is not installed. Install it, then run:\n  "
                           + " ".join(command))
    subprocess.run(command, check=True)
    logger.info(f"Started Elasticsearch as container '{CONTAINER_NAME}'. It takes "
                "a minute to open the indices; then `ngec doctor` should find both.")


def download_index(dest: Path = DEFAULT_DEST, url: str = INDEX_URL,
                   keep_archive: bool = False, start: bool = False) -> Path:
    """Download, verify and unpack the index; return the data directory.

    Raises RuntimeError with a message meant for the user when something is
    in the way (not enough disk, a bad checksum, an existing directory).
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise RuntimeError(
            "Run this as your ordinary user, not as root: Elasticsearch refuses "
            "to run as root, and files unpacked by root would need chown to fix.")
    dest.mkdir(parents=True, exist_ok=True)
    data_dir = dest / UNPACKS_TO
    if data_dir.exists():
        # Already downloaded. With --start, just start Elasticsearch over it.
        if start:
            start_elasticsearch(data_dir)
            return data_dir
        raise RuntimeError(
            f"{data_dir} already exists. To serve it, run `ngec download-index "
            f"--start` (or `ngec doctor` to check one that is running). To replace "
            "it, stop the container serving it and delete the directory first.")

    archive = dest / os.path.basename(url)
    need = UNPACKED_BYTES + (0 if archive.exists() else INDEX_BYTES)
    free = shutil.disk_usage(dest).free
    if free < need:
        raise RuntimeError(
            f"{dest} has {free / 1e9:.0f} GB free; the archive and the unpacked "
            f"index need about {need / 1e9:.0f} GB. Pass --dest to put it elsewhere.")

    logger.info(f"Downloading {url}\n  to {archive} (about {INDEX_BYTES / 1e9:.1f} GB)")
    download(url, archive)

    expected = published_sha256(url)
    if sha256_of(archive) != expected:
        raise RuntimeError(
            f"{archive} does not match its published checksum. Delete it and run "
            "this again; if it happens twice, the download is being corrupted "
            "on the way (a proxy, a full disk).")
    logger.info("Checksum OK. Unpacking (a few minutes) ...")
    unpack(archive, dest)
    if not keep_archive:
        archive.unlink()

    logger.info(f"Unpacked to {data_dir}.")
    if start:
        start_elasticsearch(data_dir)
    else:
        logger.info("\nStart Elasticsearch over it with `ngec download-index --start`, "
                    "or yourself:\n\n  " + " ".join(docker_command(data_dir)))
    return data_dir
