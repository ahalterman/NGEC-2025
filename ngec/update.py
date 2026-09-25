"""`ngec update`: bring the models and the Elasticsearch index up to date.

Two things NGEC downloads can change after installation:

- **Models on Hugging Face** (the attribute LLM and the sentence encoders).
  The hub versions every repository, so a model is out of date when the
  commit the hub serves differs from the one in the local cache.
  `ngec download-models` already fetches only what changed; `ngec update`
  says first which models that would be.
- **The pre-built index.** A release is described by a small file on the
  server (`index_download.LATEST_URL`) that records each index's build date.
  The index is out of date when those dates are newer than the `_meta`
  build dates in the Elasticsearch that is running.

Without `--apply`, nothing is changed: it reports what is out of date and what
`--apply` would do. With `--apply` it updates the models, and replaces the index
by downloading the new release next to the old one, swapping the Elasticsearch
container over to it, checking that both indices come up with the published
document counts, and only then deleting the old container and data directory.
If the new index does not come up, the old container is put back.

The container is only replaced automatically when it is the one
`ngec download-index --start` creates (named `ngec-es`); anything else is left
alone and the commands are printed instead.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from . import index_download as idx

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ models

def hub_models() -> list[str]:
    """The Hugging Face ids of the models the pipeline loads by name."""
    from .actors.common import ModelManager
    from .attribute_model import resolve_model_name
    from .models import classifier_encoder_name

    manager = ModelManager(device="cpu")
    names = [classifier_encoder_name(), manager.encoder_name,
             manager.agent_encoder_name, resolve_model_name(None)]
    # A model given as a local directory has no hub version to compare.
    return [n for n in dict.fromkeys(names) if not Path(n).expanduser().is_dir()]


def cached_revision(name: str) -> str | None:
    """The commit of `name` in the local Hugging Face cache, or None."""
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = Path(HF_HUB_CACHE) / f"models--{name.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def check_models() -> list[tuple[str, str | None, str]]:
    """(name, cached commit, hub commit) for every model that is out of date."""
    from huggingface_hub import HfApi

    api = HfApi()
    stale = []
    for name in hub_models():
        remote = api.model_info(name).sha
        local = cached_revision(name)
        if local != remote:
            stale.append((name, local, remote))
    return stale


# ------------------------------------------------------------------- index

def es_get(port: int, path: str):
    with urllib.request.urlopen(f"http://localhost:{port}/{path}", timeout=10) as r:
        return json.load(r)


def installed_index(port: int) -> dict | None:
    """{index: {"doc_count", "meta"}} for the running Elasticsearch, or None
    if nothing answers on `port`."""
    try:
        es_get(port, "")
    except Exception:
        return None
    found = {}
    for name in ("wiki", "geonames"):
        try:
            mapping = es_get(port, f"{name}/_mapping")
            meta = list(mapping.values())[0]["mappings"].get("_meta", {})
            count = es_get(port, f"{name}/_count")["count"]
            found[name] = {"doc_count": count, "meta": meta}
        except Exception:
            found[name] = None
    return found


def index_is_current(installed: dict, release: dict) -> bool | None:
    """Whether every published index is no newer than the installed one.
    None when the release does not say (the server has no latest file)."""
    if not release["indices"]:
        return None
    for name, published in release["indices"].items():
        mine = installed.get(name)
        if mine is None:
            return False
        if (published.get("meta") or {}).get("build_date", "") > (mine["meta"] or {}).get("build_date", ""):
            return False
    return True


def serving_container(port: int) -> tuple[str, Path | None] | None:
    """(name, data directory) of the docker container publishing `port`."""
    if not shutil.which("docker"):
        return None
    names = subprocess.run(["docker", "ps", "--filter", f"publish={port}",
                            "--format", "{{.Names}}"],
                           capture_output=True, text=True).stdout.split()
    if not names:
        return None
    mounts = json.loads(subprocess.run(
        ["docker", "inspect", "-f", "{{json .Mounts}}", names[0]],
        capture_output=True, text=True).stdout or "[]")
    data = [m["Source"] for m in mounts
            if m.get("Destination") == "/usr/share/elasticsearch/data"]
    return names[0], Path(data[0]) if data else None


def wait_until_ready(port: int, release: dict, timeout: int = 600) -> bool:
    """True once both indices are green with the published document counts."""
    expected = {name: d.get("doc_count") for name, d in (release["indices"] or {}).items()}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            health = {row["index"]: row for row in
                      es_get(port, "_cat/indices?format=json&h=index,health,docs.count")}
            if all(name in health and health[name]["health"] == "green"
                   and (expected.get(name) is None
                        or int(health[name]["docs.count"]) == expected[name])
                   for name in ("wiki", "geonames")):
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def docker(*args: str, check: bool = True) -> None:
    subprocess.run(["docker", *args], check=check, capture_output=True)


def swap_index(release: dict, container: str, old_dir: Path, port: int,
               keep_old: bool = False) -> Path:
    """Download `release` next to `old_dir`, move `container` over to it, and
    remove the old container and directory once the new one checks out."""
    stem = Path(release["url"]).name.removesuffix(".tar.gz")
    new_dir = old_dir.parent / stem
    if new_dir.exists():
        raise RuntimeError(f"{new_dir} already exists; delete it and run this again.")
    staging = old_dir.parent / ".ngec-update"
    staging.mkdir(exist_ok=True)
    idx.fetch_and_unpack(release, staging).rename(new_dir)
    shutil.rmtree(staging)

    previous = container + "-previous"
    logger.info(f"Stopping '{container}' and starting it again on {new_dir} "
                "(Elasticsearch is unavailable for a minute or so) ...")
    docker("stop", container)
    docker("rename", container, previous)
    subprocess.run(idx.docker_command(new_dir, name=container, port=port),
                   check=True, capture_output=True)
    if not wait_until_ready(port, release):
        logger.error("The new index did not come up; putting the old container back.")
        docker("rm", "-f", container, check=False)
        docker("rename", previous, container)
        docker("start", container)
        raise RuntimeError(
            f"The new index in {new_dir} did not come up green with the published "
            f"document counts, so '{container}' is back on {old_dir}. The new "
            f"index is left in {new_dir} to look into; delete it when done.")
    docker("rm", previous)
    # Only a directory that looks like one of ours is deleted unasked.
    if keep_old or not old_dir.name.startswith(idx.UNPACKS_TO):
        logger.info(f"Kept the old index in {old_dir}.")
    else:
        shutil.rmtree(old_dir)
        logger.info(f"Deleted the old index in {old_dir}.")
    return new_dir


# ------------------------------------------------------------------ driver

def update(apply: bool = False, models: bool = True, index: bool = True,
           port: int = 9200, container: str = idx.CONTAINER_NAME,
           keep_old: bool = False) -> int:
    """Report what is out of date and, with `apply`, update it. Returns an
    exit code: 0 when everything is (now) current, 1 otherwise."""
    # huggingface_hub logs every API request at INFO through httpx.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    todo = False       # something is out of date
    left = False       # ... and this run did not (or could not) update it
    manual = False     # ... and --apply would not update it either

    if models:
        stale = check_models()
        if not stale:
            logger.info("Models: all up to date.")
        else:
            todo = True
            for name, local, remote in stale:
                logger.info(f"Models: {name} is out of date "
                            f"({(local or 'not downloaded')[:10]} -> {remote[:10]}).")
            left = not apply
            if apply:
                from .models import download_models
                download_models()
                logger.info("Models updated. Restart anything that has them loaded "
                            "(a demo, a vllm process) to use the new versions.")

    if index:
        release = idx.latest_release()
        installed = installed_index(port)
        current = None if installed is None else index_is_current(installed, release)
        served = serving_container(port)
        if installed is None:
            logger.info(f"Index: nothing answers on localhost:{port}. "
                        "`ngec download-index --start` installs it.")
            todo = left = True
        elif current is None:
            logger.info("Index: cannot tell whether it is current (the server has no "
                        f"{Path(idx.LATEST_URL).name}).")
        elif current:
            logger.info("Index: up to date (" + ", ".join(
                f"{n} built {(d['meta'] or {}).get('build_date', '?')}"
                for n, d in installed.items() if d) + ").")
        else:
            todo = True
            published = {n: (d.get("meta") or {}).get("build_date") for n, d in release["indices"].items()}
            logger.info(f"Index: a newer release is published, {Path(release['url']).name} "
                        f"(built {published}).")
            ours = served is not None and served[0] == container and served[1] is not None
            left = not (ours and apply)
            if not ours:
                manual = True
                logger.info(
                    f"  The Elasticsearch on port {port} is not the '{container}' container "
                    "that `ngec download-index --start` creates, so it will not be replaced "
                    "automatically. Stop it yourself, then run `ngec download-index --start`.")
            elif not apply:
                logger.warning(
                    f"  `ngec update --apply` will download it (about "
                    f"{release['size_bytes'] / 1e9:.0f} GB) next to {served[1]}, then STOP "
                    f"the '{served[0]}' container and replace it with one on the new index "
                    "-- Elasticsearch is down for a minute or so -- and delete the old "
                    f"index directory once the new one is verified"
                    f"{' (--keep-old keeps it)' if not keep_old else ''}.")
            else:
                idx.refuse_root()
                swap_index(release, served[0], served[1], port, keep_old=keep_old)
                logger.info("Index updated.")

    if todo and not apply and not manual:
        logger.info("\nRun `ngec update --apply` to update.")
    return 1 if left else 0
