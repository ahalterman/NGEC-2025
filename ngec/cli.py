"""The `ngec` command.

Argparse rather than click, to match `ngec/doctor.py`, the other command in the
package.

`ngec doctor` hands its arguments straight to `ngec/doctor.py`, which is also
installed as `ngec-doctor`; its options are documented there. The doctor stays
in its own module because the two kinds of command do opposite things: doctor
only ever reports, and `download-models` / `download-index` change the
environment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import doctor
from .guide import TOPICS, read_guide, write_agents_file
from .index_download import CONTAINER_NAME, DEFAULT_DEST, download_index
from .models import download_models


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # The doctor has its own parser; hand everything after "doctor" to it
    # rather than declaring its options a second time here.
    if argv[:1] == ["doctor"]:
        return doctor.main(argv[1:])

    parser = argparse.ArgumentParser(
        prog="ngec",
        description="NGEC command-line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-models",
        help="download the models NGEC needs",
        description=(
            "Download the models NGEC needs, 3 to 4 GB together: the two spaCy "
            "models, the sentence encoders used for event classification and "
            "actor resolution, and the attribute-extraction LLM (with its GGUF "
            "file, when llama-cpp-python is installed). None of them "
            "come with installing ngec, and without this step they would "
            "download the first time the pipeline runs (except the spaCy "
            "models, which do not download on their own at all)."))
    download.add_argument("--force", action="store_true",
                          help="reinstall the spaCy models and re-download the "
                               "attribute model even if they are already present")
    download.add_argument("--attribute-model", metavar="NAME",
                          help="the attribute model to download, as a Hugging Face "
                               "id (default: $NGEC_ATTRIBUTE_MODEL, or the model "
                               "AttributeModel uses by default)")
    download.add_argument("--no-attribute-model", action="store_true",
                          help="skip the attribute model, e.g. when it runs on a "
                               "llama.cpp server")
    download.add_argument("--gguf", action="store_true", default=None,
                          help="also download the attribute model's GGUF file, "
                               "for the llamacpp backend (the default when "
                               "llama-cpp-python is installed)")

    index = subparsers.add_parser(
        "download-index",
        help="download the pre-built Elasticsearch index (wiki + geonames)",
        description=(
            "Download the pre-built Elasticsearch data directory holding the "
            "`wiki` and `geonames` indices (about 11.6 GB, 15 GB unpacked), "
            "check its checksum, unpack it, and print the `docker run` command "
            "that serves it on port 9200. Needs Docker to run. Building the "
            "indices yourself instead is described in elasticsearch/SETUP.md."))
    index.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                       help=f"where to unpack it (default: {DEFAULT_DEST})")
    index.add_argument("--url", default=None,
                       help="where to download it from (default: the current published release)")
    index.add_argument("--start", action="store_true",
                       help="also start Elasticsearch over it with docker")
    index.add_argument("--keep-archive", action="store_true",
                       help="keep the downloaded .tar.gz after unpacking it")

    upd = subparsers.add_parser(
        "update",
        help="check whether the models and the index are current; --apply updates them",
        description=(
            "Report which Hugging Face models (the attribute LLM, the sentence "
            "encoders) have newer versions on the hub, and whether a newer "
            "pre-built Elasticsearch index has been published. Changes nothing "
            "without --apply. With --apply, updates the models and replaces the "
            "index: the new release is downloaded next to the old one, the "
            "Elasticsearch container is stopped and started again on it (a minute "
            "or so without Elasticsearch), and the old container and index are "
            "deleted once both new indices are green with the published counts. "
            "If they are not, the old container is put back. Only the container "
            "`ngec download-index --start` creates is replaced automatically."))
    upd.add_argument("--apply", action="store_true", help="update what is out of date")
    upd.add_argument("--no-models", action="store_true", help="leave the models alone")
    upd.add_argument("--no-index", action="store_true", help="leave the index alone")
    upd.add_argument("--keep-old", action="store_true",
                     help="keep the old index directory after replacing it")
    upd.add_argument("--port", type=int, default=9200,
                     help="the port Elasticsearch is published on (default: 9200)")
    upd.add_argument("--container", default=CONTAINER_NAME,
                     help=f"the container --apply may replace (default: {CONTAINER_NAME})")

    subparsers.add_parser(
        "doctor",
        help="check the installation and report what is wrong with it",
        description="The same as `ngec-doctor`; see `ngec doctor --help`.")

    guide = subparsers.add_parser(
        "guide",
        help="print instructions for using NGEC, written for coding agents",
        description=(
            "Print the NGEC guide, which is written for AI coding agents but "
            "readable by anyone. It ships with the package, so it describes the "
            "installed version. With --init, add a short section to AGENTS.md "
            "in the current directory telling agents to run this command."))
    guide.add_argument("topic", nargs="?", default="overview", choices=list(TOPICS),
                       help="which part of the guide to print (default: overview)")
    guide.add_argument("--init", action="store_true",
                       help="add the NGEC section to ./AGENTS.md instead of printing")

    args = parser.parse_args(argv)

    if args.command == "guide":
        if args.init:
            path, action = write_agents_file(".")
            print(f"{action.capitalize()} {path}" if action != "unchanged"
                  else f"{path} already has the NGEC section; nothing to do.")
        else:
            print(read_guide(args.topic))
        return 0

    # The work is reported through logging, and nothing has configured a handler
    # in a fresh interpreter running the console script.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "download-models":
        try:
            download_models(force=args.force,
                            attribute_model=args.attribute_model,
                            include_attribute_model=not args.no_attribute_model,
                            gguf=args.gguf)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    if args.command == "update":
        from .update import update
        try:
            return update(apply=args.apply, models=not args.no_models,
                          index=not args.no_index, port=args.port,
                          container=args.container, keep_old=args.keep_old)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    if args.command == "download-index":
        try:
            download_index(dest=args.dest, url=args.url,
                           keep_archive=args.keep_archive, start=args.start)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
