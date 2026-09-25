"""The `ngec` command.

Argparse rather than click, to match `ngec/doctor.py`, the other command in the
package.

`ngec doctor` hands its arguments straight to `ngec/doctor.py`, which is also
installed as `ngec-doctor`; its options are documented there. The doctor stays
in its own module because the two kinds of command do opposite things: doctor
only ever reports, and `download-models` changes the environment.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import doctor
from .guide import TOPICS, read_guide, write_agents_file
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
            "Download the models NGEC needs, about 3 GB together: the two spaCy "
            "models, the sentence encoders used for event classification and "
            "actor resolution, and the attribute-extraction LLM. None of them "
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
                            include_attribute_model=not args.no_attribute_model)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
