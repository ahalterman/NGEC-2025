"""The `ngec` command.

Argparse rather than click, to match `ngec/doctor.py`, the other command in the
package.

There is one subcommand so far, and it is here rather than in `doctor.py`
because the two do opposite things: doctor only ever reports, and this changes
the environment.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .models import download_models


def main(argv: list[str] | None = None) -> int:
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

    args = parser.parse_args(argv)

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
