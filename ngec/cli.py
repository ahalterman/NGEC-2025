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

from .models import REQUIRED_SPACY_MODELS, download_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ngec",
        description="NGEC command-line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-models",
        help="download the spaCy models NGEC needs",
        description=(
            "Download and install the spaCy models NGEC needs "
            f"({', '.join(REQUIRED_SPACY_MODELS)}, about 900 MB together). "
            "They are not on PyPI, so installing ngec does not bring them in."))
    download.add_argument("--force", action="store_true",
                          help="re-download models that are already installed")

    args = parser.parse_args(argv)

    # The work is reported through logging, and nothing has configured a handler
    # in a fresh interpreter running the console script.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "download-models":
        try:
            download_models(force=args.force)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
