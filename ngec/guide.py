"""`ngec guide`: instructions for an AI coding agent (or a person) using NGEC.

The guide is plain Markdown shipped inside the package, in ngec/assets/guide/,
so what it says always matches the installed version. That is also why
`ngec guide --init` writes only a short pointer into a project's AGENTS.md
rather than a copy of the guide: a copy would go stale the next time NGEC is
upgraded, and nothing would say so.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

# Topic name -> file in ngec/assets/guide/. The order is the order `ngec guide`
# lists them in.
TOPICS = {
    "overview": "overview.md",
    "setup": "setup.md",
    "run": "run.md",
    "pieces": "pieces.md",
    "customize": "customize.md",
}

# The markers let `--init` find its own section again, so running it twice does
# not add the section twice.
START_MARKER = "<!-- ngec guide: start -->"
END_MARKER = "<!-- ngec guide: end -->"

AGENTS_SECTION = f"""{START_MARKER}
## NGEC

This project uses NGEC (https://github.com/ahalterman/NGEC-2025) to turn news
text into event data. Before writing or debugging code that uses NGEC, run
`ngec guide` (or `uv run ngec guide`) and read what it prints. It is written
for coding agents, and it describes the NGEC version installed here. It lists
further topics: `ngec guide setup`, `run`, `pieces` and `customize`.

When something in the environment looks wrong (a missing model, Elasticsearch
not answering, a pipeline much slower than expected), run `ngec doctor` before
guessing, and fix what it reports in the order it reports it.
{END_MARKER}
"""


def read_guide(topic: str = "overview") -> str:
    """Return the text of one guide topic."""
    if topic not in TOPICS:
        raise ValueError(f"unknown topic '{topic}'; choose from {', '.join(TOPICS)}")
    path = resources.files("ngec").joinpath("assets", "guide", TOPICS[topic])
    return path.read_text(encoding="utf-8")


def write_agents_file(directory: str | Path = ".") -> tuple[Path, str]:
    """Add the NGEC section to AGENTS.md in `directory`.

    Creates AGENTS.md if there is none, and appends the section to an existing
    one. An AGENTS.md that already has the section is left alone.

    Returns the path and what was done: "created", "appended" or "unchanged".
    """
    path = Path(directory) / "AGENTS.md"
    if not path.exists():
        path.write_text(AGENTS_SECTION, encoding="utf-8")
        return path, "created"

    existing = path.read_text(encoding="utf-8")
    if START_MARKER in existing:
        return path, "unchanged"
    separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    path.write_text(existing + separator + AGENTS_SECTION, encoding="utf-8")
    return path, "appended"
