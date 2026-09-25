"""The pipeline run behind `ngec-doctor --smoke`.

The point is to answer "does my install work?" by running three Voice of
America stories through `PloverCoder` and looking at what comes out. It says
nothing about whether the events are *right*; that is what the tests and the
benchmark corpus in `data/voa_benchmark/` are for.

Doctor does not call `run()` in its own process. It starts this module as a
separate script (`python -m ngec.smoke_test OUTFILE`) so that it can

- set Hugging Face's offline mode, which makes a missing model an error instead
  of a multi-gigabyte download (doctor only ever reports);
- run it in a temporary directory, since pipeline steps write files such as
  `*_dropped_events.jsonl` into the working directory;
- keep the pipeline's own logging and progress bars out of the report.

The three stories are copied from that corpus into `ngec/assets/` so they ship
with the package and the check works from a plain `pip install`. VOA text is a
US government work and in the public domain; see `data/voa_benchmark/README.md`.
"""

from __future__ import annotations

import json
import os
import sys
from importlib import resources

from .es_client import es_client_from_env


def load_smoke_test_stories() -> list[dict]:
    """Read the bundled stories and put them in the shape PloverCoder expects."""
    path = resources.files("ngec") / "assets" / "smoke_test_stories.jsonl"
    stories = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stories.append({
                "id": f"voa_{row['id']}",
                "event_text": row["text"],
                # "2025-03-10 14:02:11Z" -> "2025-03-10"
                "pub_date": row["published"][:10],
            })
    return stories


def describe_event(event: dict) -> str:
    """One line per event, for reading in a terminal."""
    def actor_names(actors):
        names = []
        for a in actors or []:
            label = a.get("wiki") or a.get("actor_role_query") or "?"
            code = "/".join(x for x in [a.get("country"), a.get("code_1")] if x)
            names.append(f"{label} [{code}]" if code else label)
        return ", ".join(names) or "-"

    location = (event.get("event_location") or {}).get("event_loc") or {}
    date = (event.get("date_resolved") or {}).get("resolved_date")
    event_type = f"{event.get('event_type')} {event.get('event_mode') or ''}".strip()

    return (f"{event_type}: {actor_names(event.get('actor'))} -> "
            f"{actor_names(event.get('recipient'))}; "
            f"in {location.get('name') or '-'}; on {date or '-'}")


def run() -> dict:
    """Run the pipeline over the bundled stories and summarize what came out."""
    from .plover_coder import PloverCoder

    stories = load_smoke_test_stories()
    coder = PloverCoder(es_client=es_client_from_env())
    events = coder.process(stories)

    return {
        "stories": len(stories),
        "events": [{"story": e.get("orig_id", e.get("id")),
                    "summary": describe_event(e)}
                   for e in events],
    }


if __name__ == "__main__":
    # The summary goes to a file rather than stdout, which the pipeline's
    # progress bars and logging also write to.
    result = run()
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(result, f)
