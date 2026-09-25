"""NGEC demonstration app.

Run it from this directory so that Streamlit picks up .streamlit/config.toml:

    cd demo && uv run --group demo-app streamlit run app.py

The app is organised in three sections. "See it work" is for someone who wants
to know what the pipeline produces; "Look inside" walks the five steps of the
framework, one page each, cross-referenced to the manuscript; "Make it yours"
covers the customisations the paper claims are cheap, and shows what they
actually cost.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# The package needs Python 3.10+. The usual way to get this wrong is to run
# `streamlit run app.py` with whichever streamlit happens to be on PATH (often a
# system or conda 3.9) instead of `uv run --group demo-app streamlit run app.py`.
# Without this check the first 3.10-only annotation raises a TypeError several
# imports deep, which does not point at the cause.
# (A linter will call this block dead code, because it checks against the version
# in pyproject.toml. The whole point is the interpreter that ignored that.)
if sys.version_info < (3, 10):  # noqa: UP036
    st.error(
        "This demo needs Python 3.10 or newer; it is running under "
        f"Python {sys.version_info.major}.{sys.version_info.minor} "
        f"({sys.executable}).\n\n"
        "Streamlit was probably launched from a different environment than the "
        "project's. From the `demo/` directory, run:\n\n"
        "```shell\n"
        "uv run --group demo-app streamlit run app.py\n"
        "```"
    )
    st.stop()

# Allow `import ngec_demo` when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).parent))

from ngec_demo import ui  # noqa: E402

ui.boot()

try:
    from ngec.logging import setup_logging

    setup_logging(level=logging.WARNING, quiet_third_party=True)
except Exception:  # noqa: BLE001 - logging setup must never block the app
    logging.basicConfig(level=logging.WARNING)


def page(path: str, title: str, url_path: str, icon: str = "", default: bool = False):
    return st.Page(f"pages/{path}", title=title, url_path=url_path,
                   icon=icon or None, default=default)


navigation = {
    "See it work": [
        page("home.py", "Start here", "home", default=True),
        page("end_to_end.py", "Document → events", "end_to_end"),
    ],
    "Look inside": [
        page("step1_detection.py", "1 · Event detection", "step1"),
        page("step2_attributes.py", "2 · Attribute extraction", "step2"),
        page("step3_entities.py", "3 · Entity linking", "step3"),
        page("step4_categories.py", "4 · Entity categorisation", "step4"),
        page("step5_dates_places.py", "5 · Dates and places", "step5"),
    ],
    "Hard cases": [
        page("echoes.py", "Temporal echoes", "echoes"),
        page("coref.py", "Coreference", "coref"),
        page("nonenglish.py", "Non-English text", "nonenglish"),
        page("vs_llm.py", "Against a frontier LLM", "vs_llm"),
    ],
    "Make it yours": [
        page("customize_codebook.py", "Your event types", "your_events"),
        page("customize_actors.py", "Your actor categories", "your_actors"),
        page("setup_cost.py", "What setup actually costs", "setup"),
    ],
}

st.navigation(navigation).run()
