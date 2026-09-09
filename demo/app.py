"""Entry point: `streamlit run app.py`.

Eight pages -- one end-to-end run, one per pipeline step, a bulk uploader and a
timing page. The sidebar says what is up, because half the demo needs
Elasticsearch and a page that quietly returns nothing is worse than one that
says the index is down. It also carries the GPU/CPU toggle -- both model sets
stay loaded, so switching is a radio button rather than a restart -- and the
button that loads the models.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngec_demo import resources as R  # noqa: E402
from ngec_demo.style import (health_sidebar, inject_css,  # noqa: E402
                             load_models_button, mode_badge)


def main() -> None:
    st.set_page_config(page_title="NGEC", page_icon=":material/schema:",
                       layout="centered")
    inject_css()

    # No icons: the step pages are already numbered, and a rail of emoji is the
    # first thing that breaks the deadpan look.
    pages = [
        st.Page("pages/home.py", title="NGEC", default=True),
        st.Page("pages/step1.py", title="1. Which event?"),
        st.Page("pages/step2.py", title="2. Who did what?"),
        st.Page("pages/step3.py", title="3. Which entity?"),
        st.Page("pages/step4.py", title="4. What kind of actor?"),
        st.Page("pages/step5.py", title="5. When and where?"),
        st.Page("pages/bulk.py", title="Bulk"),
        st.Page("pages/timing.py", title="Timing"),
    ]

    page = st.navigation(pages)

    # The toggle writes st.session_state["mode"] itself, and it is drawn before
    # anything else so that R.current_mode() -- which reads that key -- is right for
    # the health block and for the page.
    MODES = R.available_modes()
    if len(MODES) > 1:
        _env = os.environ.get("NGEC_DEMO_MODE", "").strip().lower()
        st.sidebar.radio("Compute", MODES, key="mode", horizontal=True,
                         index=MODES.index(_env) if _env in MODES else 0,
                         format_func=mode_badge)
    else:
        st.session_state["mode"] = MODES[0]
        st.sidebar.caption(f"Compute · {mode_badge(MODES[0])}")

    # Both blocks below say whether the models are loaded, and the page is what
    # loads them: a run that starts cold finishes warm, and a sidebar drawn
    # first would still say "not loaded" beside a page saying "models loaded in
    # 11 s". So their places in the sidebar are reserved here, the page runs,
    # and the slots are filled afterwards, with the state at the end of the run.
    load_slot = st.sidebar.container()
    health_slot = st.sidebar.container()

    page.run()

    # Loading the models is the demo's one long wait. It happens on the first
    # click of any page anyway; the button lets a visitor start it deliberately,
    # with a status that names each component, instead of meeting it as a hang.
    with load_slot:
        load_models_button(st.session_state["mode"])

    # The classifier's DemoModelWarning is captured at load time and parked in
    # session state; showing it here means it is said once, not on every rerun.
    with health_slot:
        health_sidebar(R.health(st.session_state["mode"]),
                       st.session_state.get("model_notes", []),
                       mode=st.session_state["mode"])


# vllm starts its engine in a *spawned* child process, and spawn re-imports the
# main script -- which under Streamlit is this file. Without the guard the child
# would run the app body, hit session state with no runtime, and the engine
# would die before it started. Spawned children import as "__mp_main__".
if __name__ == "__main__":
    main()
