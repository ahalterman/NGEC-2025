"""Entry point: `streamlit run app.py`.

Seven pages -- one end-to-end run, one per pipeline step, and a timing page.
The sidebar says what is up, because half the demo needs Elasticsearch and a
page that quietly returns nothing is worse than one that says the index is down.
It also carries the GPU/CPU toggle: both model sets stay loaded, so switching is
a radio button rather than a restart.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngec_demo import resources as R  # noqa: E402
from ngec_demo.style import health_sidebar, inject_css, mode_badge  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="NGEC", page_icon="🌍", layout="centered")
    inject_css()

    pages = [
        st.Page("pages/home.py", title="NGEC", icon="🌍", default=True),
        st.Page("pages/step1.py", title="1. Which event?", icon="1️⃣"),
        st.Page("pages/step2.py", title="2. Who did what?", icon="2️⃣"),
        st.Page("pages/step3.py", title="3. Which entity?", icon="3️⃣"),
        st.Page("pages/step4.py", title="4. What kind of actor?", icon="4️⃣"),
        st.Page("pages/step5.py", title="5. When and where?", icon="5️⃣"),
        st.Page("pages/timing.py", title="Timing", icon="⏱️"),
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

    # The classifier's DemoModelWarning is captured at load time and parked in
    # session state; showing it here means it is said once, not on every rerun.
    health_sidebar(R.health(st.session_state["mode"]),
                   st.session_state.get("model_notes", []))
    page.run()


# vllm starts its engine in a *spawned* child process, and spawn re-imports the
# main script -- which under Streamlit is this file. Without the guard the child
# would run the app body, hit session state with no runtime, and the engine
# would die before it started. Spawned children import as "__mp_main__".
if __name__ == "__main__":
    main()
