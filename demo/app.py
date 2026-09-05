"""Entry point: `streamlit run app.py`.

Six pages -- one end-to-end run, then one page per pipeline step. The sidebar
says what is up, because half the demo needs Elasticsearch and a page that
quietly returns nothing is worse than one that says the index is down.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngec_demo import resources as R  # noqa: E402
from ngec_demo.style import health_sidebar, inject_css  # noqa: E402

st.set_page_config(page_title="NGEC", page_icon="🌍", layout="centered")
inject_css()

pages = [
    st.Page("pages/home.py", title="NGEC", icon="🌍", default=True),
    st.Page("pages/step1.py", title="1. Which event?", icon="1️⃣"),
    st.Page("pages/step2.py", title="2. Who did what?", icon="2️⃣"),
    st.Page("pages/step3.py", title="3. Which entity?", icon="3️⃣"),
    st.Page("pages/step4.py", title="4. What kind of actor?", icon="4️⃣"),
    st.Page("pages/step5.py", title="5. When and where?", icon="5️⃣"),
]

page = st.navigation(pages)

# The classifier's DemoModelWarning is captured at load time and parked in
# session state; showing it here means it is said once, not on every rerun.
health_sidebar(R.health(), st.session_state.get("model_notes", []))
page.run()
