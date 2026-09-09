"""Timing: the same document coded in every mode, component by component."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.examples import DOCUMENTS
from ngec_demo.style import lede, mode_badge, timing_table

MODES = R.available_modes()

st.title("Timing")
lede("The same document through the whole pipeline in every available mode, "
     "timed down to the encoder calls.")

if "tm_doc" not in st.session_state:
    st.session_state.tm_doc = DOCUMENTS[0].text
    st.session_state.tm_pub = pd.to_datetime(DOCUMENTS[0].pub_date).date()

cols = st.columns(len(DOCUMENTS))
for col, doc in zip(cols, DOCUMENTS):
    if col.button(doc.title, width="stretch", key=f"tm_ex_{doc.key}"):
        st.session_state.tm_doc = doc.text
        st.session_state.tm_pub = pd.to_datetime(doc.pub_date).date()
        st.session_state.pop("tm_results", None)
        st.rerun()

st.text_area("Document", key="tm_doc", height=170)
left, right = st.columns([1, 3])
pub_date = left.date_input("Published", key="tm_pub")
right.write("")
run = right.button("Time it in every mode", type="primary")

if run:
    results = {}
    for mode in MODES:
        with st.spinner(f"Coding the document in {mode_badge(mode)} mode…"):
            results[mode] = steps.run_pipeline(st.session_state.tm_doc,
                                               str(pub_date), mode=mode)
    st.session_state["tm_results"] = results

results = st.session_state.get("tm_results")
if results:
    st.subheader("Pipeline")
    timing_table(columns={mode: result["breakdown"]
                          for mode, result in results.items()})
    st.caption(" · ".join(f"{mode_badge(mode)} {result['timing']['total']:.1f} s"
                          for mode, result in results.items()))

st.subheader("What runs where")
devices = {mode: R.component_devices(mode) for mode in MODES}
names = list(dict.fromkeys(name for row in devices.values() for name in row))
st.dataframe(pd.DataFrame([{"component": name,
                            **{mode_badge(mode): row.get(name, "—")
                               for mode, row in devices.items()}}
                           for name in names]),
             hide_index=True, width="stretch")

st.subheader("Load and warm-up")
report = R.load_report()
if report:
    st.dataframe(pd.DataFrame(report), hide_index=True, width="stretch")
else:
    st.caption("Nothing has been loaded in this process yet.")
st.caption(f"CPU mode runs torch on {R.CPU_THREADS} threads, to match the "
           "deployment box; spaCy and the geoparser are on CPU in both modes. "
           "Loading and warm-up happen once per mode and are not in the table "
           "above; every number is a single run.")
