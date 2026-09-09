"""End to end: a document in, coded event records out."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.examples import DOCUMENTS
from ngec_demo.style import json_block, lede, mode_badge, step_links, timing_table

st.title("NGEC")
lede("Next Generation Event Coder — turns a news story into structured political "
     "event records: who did what to whom, where, and when.")

if "doc" not in st.session_state:
    st.session_state.doc = DOCUMENTS[0].text
    st.session_state.pub_date = pd.to_datetime(DOCUMENTS[0].pub_date).date()

cols = st.columns(len(DOCUMENTS))
for col, doc in zip(cols, DOCUMENTS):
    if col.button(doc.title, width="stretch", key=f"ex_{doc.key}"):
        st.session_state.doc = doc.text
        st.session_state.pub_date = pd.to_datetime(doc.pub_date).date()
        st.session_state.pop("result", None)
        st.rerun()

text = st.text_area("Document", key="doc", height=170)
left, right = st.columns([1, 3])
pub_date = left.date_input("Published", key="pub_date")
right.write("")
run = right.button("Run NGEC", type="primary")

mode = R.current_mode()
health = R.health(mode)
missing = [name for name, row in health.items() if not row.get("ok")]
if missing:
    st.warning(f"Not available: {', '.join(missing)}. "
               "The steps that need them will return nothing.")

if run:
    with st.spinner("Loading models and coding the document…"):
        _, notes = R.get_classifier(mode)
        st.session_state["model_notes"] = notes
        st.session_state["result"] = steps.run_pipeline(text, str(pub_date),
                                                        mode=mode)

result = st.session_state.get("result")
if result:
    st.subheader("Events")
    rows = steps.events_table(result["events"])
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No event cleared the classifier's thresholds, or the attribute "
                "model declined every candidate.")
    if result.get("error"):
        st.warning(result["error"])
    if result["dropped"]:
        st.caption("Classified but dropped by the attribute model: "
                   + ", ".join(result["dropped"]))

    json_block(result["events"], label="Full records (JSON)")

    st.subheader("Timing")
    timing_table(result["breakdown"])
    st.caption(f"{result['timing']['total']:.1f} s · {mode_badge(result['mode'])} · "
               "model load and warm-up are not counted — they happen once per "
               "mode and are on the Timing page.")

st.subheader("Five steps")
lede("Each stage of the pipeline, on its own, with its own inputs.")
step_links()
