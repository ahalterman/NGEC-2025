"""End to end: a document in, coded event records out."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.examples import DOCUMENTS
from ngec_demo.style import (code_glossary, demo_model_note, event_heading,
                             field_table, json_block, lede, mode_badge,
                             role_codes_in, running, step_links, timing_table)

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
    with running(mode, "Coding the document…"):
        _, notes = R.get_classifier(mode)
        st.session_state["model_notes"] = notes
        st.session_state["result"] = steps.run_pipeline(text, str(pub_date),
                                                        mode=mode)

result = st.session_state.get("result")
if result:
    st.subheader("Events")
    events = result["events"]
    if events:
        # One story can fire an event type under several modes, and each mode is
        # coded separately, so the records can come back identical apart from
        # the mode. Say so once rather than leaving the reader to spot it.
        types = [e.get("event_type", "") for e in events]
        repeated = sorted({t for t in types if types.count(t) > 1})
        if repeated:
            st.caption(f"{', '.join(repeated)} fired under more than one mode, "
                       "so it appears once per mode; the two records can "
                       "otherwise be identical.")
        codes: list[str] = []
        for i, event in enumerate(events, start=1):
            fields = steps.event_fields(event)
            # The type and the mode are the first two rows of the table below,
            # so the heading only numbers the events.
            event_heading(i)
            field_table(fields)
            codes += [code for name, value in fields if name.endswith("code")
                      for code in role_codes_in(value)]
        # Only the codes actually on screen: the full PLOVER list is forty
        # entries and would bury the three this document produced.
        glossary = code_glossary(list(dict.fromkeys(codes)))
        if glossary:
            st.caption(f"Codes: {glossary}. A code is the actor's country "
                       "followed by its role.")
    else:
        st.info("No event cleared the classifier's thresholds, or the attribute "
                "model declined every candidate.")
    if result.get("error"):
        st.warning(result["error"])
    if result["dropped"]:
        st.caption("Classified but dropped by the attribute model: "
                   + ", ".join(result["dropped"]))

    json_block(result["events"], label="Full records (JSON)")
    demo_model_note()

    st.subheader("Timing")
    timing_table(result["breakdown"])
    st.caption(f"{result['timing']['total']:.1f} s · {mode_badge(result['mode'])} · "
               "model load and warm-up are not counted — they happen once per "
               "mode and are on the Timing page.")

st.subheader("Five steps")
lede("Each stage of the pipeline, on its own, with its own inputs.")
step_links()
