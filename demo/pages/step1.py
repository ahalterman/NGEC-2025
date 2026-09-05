"""Step 1: which event type? Sixteen binary classifiers, sixteen thresholds."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import hbar_chart, lede

# (button label, text). One text that fires a single type, one that fires
# several, and one that should fire nothing at all -- a classifier demo that
# only ever shows hits hides the half of the job that is saying "no".
EXAMPLES: list[tuple[str, str]] = [
    ("Protest",
     "Thousands of protesters blocked traffic in central Paris on Tuesday to "
     "demonstrate against the pension reform."),
    ("Protest + arrests",
     "Police in Nairobi arrested at least forty people on Saturday during a "
     "demonstration against a proposed finance bill. The interior ministry said "
     "the arrests were made to protect public order."),
    ("Not an event",
     "Heavy rain washed out the second day of the test match in Wellington, and "
     "forecasters expect the storm to clear by Thursday morning."),
]

st.title("1. Which event?")
lede("Every text is scored by all sixteen PLOVER classifiers; each fires at its "
     "own tuned threshold, so a story can be several events, or none.")

if "s1_text" not in st.session_state:
    st.session_state.s1_text = EXAMPLES[0][1]

cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES):
    if col.button(label, width="stretch", key=f"s1_ex_{label}"):
        st.session_state.s1_text = text
        st.session_state.pop("s1_result", None)
        st.rerun()

st.text_area("Text", key="s1_text", height=120)
if st.button("Classify", type="primary"):
    with st.spinner("Loading the classifiers and scoring…"):
        _, notes = R.get_classifier()
        st.session_state["model_notes"] = notes
        st.session_state["s1_result"] = steps.classify(st.session_state.s1_text)

result = st.session_state.get("s1_result")
if result:
    types = pd.DataFrame(result["types"])
    st.altair_chart(hbar_chart(types, "event_type", "probability", "threshold"),
                    width="stretch")
    fired = [row["event_type"] for row in result["types"] if row["fired"]]
    st.caption(f"{result['seconds']:.2f}s · ticks are each class's threshold · "
               + (f"fired: {', '.join(fired)}" if fired else "nothing fired"))

    st.subheader("Modes")
    if not fired:
        st.caption("Nothing fired, so these are the modes of the top-scoring type — "
                   "mode models are trained conditional on their parent type.")
    for event_type, rows in result["modes"].items():
        st.markdown(f"**{event_type}**")
        modes = pd.DataFrame(rows)
        st.altair_chart(
            hbar_chart(modes, "mode", "probability", "threshold",
                       height=max(70, 20 * len(modes))),
            width="stretch")

with st.expander("Customizing event classification"):
    st.write("Coming soon: swapping in your own ontology and your own classifiers.")
