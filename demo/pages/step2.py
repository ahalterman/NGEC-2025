"""Step 2: who did what to whom? A fine-tuned model reading a definition."""

import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

from ngec_demo import steps
from ngec_demo.style import json_block, lede

# (label, text, event type, definition override or None). The third is the
# point of the page: DETAIN is not a PLOVER type and the model has never been
# trained on it, so it arrives as a definition instead of a label.
EXAMPLES: list[tuple[str, str, str, str | None]] = [
    ("Protest",
     "Thousands of protesters blocked traffic in central Paris on Tuesday to "
     "demonstrate against the pension reform, and union leaders said they would "
     "call further strikes next week.",
     "PROTEST", None),
    ("Agreement",
     "Officials from Ethiopia and Eritrea signed a ceasefire agreement in Cairo "
     "on Monday, ending three weeks of fighting along their shared border.",
     "AGREE", None),
    ("DETAIN (custom type)",
     "Police in Nairobi arrested at least forty people on Saturday outside "
     "parliament and said those detained would appear in court on Monday.",
     "DETAIN",
     "One actor takes another into physical custody: arrests, detentions, "
     "kidnappings, and the seizure of hostages. The detaining party is the "
     "ACTOR and the person or group taken into custody is the RECIPIENT."),
]

# The codebook ships with the package; find it without importing ngec, which
# would pull the whole pipeline in on every rerun of this page.
_ngec = importlib.util.find_spec("ngec")
CODEBOOK = Path(_ngec.origin).parent / "assets" / "PLOVER_structured_codebook_updated.csv"


@st.cache_data(show_spinner=False)
def event_definitions() -> dict[str, str]:
    """{event type: definition}, in codebook order.

    The codebook has one row per mode, all repeating their type's definition;
    rows with no mode are the type-level rows, but six of the sixteen types have
    modes on every row, so the first non-empty definition is the fallback.
    """
    df = pd.read_csv(CODEBOOK)
    defs: dict[str, str] = {}
    for event_type, rows in df.groupby("event", sort=False):
        typed = rows[rows["mode"].isna()]
        source = typed if len(typed) else rows
        values = source["event_def"].dropna()
        defs[str(event_type)] = str(values.iloc[0]) if len(values) else ""
    return defs


DEFS = event_definitions()
TYPES = list(DEFS)

st.title("2. Who did what?")
lede("The extractor is given a definition, not a label, so it can pull spans "
     "for an event type it was never trained on — edit the definition and see.")

if "s2_text" not in st.session_state:
    st.session_state.s2_text = EXAMPLES[0][1]
    st.session_state.s2_type = EXAMPLES[0][2]
    st.session_state.s2_def = DEFS[EXAMPLES[0][2]]

cols = st.columns(len(EXAMPLES))
for col, (label, text, event_type, event_def) in zip(cols, EXAMPLES):
    if col.button(label, width="stretch", key=f"s2_ex_{event_type}"):
        st.session_state.s2_text = text
        st.session_state.s2_type = event_type
        st.session_state.s2_def = event_def or DEFS.get(event_type, "")
        st.session_state.pop("s2_result", None)
        st.rerun()


def _refill_definition() -> None:
    """A new type replaces the definition box with that type's codebook entry."""
    st.session_state.s2_def = DEFS.get(st.session_state.s2_type, "")


st.text_area("Text", key="s2_text", height=120)

current = st.session_state.s2_type
options = TYPES + ([current] if current not in TYPES else [])
st.selectbox("Event type", options, key="s2_type", accept_new_options=True,
             on_change=_refill_definition,
             help="Type a name that is not in the list to invent an event type.")
st.text_area("Definition", key="s2_def", height=110,
             help="Sent to the model verbatim. Edit it, or write one for a new type.")

if st.button("Extract", type="primary"):
    given = st.session_state.s2_def.strip()
    # Only override when the box no longer matches the codebook: an untouched
    # box lets the model do its own lookup, which is the production path.
    override = given if given != DEFS.get(st.session_state.s2_type, "") else None
    with st.spinner("Loading the extractor and reading the text…"):
        st.session_state["s2_result"] = steps.extract_attributes(
            st.session_state.s2_text, st.session_state.s2_type, event_def=override)

result = st.session_state.get("s2_result")
if result:
    def _cell(value):
        if isinstance(value, (list, tuple)):
            return "; ".join(str(v) for v in value)
        return "" if value is None else str(value)

    rows = [{"actor": _cell(r.get("actor")), "recipient": _cell(r.get("recipient")),
             "location": _cell(r.get("location")), "date": _cell(r.get("date")),
             "anchor quote": _cell(r.get("anchor_quote"))}
            for r in result["records"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("The model found no event of this type in the text.")
    st.caption(f"{result['seconds']:.1f}s · {len(rows)} record(s)")

    json_block(result["records"], label="Attributes (JSON)")
    with st.expander("Prompt sent to the model"):
        st.code(result["prompt"] or "(no prompt: the model did not load)",
                language="text")
