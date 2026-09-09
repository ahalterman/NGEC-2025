"""Step 2: who did what to whom? A fine-tuned model reading a definition."""

import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import (field_table, json_block, lede, mode_badge,
                             running, timing_table)

# (label, text, event type, definition override or None). The last two are the
# point of the page: neither DETAIN nor ELECTORAL_CONTENTION is a PLOVER type
# and the model has never been trained on either, so they arrive as definitions
# instead of labels. The texts are written for the demo in wire-service
# register rather than copied from a news outlet.
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
    # An ontology from another project entirely. The definition condenses the
    # ECAV codebook's "Defining electoral contention" (Daxecker, Amicarelli &
    # Jung, "Electoral contention and violence (ECAV): A new dataset", Journal
    # of Peace Research 56(5): 714-723, 2019, pp. 716-717): the italicised
    # definition itself, the two-actor requirement and the rally exclusion, the
    # "publicness" list of act types, and the substantive-plus-temporal link to
    # an identifiable election. ECAV's own variables call the two sides actor
    # and target; the last sentence maps them onto NGEC's ACTOR and RECIPIENT.
    # TODO(andy): replace with the verbatim definition from the ECAV codebook
    # (Daxecker, Amicarelli & Jung 2019). The paraphrase below was written from
    # memory of the codebook and has not been checked against it, so the button
    # says "(draft)" until it has been.
    ("ECAV contention (draft)",
     "Supporters of the opposition Unity Party blocked the main road into "
     "Kisumu on Thursday, three days after the parliamentary election, "
     "accusing the electoral commission of tampering with the count. Police "
     "fired tear gas to disperse the crowd of about 2,000, a party spokesman "
     "said.",
     "ELECTORAL_CONTENTION",
     "Public acts of mobilization, contestation, or coercion by state or "
     "non-state actors used to affect the electoral process, or arising in the "
     "context of electoral competition. A contentious event involves at least "
     "two actors who are on opposite sides of an issue, or an actor "
     "threatening or using violence against civilians; a rally or celebration "
     "in support of a candidate is not contention. The act must be publicly "
     "observable -- protests, clashes, arrests, arson, boycotts, intimidation, "
     "strikes, shootings, killings -- and tied to an identifiable election in "
     "both substance and timing. The party carrying out the act is the ACTOR "
     "and the party it is directed against is the RECIPIENT."),
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
        # The prompt box quotes the document, so it is re-rendered for the new
        # one rather than left showing the old example's text.
        st.session_state.pop("s2_prompt", None)
        st.session_state.pop("s2_prompt_base", None)
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


def _definition_override() -> str | None:
    """The definition box, unless it still matches the codebook.

    An untouched box lets the model do its own lookup, which is the production
    path.
    """
    given = st.session_state.s2_def.strip()
    return given if given != DEFS.get(st.session_state.s2_type, "") else None


def _rendered_prompt() -> str:
    """The prompt the boxes above would produce, extra attributes included."""
    return steps.attribute_prompt(
        st.session_state.s2_text, st.session_state.s2_type,
        event_def=_definition_override(),
        extra_fields=st.session_state.get("s2_extra", ""))


with st.expander("Advanced: edit the prompt"):
    st.text_input("Extra attributes", key="s2_extra",
                  placeholder="participant_count, weapons_used",
                  help="Comma-separated names, added to the JSON object the "
                       "model is asked to fill in.")
    if st.session_state.get("s2_show_prompt"):
        rendered = _rendered_prompt()
        if rendered and rendered != st.session_state.get("s2_prompt_base"):
            # Something above changed. Re-render an untouched box; leave an
            # edited one alone, since throwing away someone's edit is worse
            # than showing them a stale prompt they can reset.
            if st.session_state.get("s2_prompt") == st.session_state.get("s2_prompt_base"):
                st.session_state.s2_prompt = rendered
            st.session_state.s2_prompt_base = rendered
        st.text_area("Prompt", key="s2_prompt", height=320,
                     help="Everything the model sees, chat template included.")
        if st.session_state.s2_prompt != rendered:
            st.caption("Edited — sent as written, and no longer constrained to "
                       "the standard fields. Keep the role markers: on CPU it is "
                       "the blocks between them that reach the model.")
            if st.button("Reset to the generated prompt"):
                st.session_state.s2_prompt = rendered
                st.rerun()
    elif st.button("Show the prompt"):
        # A plain spinner, not the two-phase status the other buttons use:
        # st.status is an expander and this is already inside one.
        with st.spinner("Loading the extractor (about a minute the first time)…"):
            prompt = _rendered_prompt()
        st.session_state.s2_prompt = st.session_state.s2_prompt_base = prompt
        st.session_state.s2_show_prompt = True
        st.rerun()
    st.caption("The model fills in whatever fields the prompt asks for, but it "
               "was fine-tuned on the six standard ones: concrete additions (a "
               "crowd size, a weapon) come back well, while \"who reported it\" "
               "usually comes back N/A even when the text attributes the report.")

if st.button("Extract", type="primary"):
    # An edited prompt goes to the model as written; otherwise the extra
    # attributes are added to the standard one.
    edited = st.session_state.get("s2_prompt", "")
    override = (edited if edited and edited != st.session_state.get("s2_prompt_base")
                else None)
    with running(R.current_mode(), "Reading the text…"):
        st.session_state["s2_result"] = steps.extract_attributes(
            st.session_state.s2_text, st.session_state.s2_type,
            event_def=_definition_override(),
            extra_fields=st.session_state.get("s2_extra", ""),
            prompt_override=override)

result = st.session_state.get("s2_result")
if result:
    def _spans(value) -> str:
        """The extracted spans, quoted, joined with "; ".

        The typographic quotes are not decoration: every one of these values is
        a run of characters copied out of the document, and the quotes are what
        say so — especially for a one-word span that would otherwise read like a
        label the model chose.
        """
        values = value if isinstance(value, (list, tuple)) else [value]
        kept = [str(v) for v in values if v is not None and str(v).strip()]
        return "; ".join(f"“{v}”" for v in kept)

    # The six the model was trained on, in the order a reader wants them: what
    # kind of event, then who did it to whom, then where and when, then the
    # sentence all of it came from. The event type is the model's own echo of
    # what it was asked for, so it is not quoted as a span.
    STANDARD_ROWS = [("actor", "actor"), ("recipient", "recipient"),
                     ("location", "location"), ("date", "date"),
                     ("anchor quote", "anchor_quote")]

    records = result["records"]
    for index, record in enumerate(records, start=1):
        if len(records) > 1:
            # One text can yield several events. The heading uses the same mono
            # rule the results page puts over each event, so the records read as
            # separate records rather than one long table.
            st.markdown(f'<p class="event-head">Record {index}</p>',
                        unsafe_allow_html=True)
        fields = [("event type", str(record.get("event_type") or "—"))]
        fields += [(label, _spans(record.get(key)) or "—")
                   for label, key in STANDARD_ROWS]
        # Anything beyond the six is shown after them, in the order the model
        # returned it, so an extra attribute someone asked for is visibly the
        # model's answer and not one of the trained fields.
        fields += [(key.replace("_", " "), _spans(record.get(key)) or "—")
                   for key in record if key not in steps.STANDARD_ATTRIBUTES]
        field_table(fields)
    if not records:
        st.info("The model found no event of this type in the text.")
    if result.get("error"):
        st.warning(result["error"])
    st.caption(f"{result['seconds']:.1f} s · {mode_badge(result['mode'])} · "
               f"{len(records)} record(s)")

    json_block(result["records"], label="Attributes (JSON)")
    with st.expander("Prompt sent to the model"):
        st.code(result["prompt"] or "(no prompt: the model did not load)",
                language="text")
    with st.expander("Timing breakdown"):
        timing_table(result["timing"])
