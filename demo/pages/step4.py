"""Step 4: what kind of actor is it? Wikipedia or a dictionary, then codes."""

import datetime as dt
from pathlib import Path

import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import json_block, lede

# One span per route through the coder: a named person only Wikipedia knows, a
# generic role the actor dictionary matches on its own, and a bare country.
EXAMPLES: list[tuple[str, str, str]] = [
    ("Angela Merkel", "Angela Merkel",
     "Former German chancellor Angela Merkel criticised the government's handling "
     "of the energy crisis."),
    ("riot police", "riot police",
     "Riot police fired tear gas at protesters who had blocked the road to the "
     "airport."),
    ("Ethiopia", "Ethiopia",
     "Ethiopia and Eritrea signed a ceasefire agreement in Cairo on Monday."),
]

# The agents-file format: a pattern, then its code in square brackets, with
# underscores for spaces. ENV is not a PLOVER code -- the point is that a project
# can invent one. The bundled dictionary codes "climate activists" as CVL and
# "Extinction Rebellion" as REB (it matches on "rebellion"); the custom file
# gives both ENV.
CUSTOM_AGENTS = """CLIMATE_ACTIVISTS [ENV]
CLIMATE_PROTESTERS [ENV]
EXTINCTION_REBELLION [ENV]
GREENPEACE [ENV]"""
CUSTOM_SPAN = "Extinction Rebellion"

TMP_DIR = Path(__file__).resolve().parent.parent / ".tmp"

st.title("4. What kind of actor?")
lede("A span becomes a PLOVER actor code — a role and a country — from its "
     "Wikipedia article, from the actor dictionary, or from the country name alone.")

if "s4_span" not in st.session_state:
    st.session_state.s4_span = EXAMPLES[0][1]
    st.session_state.s4_context = EXAMPLES[0][2]
    st.session_state.s4_date = dt.date.today()

cols = st.columns(len(EXAMPLES))
for col, (label, span, context) in zip(cols, EXAMPLES):
    if col.button(label, width="stretch", key=f"ex4_{label}"):
        st.session_state.s4_span = span
        st.session_state.s4_context = context
        st.session_state.s4_date = dt.date.today()
        st.session_state.pop("s4_result", None)
        st.rerun()

st.text_input("Entity span", key="s4_span")
st.text_area("Context — the sentence the span came from", key="s4_context", height=90)
left, right = st.columns([1, 3])
query_date = left.date_input("Query date", key="s4_date")
right.write("")
run = right.button("Code the actor", type="primary")
st.caption("Codes are as of the query date: Angela Merkel is GOV while in office "
           "and ELI — a former official — after it.")

health = R.health()
missing = [name for name, row in health.items() if not row.get("ok")]
if missing:
    st.warning(f"Not available: {', '.join(missing)}. "
               "This page needs Elasticsearch and the wiki index.")

if run:
    with st.spinner("Coding the actor…"):
        st.session_state["s4_result"] = steps.categorize_entity(
            st.session_state.s4_span, context=st.session_state.s4_context,
            query_date=str(query_date))


def _codes(result: dict) -> None:
    """The three coded fields, side by side."""
    one, two, three = st.columns(3)
    one.metric("code_1", result["code_1"] or "—")
    two.metric("code_2", result["code_2"] or "—")
    three.metric("country", result["country"] or "—")


result = st.session_state.get("s4_result")
if result:
    st.subheader("Codes")
    _codes(result)

    st.subheader("How it was decided")
    if result["used_wikipedia"]:
        st.markdown(f"Resolved through Wikipedia: [{result['wiki']}]({result['url']})")
    elif result["source"] == "country only":
        st.markdown("A country name and nothing else — a country code, no role code.")
    else:
        st.markdown("Matched against the agents file, no Wikipedia lookup.")
    st.caption(f"source: {result['source'] or '—'} · reason: "
               f"{result['best_reason'] or '—'}")
    if result["description"]:
        st.write(result["description"])
    codes = " ".join(f"`{c}`" for c in result["all_code1s"] + result["all_code2s"])
    st.markdown(f"Codes considered: {codes}" if codes else "Codes considered: none")
    st.caption(f"{result['seconds']:.1f} s")

    json_block(result, label="Raw JSON")

st.subheader("4.1 Custom agents file")
lede("Swap in your own actor dictionary: one pattern per line, its code in "
     "brackets, underscores for spaces.")

if "s4_agents" not in st.session_state:
    st.session_state.s4_agents = CUSTOM_AGENTS
    st.session_state.s4_custom_span = CUSTOM_SPAN

st.text_area("Agents file", key="s4_agents", height=120)
st.text_input("Entity span", key="s4_custom_span")
run_custom = st.button("Code with both dictionaries")

if run_custom:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "custom_agents.txt"
    path.write_text(st.session_state.s4_agents, encoding="utf-8")
    with st.spinner("Embedding the custom patterns and coding the span…"):
        st.session_state["s4_custom"] = {
            "default": steps.categorize_entity(st.session_state.s4_custom_span,
                                               query_date=str(query_date)),
            "custom": steps.categorize_entity(st.session_state.s4_custom_span,
                                              query_date=str(query_date),
                                              agents_file=str(path)),
        }

custom = st.session_state.get("s4_custom")
if custom:
    for label, key in (("Bundled dictionary", "default"), ("Your dictionary", "custom")):
        st.markdown(f"**{label}**")
        _codes(custom[key])
        st.caption(f"matched: {custom[key]['description'] or '—'} · "
                   f"source: {custom[key]['source'] or '—'} · "
                   f"{custom[key]['seconds']:.1f} s")
    st.caption("A new file costs one embedding pass over its patterns; the "
               "resolver is then cached per file.")
