"""Step 4: what kind of actor is it? Wikipedia or a dictionary, then codes."""

import datetime as dt
from pathlib import Path

import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import (code_chips, code_metrics, json_block, lede,
                             mode_badge, running, timing_table)

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
# can invent one. The bundled dictionary (ngec/assets/PLOVER_agents.txt) has no
# pattern for the Sunrise Movement, so the bundled column codes the span from
# its Wikipedia article instead and returns some general civil-society code
# with this file it is ENV either way. The exact bundled code is whatever the
# wiki coder makes of the article, so read it off the page's own "Bundled
# dictionary" row rather than trusting a code written in a comment.
# "Extinction Rebellion" is worth typing in as a second span: the bundled
# dictionary has a REBELLION pattern, so it goes somewhere else again.
CUSTOM_AGENTS = """CLIMATE_ACTIVISTS [ENV]
CLIMATE_PROTESTERS [ENV]
SUNRISE_MOVEMENT [ENV]
SUNRISE_MOVEMENT [ENV]
GREENPEACE [ENV]

REBELS [REB]
INSURGENTS [REB]
"""
CUSTOM_SPAN = "Extinction Rebellion"

TMP_DIR = Path(__file__).resolve().parent.parent / ".tmp"

st.title("4. Categorizing Actors and Recipients")
lede("""At this point in the pipeline, we've extracted the actors and recipients
for an event, and resolved named entities to Wikipedia. But to be useful in downstream
     inference, we usually want to *categorize* actors and recipients into groups.

We assign entities a country code and role codes by comparing them to a list of countries
     and a defined list of patterns, using semantic similarity.
     """)

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
st.text_area("Context (optional): the sentence the span came from", key="s4_context", height=90)
left, right = st.columns([1, 3])
query_date = left.date_input("Query date", key="s4_date")
right.write("")
run = right.button("Code the actor", type="primary")
st.caption("Codes are as of the query date: Angela Merkel is GOV while in office "
           "and ELI — a former official — after it.")

health = R.health(R.current_mode())
missing = [name for name in ("Elasticsearch", "wiki index")
           if not health.get(name, {}).get("ok")]
if missing:
    st.warning(f"Not available: {', '.join(missing)}. "
               "This page needs Elasticsearch and the wiki index.")

if run:
    with running(R.current_mode(), "Coding the actor…"):
        st.session_state["s4_result"] = steps.categorize_entity(
            st.session_state.s4_span, context=st.session_state.s4_context,
            query_date=str(query_date))


def _codes(result: dict) -> None:
    """The three coded fields, side by side.

    The role code and the country code are the answer, so they are the largest
    thing on the page; code_2 is a second role the resolver sometimes adds and
    is usually empty, and drawing it at the same weight made the answer harder
    to find than it should be.
    """
    code_metrics([("country", result["country"]),
                  ("code_1", result["code_1"]),
                  ("code_2", result["code_2"]),
                  ],
                 lead=("country", "code_1"))


result = st.session_state.get("s4_result")
if result:
    st.subheader("Codes")
    _codes(result)

    st.subheader("How it was decided")
    if result["used_wikipedia"]:
        st.markdown(f"Resolved through Wikipedia: [{result['wiki']}]({result['url']})")
    elif result["source"] == "country only":
        st.markdown("A country name only: resolves to a country code with no role code.")
    else:
        st.markdown("Good match against the agents file, skipping Wikipedia lookup.")
    st.caption(f"source: {result['source'] or '—'} · reason: "
               f"{result['best_reason'] or '—'}")
    if result["description"]:
        st.write(result["description"])
    # Every code the resolver weighed, with the one it chose at full contrast:
    # the list is only interesting next to the answer it produced.
    considered = list(dict.fromkeys(
        c for c in result["all_code1s"] + result["all_code2s"] if c))
    chips = code_chips(considered, winner=result["code_1"])
    st.markdown(f"Codes considered: {chips}" if chips else "Codes considered: none",
                unsafe_allow_html=True)
    st.caption(f"{result['seconds']:.1f} s · {mode_badge(result['mode'])}")

    json_block(result, label="Raw JSON")
    with st.expander("Timing breakdown"):
        timing_table(result["timing"])

st.subheader("4.1 Custom agents file")
lede("""
This is one of the richest (and easiest) places for customization. A project on subnational politics
probably needs to make finer distinctions between groups than an international relations project.

     To customize the codes, write a small set of patterns with the role code they should be assigned.

In this example, "Extinction Rebellion" gets assigned a "rebel" role using the generic agents file that
     NGEC ships with. Creating a small number of `ENV` patterns changes this, *even though "Extinction Rebellion" is not one of the patterns.*

     (Try coding "soldiers": you should see that it has no match in the custom agents list).
     """)

if "s4_agents" not in st.session_state:
    st.session_state.s4_agents = CUSTOM_AGENTS
    st.session_state.s4_custom_span = CUSTOM_SPAN

st.text_area("Custom agents file", key="s4_agents", height=220)
st.text_input("Entity span", key="s4_custom_span")
run_custom = st.button("Code with each agents file")

if run_custom:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "custom_agents.txt"
    path.write_text(st.session_state.s4_agents, encoding="utf-8")
    with running(R.current_mode(),
                 "Embedding the custom patterns and coding the span..."):
        st.session_state["s4_custom"] = {
            "default": steps.categorize_entity(st.session_state.s4_custom_span,
                                               query_date=str(query_date)),
            "custom": steps.categorize_entity(st.session_state.s4_custom_span,
                                              query_date=str(query_date),
                                              agents_file=str(path)),
        }

custom = st.session_state.get("s4_custom")
if custom:
    for label, key in (("Bundled dictionary", "default"), ("Custom dictionary", "custom")):
        st.markdown(f"**{label}**")
        _codes(custom[key])
        st.caption(f"matched: {custom[key]['description'] or '—'} · "
                   f"source: {custom[key]['source'] or '—'} · "
                   f"{custom[key]['seconds']:.1f} s · "
                   f"{mode_badge(custom[key]['mode'])}")
    st.caption("The new dictionary is embedded once, then new queries are compared to the cached version.")
