"""Step 3: which entity is this span? Wikipedia search, then a ranker."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import hbar_chart, json_block, lede, mode_badge, timing_table

# Two full names, which the ranker resolves with near certainty, and an acronym
# that is a dozen organisations until the context names a country. A bare
# surname is a bad example *without* the context expansion -- searched
# literally it returns 200 title matches and the right article is usually not
# among them -- so the fourth example is the one that shows the expansion
# working: the step searches "Boris Johnson" because the context says so.
EXAMPLES: list[tuple[str, str, str]] = [
    ("Macron", "Emmanuel Macron",
     "President Emmanuel Macron said the pension reforms would proceed despite "
     "the demonstrations in Paris."),
    ("Boris Johnson", "Boris Johnson",
     "The prime minister faced questions in the House of Commons over parties "
     "held in Downing Street during lockdown."),
    ("ANC · South Africa", "the ANC",
     "The ANC lost its parliamentary majority in South Africa for the first time "
     "since 1994."),
    # The bare surname: only the context expansion makes this resolvable.
    ("Johnson · surname", "Johnson",
     "Boris Johnson faced questions in the House of Commons over parties held in "
     "Downing Street during lockdown. Johnson, who has been PM since 2019, has "
     "faced criticism."),
]

st.title("3. Which entity?")
lede("A span is searched against a Wikipedia index and every candidate article "
     "is scored by a ranker that reads the surrounding sentence.")

if "s3_span" not in st.session_state:
    st.session_state.s3_span = EXAMPLES[0][1]
    st.session_state.s3_context = EXAMPLES[0][2]

cols = st.columns(len(EXAMPLES))
for col, (label, span, context) in zip(cols, EXAMPLES):
    if col.button(label, width="stretch", key=f"ex3_{label}"):
        st.session_state.s3_span = span
        st.session_state.s3_context = context
        st.session_state.pop("s3_result", None)
        st.rerun()

st.text_input("Entity span", key="s3_span")
st.text_area("Context — the sentence the span came from", key="s3_context", height=90)
run = st.button("Look up", type="primary")

health = R.health(R.current_mode())
missing = [name for name in ("Elasticsearch", "wiki index")
           if not health.get(name, {}).get("ok")]
if missing:
    st.warning(f"Not available: {', '.join(missing)}. "
               "This page needs Elasticsearch and the wiki index.")

if run:
    with st.spinner("Searching Wikipedia and scoring the candidates…"):
        st.session_state["s3_asked"] = st.session_state.s3_span
        st.session_state["s3_result"] = steps.resolve_entity(
            st.session_state.s3_span, context=st.session_state.s3_context)

result = st.session_state.get("s3_result")
if result:
    best = result["best"]
    if best:
        st.subheader("Article")
        st.markdown(f"### [{best['title']}]({best['url']})")
        if best.get("short_desc"):
            st.caption(best["short_desc"])
        intro = (best.get("intro_para") or "").strip()
        if intro:
            st.write(intro[:300] + ("…" if len(intro) > 300 else ""))
        st.metric("Ranker score", f"{best['ranker_score']:.3f}")
    else:
        top = (result["candidates"] or [{}])[0]
        st.info("No article cleared the ranker's bar. Highest scoring candidate: "
                f"{top.get('title', '—')} ({top.get('ranker_score', 0):.3f}).")
    st.caption(f"{result['seconds']:.1f} s · {mode_badge(result['mode'])}")

    # The span is not always what got searched: NER over the context expands a
    # bare surname, and a second search covers the form that was dropped.
    if result.get("query") and result["query"] != st.session_state.get("s3_asked"):
        note = f'searched as "{result["query"]}" (expanded from the context)'
        if result.get("alt_query"):
            note += f' · also searched "{result["alt_query"]}"'
        st.caption(note)

    st.subheader("3.1 Candidates")
    candidates = result["candidates"]
    if candidates:
        table = pd.DataFrame([{
            "title": c.get("title", ""),
            "short description": c.get("short_desc") or "",
            "ranker": round(float(c.get("ranker_score") or 0), 3),
            "ES score": round(float(c.get("raw_es_score") or 0), 1),
            "exact title": bool(c.get("exact_title_match")),
            "redirect": bool(c.get("redirect_match")),
        } for c in candidates])
        st.dataframe(table, hide_index=True, width="stretch", height=280)
        st.caption(f"{len(table)} candidates returned by the search.")
        top10 = table.head(10)[["title", "ranker"]]
        st.altair_chart(hbar_chart(top10, "title", "ranker"), width="stretch")
    else:
        st.info("The search returned nothing for this span.")

    json_block(result, label="Raw JSON")
    with st.expander("Timing breakdown"):
        timing_table(result["timing"])
