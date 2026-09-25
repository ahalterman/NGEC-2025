"""Step 3: which entity is this span? Wikipedia search, then a ranker."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import (field_table, hbar_chart, json_block, lede,
                             mode_badge, running, timing_table)

# A full name, which the ranker resolves with near certainty; two mentions the
# way the attribute model actually returns them -- a title, a country and a
# name in one span -- which only resolve once the span is taken apart and the
# name alone is searched; two organisations that are a dozen candidates until
# the context names a country. A bare surname is a bad example *without* the
# context expansion -- searched literally it returns 200 title matches and the
# right article is usually not among them -- so the last example is the one
# that shows the expansion working: the step searches "Boris Johnson" because
# the context says so.
EXAMPLES: list[tuple[str, str, str]] = [
    ("Macron", "Emmanuel Macron",
     "President Emmanuel Macron said the pension reforms would proceed despite "
     "the demonstrations in Paris."),
    # Title + country + name. Searched whole, the ranker gives "Colin Powell"
    # 0.6; searched as the name, with the title and country as evidence, 0.99.
    ("Powell · with title", "former US Secretary of State Colin Powell",
     "Former US Secretary of State Colin Powell said the intelligence that led "
     "to the war had been flawed."),
    # The name comes *after* the title, with a country inside the title.
    ("Peña Nieto · with title", "the governor of Mexico State, Enrique Pena Nieto",
     "The governor of Mexico State, Enrique Pena Nieto, announced a new "
     "security plan for the state on Tuesday."),
    ("Myanmar armed groups", "Pyu Saw Htee",
     "148. The 348 abductions of 3437 "
     "children (231 boys, 112 girls) were attributed to the "
     "Myanmar armed forces, including related forces and affiliated militias (104) "
     "(Myanmar armed forces (89), jointly by Myanmar armed forces and affiliated militia "
     "(5), militia groups (3), jointly by Myanmar armed forces and Pyu Saw Htee (2), jointly "
     "by Myanmar Police Force and Pyu Saw Htee (2), jointly by Myanmar armed forces "
     "and Myanmar Police Force (2), by border guard forces (1)); United Wa State Army "
     "(UWSA) (137); AA (40); KIA (28); TNLA (15); ARSA (8); Pyu Saw Htee (4); "
     "MNDAA (3); unidentified perpetrators (3); People’s Defence Forces/local defence "
     "groups (2); SSPP/SSA (2); CNF (1) and Shanni Nationalities Army (1). Most children "
     "were abducted for the purpose of recruitment and use.") ,
    ("ELN · Colombia", "ELN",
     "41. Some 371 children (228 boys, 139 girls, 4 sex unknown) were recruited and"
     "used by Fuerzas Armadas Revolucionarias de Colombia-Ejército del Pueblo "
     "(FARC- EP) dissident groups (262), including Estado Mayor Central Fuerzas Armadas "
     "Revolucionarias de Colombia-Ejército del Pueblo (EMC FARC-EP) (126)), Ejército "
     "de Liberación Nacional (ELN) (54), Clan del Golfo (also known as Autodefensas "
     "Gaitanistas de Colombia) (29) and unidentified perpetrators (26). Some 57 children "
     "were used in combat roles and 182 children were released."),
    # The bare surname: only the context expansion makes this resolvable.
    ("Johnson · surname", "Johnson",
     "Boris Johnson faced questions in the House of Commons over parties held in "
     "Downing Street during lockdown. Johnson, who has been PM since 2019, has "
     "faced criticism."),
]

st.title("3. Resolving to Wikipedia")
lede("A entity span is searched against a Wikipedia index and every candidate article "
     "is scored by a ranker using the mention's context, if available..")

if "s3_span" not in st.session_state:
    st.session_state.s3_span = EXAMPLES[0][1]
    st.session_state.s3_context = EXAMPLES[0][2]

# Three to a row: six labels in one row of the centered layout get cut off.
for row in (EXAMPLES[:3], EXAMPLES[3:]):
    cols = st.columns(3)
    for col, (label, span, context) in zip(cols, row):
        if col.button(label, width="stretch", key=f"ex3_{label}"):
            st.session_state.s3_span = span
            st.session_state.s3_context = context
            st.session_state.pop("s3_result", None)
            st.rerun()

st.text_input("Entity span", key="s3_span")
st.text_area("Context (optional): the sentence or paragraph the span came from", key="s3_context", height=90)
run = st.button("Look up", type="primary")

health = R.health(R.current_mode())
missing = [name for name in ("Elasticsearch", "wiki index")
           if not health.get(name, {}).get("ok")]
if missing:
    st.warning(f"Not available: {', '.join(missing)}. "
               "This page needs Elasticsearch and the wiki index to run.")

if run:
    with running(R.current_mode(), "Searching Wikipedia and scoring the candidates…"):
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

    # The span is rarely what got searched. Say what was done to it, in the
    # order it happened: the country stripped off, the name cut out of the
    # title around it, the context expansion, the second surface form, and
    # any retry after a miss. This is the same `split_mention` the pipeline
    # runs, so the page cannot drift from step 4.
    asked = st.session_state.get("s3_asked", "")
    split = result.get("split") or {}
    st.subheader("How the search was constructed")
    st.markdown('Part of our "bag of tricks" are a set of techniques to help improve search performance, including expanding acronyms, cleaning up long spans, and distinguishing between searchable entities and broader context.')
    rows = []
    if split.get("country"):
        rows.append(("country", f'{split["country_name"] or split["country"]} '
                                f'({split["country"]}) was found in the span and removed before search'))
    elif split.get("country_name"):
        rows.append(("country", f'{split["country_name"]} — from the context'))
    if split.get("actor_desc"):
        rows.append(("description", split["actor_desc"]))
    rows.append(("searched as", result.get("query") or asked))
    field_table(rows)
    steps_taken = []
    if split.get("country"):
        steps_taken.append(f'stripped the country: "{split["trimmed_text"]}"')
    if split.get("core_query") and split["core_query"] != split.get("trimmed_text"):
        steps_taken.append(f'NER trimmed the span down to "{split["core_query"]}"i')
    if result.get("expanded"):
        steps_taken.append(f'expanded to "{result["query"]}" from the context')
    if result.get("alt_query"):
        steps_taken.append(f'also searched "{result["alt_query"]}"')
    for term in result.get("retries") or []:
        steps_taken.append(f'no article cleared the bar, so retried as "{term}"')
    if steps_taken:
        st.markdown("\n".join(f"- {line}" for line in steps_taken))
    else:
        st.caption("Searched as written: no country in the span, no title "
                   "around a name, nothing to expand from the context.")
    if split.get("actor_desc") or split.get("country_name"):
        st.caption("The description and country are not searched for; they "
                   "are scored against each candidate's introduction and "
                   "short description as ranker features.")

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
