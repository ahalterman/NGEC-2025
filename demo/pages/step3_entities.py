"""Step 3: linking named entities to Wikipedia pages."""

import streamlit as st

from ngec_demo import paper, pipeline, ui
from ngec_demo.theme import kv_table, note, rule

ui.setup()
ui.step_rail("step3")

ui.page_header(
    kicker="Step 3 of 5",
    title="Entity linking",
    standfirst=(
        "“Macron”, “President Macron” and “the French president” should end up as the same "
        "analytical unit. This step links a name to its Wikipedia article using an offline "
        "index, then reads role and country off that page."
    ),
    paper_keys=("step3", "wiki_model"),
)

st.markdown(
    """
This is where the pipeline replaces the actor dictionary. Instead of a hand-built list mapping
strings to codes, a name is looked up in a local copy of Wikipedia and the article's infobox
and categories supply the coding. Wikipedia is maintained by someone else, covers far more
entities than any dictionary, and — crucially — records **when** someone held an office, so
the same name can code differently depending on the event's date.

Linking is retrieve-then-rank: Elasticsearch returns up to 200 candidate articles, and an
XGBoost ranker picks among them using title match, redirects, context similarity and other
features. A linking failure is usually the ranker choosing wrongly, not the page being absent.
"""
)

rule()

st.markdown("## Trace one mention")

col_a, col_b = st.columns([2, 1], gap="large")
with col_a:
    mention = st.text_input("Actor mention", value="Emmanuel Macron")
with col_b:
    query_date = st.text_input("Event date", value="2023-03-15",
                               help="Offices are read as of this date.")

context = st.text_input(
    "Document context (optional)",
    value="Thousands of protesters gathered in Paris against pension reforms.",
    help="The ranker uses context similarity to disambiguate between candidate pages.",
)

if not mention.strip():
    st.stop()

if pipeline.resources.get_es() is None:
    note(
        "This panel needs the Wikipedia Elasticsearch index, which is not reachable from this "
        "server right now. The retrieval and ranking steps below cannot run.",
        title="Index unavailable",
        tone="bad",
    )
    st.stop()

st.markdown("### 1 · Retrieve")
st.markdown(
    '<div class="ngec-meta">What Elasticsearch returns for this query, before ranking. '
    "The ranker sees up to 200 of these.</div>",
    unsafe_allow_html=True,
)

with st.spinner("Searching the index…"):
    candidates = pipeline.wiki_candidates(mention, limit=10)

if candidates:
    rows = [
        (c["title"], f'{c["short_description"] or "—"}')
        for c in candidates
    ]
    st.markdown(kv_table(rows), unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="ngec-meta">No candidates returned.</div>', unsafe_allow_html=True
    )

st.markdown("### 2 · Rank, then read the page")

with st.spinner("Ranking and coding…"):
    coded = pipeline.resolve_actor(mention, context=context, query_date=query_date)

if not coded:
    note(
        "No code was produced. That happens when nothing clears the ranker's bar, which is "
        "the intended behaviour for a mention that is not a resolvable entity — but it also "
        "happens when the ranker picks nothing for an entity that is genuinely there.",
        title="Unresolved",
        tone="warn",
    )
elif coded.get("error"):
    st.error(coded["error"])
else:
    wiki = coded.get("wiki") or ""
    wiki_link = (
        f'<a href="https://en.wikipedia.org/wiki/{wiki.replace(" ", "_")}" '
        f'target="_blank">{wiki}</a>'
        if wiki
        else ""
    )
    st.markdown(
        kv_table([
            ("Query", coded.get("query", mention)),
            ("Wikipedia page", wiki_link),
            ("Country", coded.get("country", "")),
            ("Role code", f'<strong>{coded.get("code_1", "")}</strong>'),
            ("Secondary code", coded.get("code_2", "")),
            ("Job at that date", coded.get("actor_wiki_job", "")),
            ("Source of the code", coded.get("source", "")),
            ("Other codes considered", ", ".join(coded.get("all_code1s", []) or [])),
        ]),
        unsafe_allow_html=True,
    )
    ui.raw_json("Full coded actor record", coded)

note(
    "Change the event date to 2016 and re-run a mention like “the German Chancellor” or a "
    "politician who has since left office. The Wikipedia infobox carries office start and end "
    "dates, and the coder reads the office that was current on the date you give it. A static "
    "actor dictionary cannot do this, and it is the main reason entity linking is worth the "
    "infrastructure.",
    title="Try changing the date",
)

rule()

st.markdown("## In the pipeline")

# This step never sees the article: it is handed the mention strings step 2
# extracted, so those — and not the prose — are what the page shows as input.
text, pub_date, example = ui.example_picker("step3", show_source=False)
ui.what_to_notice(example)

result = ui.run_with_progress(text, pub_date, stop_after="actors",
                              label="Running through actor resolution")

if not ui.run_error(result):
    records = result.records_after("actors")

    ui.stage_input(
        ui.mention_rows(result.records_after("attributes")),
        explain="Each mention is looked up on its own, with the document supplied only as "
                "context for the ranker.",
    )

    st.markdown(
        f'<div class="ngec-meta">{result.step("actors").note}</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    ui.render_records(records)

    with st.expander("Why some mentions get no code", expanded=False):
        st.markdown(
            """
Three different things produce an empty `actor` list:

- **The attribute model returned `N/A`.** There was no actor span to resolve. Nothing failed.
- **The mention is generic** — “protesters”, “riot police”. There is no Wikipedia page to
  find, and the mention is handled by step 4 instead, from the pattern file.
- **The ranker chose nothing** for a name that does have a page. This is the real failure
  mode, and it is why the trace above shows the candidate list separately from the choice.
"""
        )

paper.ref("step3", "wiki_model", "eval_wiki")
