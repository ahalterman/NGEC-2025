"""Coreference: what the pipeline resolves, and what it does not."""

import streamlit as st

from ngec_demo import paper, pipeline, ui
from ngec_demo.theme import kv_table, note, rule

ui.setup()

ui.page_header(
    kicker="Hard cases",
    title="Coreference",
    standfirst=(
        "One person, four surface forms: “Recep Tayyip Erdogan”, “Erdogan”, “the president”, "
        "“he”. The pipeline handles the second of those and not the last two."
    ),
    paper_keys=("step3", "limitations"),
)

st.markdown(
    """
Three distinct problems travel under the coreference label, and they have different costs.

**Within-document entity coreference** — linking “Assad”, “the Syrian president” and “he” to
one actor record inside one article. This is what the pipeline partially addresses.

**Cross-document event deduplication** — the same event reported by four wire services, or
referenced again in a follow-up. The pipeline does not address this at all; `orig_id` tracks
a record back to its source story, but nothing merges across stories.

**Temporal echoes** — the same event re-reported later as history. That gets
[its own page](echoes).

Both of the first two are active NLP research areas (see Lu and Ng 2018, Liu et al. 2021).
What is here is a heuristic, not a solution, and this page shows exactly where it stops.
"""
)

rule()

st.markdown("## What the heuristic does")

st.markdown(
    """
When the attribute model returns a bare surname or an acronym, the resolver looks earlier in
the same document for an expanded named entity from spaCy's NER and searches Wikipedia with
that instead. “Erdogan” becomes “Recep Tayyip Erdogan”; “the SDF” becomes “Syrian Democratic
Forces”. That covers a real and common case — wire copy names someone in full once and uses
the surname thereafter.

It does nothing for a mention that is not a name.
"""
)

text, pub_date, example = ui.example_picker("coref")
ui.what_to_notice(example)

result = ui.run_with_progress(text, pub_date, stop_after="actors",
                              label="Running through actor resolution")

if not ui.run_error(result):
    st.markdown("### Coded actors from this document")
    ui.render_records(result.records_after("actors"), text)

rule()

st.markdown("## The same mention, with and without its document")

st.markdown(
    "Entity linking takes a context string as well as the mention. This is where the "
    "expansion shows up: the mention alone is often ambiguous or unresolvable, and the "
    "document supplies what disambiguates it."
)

mention = st.text_input("Mention", value="Erdogan")
context = st.text_area(
    "Context",
    value=("Turkish President Recep Tayyip Erdogan met European Union officials in Brussels "
           "on Thursday to discuss migration."),
    height=90,
)

if mention.strip() and pipeline.resources.get_es() is not None:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Mention alone**")
        bare = pipeline.resolve_actor(mention, context="")
        if bare and not bare.get("error"):
            st.markdown(
                kv_table([
                    ("Wikipedia", bare.get("wiki", "")),
                    ("Code", bare.get("code_1", "")),
                    ("Country", bare.get("country", "")),
                    ("Confidence", f'{bare.get("conf", 0):.3f}'
                        if bare.get("conf") is not None else ""),
                ]),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="ngec-meta">Unresolved.</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("**With context**")
        withctx = pipeline.resolve_actor(mention, context=context)
        if withctx and not withctx.get("error"):
            st.markdown(
                kv_table([
                    ("Wikipedia", withctx.get("wiki", "")),
                    ("Code", withctx.get("code_1", "")),
                    ("Country", withctx.get("country", "")),
                    ("Confidence", f'{withctx.get("conf", 0):.3f}'
                        if withctx.get("conf") is not None else ""),
                ]),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="ngec-meta">Unresolved.</div>', unsafe_allow_html=True)

st.markdown("### Try these")
st.markdown(
    """
Replace the mention with each of these, keeping the same context:

- `Erdogan` — a surname. Expanded and resolved.
- `the president` — a role mention. Not expanded; either unresolved or coded generically
  from the pattern file, which loses the identity.
- `He` — a pronoun. Nothing to search for.
"""
)

rule()

st.markdown("## An honest scorecard")

st.markdown(
    kv_table([
        ("Full name → page", "Handled. This is the ordinary case."),
        ("Surname → full name", "Handled, when the full name appears earlier in the document."),
        ("Acronym → expansion", "Handled, same mechanism."),
        ("Role mention (“the president”)",
         "<strong>Not handled.</strong> Coded generically at best; the identity is lost."),
        ("Pronoun (“he”, “they”)",
         "<strong>Not handled.</strong> Usually not extracted as an actor in the first place."),
        ("Definite description (“the former senior member”)",
         "<strong>Not handled.</strong>"),
        ("Same event across wire services",
         "<strong>Not addressed.</strong> Every story produces its own records."),
        ("Same event in a follow-up story",
         "<strong>Not addressed.</strong> See the temporal echoes page."),
    ]),
    unsafe_allow_html=True,
)

note(
    "The cost is not symmetric across research designs. If you are counting events, "
    "unresolved role mentions mostly cost you actor detail on events you already have. If you "
    "are building an actor-level network, they cost you edges — and they do so "
    "non-randomly, since prominent actors are exactly the ones journalists refer to by role "
    "after first mention.",
    title="Who this hurts",
)

paper.ref("step3", "limitations")
