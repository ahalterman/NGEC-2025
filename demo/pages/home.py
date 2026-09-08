"""Landing page: what this is, and a short guided tour."""

import streamlit as st

from ngec_demo import paper, ui
from ngec_demo.theme import note, rule

ui.setup()

ui.page_header(
    kicker="Creating Custom Event Data: A Bag-of-Tricks",
    title="NGEC, running",
    standfirst=(
        "This is the pipeline from the paper, live. Paste a news story and watch it become "
        "structured event records — or open any single step and look at what that component "
        "actually does with the text."
    ),
    paper_keys=("framework",),
)

st.markdown(
    """
Automated event data has usually meant taking someone else's ontology and someone else's
dictionaries. This pipeline is built the other way round: five steps, each replaceable, so
that a researcher can change the parts that do not fit their question and keep the rest.

The demo exists because that claim is easy to make and hard to check. Every panel here runs
the real package on the text you give it — nothing is pre-rendered, and the cases where the
pipeline does badly are on the site too, marked as such.
"""
)

rule()

st.markdown("## A ten-minute tour")

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("**Start with the whole thing**")
    st.page_link("pages/end_to_end.py", label="A document becomes events")
    st.markdown(
        '<div class="ngec-meta">One story in, one coded record per event out, with every '
        "intermediate state visible and each step timed.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.page_link("pages/step2_attributes.py", label="…and where it goes wrong")
    st.markdown(
        '<div class="ngec-meta">The attribute model is the step that decides whether a '
        "record is usable. Its failures are shown rather than filtered out.</div>",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown("**Then open up a step**")
    for path, label, blurb in [
        ("pages/step1_detection.py", "1 · Event detection",
         "Sixteen binary classifiers, each with its own threshold."),
        ("pages/step2_attributes.py", "2 · Attribute extraction",
         "A 0.6B fine-tuned model pulling four spans out of the document."),
        ("pages/step3_entities.py", "3 · Entity linking",
         "Names to Wikipedia pages, through an offline index."),
        ("pages/step4_categories.py", "4 · Entity categorisation",
         "Pages and generic mentions to ontology codes."),
        ("pages/step5_dates_places.py", "5 · Dates and places",
         "“last Tuesday” to a date; “Aleppo” to coordinates."),
    ]:
        st.page_link(path, label=label)
        st.markdown(f'<div class="ngec-meta">{blurb}</div>', unsafe_allow_html=True)

rule()

st.markdown("## Where it breaks")

st.markdown(
    """
Three classes of failure get their own pages, because they are the ones most likely to bite a
researcher and least likely to show up in a demo built to impress.
"""
)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.page_link("pages/echoes.py", label="Temporal echoes")
    st.markdown(
        '<div class="ngec-meta">News reports old events in the present tense. '
        "Anniversaries, retrospectives, campaign summaries — all grammatically identical "
        "to current reporting.</div>",
        unsafe_allow_html=True,
    )
with c2:
    st.page_link("pages/coref.py", label="Coreference")
    st.markdown(
        '<div class="ngec-meta">The pipeline expands surnames to full names. It does not '
        "resolve “the president” or “he”, and you can see exactly where it stops.</div>",
        unsafe_allow_html=True,
    )
with c3:
    st.page_link("pages/nonenglish.py", label="Non-English text")
    st.markdown(
        '<div class="ngec-meta">Every component is English-only. This page shows which ones '
        "degrade gracefully and which fail outright.</div>",
        unsafe_allow_html=True,
    )

st.write("")
st.page_link("pages/vs_llm.py", label="How this compares to just prompting a frontier LLM",
             )

rule()

st.markdown("## Making it yours")

st.markdown(
    """
The pipeline's value is supposed to be that you can replace the parts that do not fit. These
pages make that concrete rather than aspirational — including an honest accounting of the
setup burden, which is larger than a table in the paper can convey.
"""
)

d1, d2, d3 = st.columns(3, gap="large")
with d1:
    st.page_link("pages/customize_codebook.py", label="Your event types")
    st.markdown(
        '<div class="ngec-meta">Upload a codebook, have an LLM apply it to real news, and '
        "fit classifiers you can drop into the pipeline.</div>",
        unsafe_allow_html=True,
    )
with d2:
    st.page_link("pages/customize_actors.py", label="Your actor categories")
    st.markdown(
        '<div class="ngec-meta">Edit the category patterns and re-code the same document '
        "immediately. No retraining involved.</div>",
        unsafe_allow_html=True,
    )
with d3:
    st.page_link("pages/setup_cost.py", label="What setup actually costs")
    st.markdown(
        '<div class="ngec-meta">Disk, hours, and hardware for the Wikipedia index and the '
        "models, measured rather than estimated.</div>",
        unsafe_allow_html=True,
    )

rule()

note(
    "The event classifiers in this demo are <strong>not</strong> the models behind POLECAT. "
    "They are demonstration models trained on Voice of America articles labelled by an LLM "
    "applying the PLOVER codebook, and they are noticeably worse — macro F1 around 0.52 "
    "across sixteen types. That is enough to show the pipeline working and not enough to "
    "build a dataset on. The attribute, entity-linking and resolution models are the real "
    "ones described in the paper.",
    title="About the models you are about to see",
    tone="warn",
)

paper.ref("framework", "customizing", label="Read alongside")
