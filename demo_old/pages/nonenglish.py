"""Non-English text: which components transfer and which do not."""

import streamlit as st

from ngec_demo import examples as ex_mod
from ngec_demo import paper, ui
from ngec_demo.theme import kv_table, note, rule

ui.setup()

ui.page_header(
    kicker="Hard cases",
    title="Non-English text",
    standfirst=(
        "Every component of this pipeline is English-only. The paper says so in one sentence; "
        "this page shows what that actually means, component by component."
    ),
    paper_keys=("customizing", "limitations"),
)

st.markdown(
    """
The conflict literature increasingly draws on non-English sources, and one sentence in the
limitations section is not enough. The barriers differ a lot across the five steps, and
knowing which is which tells a researcher whether their project is a week of work or a year
of it.
"""
)

rule()

st.markdown("## Run a Spanish story through an English pipeline")

spanish = ex_mod.BY_KEY["spanish"]
text = st.text_area("Spanish text", value=spanish.text, height=150)
pub_date = st.text_input("Publication date", value=spanish.pub_date)

if st.button("Run it", type="primary"):
    st.session_state["_run_spanish"] = True

if st.session_state.get("_run_spanish"):
    result = ui.run_with_progress(text, pub_date, label="Running the pipeline")
    if not ui.run_error(result):
        ui.render_records(result.final, text)
    st.markdown(
        '<div class="ngec-meta" style="margin-top:0.7rem">'
        "Whatever came out, do not read it as “mostly working”. The sentence encoder behind "
        "the classifiers was trained on English; the attribute model was fine-tuned on English "
        "synthetic news; the Wikipedia index holds English articles. Anything that looks "
        "right here is the transfer you get for free from multilingual pretraining and from "
        "proper nouns being spelled the same way, not from the system supporting Spanish."
        "</div>",
        unsafe_allow_html=True,
    )

rule()

st.markdown("## Translate first, then code")

st.markdown(
    """
The pragmatic route is machine translation into English, then the pipeline unchanged. Below
is the same story translated, so you can compare. Translation is not run live here — this is
a fixed human translation, so the comparison isolates the pipeline's behaviour rather than
mixing in a translation model's.
"""
)

TRANSLATED = (
    "Hundreds of demonstrators gathered on Tuesday in front of the presidential palace in "
    "Bogota to protest against the tax reform. Riot police used tear gas to disperse the "
    "crowd. The president declared that the reform would go ahead. The unions announced a "
    "general strike for next week."
)

st.markdown(f'<div class="ngec-doc">{TRANSLATED}</div>', unsafe_allow_html=True)
st.write("")

if st.button("Run the translation"):
    st.session_state["_run_translated"] = True

if st.session_state.get("_run_translated"):
    result_t = ui.run_with_progress(TRANSLATED, spanish.pub_date,
                                    label="Running the translated text")
    if not ui.run_error(result_t):
        ui.render_records(result_t.final, TRANSLATED)

note(
    "Translation introduces its own validity problems, and they are not evenly distributed "
    "across the four attributes. Dates and locations survive translation well. Actor "
    "identification does not: honorifics, transliteration choices and organisation names are "
    "exactly what machine translation handles inconsistently, and “the president” in "
    "translated text is a role mention the coreference heuristic cannot expand — see "
    "<a href='coref'>coreference</a>. A translated corpus systematically loses more actor "
    "detail than an English one, which biases actor-level analyses rather than just adding "
    "noise.",
    title="What translation costs, and where",
    tone="warn",
)

rule()

st.markdown("## Component by component")

st.markdown(
    kv_table([
        ("Step 1 · Event detection",
         "<strong>Retrainable.</strong> Swap the sentence encoder for a multilingual one and "
         "annotate in the target language. This is the cheapest step to port — it is document "
         "classification, and the codebook-LLM annotation route works in any language the "
         "annotating model handles."),
        ("Step 2 · Attribute extraction",
         "<strong>Needs retraining.</strong> The synthetic training data generator would have "
         "to produce target-language news. The base model (Qwen3) is multilingual, so this is "
         "fine-tuning rather than starting over."),
        ("Step 3 · Entity linking",
         "<strong>The real bottleneck.</strong> The index is English Wikipedia. Other "
         "language editions exist and are smaller, but cross-lingual linking — an Arabic "
         "mention to an English page — is a different problem than the ranker was trained on. "
         "Wikidata IDs are the obvious bridge and are not currently used."),
        ("Step 4 · Entity categorisation",
         "<strong>Nearly free.</strong> The patterns are a text file and the matching is "
         "embedding similarity. A multilingual encoder plus a translated pattern file gets "
         "most of the way."),
        ("Step 5 · Dates and places",
         "<strong>Mixed.</strong> Geonames is multilingual and carries alternate names, so "
         "place resolution ports well. The date resolver is a cascade of English-language "
         "patterns and would need rewriting per language — a large, tedious, and entirely "
         "tractable job."),
    ]),
    unsafe_allow_html=True,
)

note(
    "The honest summary is that steps 1, 4 and half of 5 are a few weeks of work per "
    "language, step 2 is a fine-tuning run, and step 3 is a research project. A paper that "
    "presents this as a general-purpose toolkit should say that, rather than compressing it "
    "into one sentence.",
    title="The short version",
)

paper.ref("customizing", "limitations")
