"""Step 1: event detection as document classification."""

import streamlit as st

from ngec_demo import paper, pipeline, ui
from ngec_demo.theme import bar_rows, note, rule

ui.setup()
ui.step_rail("step1")

ui.page_header(
    kicker="Step 1 of 5",
    title="Event detection",
    standfirst=(
        "Which event types does this document report? The pipeline treats this as document "
        "classification — one binary classifier per event type, each with its own decision "
        "threshold — rather than as a parsing or pattern-matching problem."
    ),
    paper_keys=("step1",),
)

st.markdown(
    """
Posing detection this way is what makes the step cheap to customise. Annotating documents
for one event type at a time is something a political scientist can plan and cost; writing a
verb dictionary is not. It also means the standard toolkit applies — active learning,
synthetic augmentation, or a fine-tuned encoder — without touching the rest of the pipeline.
"""
)

text, pub_date, example = ui.example_picker("step1")
ui.what_to_notice(example)

result = ui.run_with_progress(text, pub_date, stop_after="classify",
                              label="Classifying")

rule()

st.markdown("## Every classifier's score")
st.markdown(
    "Each of the sixteen event types gets its own model and its own threshold, shown as the "
    "vertical tick. Bars that cleared their threshold are in colour. `process()` only reports "
    "the ones that fired; the near misses are here because they are usually the interesting part."
)

scores = pipeline.classify_scores(text)
st.markdown(bar_rows(scores), unsafe_allow_html=True)

fired = [r for r in scores if r["fired"]]
near = [r for r in scores if not r["fired"] and r["margin"] > -0.15]

col1, col2 = st.columns(2, gap="large")
with col1:
    if fired:
        st.markdown(
            f'<div class="ngec-meta"><strong>Fired:</strong> '
            f'{", ".join(r["event_type"] for r in fired)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ngec-meta"><strong>Nothing fired.</strong> The pipeline stops here — '
            "with no event type there is nothing to extract attributes for.</div>",
            unsafe_allow_html=True,
        )
with col2:
    if near:
        st.markdown(
            f'<div class="ngec-meta"><strong>Within 0.15 of firing:</strong> '
            f'{", ".join(r["event_type"] for r in near)}</div>',
            unsafe_allow_html=True,
        )

note(
    "The thresholds are not a single number you pass in. Each is the value that maximised that "
    "class's F1 at training time, recorded in the model directory's <code>metadata.json</code> "
    "and loaded with the models. ACCUSE and CONCEDE have very different base rates and very "
    "different thresholds, and collapsing them to one global cut-off — which passing "
    "<code>threshold=0.9</code> would do — throws that away.",
    title="Where the thresholds come from",
)

rule()

st.markdown("## Modes: the second stage")
st.markdown(
    "A mode model is fit only on documents where its parent type is present, and applied the "
    "same way. This is the machinery the paper points at for sub-types, and it is also the "
    "obvious place to hang a temporal-status label — see "
    "[the temporal echoes page](echoes)."
)

if fired:
    tabs = st.tabs([r["event_type"] for r in fired])
    for tab, row in zip(tabs, fired):
        with tab:
            modes = pipeline.mode_scores(text, row["event_type"])
            if not modes:
                st.markdown(
                    '<div class="ngec-meta">No mode models were fit for this type.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    bar_rows(modes, name_key="mode"),
                    unsafe_allow_html=True,
                )
else:
    st.markdown(
        '<div class="ngec-meta">No event type fired, so no mode models ran.</div>',
        unsafe_allow_html=True,
    )

rule()

st.markdown("## How good are these classifiers?")

meta = pipeline.model_metadata()
metrics = meta.get("metrics", {})

if metrics:
    rows = []
    for name in sorted(metrics, key=lambda n: -metrics[n].get("f1", 0)):
        m = metrics[name]
        rows.append({
            "event_type": name,
            "probability": m.get("f1", 0.0),
            "threshold": None,
            "fired": m.get("f1", 0) >= 0.6,
        })
    st.markdown(
        "F1 on a stratified split of the annotated data, from `metadata.json`. Bars at or "
        "above 0.6 are coloured only to make the spread readable."
    )
    st.markdown(bar_rows(rows, threshold_key=None), unsafe_allow_html=True)

note(
    "These are demonstration models, not the ones behind POLECAT. They were trained on about "
    "2,000 Voice of America articles labelled by an LLM applying the PLOVER codebook, with one "
    "annotator and no adjudication. On a never-seeded random holdout they reach a macro F1 of "
    "0.52 across the sixteen types — better than the synthetic-text models they replaced "
    "(0.39), and well short of production quality. The numbers above come from the annotated "
    "split, which was enriched for positives on purpose, so they read higher than the holdout. "
    "Treat them as evidence that the step works, not as an accuracy claim.",
    title="An honest reading of these numbers",
    tone="warn",
)

with st.expander("The model directory's metadata.json", expanded=False):
    st.markdown(
        '<div class="ngec-meta">This is what makes the models self-describing: the encoder '
        "name, the per-class thresholds, the training metrics, and the screening error rates "
        "used to weight the implied negatives all travel with the models.</div>",
        unsafe_allow_html=True,
    )
    st.json(meta, expanded=False)

paper.ref("step1", "customizing")
