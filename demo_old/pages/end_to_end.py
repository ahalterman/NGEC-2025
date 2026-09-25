"""The whole pipeline on one document."""

import json

import streamlit as st

from ngec_demo import paper, ui
from ngec_demo.theme import rule

ui.setup()

ui.page_header(
    kicker="See it work",
    title="A document becomes events",
    standfirst=(
        "One news story in; one structured record per event out. Each of the six steps runs "
        "in front of you, and its intermediate output is kept so you can see where any "
        "particular field came from."
    ),
    paper_keys=("framework",),
)

text, pub_date, example = ui.example_picker("end_to_end")
ui.what_to_notice(example)

result = ui.run_with_progress(text, pub_date)

if not ui.run_error(result):
    records = result.final

    st.markdown("## Coded events")
    st.markdown(
        f'<div class="ngec-meta">One story produced '
        f"<strong>{len(records)}</strong> event record"
        f'{"s" if len(records) != 1 else ""}.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    ui.render_records(records, text)

    ui.repeated_modes_note(records)

    rule()

    ui.dropped_events(result)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("### What each step cost")
        ui.timing_table(result)
        st.markdown(
            '<div class="ngec-meta" style="margin-top:0.7rem">'
            "Timings are for this host, one document at a time, with the models already "
            "loaded. Attribute extraction dominates because it is the only generative "
            "model in the pipeline; everything else is a classifier, an index lookup, or "
            "arithmetic on dates.</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Take it with you")
        st.download_button(
            "Download these records as JSON",
            data=json.dumps(records, indent=2, default=str),
            file_name="ngec_events.json",
            mime="application/json",
        )
        st.markdown(
            '<div class="ngec-meta" style="margin-top:0.6rem">'
            "This is exactly the shape <code>PloverCoder.process()</code> returns, and what "
            "gets written to <code>events_processed.jsonl</code> in a corpus run.</div>",
            unsafe_allow_html=True,
        )

    rule()

    st.markdown("### Intermediate output, step by step")
    st.markdown(
        "Each step mutates and passes along a list of dicts. Opening these in order shows "
        "the record growing: the classifier adds `event_type`, the attribute model replaces "
        "one record with one per extracted event, actor resolution adds a top-level `actor`, "
        "and the formatter adds `date_resolved` and `event_location`."
    )
    for step in result.steps:
        label = f"After {step.label.lower()} — {step.note}"
        ui.raw_json(label, step.records)

paper.ref("framework", "customizing")
