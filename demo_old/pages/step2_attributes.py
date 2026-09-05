"""Step 2: pulling actor, recipient, date and location spans out of the document."""

import html

import streamlit as st
from ngec_demo import event_types, paper, pipeline, ui
from ngec_demo.theme import note, rule

ui.setup()
ui.step_rail("step2")

ui.page_header(
    kicker="Step 2 of 5",
    title="Attribute extraction",
    standfirst=(
        "An event is four things: who did it, to whom, when, and where. "
        "This step is a 0.6B-parameter fine-tuned model that reads the whole document and "
        "copies out the spans that fill those slots."
    ),
    paper_keys=("step2", "attr_model"),
)

st.markdown(
    """
Two design choices matter here. The model reads the **whole document**, not a sentence, so an
actor named in the lede can fill a slot for an event described four paragraphs down. And it
extracts **per event type** — the same document classified as both PROTEST and ASSAULT is
asked about each separately, because the actor of the protest is usually not the actor of the
assault.

It is also small on purpose. A 0.6B model that runs on a laptop and can be re-fine-tuned on
new attributes is a different proposition from an API call per document, both for cost and
for reproducibility.
"""
)

ui.attribute_model_provenance()

# This step reads the whole document, so the whole document is its input.
text, pub_date, example = ui.example_picker("step2")
ui.what_to_notice(example)

result = ui.run_with_progress(text, pub_date, stop_after="attributes",
                              label="Classifying and extracting")

if not ui.run_error(result):
    records = result.records_after("attributes")

    rule()

    st.markdown("## What the model found")
    st.markdown(
        f'<div class="ngec-meta">{result.step("split").note} · '
        f"{result.step('attributes').note}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    for i, record in enumerate(records):
        if len(records) > 1:
            st.markdown(
                f'<div class="ngec-kicker" style="margin-top:1.3rem">'
                f"Extracted event {i + 1} of {len(records)}"
                f"{ui.record_label(record)}</div>",
                unsafe_allow_html=True,
            )
        ui.render_attributes(record, text)
        st.write("")

    ui.repeated_modes_note(records)

    rule()

    ui.dropped_events(result)

    st.markdown("## One record per event")
    st.markdown(
        """
The model may find no events in a document, one, or several. `explode_events` turns each
extracted event into its own record with a single `attributes` dict, so that by the time
actor resolution runs, one record means one event.

Records with **zero** extractions are dropped rather than emitted with empty attributes —
people are bad at filtering pipeline output, so empty junk does not go in it. Rather than
vanishing into a log, they are listed above.
"""
    )

    ui.raw_json("The attributes dict for each record", [r.get("attributes") for r in records])

rule()

# --------------------------------------------------------------------------
# PLOVER, and everything else
# --------------------------------------------------------------------------
#
# Outside the `if not ui.run_error(...)` block above on purpose. This section
# does not use the pipeline run at all — it prompts the attribute model directly
# — so it still works on a document where the classifier fired on nothing, which
# is exactly the document a visitor with their own ontology is likely to bring.

st.markdown("## PLOVER, and event types the model has never seen")

st.markdown(
    """
The label in the prompt is a handle. What the model is actually given is a **definition**,
and it was fine-tuned to read one — which is why a new ontology costs a step-1 classifier and
nothing else. That claim is easy to make and worth checking, so here is the model with a
definition you write, on text you choose, with no classifier and no codebook in between.
"""
)

PRESETS: dict[str, tuple[str, str]] = {
    f"Not in any ontology — {event_types.INVENTED_LABEL}":
        (event_types.INVENTED_LABEL, event_types.INVENTED_DEFINITION),
    **{f"ECAV, a different project's ontology — {label}": (label, definition)
       for label, definition in event_types.UNSEEN.items()},
    **{f"PLOVER, in the model's training data — {label}": (label, definition)
       for label, definition in event_types.plover().items()},
}

LABEL_KEY, DEF_KEY, PRESET_KEY = "_custom_label", "_custom_def", "_custom_preset"


def _load_preset() -> None:
    label, definition = PRESETS[st.session_state[PRESET_KEY]]
    st.session_state[LABEL_KEY] = label
    st.session_state[DEF_KEY] = definition


st.session_state.setdefault(LABEL_KEY, event_types.INVENTED_LABEL)
st.session_state.setdefault(DEF_KEY, event_types.INVENTED_DEFINITION)

st.selectbox("Start from", list(PRESETS), key=PRESET_KEY, on_change=_load_preset)

with st.form("custom_event_type"):
    st.text_input("Event type", key=LABEL_KEY)
    st.text_area("Definition", key=DEF_KEY, height=150,
                 help="What a human coder would be given. Anything after "
                      "'## Extraction Note:' tells the model how to read the four slots "
                      "for this type.")
    custom_text = st.text_area("Document", value=text, height=170,
                               help="Prefilled with the document above; overwrite it freely.")
    extract = st.form_submit_button("Extract with this definition", type="primary")

if extract:
    with st.spinner("Prompting the attribute model…"):
        st.session_state["_custom_extraction"] = (
            pipeline.extract_with_definition(
                custom_text, st.session_state[LABEL_KEY], st.session_state[DEF_KEY]),
            st.session_state[LABEL_KEY],
            custom_text,
        )

held = st.session_state.get("_custom_extraction")
if held is not None:
    extraction, used_label, used_text = held
    if extraction.error:
        st.error(extraction.error)
    else:
        # The label alone decides this. A visitor who reuses "PROTEST" for their
        # own concept still gets a label the model saw thousands of times, and
        # the page should say so rather than credit the model with generalising.
        known = event_types.is_plover(used_label)
        note(
            f"<code>{html.escape(used_label.upper())}</code> "
            + ("is one of the sixteen PLOVER types, so this label and a definition very "
               "like it were in the model's training data. Extraction here is the case "
               "the model was built for, and it is the baseline the other option should "
               "be judged against."
               if known else
               "is not a PLOVER type. Neither this label nor this definition was in the "
               "model's training data; it is reading the definition and applying it.")
            + " Nothing else in the pipeline was told about it: no classifier was trained, "
              "no codebook row was added.",
            title="A PLOVER type" if known else "Not a PLOVER type",
        )
        if not extraction.records:
            st.markdown(
                '<div class="ngec-note"><span class="ngec-note-title">'
                "Nothing extracted</span>"
                "The model returned no event of this type in this document, and the record "
                "was dropped. That is a real answer, and on a definition the model has never "
                "seen it is the failure mode to watch for — but check the document actually "
                "contains such an event before reading it as one.</div>",
                unsafe_allow_html=True,
            )
        for i, record in enumerate(extraction.records):
            if len(extraction.records) > 1:
                st.markdown(
                    f'<div class="ngec-kicker" style="margin-top:1.1rem">'
                    f"Extracted event {i + 1} of {len(extraction.records)}</div>",
                    unsafe_allow_html=True,
                )
            ui.render_attributes(record, used_text)
        st.markdown(
            f'<div class="ngec-meta" style="margin-top:0.6rem">'
            f"{extraction.seconds:.1f}s, one call to the attribute model. No classifier, no "
            f"codebook lookup, no index — a new event type reaches this step as a string.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("The prompt the model was given", expanded=False):
            st.markdown(
                '<div class="ngec-meta">This is the whole briefing. Everything the model '
                "knows about your event type is in it — which is the reason a new type needs "
                "a definition written for it rather than a retrained model, and also the "
                "reason a vague definition extracts badly.</div>",
                unsafe_allow_html=True,
            )
            st.code(extraction.prompt, language="text")

note(
    "The evaluation behind this is ECAV, a hand-annotated armed-violence dataset with its own "
    "ontology. Thirteen of its fourteen event types are not PLOVER labels, and none of its "
    "definitions were in training; prompted with them, the model returned an event on "
    "<strong>98.6%</strong> of the documents whose annotators had recorded one. Two honest "
    "qualifications. That evaluation also passed the model the annotated passage to focus on, "
    "which this panel does not, so it is an easier setting than the one above. And ECAV's "
    "categories are still political-violence categories — near neighbours of PLOVER's, not an "
    "arbitrary ontology — so read the number as evidence that the model reads definitions "
    "rather than as a guarantee for any category you can write.",
    title="What this is measured on",
)

rule()

st.markdown("## Where this model is weak")

note(
    "The <code>anchor_quote</code> is meant to be a verbatim span from the article that "
    "justifies the extraction. It sometimes comes back as a paraphrase of the codebook's "
    "definition of the event type instead — “All rejections and refusals.” rather than a "
    "sentence from the story — and when that happens the rest of the extraction is usually "
    "thin, N/A in the actor and recipient slots. It happened twice in thirteen documents "
    "on a corpus run. It does <em>not</em> happen on any of the eleven curated examples "
    "here, with either the published model or the retrained one, so this page cannot show "
    "it to you on demand — which is exactly why every page checks each span against the "
    "document as it renders, and says so above the text when one does not appear there. "
    "Paste your own article and the check runs on that too.",
    title="Does the anchor quote actually come from the article?",
    tone="warn",
)

note(
    "Slots the model cannot fill come back as the string <code>N/A</code>, which this page "
    "treats as empty. A recipient of N/A is not the same as an event with no recipient — "
    "the model is saying it did not find one, which for many event types is the correct "
    "answer.",
    title="N/A is a real value",
)

note(
    "The four attributes are the schema. A mediator, a third party, a casualty count or a "
    "crowd size has nowhere to go. The paper's answer is that extra attributes mean "
    "extending the synthetic training data and re-fine-tuning — the one customisation on "
    "the list that genuinely requires touching a model.",
    title="Four slots, and only four",
)

with st.expander("Model card", expanded=False):
    st.markdown(
        """
| | |
|---|---|
| Model | `ahalt/event-attribute-extractor` |
| Base | Qwen3-0.6B |
| Training data | Synthetic news generated per event type, with attributes known by construction |
| Backends | vllm (Linux/CUDA), transformers (portable), mlx (macOS) |
| Sampling | temperature 0.5 — output, and therefore event `id` suffixes, vary between runs |

Because sampling is stochastic, re-running the same document can yield a different number of
events. `orig_id` is the stable key back to the source story; the `_0`, `_1` suffixes on
`id` are per-run artefacts.
"""
    )

paper.ref("step2", "attr_model", "eval_attr")
