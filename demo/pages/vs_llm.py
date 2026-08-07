"""Side by side against a one-shot frontier-LLM prompt."""

import html

import streamlit as st

from ngec_demo import llm_baseline, paper, ui
from ngec_demo.theme import PALETTE, kv_table, note, rule

ui.setup()

ui.page_header(
    kicker="Hard cases",
    title="Against a frontier LLM",
    standfirst=(
        "The most relevant competitor is not another event dataset. It is a researcher who "
        "writes one prompt, calls an API, and gets actor, recipient, date and location back. "
        "This page runs both on the same document."
    ),
    paper_keys=("llms",),
)

st.markdown(
    """
"Processing a corpus through a large LLM is expensive" is a weaker argument every year.
The comparison is worth making directly.

Three things separate the two approaches, and only the first is about accuracy:

1. **Extraction quality** — do the spans match what a human coder would pick? That is what
   the panel below and the ECAV evaluation test.
2. **Canonical resolution** — an extracted string is not an analytical unit. "Macron" has to
   become a Wikipedia page, a country, and a role code *as of the event date*; "Aleppo" has
   to become coordinates. That needs external knowledge stores, and it is most of what this
   pipeline is.
3. **Reproducibility and cost** — a closed model that changes under you is a problem for a
   dataset meant to be re-derivable. So is the bill on a million documents.

The honest version of point 3: this pipeline also uses a closed model to *generate* its
training data and to label the demonstration classifiers. What it ships is open weights and
a fixed index, so a re-run in three years reproduces; that is the claim, not that no
proprietary model was ever involved.
"""
)

rule()

st.markdown("## Run both on the same document")

text, pub_date, example = ui.example_picker("end_to_end")
ui.what_to_notice(example)

col_cfg, col_run = st.columns([2, 1], gap="large")
with col_cfg:
    effort = st.selectbox(
        "LLM effort level", ["low", "medium", "high"], index=1,
        help="Higher effort means more reasoning, more tokens, and more latency.",
    )
with col_run:
    st.write("")
    go = st.button("Run the comparison", type="primary")

if go:
    st.session_state["_ran_comparison"] = True

if not llm_baseline.available():
    note(
        "No API key is configured on this server, so the LLM side cannot run live. The "
        "NGEC side below still runs. To enable the live comparison, set "
        "<code>ANTHROPIC_API_KEY</code> in the environment the demo runs under.",
        title="Live comparison unavailable",
        tone="warn",
    )

if st.session_state.get("_ran_comparison"):
    ngec_result = ui.run_with_progress(text, pub_date, label="Running NGEC")

    llm_result = None
    if llm_baseline.available():
        with st.spinner("Prompting the LLM…"):
            llm_result = llm_baseline.cached_extract(text, pub_date, effort=effort)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### NGEC")
        if not ngec_result.error:
            ui.render_records(ngec_result.final, text)
        else:
            ui.run_error(ngec_result)

    with right:
        st.markdown("### One prompt to a frontier LLM")
        if llm_result is None:
            st.markdown(
                '<div class="ngec-meta">Not run — no API key configured.</div>',
                unsafe_allow_html=True,
            )
        elif llm_result.error:
            st.markdown(
                f'<div class="ngec-note bad">{html.escape(llm_result.error)}</div>',
                unsafe_allow_html=True,
            )
        elif not llm_result.events:
            st.markdown(
                '<div class="ngec-meta">The model reported no events.</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, event in enumerate(llm_result.events):
                if len(llm_result.events) > 1:
                    st.markdown(
                        f'<div class="ngec-kicker" style="margin-top:1.2rem">'
                        f"Event {i + 1} of {len(llm_result.events)}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    kv_table([
                        ("Event type", f'<strong>{html.escape(str(event.get("event_type") or ""))}</strong>'),
                        ("Mode", ""),
                        ("Actor", html.escape(str(event.get("actor") or ""))),
                        ("Recipient", html.escape(str(event.get("recipient") or ""))),
                        ("Location", html.escape(str(event.get("location") or ""))),
                        ("Date", html.escape(str(event.get("date") or ""))),
                        ("Anchor quote", html.escape(str(event.get("anchor_quote") or ""))),
                    ]),
                    unsafe_allow_html=True,
                )

    rule()

    st.markdown("### What each one cost")

    ngec_seconds = ngec_result.total_seconds
    rows = [
        ("Wall clock — NGEC", f"{ngec_seconds:.1f}s <span class='none'>on CPU, models "
                              f"already loaded</span>"),
    ]
    if llm_result and not llm_result.error:
        rows.append(("Wall clock — LLM", f"{llm_result.seconds:.1f}s"))
        rows.append(("Tokens — LLM", f"{llm_result.input_tokens:,} in / "
                                     f"{llm_result.output_tokens:,} out"))
        rows.append(("Cost — LLM, this document", f"${llm_result.cost_usd:.4f}"))
        rows.append(("Cost — LLM, 100,000 documents",
                     f"<strong>${llm_result.cost_usd * 100_000:,.0f}</strong>"))
        rows.append(("Model", html.escape(llm_result.model)))
    st.markdown(kv_table(rows), unsafe_allow_html=True)

    st.markdown(
        '<div class="ngec-meta" style="margin-top:0.6rem">'
        "The extrapolation is the honest way to read the cost column: the per-document "
        "figure looks trivial, and a corpus-scale project is where the difference lives. "
        "NGEC's marginal cost per document after setup is electricity — but the setup is "
        "real, and it is itemised on "
        "<a href='setup'>the setup page</a>.</div>",
        unsafe_allow_html=True,
    )

    rule()

    st.markdown("### Run it twice")
    st.markdown(
        "Reproducibility is the argument that does not depend on which system extracts "
        "better. Re-running the same document through the LLM gives a second sample from "
        "the same distribution; the classifiers and the entity linker are deterministic "
        "given fixed weights and a fixed index."
    )

    if llm_baseline.available() and st.button("Run the LLM a second time"):
        with st.spinner("Prompting again…"):
            second = llm_baseline.cached_extract(text, pub_date, effort=effort, nonce=1)

        if second.error:
            st.error(second.error)
        else:
            first_events = (llm_result.events if llm_result else [])
            same_count = len(first_events) == len(second.events)

            def signature(events):
                return [
                    (e.get("event_type"), e.get("actor"), e.get("recipient"),
                     e.get("date"), e.get("location"))
                    for e in events
                ]

            identical = signature(first_events) == signature(second.events)
            colour = PALETTE["good"] if identical else PALETTE["bad"]
            verdict = (
                "Identical to the first run."
                if identical
                else "Different from the first run."
            )
            st.markdown(
                f'<div class="ngec-note" style="border-left-color:{colour}">'
                f"<span class='ngec-note-title'>{verdict}</span>"
                f"Run 1 produced {len(first_events)} event(s); run 2 produced "
                f"{len(second.events)}."
                f"{'' if same_count else ' The event count itself changed.'}</div>",
                unsafe_allow_html=True,
            )
            ui.raw_json("Second run, raw", second.events)

    rule()

    st.markdown("### The part the prompt cannot do")
    st.markdown(
        """
Look at the actor rows in the two panels. The LLM returns a string — "President Emmanuel
Macron". NGEC returns a string *plus* a Wikipedia page, a country code, and a role code read
off that page's infobox **as of the event date**, so the same name codes differently for an
event in 2015 and an event in 2023.

You can ask an LLM to guess the country and role from parametric knowledge, and it will
often be right. What it cannot do without an external store is be consistent across a
corpus, be checkable against a source, or be correct about who held an office on a
particular day. That is the actual argument for the infrastructure, and it is narrower and
more defensible than "LLMs are expensive".
"""
    )

rule()

st.markdown("## The ECAV evaluation")

st.markdown(
    """
A single document is an anecdote. The cross-ontology validation against the Electoral
Contention and Violence dataset is the systematic version: ECAV was coded by different
people, to a different ontology, for a different purpose, so it tests whether the attribute
model generalises past what it was trained on.

Running the same evaluation with the prompt above, and reporting both F1 scores in the same
table, is the comparison the paper should carry. It is a bounded amount of work — the
evaluation harness already exists — and either outcome improves the paper: if NGEC wins, the
value-added argument stops being rhetorical; if it does not, the honest claim is about
resolution, reproducibility and cost rather than extraction accuracy, and the paper is
stronger for saying so.
"""
)

note(
    "This panel is a placeholder for those numbers. The demo deliberately does not show a "
    "made-up table here — when the LLM arm of the ECAV evaluation has been run, its results "
    "belong in this space alongside the existing NGEC scores.",
    title="Not yet run",
    tone="warn",
)

paper.ref("llms", "ecav", "ecav_attr", "ecav_actors")
