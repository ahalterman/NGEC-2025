"""Temporal echoes: historical events reported in the grammar of current ones."""

import html
import re
from datetime import datetime

import streamlit as st

from ngec_demo import paper, ui
from ngec_demo.theme import PALETTE, kv_table, note, rule

ui.setup()

ui.page_header(
    kicker="Hard cases",
    title="Temporal echoes",
    standfirst=(
        "News routinely reports old events in the present tense: anniversaries, "
        "retrospectives, campaign summaries. They are grammatically almost identical to "
        "reporting on something that happened yesterday, and the pipeline does not currently "
        "tell them apart."
    ),
    paper_keys=("step5", "limitations"),
)

st.markdown(
    """
A commemoration piece contains a real ASSAULT event
in the text — the massacre being commemorated — and a pipeline that extracts it produces an
event record dated to roughly the publication date. Do that across a corpus and derived event
counts pick up a systematic, non-random bias: conflict-heavy places generate more
commemorative coverage, so they accumulate more phantom events, and the anniversaries cluster
on the same calendar days every year.

There are three distinguishable versions of the problem.
"""
)

cols = st.columns(3, gap="large")
for col, (title, body) in zip(cols, [
    ("Commemorative",
     "“On the fifth anniversary of the massacre, survivors gathered.” Stylistically marked — "
     "there is distinctive vocabulary to learn."),
    ("Retrospective",
     "“The government has faced criticism since 2019, when security forces opened fire.” "
     "Ordinary declarative prose. Only the date gives it away."),
    ("Campaign summary",
     "“The offensive has killed hundreds since March.” A real, current, ongoing process "
     "reported as a single aggregate — arguably not an echo at all, but it codes like one."),
]):
    with col:
        st.markdown(f"**{title}**")
        st.markdown(f'<div class="ngec-meta">{body}</div>', unsafe_allow_html=True)

rule()

st.markdown("## What the pipeline does today")

text, pub_date, example = ui.example_picker("echoes")
ui.what_to_notice(example)

result = ui.run_with_progress(text, pub_date, label="Running the pipeline")

if not ui.run_error(result):
    ui.render_records(result.final, text)

    st.markdown(
        '<div class="ngec-meta" style="margin-top:0.8rem">'
        "Read the dates on those records against the publication date. The date resolver is "
        "doing what it was designed to do — it refuses spans like “the fifth anniversary of” "
        "because they need world knowledge, and the caller stamps the publication date "
        "flagged <code>unresolved</code>. Nothing is malfunctioning. But the resulting record "
        "asserts that an event happened around the publication date, and that assertion is "
        "wrong.</div>",
        unsafe_allow_html=True,
    )

rule()

st.markdown("## Two diagnostics that would catch most of it")

note(
    "Neither of the checks below is part of the pipeline. They are computed here, on this "
    "page, from output the pipeline already produces, to show that the signal is available "
    "and where each intervention would sit. Nothing on the rest of this site applies them.",
    title="Not currently implemented",
    tone="warn",
)

# --- diagnostic 1: publication-date lag -----------------------------------

st.markdown("### 1 · Publication-date lag  ·  a filter at step 5")

st.markdown(
    "The date resolver already compares each span against the publication date. An event "
    "whose resolved date falls well before publication is either a genuine late report or a "
    "retrospective reference, and for most research designs both are worth flagging."
)

lag_days = st.slider("Flag events resolved more than N days before publication",
                     min_value=1, max_value=365, value=30, step=1)

if not result.error and result.final:
    rows = []
    for record in result.final:
        resolved = record.get("date_resolved") or {}
        raw = resolved.get("resolved_date")
        date_type = resolved.get("date_type") or ""
        try:
            resolved_dt = datetime.fromisoformat(str(raw)[:19])
            pub_dt = datetime.fromisoformat(str(pub_date)[:19])
            lag = (pub_dt - resolved_dt).days
        except (TypeError, ValueError):
            lag = None

        if lag is None:
            verdict = '<span class="none">no comparable date</span>'
        elif date_type == "unresolved":
            verdict = (
                f'<span style="color:{PALETTE["warn"]}">unresolved — stamped with the '
                f"publication date, so the lag is 0 by construction and this filter "
                f"cannot see it</span>"
            )
        elif lag > lag_days:
            verdict = (
                f'<span style="color:{PALETTE["bad"]}"><strong>flagged</strong> — '
                f"{lag} days before publication</span>"
            )
        else:
            verdict = f"kept — {lag} days before publication"
        rows.append((str(record.get("event_type", "")), verdict))
    st.markdown(kv_table(rows), unsafe_allow_html=True)

note(
    "Note the failure mode this diagnostic has on the commemoration example: because the "
    "resolver <em>refuses</em> “the fifth anniversary of” and falls back to the publication "
    "date, the lag is zero and the filter sees nothing. The lag filter catches the "
    "<em>retrospective</em> case, where a real date is present and resolves to 2019. The two "
    "echo types need different interventions, which is the argument for doing both.",
    title="Why one filter is not enough",
)

# --- diagnostic 2: commemorative language ---------------------------------

st.markdown("### 2 · Commemorative language  ·  a label at step 1")

st.markdown(
    "Commemorative writing is stylistically marked in a way retrospective writing is not. "
    "Below is a deliberately crude lexical probe — not a model, just a regular expression over "
    "the document — to show that the signal is on the surface of the text. A mode-style "
    "classifier trained on a few hundred anniversary stories would do considerably better, "
    "and would slot into the existing second-stage classifier infrastructure without touching "
    "the attribute model or the entity linker."
)

ECHO_CUES = [
    r"\banniversar(y|ies)\b",
    r"\b(five|ten|twenty|thirty|forty|fifty|\d+)\s+years?\s+(ago|since|after|on)\b",
    r"\bin memory of\b",
    r"\bcommemorat\w*",
    r"\bmark(ed|s|ing)?\s+the\s+\w+\s+(anniversary|year)",
    r"\bmemorial\b",
    r"\bremembrance\b",
    r"\blaid wreaths?\b",
    r"\bsurvivors?\b",
    r"\bmass graves?\b",
]

hits = []
for pattern in ECHO_CUES:
    for match in re.finditer(pattern, text, re.IGNORECASE):
        hits.append((match.group(0), match.start()))

if hits:
    seen, spans = set(), []
    for phrase, start in sorted(hits, key=lambda h: h[1]):
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            spans.append(phrase)
    st.markdown(
        f'<div class="ngec-note warn"><span class="ngec-note-title">'
        f"{len(hits)} commemorative cue(s) found</span>"
        f'{", ".join(f"<em>{html.escape(s)}</em>" for s in spans)}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="ngec-meta">No commemorative cues in this document.</div>',
        unsafe_allow_html=True,
    )

rule()

st.markdown("## What it would take")

st.markdown(
    """
Both interventions fit the existing architecture, and neither requires retraining the
attribute model or touching entity linking.

**At step 1.** The second-stage mode classifier already trains a per-(type, mode) model on
documents where the parent type fired. A temporal-status label — contemporaneous versus
historical reference — is the same shape of problem and would be fit the same way. The
training data is the harder part: the synthetic story generator would need an anniversary or
retrospective scenario added to its `event_placement` prompt so that echo documents exist in
the training set as a distinct category.

**At step 5.** The lag filter above is a handful of lines against a field the pipeline already
emits. The design question is not how to compute it but what to do with it: dropping flagged
events silently would repeat the mistake the pipeline avoids elsewhere, so the right output is
a flag on the record that a researcher chooses to filter on.

**What neither fixes.** A commemoration whose date span the resolver refuses gets stamped with
the publication date, so it is invisible to the lag filter and depends entirely on the step 1
label. And a document can be *both* — a story about an anniversary protest contains a real
current PROTEST and a historical ASSAULT, and the correct output keeps one and drops the
other. That is a per-event judgement, not a per-document one, which means the label belongs
on the extracted event rather than on the story.
"""
)

paper.ref("step1", "step5", "limitations")
