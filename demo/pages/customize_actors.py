"""Customising the actor category ontology — the cheapest change in the paper's Table 1."""

import html
import re

import streamlit as st

from ngec_demo import paper, pipeline, ui
from ngec_demo.theme import PALETTE, kv_table, note, rule

ui.setup()

ui.page_header(
    kicker="Make it yours",
    title="Your actor categories",
    standfirst=(
        "PLOVER's actor codes will not fit every project. Changing them means editing a text "
        "file — no training, no labelled data, no GPU. This page lets you do it and see the "
        "effect immediately."
    ),
    paper_keys=("step4", "customizing"),
)

st.markdown(
    """
The ontology lives in `ngec/assets/PLOVER_agents.txt`, one pattern per line with its code in
square brackets:

```
riot police [COP]
internal security forces [COP]
prime minister [GOV]
opposition party [OPP]
```

A mention is matched to this list by encoding both with a sentence transformer and taking
the nearest pattern above a similarity threshold. Because the match is semantic rather than
exact, a handful of patterns per category covers a lot of surface forms — you are writing
examples, not a dictionary.

**Codes are three letters.** `AgentMatcher._load_and_clean_agents` splits the bracketed
string at position three: the first block is the category (`code_1`) and anything after it is
a modifier (`code_2`), which is how compounds like `[GOVMIL]` — a defence ministry, both
government and military — and `[CVLOPP]` — civilians identified with the opposition — work.
A four-letter code is therefore not a new category; it is a three-letter category plus a
one-letter modifier nobody defined. The editor below flags that rather than silently
splitting it.
"""
)

patterns = pipeline.agent_patterns()
if not patterns:
    note(
        "The actor resolver could not be loaded on this server, so the live matching below "
        "is unavailable. The rest of the page still describes the mechanism.",
        title="Resolver unavailable",
        tone="bad",
    )

rule()

st.markdown("## Four actor categories PLOVER does not distinguish")

st.markdown(
    """
The editor below is preloaded with a worked example rather than a placeholder, because the
argument for a replaceable actor ontology is only as good as the cases that motivate it.
These four are not gaps in the sense of "PLOVER forgot them". PLOVER codes each of them
already — into a category that erases the distinction a researcher studying them is trying to
measure. That is the failure mode worth demonstrating, and it is invisible unless you look at
what the nearest pattern actually is.

Every "coded today as" below is what the live matcher returns for that mention on this
server, not an illustration.
"""
)

st.markdown(
    """
| New code | Actors | Coded today as | Why the difference matters |
|---|---|---|---|
| `CHF` | Chiefs, elders, headmen — customary authority | *a traditional leader* → `GOV`, from *supreme leader* (0.76); *the village elders* → `CVL`, from *elders* (0.85) | A chief becomes either the state or an ordinary civilian. Both erase what makes customary authority a distinct object: power that is neither elected nor delegated by the centre. Work on traditional leaders and local public-goods provision (Baldwin 2016) needs them separable from government. |
| `VIG` | Vigilante and community self-defence groups | *a vigilante group* → `UAF`, from *violent group* (0.77); *the neighbourhood self-defence militia* → `PRM`, from *self defense militia* (0.88) | `UAF` means "we could not identify them"; `PRM` is for private armies and security contractors. Vigilantism is defined against both — locally known people claiming to enforce order *without* state sanction and without being paid for it (Bateson 2021). |
| `EMB` | Election management bodies | *the electoral commission* → `GOV`, from *electoral commission* (0.98) | An EMB's independence *from* the executive is the variable in most electoral-integrity research. Coding it as government assumes the answer, and does so at a similarity score that looks like a correct match. |
| `DSP` | Diaspora organisations | *a diaspora association* → `NON`, from *international community* (0.71) | `NON` is the bucket for strings that are not actors at all — days of the week, "power", "the border". The record survives, with its country blanked by `actor_resolution.py`, carrying a code that tells every downstream user to ignore it. For transnational politics that is worse than a wrong category: it is an instruction to discard the case. |
"""
)

note(
    "These four share a frame, which is the point: each is an actor whose authority or "
    "identity is defined by its relationship to the state, and PLOVER's categories are built "
    "around the state itself. That is a defensible design for a general-purpose ontology and "
    "a poor one for a project about hybrid governance, contested elections, or transnational "
    "politics. Replace what does not fit; that is the argument.",
    title="Why these four and not any other four",
)

st.markdown("## Add them, and see what changes")

st.markdown(
    "Write one pattern per line in the same `pattern [CODE]` format. These are encoded when "
    "you run the match — the built-in patterns keep their cached embeddings, so adding a "
    "category costs a single encode rather than a re-encode of several thousand strings."
)

col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    custom_text = st.text_area(
        "Your patterns",
        value=(
            "# Customary authority — neither the state nor ordinary civilians\n"
            "traditional leader [CHF]\n"
            "village elder [CHF]\n"
            "paramount chief [CHF]\n"
            "# Order-keeping without state sanction\n"
            "vigilante group [VIG]\n"
            "community self-defence group [VIG]\n"
            "# Election administration, independent of the executive\n"
            "independent electoral commission [EMB]\n"
            "election management body [EMB]\n"
            "# Politics conducted from outside the country\n"
            "diaspora association [DSP]\n"
            "emigrant community organisation [DSP]"
        ),
        height=270,
    )

with col_r:
    mention = st.text_input("Actor mention to code", value="a traditional leader")
    st.markdown(
        '<div class="ngec-meta">Try mentions your own project cares about. The interesting '
        "cases are the ones PLOVER has no good category for — those are exactly the ones "
        "worth adding patterns for.</div>",
        unsafe_allow_html=True,
    )


# A code is one or more three-letter blocks: the first is the category, the rest
# are modifiers. This mirrors AgentMatcher._load_and_clean_agents, which slices
# at position three unconditionally — so a four-letter code there does not fail,
# it quietly becomes a three-letter category plus a one-letter modifier. Catching
# it here is the whole reason this returns problems alongside patterns.
CODE = re.compile(r"^(?:[A-Za-z]{3})+$")
LINE = re.compile(r"^(.*?)\s*\[([A-Za-z0-9]*)\]\s*$")


def parse_patterns(raw: str) -> tuple[list[dict], list[str]]:
    """`pattern [CODE]` lines → (patterns for the matcher, complaints to show)."""
    out: list[dict] = []
    problems: list[str] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            problems.append(f"line {number}: no code in square brackets — “{line}”")
            continue
        text, code = match.group(1).strip(), match.group(2).strip()
        if not text:
            problems.append(f"line {number}: a code with no pattern in front of it")
            continue
        if not CODE.match(code):
            problems.append(
                f"line {number}: “{code}” is {len(code)} letters. A code is three letters, "
                f"or several three-letter blocks for a compound like GOVMIL"
            )
            continue
        out.append({"pattern": text.lower(), "code_1": code[:3].upper(),
                    "code_2": code[3:].upper()})
    return out, problems


custom, problems = parse_patterns(custom_text)

if problems:
    items = "".join(f"<li>{html.escape(p)}</li>" for p in problems)
    st.markdown(
        f'<div class="ngec-note warn"><span class="ngec-note-title">'
        f'{len(problems)} line{"s" if len(problems) > 1 else ""} not used</span>'
        f"<ul style='margin:0.4rem 0 0 1.1rem;padding:0'>{items}</ul></div>",
        unsafe_allow_html=True,
    )

if custom:
    st.markdown(
        f'<div class="ngec-meta">Parsed <strong>{len(custom)}</strong> pattern(s) into '
        f'<strong>{len(set(c["code_1"] for c in custom))}</strong> code(s).</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="ngec-meta">No patterns parsed — each line needs a pattern and a '
        "three-letter code in square brackets.</div>",
        unsafe_allow_html=True,
    )

if mention.strip() and patterns:
    with st.spinner("Encoding and matching…"):
        before = pipeline.match_against_patterns(mention, extra=None, top_k=5)
        after = pipeline.match_against_patterns(mention, extra=custom or None, top_k=5)

    st.markdown("### The same mention, before and after")

    def render(rows, title):
        st.markdown(f"**{title}**")
        body = []
        for r in rows:
            code = r["code_1"] + (f'·{r["code_2"]}' if r["code_2"] else "")
            is_yours = r["source"] == "yours"
            colour = PALETTE["accent"] if is_yours else PALETTE["ink_faint"]
            tag = (
                f' <span style="color:{PALETTE["accent"]};font-weight:650">yours</span>'
                if is_yours else ""
            )
            body.append(
                f'<div class="ngec-bar-row {"on" if is_yours else "off"}">'
                f'<span class="ngec-bar-name" style="color:{colour}">'
                f"{html.escape(code)}</span>"
                f'<span class="ngec-bar-track"><span class="ngec-bar-fill" '
                f'style="width:{max(0, min(1, r["similarity"])) * 100:.1f}%"></span></span>'
                f'<span class="ngec-bar-val">{r["similarity"]:.2f}</span></div>'
                f'<div class="ngec-meta" style="margin:-0.1rem 0 0.35rem 7.8rem">'
                f'{html.escape(r["pattern"])}{tag}</div>'
            )
        st.markdown("".join(body), unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        render(before, "PLOVER only")
    with col_b:
        render(after, "With your patterns")

    top_before = before[0] if before else None
    top_after = after[0] if after else None
    if top_before and top_after and top_before["pattern"] != top_after["pattern"]:
        st.markdown(
            f'<div class="ngec-note good"><span class="ngec-note-title">The code changed'
            f"</span>“{html.escape(mention)}” was coded "
            f'<code>{html.escape(top_before["code_1"])}</code> from '
            f'“{html.escape(top_before["pattern"])}” and is now coded '
            f'<code>{html.escape(top_after["code_1"])}</code> from '
            f'“{html.escape(top_after["pattern"])}”. No model was retrained.</div>',
            unsafe_allow_html=True,
        )
    elif top_after:
        st.markdown(
            '<div class="ngec-meta">Your patterns did not outrank the built-in ones for this '
            "mention. That is informative: either the existing category already fits, or the "
            "pattern needs to be worded closer to how the mention actually appears in "
            "text.</div>",
            unsafe_allow_html=True,
        )

    note(
        "Type <em>the electoral commission</em>, without “independent”. PLOVER's file contains "
        "that exact phrase, coded <code>GOV</code>, so it matches at 0.98 and your "
        "<code>EMB</code> pattern cannot beat it — adding a category does not displace a "
        "built-in pattern that says the same words. Overriding a phrase PLOVER already lists "
        "means editing its line, not adding yours, which is why the real customisation is a "
        "copy of the file rather than a supplement to it. The same mention with “independent” "
        "in front of it does flip, which tells you how narrow the escape is.",
        title="Adding a category does not override a phrase already in the file",
        tone="warn",
    )

rule()

st.markdown("## Using this for real")

st.markdown(
    """
The panel above matches against your patterns in memory. To use a custom ontology in an
actual run, put the patterns in a file and point the resolver at it:

```python
from ngec import ActorResolver

resolver = ActorResolver(
    spacy_model=nlp,
    es_client=es,
    # AgentMatcher takes an agents_file; pass your own copy of the
    # PLOVER_agents.txt format.
)
```

The pattern file is read by `AgentMatcher._load_and_clean_agents`, which strips comments,
parses the bracketed code into `code_1` (first three characters) and `code_2` (the rest), and
expands two placeholders: `!minist!` becomes Minister / Ministers / Ministry / Ministries,
and `!person!` becomes person / man / woman / men / women. Embeddings for the whole file are
computed once and cached, so the first run after an edit is slower than the rest.
"""
)

st.download_button(
    "Download your patterns as a file",
    data="\n".join(
        f'{c["pattern"]} [{c["code_1"]}{c["code_2"]}]' for c in custom
    ) or "# add patterns above",
    file_name="my_agents.txt",
    mime="text/plain",
)

rule()

st.markdown("## What this does and does not buy you")

st.markdown(
    kv_table([
        ("Cost to change", "Editing a text file. No labelled data, no training, no GPU."),
        ("Takes effect", "On the next run, after one embedding pass over the file."),
        ("What it covers",
         "Generic mentions — “riot police”, “farmers”, “the village elders” — and named "
         "entities whose Wikipedia page yields a role description that then matches a "
         "pattern."),
        ("What it does not cover",
         "Named entities coded straight from a Wikipedia infobox. Those take their code from "
         "the infobox route in step 3, not from this file. Nor does it override a phrase "
         "PLOVER already lists, as the electoral-commission case above shows."),
        ("The honest caveat",
         "This is a semantic nearest-neighbour lookup, not a classifier that understands your "
         "ontology. A mention matching its best pattern at 0.55 is being coded on a "
         "resemblance you may not endorse — and a high score is no defence either, since "
         "<em>electoral commission</em> is coded <code>GOV</code> at 0.98. Read the pattern "
         "that won, not just the code and the number."),
    ]),
    unsafe_allow_html=True,
)

note(
    "The paper's Table 1 lists this as a low-effort customisation, and at the level of "
    "<em>this step</em> that is accurate — it really is a text file. What the table does not "
    "convey is that you must first install the package, stand up Elasticsearch, and index "
    "Wikipedia before any of it runs. That total is on "
    "<a href='setup'>the setup page</a>, itemised.",
    title="Cheap step, expensive prerequisite",
    tone="warn",
)

paper.ref("step4", "customizing")
