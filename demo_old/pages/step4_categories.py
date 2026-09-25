"""Step 4: assigning ontology categories to actors."""

import html

import streamlit as st

from ngec_demo import paper, pipeline, ui
from ngec_demo.theme import kv_table, note, rule

ui.setup()
ui.step_rail("step4")

ui.page_header(
    kicker="Step 4 of 5",
    title="Entity categorisation",
    standfirst=(
        "“Riot police”, “internal security forces” and “the gendarmerie” should all land in "
        "the same category. This step maps a mention — named or generic — onto the "
        "researcher's own ontology."
    ),
    paper_keys=("step4",),
)

st.markdown(
    """
There is no model trained on the categories here. The ontology is a **plain text file** of
role descriptions with codes attached, and a mention is matched to it by encoding both with a
sentence transformer and taking the nearest pattern above a similarity threshold. Adding a
category means adding lines to that file. Nothing is retrained, and no embeddings of the
existing patterns are recomputed except for the lines you added.

This is the cheapest customisation in the paper's Table 1, and the one claim there the demo
can demonstrate literally — see [your actor categories](your_actors).
"""
)

patterns = pipeline.agent_patterns()
if patterns:
    codes = {}
    for pattern in patterns:
        codes.setdefault(pattern.get("code_1", "?"), 0)
        codes[pattern.get("code_1", "?")] += 1
    top = sorted(codes.items(), key=lambda kv: -kv[1])
    st.markdown(
        f'<div class="ngec-meta"><strong>{len(patterns):,} patterns</strong> across '
        f"<strong>{len(codes)}</strong> primary codes, loaded from "
        f"<code>ngec/assets/PLOVER_agents.txt</code>. Most common: "
        f'{", ".join(f"{c} ({n})" for c, n in top[:10])}.</div>',
        unsafe_allow_html=True,
    )

rule()

st.markdown("## Match a mention")

mention = st.text_input("Actor mention", value="riot police")

if mention.strip():
    if pipeline.resources.get_actor_resolver() is None:
        note(
            "This panel needs the actor resolver, which could not be loaded on this server.",
            title="Unavailable",
            tone="bad",
        )
    else:
        with st.spinner("Encoding and matching…"):
            ranked = pipeline.match_against_patterns(mention, top_k=8)
            coded = pipeline.agent_match(mention)

        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("### Nearest patterns")
            rows = []
            for r in ranked:
                bar_width = max(0.0, min(1.0, r["similarity"])) * 100
                code = r["code_1"] + (f'·{r["code_2"]}' if r["code_2"] else "")
                rows.append(
                    f'<div class="ngec-bar-row on">'
                    f'<span class="ngec-bar-name">{html.escape(code)}</span>'
                    f'<span class="ngec-bar-track">'
                    f'<span class="ngec-bar-fill" style="width:{bar_width:.1f}%"></span></span>'
                    f'<span class="ngec-bar-val">{r["similarity"]:.2f}</span>'
                    f"</div>"
                    f'<div class="ngec-meta" style="margin:-0.1rem 0 0.35rem 7.8rem">'
                    f'{html.escape(r["pattern"])}</div>'
                )
            st.markdown("".join(rows), unsafe_allow_html=True)

        with right:
            st.markdown("### The code assigned")
            if coded and not coded.get("error"):
                st.markdown(
                    kv_table([
                        ("Code", f'<strong>{coded.get("code_1", "")}</strong>'),
                        ("Secondary", coded.get("code_2", "")),
                        ("Matched pattern", coded.get("pattern", "")),
                        ("Country", coded.get("country", "")),
                        ("Confidence", f'{coded.get("conf", 0):.3f}'
                            if coded.get("conf") is not None else ""),
                        ("How", coded.get("source", "")),
                    ]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ngec-meta">No pattern cleared the similarity threshold, so '
                    "the mention goes uncoded rather than being forced into the nearest "
                    "category.</div>",
                    unsafe_allow_html=True,
                )

note(
    "Try “gendarmerie”, “the lads from the estate”, “Ministry of Agriculture”, or something "
    "from your own domain that PLOVER has no category for. The similarity scores are the "
    "useful part: a mention that matches its best pattern at 0.62 is being coded on a "
    "resemblance the researcher may not endorse. This is a semantic nearest-neighbour lookup, "
    "not a classifier that knows the ontology.",
    title="Where the soft matching shows its seams",
)

rule()

st.markdown("## In the pipeline")

# Like step 3, this step is handed mentions rather than prose.
text, pub_date, example = ui.example_picker("step4", show_source=False)
ui.what_to_notice(example)

result = ui.run_with_progress(text, pub_date, stop_after="actors",
                              label="Running through categorisation")

if not ui.run_error(result):
    ui.stage_input(
        ui.mention_rows(result.records_after("attributes")),
        explain="A mention reaches this step only if step 3 found no Wikipedia page for it, "
                "or found one that carries no code.",
    )
    ui.render_records(result.records_after("actors"))

    st.markdown(
        '<div class="ngec-meta" style="margin-top:0.8rem">'
        "Named entities are coded from their Wikipedia page where one was found (step 3) and "
        "from the pattern file otherwise. The <code>source</code> field on each coded actor "
        "records which route was taken — “Infobox” for a Wikipedia infobox, “similarity” or "
        "“BERT matching full text” for the pattern file.</div>",
        unsafe_allow_html=True,
    )

paper.ref("step4", "customizing", "ecav_actors")
