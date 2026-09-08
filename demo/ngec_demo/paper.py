"""Cross-references from demo pages to sections of the manuscript.

Every step page in this demo corresponds to a numbered section of "Creating
Custom Event Data: A Bag-of-Tricks". Keeping the mapping in one table means the
links can be re-pointed in one place when the manuscript is repaginated, and it
lets the response-to-reviewers cite a demo page and a paper section together.

Page numbers are PDF page numbers (1-indexed from the title page), used to build
`#page=N` fragments, which Chrome, Firefox, Safari and Acrobat all honour.
"""

import html
import os
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Section:
    number: str
    title: str
    page: int


# Extracted from the compiled manuscript. Update `page` if the PDF is recompiled
# with different pagination; `scripts/check_paper_sections.py` re-derives these.
SECTIONS: dict[str, Section] = {
    "intro": Section("1", "Introduction", 2),
    "existing": Section("2", "Existing Approaches to Automated Event Extraction", 3),
    "framework": Section("3", "A Five-Part Framework for Coding Event Data", 5),
    "llms": Section("3.1", "LLMs and the “Test of Time”", 6),
    "customizing": Section("4", "Customizing Event Data", 6),
    "step1": Section("4.1", "Step 1: Event Detection", 7),
    "step2": Section("4.2", "Step 2: Extracting Event Attributes", 8),
    "attr_model": Section("4.2.1", "A New Model for Event Attribute Extraction", 9),
    "step3": Section("4.3", "Step 3: Resolving and Disambiguating Entities", 10),
    "wiki_model": Section("4.4", "A New Model for Linking Entities to Their Wikipedia Pages", 11),
    "step4": Section("4.5", "Step 4: Entity Categorization", 12),
    "step5": Section("4.6", "Step 5: Dates and Locations", 12),
    "evaluation": Section("5", "Evaluation", 13),
    "eval_attr": Section("5.1", "Internal Validation of the Attribute Model", 13),
    "eval_wiki": Section("5.2", "Internal Evaluation of the Wikipedia Model", 14),
    "ecav": Section("5.3", "Cross-Ontology Validation: ECAV", 14),
    "ecav_attr": Section("5.3.1", "Evaluating the Attribute Model", 15),
    "ecav_actors": Section("5.3.2", "ECAV Actor Resolution Results", 16),
    "limitations": Section("5.4", "Limitations and Future Work", 17),
}

# Where the PDF is served from. Set NGEC_DEMO_PAPER_URL to a public URL (e.g. the
# preprint) when deploying; otherwise the app serves the copy in demo/static/.
PAPER_URL = os.environ.get("NGEC_DEMO_PAPER_URL", "app/static/paper.pdf")


def url(key: str) -> str:
    """Deep link to a section of the PDF."""
    sec = SECTIONS[key]
    return f"{PAPER_URL}#page={sec.page}"


def cite(key: str) -> str:
    """Inline markdown link, e.g. [§4.2 Step 2: ...](...#page=8).

    For prose passed to `st.markdown` as markdown. Anything embedded in a raw
    HTML block wants `cite_html` instead — see there.
    """
    sec = SECTIONS[key]
    return f"[§{sec.number} {sec.title}]({url(key)})"


def cite_html(key: str) -> str:
    """The same link as real HTML.

    Markdown syntax inside a raw HTML block is *not* processed: by CommonMark a
    line starting with `<div` opens an HTML block that runs to the next blank
    line, and everything inside is passed through verbatim. A `[§4.1 …](…)` put
    there renders as literal brackets and a bare URL, which is what the
    cross-reference line was doing. Emitting the anchor directly sidesteps the
    question.
    """
    sec = SECTIONS[key]
    text = html.escape(f"§{sec.number} {sec.title}")
    return f'<a href="{html.escape(url(key))}" target="_blank">{text}</a>'


def ref(*keys: str, label: str = "In the paper") -> None:
    """Render a small 'see also' line pointing into the manuscript.

    Set as a mono section label in the accent colour. This is the one element the
    design handoff names as the accent's job on a content page — it is the
    cross-reference a reviewer is looking for — so it should be the first orange
    thing the eye finds.
    """
    links = " · ".join(cite_html(k) for k in keys)
    st.markdown(
        f'<div class="po-label po-label--orange ngec-paperref">'
        f"{html.escape(label)} · {links}</div>",
        unsafe_allow_html=True,
    )
