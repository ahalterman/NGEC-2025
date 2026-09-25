"""An honest accounting of what it takes to run this yourself."""

import streamlit as st

from ngec_demo import paper, resources, ui
from ngec_demo.theme import PALETTE, kv_table, note, rule

ui.setup()

ui.page_header(
    kicker="Make it yours",
    title="What setup actually costs",
    standfirst=(
        "The paper's Table 1 is right that changing the actor ontology means editing a text "
        "file. It is quiet about what you must have running before that edit does anything. "
        "This page fills the gap, with measured numbers."
    ),
    paper_keys=("customizing",),
)

st.markdown(
    """
Two things stay separate here, because conflating them is how a customisation table becomes
misleading:

- **Marginal effort** — what it costs to change one component once the system runs. This is
  genuinely low for actor categories, moderate for event types, and high for attributes.
- **Setup effort** — what it costs to get to a working system at all. This is front-loaded,
  mostly one-time, and larger than any row of Table 1 suggests.
"""
)

rule()

st.markdown("## This server, right now")

st.markdown(
    "The demo runs the real package against a real index, so its own status is the most "
    "honest illustration of what has to be standing up."
)

health = resources.health()
rows = []
for name, h in health.items():
    mark = "●" if h.ok else "○"
    colour = PALETTE["good"] if h.ok else PALETTE["bad"]
    rows.append((
        name,
        f'<span style="color:{colour}">{mark}</span> {h.detail}',
    ))
st.markdown(kv_table(rows), unsafe_allow_html=True)

rule()

st.markdown("## The one-time cost, itemised")

st.markdown(
    kv_table([
        ("Python 3.10+",
         "The package uses <code>X | Y</code> types and <code>match</code>. A base conda "
         "environment is often 3.9 and will not import it."),
        ("Package install",
         "Managed with <code>uv</code>. You must choose exactly one of the "
         "<code>cpu</code> / <code>cu12</code> / <code>cu13</code> extras, which redirect "
         "PyTorch to the matching index. Choosing none silently installs a CUDA 13 build that "
         "falls back to CPU on an older driver — a real trap, and the reason "
         "<code>install.py</code> exists to detect the driver for you."),
        ("Model downloads",
         "<strong>~2.5 GB.</strong> The attribute model (1.2 GB), the sentence encoder "
         "(419 MB), and the two spaCy models — <code>en_core_web_trf</code> (478 MB) and "
         "<code>en_core_web_lg</code> (425 MB). Plus ~90 MB of assets in the package."),
        ("Elasticsearch 7.x",
         "Not optional, and not a library — a service you run and keep running. The client is "
         "pinned to <code>elasticsearch==7.17.9</code>; an ES 8 or 9 server will not work "
         "without changing that pin. The reference setup is 7.10.1 in Docker."),
        ("The geonames index",
         "<strong>12,571,784 documents, 1.9 GB.</strong> Needed for step 5. Built by the "
         "scripts in <code>setup/</code>."),
        ("The Wikipedia index",
         "<strong>7,854,807 documents, 11 GB.</strong> The big one. You download a full "
         "English Wikipedia dump, parse it, and index it. This is the single largest barrier "
         "to adoption in the whole system."),
        ("Disk, total",
         "<strong>~15 GB for the indices, ~3 GB for models</strong>, before your corpus."),
        ("GPU",
         "<strong>Optional, and worth about 10x.</strong> This demo is served from a CPU-only "
         "host, where the attribute model is a quantized copy served through llama.cpp and the "
         "whole pipeline takes roughly ten seconds per document — fine for reading a page, slow "
         "for a corpus. On one RTX 4090 with the vllm backend the same pipeline measured about "
         "<strong>one second per document</strong>, extraction itself being 0.2–0.3s of it. "
         "Nothing about the method needs a GPU; the wait you see here is the machine, not the "
         "pipeline."),
    ]),
    unsafe_allow_html=True,
)

note(
    "A stale system CUDA on <code>LD_LIBRARY_PATH</code> shadows the PyTorch wheels' own "
    "libraries and produces <code>undefined symbol: __nvJitLinkGetErrorLogSize_12_9</code>. "
    "The fix is to run with <code>env -u LD_LIBRARY_PATH</code>. It is documented, and it is "
    "the kind of thing that costs someone an afternoon.",
    title="One specific trap",
    tone="warn",
)

rule()

st.markdown("## Where the documentation now stands")

st.markdown(
    """
Several documentation gaps have been addressed. Specifically:

- `RUNNING.md` documents the end-to-end run, the install extras, the CUDA caveat, and the
  Elasticsearch requirement with the actual index sizes.
- `README.md` covers installation, the `cpu`/`cu12`/`cu13` choice, spaCy models, and backends.
- `PIPELINE.md` documents the data contract at each step, including the seam that used to
  crash on empty extractions.
- `CLASSIFIERS.md` documents where the demonstration classifiers came from, what is still
  wrong with them, and where the training data lives.
- `setup/` holds the scripts that build the Elasticsearch indices and train the classifiers.
- `install.py` detects the driver and picks the right extra.

What remains honestly incomplete: `context_class.py` and `mode_class.py` are vestigial stubs
from an earlier architecture and should be finished or deleted, and `formatter.py` still
carries commented-out dead code using an older span-based format that does not reflect the
current contract.
"""
)

rule()

st.markdown("## The lever that would matter most")

st.markdown(
    """
The Wikipedia index is the one that turns "an afternoon" into "a project". Eleven gigabytes,
requires downloading and parsing a full dump, and it is the prerequisite for the
entity-linking step — arguably the paper's most useful standalone piece.

A **hosted, read-only index** removes that barrier entirely for anyone evaluating the system
or teaching with it — which is what this demo already runs against. Publishing that endpoint
would do more for adoption than any documentation improvement, and it would make "editing a
category mapping file" a true description of the total effort for a large class of users
rather than a description of the last step only.
"""
)

note(
    "The recommendation for the paper is not to hide the setup cost but to state it: a short, "
    "concrete paragraph giving the index sizes, the model downloads, and the Elasticsearch "
    "requirement, next to Table 1. A reader who knows what they are getting into and proceeds "
    "is a better outcome than one who discovers it after cloning.",
    title="What to put in the paper",
)

paper.ref("customizing", "limitations")
