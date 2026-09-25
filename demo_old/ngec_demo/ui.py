"""Shared page furniture: headers, the step rail, the document box, record views.

Three conventions hold the demo together and are worth stating:

1. **The document box is the page's call to action.** Coding your own text is
   the thing the demo exists to invite, so on the pages that read an article the
   box holding it is editable, always visible, and the first thing under the
   header. It used to be the last option in a dropdown, which read as an
   afterthought.
2. **No empty text boxes.** The box arrives with a worked example already in it,
   because a visiting reviewer will not type one in. The curated examples are
   starting points for the box, not a separate mode.
3. **The example follows you.** The chosen document is kept in the URL query
   string, so the step rail can carry it from page to page and so any panel can
   be linked to directly from the response-to-reviewers. Text a visitor typed is
   kept in session state instead — it is theirs, and it does not go in a URL.
"""

from __future__ import annotations

import html
import json

import streamlit as st

from . import examples as ex_mod
from . import paper, pipeline, resources
from .theme import (
    PALETTE,
    SPAN_COLORS,
    inject_css,
    kv_table,
    tag,
    unmatched_spans,
)

# The paper's five-part framework, which is what the rail shows. The code has
# six steps because "story → events" is an implementation detail, not a
# conceptual one.
#
# The number is a separate field from the name so the rail can set it in the
# boxed numeral the design uses for ordered items; it is not decoration, it is
# how a visitor keeps their place in a five-step argument.
RAIL = [
    ("step1", "01", "Detection"),
    ("step2", "02", "Attributes"),
    ("step3", "03", "Entity linking"),
    ("step4", "04", "Categories"),
    ("step5", "05", "Dates & places"),
]


def boot() -> None:
    """Called once by app.py, before navigation resolves a page."""
    st.set_page_config(
        page_title="NGEC — custom event data",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def setup() -> None:
    """Per-page boilerplate. Call first, before anything renders.

    The page title itself comes from `st.Page(title=...)` in app.py, so this
    only has to apply the theme, check the services, and draw the sidebar.
    """
    inject_css()
    startup_check()
    sidebar_status()


def startup_check() -> None:
    """Say plainly, at the top of the page, when a service the demo needs is down.

    The pipeline's dependencies — Elasticsearch and, on a CPU host, the
    llama-server carrying the attribute model — are separate processes that do
    not come back on their own after a reboot. Without this the pages still
    degrade rather than crash, but a visitor cannot tell "this step is
    unavailable right now" from "this is all the pipeline does", which is a
    misreading the demo cannot afford. So the failure is named up front, along
    with what it costs.
    """
    down = resources.degraded()
    if not down:
        return

    lines = "".join(
        f"<li><strong>{html.escape(name)}</strong> — {html.escape(h.detail)}. "
        f"Unavailable while it is down: {html.escape(h.blocks)}.</li>"
        for name, h in down
    )
    st.markdown(
        f'<div class="ngec-note bad"><span class="ngec-note-title">'
        f"Not everything the demo needs is running</span>"
        f"<ul style='margin:0.4rem 0 0 1.1rem;padding:0'>{lines}</ul>"
        f"<div style='margin-top:0.5rem'>The rest of the page works as usual. "
        f"Anything that depends on a service listed above says so in place of the "
        f"output it would otherwise show.</div></div>",
        unsafe_allow_html=True,
    )

    fixes = [(name, h.fix) for name, h in down if h.fix]
    if fixes:
        with st.expander("Restarting these (for whoever is running the demo)", expanded=False):
            for name, fix in fixes:
                st.markdown(f"**{name}**")
                st.code(fix, language="shell")
            # The health check and the failed connections are both cached, so a
            # service that has just come back would keep showing as down.
            if st.button("Check again", key="_recheck_health"):
                resources.recheck()
                st.rerun()


def page_header(kicker: str, title: str, standfirst: str, paper_keys=()) -> None:
    st.markdown(
        f'<div class="ngec-kicker">{html.escape(kicker)}</div>'
        f"<h1>{html.escape(title)}</h1>"
        f'<div class="ngec-standfirst">{standfirst}</div>',
        unsafe_allow_html=True,
    )
    if paper_keys:
        paper.ref(*paper_keys)


def step_rail(current: str) -> None:
    """Row of links to the five step pages, carrying the current example along."""
    current_ex = st.query_params.get("ex", "")
    suffix = f"?ex={current_ex}" if current_ex else ""

    cells = []
    for key, number, label in RAIL:
        inner = f'<span class="n">{number}</span>{html.escape(label)}'
        if key == current:
            cells.append(f'<span class="s here">{inner}</span>')
        else:
            cells.append(f'<a class="s" href="{key}{suffix}" target="_self">{inner}</a>')
    st.markdown(
        f'<div class="ngec-steprail">{"".join(cells)}</div>', unsafe_allow_html=True
    )


def sidebar_status() -> None:
    """Dependency status and a short orientation note."""
    with st.sidebar:
        # A filled black tag rather than a heading: in this design the product
        # name is a mark on a front plate, not a section title.
        st.markdown(tag("NGEC", ink=True), unsafe_allow_html=True)
        st.markdown(
            '<div class="ngec-meta" style="margin-top:10px">A demonstration of the '
            "pipeline described in "
            '“Creating Custom Event Data: A Bag-of-Tricks”. Every panel runs the '
            "real code on the text you give it.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Open by default when something is down, so the detail is one glance
        # away from the banner rather than one click.
        with st.expander("System status", expanded=bool(resources.degraded())):
            for name, h in resources.health().items():
                mark = "●" if h.ok else "○"
                color = PALETTE["good"] if h.ok else PALETTE["ink_faint"]
                st.markdown(
                    f'<div class="ngec-meta"><span style="color:{color}">{mark}</span> '
                    f"<strong>{html.escape(name)}</strong><br>"
                    f'<span style="padding-left:1em">{html.escape(h.detail)}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            f'<div class="ngec-meta" style="margin-top:1em">'
            f'<a href="{paper.PAPER_URL}" target="_blank">Read the paper (PDF)</a><br>'
            f'<a href="https://github.com/ahalterman/NGEC-2025" target="_blank">'
            f"Source on GitHub</a></div>",
            unsafe_allow_html=True,
        )


# The document every page is currently working on, session-wide rather than
# per-page: a visitor who pastes a story on the end-to-end page and then opens a
# step page expects to still be looking at their story.
ACTIVE = "_active_document"


def _active(page: str) -> dict:
    """The document in the box: {'text', 'pub_date', 'key'}.

    `key` is an example key, or None for text the visitor supplied. Resolution
    order is the URL (so a link to a particular example works), then whatever is
    already in session state, then this page's default example — the last of
    which is why no page ever opens on an empty box.
    """
    from_url = st.query_params.get("ex")
    if from_url and from_url in ex_mod.BY_KEY and st.session_state.get(ACTIVE, {}).get("key") != from_url:
        _set_active(ex_mod.BY_KEY[from_url])

    if ACTIVE not in st.session_state:
        _set_active(ex_mod.default_for(page))
    return st.session_state[ACTIVE]


def _set_active(example: ex_mod.Example | None, text: str = "", pub_date: str = "") -> None:
    """Put a document in the box, from an example or from typed text."""
    if example is not None:
        st.session_state[ACTIVE] = {"text": example.text, "pub_date": example.pub_date,
                                    "key": example.key}
        st.query_params["ex"] = example.key
    else:
        st.session_state[ACTIVE] = {"text": text, "pub_date": pub_date, "key": None}
        # A visitor's own text is not carried in the URL: it is theirs, and a
        # stale `ex` would make the page claim to be showing an example.
        if "ex" in st.query_params:
            del st.query_params["ex"]


def example_picker(page: str, show_source: bool = True
                   ) -> tuple[str, str, ex_mod.Example | None]:
    """The document box, plus the examples that seed it.

    Returns (text, pub_date, example_or_None) — `None` when the text is the
    visitor's own, which is the signal the "what to notice" note keys off.

    There is deliberately no "curated example *or* your own text" mode switch.
    There is one box; it starts with an example in it and the visitor overwrites
    it. Making that the top of the page is the point: pasting your own text is
    the demo's central invitation, and it previously sat behind a dropdown
    option that no one found.

    `show_source` says whether this page's step reads the article. Detection,
    attribute extraction and the hard-case pages do, and get the box inline.
    Steps 3 to 5 do not — they receive a handful of spans that earlier steps
    pulled out, and putting a full article at the top of those pages implied
    otherwise and pushed the part that matters below the fold. There the box is
    a collapsed panel under the examples, and `stage_input` shows what the step
    actually receives.
    """
    options = ex_mod.for_page(page) or ex_mod.EXAMPLES
    active = _active(page)

    if show_source:
        _document_box(page, active)
        _example_pills(page, options, active)
    else:
        _example_pills(page, options, active)
        # Opened when it holds the visitor's own text, so that what they pasted
        # is visible rather than hidden behind a click on the page that used it.
        with st.expander("✎ Code your own text", expanded=active["key"] is None):
            _document_box(page, active, compact=True)

    return active["text"], active["pub_date"], ex_mod.BY_KEY.get(active["key"] or "")


def _document_box(page: str, active: dict, compact: bool = False) -> None:
    """The editable document, in a form so it codes on submit and not per keystroke.

    A bare text_area would re-run the pipeline on every edit, because each new
    string is a new cache key — a live run per pause in typing.

    This box replaces the read-only article panel the pages used to print above
    their output. One box that both shows the document and accepts a new one is
    less to explain than two panels showing the same text, and it removes the
    question the old layout raised: which of these is the thing being coded?
    """
    with (
        st.container(key=f"docbox_{page}"),
        st.form(f"doc_form_{page}", clear_on_submit=False, border=not compact),
    ):
        if not compact:
            st.markdown(
                '<div class="ngec-kicker">The document — paste your own over it</div>',
                unsafe_allow_html=True,
            )
        text = st.text_area(
            "Document text",
            value=active["text"],
            height=150 if compact else 230,
            label_visibility="collapsed",
            placeholder="Paste the text of a news story here — a few paragraphs is plenty.",
        )
        left, right = st.columns([1, 2], gap="medium", vertical_alignment="bottom")
        with left:
            pub_date = st.text_input(
                "Published",
                value=active["pub_date"],
                help="Relative dates in the text ('Tuesday', 'last week') resolve "
                     "against this.",
            )
        with right:
            submitted = st.form_submit_button("Code this document", type="primary")

    if not submitted:
        return
    if not text.strip():
        st.markdown(
            '<div class="ngec-note warn"><span class="ngec-note-title">'
            "Nothing to code</span>Paste some text first.</div>",
            unsafe_allow_html=True,
        )
        return

    # Unchanged example text stays an example: its pre-coded run is then still a
    # cache hit, so pressing the button on an example is instant rather than a
    # minute of live coding.
    match = next((e for e in ex_mod.EXAMPLES
                  if e.text == text and e.pub_date == pub_date), None)
    _set_active(match, text=text, pub_date=pub_date)
    # Rerun rather than falling through, so that everything below this point --
    # including the pills, which have their own state -- is drawn from one
    # consistent active document.
    st.rerun()


def _example_pills(page: str, options: list[ex_mod.Example], active: dict) -> None:
    """The curated documents, as one-click seeds for the box.

    Titles only. Each example used to carry a blurb under the picker and a "what
    to notice" note above the output, which together said more about the
    examples than about the pipeline — and the examples are scaffolding, not the
    argument. The blurb is now a single line under the pills, and the longer
    note is a click away.
    """
    labels = {e.title + ("  ⚠" if e.honest else ""): e for e in options}
    key = f"ex_pills_{page}"
    current = next((label for label, e in labels.items() if e.key == active["key"]), None)

    def picked() -> None:
        chosen = labels.get(st.session_state.get(key) or "")
        if chosen is not None:
            _set_active(chosen)

    # Seeded once, then owned by the widget: writing to it on every run would
    # overwrite the visitor's click before the callback saw it.
    st.session_state.setdefault(key, current)
    st.pills("Or start from a prepared example", list(labels), key=key,
             selection_mode="single", on_change=picked)

    example = labels.get(current or "")
    if example is not None and example.blurb:
        st.markdown(
            f'<div class="ngec-meta" style="margin:-0.3rem 0 0.6rem 0">'
            f"{html.escape(example.blurb)}</div>",
            unsafe_allow_html=True,
        )
    elif active["key"] is None:
        st.markdown(
            '<div class="ngec-meta" style="margin:-0.3rem 0 0.6rem 0">'
            "Coding your own text. It is not saved anywhere and is not carried in the "
            "URL, so unlike the prepared examples this panel cannot be linked to — and "
            "it is coded from scratch rather than served pre-coded.</div>",
            unsafe_allow_html=True,
        )


def stage_input(rows: list[tuple[str, str]], explain: str = "") -> None:
    """What this step receives, for a step that does not read the article.

    Steps 3, 4 and 5 are handed spans, not prose: a name to look up, a mention
    to categorise, a date expression to resolve against a publication date.
    Showing them the whole article said the opposite. Rows are the same
    `(key, value, role)` triples the record tables take, so a location row here
    and a location row in the output carry the same colour.
    """
    st.markdown(
        '<div class="ngec-kicker">What this step receives</div>', unsafe_allow_html=True
    )
    st.markdown(kv_table(rows), unsafe_allow_html=True)
    tail = f"{explain} " if explain else ""
    st.markdown(
        f'<div class="ngec-meta" style="margin-top:0.45rem">{tail}'
        f'The article these came out of is on the <a href="end_to_end" target="_self">'
        f"end-to-end page</a>, and in the document box above.</div>",
        unsafe_allow_html=True,
    )
    st.write("")


def attribute_model_provenance() -> None:
    """Say which attribute model produced what is on the page.

    On this training box ATTRIBUTE_MODEL is pointed at a local checkpoint
    directory so the demo skips re-downloading weights already on disk (see
    LOCAL_ATTRIBUTE_MODEL in resources.py) — it is the same weights as the
    published `ahalt/qwen3-event-extraction-exp5.1`, not a different model.
    Worth saying anyway, since NGEC_ATTRIBUTE_MODEL could be pointed at
    something else entirely, e.g. the older `event-attribute-extractor` for a
    baseline comparison.
    """
    if resources.ATTRIBUTE_MODEL != resources.LOCAL_ATTRIBUTE_MODEL:
        return
    st.markdown(
        '<div class="ngec-note"><span class="ngec-note-title">'
        "Which model this is</span>"
        "This demo is using a local copy of the attribute model "
        "(<code>qwen3-event-extraction-exp5.1</code>) instead of downloading it — the "
        "same weights published at "
        '<a href="https://huggingface.co/ahalt/qwen3-event-extraction-exp5.1">'
        "ahalt/qwen3-event-extraction-exp5.1</a>. On its held-out evaluation this model is "
        "18 points better on exact actor match and 19 on location than "
        "<code>ahalt/event-attribute-extractor</code>, the model this package used to "
        "default to.</div>",
        unsafe_allow_html=True,
    )


def what_to_notice(example: ex_mod.Example | None) -> None:
    """The note attached to a curated example, if one is loaded.

    Only the cases the pipeline gets *wrong* get a note box. Those are an
    argument the demo is making and the visitor should not be able to miss them.
    The rest are scaffolding, and their commentary is one click away: a page that
    opens with two paragraphs about the example before any output says the
    examples are what it is about, and they are not.
    """
    if not example or not example.notice:
        return
    if example.honest:
        st.markdown(
            f'<div class="ngec-note warn"><span class="ngec-note-title">'
            f"A case the pipeline gets wrong</span>"
            f"{html.escape(example.notice)}</div>",
            unsafe_allow_html=True,
        )
        return
    with st.expander("What to notice in this example", expanded=False):
        st.markdown(
            f'<div class="ngec-meta">{html.escape(example.notice)}</div>',
            unsafe_allow_html=True,
        )


def run_with_progress(text: str, pub_date: str, stop_after: str | None = None,
                      event_type_override: str | None = None,
                      label: str = "Running the pipeline") -> pipeline.Run:
    """Run (or fetch from cache) with a live per-step status display."""
    # Keyed rather than single-slot: a visitor moving between step pages runs the
    # same document to different depths, and a single slot would re-run the
    # pipeline on every switch.
    cache: dict = st.session_state.setdefault("_runs", {})
    cache_key = (text, pub_date, stop_after, event_type_override)
    if cache_key in cache:
        return cache[cache_key]

    # Curated examples are pre-coded by build_example_cache.py. A document costs
    # about a minute on CPU, so without this a visitor clicking between examples
    # would wait a minute each time.
    if not event_type_override:
        precoded = pipeline.cached_example(text, pub_date, stop_after)
        if precoded is not None:
            cache[cache_key] = precoded
            return precoded

    # A live run reaches the attribute model, which is the slow part; say so
    # rather than showing an unexplained spinner. And say why it is slow — a
    # visitor watching a ten-second spinner should not conclude that the method
    # is slow when what is slow is the CPU this happens to be served from.
    reaches_attributes = stop_after is None or stop_after in ("attributes", "actors", "format")
    if reaches_attributes and resources.BACKEND == "llamacpp":
        hint = (" — the attribute model runs on this server's CPU, so allow "
                "ten seconds or so; on a GPU it is about one second")
    elif reaches_attributes and resources.BACKEND == "transformers":
        hint = " — the attribute model runs on CPU unquantized here, so allow about a minute"
    else:
        hint = ""

    with st.status(f"{label}{hint}", expanded=True) as status:
        placeholder = st.empty()

        def progress(key, step_label):
            note = " (the slow step)" if key == "attributes" else ""
            placeholder.markdown(
                f'<div class="ngec-meta">{html.escape(step_label)}{note}…</div>',
                unsafe_allow_html=True,
            )

        result = pipeline.run(text, pub_date, stop_after=stop_after,
                              event_type_override=event_type_override,
                              progress=progress)
        placeholder.empty()
        elapsed = result.total_seconds
        status.update(label=f"{label} — {elapsed:.1f}s", state="complete", expanded=False)

    # A run whose dependencies were down shouldn't be cached — the visitor may be
    # looking at the page again precisely because the index came back.
    if not any(s.skipped for s in result.steps):
        cache[cache_key] = result
        # Bound the cache; these hold whole record lists.
        for stale in list(cache)[:-12]:
            cache.pop(stale, None)
    return result


def repeated_modes_note(records: list[dict]) -> None:
    """Explain records that share an event type and differ only in mode.

    Shown only when it actually happens, because it is a specific and slightly
    surprising property rather than something worth saying on every page: the
    document is sent to the attribute model once per detected mode, so a story
    classified as ASSAULT-explosives *and* ASSAULT-heavy-weapons is asked about
    twice and yields two records describing the same attack.
    """
    by_type: dict[str, list[str]] = {}
    for record in records:
        event_type = str(record.get("event_type") or "")
        mode = str(record.get("event_mode") or "")
        if event_type:
            by_type.setdefault(event_type, []).append(mode)

    repeated = {t: ms for t, ms in by_type.items() if len(ms) > 1 and any(ms)}
    if not repeated:
        return

    described = "; ".join(
        f"<strong>{html.escape(t)}</strong> as "
        + ", ".join(f"<em>{html.escape(m)}</em>" for m in modes if m)
        for t, modes in repeated.items()
    )
    st.markdown(
        f'<div class="ngec-note warn"><span class="ngec-note-title">'
        f"Several records, one event</span>"
        f"This document was classified under more than one <em>mode</em> of the same "
        f"event type — {described}. Extraction runs once per mode, so each one "
        f"produces its own record, and when the modes describe the same underlying "
        f"event those records come out nearly identical. They are not duplicates in "
        f"the pipeline's terms: each is a distinct type-mode pair. But anyone counting "
        f"events in a corpus has to decide what to do with them, and the demo would "
        f"rather show that than quietly merge them.</div>",
        unsafe_allow_html=True,
    )


def dropped_events(result: pipeline.Run) -> None:
    """The event types the classifier proposed and the attribute model declined.

    This is the pipeline's most interesting self-correction and it used to be
    invisible: the count appeared in a step note and the records themselves went
    to a `*_dropped_events.jsonl` file that a visitor cannot see. The records are
    already in the trace, so the demo can just diff the two steps.

    Matching is on the (event type, mode) pair rather than on `id`, because
    `explode_events` appends an `_<index>` suffix to the ids it emits and the
    pair is exact.
    """
    before = result.records_after("split")
    after = result.records_after("attributes")
    if not before:
        return

    def key(record: dict) -> tuple[str, str]:
        return (str(record.get("event_type") or ""), str(record.get("event_mode") or ""))

    kept = {key(r) for r in after}
    dropped = [r for r in before if key(r) not in kept]
    if not dropped:
        return

    items = "".join(
        f"<li><strong>{html.escape(event_type)}</strong>"
        + (f" · <em>{html.escape(mode)}</em>" if mode else "")
        + "</li>"
        for event_type, mode in dict.fromkeys(key(r) for r in dropped)
    )
    st.markdown(
        f"## The classifier proposed {len(dropped)}, the model declined them\n"
    )
    st.markdown(
        f'<div class="ngec-note"><span class="ngec-note-title">'
        f"Dropped before the record was made</span>"
        f"The event classifier fired for these, and the attribute model was asked "
        f"about each one. It returned nothing — there is no such event in this text — "
        f"so no record was made:"
        f"<ul style='margin:0.5rem 0 0 1.1rem;padding:0'>{items}</ul>"
        f"<div style='margin-top:0.6rem'>This is the extraction step correcting a "
        f"classifier that over-fires, and it is why the coded output is cleaner than "
        f"the detection scores alone would suggest. It also means <strong>classifier "
        f"quality cannot be read off this page</strong> — for that, see the scores on "
        f"the <a href='step1' target='_self'>detection page</a>. In a corpus run these "
        f"are counted in a warning rather than shown.</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")


def run_error(result: pipeline.Run) -> bool:
    """Render whatever went wrong. Returns True if the caller should stop."""
    if result.error == "no_event":
        st.markdown(
            '<div class="ngec-note warn"><span class="ngec-note-title">'
            "No event type passed its threshold</span>"
            "The pipeline stops here: with no event type there is nothing to extract "
            "attributes for. On the step 1 page you can see how close each classifier "
            "came, and you can force an event type to see the later steps anyway.</div>",
            unsafe_allow_html=True,
        )
        return True
    if result.error == "no_attributes":
        st.markdown(
            '<div class="ngec-note warn"><span class="ngec-note-title">'
            "The attribute model extracted nothing</span>"
            "The record is dropped rather than emitted with empty attributes. In a corpus "
            "run these are counted in a warning and written to a "
            "<code>*_dropped_events.jsonl</code> file.</div>",
            unsafe_allow_html=True,
        )
        return True
    if result.error:
        st.error(f"The pipeline raised an error: {result.error}")
        return True
    return False


# --------------------------------------------------------------------------
# rendering pipeline output
# --------------------------------------------------------------------------


def attribute_spans(record: dict) -> dict[str, list[str]]:
    """The four attribute span lists from a record, normalised to lists."""
    attrs = record.get("attributes") or {}
    out = {}
    for role in ("actor", "recipient", "date", "location"):
        value = attrs.get(role) or []
        if isinstance(value, str):
            value = [value]
        # The model writes "N/A" when it finds no filler for a slot.
        out[role] = [v for v in value if v and str(v).strip().upper() not in ("N/A", "NA", "NONE")]
    return out


def span_list(spans: list[str]) -> str:
    """The spans filling one slot, one per line, verbatim."""
    return "".join(f'<span class="span">{html.escape(str(s))}</span>' for s in spans)


def render_attributes(record: dict, text: str | None = None) -> None:
    """What the attribute model pulled out of the document, as a table.

    One row per slot, listing the spans that filled it. This replaced marking
    the spans up inside the article: with four slots, several records per
    document and spans scattered across paragraphs, the marked-up version was
    hard to read off and impossible to attribute to a particular record. A table
    per record answers "what did the model say the actor was" directly, and it
    is the same table `render_record` uses for the coded version of the same
    slots, so the two read as before and after.

    `text` turns on the verbatim check, which is the one thing the marked-up
    version did for free: a span that cannot be found in the article is the
    model paraphrasing rather than extracting.
    """
    spans = attribute_spans(record)
    attrs = record.get("attributes") or {}

    rows = [
        (SPAN_COLORS[role]["label"], span_list(spans[role]), role)
        for role in ("actor", "recipient", "date", "location")
    ]
    rows.append(("Anchor quote", html.escape(str(attrs.get("anchor_quote") or ""))))
    st.markdown(kv_table(rows), unsafe_allow_html=True)

    missing = unmatched_spans(text, spans) if text else []
    if missing:
        items = ", ".join(f"<em>{html.escape(s)}</em> ({r})" for r, s in missing)
        st.markdown(
            f'<div class="ngec-note warn"><span class="ngec-note-title">'
            f"Not found verbatim in the document</span>"
            f"The model returned {items}. Extraction should copy spans out of the text; "
            f"a span that is not there is the model paraphrasing, which is worth "
            f"knowing about.</div>",
            unsafe_allow_html=True,
        )


def mention_rows(records: list[dict]) -> list[tuple[str, str, str]]:
    """The actor and recipient mentions across records, as `stage_input` rows.

    These strings are what steps 3 and 4 work from. Repeats are collapsed, since
    a name extracted for two records is the same name being coded, but the order
    the records give them is kept.
    """
    rows = []
    for role in ("actor", "recipient"):
        seen: list[str] = []
        for record in records:
            for span in attribute_spans(record)[role]:
                if span not in seen:
                    seen.append(span)
        if seen:
            rows.append((SPAN_COLORS[role]["label"], span_list(seen), role))
    if not rows:
        rows = [("Mentions", '<span class="none">none extracted</span>')]
    return rows


def record_label(record: dict) -> str:
    """" · ASSAULT · explosives" — the type and mode that identify a record.

    Two records of the same event type but different modes are separate records
    with, often, identical attributes: the model is asked about the document once
    per mode and only the sub-event definition changes. Labelling with the type
    alone makes those look like the same event emitted twice, which is the single
    most confusing thing the output does. The mode is what tells them apart, so
    it goes in every heading that names a record.
    """
    bits = [str(record.get("event_type") or "")]
    mode = str(record.get("event_mode") or "")
    if mode:
        bits.append(mode)
    return "".join(f" · {html.escape(b)}" for b in bits if b)


def _fmt_actor(coded: dict) -> str:
    """One coded actor as a compact readable string."""
    bits = []
    query = coded.get("actor_role_query") or coded.get("query") or ""
    if query:
        bits.append(html.escape(str(query)))
    code = " ".join(str(coded.get(k) or "") for k in ("country", "code_1")).strip()
    if code:
        bits.append(f"<strong>{html.escape(code)}</strong>")
    wiki = coded.get("wiki")
    if wiki:
        url = f"https://en.wikipedia.org/wiki/{str(wiki).replace(' ', '_')}"
        bits.append(f'<a href="{html.escape(url)}" target="_blank">{html.escape(str(wiki))}</a>')
    return " → ".join(bits) if bits else '<span class="none">unresolved</span>'


def render_record(record: dict) -> None:
    """A coded event record as a definition table."""
    attrs = record.get("attributes") or {}

    def actors(field_name):
        coded = record.get(field_name) or []
        if coded:
            return "<br>".join(_fmt_actor(c) for c in coded)
        raw = attrs.get(field_name) or []
        raw = [r for r in raw if str(r).strip().upper() not in ("N/A", "NA")]
        if raw:
            return (
                html.escape("; ".join(map(str, raw)))
                + ' <span class="none">— extracted but not coded</span>'
            )
        return ""

    geo = (record.get("event_location") or {}).get("event_loc") or {}
    place_bits = [geo.get("resolved_placename"), geo.get("admin1_name"), geo.get("country_code3")]
    place = ", ".join(str(b) for b in place_bits if b)
    if geo.get("lat") is not None:
        place += f'  <span class="none">({geo["lat"]:.3f}, {geo["lon"]:.3f})</span>'

    date_res = record.get("date_resolved") or {}
    date_str = str(date_res.get("resolved_date") or "")[:10]
    if date_res.get("date_end"):
        date_str += f" → {str(date_res['date_end'])[:10]}"
    if date_res.get("date_type"):
        date_str += f'  <span class="none">({date_res["date_type"]}, ' \
                    f'{date_res.get("granularity", "")})</span>'

    rows = [
        ("Event type", f'<strong>{html.escape(str(record.get("event_type") or ""))}</strong>'),
        ("Mode", html.escape(str(record.get("event_mode") or ""))),
        ("Actor", actors("actor"), "actor"),
        ("Recipient", actors("recipient"), "recipient"),
        ("Location", place, "location"),
        ("Date", date_str, "date"),
        ("Anchor quote", html.escape(str(attrs.get("anchor_quote") or ""))),
    ]
    st.markdown(kv_table(rows), unsafe_allow_html=True)


def render_records(records: list[dict], text: str | None = None) -> None:
    """Several coded events, one table each.

    `text`, if given, turns on the verbatim check on the extracted spans; the
    document itself is shown once at the top of the page, not once per record.
    """
    if not records:
        st.markdown('<div class="ngec-meta">No events.</div>', unsafe_allow_html=True)
        return

    for i, record in enumerate(records):
        if len(records) > 1:
            st.markdown(
                f'<div class="ngec-kicker" style="margin-top:1.2rem">'
                f"Event {i + 1} of {len(records)}{record_label(record)}</div>",
                unsafe_allow_html=True,
            )
        render_record(record)
        if text:
            missing = unmatched_spans(text, attribute_spans(record))
            if missing:
                items = ", ".join(f"<em>{html.escape(s)}</em> ({r})" for r, s in missing)
                st.markdown(
                    f'<div class="ngec-note warn"><span class="ngec-note-title">'
                    f"Not found verbatim in the document</span>"
                    f"The model returned {items}, which does not appear in the article. "
                    f"Extraction should copy spans out of the text, so this is the model "
                    f"paraphrasing.</div>",
                    unsafe_allow_html=True,
                )


def raw_json(label: str, obj, expanded: bool = False) -> None:
    with st.expander(label, expanded=expanded):
        st.code(json.dumps(obj, indent=2, default=str), language="json")


def timing_table(result: pipeline.Run) -> None:
    rows = []
    for step in result.steps:
        state = "skipped" if step.skipped else f"{step.seconds:.2f}s"
        rows.append((step.label, f'{state} <span class="none">— {html.escape(step.note)}</span>'))
    rows.append(("Total", f"<strong>{result.total_seconds:.2f}s</strong>"))
    st.markdown(kv_table(rows), unsafe_allow_html=True)
