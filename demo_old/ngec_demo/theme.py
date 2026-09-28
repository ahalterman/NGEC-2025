"""Visual theme for the demo: palette, typography, and the record tables.

The look is the "Pocket Operator" design handoff: white paper, black hairlines,
grey chips, and exactly one orange. Nothing is rounded and nothing glows; depth
comes from 1px rules, never from shadows. Body copy is Space Grotesk, and IBM
Plex Mono at 10px uppercase is reserved for things the machine says about itself
— section labels, step numbers, timings, units.

Two stylesheets are involved and the split matters:

* `demo/handoff/pocket-operator.css` is the handoff **as authored**. It owns the
  tokens and the Streamlit widget overrides. Treat it as read-only: when a
  Streamlit upgrade moves a `data-testid`, re-point the selector there rather
  than restyling the component here.
* `_components_css()` below is this app's own furniture — the notes, the record
  tables, the classifier bars, the step rail. It is written entirely against the
  `--po-*` custom properties, so a token change in the handoff file propagates
  without touching Python.

The one accent is a signal, not a colour scheme: it marks the thing you should
look at, and never more than one kind of thing per view. If two oranges compete
on a page, one of them is wrong.
"""

import html
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st

# The handoff stylesheet, which owns the tokens.
_CSS_PATH = Path(__file__).resolve().parent.parent / "handoff" / "pocket-operator.css"

# Mirror of the `--po-*` tokens, for the handful of places that build an inline
# style from Python (a status dot, a coloured word inside a sentence). The names
# are the ones the pages already use, so remapping the look did not mean editing
# every call site.
#
# Note what happened to the status colours. The old theme had a green/amber/red
# triad; this design has one accent, and "coloured status pills in more than one
# hue" is a named failure mode. So state is carried by ink-versus-grey, and
# orange is spent only on the thing that wants attention: `good` is simply black
# and `bad`/`warn` are the accent. That is a real loss of resolution — a page
# cannot now distinguish "degraded" from "failed" by colour — so anything making
# that distinction has to say it in words, which every current call site does.
PALETTE = {
    "paper": "#FFFFFF",
    "card": "#F2F2F0",
    "rule": "#E6E6E4",
    "rule_hard": "#111111",
    "ink": "#111111",
    "ink_soft": "#43433F",
    "ink_faint": "#8E8E8A",
    "body": "#2A2A28",
    "mute": "#6C6C68",
    "accent": "#F0521E",
    "accent_hover": "#D9451A",
    # Status, collapsed onto the one-accent rule; see the note above.
    "good": "#111111",
    "warn": "#F0521E",
    "bad": "#F0521E",
}

# The four attributes the paper defines an event as having (Section 3, step 2).
# One colour each, carried by the label cell of the row holding that attribute,
# in every table on the site: extracted spans, the step input panels, and the
# coded record.
#
# `hue` is the identity carrier and the *only* place these colours appear: a
# saturated 3px rule down the left edge of an otherwise neutral cell. The cell
# itself sits on the standard grey chip. The previous theme tinted each cell with
# a pale wash of its hue, which under this design would put four competing
# colour fields on the page — exactly the "coloured pills in more than one hue"
# that the handoff rules out. Confining the hue to the hairline keeps the paper
# neutral and keeps the identity.
#
# Recipient moved from orange `#eb6834` to yellow, because the old hue was
# essentially the new accent `#F0521E` and a reader would fairly have read an
# orange rule as "this is the row to look at". The four were then re-validated
# **as a set with the accent included**, since an accented label and an attribute
# table share a screen: worst all-pairs CVD ΔE 9.1 (protan), worst normal-vision
# ΔE 16.3, on the `#F2F2F0` chip.
#
# Every other fourth hue fails once the accent is in the pairlist — red is ΔE 4.9
# from it under deuteranopia, green 1.5 under protanopia, magenta 14.6 to normal
# vision. Darkening the yellow to lift its contrast backfires for the same
# reason: `#c98500` collapses to ΔE 1.5 from the accent, because brightness is
# precisely what separates yellow from orange under CVD. So the bright step
# stands, and its low contrast against the chip (1.93:1) is relieved the way it
# always was here — every row is labelled in words, so identity is never colour
# alone.
SPAN_COLORS = {
    "actor": {"hue": "#2a78d6", "label": "Actor"},
    "recipient": {"hue": "#eda100", "label": "Recipient"},
    "location": {"hue": "#1baf7a", "label": "Location"},
    "date": {"hue": "#4a3aa7", "label": "Date"},
}

# Kept as names rather than literals so pages read declaratively. There is no
# serif in this design; SERIF is retained as an alias because removing it would
# be a silent breakage for anything outside this file that still asks for it.
SANS = 'var(--po-sans)'
MONO = 'var(--po-mono)'
SERIF = SANS


@lru_cache(maxsize=1)
def _handoff_css() -> str:
    """The handoff stylesheet, read once per process.

    A missing file is reported in the app rather than raised: the demo losing its
    typography is bad, but a visitor meeting a traceback instead of the pipeline
    is worse.
    """
    try:
        return _CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def _span_css() -> str:
    """Per-role colour rules, generated so the four hues are declared once.

    The label cell of an attribute row carries its hue as a rule down the left
    edge. `box-shadow` rather than `border-left` so it does not fight the table's
    collapsed borders.
    """
    return "\n".join(
        f'.ngec-kv td.k.ngec-{role} {{ box-shadow: inset 3px 0 0 {c["hue"]}; '
        f"background: var(--po-chip); padding-left: 9px; }}"
        for role, c in SPAN_COLORS.items()
    )


def _components_css() -> str:
    """This app's own furniture, written against the handoff's tokens."""
    return """
/* --- page furniture ---------------------------------------------------- */

/* Mono eyebrow above a title. The machine's voice: what this page is. */
.ngec-kicker {
    font-family: var(--po-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--po-label);
    margin-bottom: 6px;
}

/* The one-sentence summary under a page title. Sans, not mono: this is
   written for a person, and mono is for labels only. */
.ngec-standfirst {
    font-family: var(--po-sans);
    font-size: 19px;
    line-height: 1.5;
    color: var(--po-mute);
    margin: 6px 0 20px 0;
    max-width: 62ch;
    text-wrap: pretty;
}

.ngec-rule { border: none; border-top: var(--po-rule); margin: 26px 0 20px 0; }

/* Caption-ish metadata. Prose, so sans — a mono paragraph is a failure mode. */
.ngec-meta {
    font-family: var(--po-sans);
    font-size: 14px;
    line-height: 1.5;
    color: var(--po-mute);
}

/* A bordered aside. The default is a plain grey chip; only the tones that want
   attention spend the accent, as a 3px rule. `good` takes a black rule instead,
   so a page can still show three states without a second hue. */
.ngec-note {
    background: var(--po-chip);
    padding: 14px;
    margin: 16px 0;
    font-size: 15px;
    line-height: 1.6;
    color: var(--po-body-2);
}
.ngec-note.warn,
.ngec-note.bad  { box-shadow: inset 3px 0 0 var(--po-orange); padding-left: 17px; }
.ngec-note.good { box-shadow: inset 3px 0 0 var(--po-ink);    padding-left: 17px; }
.ngec-note p { font-size: 15px; line-height: 1.6; color: var(--po-body-2); margin: 0; }
.ngec-note .ngec-note-title {
    font-family: var(--po-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--po-ink);
    display: block;
    margin-bottom: 6px;
}
.ngec-note.warn .ngec-note-title,
.ngec-note.bad  .ngec-note-title { color: var(--po-orange); }
.ngec-note ul { margin: 8px 0 0 18px; padding: 0; }
.ngec-note li { font-size: 15px; line-height: 1.6; color: var(--po-body-2); }

/* The document, as given: the quote panel of the design language. */
.ngec-doc {
    font-family: var(--po-sans);
    font-size: 16px;
    line-height: 1.65;
    color: var(--po-body-2);
    background: var(--po-chip);
    padding: 16px 18px;
    max-width: 68ch;
}
/* The document box (ui._document_box): the editable panel holding whatever is
   being coded, and the page's primary call to action. A visitor reads the
   article in it as much as they type into it, so it is set in the same face as
   the `.ngec-doc` panel it replaced rather than in a form-control font. */
[class*="st-key-docbox_"] textarea {
    font-family: var(--po-sans) !important;
    font-size: 15.5px !important;
    line-height: 1.6 !important;
}
[class*="st-key-docbox_"] [data-testid="stForm"] {
    background: var(--po-chip);
}

/* --- horizontal bar rows (step 1 class scores) ------------------------- */

.ngec-bars { margin: 8px 0 4px 0; }
.ngec-bar-row {
    display: grid;
    grid-template-columns: 8.5rem 1fr 3rem;
    align-items: center;
    gap: 10px;
    padding: 2px 0;
}
.ngec-bar-name {
    font-family: var(--po-mono);
    font-size: 11px;
    letter-spacing: 0.06em;
    color: var(--po-ink);
    text-align: right;
    white-space: nowrap;
}
.ngec-bar-row.off .ngec-bar-name { color: var(--po-label); }
.ngec-bar-track {
    position: relative;
    height: 13px;
    background: var(--po-chip);
}
.ngec-bar-fill { position: absolute; left: 0; top: 0; bottom: 0; background: #D8D8D4; }
/* The bars that cleared their threshold are the answer to the question the page
   asks, so they are what the accent marks. There are rarely more than three. */
.ngec-bar-row.on .ngec-bar-fill { background: var(--po-orange); }
.ngec-bar-thresh {
    position: absolute;
    top: -2px; bottom: -2px;
    width: 1px;
    background: var(--po-ink);
}
.ngec-bar-val {
    font-family: var(--po-mono);
    font-size: 11px;
    color: var(--po-mute);
}
.ngec-bar-row.off .ngec-bar-val { color: var(--po-label); }

/* Small mono chip. `on` is a filled black tag; nothing here takes the accent,
   because these appear in rows of a dozen. */
.ngec-chip {
    display: inline-block;
    font-family: var(--po-mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 7px;
    background: var(--po-chip);
    color: var(--po-mute);
    margin: 0 5px 5px 0;
}
.ngec-chip.on { background: var(--po-ink); color: #FFFFFF; }
.ngec-chip.off { color: var(--po-label); }

/* --- step rail --------------------------------------------------------- */

/* The boxed numeral is the most characteristic detail in this design, and the
   five steps of the framework are exactly what it is for. Each cell is a boxed
   `01`..`05` followed by the step name; the current step fills black. */
.ngec-steprail {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: stretch;
    margin: 0 0 20px 0;
}
.ngec-steprail .s {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 9px 4px 4px;
    border: var(--po-rule);
    background: var(--po-paper);
    font-family: var(--po-sans);
    font-size: 13px;
    color: var(--po-ink);
    text-decoration: none;
}
.ngec-steprail .s .n {
    font-family: var(--po-mono);
    font-size: 11px;
    line-height: 1;
    padding: 3px 4px;
    border: var(--po-rule);
    color: var(--po-ink);
}
.ngec-steprail .s:hover { background: var(--po-chip); text-decoration: none; }
.ngec-steprail .s.here { background: var(--po-ink); border-color: var(--po-ink); color: #FFFFFF; }
.ngec-steprail .s.here .n { border-color: #FFFFFF; color: #FFFFFF; }

/* --- record tables ----------------------------------------------------- */

.ngec-kv {
    width: 100%;
    border-collapse: collapse;
    margin: 4px 0 8px 0;
}
.ngec-kv td {
    padding: 8px 10px 8px 0;
    border-bottom: var(--po-hair);
    vertical-align: top;
}
.ngec-kv td.k {
    width: 11rem;
    font-family: var(--po-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--po-label);
    padding-top: 11px;
}
.ngec-kv td.v {
    font-family: var(--po-mono);
    font-size: 13px;
    line-height: 1.5;
    color: var(--po-ink);
}
.ngec-kv td.v .none { color: var(--po-label); }
.ngec-kv td.v .span { display: block; padding: 1px 0; }
.ngec-kv td.v a { color: var(--po-orange); }

/* --- links ------------------------------------------------------------- */

/* `st.page_link` renders as a full-width navigation row with no underline and
   body-text colour, which reads as plain text until the cursor lands on one. On
   the landing page those links *are* the tour, so they get the accent, an
   underline and an arrow, on a box no wider than the label. The colour has to be
   set on the inner elements — Streamlit sets it on the span wrapping the label,
   which would otherwise win. */
a[data-testid="stPageLink-NavLink"] {
    width: auto;
    padding-left: 0;
    padding-right: 0;
    background: transparent !important;
}
a[data-testid="stPageLink-NavLink"] span,
a[data-testid="stPageLink-NavLink"] p {
    font-family: var(--po-sans);
    font-size: 15px;
    color: var(--po-orange) !important;
    font-weight: 500;
    text-decoration: underline;
    text-underline-offset: 0.18em;
    text-decoration-thickness: 1px;
}
a[data-testid="stPageLink-NavLink"]::after {
    content: "→";
    color: var(--po-orange);
    font-weight: 500;
}
a[data-testid="stPageLink-NavLink"]:hover span,
a[data-testid="stPageLink-NavLink"]:hover p { text-decoration-thickness: 2px; }

.stMarkdown p a, .ngec-note a, .ngec-meta a {
    color: var(--po-orange);
    text-decoration: underline;
    text-underline-offset: 0.15em;
}

/* The manuscript cross-reference. The section titles are long, so this is
   allowed to wrap; line-height is set because 0.2em tracking on two lines of
   10px mono otherwise collides. */
.ngec-paperref { margin: 2px 0 16px 0; line-height: 1.9; }
.ngec-paperref a {
    font-family: var(--po-mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--po-orange) !important;
    text-decoration: none;
}
.ngec-paperref a:hover { text-decoration: underline; text-underline-offset: 0.2em; }

/* --- sidebar navigation ------------------------------------------------ */

/* The app routes with `st.navigation`, not the handoff's `st.button` sidebar,
   because a demo cited in a response-to-reviewers has to have linkable URLs and
   buttons do not produce them. So the built-in nav is styled to match what the
   buttons would have looked like: flat grey rows, square, active row filled in
   the accent. Selectors are Streamlit's; if an upgrade moves them, re-point
   these rather than changing how the nav works. */
[data-testid="stSidebarNav"] ul { padding: 0; }
[data-testid="stSidebarNav"] li { margin: 0 0 4px 0; list-style: none; }
[data-testid="stSidebarNav"] a {
    border-radius: var(--po-radius) !important;
    background: var(--po-chip);
    padding: 8px 10px;
    text-decoration: none;
}
[data-testid="stSidebarNav"] a:hover { background: #E8E8E5; text-decoration: none; }
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] a p {
    font-family: var(--po-sans) !important;
    font-size: 14px !important;
    font-weight: 400;
    color: var(--po-ink) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--po-orange); }
[data-testid="stSidebarNav"] a[aria-current="page"] span,
[data-testid="stSidebarNav"] a[aria-current="page"] p {
    color: #FFFFFF !important;
    font-weight: 700;
}
/* Section headings ("See it work", "Look inside") are the machine's labels. */
[data-testid="stNavSectionHeader"] {
    font-family: var(--po-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--po-label) !important;
    margin: 18px 0 6px !important;
}

/* Expanders and status blocks: square, hairline, mono summary. */
[data-testid="stExpander"] details {
    border: var(--po-hair);
    border-radius: var(--po-radius);
}
[data-testid="stExpander"] summary p {
    font-family: var(--po-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--po-label) !important;
}
"""


def inject_css() -> None:
    """Apply the theme. Call once at the top of every page.

    Everything goes in **one** `<style>` element, and the string handed to
    `st.markdown` starts at column 0. Both matter: `st.markdown` runs
    `textwrap.dedent(...).strip()` on its argument, and these blocks have no
    common leading indentation, which makes the dedent a no-op for the whole
    string. The `.strip()` then rescues only the *first* line, so a second
    `<style>` further down stays indented, and markdown renders an indented line
    as a code block — the CSS appeared on the page as text.
    """
    handoff = _handoff_css()
    st.markdown(
        f"<style>\n{handoff}\n{_components_css()}\n{_span_css()}\n</style>",
        unsafe_allow_html=True,
    )
    if not handoff:
        st.warning(
            f"The theme stylesheet is missing ({_CSS_PATH}), so this page is "
            "unstyled. The pipeline itself is unaffected."
        )


# --------------------------------------------------------------------------
# small HTML helpers
# --------------------------------------------------------------------------


def kicker(text: str) -> None:
    st.markdown(f'<div class="ngec-kicker">{html.escape(text)}</div>', unsafe_allow_html=True)


def standfirst(text: str) -> None:
    """The one-sentence summary under a page title."""
    st.markdown(f'<div class="ngec-standfirst">{text}</div>', unsafe_allow_html=True)


def rule() -> None:
    st.markdown('<hr class="ngec-rule">', unsafe_allow_html=True)


def note(body: str, title: str | None = None, tone: str = "info") -> None:
    """A bordered aside. tone: info | warn | bad | good."""
    cls = "" if tone == "info" else f" {tone}"
    head = f'<span class="ngec-note-title">{html.escape(title)}</span>' if title else ""
    st.markdown(f'<div class="ngec-note{cls}">{head}{body}</div>', unsafe_allow_html=True)


# --- the design language's own pieces, for pages that want them directly ---


def label(text: str, tone: str = "") -> str:
    """A mono section label. tone: "" | "ink" | "orange"."""
    extra = f" po-label--{tone}" if tone in ("ink", "orange") else ""
    return f'<span class="po-label{extra}">{html.escape(text)}</span>'


def tag(text: str, ink: bool = False) -> str:
    """A solid tag. Orange by default; `ink` makes it black."""
    cls = "po-tag po-tag--ink" if ink else "po-tag"
    return f'<span class="{cls}">{html.escape(text)}</span>'


def num(value) -> str:
    """A boxed numeral, zero-padded to two digits."""
    try:
        text = f"{int(value):02d}"
    except (TypeError, ValueError):
        text = str(value)
    return f'<span class="po-num">{html.escape(text)}</span>'


def statusbar(left: str, right: str = "") -> None:
    """The thin bar above a page title: a tag on the left, meta on the right."""
    st.markdown(
        f'<div class="po-statusbar">{left}'
        f'<span class="po-label">{html.escape(right)}</span></div>',
        unsafe_allow_html=True,
    )


def bar_rows(rows: list[dict], value_key: str = "probability",
             threshold_key: str | None = "threshold",
             name_key: str = "event_type", fired_key: str = "fired",
             value_fmt: str = "{:.2f}") -> str:
    """Horizontal bars with a per-row threshold marker.

    Used for the classifier score panels. This is a table with bars rather than a
    chart proper: with sixteen classes that all carry meaning, the label has to be
    readable on every row. Bars that cleared their threshold take the accent and
    the rest go grey — emphasis, not sixteen categorical hues. Every row shows its
    name and its number as text, which is the relief the grey requires.
    """
    out = []
    for r in rows:
        fired = bool(r.get(fired_key))
        value = float(r[value_key])
        pct = max(0.0, min(1.0, value)) * 100
        marker = ""
        if threshold_key and r.get(threshold_key) is not None:
            tpct = max(0.0, min(1.0, float(r[threshold_key]))) * 100
            marker = (
                f'<span class="ngec-bar-thresh" style="left:{tpct:.1f}%" '
                f'title="threshold {r[threshold_key]:.2f}"></span>'
            )
        out.append(
            f'<div class="ngec-bar-row {"on" if fired else "off"}">'
            f'<span class="ngec-bar-name">{html.escape(str(r[name_key]))}</span>'
            f'<span class="ngec-bar-track">'
            f'<span class="ngec-bar-fill" style="width:{pct:.1f}%"></span>{marker}</span>'
            f'<span class="ngec-bar-val">{value_fmt.format(value)}</span>'
            f"</div>"
        )
    return f'<div class="ngec-bars">{"".join(out)}</div>'


def chips(items, active=()) -> str:
    """Return HTML for a row of small chips; `active` ones are filled black."""
    out = []
    for it in items:
        cls = "on" if it in active else "off"
        out.append(f'<span class="ngec-chip {cls}">{html.escape(str(it))}</span>')
    return "".join(out)


# --------------------------------------------------------------------------
# the document, and checking spans against it
# --------------------------------------------------------------------------


# The `.ngec-doc` panel is what fixed text is shown in — the translation on the
# non-English page, for instance. The document being coded is no longer rendered
# through it: it lives in an editable box instead (`ui._document_box`), because
# inviting a visitor to paste their own text is the demo's main call to action
# and a read-only panel with the paste box elsewhere buried it.
#
# Two earlier versions of that panel are worth not repeating. Extracted spans
# were once highlighted inline with small uppercase tags: it read well on a
# two-line example and badly on a real article, where four spans scattered
# through six paragraphs are hard to collect by eye, a span belonging to one of
# three records cannot be told from the others, and the tags break the line
# spacing of the prose they annotate. The spans are listed in a table instead —
# see `ui.render_attributes`.


# Typographic variants the model swaps freely: it may return a curly apostrophe
# where the article has a straight one, or the reverse. Left unhandled, an
# otherwise perfect span fails to match and the page accuses the model of
# paraphrasing — on this demo's own examples that was three of forty-one spans,
# every one of them a genuine verbatim quote with the wrong apostrophe.
#
# Every mapping here MUST be one character to one character, so that folding a
# string cannot change its length. Nothing here slices the text by offset today,
# but `check_extractions.py` imports this table to compare spans against the
# article, and a substitution that shifted positions (an ellipsis becoming three
# dots, say) would make any offset computed from a folded string wrong.
_TYPOGRAPHIC = str.maketrans({
    "’": "'", "‘": "'",           # curly single quotes
    "“": '"', "”": '"',           # curly double quotes
    "–": "-", "—": "-",           # en and em dash
    " ": " ",                          # non-breaking space
})


def _normalise_typography(text: str) -> str:
    """Fold typographic variants, preserving length and therefore offsets."""
    return text.translate(_TYPOGRAPHIC)


def appears_in(text: str, needle: str) -> bool:
    """Does `needle` occur in `text`, allowing for how the model writes it?

    Matching is case-insensitive, whitespace-tolerant and tolerant of
    typographic variants, because the attribute model returns the span as it
    chose to write it, which is not always character-identical to the article.
    """
    needle = needle.strip()
    if not needle:
        return False
    words = _normalise_typography(needle).split()
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)
    return pattern.search(_normalise_typography(text)) is not None


def unmatched_spans(text: str, spans_by_role: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Spans the model returned that do not appear verbatim in the document."""
    missing = []
    for role, spans in spans_by_role.items():
        for span in spans or []:
            if span and not appears_in(text, span):
                missing.append((role, span))
    return missing


def kv_table(rows) -> str:
    """A definition-list style table. Values are pre-escaped by the caller if HTML.

    A row is `(key, value)`, or `(key, value, role)` where `role` is one of the
    four attribute roles; that row's label then carries the role's colour as a
    rule down its left edge, so the actor row of an extraction table and the
    actor row of the coded record below it are recognisably the same slot.
    """
    body = []
    for row in rows:
        key, value = row[0], row[1]
        role = row[2] if len(row) > 2 else None
        cls = f"k ngec-{role}" if role in SPAN_COLORS else "k"
        shown = value if value not in (None, "", []) else '<span class="none">none</span>'
        body.append(
            f'<tr><td class="{cls}">{html.escape(key)}</td><td class="v">{shown}</td></tr>'
        )
    return f'<table class="ngec-kv">{"".join(body)}</table>'
