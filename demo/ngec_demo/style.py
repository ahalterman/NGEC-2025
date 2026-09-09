"""The demo's look, and the two or three widgets every page reuses.

The look is the "Pocket Operator" design the earlier demo used: white paper,
grey chips, black hairlines, and exactly one orange. Nothing is rounded and
nothing glows; separation comes from 1px rules, never from shadows. Body copy is
Space Grotesk; IBM Plex Mono in small uppercase is reserved for what the machine
says about itself -- section labels, units, timings, status.

The one accent is a signal rather than a colour scheme: it marks the thing to
look at (a classifier that fired, the primary button), and never two kinds of
thing in the same view.
"""

from __future__ import annotations

import html
from contextlib import contextmanager

import altair as alt
import pandas as pd
import streamlit as st

from . import resources as R

# The palette, mirrored as CSS variables below. Charts and status dots build
# colours in Python, everything else reads the variables.
PAPER = "#FFFFFF"
CHIP = "#F2F2F0"      # grey chip: inputs' fill, code blocks, quote panels
HAIR = "#E6E6E4"      # soft hairline
INK = "#111111"       # headings, hard rules, "this is fine" status
BODY = "#2A2A28"      # body copy
MUTED = "#6C6C68"     # captions, secondary copy, bars that did not fire
LABEL = "#8E8E8A"     # mono uppercase labels
ACCENT = "#F0521E"    # the one accent
ACCENT_HOVER = "#D9451A"
OK = "#2E7D53"        # the status green: only ever a dot, never text

# The Google Fonts import has to be the first thing in the stylesheet: CSS drops
# an @import that follows any style rule, and the two webfaces then fall back to
# Helvetica without saying so.
_CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap");

:root {{
  --po-paper: {PAPER};
  --po-chip: {CHIP};
  --po-hair: {HAIR};
  --po-ink: {INK};
  --po-body: {BODY};
  --po-mute: {MUTED};
  --po-label: {LABEL};
  --po-accent: {ACCENT};
  --po-ok: {OK};
  --po-sans: "Space Grotesk", "Helvetica Neue", Helvetica, sans-serif;
  --po-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}

html, body, .stApp, [class*="css"] {{
  background: var(--po-paper);
  color: var(--po-body);
  font-family: var(--po-sans);
}}
.stApp a {{ color: var(--po-accent); text-decoration: none; }}
.stApp a:hover {{ color: var(--po-ink); text-decoration: underline; }}

.block-container {{ padding-top: 2.6rem; max-width: 60rem; }}

h1, h2, h3 {{
  font-family: var(--po-sans);
  font-weight: 700;
  color: var(--po-ink);
}}
h1 {{ font-size: 2.4rem; letter-spacing: -0.04em; margin-bottom: 0.2rem; }}
h2 {{ font-size: 1.35rem; letter-spacing: -0.03em; margin-top: 1.8rem; }}
h3 {{ font-size: 1.0rem; letter-spacing: -0.01em; }}
p, li {{ color: var(--po-body); }}
.lede {{ color: var(--po-mute); margin: 0 0 1.4rem 0; }}

/* Buttons: square, hairline, black on paper; the accent is spent on the one
   primary button per page. */
.stButton > button {{
  font-family: var(--po-sans); font-weight: 700; font-size: 0.82rem;
  color: var(--po-ink); background: var(--po-paper);
  border: 1px solid var(--po-ink); border-radius: 0;
  padding: 0.3rem 0.8rem; box-shadow: none;
}}
.stButton > button:hover {{ background: var(--po-chip); color: var(--po-ink); }}
/* A disabled button should read as disabled: the "Load models" button stays on
   screen once the models are up, and at full contrast it looked clickable. */
.stButton > button:disabled, .stButton > button:disabled:hover {{
  color: var(--po-mute); border-color: var(--po-hair);
  background: var(--po-paper); opacity: 0.55;
}}
.stButton > button[kind="primary"] {{
  background: var(--po-accent); color: #FFFFFF; border-color: var(--po-accent);
}}
.stButton > button[kind="primary"]:hover {{
  background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; color: #FFFFFF;
}}

/* Inputs: mono, square, hairline; the label above them is a machine label. */
.stTextInput input, .stTextArea textarea, .stDateInput input,
.stNumberInput input {{
  font-family: var(--po-mono); font-size: 0.82rem; color: var(--po-ink);
  background: var(--po-paper); border-radius: 0 !important;
  box-shadow: none !important;
}}
.stTextInput div[data-baseweb="input"], .stTextArea div[data-baseweb="base-input"],
.stDateInput div[data-baseweb="input"] {{
  border: 1px solid var(--po-ink) !important; border-radius: 0 !important;
}}
.stTextInput label p, .stTextArea label p, .stDateInput label p,
.stRadio label p, .stSelectbox label p {{
  font-family: var(--po-mono) !important; font-size: 0.65rem !important;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--po-label) !important;
}}

/* Expanders: square, hairline, mono summary. */
[data-testid="stExpander"] details {{
  border: 1px solid var(--po-hair); border-radius: 0; background: var(--po-paper);
}}
[data-testid="stExpander"] summary p {{
  font-family: var(--po-mono) !important; font-size: 0.65rem !important;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--po-label) !important;
}}

/* Tables and code: mono, hairline, unrounded. */
[data-testid="stDataFrame"], [data-testid="stTable"], .stCodeBlock, pre {{
  border-radius: 0 !important;
}}
code, pre, .stCode, [data-testid="stJson"] {{
  font-family: var(--po-mono) !important; font-size: 0.78rem !important;
}}
.stCodeBlock, pre {{ background: var(--po-chip) !important; }}

hr, [data-testid="stDivider"] {{
  border: none; border-top: 1px solid var(--po-ink); margin: 1.6rem 0;
}}

[data-testid="stSidebar"] {{
  background: var(--po-paper);
  border-right: 1px solid var(--po-ink);
  font-size: 0.8rem;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
  font-family: var(--po-mono); font-size: 0.68rem; color: var(--po-mute);
}}
[data-testid="stSidebar"] strong {{
  font-family: var(--po-mono); font-size: 0.65rem;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--po-label);
}}

[data-testid="stMetricValue"] {{
  font-size: 1.3rem; font-weight: 700; letter-spacing: -0.03em; color: var(--po-ink);
}}
[data-testid="stMetricLabel"] p {{
  font-family: var(--po-mono); font-size: 0.65rem;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--po-label);
}}

/* Status dot: a filled square, green when the dependency is up and orange when
   it is not. The whole block was grey, so "Elasticsearch is down" looked the
   same as everything else at a glance. Only the dot is coloured -- the caption
   beside it still says in words which state it is in, so the block reads the
   same to someone who cannot tell the two colours apart. */
.dot {{
  display: inline-block; width: 7px; height: 7px; margin-right: 0.45em;
  background: var(--po-ok); vertical-align: 0.05em;
}}
.dot.down {{ background: var(--po-accent); }}

/* The per-event field tables on the results page. */
.fields {{ border-collapse: collapse; width: 100%; margin: 0 0 1.1rem 0; }}
.fields td {{
  border-top: 1px solid var(--po-hair); padding: 0.34rem 0.5rem 0.34rem 0;
  vertical-align: top; font-size: 0.86rem; color: var(--po-body);
}}
.fields td.name {{
  font-family: var(--po-mono); font-size: 0.62rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--po-label);
  /* Wide enough for the longest name the table has ("Recipient span") at this
     size and letter-spacing, and told not to wrap in any case: a field name
     broken over two lines pushed its value out of line with the rest. */
  width: 10.5rem; padding-top: 0.5rem; white-space: nowrap;
}}
.event-head {{
  font-family: var(--po-mono); font-size: 0.68rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--po-ink);
  border-bottom: 1px solid var(--po-ink); padding-bottom: 0.3rem;
  margin: 1.2rem 0 0 0;
}}
.event-head .mode {{ color: var(--po-accent); }}
/* A code with a tooltip: the dotted rule is the only hint that hovering it
   says what the three letters mean. */
.fields span.code {{ border-bottom: 1px dotted var(--po-label); cursor: help; }}

/* The coded fields on the actor page. The role code and the country code are
   the answer the page exists to give, so they are ink and large; the secondary
   code, which is usually empty, is kept at body size so it does not compete. */
.codes {{ display: flex; flex-wrap: wrap; gap: 2.4rem; margin: 0.2rem 0 0.6rem 0; }}
/* A fixed slot width so two of these rows stacked -- the bundled dictionary and
   yours -- line their codes up under each other. */
.codes > div {{ min-width: 7rem; }}
.codes .key {{
  font-family: var(--po-mono); font-size: 0.62rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--po-label);
}}
.codes .val {{
  font-family: var(--po-sans); font-size: 1.1rem; font-weight: 500;
  letter-spacing: -0.02em; color: var(--po-body); margin-top: 0.15rem;
}}
.codes .val.lead {{ font-size: 1.9rem; font-weight: 700; color: var(--po-ink); }}
.codes .val.none {{ color: var(--po-mute); font-weight: 400; }}

/* A code among the ones the resolver weighed. The one it chose is the point of
   the list, so it is the only one at full contrast. */
.chip {{
  display: inline-block; font-family: var(--po-mono); font-size: 0.76rem;
  padding: 0.05rem 0.4rem; margin: 0 0.3rem 0.2rem 0;
  border: 1px solid var(--po-hair); background: var(--po-chip);
  color: var(--po-mute);
}}
.chip.win {{
  border-color: var(--po-ink); background: var(--po-paper);
  color: var(--po-ink); font-weight: 600;
}}

/* The five step links at the foot of the home page. `st.page_link` is a bare
   anchor, so the list read as body copy; a hairline box and a chip on hover
   make it look like a row to click. It stays a page_link rather than an <a
   href="/step1"> so navigation happens inside the app, without a reload. The
   -1px margin collapses each row's border into the one above it. */
.st-key-step_links [data-testid="stPageLink"] {{ margin-bottom: -1px; }}
.st-key-step_links [data-testid="stPageLink-NavLink"] {{
  display: block; width: 100%; border: 1px solid var(--po-hair); border-radius: 0;
  background: var(--po-paper); padding: 0.5rem 0.75rem;
}}

.st-key-step_links [data-testid="stPageLink-NavLink"]:hover {{
  background: var(--po-chip); border-color: var(--po-ink);
  text-decoration: none;
}}
/* The blurb is whatever element Streamlit wraps the label in, so all three are
   named rather than guessing which one carries the text. */
.st-key-step_links [data-testid="stPageLink-NavLink"],
.st-key-step_links [data-testid="stPageLink-NavLink"] p,
.st-key-step_links [data-testid="stPageLink-NavLink"] span {{
  margin: 0; font-size: 0.86rem; color: var(--po-mute);
}}
.st-key-step_links [data-testid="stPageLink-NavLink"] strong {{
  color: var(--po-ink); font-weight: 700;
}}
</style>
"""


def inject_css() -> None:
    """Apply the theme. `app.py` calls it once, before the page runs."""
    st.markdown(_CSS, unsafe_allow_html=True)


def lede(text: str) -> None:
    """The one grey sentence under a page title."""
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def demo_model_note() -> None:
    """The one line saying the bundled classifiers are not the POLECAT models.

    The classifier says the same thing at load time and the sidebar carries that
    warning, but the sidebar is peripheral: a reader looking at coded output
    should not have to go find it.
    """
    st.caption("\\* The bundled event classifiers are demonstration models for "
               "the PLOVER ontology, not the production models behind POLECAT.")


@contextmanager
def running(mode: str, message: str):
    """Run a step, saying what is happening while it runs.

    Once the models are up this is just the step's spinner. The first time,
    though, the click also pays for a minute of model loading, and a spinner
    that says "coding the document" while nothing happens for ninety seconds
    reads as a hang. So a cold process gets a status with two labelled phases --
    the loading, then the step -- and the wait is at least explained.
    """
    if R.is_loaded(mode):
        with st.spinner(message):
            yield None
        return
    with st.status("Loading models (about a minute, once per session)") as status:
        R.load_all(mode, on_step=lambda name: status.update(label=f"Loading models — {name}"))
        status.update(label=message)
        yield status
        status.update(label=f"Models loaded in {R.load_seconds(mode):.0f} s",
                      state="complete")


def hbar_chart(df: pd.DataFrame, label_col: str, value_col: str,
               threshold_col: str | None = None, height: int | None = None,
               muted: bool = False):
    """Horizontal bars sorted by value, accent for rows over their threshold.

    When `threshold_col` is given, each bar also gets a tick at its own
    threshold, because the classifier's thresholds are per class -- a 0.4 that
    fires and a 0.6 that does not is the interesting case, and a single bar
    length cannot show it.

    `muted` greys every bar whatever its threshold. Step 1 uses it for the modes
    of a type that did not fire: those scores are real, but the accent means
    "this fired", and spending it on a mode under a silent parent says the
    opposite of what happened.

    The default height gives every row about 24px and the axis is told not to
    thin its labels out: sixteen event types in a short chart is exactly the
    case where Altair starts dropping every other name, and a bar whose label is
    missing is worse than no chart.
    """
    df = df.copy()
    if muted:
        df["_fired"] = False
    elif threshold_col:
        df["_fired"] = df[value_col] >= df[threshold_col]
    else:
        df["_fired"] = True

    height = height or 24 * len(df) + 24
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label_col}:N", sort="-x", title=None,
                axis=alt.Axis(labelFontSize=11, labelOverlap=False,
                              labelLimit=400, labelColor=INK,
                              labelFont="IBM Plex Mono, monospace",
                              domainColor=INK, tickColor=HAIR)),
    )
    bars = base.mark_bar(height=12).encode(
        x=alt.X(f"{value_col}:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelColor=MUTED, domainColor=HAIR, tickColor=HAIR,
                              gridColor=HAIR)),
        color=alt.condition("datum._fired", alt.value(ACCENT), alt.value(MUTED)),
        tooltip=[label_col, alt.Tooltip(f"{value_col}:Q", format=".3f")],
    )
    if threshold_col:
        ticks = base.mark_tick(color=INK, thickness=1.5, size=16).encode(
            x=alt.X(f"{threshold_col}:Q"),
            tooltip=[alt.Tooltip(f"{threshold_col}:Q", title="threshold", format=".2f")],
        )
        chart = bars + ticks
    else:
        chart = bars
    return chart.properties(height=height).configure_view(strokeWidth=0)


def json_block(obj, expanded: bool = False, label: str = "Raw JSON") -> None:
    """The full record, behind an expander, already JSON-sanitised.

    The expander stays shut, but the JSON inside it is fully unfolded: someone
    who opened it wants to read the record, not click through its nesting.
    """
    from .steps import jsonable

    with st.expander(label, expanded=expanded):
        st.json(jsonable(obj), expanded=True)


# PLOVER's actor role codes, in the order `ngec/assets/PLOVER_priorities.csv`
# lists them -- that file is the authoritative list of the codes the resolver
# can emit, but it holds no names, so the names are written here.
#
# The ones marked (agents) were checked against the patterns filed under each
# code in `ngec/assets/PLOVER_agents.txt`, which is what the agent matcher
# actually assigns; MED is the one that matters, since it reads like "media"
# and is in fact doctors and nurses (media is JRN). The rest carry no patterns
# -- they come from the Wikipedia side, whose training data is outside this
# repo -- so their names follow the PLOVER/CAMEO manual and are marked
# (unverified here).
ROLE_CODES = {
    "IGO": "intergovernmental organisation",   # (agents)
    "ISM": "international social movement",    # (unverified here)
    "IMG": "international militarised group",  # (unverified here)
    "PRE": "unrecognised or partially recognised state",  # (agents)
    "REB": "rebels",                           # (agents)
    "SPY": "intelligence services",            # (agents)
    "JUD": "judiciary",                        # (agents)
    "OPP": "political opposition",             # (unverified here)
    "GOV": "government",                       # (agents)
    "LEG": "legislature",                      # (agents)
    "MIL": "military",                         # (agents)
    "COP": "police",                           # (agents)
    "PRM": "private or paramilitary force",    # (agents)
    "ELI": "elites",                           # (agents)
    "PTY": "political party",                  # (agents)
    "BUS": "business",                         # (agents)
    "NON": "not an actor",                     # (agents)
    "JNK": "not an actor: weather or disaster",  # (agents)
    "UAF": "unidentified armed force",         # (agents)
    "CRM": "criminals",                        # (agents)
    "LAB": "labour",                           # (agents)
    "MED": "medical personnel",                # (agents)
    "NGO": "NGO",                              # (agents)
    "SOC": "social movement or civil society",  # (agents)
    "EDU": "education",                        # (agents)
    "JRN": "journalists and media",            # (agents)
    "ENV": "environmental",                    # (unverified here)
    "HRI": "human rights",                     # (unverified here)
    "UNK": "unknown",                          # (agents)
    "REF": "refugees and displaced people",    # (agents)
    "AGR": "agriculture",                      # (agents)
    "RAD": "radical or extremist",             # (unverified here)
    "CVL": "civilians",                        # (agents)
    "BUD": "Buddhist",                         # (unverified here)
    "CHR": "Christian",                        # (unverified here)
    "HIN": "Hindu",                            # (unverified here)
    "JEW": "Jewish",                           # (unverified here)
    "MUS": "Muslim",                           # (unverified here)
    "REL": "religious",                        # (agents)
    "EUR": "European Union",                   # (unverified here)
    "MNC": "multinational corporation",        # (unverified here)
    "UNO": "United Nations",                   # (unverified here)
}


def role_codes_in(text: str) -> list[str]:
    """The role codes inside a coded string, in order and without repeats.

    A code cell is the country and the role run together ("FRAGOV"), and a role
    can itself be a category plus a modifier ("CVLOPP"), so the string is read
    three characters at a time and every chunk that is a known role is kept.
    """
    found = []
    for token in str(text).replace(";", " ").split():
        for start in range(0, len(token) - 2, 3):
            chunk = token[start:start + 3]
            if chunk in ROLE_CODES and chunk not in found:
                found.append(chunk)
    return found


def code_glossary(codes: list[str]) -> str:
    """"CVL civilians · GOV government" -- the codes present, expanded."""
    return " · ".join(f"{code} {ROLE_CODES[code]}" for code in codes
                      if code in ROLE_CODES)


def field_table(fields: list[tuple[str, str]]) -> None:
    """One event as a two-column table: the field name, then its value.

    A code cell gets its expansion as a tooltip: "BLRGOV" is unreadable on
    sight, and the codes are the part of the record a newcomer stalls on.
    """
    rows = []
    for name, value in fields:
        cell = html.escape(value)
        glossary = code_glossary(role_codes_in(value)) if name.endswith("code") else ""
        if glossary:
            cell = f'<span class="code" title="{html.escape(glossary, quote=True)}">{cell}</span>'
        rows.append(f'<tr><td class="name">{html.escape(name)}</td>'
                    f"<td>{cell}</td></tr>")
    st.markdown(f'<table class="fields">{"".join(rows)}</table>',
                unsafe_allow_html=True)


def event_heading(index: int, event_type: str = "", event_mode: str = "") -> None:
    """"Event 1 · PROTEST · demonstrate", or just "Event 1" on its own.

    One story can produce several records that differ only in their mode, so
    wherever the type and mode are not shown anywhere else they belong here:
    without the mode two such records look like the same event twice. The
    results table now carries both as its first two rows, so a caller with a
    table under the heading asks for the number alone.
    """
    parts = [f"Event {index}"]
    if event_type:
        parts.append(html.escape(event_type))
    if event_mode:
        parts.append(f'<span class="mode">{html.escape(event_mode)}</span>')
    st.markdown(f'<p class="event-head">{" · ".join(parts)}</p>',
                unsafe_allow_html=True)


def code_metrics(items: list[tuple[str, str]], lead: tuple[str, ...] = ()) -> None:
    """A row of coded fields: the field name, then its code.

    The names in `lead` are the codes the page is really answering with, and
    they are set large and in ink; the rest stay at body size. `st.metric` drew
    all of them at one weight, which left the answer no more visible than the
    field beside it that came back empty.
    """
    cells = []
    for name, value in items:
        classes = "val lead" if name in lead else "val"
        if not value or value == "—":
            value, classes = "—", classes + " none"
        cells.append(f'<div><div class="key">{html.escape(name)}</div>'
                     f'<div class="{classes}">{html.escape(value)}</div></div>')
    st.markdown(f'<div class="codes">{"".join(cells)}</div>',
                unsafe_allow_html=True)


def code_chips(codes: list[str], winner: str = "") -> str:
    """The codes the resolver considered, with the one it chose picked out."""
    return "".join(
        f'<span class="chip{" win" if code and code == winner else ""}">'
        f"{html.escape(code)}</span>" for code in codes)


def mode_badge(mode: str | None) -> str:
    """The mode as the captions show it: "GPU" or "CPU"."""
    return {"gpu": "GPU", "cpu": "CPU"}.get(mode or "", str(mode or "—"))


def _with_remainder(rows: list[dict] | None) -> list[dict]:
    """The timing rows plus a "(other)" row wherever children miss the parent.

    A parent's seconds is its own wall time, so the children of an instrumented
    call rarely add up to it; the gap is real work (tokenising, python glue) and
    hiding it makes the sub-rows look like the whole story.
    """
    rows = [dict(r) for r in rows or []]
    by_path = {r["path"]: r for r in rows}
    spent: dict[str, float] = {}
    last: dict[str, int] = {}
    for i, row in enumerate(rows):
        parts = str(row["path"]).split("/")
        if len(parts) > 1:
            parent = "/".join(parts[:-1])
            spent[parent] = spent.get(parent, 0.0) + float(row["seconds"])
        for k in range(1, len(parts)):
            last["/".join(parts[:k])] = i

    extra: dict[int, list[dict]] = {}
    for parent, used in spent.items():
        row = by_path.get(parent)
        if row is None:
            continue
        rest = float(row["seconds"]) - used
        if rest > 0.005:
            extra.setdefault(last[parent], []).append(
                {"path": f"{parent}/(other)", "label": "(other)",
                 "depth": int(row["depth"]) + 1, "seconds": rest, "calls": 0})

    out: list[dict] = []
    for i, row in enumerate(rows):
        out.append(row)
        out += sorted(extra.get(i, []), key=lambda r: -r["depth"])
    return out


def _label(row: dict) -> str:
    return "  " * int(row["depth"]) + str(row["label"])


def timing_table(rows: list[dict] | None = None,
                 columns: dict[str, list[dict]] | None = None) -> None:
    """The timing tree as a table: the path indented by depth, seconds to 3 dp.

    `columns` is {mode: rows} and gives one seconds column per mode plus the
    cpu/gpu ratio, for the side-by-side on the Timing page.
    """
    if columns:
        tables = {mode: {r["path"]: r for r in _with_remainder(rs)}
                  for mode, rs in columns.items()}
        both = "gpu" in tables and "cpu" in tables
        # The union of the paths, sorted by where each path's ancestors were
        # first seen, so a row that only one mode has (prefill and decode are
        # CPU-only) sits under its parent rather than at the bottom, and
        # "(other)" stays last among its siblings.
        seen: list[str] = []
        for table in tables.values():
            seen += [p for p in table if p not in seen]
        rank = {p: i for i, p in enumerate(seen)}

        def key(path: str):
            parts = path.split("/")
            return tuple((parts[i] == "(other)", rank["/".join(parts[: i + 1])])
                         for i in range(len(parts)))

        order = sorted(seen, key=key)
        data = []
        for path in order:
            row = next(t[path] for t in tables.values() if path in t)
            entry = {"step": _label(row)}
            for mode, table in tables.items():
                hit = table.get(path)
                entry[mode_badge(mode)] = (round(float(hit["seconds"]), 3)
                                           if hit else None)
            if both:
                gpu, cpu = tables["gpu"].get(path), tables["cpu"].get(path)
                entry["×"] = (round(float(cpu["seconds"]) / float(gpu["seconds"]), 1)
                              if gpu and cpu and float(gpu["seconds"]) > 0 else None)
            data.append(entry)
        st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")
        return

    rows = _with_remainder(rows)
    if not rows:
        st.caption("Nothing was timed on this run.")
        return
    data = [{"step": _label(r),
             "seconds": round(float(r["seconds"]), 3),
             "calls": str(r["calls"]) if int(r.get("calls") or 0) > 1 else ""}
            for r in rows]
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")


STEPS = [
    ("pages/step1.py", "1. Which event?", "Sixteen event types, one classifier each."),
    ("pages/step2.py", "2. Who did what?", "A fine-tuned model pulls out the spans."),
    ("pages/step3.py", "3. Which entity?", "Each span is matched to a Wikipedia article."),
    ("pages/step4.py", "4. What kind of actor?", "The article becomes role and country codes."),
    ("pages/step5.py", "5. When and where?", "Dates become real dates; place names become geonames entries."),
]


def step_links() -> None:
    """The five step pages, as boxed rows that look like links.

    The container's key is what the stylesheet hooks on (Streamlit turns it into
    a `st-key-step_links` class), so the boxes are confined to this list and the
    sidebar's own navigation is left alone. The label is bold and the blurb is
    not, which the stylesheet then colours: the step is the link, the blurb is
    the explanation.
    """
    with st.container(key="step_links"):
        for path, label, blurb in STEPS:
            st.page_link(path, label=f"**{label}** — {blurb}", width="stretch")


def load_models_button(mode: str) -> None:
    """The "Load models" button, and what it says once it is done.

    The minute of loading has to happen somewhere. Offering it here lets a
    visitor start it while they read the page, instead of discovering it under
    their first click.

    Drawn into whatever container the caller is in rather than straight into
    `st.sidebar`: `app.py` reserves a slot for it, runs the page, and fills the
    slot afterwards, so the button reflects whether the page just loaded the
    models.
    """
    if R.is_loaded(mode):
        st.button("Load models", disabled=True, width="stretch")
        st.caption(f"Models loaded · {R.load_seconds(mode):.0f} s")
        return
    if st.button("Load models", width="stretch"):
        with st.status("Loading models (about a minute, once per process)") as status:
            seconds = R.load_all(
                mode, on_step=lambda name: status.update(label=f"Loading {name}"))
            status.update(label=f"Loaded in {seconds:.0f} s", state="complete")
        # No rerun: the finished status, with the component it ended on, is
        # worth leaving on screen.


def health_sidebar(health: dict, notes: list[str] | None = None,
                   mode: str | None = None) -> None:
    """A compact status block: one line per dependency, a dot for its state.

    With a `mode`, the block also says whether that mode's models are loaded,
    so the state the "Load models" button changes is visible on every page.
    Like the button above, this draws into the caller's container so that
    `app.py` can fill it in after the page has run.
    """
    st.markdown("**Status**")
    rows = dict(health)
    if mode is not None:
        rows["models"] = {"ok": R.is_loaded(mode),
                          "detail": "loaded" if R.is_loaded(mode) else "not loaded"}
    for name, row in rows.items():
        dot = '<span class="dot"></span>' if row.get("ok") else '<span class="dot down"></span>'
        st.caption(f"{dot}{name} · {row.get('detail', '')}",
                   unsafe_allow_html=True)
    for note in notes or []:
        # The classifier's own warning begins "... loaded. NOTE: these models
        # are not ...", and this block adds a NOTE of its own; two in one line
        # read as a stutter, so the inner one goes.
        st.caption("NOTE · " + str(note).replace("NOTE: ", ""))
