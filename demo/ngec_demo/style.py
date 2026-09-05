"""The demo's look, and the two or three widgets every page reuses."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

ACCENT = "#F97316"
MUTED = "#5B5F66"
INK = "#E6E6E6"

_CSS = f"""
<style>
  h1, h2, h3 {{
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    letter-spacing: -0.02em;
  }}
  h1 {{ font-size: 2.1rem; margin-bottom: 0.1rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 1.6rem; }}
  h3 {{ font-size: 1.0rem; }}
  .block-container {{ padding-top: 2.6rem; max-width: 60rem; }}
  .lede {{ color: {MUTED}; margin: 0 0 1.4rem 0; }}
  .stButton > button {{ font-size: 0.82rem; padding: 0.25rem 0.7rem; }}
  code, .stCode {{ font-size: 0.82rem; }}
  [data-testid="stSidebar"] {{ font-size: 0.8rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
</style>
"""


def inject_css() -> None:
    """Monospace headings, a narrower column, smaller buttons."""
    st.markdown(_CSS, unsafe_allow_html=True)


def lede(text: str) -> None:
    """The one grey sentence under a page title."""
    st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def hbar_chart(df: pd.DataFrame, label_col: str, value_col: str,
               threshold_col: str | None = None, height: int | None = None):
    """Horizontal bars sorted by value, accent for rows over their threshold.

    When `threshold_col` is given, each bar also gets a tick at its own
    threshold, because the classifier's thresholds are per class -- a 0.4 that
    fires and a 0.6 that does not is the interesting case, and a single bar
    length cannot show it.
    """
    df = df.copy()
    if threshold_col:
        df["_fired"] = df[value_col] >= df[threshold_col]
    else:
        df["_fired"] = True

    height = height or max(120, 22 * len(df))
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label_col}:N", sort="-x", title=None,
                axis=alt.Axis(labelFontSize=11)),
    )
    bars = base.mark_bar(height=12, cornerRadiusEnd=2).encode(
        x=alt.X(f"{value_col}:Q", title=None, scale=alt.Scale(domain=[0, 1])),
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
    """The full record, behind an expander, already JSON-sanitised."""
    from .steps import jsonable

    with st.expander(label, expanded=expanded):
        st.json(jsonable(obj), expanded=False)


STEPS = [
    ("pages/step1.py", "1. Which event?", "Sixteen event types, one classifier each."),
    ("pages/step2.py", "2. Who did what?", "A fine-tuned model pulls out the spans."),
    ("pages/step3.py", "3. Which entity?", "Each span is matched to a Wikipedia article."),
    ("pages/step4.py", "4. What kind of actor?", "The article becomes role and country codes."),
    ("pages/step5.py", "5. When and where?", "Dates and places are resolved to real values."),
]


def step_links() -> None:
    """The five step pages, as links."""
    for path, label, blurb in STEPS:
        st.page_link(path, label=f"{label} — {blurb}")


def health_sidebar(health: dict, notes: list[str] | None = None) -> None:
    """A compact status block: one line per dependency, green or orange."""
    st.sidebar.markdown("**Status**")
    for name, row in health.items():
        mark = "🟢" if row.get("ok") else "🟠"
        st.sidebar.caption(f"{mark} {name} · {row.get('detail', '')}")
    for note in notes or []:
        st.sidebar.caption(f"ℹ️ {note}")
