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
