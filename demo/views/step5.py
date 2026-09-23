"""Step 5: when and where? A date phrase against the publication date, places
against geonames."""

import datetime as dt

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import (field_table, json_block, lede, mode_badge, running,
                             timing_table)

# (phrase, publication date) -- a weekday, a relative offset, an open-ended
# range and a bounded one, which are the four shapes the resolver has to tell
# apart. The last is published on a Friday, so "Monday" and "Wednesday" both
# fall earlier in the same week.
DATE_EXAMPLES: list[tuple[str, str]] = [
    ("last Tuesday", "2023-03-15"),
    ("three weeks ago", "2024-06-11"),
    ("since early 2015", "2024-02-20"),
    ("between Monday and Wednesday", "2024-06-14"),
]

# (button label, text). The second is the interesting one: "Paris" and
# "Springfield" are only resolvable from the states named around them.
GEO_EXAMPLES: list[tuple[str, str]] = [
    ("Nairobi",
     "Police in Nairobi arrested at least forty people on Saturday during a "
     "demonstration outside parliament, the interior ministry said. Similar "
     "protests were held in Mombasa and Kisumu."),
    ("Paris, Texas",
     "The city council in Paris, Texas voted on Tuesday to expand the water "
     "treatment plant, after a delegation from Springfield, Illinois toured the "
     "site last month."),
    ("Cairo",
     "Officials from Ethiopia and Eritrea signed a ceasefire agreement in Cairo "
     "on Monday, ending three weeks of fighting along their shared border."),
]

st.title("5. When and where?")
lede("A date phrase from the story is resolved against the story's publication "
     "date; place names are resolved against geonames and matched to the "
     "event's location span.")

health = R.health(R.current_mode())
missing = [name for name in ("Elasticsearch", "geonames index")
           if not health.get(name, {}).get("ok")]
if missing:
    st.warning(f"Not available: {', '.join(missing)}. Geoparsing needs the "
               "geonames index; date resolution does not.")

st.subheader("5.1 Dates")

if "s5_phrase" not in st.session_state:
    st.session_state.s5_phrase = DATE_EXAMPLES[0][0]
    st.session_state.s5_pub_text = "today"

cols = st.columns(len(DATE_EXAMPLES))
for col, (example_phrase, example_date) in zip(cols, DATE_EXAMPLES):
    if col.button(example_phrase, width="stretch", key=f"s5_date_{example_phrase}"):
        st.session_state.s5_phrase = example_phrase
        st.session_state.s5_pub_text = example_date
        st.rerun()

left, right = st.columns(2)
phrase = left.text_input("Date phrase", key="s5_phrase")
pub_text = right.text_input(
    "Published", key="s5_pub_text",
    help="A date or a phrase: 2024-06-11, June 11, 2024, 11 June 2024, "
         "today, yesterday, last Friday.")


def plain_english(phrase: str, pub_date: str, result: dict) -> str:
    """The result as one sentence, for readers who won't read the table."""
    published = dt.date.fromisoformat(pub_date)
    when = f"{pub_date} ({published:%A})"
    resolved = (result["resolved_date"] or "")[:10]
    if result["date_type"] == "unresolved" or not resolved:
        return (f"“{phrase}” relative to {when} → not resolved; "
                "the publication date is used instead.")
    span = resolved
    if result["date_end"]:
        span = f"{resolved} to {result['date_end'][:10]}"
    detail = ", ".join(part for part in
                       (result["date_type"],
                        f"{result['granularity']} granularity"
                        if result["granularity"] else "") if part)
    return f"“{phrase}” relative to {when} → {span}, {detail}."


pub_date = steps.parse_pub_date(pub_text)
if pub_date is None:
    st.info(f"Couldn't read “{pub_text}” as a publication date. Try a date "
            "like 2024-06-11, or a phrase like “yesterday”.")
else:
    # What the typed text was taken to mean, since "today" and "last Friday"
    # are only anchored once they are read.
    published = dt.date.fromisoformat(pub_date)
    st.caption(f"Published {pub_date} ({published:%A}).")

if pub_date is not None and phrase.strip():
    # No models and no network: cheap enough to redo on every keystroke.
    date_result = steps.resolve_date(phrase, pub_date)
    # Fixed narrow columns for the four short fields: left to itself the table
    # gave the space to the widest value and truncated the dates to "2026-09-0".
    # The resolver's reason is a sentence, so it goes under the table instead of
    # squeezing the columns beside it.
    st.dataframe(pd.DataFrame([{
        "resolved": (date_result["resolved_date"] or "—")[:10],
        "end": (date_result["date_end"] or "—")[:10],
        "granularity": date_result["granularity"] or "—",
        "type": date_result["date_type"] or "—",
    }]), hide_index=True, width="stretch",
        column_config={name: st.column_config.TextColumn(name, width="small")
                       for name in ("resolved", "end", "granularity", "type")})
    st.markdown(plain_english(phrase, pub_date, date_result))
    if date_result["reason"]:
        st.caption(f"Why: {date_result['reason']}")

# --- 5.2 places --------------------------------------------------------------

st.subheader("5.2 Places")

if "s5_geo_text" not in st.session_state:
    st.session_state.s5_geo_text = GEO_EXAMPLES[0][1]

cols = st.columns(len(GEO_EXAMPLES))
for col, (label, example_text) in zip(cols, GEO_EXAMPLES):
    if col.button(label, width="stretch", key=f"s5_geo_{label}"):
        st.session_state.s5_geo_text = example_text
        st.session_state.pop("s5_geo_result", None)
        # A pick made from the previous text would otherwise sit under the new
        # places, reading as a result of the new run.
        st.session_state.pop("s5_pick", None)
        st.rerun()

st.text_area("Text", key="s5_geo_text", height=110)
if st.button("Geoparse", type="primary"):
    with running(R.current_mode(), "Finding and resolving place names…"):
        st.session_state["s5_geo_result"] = steps.geoparse(
            st.session_state.s5_geo_text)
    st.session_state.pop("s5_pick", None)
    entities = st.session_state["s5_geo_result"]["entities"]
    if entities:
        # 5.3 starts from the first place found, which is what the attribute
        # model's location span usually points at.
        st.session_state["s5_span"] = entities[0]["search_name"]

geo_result = st.session_state.get("s5_geo_result")
if geo_result:
    entities = geo_result["entities"]
    if entities:
        # Explicit widths: left to itself the table gives its space to the
        # longest value and truncates the resolved name, which is the column
        # the whole step is about.
        table = pd.DataFrame([{
            "span": e["search_name"],
            "resolved place": e["resolved_placename"],
            "admin1": e["admin1_name"],
            "country": e["country_code3"],
            "feature code": e["feature_code"],
            "score": e["score"],
        } for e in entities])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "span": st.column_config.TextColumn("span", width="small"),
            "resolved place": st.column_config.TextColumn("resolved place",
                                                          width="medium"),
            "admin1": st.column_config.TextColumn("admin1", width="medium"),
            "country": st.column_config.TextColumn("country", width="small"),
            "feature code": st.column_config.TextColumn("feature code",
                                                        width="small"),
            "score": st.column_config.NumberColumn("score", width="small",
                                                   format="%.2f"),
        })
    else:
        st.info("No place names were resolved in this text.")
    st.caption(f"{geo_result['seconds']:.1f} s · {mode_badge(geo_result['mode'])} · "
               f"{len(entities)} place(s)")
    json_block(entities, label="Raw JSON")
    with st.expander("Timing breakdown"):
        timing_table(geo_result["timing"])

# --- 5.3 event location ------------------------------------------------------

st.subheader("5.3 Event location")

if not geo_result:
    st.caption("Run the geoparser above first — this step chooses among its results.")
else:
    span = st.text_input(
        "Location span", key="s5_span",
        help="The location the attribute model extracted, matched by word "
             "overlap against the geoparsed place names.")
    if st.button("Pick location"):
        st.session_state["s5_pick"] = steps.pick_location(span, geo_result["entities"])
    pick = st.session_state.get("s5_pick")
    if pick:
        loc = pick["event_loc"]
        if loc:
            field_table([("chosen", str(loc.get("resolved_placename") or "—")),
                         ("admin1", str(loc.get("admin1_name") or "—")),
                         ("country", str(loc.get("country_code3") or "—")),
                         ("feature code", str(loc.get("feature_code") or "—"))])
        else:
            st.info("No geoparsed place was close enough to the span.")
        st.caption(f"{pick['seconds']:.2f} s · {mode_badge(pick['mode'])} · "
                   f"{pick['reason']}")
