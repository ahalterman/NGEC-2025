"""Step 5: when and where? Date phrases against a pub date, places against geonames."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import lede

# (phrase, publication date) -- a weekday, a relative offset, and an open-ended
# range, which are the three shapes the resolver has to tell apart.
DATE_EXAMPLES: list[tuple[str, str]] = [
    ("last Tuesday", "2023-03-15"),
    ("three weeks ago", "2024-06-11"),
    ("since early 2015", "2024-02-20"),
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
lede("A date phrase is resolved against the story's publication date; place "
     "names are resolved against geonames, then matched to the event's location span.")

health = R.health()
missing = [name for name, row in health.items() if not row.get("ok")]
if missing:
    st.warning(f"Not available: {', '.join(missing)}. Geoparsing needs the "
               "geonames index; date resolution does not.")

# --- 5.1 dates ---------------------------------------------------------------

st.subheader("5.1 Dates")

if "s5_phrase" not in st.session_state:
    st.session_state.s5_phrase = DATE_EXAMPLES[0][0]
    st.session_state.s5_pub = pd.to_datetime(DATE_EXAMPLES[0][1]).date()

cols = st.columns(len(DATE_EXAMPLES))
for col, (example_phrase, example_date) in zip(cols, DATE_EXAMPLES):
    if col.button(example_phrase, width="stretch", key=f"s5_date_{example_phrase}"):
        st.session_state.s5_phrase = example_phrase
        st.session_state.s5_pub = pd.to_datetime(example_date).date()
        st.rerun()

left, right = st.columns(2)
phrase = left.text_input("Date phrase", key="s5_phrase")
pub_date = right.date_input("Published", key="s5_pub")

if phrase.strip():
    # No models and no network: cheap enough to redo on every keystroke.
    date_result = steps.resolve_date(phrase, str(pub_date))
    st.dataframe(pd.DataFrame([{
        "resolved": (date_result["resolved_date"] or "—")[:10],
        "end": (date_result["date_end"] or "—")[:10],
        "granularity": date_result["granularity"] or "—",
        "type": date_result["date_type"] or "—",
        "reason": date_result["reason"] or "—",
    }]), hide_index=True, width="stretch")

# --- 5.2 geoparsing ----------------------------------------------------------

st.subheader("5.2 Geoparsing")

if "s5_geo_text" not in st.session_state:
    st.session_state.s5_geo_text = GEO_EXAMPLES[0][1]

cols = st.columns(len(GEO_EXAMPLES))
for col, (label, text) in zip(cols, GEO_EXAMPLES):
    if col.button(label, width="stretch", key=f"s5_geo_{label}"):
        st.session_state.s5_geo_text = text
        st.session_state.pop("s5_geo_result", None)
        st.session_state.pop("s5_pick", None)
        st.rerun()

st.text_area("Text", key="s5_geo_text", height=110)
if st.button("Geoparse", type="primary"):
    with st.spinner("Loading spaCy and the geoparser…"):
        result = steps.geoparse(st.session_state.s5_geo_text)
    st.session_state["s5_geo_result"] = result
    # A pick from the previous text would otherwise sit under the new places.
    st.session_state.pop("s5_pick", None)
    if result["entities"]:
        # 5.3 starts from the first place found, which is what the attribute
        # model's location span usually points at.
        st.session_state["s5_span"] = result["entities"][0]["search_name"]

geo_result = st.session_state.get("s5_geo_result")
if geo_result:
    entities = geo_result["entities"]
    if entities:
        table = pd.DataFrame([{
            "span": e["search_name"],
            "resolved": e["resolved_placename"],
            "admin1": e["admin1_name"],
            "country": e["country_code3"],
            "feature": e["feature_code"],
            "score": e["score"],
            "lat": e["lat"],
            "lon": e["lon"],
        } for e in entities])
        st.dataframe(table, hide_index=True, width="stretch")

        points = table[["lat", "lon"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(points):
            st.map(points, latitude="lat", longitude="lon", size=20000)
    else:
        st.info("No place names were resolved in this text.")
    st.caption(f"{geo_result['seconds']:.1f}s · {len(entities)} place(s)")

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
            st.dataframe(pd.DataFrame([{
                "chosen": loc.get("resolved_placename"),
                "admin1": loc.get("admin1_name"),
                "country": loc.get("country_code3"),
                "feature": loc.get("feature_code"),
                "score": loc.get("score"),
            }]), hide_index=True, width="stretch")
        else:
            st.info("No geoparsed place was close enough to the span.")
        st.caption(f"{pick['seconds']:.2f}s · {pick['reason']}")
