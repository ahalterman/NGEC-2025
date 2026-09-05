"""Step 5: resolving date expressions and place names."""

import html

import streamlit as st

from ngec_demo import examples as ex_mod
from ngec_demo import paper, pipeline, ui
from ngec_demo.theme import PALETTE, kv_table, note

ui.setup()
ui.step_rail("step5")

ui.page_header(
    kicker="Step 5 of 5",
    title="Dates and places",
    standfirst=(
        "“Last Tuesday” becomes a date; “Aleppo” becomes a point on the earth. Both are "
        "resolution against an external reference — the publication date in one case, the "
        "Geonames gazetteer in the other."
    ),
    paper_keys=("step5",),
)

tab_dates, tab_places = st.tabs(["Dates", "Places"])

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

with tab_dates:
    st.markdown(
        """
The attribute model returns the date as the article wrote it. Turning that into something a
researcher can put on an axis means resolving it against the publication date, and news
prose is far more varied than "yesterday": *late Tuesday night*, *since Thursday*, *over the
weekend*, *a week before last Friday*, *the second week of April*.

The resolver is an ordered cascade of strategies. Each either resolves the span or falls
through to the next, and the structural ones recurse on a strictly shorter sub-expression, so
nested relatives work and the recursion always terminates.
"""
    )

    note(
        "The design rule is <strong>fail-safe, not best-effort</strong>. A step that cannot "
        "resolve returns nothing rather than guessing, and the caller stamps the publication "
        "date flagged <code>unresolved</code>. Several steps exist purely to refuse: “three "
        "days of clashes” is a duration, not an offset, and must not silently become “three "
        "days ago”. A wrong date is worse than a missing one, because a missing one can be "
        "filtered.",
        title="Refusing is a feature",
    )

    pub_date = st.text_input("Publication date", value="2023-03-15", key="date_pub")

    st.markdown("### The cascade, on a battery of spans")

    for group, phrases in ex_mod.DATE_PHRASES.items():
        refuses = "Refused" in group
        st.markdown(
            f'<div class="ngec-kicker" style="margin-top:1.1rem">{html.escape(group)}</div>',
            unsafe_allow_html=True,
        )
        rows = []
        for phrase in phrases:
            res = pipeline.resolve_date_phrase(phrase, pub_date)
            resolved = str(res.get("resolved_date") or "")[:10]
            end = str(res.get("date_end") or "")[:10]
            date_type = res.get("date_type") or ""

            if date_type == "unresolved":
                value = (
                    f'<span style="color:{PALETTE["ink_faint"]}">unresolved — '
                    f"stamped {resolved}</span>"
                )
            else:
                value = f"<strong>{resolved}</strong>"
                if end:
                    value += f" → <strong>{end}</strong>"
                value += (
                    f' <span class="none">{date_type}, '
                    f'{res.get("granularity", "")}</span>'
                )
            rows.append((phrase, value))
        st.markdown(kv_table(rows), unsafe_allow_html=True)

        if refuses:
            st.markdown(
                '<div class="ngec-meta" style="margin-top:0.4rem">'
                "These need world knowledge the pipeline does not have — when the coup was, "
                "when Ramadan fell that year. They stay unresolved by design, and "
                "<code>tests/test_date_resolver.py::test_correct_rejections</code> locks that "
                "in so a future change cannot quietly start guessing at them.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Resolve your own")
    own = st.text_input("Date expression", value="a week before last Friday", key="date_own")
    if own.strip():
        res = pipeline.resolve_date_phrase(own, pub_date)
        st.markdown(
            kv_table([
                ("Resolved date", str(res.get("resolved_date") or "")[:10]),
                ("Range end", str(res.get("date_end") or "")[:10]),
                ("Granularity", res.get("granularity") or ""),
                ("Type", res.get("date_type") or ""),
                ("Trail", html.escape(str(res.get("reason") or ""))),
            ]),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ngec-meta">The trail is chained with ← through nested resolutions, '
            "so a compound expression shows every step that fired.</div>",
            unsafe_allow_html=True,
        )

    note(
        "A date that resolves far before the publication date is the main signal available for "
        "catching historical events reported as current ones. That is the subject of "
        "<a href='echoes'>the temporal echoes page</a>, and it is a filter the pipeline does "
        "not currently apply.",
        title="Dates are also an echo detector",
    )

# ---------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------

with tab_places:
    st.markdown(
        """
Place names are resolved with [mordecai3](https://github.com/ahalterman/mordecai3) against a
local Geonames index: every place mention in the document is geolocated, and then the event's
extracted location span is matched to one of those resolutions.

Geolocation is a ranking problem, not a lookup. "Georgia" is a country and a US state;
"Springfield" is dozens of places. The surrounding document is what disambiguates.
"""
    )

    text, pub_date_p, example = ui.example_picker("step5", show_source=False)
    ui.what_to_notice(example)

    result = ui.run_with_progress(text, pub_date_p, label="Running the pipeline")

    if not ui.run_error(result):
        # Two different inputs meet here, which is worth stating: mordecai3 reads
        # the whole document to resolve every place mention in it, and the event's
        # own location is then chosen from those by matching the span step 2
        # extracted. The date half of the page needs no document at all.
        spans = []
        for record in result.final:
            for span in ui.attribute_spans(record)["location"]:
                if span not in spans:
                    spans.append(span)
        ui.stage_input(
            [
                ("Extracted location", ui.span_list(spans), "location"),
                ("Publication date", html.escape(pub_date_p), "date"),
            ],
            explain="Geolocation is the one part of this step that reads the whole article: "
                    "the surrounding text is what tells “Georgia” the country from "
                    "“Georgia” the state.",
        )

        geo_step = result.step("geolocate")
        if geo_step and geo_step.skipped:
            note(
                "The Geonames index is not reachable from this server, so geolocation could "
                "not run.",
                title="Index unavailable",
                tone="bad",
            )
        else:
            st.markdown("### Every place mention in the document")
            story = (geo_step.records or [{}])[0]
            ents = story.get("geolocated_ents") or []
            if ents:
                rows = []
                for ent in ents:
                    if not ent:
                        continue
                    name = ent.get("search_name") or ent.get("placename") or "—"
                    place = ", ".join(
                        str(b) for b in [ent.get("resolved_placename"),
                                         ent.get("admin1_name"),
                                         ent.get("country_code3")] if b
                    )
                    coords = (
                        f' <span class="none">({ent["lat"]:.3f}, {ent["lon"]:.3f}) '
                        f'{ent.get("feature_code", "")}</span>'
                        if ent.get("lat") is not None else ""
                    )
                    rows.append((name, place + coords))
                st.markdown(kv_table(rows), unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="ngec-meta">No place mentions were resolved.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### The location picked for each event")
            for record in result.final:
                loc = (record.get("event_location") or {})
                event_loc = loc.get("event_loc") or {}
                extracted = ((record.get("attributes") or {}).get("location") or ["—"])[0]
                st.markdown(
                    kv_table([
                        ("Event", f'<strong>{record.get("event_type", "")}</strong>'),
                        ("Extracted span", html.escape(str(extracted))),
                        ("Resolved to", ", ".join(str(b) for b in [
                            event_loc.get("resolved_placename"),
                            event_loc.get("admin1_name"),
                            event_loc.get("country_code3")] if b)),
                        ("Coordinates", f'{event_loc["lat"]:.4f}, {event_loc["lon"]:.4f}'
                            if event_loc.get("lat") is not None else ""),
                        ("Geonames ID", str(event_loc.get("geonameid") or "")),
                        ("Outcome", loc.get("reason", "")),
                    ]),
                    unsafe_allow_html=True,
                )
                st.write("")

paper.ref("step5", "framework")
