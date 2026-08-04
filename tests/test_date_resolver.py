# Tests for the date resolver in ngec/formatter.py.
#
# These live here rather than in a test_formatter.py because the tests/ package
# conftest.py needs Elasticsearch, and date resolution is pure logic that should
# be testable without it.
#
# Two levels are exercised:
#   * _resolve_date(date_string, ref_date) -- the resolver itself, returning a
#     ResolvedDate. Most tests use this.
#   * resolve_date(event) -- the pipeline entry point, which reads
#     event['attributes']['date'] and writes a top-level event['date_resolved'].
#     One record is one event (see explode_events), so 'attributes' is a single
#     dict and the resolved value is top-level.

from ngec.formatter import resolve_date, _resolve_date

REF = "2025-05-16"  # a Friday


def test_resolution():
    event = {"pub_date": "June 20, 2012",
             "attributes": {"date": ["last Sunday"]}
             }
    res = resolve_date(event)
    dr = res["date_resolved"]
    # "last Sunday" relative to Wed June 20, 2012 resolves to Sun June 17, 2012.
    # dateparser can't parse "last Sunday" directly, but the cascade strips the
    # "last" modifier and re-parses with a past preference.
    assert dr["resolved_date"].strftime("%Y-%m-%d") == "2012-06-17"
    assert dr["granularity"] == "day"
    assert dr["date_type"] == "exact"
    assert "past-tense modifier" in dr["reason"]


def test_resolution_fallback():
    # Non-Gregorian / world-knowledge references can't be resolved, so we fall
    # back to the publication date, flagged unresolved.
    event = {"pub_date": "June 20, 2012",
             "attributes": {"date": ["last Ramadan"]}
             }
    res = resolve_date(event)
    dr = res["date_resolved"]
    assert dr["resolved_date"].strftime("%Y-%m-%d") == "2012-06-20"
    assert dr["date_type"] == "unresolved"
    assert dr["granularity"] is None
    assert "failed to convert relative date" in dr["reason"]


def test_weekend_is_approximate():
    # "over the weekend" is resolution ambiguity (not a genuine multi-day event):
    # approximated to the preceding Saturday, flagged approximate at week level,
    # with no end date.
    res = _resolve_date("over the weekend", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-10"
    assert res.date_type == "approximate"
    assert res.granularity == "week"
    assert res.date_end is None


def test_vague_vs_precise_count():
    # Vague counts are approximate and coarsened a unit; definite counts exact.
    vague = _resolve_date("a few days ago", REF)
    assert vague.date_type == "approximate"
    assert vague.granularity == "week"
    precise = _resolve_date("two weeks ago", REF)
    assert precise.date_type == "exact"
    assert precise.granularity == "week"


def test_hedged_count_is_approximate():
    # A definite count the source hedges ("up to", "nearly", "about") resolves to
    # the same date but must not claim to be exact.
    for expr in ["up to three weeks ago", "nearly three weeks ago",
                 "about three weeks ago"]:
        res = _resolve_date(expr, REF)
        assert res.resolved_date.strftime("%Y-%m-%d") == "2025-04-25", expr
        assert res.date_type == "approximate", expr


def test_bounded_range_has_end_date():
    # A genuine range: resolves to the first endpoint, captures the second as
    # date_end, and is typed "range".
    res = _resolve_date("March 15 to March 20", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-03-15"
    assert res.date_end.strftime("%Y-%m-%d") == "2025-03-20"
    assert res.date_type == "range"
    assert res.granularity == "day"


def test_range_end_distinguishes_open_vs_closed():
    # Both are "range", but a closed range carries date_end while an open-ended
    # one does not.
    closed = _resolve_date("last Tuesday to Thursday", REF)
    assert closed.date_type == "range"
    assert closed.resolved_date.strftime("%Y-%m-%d") == "2025-05-13"
    assert closed.date_end.strftime("%Y-%m-%d") == "2025-05-15"

    open_ended = _resolve_date("since March 2024", REF)
    assert open_ended.date_type == "range"
    assert open_ended.resolved_date.strftime("%Y-%m-%d") == "2024-03-16"
    assert open_ended.date_end is None
    assert open_ended.granularity == "month"


def test_range_endpoints_are_never_inverted():
    # The two endpoints of a range are resolved independently, which can put the
    # end *before* the start in two ways. Both must be repaired.

    # (a) only the second endpoint states a year, so the first would default to
    #     the pub year and land a year late.
    res = _resolve_date("March to April 2024", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2024-03-16"
    assert res.date_end.strftime("%Y-%m-%d") == "2024-04-16"

    # (b) both endpoints relative, and the past preference pushes the end behind
    #     the start (the Tuesday before publication precedes the Thursday).
    res = _resolve_date("Thursday through Tuesday", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-15"
    assert res.date_end.strftime("%Y-%m-%d") == "2025-05-20"

    # Whatever the expression, an inverted range must never be emitted.
    for expr in ["March to April 2024", "Thursday through Tuesday",
                 "November to January", "September to March",
                 "March 15 to March 20", "between July and September",
                 "January 2020 - March 2021", "12-14 May"]:
        res = _resolve_date(expr, REF)
        if res.date_end is not None:
            assert res.date_end >= res.resolved_date, f"{expr!r} -> {res}"


def test_non_date_after_connector_is_not_a_range():
    # "Monday to discuss the deal" shares its connector with a real range, but
    # the right-hand side isn't a date. Keep Monday; don't call it a range.
    res = _resolve_date("Monday to discuss the deal", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-12"
    assert res.date_type != "range"
    assert res.date_end is None


def test_quarter_reference():
    res = _resolve_date("Q2 2023", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2023-04-01"
    assert res.granularity == "quarter"
    assert res.date_type == "exact"


def test_n_day_old():
    res = _resolve_date("5-day-old", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-11"
    assert res.granularity == "day"
    assert res.date_type == "exact"


def test_before_after_offset():
    res = _resolve_date("a week before March 10", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-03-03"
    assert res.granularity == "day"
    assert res.date_type == "exact"


def test_nested_relative_in_offset():
    res = _resolve_date("a week before last Friday", REF)
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-02"
    assert res.granularity == "day"
    assert res.date_type == "exact"


def test_parentheses_preferred():
    res = _resolve_date("yesterday (Friday)", REF)
    # the parenthetical "Friday" resolves to the preceding Friday
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-09"


def test_year_and_decade():
    # plain years resolve exact at year granularity; a decade is approximate
    # (sometime in the 90s) at the first year of the decade.
    for expr, expected in [("1992", "1992"), ("last year", "2024"),
                           ("two years ago", "2023")]:
        res = _resolve_date(expr, REF)
        assert res.resolved_date.year == int(expected)
        assert res.granularity == "year"
        assert res.date_type == "exact"
    decade = _resolve_date("the 1990s", REF)
    assert decade.resolved_date.strftime("%Y-%m-%d") == "1990-01-01"
    assert decade.granularity == "year"
    assert decade.date_type == "approximate"


def test_previous_year_synonym():
    # "previous year" must resolve like "last year" (2024), not collapse to a
    # bare "year" that dateparser reads as today.
    res = _resolve_date("previous year", REF)
    assert res.resolved_date.year == 2024
    assert res.granularity == "year"
    assert res.date_type == "exact"


# --- Robustness on noisy extractive-model spans -----------------------------
# These lock in behavior on the kind of messy spans a QA/extractive model emits
# when it grabs the text reporting *when* an event happened.

def test_numeric_and_intl_formats():
    # Unambiguous numeric / international / ISO formats and trailing punctuation.
    cases = [
        ("13/05/2025", "2025-05-13"),       # D/M/Y
        ("05/13/2025", "2025-05-13"),       # M/D/Y (13 can't be a month)
        ("Aug. 1, 2019", "2019-08-01"),
        ("2019-08-01T14:30:00", "2019-08-01"),
        ("the 15th", "2025-05-15"),
        ("July 31st", "2024-07-31"),
        ("on 8 July", "2024-07-08"),
        ("May 13, 2025,", "2025-05-13"),    # trailing-comma noise
    ]
    for expr, expected in cases:
        res = _resolve_date(expr, REF)
        assert res.resolved_date.strftime("%Y-%m-%d") == expected, expr
        assert res.date_type == "exact", expr


def test_weekday_abbreviations_and_timeofday_noise():
    for expr, expected in [("Tues", "2025-05-13"), ("Wed.", "2025-05-14"),
                           ("Thursday evening", "2025-05-15")]:
        res = _resolve_date(expr, REF)
        assert res.resolved_date.strftime("%Y-%m-%d") == expected, expr


def test_correct_rejections():
    # Spans that SHOULD NOT resolve: holidays/seasons and "the YEAR <event>"
    # (need world knowledge), anaphora (no anchor in the span), and garbage
    # grabs. The resolver must degrade to 'unresolved', never hallucinate.
    for expr in ["Christmas Day", "last summer", "during Ramadan", "the 2024 election",
                 "the day before", "that day", "according to police", "sometime soon",
                 "the anniversary of the coup", "the day after the vote",
                 "the summer of 2020", "spring 2019", "the second half of 2022",
                 "Wednesday, officials said", "refused to comment"]:
        res = _resolve_date(expr, REF)
        assert res.date_type == "unresolved", f"{expr!r} should be unresolved, got {res}"


def test_hyphen_day_ranges():
    # Day-of-month hyphen ranges are genuine ranges, not a single date with the
    # second number misread as a year. Endpoints share a year (the end must not
    # flip to the prior year when it falls after the pub date under past-pref).
    cases = [
        ("March 15-20", "2025-03-15", "2025-03-20"),
        ("12-14 May", "2025-05-12", "2025-05-14"),
        ("May 15-20, 2025", "2025-05-15", "2025-05-20"),
        ("May 15-20", "2025-05-15", "2025-05-20"),  # end stays 2025, not 2024
    ]
    for expr, start, end in cases:
        res = _resolve_date(expr, REF)
        assert res.date_type == "range", expr
        assert res.resolved_date.strftime("%Y-%m-%d") == start, expr
        assert res.date_end.strftime("%Y-%m-%d") == end, expr


def test_spaced_hyphen_range():
    # A hyphen with whitespace on both sides is a range separator. A bare hyphen
    # is not, or we'd wreck "mid-March", ISO dates, and "Covid-19".
    res = _resolve_date("January 2020 - March 2021", REF)
    assert res.date_type == "range"
    assert res.resolved_date.strftime("%Y-%m") == "2020-01"
    assert res.date_end.strftime("%Y-%m") == "2021-03"

    assert _resolve_date("mid-March", REF).resolved_date.strftime("%Y-%m-%d") == "2025-03-16"
    assert _resolve_date("2025-05-13", REF).resolved_date.strftime("%Y-%m-%d") == "2025-05-13"
    assert _resolve_date("12-14 May", REF).resolved_date.strftime("%Y-%m-%d") == "2025-05-12"


def test_bare_period_modifier_anchors_to_pubdate():
    # "beginning/end of the year" used to collapse to *today*; now they anchor to
    # the pub-date period and are flagged approximate, never exact.
    begin = _resolve_date("beginning of the year", REF)
    assert begin.resolved_date.strftime("%Y-%m-%d") == "2025-01-01"
    assert begin.granularity == "year"
    assert begin.date_type == "approximate"

    end = _resolve_date("end of the year", REF)
    assert end.resolved_date.strftime("%Y-%m-%d") == "2025-12-31"
    assert end.date_type == "approximate"

    month = _resolve_date("beginning of the month", REF)
    assert month.resolved_date.strftime("%Y-%m-%d") == "2025-05-01"
    assert month.granularity == "month"
    assert month.date_type == "approximate"

    # A modifier on a concrete month must still work (regression guard).
    mid = _resolve_date("mid-March", REF)
    assert mid.resolved_date.strftime("%Y-%m-%d") == "2025-03-16"
    assert mid.date_type == "approximate"


def test_duration_phrase_is_unresolved():
    # "N units of <noun>" is a duration, not an offset: it must not be misread as
    # "N units ago". Safe fallback is unresolved, never a confident date.
    for expr in ["three days of clashes", "5 months of fighting", "two weeks of protests"]:
        res = _resolve_date(expr, REF)
        assert res.date_type == "unresolved", f"{expr!r} should be unresolved, got {res}"


def test_day_idioms():
    # "this morning"->pub day, "last night"/"overnight"->day before, exact at day.
    for expr in ["this morning", "this afternoon", "tonight"]:
        res = _resolve_date(expr, REF)
        assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-16", expr
        assert res.date_type == "exact" and res.granularity == "day", expr
    for expr in ["last night", "overnight", "last evening"]:
        res = _resolve_date(expr, REF)
        assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-15", expr
        assert res.date_type == "exact" and res.granularity == "day", expr

    # Guard: a weekday/parenthetical must NOT be stolen by the idiom branch.
    assert _resolve_date("Thursday evening", REF).resolved_date.strftime("%Y-%m-%d") == "2025-05-15"
    assert _resolve_date("yesterday (Friday)", REF).resolved_date.strftime("%Y-%m-%d") == "2025-05-09"


def test_recency_window_is_approximate():
    res = _resolve_date("in the past few days", REF)
    assert res.date_type == "approximate"
    assert res.granularity == "week"
    # Bare "last week" (no count) stays exact -- not swept up as a recency window.
    assert _resolve_date("last week", REF).date_type == "exact"


def test_week_of_month():
    # Pinned to the pub-date year so sibling references stay in the same year.
    june = _resolve_date("first week of June", REF)
    assert june.resolved_date.strftime("%Y-%m-%d") == "2025-06-01"
    assert june.date_type == "approximate" and june.granularity == "week"
    april = _resolve_date("third week of April", REF)
    assert april.resolved_date.strftime("%Y-%m-%d") == "2025-04-15"
    assert april.resolved_date.year == june.resolved_date.year


def test_leading_prefix_and_clock_noise():
    # "back in March" -> plain month (exact); "as recently as X" carries a bound
    # (approximate); a clock-time prefix is stripped to leave the weekday.
    back = _resolve_date("back in March", REF)
    assert back.resolved_date.strftime("%Y-%m") == "2025-03"
    assert back.date_type == "exact"

    recent = _resolve_date("as recently as Monday", REF)
    assert recent.resolved_date.strftime("%Y-%m-%d") == "2025-05-12"
    assert recent.date_type == "approximate"

    clock = _resolve_date("around 6:30 a.m. on Tuesday", REF)
    assert clock.resolved_date.strftime("%Y-%m-%d") == "2025-05-13"
    assert clock.date_type == "exact"

    # Guard: the am/pm gate keeps the clock branch off ISO datetimes and "on 8 July".
    assert _resolve_date("2019-08-01T14:30:00", REF).resolved_date.strftime("%Y-%m-%d") == "2019-08-01"
    assert _resolve_date("on 8 July", REF).resolved_date.strftime("%Y-%m-%d") == "2024-07-08"


def test_ranges_from_messy_text():
    res = _resolve_date("from Monday to Wednesday", REF)
    assert res.date_type == "range"
    assert res.resolved_date.strftime("%Y-%m-%d") == "2025-05-12"
    assert res.date_end.strftime("%Y-%m-%d") == "2025-05-14"
    res = _resolve_date("between July and September", REF)
    assert res.date_type == "range"
    assert res.date_end is not None


def test_pub_date_is_normalized_to_naive_midnight():
    # Publication dates arrive with times and timezones. If we kept them, some
    # branches would return tz-aware datetimes (arithmetic off the reference)
    # and others naive ones (dates we construct), which is miserable in pandas.
    for pub in ["2019-08-01", "2019-08-01T14:30:00", "2019-08-01T14:30:00+00:00"]:
        for expr in ["yesterday", "beginning of the year", "March 15-20", "last Ramadan"]:
            res = _resolve_date(expr, pub)
            assert res.resolved_date.tzinfo is None, (pub, expr)
            assert (res.resolved_date.hour, res.resolved_date.minute) == (0, 0), (pub, expr)
            if res.date_end is not None:
                assert res.date_end.tzinfo is None, (pub, expr)


def test_resolve_date_event_integration():
    # Full public path: resolve_date(event) writes asdict(result) to the
    # top-level 'date_resolved' key. One record is one event, so 'attributes' is
    # a single dict.
    pub = "2025-05-16"

    def resolved(date_string):
        event = resolve_date({"pub_date": pub, "attributes": {"date": [date_string]}})
        return event["date_resolved"]

    rng = resolved("March 15-20")
    assert set(rng) == {"resolved_date", "date_end", "granularity", "date_type", "reason"}
    assert rng["date_type"] == "range"
    assert rng["resolved_date"].strftime("%Y-%m-%d") == "2025-03-15"
    assert rng["date_end"].strftime("%Y-%m-%d") == "2025-03-20"

    approx = resolved("beginning of the year")
    assert approx["date_type"] == "approximate"
    assert approx["resolved_date"].strftime("%Y-%m-%d") == "2025-01-01"
    assert approx["date_end"] is None

    dur = resolved("three days of clashes")
    # Duration phrase doesn't resolve, so it falls back to the pub date, unresolved.
    assert dur["date_type"] == "unresolved"
    assert dur["resolved_date"].strftime("%Y-%m-%d") == pub


def test_resolve_date_reads_attributes_defensively():
    # The attribute model may omit 'date', hand back an empty list, or emit a
    # bare string instead of a list. None of those may crash, and a bare string
    # must not be indexed character-wise.
    pub = "2025-05-16"

    assert resolve_date({"pub_date": pub, "attributes": {}})["date_resolved"]["date_type"] == "unresolved"
    assert resolve_date({"pub_date": pub, "attributes": {"date": []}})["date_resolved"]["date_type"] == "unresolved"
    assert resolve_date({"pub_date": pub})["date_resolved"]["date_type"] == "unresolved"

    bare = resolve_date({"pub_date": pub, "attributes": {"date": "yesterday"}})["date_resolved"]
    assert bare["resolved_date"].strftime("%Y-%m-%d") == "2025-05-15"

    # No publication date to anchor against: nothing to resolve to.
    no_pub = resolve_date({"attributes": {"date": ["yesterday"]}})["date_resolved"]
    assert no_pub["resolved_date"] is None
    assert no_pub["date_type"] == "unresolved"
