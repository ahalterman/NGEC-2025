
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from importlib import resources
import logging
import os
import re
import warnings

import dateparser
import jsonlines
import numpy as np
import pandas as pd
from rich import print


logger = logging.getLogger(__name__)


# silence dateparser warning. https://github.com/scrapinghub/dateparser/issues/1013
warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)


def country_name_dict(file_path: str | None=None) -> dict:
    if file_path is None:
        file = str(resources.files('ngec').joinpath("assets", "countries.csv"))
    else:
        file = file_path
    countries = pd.read_csv(file)
    country_name_dict = {i:j for i, j in zip(countries['CCA3'], countries['Name'])}
    country_name_dict.update({"": ""})
    country_name_dict.update({"IGO": "Intergovernmental Organization"})
    return country_name_dict


def _first_attribute_value(value):
    """
    Pull the leading span out of an attribute value. The attribute model is
    supposed to emit lists (``['last week']``), but it sometimes emits a bare
    string and sometimes omits the key entirely, so accept all three. Returns
    ``None`` when there's nothing there.
    """
    if isinstance(value, list):
        return value[0] if value else None
    return value  # str or None


def resolve_date(event: dict) -> dict:
    """
    Add a top-level 'date_resolved' key to an event, resolved from its
    attributes' 'date' value and the event's publication date.

    One record is one event, so 'attributes' is a single dict and there is a
    single date to resolve. The value stored is ``asdict()`` of a
    :class:`ResolvedDate`.

    >>> DateDataParser().get_date_data('March 2015')
    DateData(date_obj=datetime.datetime(2015, 3, 16, 0, 0), period='month', locale='en')
    """
    attributes = event.get('attributes') or {}
    date_string = _first_attribute_value(attributes.get('date'))
    res = _resolve_date(date_string=date_string, ref_date=event.get('pub_date'))
    event['date_resolved'] = asdict(res)
    return event


@dataclass
class ResolvedDate:
    """
    The outcome of resolving a date expression.

    The three pieces of resolution information are deliberately kept separate,
    because they answer different questions a data user will ask.

    Attributes
    ----------
    resolved_date : datetime | str | None
        The single resolved date, or the *start* of a range.
    date_end : datetime | None
        The *end* of a genuine, source-stated range (e.g. "Tuesday to
        Thursday"). ``None`` for a single point or an open-ended range.
    granularity : str | None
        The precision *unit* of the resolution: ``day``, ``week``, ``month``,
        ``quarter``, or ``year``. ``None`` when nothing could be resolved.
        This says how coarsely the date is known -- it does NOT say whether the
        event spanned multiple days (that's ``date_end``) or whether the
        resolution is fuzzy (that's ``date_type``).
    date_type : str | None
        How to interpret the resolution:

        - ``"exact"``       -- a point the source states plainly (to its
          granularity): "May 3 2021", "last Tuesday", "March 2015".
        - ``"approximate"`` -- a point we resolved but the source is vague, so
          the exact value shouldn't be over-trusted: "over the weekend",
          "a few days ago", "early May", "the 1990s".
        - ``"range"``       -- a genuine multi-period event. ``date_end`` holds
          the end (or is ``None`` for an open-ended range like "since 2015").
        - ``"unresolved"``  -- nothing resolved; fell back to the pub date/None.
    reason : str | None
        Human-readable trail describing how the value was reached.
    """
    resolved_date: datetime | str | None
    date_end: datetime | None = None
    granularity: str | None = None
    date_type: str | None = None
    reason: str | None = None

    def __post_init__(self):
        if isinstance(self.resolved_date, str):
            try:
                self.resolved_date = datetime.strptime(self.resolved_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"resolved_date string not in YYYY-MM-DD format: {self.resolved_date}")


def _try_get_date(date_string: str,
                  base_date: datetime,
                  prefer: str) -> tuple[datetime, str] | None:
    """
    Run dateparser's ``DateDataParser`` and return ``(date_obj, period)`` if it
    succeeds, otherwise ``None``.

    Unlike ``dateparser.parse``, ``get_date_data`` exposes the ``period``
    (granularity) of the match, which is the date-resolution information we want
    to preserve all the way through the cascade below.

    Parameters
    ----------
    date_string : str
        Text to parse.
    base_date : datetime
        Reference date for relative expressions (RELATIVE_BASE).
    prefer : str
        ``"past"`` or ``"future"`` (PREFER_DATES_FROM).
    """
    parser = dateparser.DateDataParser(
        languages=['en'],
        settings={'RELATIVE_BASE': base_date, 'PREFER_DATES_FROM': prefer},
    )
    res = parser.get_date_data(date_string)
    if res.date_obj is not None:
        return res.date_obj, res.period
    return None


# Phrases that carry no usable date information; short-circuit to a failure so
# the caller falls back to the publication date.
_NOISE_PHRASES = {
    "", "not specified", "unspecified", "ongoing", "unknown", "n/a", "na",
    "tba", "recently", "recent", "no date", "none",
}

_MONTHS = (r"January|February|March|April|May|June|July|August|September|"
           r"October|November|December")

_WEEKDAYS = (r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
             r"mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun")

# map weekday names to their Python weekday index (Monday = 0), for
# the same-day rule in _resolve_date.
_WEEKDAY_INDEX = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "weds": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

# Qualifiers, which reduces our reported confidence 
_HEDGES = (r"up to|at least|as many as|as much as|nearly|almost|roughly|"
           r"approximately|about|around|sometime|on or about")

# First month of each quarter, used to resolve quarter references to a date.
_QUARTER_TO_MONTH = {
    "first": 1, "1st": 1, "q1": 1,
    "second": 4, "2nd": 4, "q2": 4,
    "third": 7, "3rd": 7, "q3": 7,
    "fourth": 10, "4th": 10, "q4": 10,
}

_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "couple": 2, "few": 3, "a few": 3, "several": 3}

_UNIT_TO_DAYS = {"day": 1, "week": 7, "month": 30}


def _phrase_to_days(phrase: str) -> int | None:
    """Turn an amount phrase ("a week", "two days", "3 months") into a day count."""
    phrase = phrase.lower().strip()
    m = re.search(r"(\d+)\s*(day|week|month)s?", phrase)
    if m:
        return int(m.group(1)) * _UNIT_TO_DAYS[m.group(2)]
    m = re.search(r"(a few|couple|few|several|an|a|one|two|three|four|five)\s*(day|week|month)s?", phrase)
    if m:
        return _NUMBER_WORDS.get(m.group(1), 1) * _UNIT_TO_DAYS[m.group(2)]
    return None


def _is_hedged(raw: str) -> bool:
    """Does the span hedge the date it states ("up to two weeks ago")?"""
    return re.search(rf"\b(?:{_HEDGES})\b", raw, re.IGNORECASE) is not None


def _same_day_weekday(raw: str, base_date: datetime) -> ResolvedDate | None:
    """
    Resolve a bare weekday that names the publication day itself.

    dateparser's ``PREFER_DATES_FROM='past'`` means *strictly* before the
    reference date, so "Saturday" in a story filed on a Saturday resolves to the
    Saturday a week earlier. In news copy a bare weekday naming the filing day
    means that day, so return the base date. Every other weekday is left to the
    normal past-preference parse and still lands on its most recent occurrence
    before publication.

    Deliberately narrow: it fires only when the *whole* span is a weekday name
    (optionally prefixed with "on" or "this"). Anything carrying a modifier
    ("last Tuesday", "next Tuesday"), a time of day ("Thursday evening"), a
    month, or a digit goes through the cascade untouched.
    """
    m = re.fullmatch(rf"(?:on\s+|this\s+)?({_WEEKDAYS})", raw.strip(" -,."),
                     re.IGNORECASE)
    if m is None or _WEEKDAY_INDEX[m.group(1).lower()] != base_date.weekday():
        return None
    return ResolvedDate(resolved_date=base_date, granularity="day", date_type="exact",
                        reason="<Bare weekday naming the publication day, resolved to the pub date>")


def _anchor_bare_period(modifier: str, period: str, base_date: datetime) -> ResolvedDate | None:
    """
    Resolve a within-period modifier applied to a *bare* period word
    ("beginning of the year", "end of the year", "mid-year", "beginning of the
    month"). Without an explicit month/year, dateparser resolves the leftover
    "the year" to *today*, so we anchor to the current period derived from
    ``base_date`` instead. Returns ``None`` for periods we don't anchor (safe:
    the caller falls through rather than emitting today).
    """
    modifier = modifier.lower()
    if "early" in modifier or "begin" in modifier:
        pos = "start"
    elif "mid" in modifier:
        pos = "mid"
    else:  # "late", "end of"
        pos = "end"

    if period == "year":
        month = {"start": 1, "mid": 7, "end": 12}[pos]
        day = 31 if pos == "end" else 1
        return ResolvedDate(resolved_date=datetime(base_date.year, month, day),
                            granularity="year", date_type="approximate",
                            reason=f"<Anchored '{modifier} {period}' to the {pos} of the pub-date year>")
    if period == "month":
        if pos == "end":
            nxt = datetime(base_date.year + 1, 1, 1) if base_date.month == 12 \
                else datetime(base_date.year, base_date.month + 1, 1)
            anchored = nxt - timedelta(days=1)
        else:
            anchored = datetime(base_date.year, base_date.month, 15 if pos == "mid" else 1)
        return ResolvedDate(resolved_date=anchored,
                            granularity="month", date_type="approximate",
                            reason=f"<Anchored '{modifier} {period}' to the {pos} of the pub-date month>")
    return None


def _align_range(res: ResolvedDate, second: str, base_date: datetime) -> ResolvedDate:
    """
    Make a bounded range internally consistent, i.e. ensure ``date_end`` is not
    *before* ``resolved_date``.

    The two endpoints are resolved independently, so an inverted range can come
    out in two ways:

    - Only the second endpoint carries a year ("March to April 2024"). The first
      then defaults to the publication year and lands a year late.
    - Both endpoints are relative and the past preference pushes the end behind
      the start ("Thursday through Tuesday": the Tuesday before publication is
      earlier than the Thursday before it).

    We repair the first by pulling the end's year onto the start, and the second
    by re-resolving the end *forward* from the start. If neither repair works,
    we drop ``date_end``: an open-ended range is honest, an inverted one is not.
    """
    if res.resolved_date is None or res.date_end is None or res.date_end >= res.resolved_date:
        return res

    # The second endpoint states a year the first one lacks: adopt it.
    if re.search(r"\b\d{4}\b", second):
        try:
            pulled = res.resolved_date.replace(year=res.date_end.year)
        except ValueError:   # Feb 29 in a non-leap year
            pulled = None
        if pulled is not None and pulled <= res.date_end:
            return replace(res, resolved_date=pulled,
                           reason=f"{res.reason.rstrip('>')}, start pinned to the end's year>")

    # Both endpoints relative: re-resolve the end going forward from the start.
    hit = _try_get_date(second.strip(" -,."), res.resolved_date, "future")
    if hit is not None and hit[0] >= res.resolved_date:
        return replace(res, date_end=hit[0],
                       reason=f"{res.reason.rstrip('>')}, end re-resolved forward from the start>")

    return replace(res, date_end=None,
                   reason=f"{res.reason.rstrip('>')}, end dropped (resolved before the start)>")


def _resolve_core(raw: str, base_date: datetime) -> ResolvedDate | None:
    """
    Resolve a single date expression relative to ``base_date``.

    Returns a :class:`ResolvedDate` on success or ``None`` if the expression
    can't be resolved (the caller decides on a fallback). Structural handlers
    (parentheses, ranges, "before/after", "since/from", ...) recurse back into
    this function on the *sub-expression* they extract, so nested relatives such
    as "last Tuesday to Thursday" or "since last March" resolve correctly.
    Recursion always runs on a strictly shorter substring, which guarantees
    termination.

    The steps are tried in order and each one bails out rather than guessing, so
    an unrecognized span ends up unresolved instead of confidently wrong.
    """
    raw = raw.strip()
    low = raw.lower()

    if low in _NOISE_PHRASES or re.match(r"^not specified\b", low):
        return None

    def recurse(sub: str, prefix: str, granularity: str | None = None,
                date_type: str | None = None, date_end="inherit") -> ResolvedDate | None:
        """
        Resolve a shorter sub-expression and re-wrap its result. ``prefix`` is
        plain text describing this layer; it's chained onto the inner reason with
        ``←`` so nested resolutions read as a single legible trail. ``granularity``,
        ``date_type``, and ``date_end`` override the inner result when given;
        otherwise they're inherited (``date_end`` inherits on the sentinel
        ``"inherit"`` so ``None`` can be passed to force an open-ended range).
        """
        sub = sub.strip(" -,.")
        if not sub or len(sub) >= len(raw):
            return None
        inner = _resolve_core(sub, base_date)
        if inner is None or inner.resolved_date is None:
            return None
        return ResolvedDate(
            resolved_date=inner.resolved_date,
            date_end=inner.date_end if date_end == "inherit" else date_end,
            granularity=granularity if granularity is not None else inner.granularity,
            date_type=date_type if date_type is not None else inner.date_type,
            reason=f"<{prefix} ← {inner.reason.strip('<>')}>",
        )

    # 0. Decade references ("the 1990s", "2010s"). dateparser misreads the
    #    trailing 's' (it returns a near-today date), so intercept before the
    #    direct parse and resolve to the first year of the decade. Treated as
    #    approximate -- "in the 1990s" means "sometime in the 90s", not a
    #    10-year event. (Judgment call; flip to date_type="range" if you'd
    #    rather model a decade as a span.)
    decade = re.search(r"\b(?:the\s+)?((?:1[89]|20)\d0)s\b", raw, re.IGNORECASE)
    if decade:
        return ResolvedDate(resolved_date=datetime(int(decade.group(1)), 1, 1),
                            granularity="year", date_type="approximate",
                            reason=f"<Resolved decade reference '{decade.group(0).strip()}' to start of decade>")

    # 0b. Day-of-month hyphen ranges ("March 15-20", "12-14 May",
    #     "May 15-20, 2025"). Caught before the direct parse, which otherwise
    #     misreads "March 15-20" as March 15, 2020. We deliberately do NOT add a
    #     bare "-" to the general range separator (step 7): that would wreck
    #     "mid-March", "Covid-19", and ISO dates "2025-05-13". Instead we match
    #     only a digit-hyphen-digit *flanked by a month name*, reconstruct both
    #     endpoints, and resolve each. The end endpoint is resolved with the
    #     start's year pinned, so "May 15-20" (where May 20 is after the pub
    #     date) doesn't flip the end to the prior year under the past preference.
    hy = re.search(rf"\b({_MONTHS})\s+(\d{{1,2}})\s*-\s*(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b", raw, re.IGNORECASE)
    hy_order = "MD" if hy else None
    if not hy:
        hy = re.search(rf"\b(\d{{1,2}})\s*-\s*(\d{{1,2}})\s+({_MONTHS})(?:,?\s+(\d{{4}}))?\b", raw, re.IGNORECASE)
        hy_order = "DM" if hy else None
    if hy:
        if hy_order == "MD":
            month, d1, d2, year = hy.group(1), hy.group(2), hy.group(3), hy.group(4)
            start_str = f"{month} {d1}" + (f" {year}" if year else "")
        else:
            d1, d2, month, year = hy.group(1), hy.group(2), hy.group(3), hy.group(4)
            start_str = f"{d1} {month}" + (f" {year}" if year else "")
        start_res = _resolve_core(start_str, base_date)
        if start_res is not None and start_res.resolved_date is not None:
            pinned_year = year if year else start_res.resolved_date.year
            end_str = (f"{month} {d2} {pinned_year}" if hy_order == "MD"
                       else f"{d2} {month} {pinned_year}")
            end_res = _resolve_core(end_str, base_date)
            end_date = end_res.resolved_date if (end_res is not None and end_res.resolved_date is not None) else None
            return ResolvedDate(resolved_date=start_res.resolved_date, date_end=end_date,
                                granularity=start_res.granularity, date_type="range",
                                reason=f"<Resolved day-of-month hyphen range '{hy.group(0).strip()}'>")

    # 0c. Duration phrase ("three days of clashes", "5 months of fighting"). A
    #     bare "N units of <noun>" is a *duration*, not an offset, but the direct
    #     parse would read "three days" as "three days ago". Refuse to resolve it
    #     (safe: the caller falls back to the pub date, flagged unresolved) rather
    #     than emit a confidently-wrong date.
    if re.search(r"\b(?:\d+|a|an|one|two|three|four|five|few|several|couple)\s+"
                 r"(?:day|week|month|year)s?\s+of\s+[a-z]", low):
        return None

    # 0d. Common day idioms ("this morning" -> pub day, "last night"/"overnight"
    #     -> day before). dateparser doesn't resolve these. Fires ONLY when no
    #     weekday, month, or digit is present, so more specific dated spans
    #     ("Thursday evening", "yesterday (Friday)") keep their own resolution.
    if not re.search(rf"\b(?:{_WEEKDAYS})\b", low) \
            and not re.search(rf"\b(?:{_MONTHS})\b", raw, re.IGNORECASE) \
            and not re.search(r"\d", raw):
        if re.search(r"\b(?:this (?:morning|afternoon|evening)|today|tonight)\b", low):
            return ResolvedDate(resolved_date=base_date, granularity="day", date_type="exact",
                                reason="<Resolved day idiom to the publication day>")
        if re.search(r"\b(?:last night|last evening|overnight|yesterday)\b", low):
            return ResolvedDate(resolved_date=base_date - timedelta(days=1),
                                granularity="day", date_type="exact",
                                reason="<Resolved day idiom to the day before publication>")

    # 0e. Recency window ("in the past few days", "the last several weeks"). Like
    #     "N units ago" but the window is vague, so it's approximate at a coarser
    #     unit. Bare "last week"/"last month" (no count) are NOT matched -- they
    #     are exact and handled by the direct parse / past-modifier steps.
    recency = re.search(r"\b(?:in\s+)?(?:the\s+)?(?:past|last)\s+"
                        r"(few|several|couple|a few|one|two|three|four|five|\d+)\s+"
                        r"(day|week|month)s?\b", low)
    if recency:
        days = _phrase_to_days(f"{recency.group(1)} {recency.group(2)}")
        if days is not None:
            coarser = {"day": "week", "week": "month", "month": "year"}[recency.group(2)]
            return ResolvedDate(resolved_date=base_date - timedelta(days=days),
                                granularity=coarser, date_type="approximate",
                                reason=f"<Approximate recency window '{recency.group(0).strip()}' relative to pub date>")

    # 0f. "Nth week of <month>" ("first week of June", "third week of April").
    #     Approximate to the start of that week. The month's year is pinned to the
    #     pub-date year (not dateparser's past preference) so sibling references
    #     like "first week of June" / "third week of April" stay in the same year.
    weekof = re.search(r"\b(first|second|third|fourth|last|1st|2nd|3rd|4th)\s+week\s+(?:of|in)\s+(.+)",
                       raw, re.IGNORECASE)
    if weekof:
        ord_map = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2,
                   "3rd": 2, "fourth": 3, "4th": 3, "last": 3}
        month_res = _resolve_core(weekof.group(2).strip(), base_date)
        if month_res is not None and month_res.resolved_date is not None:
            # Pin to the pub-date year only when the source gave no explicit year
            # ("first week of June"); an explicit "... June 2023" is kept as-is.
            has_explicit_year = re.search(r"\b\d{4}\b", weekof.group(2)) is not None
            year = month_res.resolved_date.year if has_explicit_year else base_date.year
            first_of_month = month_res.resolved_date.replace(year=year, day=1)
            offset = ord_map.get(weekof.group(1).lower(), 0) * 7
            return ResolvedDate(resolved_date=first_of_month + timedelta(days=offset),
                                granularity="week", date_type="approximate",
                                reason=f"<Resolved '{weekof.group(0).strip()}' to that week of the pub-date-year month>")

    # 0g. Leading recency/filler prefixes ("back in March", "as recently as
    #     Monday"). Strip the prefix and resolve the remainder. "as recently as"
    #     states an upper bound (not a plain date), so its result is approximate.
    prefix = re.search(r"^(back in|as recently as)\s+(.+)", raw, re.IGNORECASE)
    if prefix:
        approximate = prefix.group(1).lower() == "as recently as"
        res = recurse(prefix.group(2), f"stripped leading prefix '{prefix.group(1).strip()}'",
                      date_type="approximate" if approximate else None)
        if res is not None:
            return res

    # 0h. Clock-time prefix ("around 6:30 a.m. on Tuesday"). Gated on an
    #     am/pm/o'clock marker -- a bare colon would wrongly fire on ISO datetimes
    #     ("2019-08-01T14:30:00"), and the am/pm gate is also what keeps the
    #     connector strip from touching "on 8 July". Strip the clock time and its
    #     leading connectors, then resolve the remaining date.
    if re.search(r"\b(?:a\.?m\.?|p\.?m\.?|o'?clock)\b", low):
        stripped = re.sub(r"\b(?:around|about|at|approximately)\b", "", raw, flags=re.IGNORECASE)
        stripped = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|o'?clock)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"^\s*(?:on|at)\s+", "", stripped.strip(" -,"), flags=re.IGNORECASE)
        stripped = re.sub(r"\s+", " ", stripped).strip(" -,")
        if stripped and stripped.lower() != low:
            res = recurse(stripped, "stripped clock time")
            if res is not None:
                return res

    # 1. Direct parse (past preference). Handles absolute dates and the simple
    #    relative forms dateparser already knows, preserving its granularity.
    #    A hedged span ("about two weeks ago") resolves the same way but is
    #    reported as approximate.
    hit = _try_get_date(raw, base_date, "past")
    if hit is not None:
        return ResolvedDate(resolved_date=hit[0], granularity=hit[1],
                            date_type="approximate" if _is_hedged(raw) else "exact",
                            reason="<Resolved relative date with past reference>")

    # 2. Parentheses often hold the actual date ("yesterday (Friday)"). Skip
    #    parentheses that just contain explanatory prose.
    paren = re.search(r"\(([^)]+)\)", raw)
    if paren:
        content = paren.group(1).strip()
        if not re.search(r"\b(when|specific|exact|not provided|context|implies|but|described)\b", content, re.IGNORECASE):
            res = recurse(content, "date from parentheses")
            if res is not None:
                return res

    # 3. Number-word "ago" ("a few days ago", "two weeks ago"). Vague counts
    #    ("a few", "several") are approximate and resolved one unit coarser;
    #    definite counts ("two", digits) are exact unless the span hedges them
    #    ("up to three weeks ago"). Classification is by the matched word, not by
    #    which branch happens to catch it.
    ago = re.search(r"\b(a few|few|couple|several|one|two|three|four|five)\s+(day|week|month)s?\s+ago", raw, re.IGNORECASE)
    if ago:
        word, unit = ago.group(1).lower(), ago.group(2).lower()
        days = _phrase_to_days(f"{word} {unit}")
        if days is not None:
            coarser = {"day": "week", "week": "month", "month": "year"}[unit]
            if word in {"a few", "few", "couple", "several"}:
                return ResolvedDate(resolved_date=base_date - timedelta(days=days),
                                    granularity=coarser, date_type="approximate",
                                    reason=f"<Approximate '{word} {unit}s ago' relative to pub date>")
            if _is_hedged(raw):
                return ResolvedDate(resolved_date=base_date - timedelta(days=days),
                                    granularity=unit, date_type="approximate",
                                    reason=f"<Hedged '{word} {unit}s ago' relative to pub date>")
            return ResolvedDate(resolved_date=base_date - timedelta(days=days),
                                granularity=unit, date_type="exact",
                                reason=f"<Resolved '{word} {unit}s ago' relative to pub date>")

    # 4. "N-day-old" / "N-week-old" / "N-month-old": count back from pub date.
    age = re.search(r"(\d+)[\s-]*(day|week|month)s?[\s-]*old", low)
    if age:
        num, unit = int(age.group(1)), age.group(2)
        return ResolvedDate(resolved_date=base_date - timedelta(days=num * _UNIT_TO_DAYS[unit]),
                            granularity=unit, date_type="exact",
                            reason=f"<Resolved '{age.group(0)}' counting back from pub date>")

    # 5. "X before/after Y": resolve Y (recursively, it may be relative) and
    #    shift by the offset. The result is a specific day.
    rel = re.search(r"((?:a few|a|an|one|two|three|four|five|\d+)?\s*(?:days?|weeks?|months?))\s+(before|after)\s+(.+)", raw, re.IGNORECASE)
    if rel:
        days = _phrase_to_days(rel.group(1))
        if days is not None:
            anchor = _resolve_core(rel.group(3).strip(), base_date)
            if anchor is not None and anchor.resolved_date is not None:
                direction = rel.group(2).lower()
                shifted = (anchor.resolved_date - timedelta(days=days)) if direction == "before" \
                    else (anchor.resolved_date + timedelta(days=days))
                return ResolvedDate(resolved_date=shifted, granularity="day", date_type="exact",
                                    reason=f"<'{rel.group(1).strip()} {direction}' offset ← {anchor.reason.strip('<>')}>")

    # 6. Open-ended range ending at the present ("March 2024 onwards",
    #    "since June to present"). Resolve to the start; it's a range with no end.
    onwards = re.search(r"^(.+?)\s+(?:onwards?|to present|to date|to the present|till date)\b", raw, re.IGNORECASE)
    if onwards:
        res = recurse(onwards.group(1), f"open-ended range '{raw}', resolved to start (ongoing)",
                      date_type="range", date_end=None)
        if res is not None:
            return res

    # 7. Bounded range ("between X and Y", "March 15 to March 20",
    #    "Monday through Wednesday", "January 2020 - March 2021"). Resolve to the
    #    first endpoint and capture the second as date_end; granularity is
    #    inherited from the start.
    #
    #    Two guards keep this branch from over-firing on prose. First, the
    #    separators are words plus an *spaced* dash -- a bare "-" would wreck
    #    "mid-March" and ISO dates, but " - " with whitespace on both sides is
    #    safe. Second, we only call it a range if the right-hand side actually
    #    resolves to a date: "Monday to discuss the deal" should give us Monday,
    #    not a bogus one-ended range.
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)", raw, re.IGNORECASE)
    if between:
        first, second = between.group(1), between.group(2)
    else:
        sep = re.search(r"^(.+?)\s+(?:through|thru|to|till|until|and)\s+(.+)$", raw, re.IGNORECASE) \
            or re.search(r"^(.+?)\s*[–—]\s*(.+)$", raw) \
            or re.search(r"^(.+?)\s+-\s+(.+)$", raw)
        first, second = (sep.group(1), sep.group(2)) if sep else (None, None)
    if first is not None:
        # Drop leading open-ended words so "from January to March" -> "January".
        first_clean = re.sub(r"^(from|since|starting|beginning)\s+", "", first, flags=re.IGNORECASE)
        end_res = _resolve_core(second.strip(), base_date) if second else None
        end_date = end_res.resolved_date if (end_res is not None and end_res.resolved_date is not None) else None
        if end_date is not None:
            res = recurse(first_clean, f"range '{first.strip()}'–'{second.strip()}', resolved to start",
                          date_type="range", date_end=end_date)
            if res is not None:
                return _align_range(res, second, base_date)
        else:
            # Right-hand side isn't a date, so this isn't a range: keep whatever
            # the left-hand side resolves to and leave its type alone.
            res = recurse(first_clean, f"resolved '{first.strip()}'; '{second.strip()}' is not a date")
            if res is not None:
                return res

    # 8. Open-ended range starting at a date ("since March 2024",
    #    "starting in April", "from January 2020"). Resolve to the start.
    #     The negative lookahead keeps "beginning of <period>" out of this branch
    #     (that's a within-period reference handled at step 13, not an open-ended
    #     range); "beginning in April" still reads as "starting in April".
    since = re.search(r"^(?:starting|beginning(?!\s+of\b)|since|from)\s+(.+)", raw, re.IGNORECASE)
    if since:
        date_part = re.sub(r"^(in|on|the)\s+", "", since.group(1).strip(), flags=re.IGNORECASE)
        res = recurse(date_part, f"open-ended range, resolved to start ('{date_part}')",
                      date_type="range", date_end=None)
        if res is not None:
            return res

    # 9. Quarter references ("Q2 2023", "first quarter of 2024"). Resolve to the
    #    first day of the quarter, with a quarter-level granularity.
    quarter = re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th|q[1-4])\s+quarter\s+(?:of\s+)?(\d{4})", raw, re.IGNORECASE) \
        or re.search(r"\b(q[1-4])\s+(\d{4})", raw, re.IGNORECASE)
    if quarter:
        month = _QUARTER_TO_MONTH.get(quarter.group(1).lower())
        if month is not None:
            return ResolvedDate(resolved_date=datetime(int(quarter.group(2)), month, 1),
                                granularity="quarter", date_type="exact",
                                reason=f"<Resolved quarter reference '{quarter.group(0)}' to start of quarter>")

    # 10. Future references ("next Tuesday", "later this week", "coming Monday").
    #     "later" is a vague within-period qualifier we can't honor (like
    #     "early"/"mid"), so a "later ..." resolution is approximate; "next" /
    #     "coming" / "upcoming" point at a specific occurrence and are exact.
    if re.search(r"\b(next|later|coming|upcoming)\b", raw, re.IGNORECASE):
        cleaned = re.sub(r"\b(next|later|coming|upcoming)\b", "", raw, flags=re.IGNORECASE).strip(" -")
        hit = _try_get_date(cleaned, base_date, "future")
        if hit is not None:
            vague = re.search(r"\blater\b", raw, re.IGNORECASE) is not None
            return ResolvedDate(resolved_date=hit[0], granularity=hit[1],
                                date_type="approximate" if vague else "exact",
                                reason="<Resolved relative date with future reference>")

    # 11. Past-tense modifiers. dateparser handles "last year/month/week/decade"
    #     natively but not the synonyms ("previous year") nor "last <weekday>".
    #     First normalize the synonyms to "last" and retry the raw parse; if that
    #     fails, strip the modifier for weekday-style references ("last Tuesday").
    #     Bare period words are skipped because dateparser resolves them to
    #     *today*, which would be a garbage result ("previous year" -> "year").
    if re.search(r"\b(last|previous|prior|past|earlier)\b", raw, re.IGNORECASE):
        normalized = re.sub(r"\b(previous|prior|past|earlier)\b", "last", raw, flags=re.IGNORECASE)
        if normalized.lower() != raw.lower():
            hit = _try_get_date(normalized, base_date, "past")
            if hit is not None:
                return ResolvedDate(resolved_date=hit[0], granularity=hit[1], date_type="exact",
                                    reason="<Resolved after normalizing past-tense modifier to 'last'>")
        cleaned = re.sub(r"\b(last|previous|prior|past|earlier)\b", "", raw, flags=re.IGNORECASE).strip(" -,")
        if cleaned.lower() not in {"year", "month", "week", "day", "decade", "quarter", "weekend"}:
            res = recurse(cleaned, "removed past-tense modifier")
            if res is not None:
                return res

    # 12. Weekend references ("over the weekend"). Approximate to the preceding
    #     Saturday; we only know it happened within that week, so the
    #     granularity is week-level and the type is approximate.
    if re.search(r"\bweekend\b", raw, re.IGNORECASE):
        hit = _try_get_date("Saturday", base_date, "past")
        if hit is not None:
            return ResolvedDate(resolved_date=hit[0], granularity="week", date_type="approximate",
                                reason="<Resolved weekend reference, approximated to the preceding Saturday>")

    # 13. Approximate within-period modifiers ("early May", "mid-March"). These
    #     usually resolve at month granularity, so the day component is filler;
    #     the source is vague, so the type is approximate.
    mod = re.search(r"\b(early|mid|late|beginning of|end of)\b", raw, re.IGNORECASE)
    if mod:
        cleaned = re.sub(r"\b(early|mid|late|beginning of|end of)\b", "", raw, flags=re.IGNORECASE)
        # A bare period word ("the year", "year", "the month") left after
        # stripping the modifier would parse to *today*; anchor it to the
        # pub-date period instead. Other bare periods (week/quarter/decade) are
        # left unanchored -- returning None is safer than emitting today.
        period = re.sub(r"\b(of|the|this|current|in)\b", "", cleaned, flags=re.IGNORECASE).strip(" -,")
        if period.lower() in {"year", "month", "week", "quarter", "decade"}:
            return _anchor_bare_period(mod.group(1), period.lower(), base_date)
        res = recurse(cleaned, "approximate within-period reference (early/mid/late), day is filler",
                      date_type="approximate")
        if res is not None:
            return res

    # 14. Explicit date embedded in noise ("violence started March 5",
    #     "8 July"). Extract and resolve it.
    explicit = re.search(rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b", raw, re.IGNORECASE) \
        or re.search(rf"\b\d{{1,2}}\s+(?:{_MONTHS})(?:,?\s+\d{{4}})?\b", raw, re.IGNORECASE)
    if explicit:
        res = recurse(explicit.group(0), "extracted explicit date")
        if res is not None:
            return res

    # 15. Last resort: strip time-of-day and event-onset noise, then re-resolve.
    cleaned = re.sub(r"\b(night|morning|afternoon|evening)\b|early hours of", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(the\s+)?(violence|trouble|conflict|unrest|fighting|clashes?|protests?|war)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(broke out|started|began|erupted|happened|occurred)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,")
    res = recurse(cleaned, "stripped noise words")
    if res is not None:
        return res

    return None


def _resolve_date(date_string: str | None=None,
                  ref_date: str | datetime | date | None=None
                  ) -> ResolvedDate:
    """
    Resolve Q&A string date to a specific date.

    The resolver hands the string to :func:`_resolve_core`, which tries a
    cascade of strategies (direct dateparser parse, parentheses, "N units ago",
    "N-day-old", "before/after", ranges, "since/from", quarters, future/past
    modifiers, weekend, "early/mid/late", embedded explicit dates, noise
    stripping). The result carries three orthogonal pieces of resolution
    information -- see :class:`ResolvedDate`:

    - ``granularity``: the precision *unit* (day/week/month/quarter/year).
    - ``date_type``: exact / approximate / range / unresolved.
    - ``date_end``: the end of a genuine range, else ``None``.

    Parameters
    ----------
    date_string : str | None
        Text that contains a date, possibly a relative date reference.
    ref_date : str | datetime | date | None
        Date to use as reference date, e.g. article publication date.

    Returns
    -------
    ResolvedDate
        Dataclass with resolved date and reason for resolution

    Examples
    --------
    >>> _resolve_date("yesterday", "2021-01-02")
    ResolvedDate(resolved_date=datetime(2021, 1, 1), date_end=None, granularity="day", date_type="exact", reason="...")
    """
    na = set([None, ""])

    # Either or both of the inputs might be missing, however, before we handle
    # those possibilities, we need make sure we don't end up with a missing
    # pub date due to failure to parse a non-missing string.
    if ref_date not in na:
        orig = ref_date
        ref_date = dateparser.parse(str(ref_date))
        if ref_date is None:
            logger.warning(f"<Failed to parse reference date: {orig}>")
        else:
            # Normalize to a naive midnight. Publication dates sometimes arrive
            # with a time and a timezone ("2019-08-01T14:30:00+00:00"), and if we
            # keep those, some branches below return tz-aware datetimes (day
            # arithmetic off the reference) while others return naive ones
            # (dates we construct outright). A column holding both is painful to
            # work with in pandas, and the time-of-day is below our resolution.
            ref_date = ref_date.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)

    # Handle cases were inputs are incomplete
    match (date_string in na, ref_date is None or ref_date in na):
        case (True, True):
            return ResolvedDate(resolved_date=None,
                                date_type="unresolved",
                                reason=f"<No date string or publication date, date_string={date_string}, ref_date={ref_date}>")
        case (True, False):
            return ResolvedDate(resolved_date=ref_date,
                                date_type="unresolved",
                                reason="<No date string, using pub date>")
        case (False, True):
            # Don't fall back to 'today' as backup date, but maybe in the future
            # make that a configurable option
            return ResolvedDate(resolved_date=None,
                                date_type="unresolved",
                                reason="<No publication date>")
        case (False, False):
            # We have both inputs and can proceed
            base_date = ref_date

    date_string = str(date_string).strip()

    # Handle the case where the attribute model return "N/A". Explicitly 
    # return that, rather than a more generic response that might sound like a parsing
    # failure.
    if date_string.lower() in _NOISE_PHRASES:
        return ResolvedDate(resolved_date=ref_date,
                            date_type="unresolved",
                            reason=f"<No date given in the text ('{date_string}'), using pub date>")

    # First check for a bare weekday that's the publication day itself. Check before
    # starting _resolve_core.
    same_day = _same_day_weekday(date_string, base_date)
    if same_day is not None:
        return same_day

    res = _resolve_core(date_string, base_date)
    if res is not None:
        return res

    # If we're here, nothing worked (e.g. non-Gregorian calendars like "last Ramadan" or
    # references needing world knowledge like "the anniversary of the
    # uprising"). Fall back to the publication date and flag as unresolved.
    return ResolvedDate(resolved_date=ref_date,
                        date_type="unresolved",
                        reason="<dateparser failed to convert relative date, using pub date>")




def pick_event_loc(search_term: str | None, 
                   geolocated_ents: list[dict | None],
                   geo_overlap_threshold = 0.5,
                   geo_confidence_threshold = 0.85) -> dict:
    na_equiv = [None, "", "N/A", "NA", "n/a", "na"]

    # Handle all 4 combinations of missing search term or empty geo_entities
    match (search_term in na_equiv, not geolocated_ents):
        case (False, False):
            # Both search term and geo entities are present, fallthrough to 
            # logic below
            pass
        case (False, True):
            return {"event_loc": None, "reason": "no geo entities"}
        case (True, False):
            return {"event_loc": None, "reason": "no search term"}
        case (True, True):
            return {"event_loc": None, "reason": "no search term and no geo entities"}

    # Calculate word overlap fraction between search term and each geo entity 
    # search name
    overlaps = [word_overlap_fraction(search_term, geo_entity.get("search_name", "")) for geo_entity in geolocated_ents]
    if max(overlaps) < geo_overlap_threshold:
        return {"event_loc": None, "reason": "no sufficient overlap in search terms"}
    best_match = geolocated_ents[overlaps.index(max(overlaps))]
    if best_match.get("score", 0.0) < geo_confidence_threshold:
        return {"event_loc": None, "reason": "no sufficient confidence in geo entity"}
    return {"event_loc": best_match, "reason": "success"}


def word_overlap_fraction(word1: str, word2: str) -> float:
    """
    Calculate the overlap between two words as a fraction of their aligned length.
    
    Args:
        word1: First word
        word2: Second word
    
    Returns:
        Float between 0 and 1 representing the overlap ratio
    """
    if not word1 or not word2:
        return 0.0
    
    if word1 == word2:
        return 1.0
    
    max_overlap = 0
    len1, len2 = len(word1), len(word2)
    
    # Check all possible alignments
    # word1 shifted right relative to word2
    for i in range(len1):
        overlap = sum(1 for j in range(min(len1 - i, len2)) 
                     if word1[i + j] == word2[j])
        max_overlap = max(max_overlap, overlap)
    
    # word2 shifted right relative to word1
    for i in range(1, len2):
        overlap = sum(1 for j in range(min(len2 - i, len1)) 
                     if word2[i + j] == word1[j])
        max_overlap = max(max_overlap, overlap)
    
    # The aligned length is the minimum of the two word lengths
    # at the best alignment position
    return max_overlap / max(len1, len2)




class Formatter:
    def __init__(self, quiet=False, country_csv_path: str | None=None, geolocation_threshold=0.85):
        self.quiet = quiet
        self.iso_to_name = country_name_dict(country_csv_path)
        self.geo_threshold = geolocation_threshold

    """
    event = {   'attributes': {   'ACTOR': [{   'qa_end_char': 53,
                                   'qa_score': 0.31743326783180237,
                                   'qa_start_char': 39,
                                   'text': 'Nicolas Maduro',
                                   'score': 0.23675884306430817,
                                   'wiki': 'Nicolás Maduro',
                                   'country': 'VEN',
                                   'code_1': 'ELI',
                                   'code_2': ''}],
                      'LOC': [{   'qa_end_char': 156,
                                 'qa_score': 0.4355418384075165,
                                 'qa_start_char': 148,
                                 'text': 'Barbados'}],
                      'RECIP': [{   'qa_end_char': 90,
                                   'qa_score': 0.1324695497751236,
                                   'qa_start_char': 79,
                                   'score': 0.13248120248317719,
                                   'wiki': 'Juan Guaidó',
                                   'country': 'VEN',
                                   'code_1': 'REB',
                                   'code_2': '',
                                   'text': 'Juan Guaidó'}]},
    'contexts': ['pro_democracy'],
    'date': '2019-08-01',
    'event_geolocation': {   'admin1_code': '00',
                             'admin1_name': '',
                             'admin2_code': '',
                             'admin2_name': '',
                             'country_code3': 'BRB',
                             'end_char': 156,
                             'event_location_overlap_score': 1.0,
                             'feature_class': 'A',
                             'feature_code': 'PCLI',
                             'geonameid': '3374084',
                             'lat': 13.16453,
                             'lon': -59.55165,
                             'resolved_placename': 'Barbados',
                             'score': 1.0,
                             'search_placename': 'Barbados',
                             'start_char': 148},
    'event_mode': [],
    'event_text': 'Delegates of the Venezuelan president, Nicolas Maduro, and '
                  'the leader objector Juan Guaidó resumed on Wednesday (31) '
                  'conversations on the island of Barbados, sponsored by '
                  'Norway, to seek a way out of the crisis in their country, '
                  'announced the parties. "We started another round of '
                  'sanctions under the mechanism of Oslo," indicated on '
                  'Twitter Mr Stalin González, one of the envoys of Guaidó, '
                  'parliamentary leader recognized as interim president by '
                  'half hundred countries. The vice-president of Venezuela, '
                  'Delcy Rodríguez, confirmed in a press conference that '
                  'representatives of mature traveled to Barbados for the '
                  'meetings with the opposition. Mature reaffirmed in a '
                  'message to the nation that the government seeks to '
                  'establish a "bureau for permanent dialog with the '
                  'opposition, and called entrepreneurs and social movements '
                  'to be added to the process. After exploratory '
                  'approximations and a first face to face in Oslo in mid-May, '
                  'the parties have transferred the dialog on 8 July for the '
                  'caribbean island. The opposition search in the negotiations '
                  'the output of mature and a new election, by considering '
                  'that his second term, started last January, resulted from '
                  'fraudulent elections, not recognized by almost 60 '
                  'countries, among them the United States. ',
    'event_type': 'RETREAT',
    'geolocated_ents': [   {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'BRB',
                               'end_char': 156,
                               'event_location_overlap_score': 1.0,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3374084',
                               'lat': 13.16453,
                               'lon': -59.55165,
                               'resolved_placename': 'Barbados',
                               'score': 1.0,
                               'search_placename': 'Barbados',
                               'start_char': 148},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'NOR',
                               'end_char': 177,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3144096',
                               'lat': 62.0,
                               'lon': 10.0,
                               'resolved_placename': 'Kingdom of Norway',
                               'score': 1.0,
                               'search_placename': 'Norway',
                               'start_char': 171},
                           {   'admin1_code': '12',
                               'admin1_name': 'Oslo',
                               'admin2_code': '0301',
                               'admin2_name': 'Oslo',
                               'country_code3': 'NOR',
                               'end_char': 318,
                               'feature_class': 'P',
                               'feature_code': 'PPLC',
                               'geonameid': '3143244',
                               'lat': 59.91273,
                               'lon': 10.74609,
                               'resolved_placename': 'Oslo',
                               'score': 1.0,
                               'search_placename': 'Oslo',
                               'start_char': 314},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'VEN',
                               'end_char': 502,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3625428',
                               'lat': 8.0,
                               'lon': -66.0,
                               'resolved_placename': 'Bolivarian Republic of '
                                                     'Venezuela',
                               'score': 1.0,
                               'search_placename': 'Venezuela',
                               'start_char': 493},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'BRB',
                               'end_char': 604,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3374084',
                               'lat': 13.16453,
                               'lon': -59.55165,
                               'resolved_placename': 'Barbados',
                               'score': 1.0,
                               'search_placename': 'Barbados',
                               'start_char': 596},
                           {   'admin1_code': '12',
                               'admin1_name': 'Oslo',
                               'admin2_code': '0301',
                               'admin2_name': 'Oslo',
                               'country_code3': 'NOR',
                               'end_char': 918,
                               'feature_class': 'P',
                               'feature_code': 'PPLC',
                               'geonameid': '3143244',
                               'lat': 59.91273,
                               'lon': 10.74609,
                               'resolved_placename': 'Oslo',
                               'score': 1.0,
                               'search_placename': 'Oslo',
                               'start_char': 914},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'USA',
                               'end_char': 1259,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '6252001',
                               'lat': 39.76,
                               'lon': -98.5,
                               'resolved_placename': 'United States',
                               'score': 1.0,
                               'search_placename': 'United States',
                               'start_char': 1239}],
    'headline': 'Governo e oposição da Venezuela retomam diálogo em Barbados\n',
    'id': '20190801-2309-4e081644904c_COOPERATE_R',
    'pub_date': '2019-08-01',
    'publisher': 'translateme2-pt',
    'story_id': 'AFPPT00020190801ef81000jh:50066619',
    'story_people': ['Delcy Rodríguez', 'Guaidó', 'Nicolas Maduro', 'Stalin González', 'Juan Guaidó'],
    'story_orgs': ['Mature'],
    'story_locs': ['Norway', 'United States', 'Barbados', 'Oslo', 'Venezuela'],
    'version': 'NGEC_coder-Vers001-b1-Run-001'}
    """

    def find_event_loc(self, event, geo_overlap_thresh=0.5):
        if 'LOC' not in event['attributes'].keys():
            event['event_geolocation'] = {"reason": "No LOC attribute found by the QA/attribute model",
                                          "geo": None}
            return event
        try:
            event_loc_raw = event['attributes']['LOC'][0] ## NOTE!! Assuming just one location from the QA model
        except IndexError:
            event['event_geolocation'] = {"reason": "No LOC attribute found by the QA/attribute model",
                                          "geo": None}
            return event
        if not event_loc_raw:
            event['event_geolocation'] = {"reason": "No LOC attribute found by the QA/attribute model",
                                          "geo": None}
            return event
        if 'geolocated_ents' not in event.keys():
            event['event_geolocation'] = {"reason": "No story locations were geolocated (Missing 'geolocated_ents' key).",
                                          "geo": None}
            return event
        event_loc_chars = set(range(event_loc_raw['qa_start_char'], event_loc_raw['qa_end_char']))
        geo_ent_ranges = [set(range(i['start_char'], i['end_char'])) for i in event['geolocated_ents']]
        # calculate intersection-over-union/Jaccard
        ious = np.array([len(event_loc_chars.intersection(i)) / len(event_loc_chars.union(i)) for i in geo_ent_ranges])
        if len(ious) == 0:
            event['event_geolocation'] = {"reason": f"No geolocated entities",
                                              "geo": None}
            return event
        try:
            if np.max(ious) < geo_overlap_thresh:
                event['event_geolocation'] = {"reason": f"Attribute placename ({event_loc_raw['text']}) [doesn't overlap enough with any placenames: {str(np.max(ious))}",
                                              "geo": None}
                return event
        except ValueError:
            event['event_geolocation'] = {"reason": f"Problem with intersection-overlap vector. No elements?",
                                              "geo": None}
            return event
        best_match = event['geolocated_ents'][np.argmax(ious)]
        if not best_match:
            event['event_geolocation'] = {"reason": f"No 'best_match' geolocated entity",
                                              "geo": None}
            return event
        best_match['event_location_overlap_score'] = float(np.max(ious))
        if 'score' not in best_match.keys():
            event['event_geolocation'] = {"reason": f"'best_match' identified but no 'score' key. Returning best_match anyway",
                                        "geo": best_match} 
            return event
        if best_match['score'] > self.geo_threshold:
            event['event_geolocation'] = {"reason": f": Successful overlap between attribute placename and one of the geoparser results",
                                        "geo": best_match}
            return event
        else:
            event['event_geolocation'] = {"reason": f": Successful overlap between attribute placename and one of the geoparser results BUT geoparser score was too low ({best_match['score']})",
                                        "geo": None}
            return event




    def add_meta(self, event):
        """
        Add optional metadata to the event dictionary (e.g. alternative country codes, country names,
        event intensity, event quad class, etc.)
        """
        for k, att in event['attributes'].items():
            # add stuff to actors and recipients
            if k in ["LOC", "DATE"]:
                continue
            for v in att:
                try:
                    v['country_name'] = self.iso_to_name[v['country']]
                except:
                    print(v['country'])
                    v['country_name'] = ""

        return event


    def process(self, event_list, return_raw=False):
        """
        Create and write out a final cleaned dictionary/JSON file of events.

        Parameters
        ----------
        event_list: list of dicts
          list of events after being passed through each of the processing steps
        return_raw: bool
          If true, don't write to a final and instead return the final version. Useful for 
          debugging. Defaults to False.
        """
        for n, event in enumerate(event_list):
            # 'attributes' is a single dict (one event per record).
            attributes = event.get('attributes') or {}
            event["event_location"] = pick_event_loc(
                _first_attribute_value(attributes.get('location')),
                event.get('geolocated_ents', []),
                geo_confidence_threshold=self.geo_threshold
            )
            try:
                resolve_date(event)
            except Exception as exception:
                logger.warning(f"{exception} parsing date for event number {n}")

        if return_raw:
            return event_list
        else:
            with jsonlines.open("events_processed.jsonl", "w") as f:
                f.write_all(event_list)

