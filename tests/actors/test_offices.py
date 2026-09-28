"""Office parsing from Wikipedia infoboxes, and the known_country argument.

No models or Elasticsearch: the parser only needs a country detector, and the
agent matcher it would otherwise load is replaced by a stand-in.
"""
from datetime import date, datetime

from ngec.actors.actor_resolution import COUNTRY_NAMES, WikiParser


def make_parser():
    return WikiParser(agent_matcher=object())


def test_house_seat_without_office_field():
    # Alexandria Ocasio-Cortez's infobox, as stored in the wiki index: a
    # state and a district, no `office`.
    infobox = {"term_start": "January 3, 2019", "district": "", "term_end": "",
               "state": "New York"}
    offices = make_parser().parse_offices(infobox)
    assert [o["office"] for o in offices] == [
        "Member of the U.S. House of Representatives from New York"]
    assert offices[0]["term_start"] == datetime(2019, 1, 3)


def test_senate_seat_next_to_other_offices():
    # Mitch McConnell, trimmed: a numbered Senate seat alongside `office`.
    infobox = {"office": "Chair of the Senate Rules Committee", "term_start": "January 3, 2025",
               "term_end": "", "state8": "Kentucky", "jr/sr8": "United States Senator",
               "term_start8": "January 3, 1985", "term_end8": ""}
    parser = make_parser()
    offices = parser.parse_offices(infobox)
    assert "United States Senator from Kentucky" in [o["office"] for o in offices]
    current, _ = parser.get_current_office(offices, "2020-06-01")
    assert [o["office"] for o in current] == ["United States Senator from Kentucky"]


def test_state_and_district_outside_the_us_are_not_a_house_seat():
    infobox = {"state": "Maharashtra", "district": "Pune", "term_start": "2014"}
    assert make_parser().parse_offices(infobox) == []


def test_numbered_office_is_not_duplicated():
    # A state next to an `office` with the same number is that office's.
    infobox = {"office2": "Governor of Texas", "state2": "Texas", "district2": "",
               "term_start2": "January 17, 1995", "term_end2": "December 21, 2000"}
    offices = make_parser().parse_offices(infobox)
    assert [o["office"] for o in offices] == ["Governor of Texas"]


def test_current_office_accepts_a_date_object():
    # A plain date used to be compared with datetimes, raise, and be caught,
    # so no office ever counted as current.
    parser = make_parser()
    offices = parser.parse_offices({"office": "Chancellor of Germany",
                                    "term_start": "November 22, 2005",
                                    "term_end": "December 8, 2021"})
    current, _ = parser.get_current_office(offices, date(2010, 1, 1))
    assert [o["office"] for o in current] == ["Chancellor of Germany"]


def test_known_country_codes_map_to_names():
    assert COUNTRY_NAMES["DEU"] == "Germany"
    assert COUNTRY_NAMES["USA"] == "United States"
    assert "IGO" not in COUNTRY_NAMES
