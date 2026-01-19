
from dataclasses import asdict, dataclass
from datetime import date, datetime
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


def resolve_date(event: dict) -> dict:
    """
    Create a new 'date_resolved' key with a date in YYYY-MM-DD format
    >>> DateDataParser().get_date_data('March 2015')
    DateData(date_obj=datetime.datetime(2015, 3, 16, 0, 0), period='month', locale='en')
    """
    date_string = event.get('attributes', {}).get('date', [{}])[0]
    pub_date = event.get('pub_date', None)
    res = _resolve_date(date_string=date_string, ref_date=pub_date)
    event['date_resolved'] = asdict(res)
    return event


@dataclass
class ResolvedDate:
    resolved_date: datetime | str | None
    granularity: str | None
    reason: str | None

    def _post_init__(self):
        if isinstance(self.resolved_date, str):
            try:
                self.resolved_date = datetime.strptime(self.resolved_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"resolved_date string not in YYYY-MM-DD format: {self.resolved_date}")


def _resolve_date(date_string: str | None=None, 
                  ref_date: str | datetime | date | None=None
                  ) -> ResolvedDate:
    """
    Resolve Q&A string date to a specific date

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
    >>> _resolve_date("yesterday", "2021-01-01")
    ResolvedDate(resolved_date=datetime(2021, 1, 1), granularity="day", reason="Resolved relative date with past reference")
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

    # Handle cases were inputs are incomplete
    match (date_string in na, ref_date is None or ref_date in na):
        case (True, True):
            return ResolvedDate(resolved_date=None, 
                                granularity=None,
                                reason=f"<No date string or publication date, date_string={date_string}, ref_date={ref_date}>")
        case (True, False):
            return ResolvedDate(resolved_date=ref_date, 
                                granularity="uncertain", 
                                reason="<No date string, using pub date>")
        case (False, True):
            # Don't fall back to 'today' as backup date, but maybe in the future
            # make that a configurable option
            return ResolvedDate(resolved_date=None, 
                                granularity=None, 
                                reason="<No publication date>")
        case (False, False):
            # We have both inputs and can proceed
            base_date = ref_date

    DateParser = dateparser.DateDataParser(languages=['en'], settings={'RELATIVE_BASE': base_date, 'PREFER_DATES_FROM': "past"})
    res = DateParser.get_date_data(date_string)

    # Did we succeed?
    if res.date_obj is not None:
        res = ResolvedDate(resolved_date=res.date_obj, 
                           granularity=res.period, 
                           reason="<Resolved relative date with past reference>")
    else:
        # Check whether we have a future reference
        future_pattern = r"next|later"
        if re.search(future_pattern, date_string):
            date_string = re.sub(future_pattern, "", date_string).strip()
            DateParser = dateparser.DateDataParser(languages=['en'], settings={'RELATIVE_BASE': base_date, 'PREFER_DATES_FROM': "future"})
            res = DateParser.get_date_data(date_string)

            if res.date_obj is not None:
                res = ResolvedDate(resolved_date=res.date_obj, 
                                   granularity=res.period,
                                   reason="<Resolved relative date with future reference>")
            else:
                # If still not resolved, use publication date
                res = ResolvedDate(resolved_date=ref_date, 
                                   granularity="uncertain",
                                   reason="<dateparser failed to convert future relative date, using pub date>")
        # Nope, no future reference so dateparser just failed
        else:
            res = ResolvedDate(resolved_date=ref_date, 
                               granularity= "uncertain", 
                               reason="<dateparser failed to convert relative date, using pub date>")
    
    return res
    



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
            #if n == 0:
            #    print(e)
            #event = self.find_event_loc(event)
            #event = self.add_meta(event)
            event["event_location"] = pick_event_loc(
                event.get('attributes', ["N/A"]).get('location', ["N/A"])[0],
                event.get('geolocated_ents', []),
                geo_confidence_threshold=self.geo_threshold
            )
            try:
                event = resolve_date(event)
            except Exception as exception:
                logger.warning(f"{exception} parsing date for event number {n}")
        
        if return_raw:
            return event_list
        else:
            with jsonlines.open("events_processed.jsonl", "w") as f:
                f.write_all(event_list)

