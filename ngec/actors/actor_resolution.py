from collections import Counter
from copy import deepcopy
from importlib import resources
import logging
import re
import time

import dateparser
from elasticsearch import Elasticsearch
import jsonlines
from rich.progress import track

from .common import ModelManager, clean_query, CountryDetector, strip_ents
from .agent_matcher import AgentMatcher
from .wiki_matcher import WikiMatcher


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



# Threshold constants. The only confidence high enough to skip the Wikipedia
# lookup on an agent match; see the policy comment in actor_to_code.
THRESHOLD_VERY_HIGH_CONFIDENCE = 0.95
THRESHOLD_CONFIDENT_NO_CONTEXT = 0.6  # agent match that settles a span when no context is given


# Words that make a *single-token* NER core entity worthless as a search term.
# spaCy will happily tag "House" in "House Judiciary" or "Fox" in "President
# Fox" as the entity, and the pipeline then throws the rest of the span away
# and searches Wikipedia for "House". These are the tokens that, on their own,
# name nobody: articles and determiners, honorifics and job titles, and the
# bare institution nouns that are meaningless without their qualifier.
DEGENERATE_CORE_WORDS = (
    # articles / determiners
    "the", "a", "an", "this", "that", "these", "those", "his", "her", "its",
    "their", "our", "some", "such",
    # honorifics and job titles
    "mr", "mrs", "ms", "dr", "sir", "president", "vice", "prime", "minister",
    "secretary", "senator", "representative", "governor", "mayor", "chairman",
    "chairwoman", "chief", "director", "commissioner", "ambassador", "general",
    "colonel", "captain", "spokesman", "spokeswoman", "spokesperson", "leader",
    "official", "officials", "head", "deputy", "acting", "former",
    # bare institution nouns
    "house", "senate", "congress", "parliament", "assembly", "ministry",
    "department", "court", "council", "committee", "commission", "party",
    "government", "administration", "state", "office", "authority", "agency",
    "bureau", "bank", "force", "forces", "army", "navy", "police", "union",
)


# Head nouns (as spaCy lemmas) of spans that name a *category* of people
# rather than an actor. The ontology wants a role code for these -- "police"
# is COP, "protesters" is CVL -- and Wikipedia has nothing useful to add:
# linking "police" to the article `Police`, or "residents" to `The Residents`,
# is always wrong. Measured on a probe set of real VOA sentences, the linker
# put a page on 61% of these.
COLLECTIVE_HEAD_NOUNS = (
    "police", "officer", "protester", "protestor", "demonstrator", "rioter",
    "resident", "civilian", "villager", "worker", "student", "refugee",
    "migrant", "soldier", "troop", "force", "gunman", "militant", "rebel",
    "insurgent", "fighter", "activist", "supporter", "voter", "official",
    "authority", "militia", "crowd", "mob", "youth", "farmer", "teacher",
    "doctor", "journalist", "lawmaker", "attacker", "public",
)

# The same idea, but only when the span is nothing *but* one of these words.
# "the government" is a role; "the Government Accountability Office" is a
# thing with a Wikipedia page. Note that nationality stripping runs first, so
# "the Nigerian Army" arrives here as "Army".
COLLECTIVE_BARE_TERMS = (
    "government", "opposition", "army", "military",
)

# Modifiers that turn a collective noun back into a named institution with a
# page of its own: "police" is a role, "the national police" is
# `National Police of Colombia`.
INSTITUTIONAL_MODIFIERS = (
    "national", "federal", "royal", "state", "supreme", "central",
    "presidential", "republican",
)

# Head nouns of spans that name a *body*, not a category of people. A news
# story writes "the interior ministry" or "the supreme court" and means one
# specific institution in one specific country, which is exactly what the
# Wikipedia linker is for: `Ministry of Home Affairs (Tanzania)`,
# `Supreme Court of Kenya`. Matched on the surface form, not the lemma, so
# that "authorities" stays a generic collective while "authority" (as in
# "the Palestinian Authority") does not.
INSTITUTION_HEAD_NOUNS = (
    "ministry", "department", "court", "assembly", "parliament", "senate",
    "congress", "legislature", "council", "commission", "committee", "agency",
    "bureau", "authority", "bank", "party", "cabinet", "presidency", "office",
)

# These name an institution only when something says *whose*: "the national
# police" is `National Police of Colombia`, but "police" on its own is a role.
INSTITUTION_MODIFIED_HEAD_NOUNS = (
    "police", "army", "forces", "guard",
)


# Spans that refer to people without naming them. A pronoun has no referent
# to look up, and an indefinite phrase ("two men", "a group of men", "some
# residents") names a quantity, not an actor; both used to be caught by the
# loose confidence clauses in the old gate.
PRONOUNS = (
    "he", "she", "they", "them", "him", "her", "we", "us", "i", "me", "you",
    "it", "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
    "nobody", "others",
)

# Words that open an indefinite phrase. Numbers are caught separately with
# spaCy's like_num, which also handles digits.
INDEFINITE_STARTERS = (
    "a", "an", "some", "several", "many", "few", "both", "another", "other",
    "dozens", "hundreds", "thousands",
)

# Collective synonyms that only ever appear bare: "cops", "two men", "people".
UNLINKABLE_BARE_NOUNS = (
    "cop", "cops", "man", "men", "woman", "women", "person", "people",
    # adjectives used as collective nouns: "the displaced", "the wounded"
    "displaced", "wounded", "injured", "dead", "missing", "homeless",
    "unemployed", "poor",
)


def span_is_unlinkable_reference(doc):
    """
    Does this span refer to people without naming anyone?

    Three cases, all of which the Wikipedia linker can only get wrong: a
    pronoun ("he", "they"), an indefinite phrase whose head is a common noun
    ("two men", "a group of men", "some residents"), and a bare collective
    synonym ("cops", "people"). As with the collective filter, a span with a
    PERSON/ORG/GPE/NORP entity, a capitalised word past the start, or an
    institutional head noun is left alone.

    Args:
        doc: spaCy Doc of the span, after nationality stripping

    Returns:
        bool: True if the caller should skip the Wikipedia lookup
    """
    if any(e.label_ in ("PERSON", "ORG", "GPE", "NORP") for e in doc.ents):
        return False
    if span_names_an_institution(doc):
        return False

    words = [t for t in doc if t.is_alpha or t.like_num]
    if not words:
        return False
    if words[0].lower_ in PRONOUNS:
        return len(words) == 1
    if any(t.text[0].isupper() for t in words[1:]):
        return False

    article = words[0].lower_ in ("the", "a", "an")
    body = words[1:] if article else words
    if not body:
        return False

    head = body[-1]
    if len(body) == 1 and head.lower_ in UNLINKABLE_BARE_NOUNS:
        return True
    # A single lowercase word that is not a proper noun ("yesterday",
    # "displaced") names nothing Wikipedia could have an article about.
    if len(body) == 1 and head.text[:1].islower() and head.pos_ != "PROPN":
        return True
    # An indefinite phrase: "two men", "a group of men", "some residents".
    opener = words[0]
    if opener.like_num or opener.lower_ in INDEFINITE_STARTERS:
        return head.pos_ == "NOUN" and head.text[:1].islower()
    return False


def span_names_an_institution(doc):
    """
    Does this span name an institution -- a ministry, a court, a parliament?

    Args:
        doc: spaCy Doc of the span, after nationality stripping

    Returns:
        bool: True if the span should reach the Wikipedia linker
    """
    words = [t for t in doc if t.is_alpha]
    if words and words[0].lower_ in ("the", "a", "an"):
        words = words[1:]
    if not words:
        return False

    head = words[-1]
    if head.lower_ in INSTITUTION_HEAD_NOUNS:
        return True
    if head.lower_ in INSTITUTION_MODIFIED_HEAD_NOUNS:
        return any(t.lower_ in INSTITUTIONAL_MODIFIERS for t in words[:-1])
    return False


def span_is_generic_collective(doc):
    """
    Does this span name a category of people rather than a specific actor?

    True when the span has no PERSON/ORG/GPE/NORP entity, its head noun is in
    COLLECTIVE_HEAD_NOUNS (or the whole span is one of COLLECTIVE_BARE_TERMS),
    and it carries no capitalised word other than its first. That last
    condition is what separates "the security forces" from "Kenya Police":
    a capitalised word past the start of the span is a name, and names are
    what the Wikipedia linker is for.

    Args:
        doc: spaCy Doc of the span, after nationality stripping

    Returns:
        bool: True if the caller should skip the Wikipedia lookup
    """
    if any(e.label_ in ("PERSON", "ORG", "GPE", "NORP") for e in doc.ents):
        return False
    if span_names_an_institution(doc):
        return False

    words = [t for t in doc if t.is_alpha]
    if not words:
        return False
    # Ignore a leading article so that "the Police" counts as a bare span
    # while "Kenya Police" does not.
    if words[0].lower_ in ("the", "a", "an"):
        words = words[1:]
    if not words:
        return False
    if any(t.text[0].isupper() for t in words[1:]):
        return False
    if any(t.lower_ in INSTITUTIONAL_MODIFIERS for t in words[:-1]):
        return False

    head = words[-1]
    if head.lower_ in COLLECTIVE_BARE_TERMS or head.lemma_.lower() in COLLECTIVE_BARE_TERMS:
        return len(words) == 1
    return (head.lower_ in COLLECTIVE_HEAD_NOUNS
            or head.lemma_.lower() in COLLECTIVE_HEAD_NOUNS)


def span_looks_like_organisation(text):
    """
    Does this span look like the name of an organisation rather than a
    nationality?

    Used only for spans that `search_nat` swallows whole. "U.N.", "DPRK",
    "UWSA" and "European Union" are all country *patterns* as far as
    CountryDetector is concerned, but each is also an organisation with a
    Wikipedia article; "Israeli" and "Iranian" are not. The test is an
    internal capital or a period, which is what separates an acronym or a
    multi-word proper name from a nationality adjective.

    Args:
        text: The raw actor span

    Returns:
        bool: True if the span is worth a Wikipedia lookup
    """
    text = text.strip()
    if len(text) < 2:
        return False
    return "." in text or any(c.isupper() for c in text[1:])


def core_query_is_degenerate(core_query, span):
    """
    Is the NER-extracted core entity a worse search term than the span itself?

    Two ways it can be. Either the core is a single token that names nobody
    ("the", "House", "President": see DEGENERATE_CORE_WORDS), or it throws away
    more than half of the letters of an already-short span, which is what
    happens when spaCy tags one word of a two-word name ("President Fox" ->
    "Fox"). The length rule is restricted to spans of four words or fewer so
    that it never fires on the case NER is there for: pulling "Pat Roberts"
    out of "Republican Senator Pat Roberts of Kansas".

    Args:
        core_query: The entity text NER picked out of the span
        span: The span it was picked out of (after nationality stripping)

    Returns:
        bool: True if the caller should search on `span` instead
    """
    core_letters = [c for c in core_query if c.isalpha()]
    span_letters = [c for c in span if c.isalpha()]
    if not core_letters:
        return True

    if len(core_query.split()) == 1 and core_query.strip(".,'").lower() in DEGENERATE_CORE_WORDS:
        return True

    short_span = len(span.split()) <= 4
    dropped_most = len(core_letters) * 2 < len(span_letters)
    return short_span and dropped_most



#######################################################
# Cache Management
#######################################################

def load_actor_priorities(priorities_file=None):
    """
    Load actor code priorities from a CSV asset.

    The file lists one code per line as `code,priority,special`, with `#`
    comments. Priorities are the final tie-break in
    `CodeSelector.pick_best_code`; `special` marks codes that name a polity
    rather than a role, which `clean_best` moves into the country field.

    Pulling this out of the class lets a user supply priorities for their own
    ontology, the same way `agents_file` supplies the patterns. The two go
    together: a custom agents file emits codes that the default priority table
    knows nothing about, so every one of them would otherwise tie at 0 and the
    tie-break would fall through to source order.

    Args:
        priorities_file: path to a priorities CSV. None uses the PLOVER
            priorities shipped in ngec.assets.

    Returns:
        (priorities, special_types): a dict of code -> int, and a list of codes
        flagged special.
    """
    if priorities_file is None:
        priorities_file = str(resources.files("ngec.assets") / "PLOVER_priorities.csv")

    priorities = {}
    special_types = []
    with open(priorities_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                logger.warning(f"Skipping malformed priorities line: {line!r}")
                continue
            code = parts[0]
            try:
                priorities[code] = int(parts[1])
            except ValueError:
                logger.warning(f"Skipping non-integer priority: {line!r}")
                continue
            if len(parts) > 2 and parts[2] not in ("", "0"):
                special_types.append(code)

    logger.debug(f"Loaded {len(priorities)} actor priorities from {priorities_file}")
    return priorities, special_types


class CacheManager:
    """
    Result caching utilities.
    
    This class provides methods for caching and retrieving results.
    
    Example:
        cache = CacheManager()
        result = cache.get("key")
        cache.set("key", value)
    """
    
    def __init__(self):
        """Initialize an empty cache."""
        self.cache = {}
    
    def get(self, key):
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            object: Cached value or None if not found
        """
        return self.cache.get(key)
    
    def set(self, key, value):
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}



#######################################################
# Wikipedia Parser
#######################################################

class WikiParser:
    """
    Parse and extract information from Wikipedia articles.
    
    This class provides methods for extracting actor codes and other
    information from Wikipedia articles.
    
    Example:
        parser = WikiParser()
        offices = parser.parse_offices(wiki_article['infobox'])
        actor_codes = parser.wiki_to_code(wiki_article)
    """

    # TODO #26: allow overriding assets/models
    
    # Actor priorities and special types live in ngec/assets/
    # PLOVER_priorities.csv and are held by CodeSelector, which is the only
    # class that consults them. This class previously carried an unused
    # duplicate of both.
    
    def __init__(self, 
                 country_detector=None, 
                 agent_matcher=None,
                device=None):
        """
        Initialize the Wikipedia parser.
        
        Args:
            country_detector: CountryDetector instance
            agent_matcher: AgentMatcher instance
            device: Device to use for inference ('cuda' or None)
        """
        # Initialize components or use provided ones
        if country_detector is None:
            self.country_detector = CountryDetector()
        else:
            self.country_detector = country_detector
            
        if agent_matcher is None:
            model_manager = ModelManager(device)
            trf_model = model_manager.load_trf_model()
            self.agent_matcher = AgentMatcher(trf_model, 
                                              device=device)
        else:
            self.agent_matcher = agent_matcher
    
    def _parse_office_term_dates(self, infobox, office_key, num=""):
        """
        Parse term start and end dates for an office from an infobox.
        
        Args:
            infobox: Wikipedia infobox data
            office_key: Key for the office in the infobox
            num: Number suffix for additional offices
            
        Returns:
            tuple: (term_start, term_end) as datetime objects or None

        The dates come from English Wikipedia infoboxes, so the parser is told
        the language. Without it, dateparser tries every locale it knows on
        each string, which was 60 of the 65 ms per mention this function cost.
        """
        # Try to get term end date
        term_end = None
        try:
            term_end = dateparser.parse(infobox[f"term_end{num}"], languages=["en"])
        except KeyError:
            try:
                # Sometimes no underscore is used
                term_end = dateparser.parse(infobox[f"termend{num}"], languages=["en"])
            except KeyError:
                pass
        
        # Try to get term start date
        term_start = None
        try:
            term_start = dateparser.parse(infobox[f"term_start{num}"], languages=["en"])
        except KeyError:
            try:
                # Sometimes no underscore is used
                term_start = dateparser.parse(infobox[f"termstart{num}"], languages=["en"])
            except KeyError:
                pass
                
        return term_start, term_end

    def parse_offices(self, infobox):
        """
        Extract office information from a Wikipedia infobox.
        
        Args:
            infobox: Wikipedia infobox data
            
        Returns:
            list: List of office information dictionaries
        """
        offices = []
        office_keys = [key for key in infobox.keys() if re.search("office", key)]
        logger.debug(f"Office keys: {office_keys}")
        
        for key in office_keys:
            # Determine office number
            try:
                num = re.findall(r"\d+", key)
                num = num[0] if num else ""
            except:
                num = ""  # this is the most current one
                
            # Parse dates
            term_start, term_end = self._parse_office_term_dates(infobox, key, num)
            
            # Add office to list
            try:
                office = {
                    "office": infobox[f"office{num}"],
                    "office_num": num,
                    "term_start": term_start,
                    "term_end": term_end
                }
                offices.append(office)
            except KeyError:
                continue
                
        return offices

    def get_current_office(self, offices, query_date):
        """
        Determine which offices were active at the query date.
        
        Args:
            offices: List of office dictionaries from parse_offices
            query_date: Date to check against
            
        Returns:
            tuple: (active_offices, detected_countries)
        """
        # Parse query date if it's a string
        if isinstance(query_date, str):
            query_date = dateparser.parse(query_date)
            
        active_offices = []
        detected_countries = []
        
        for office in offices:
            # Skip offices with no start date
            if not office['term_start']:
                continue
                
            # Extract country from office title
            country, _ = self.country_detector.search_nat(office['office'])
            if country:
                detected_countries.append(country)

            try:
                # Check if office was active at query date
                is_active = False
                if office['term_start'] < query_date:
                    if not office['term_end'] or office['term_end'] > query_date:
                        is_active = True
                        
                if is_active:
                    active_offices.append(office)
            except Exception:
                logger.info("Term start or end error in current office")
                
        return active_offices, detected_countries

    def _process_wiki_short_description(self, wiki, countries):
        """
        Process the short description from a Wikipedia article to extract actor code.
        
        Args:
            wiki: Wikipedia article
            countries: List to append detected countries to
            
        Returns:
            list: List containing SD code if found, otherwise empty list
        """
        if 'short_desc' not in wiki:
            return []
            
        # Extract country and text
        country, trimmed_text = self.country_detector.search_nat(wiki['short_desc'])
        if country:
            countries.append(country)
            
        # Match text to agent pattern
        sd_code = self.agent_matcher.trf_agent_match(trimmed_text, country=country)
        if sd_code:
            sd_code['source'] = "Wiki short description"
            sd_code['actor_wiki_job'] = wiki['short_desc']
            sd_code['wiki'] = wiki['title']
            sd_code['country'] = country
            return [sd_code]
            
        return []

    def _process_wiki_infobox(self, wiki, query_date, countries, office_countries):
        """
        Process the infobox from a Wikipedia article to extract actor code.
        
        Args:
            wiki: Wikipedia article
            query_date: Date to use for determining current offices
            countries: List to append detected countries to
            office_countries: List to append countries detected from offices to
            
        Returns:
            tuple: (box_codes, box_type_code, type_code)
        """
        if 'infobox' not in wiki:
            return [], None, None
            
        infobox = wiki['infobox']
        box_codes = []
        box_type_code = None
        type_code = None
        
        # Extract country from infobox
        if 'country' in infobox:
            country = self.country_detector.search_nat(infobox['country'])[0]
            if country:
                countries.append(country)
                
        # Get box type code
        if 'box_type' in wiki:
            box_type_code = self.agent_matcher.trf_agent_match(wiki['box_type'])
            if box_type_code:
                box_type_code['country'] = countries[0] if countries else None
                box_type_code['wiki'] = wiki['title']
                box_type_code['source'] = "Infobox Title"
                box_type_code['actor_wiki_job'] = wiki['box_type']
                
        # Get type code from infobox
        if 'type' in infobox:
            type_code = self.agent_matcher.trf_agent_match(infobox['type'])
            if type_code:
                type_code['country'] = countries[0] if countries else None
                type_code['wiki'] = wiki['title']
                type_code['source'] = "Infobox Type"
                type_code['actor_wiki_job'] = infobox['type']
                
        # Parse offices and get current offices
        offices = self.parse_offices(infobox)
        logger.debug(f"All offices: {offices}")
        current_offices, detected_countries = self.get_current_office(offices, query_date)
        logger.debug(f"Current offices: {current_offices}")
        office_countries.extend(detected_countries)
        
        # Handle ELI (former official) case
        if offices and not current_offices:
            eli_codes = self._handle_former_officials(offices, wiki, countries)
            if eli_codes:
                box_codes.extend(eli_codes)
        elif current_offices:
            # Handle current offices
            for office in current_offices:
                office_text = clean_query(office['office'])
                code = self.agent_matcher.short_text_to_agent(office_text, country_detector=self.country_detector)
                if code:
                    code['actor_wiki_job'] = office_text
                    code['source'] = "Infobox"
                    code['office_num'] = office['office_num'] or 0
                    code['wiki'] = wiki['title']
                    logger.debug(f"Office code: {code}")
                    box_codes.append(code)
                    break  # Only get the first one
                    
        return box_codes, box_type_code, type_code

    def _handle_former_officials(self, offices, wiki, countries):
        """
        Handle former officials (ELI code) from office history.
        
        Args:
            offices: List of office dictionaries
            wiki: Wikipedia article
            countries: List of detected countries
            
        Returns:
            list: List of ELI codes if applicable
        """
        # Get codes from past offices
        old_codes_raw = [self.agent_matcher.trf_agent_match(clean_query(o['office'])) for o in offices]
        old_codes = []
        for code in old_codes_raw:
            if not code or 'code_1' not in code:
                continue
            old_codes.append(code['code_1'])
            
        # Get countries from past offices
        old_countries = [self.country_detector.search_nat(o['office'])[0] for o in offices]
        box_country = list(set([c for c in old_countries if c]))
        
        # Check if person held government position
        if "GOV" in old_codes:
            code_1 = "ELI"
            
            # Handle different country detection scenarios
            if not box_country:
                # No country from offices
                country = countries[0] if countries else ""
                return [{'pattern': 'NA', 'code_1': code_1, 'code_2': '', 
                         'country': country, 
                         'description': "previously held a GOV role, so coded as ELI.", 
                         "source": "Infobox", "wiki": wiki['title']}]
            elif len(box_country) == 1:
                # Single country from offices
                primary_country = countries[0] if countries else ""
                if not primary_country or box_country[0] == primary_country:
                    return [{'pattern': 'NA', 'code_1': code_1, 'code_2': '', 
                             'country': box_country[0], 
                             'description': 'previously held a GOV role, so coded as ELI', 
                             "source": "Infobox", "wiki": wiki['title']}]
            
            # Multiple countries or mismatch
            primary_country = countries[0] if countries else ""
            return [{'pattern': 'NA', 'code_1': code_1, 'code_2': '', 
                     'country': primary_country, 
                     'description': f"previously held a GOV role, so coded as ELI. Countries: {box_country}", 
                     "source": "Infobox", "wiki": wiki['title']}]
                     
        return []

    def _process_wiki_categories(self, wiki, cat_countries):
        """
        Process Wikipedia categories to extract countries.
        
        Args:
            wiki: Wikipedia article
            cat_countries: List to append detected countries to
        """
        if 'categories' not in wiki:
            return
            
        for category in wiki['categories']:
            country, _ = self.country_detector.search_nat(category, categories=True)
            if country:
                cat_countries.append(country)

    def wiki_to_code(self, wiki, query_date="today", country=""):
        """
        Convert a Wikipedia article to a PLOVER actor code.
        
        Args:
            wiki: Wikipedia article
            query_date: Date to use for determining current offices
            country: Country code if already known
            
        Returns:
            list: List of actor codes derived from the article
        """
        # Handle missing wiki article
        if not wiki:
            return []
            
        # Initialize collections
        box_codes = []
        countries = []
        sd_code = []
        cat_countries = []
        office_countries = []
        
        # Extract country from first sentence
        intro_text = re.sub(r"\(.*?\)", "", wiki['intro_para']).strip()
        try:
            first_sent = intro_text.split("\n")[0]
            first_sent_country, _ = self.country_detector.search_nat(first_sent, method="first")
            if first_sent_country:
                countries.append(first_sent_country)
        except IndexError:
            first_sent_country = None
            
        # Process different parts of the wiki article
        sd_code = self._process_wiki_short_description(wiki, countries)
        box_codes_new, b_code, type_code = self._process_wiki_infobox(
            wiki, query_date, countries, office_countries
        )
        box_codes.extend(box_codes_new)
        self._process_wiki_categories(wiki, cat_countries)
        
        # Combine all codes
        all_codes = box_codes + sd_code + ([b_code] if b_code else []) + ([type_code] if type_code else [])
        all_codes = [c for c in all_codes if c]
        logger.debug(f"All codes: {all_codes}")
        
        # Collect all countries from different sources
        all_countries = [c['country'] for c in all_codes if c.get('country')]
        all_countries.extend(countries)
        all_countries.extend(office_countries)
        all_countries.extend(cat_countries)
        
        # Add first sentence country if no others found
        if not all_countries and first_sent_country:
            all_countries = [first_sent_country]
            
        # Find most common country
        country_counts = Counter([c for c in all_countries if c])
        top_country = country_counts.most_common(1)[0][0] if country_counts else None
        unique_countries = list(set([c for c in all_countries if c]))
        
        logger.debug(f"All countries: {unique_countries}")
        logger.debug(f"Top country: {top_country}")
        
        # Assign countries to codes
        if len(unique_countries) == 1 and unique_countries[0]:
            # Single country found
            for code in all_codes:
                code['country'] = unique_countries[0]
        elif top_country:
            # Multiple countries, use the most common
            for code in all_codes:
                if not code['country']:
                    code['country'] = top_country
                    
        # Handle case where no code was found but country was
        if not all_codes:
            if len(unique_countries) == 1 and unique_countries[0]:
                all_codes = [{
                    'pattern': '', 
                    'code_1': '', 
                    'code_2': '', 
                    'country': unique_countries[0], 
                    'description': "No code identified, but country found", 
                    "source": "Wiki", 
                    "wiki": wiki['title']
                }]
            elif top_country:
                all_codes = [{
                    'pattern': '', 
                    'code_1': '', 
                    'code_2': '', 
                    'country': top_country, 
                    'description': "No code identified, but country found", 
                    "source": "Wiki", 
                    "wiki": wiki['title']
                }]
                
        return all_codes


#######################################################
# Code Selection
#######################################################

class CodeSelector:
    """
    Select and clean best actor codes.
    
    This class provides methods for selecting the best actor code
    from a list of candidates and cleaning the result.
    
    Example:
        selector = CodeSelector()
        best_code = selector.pick_best_code(all_codes, country)
        cleaned = selector.clean_best(best_code)
    """
    
    # Fallback priorities, used only if the asset file cannot be read. The
    # authoritative copy is ngec/assets/PLOVER_priorities.csv; see
    # load_actor_priorities().
    ACTOR_TYPE_PRIORITIES = {
        "IGO": 200, "ISM": 195, "IMG": 192, "PRE": 190, "REB": 130,
        "SPY": 110, "JUD": 105, "OPP": 102, "GOV": 100, "LEG": 90,
        "MIL": 80, "COP": 75, "PRM": 72, "ELI": 70, "PTY": 65,
        "BUS": 60, "UAF": 50, "CRM": 48, "LAB": 47, "MED": 45,
        "NGO": 43, "SOC": 42, "EDU": 41, "JRN": 40, "ENV": 39,
        "HRI": 38, "UNK": 37, "REF": 35, "AGR": 30, "RAD": 20,
        "CVL": 10, "JEW": 5, "MUS": 5, "BUD": 5, "CHR": 5,
        "HIN": 5, "REL": 1, "": 0, "JNK": 51, "NON": 60
    }
    
    # Actor types that should be treated as countries themselves
    SPECIAL_ACTOR_TYPES = ["IGO", "MNC", "NGO", "ISM", "EUR", "UNO"]

    # Evidence sources that win outright in pick_best_code(), checked before
    # the priority tie-break. A source listed here short-circuits selection:
    # if it produced a candidate, that candidate is returned with no
    # comparison against the others. Kept as the default for backwards
    # compatibility -- see the `override_sources` argument below.
    DEFAULT_OVERRIDE_SOURCES = ("Wiki short description",)

    def __init__(self, priorities_file=None, override_sources=None):
        """
        Args:
            priorities_file: path to a priorities CSV in the format of
                ngec/assets/PLOVER_priorities.csv. None loads the PLOVER
                priorities. Supply this together with a custom `agents_file`
                when coding into a different ontology, so the tie-break in
                pick_best_code() knows how to rank the codes that file emits.
            override_sources: evidence sources that win outright over every
                other candidate, bypassing the priority tie-break. None keeps
                the historical default, `("Wiki short description",)`. Pass an
                empty sequence to disable the short-circuit entirely, so that
                conflicts between the span-text match and the Wikipedia short
                description are settled by `priorities_file` instead of by
                source. That is usually what a custom ontology wants: the
                short-circuit is only reached when the two sources disagree,
                and it resolves every such disagreement in Wikipedia's favour.
        """
        try:
            priorities, special = load_actor_priorities(priorities_file)
        except (OSError, ValueError) as e:
            if priorities_file is not None:
                raise
            logger.warning(f"Could not load priorities asset ({e}); "
                           f"falling back to the built-in table.")
            priorities, special = self.ACTOR_TYPE_PRIORITIES, self.SPECIAL_ACTOR_TYPES
        self.actor_type_priorities = priorities
        self.special_actor_types = special
        self.override_sources = tuple(
            self.DEFAULT_OVERRIDE_SOURCES if override_sources is None
            else override_sources
        )

    def _get_actor_priority(self, code_1):
        """
        Get priority value for an actor code.
        
        Args:
            code_1: Actor code
            
        Returns:
            int: Priority value
        """
        return self.actor_type_priorities.get(code_1, 0)

    def pick_best_code(self, all_codes, country):
        """
        Select the best actor code from a list of candidates.
        
        Args:
            all_codes: List of actor code dictionaries
            country: Country code if already known
            
        Returns:
            dict or None: Best actor code or None if no valid code
        """
        logger.debug(f"Running pick_best_code with input country {country}")
        
        # Handle empty code list
        if not all_codes:
            if country:
                return {
                    "country": country,
                    "code_1": "",
                    "code_2": "",
                    "source": "country only",
                    "wiki": "",
                    "query": ''
                }
            return None
            
        # Handle single code
        if len(all_codes) == 1:
            best = all_codes[0]
            best['best_reason'] = "only one code"
            if not best['country'] and country:
                best['country'] = country
            return best
            
        # Collect country information
        all_countries = [c['country'] for c in all_codes if c.get('country')]
        if country:
            all_countries.append(country)
        logger.debug(f"pick best code all_countries: {all_countries}")
        
        # Get unique codes
        unique_code_1s = list(set([c['code_1'] for c in all_codes if c.get('code_1')]))
        
        # Get wiki title if available
        wiki_titles = [c.get('wiki', '') for c in all_codes if 'wiki' in c]
        wiki_title = next((t for t in wiki_titles if t), "")
        
        # Determine best country
        if len(set(all_countries)) == 1 and all_countries:
            best_country = all_countries[0]
        elif not all_countries:
            best_country = ""
        elif country:
            best_country = country
        else:
            country_counts = Counter([c for c in all_countries if c])
            best_country = country_counts.most_common(1)[0][0] if country_counts else ""
            
        logger.debug(f"Identified best country: {best_country}")
        
        # Try different strategies to pick the best code
        
        # 1. Check for settlement (city) type
        code_sources = [c for c in all_codes if 'source' in c]
        box_type_codes = [c for c in code_sources if c['source'] == "Infobox Type"]
        if box_type_codes:
            if box_type_codes[0]['query'] in ['settlement']:
                # If it's a city, no actor/sector code applies
                best = box_type_codes[0]
                best['code_1'] = ""
                if not best['country']:
                    best['country'] = best_country
                if 'wiki' not in best:
                    best['wiki'] = wiki_title
                best['best_reason'] = "It's a city/settlement, so no code1 applies"
                return best
                
        # 2. Check for infobox entries
        info_box_codes = [c for c in code_sources if c['source'] == "Infobox"]
        if len(info_box_codes) == 1:
            # Single infobox entry
            best = info_box_codes[0]
            if best['code_1'] == "IGO":
                best['country'] = "IGO"
            else:
                best['country'] = best_country
            best['best_reason'] = "Only one entry in the info box"
            if 'wiki' not in best:
                best['wiki'] = wiki_title
            return best
        elif len(info_box_codes) > 1:
            # Multiple infobox entries, sort by office number
            info_box_codes.sort(key=lambda x: x.get('office_num', 0))
            best = info_box_codes[0]
            if best['code_1'] == "IGO":
                best['country'] = "IGO"
            else:
                best['country'] = best_country
            best['best_reason'] = "Picking highest priority Wiki info box title"
            if 'wiki' not in best:
                best['wiki'] = wiki_title
            return best
                
        # 3. Check for single unique code_1
        if len(unique_code_1s) == 1:
            logger.debug("Only one unique code_1, returning first one with that code")
            code1_entries = [c for c in all_codes if c.get('code_1')]
            wiki_codes = [c for c in code1_entries if c.get('wiki')]
            
            if wiki_codes:
                # Prefer entries with wiki info
                best = wiki_codes[0]
                best['best_reason'] = "only one unique code1, returning wiki code"
            else:
                # Otherwise sort by confidence if available
                try:
                    code1_entries.sort(key=lambda x: -x.get('conf', 0))
                    best = code1_entries[0]
                    best['best_reason'] = "only one unique code1: returning highest conf"
                except KeyError:
                    best = code1_entries[0]
                    best['best_reason'] = "only one unique code1: returning first entry"
                    
            best['country'] = best_country
            if 'wiki' not in best:
                best['wiki'] = wiki_title
            return best
                
        # 4. Check for an overriding evidence source (by default, the
        # Wikipedia short description). Steps 1-3 have already returned if the
        # candidates agree, so this branch is reached only when sources
        # conflict, and it resolves the conflict by source rather than by
        # code. Configure with CodeSelector(override_sources=...); an empty
        # sequence falls through to the priority tie-break in step 6.
        override_codes = [c for c in code_sources
                          if c['source'] in self.override_sources]
        if override_codes:
            best = override_codes[0]
            if best['code_1'] == "IGO":
                best['country'] = "IGO"
            else:
                best['country'] = best_country
            best['best_reason'] = f"Picking {best['source']}"
            return best
            
        # 5. Check for pre-wiki lookup codes.
        #
        # NOTE: this branch is currently dead. Nothing in the codebase assigns
        # the source "BERT matching on non-entity text" any more, so the list
        # is always empty and control falls through to the priority sort in
        # step 6. It matters because step 4 above is now configurable: with
        # override_sources=[], conflicts that used to be settled by step 4
        # reach step 6. If this source is ever reinstated, those conflicts
        # would silently start being settled here instead, under different
        # semantics -- decide deliberately at that point which of the two
        # should run first.
        pre_wiki_codes = [
            c for c in all_codes 
            if c.get('source') == "BERT matching on non-entity text" and c.get('country') and c.get('code_1')
        ]
        
        if len(pre_wiki_codes) == 1:
            best = pre_wiki_codes[0]
            best['best_reason'] = "Using pre-wiki lookup"
            best['country'] = best_country
            if 'wiki' not in best:
                best['wiki'] = wiki_title
            return best
            
        if pre_wiki_codes:
            # Check if all pre-wiki codes have same country and code_1
            unique_countries = set(c['country'] for c in pre_wiki_codes)
            unique_codes = set(c['code_1'] for c in pre_wiki_codes)
            
            if len(unique_countries) == 1 and len(unique_codes) == 1:
                best = pre_wiki_codes[0]
                best['country'] = best_country
                best['best_reason'] = "All pre-wiki lookups are the same"
                if 'wiki' not in best:
                    best['wiki'] = wiki_title
                return best
                
        # 6. Fall back to code priority
        logger.debug("Using code priority sorting")
        all_codes.sort(key=lambda x: -self._get_actor_priority(x.get('code_1', '')))
        
        if all_codes:
            # Get all codes with highest priority
            highest_priority = self._get_actor_priority(all_codes[0].get('code_1', ''))
            highest_priority_codes = [
                c for c in all_codes 
                if self._get_actor_priority(c.get('code_1', '')) == highest_priority
            ]
            
            # Prefer wiki codes
            wiki_codes = [c for c in highest_priority_codes if c.get('wiki')]
            
            if wiki_codes:
                best = wiki_codes[0]
                best['best_reason'] = "Ranked by code1 priority, returning wiki code"
            else:
                best = highest_priority_codes[0]
                best['best_reason'] = "Ranked by code1 priority, returning first"
                
            best['country'] = best_country
            if 'wiki' not in best:
                best['wiki'] = wiki_title
                
            return best
            
        return None

    def clean_best(self, best):
        """
        Clean and normalize the best actor code.
        
        Args:
            best: Actor code dictionary
            
        Returns:
            dict or None: Cleaned actor code or None if input was None
        """
        if not best:
            return None
            
        # Handle special codes
        if best.get('code_1') == 'JNK':
            best['country'] = ""
            
        if best.get('code_1') == 'NON':
            best['country'] = ""
            
        # Handle actor types that should be treated as countries
        if best.get('code_1') in self.special_actor_types:
            best['country'] = best['code_1']
            best['code_1'] = ""
            
        # Fix invalid country
        if best.get('country') is None or not isinstance(best.get('country'), str):
            best['country'] = ""
            
        # Handle composite country-code (e.g. "USAGOV")
        if best.get('country') and len(best['country']) == 6:
            best['code_1'] = best['country'][3:6]
            best['country'] = best['country'][0:3]
            
        return best



#######################################################
# Main Actor Resolver Class
#######################################################

class ActorResolver:
    """
    Main class for resolving actors to PLOVER codes.
    
    This class orchestrates the actor resolution process, integrating
    all the component classes.
    
    Example:
        resolver = ActorResolver()
        code = resolver.actor_to_code("German Chancellor")
        processed_events = resolver.process(events)
    """

    # TODO #26: allow overriding assets/models
    
    def __init__(self, 
                spacy_model=None,
                save_intermediate=False,
                wiki_sort_method="neural",
                gpu=False,
                es_client: None | Elasticsearch = None,
                agents_file: None | str = None,
                priorities_file: None | str = None,
                override_sources: None | list | tuple = None,
                device: None | str = None,
                ):
        """
        Initialize the ActorResolver with the necessary models and data.
        
        Args:
            spacy_model: Pre-loaded spaCy model to use
            save_intermediate: Whether to save intermediate results
            wiki_sort_method: Method to use for sorting Wikipedia results
            gpu: Whether to use GPU for model inference
            agents_file: Path to a custom PLOVER/CAMEO-format agents file. The
                default None uses the PLOVER agents file shipped in
                ngec.assets. A custom file lets a user code actors into their
                own ontology without changing any code; the same matcher is
                used both for the raw text and for Wikipedia short
                descriptions, so a custom file applies to both paths.
            priorities_file: Path to a custom actor-priority CSV, in the format
                of ngec.assets/PLOVER_priorities.csv. The default None uses the
                PLOVER priorities. Pass this whenever you pass a custom
                agents_file: codes absent from the priority table all tie at 0,
                so the tie-break in pick_best_code() silently degenerates to
                source order.
            override_sources: Evidence sources that win outright in
                pick_best_code(), bypassing the priority tie-break. The default
                None keeps the historical behaviour, in which a Wikipedia short
                description beats the span-text match whenever the two
                disagree. Pass `[]` to let priorities_file arbitrate instead.
            device: Explicit torch device ('cpu' or 'cuda') for the sentence
                encoders. Overrides `gpu`. The default None keeps the old
                behaviour ('cuda' when gpu=True, otherwise
                sentence-transformers' own choice, which is CUDA when a card is
                visible). Pass 'cpu' to hold the encoders on the CPU on a
                machine that has a GPU -- gpu=False alone does not do that.
        """
        # TODO: #26, make it possible to override models
        # This impacts all the other related classes here

        # Set device for model inference. `device` wins when given, so a
        # caller can pin the encoders to the CPU; `gpu` alone only ever asks
        # for CUDA and leaves the choice to sentence-transformers otherwise.
        self.device = device if device is not None else ('cuda' if gpu else None)
        
        # Initialize utility classes
        self.cache_manager = CacheManager()
        self.country_detector = CountryDetector()
        
        # Initialize model manager and load models
        self.model_manager = ModelManager(self.device)
        self.nlp = spacy_model if spacy_model else self.model_manager.load_spacy_lg()
        self.trf = self.model_manager.load_trf_model()
        
        # Initialize agent matcher
        self.agent_matcher = AgentMatcher(
            self.trf, 
            agents_file=agents_file,
            device=self.device, 
        )
        
        # Initialize Wikipedia components
        self.wiki_matcher = WikiMatcher(
            es_client = es_client,
            model_manager = self.model_manager, 
            wiki_sort_method=wiki_sort_method,
            device=self.device, 
        )
        self.wiki_parser = WikiParser(
            self.country_detector,
            self.agent_matcher,
            self.device
        )
        
        # Initialize code selector
        self.code_selector = CodeSelector(priorities_file=priorities_file,
                                          override_sources=override_sources)
        
        # Store configuration
        self.save_intermediate = save_intermediate
        self.wiki_sort_method = wiki_sort_method

    def _country_from_context(self, text, context, window=200):
        """
        Return the country *name* mentioned nearest to `text` in `context`.

        Looks at `window` characters on either side of the first occurrence
        of the mention (or the first 2*window characters if the mention is not
        found verbatim), mirroring how the wiki ranker's training data was
        built. Returns "" if no country or nationality is mentioned.
        """
        mention = re.sub(r"\s+", " ", text).strip()
        pos = context.find(mention)
        if pos >= 0:
            passage = context[max(0, pos - window): pos + len(mention) + window]
        else:
            passage = context[: 2 * window]
        country, _ = self.country_detector.search_nat(passage, use_name=True)
        return country or ""

    def actor_to_code(self, text, doc=None, context="", query_date="today", known_country="", search_limit_term="") -> dict | None:
        """
        Resolve an actor mention to a code representing their role.
        
        Args:
            text: Text mention of the actor to resolve
            doc: Unused, kept for signature compatibility. The span is parsed
                below, after nationality stripping, which is the only parse
                this function needs.
            context: Additional context to help with disambiguation
            query_date: Date to use when determining current offices
            known_country: Country code if already known
            search_limit_term: Term to limit Wikipedia search results
            
        Returns:
            dict or None: Actor code information or None if resolution fails
        """
        # Check cache first. The key includes the context and country because
        # the same mention ("the Liberal Party") resolves differently in
        # different documents; keying on the mention alone would return the
        # first document's answer for every later one.
        cache_key = "_".join([text, str(query_date), known_country,
                              str(hash(context)) if context else ""])
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug("Returning from cache")
            return cached_result
        
        # NB: `doc` is not used. The span has to be re-parsed below anyway,
        # after nationality stripping changes it, so parsing `text` here was
        # a spaCy call per mention whose result was thrown away. The argument
        # is kept because it is part of the public signature.

        # TODO: replace this with the new entity splitter

        country, trimmed_text = self.country_detector.search_nat(text)
        logger.debug(f"Identified country from text: {country}")
        
        # Handle country-only case
        if country and not trimmed_text:
            logger.debug("Country only, returning as-is")
            code_full_text = {
                "country": country,
                "code_1": "",
                "code_2": "",
                "source": "country only",
                "wiki": "",
                'actor_wiki_job': "",
                "query": text
            }
            # A span that search_nat swallows whole is usually a nationality
            # ("Israeli"), but it can also be an organisation that happens to
            # be one of the country patterns: "U.N.", "DPRK", "UWSA",
            # "European Union". Those have Wikipedia articles, and returning
            # here used to throw them away before retrieval even started, so
            # look the organisation-like ones up and attach the page. The
            # code itself stays country-only.
            if span_looks_like_organisation(text):
                logger.debug(f"Country-only span '{text}' looks like an organisation. Trying Wikipedia.")
                if not known_country and context:
                    known_country = self._country_from_context(text, context)
                wiki = self.wiki_matcher.query_wiki(
                    query_term=text,
                    country=known_country,
                    context=context,
                    limit_term=search_limit_term
                )
                if wiki:
                    logger.debug(f"Wikipedia page found for country-only span: {wiki['title']}")
                    code_full_text['wiki'] = wiki['title']
            self.cache_manager.set(cache_key, code_full_text)
            return self.code_selector.clean_best(code_full_text)

        # Parse entities in text
        # TODO: all of this probably goes away with the new entity splitter
        try:
            doc = self.nlp(trimmed_text)
            non_ent_text = strip_ents(doc)
            ents = [i for i in doc.ents if i.label_ in ['EVENT', 'FAC', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON']]
            token_level_ents = [i.ent_type_ for i in doc]
            ent_text = ''.join([i.text_with_ws for i in doc if i.ent_type_ != ""])
            logger.debug(f"Found named entities: {ents}")
        except IndexError:
            # Usually caused by a mismatch between token and embedding
            logger.info(f"Token alignment error on {trimmed_text}")
            doc = None
            non_ent_text = trimmed_text
            token_level_ents = ['']
            ent_text = ""
            ents = []
            
        # Try direct matching first
        code_full_text = None
        if trimmed_text:
            logger.debug(f"Trying direct matching on: {trimmed_text}")
            code_full_text = self.agent_matcher.trf_agent_match(trimmed_text, country=country)

            if code_full_text:
                logger.debug(f"Direct match found: {code_full_text}")
                code_full_text['source'] = "BERT matching full text"
                code_full_text['wiki'] = ""
                code_full_text['actor_wiki_job'] = ""
            else:
                logger.debug(f"No direct match found for {trimmed_text}")

        # The policy: a mention gets a role code and no Wikipedia page only if
        # it names a category of people rather than an actor ("police",
        # "protesters"), or if the agent matcher is nearly certain about a span
        # that names nothing in particular. A generically-worded institution --
        # "the interior ministry", "the supreme court", "the central bank" --
        # names one body in one country and goes to the linker.
        #
        # This used to be three confidence clauses, and the two loosest ones
        # swallowed almost every institution mention in the corpus: 164 of 173
        # institution probes never reached the linker, because "the defence
        # ministry" is lowercase, multi-word, and a very good agent match.
        # Detect the country from the passage around the mention when the
        # caller did not supply one, falling back to a country named in the
        # span itself ("Norway's central bank"). The wiki ranker's
        # country_match feature and the "Title (Country)" retrieval variants
        # need it, and the gate below only sends a generically named
        # institution to the linker when there is a country to disambiguate
        # with: "the defence ministry" with no context has no right answer.
        if not known_country and context:
            known_country = self._country_from_context(text, context)
            logger.debug(f"Country detected from context: {known_country!r}")
        if not known_country and country:
            known_country = self.country_detector.search_nat(text, use_name=True)[0] or ""

        skip_wiki_reason = ""
        if trimmed_text and doc is not None and span_is_generic_collective(doc):
            skip_wiki_reason = "generic collective"
        elif trimmed_text and doc is not None and span_is_unlinkable_reference(doc):
            skip_wiki_reason = "unlinkable reference"
        elif (code_full_text and doc is not None
                and not ents
                and not (span_names_an_institution(doc) and known_country)
                and (code_full_text['conf'] > THRESHOLD_VERY_HIGH_CONFIDENCE
                     # With no context and no country there is nothing to
                     # disambiguate a title like "Defense Minister" with, so
                     # a merely confident agent match settles it, as before.
                     or (not context and not known_country
                         and code_full_text['conf'] > THRESHOLD_CONFIDENT_NO_CONTEXT))):
            skip_wiki_reason = "high-confidence agent match"

        if skip_wiki_reason:
            logger.debug(f"Skipping Wikipedia lookup for '{trimmed_text}': {skip_wiki_reason}")
            if not code_full_text and not country:
                # Nothing to return: no role code, no country, and no page.
                self.cache_manager.set(cache_key, None)
                return None
            skipped_code = code_full_text if code_full_text else {
                "country": country,
                "code_1": "",
                "code_2": "",
                "query": trimmed_text,
            }
            if not code_full_text:
                skipped_code['source'] = skip_wiki_reason
            skipped_code['wiki'] = ""
            skipped_code['actor_wiki_job'] = ""
            skipped_code = self.code_selector.clean_best(skipped_code)
            self.cache_manager.set(cache_key, skipped_code)
            return skipped_code

        # Extract core entity and role from the span using NER
        # This handles noisy spans like "Republican Senator Pat Roberts of Kansas"
        # --> core_query="Pat Roberts", actor_desc="Republican Senator"
        # Prefer PERSON over ORG (most queries are people)
        # Use the longest match (NOT the first match as before: this caused problems).
        core_query = trimmed_text
        actor_desc = ""
        if ents:
            persons = [e for e in ents if e.label_ == 'PERSON']
            orgs = [e for e in ents if e.label_ == 'ORG']
            best_ent = None
            for candidates in [persons, orgs]:
                if candidates:
                    best_ent = max(candidates, key=lambda e: len(e.text))
                    break
            if best_ent:
                core_query = best_ent.text
                before = trimmed_text[:best_ent.start_char].strip().strip(',').strip()
                after = trimmed_text[best_ent.end_char:].strip().strip(',').strip()
                desc_parts = [p for p in [before, after] if p]
                actor_desc = ' '.join(desc_parts)
        # Guard against NER handing back a core entity that is a worse query
        # than the span it came from ("House Judiciary" -> "House").
        if core_query != trimmed_text and core_query_is_degenerate(core_query, trimmed_text):
            logger.debug(f"Core entity '{core_query}' is degenerate; searching on '{trimmed_text}' instead")
            core_query = trimmed_text
            actor_desc = ""
        if core_query != trimmed_text:
            logger.debug(f"Extracted core entity: '{core_query}' (desc: '{actor_desc}') from '{trimmed_text}'")

        # Try Wikipedia lookup for better resolution
        logger.debug(f"Trying Wikipedia lookup with: {core_query}")
        wiki_codes = []
        # Skip _expand_query when NER already extracted a specific multi-word entity.
        # Sometimes they also should be expanded, but can also replace the correct name with something worse.
        # Just do allow expansion for single-word entities (e.g. okay to expand "Robertson", but will also expand "Hamas").
        ner_extracted_specific = (core_query != trimmed_text and len(core_query.split()) >= 2)
        # The other surface forms of this same mention: the raw span, and the
        # span after nationality stripping but before NER cut it down. The
        # matcher searches the first one that differs from the main term and
        # merges the two candidate lists, because the form the article is
        # titled under is often not the one the pipeline settled on.
        raw_span = re.sub(r"['’]s\s*$", "", text).strip()
        alt_query_terms = []
        for term in [raw_span, trimmed_text, core_query]:
            if term and term != core_query and term not in alt_query_terms:
                alt_query_terms.append(term)
        wiki = self.wiki_matcher.query_wiki(
            query_term=core_query,
            country=known_country,
            context=context,
            actor_desc=actor_desc,
            limit_term=search_limit_term,
            skip_expansion=ner_extracted_specific,
            alt_query_terms=alt_query_terms,
        )

        if wiki:
            logger.debug(f"Wikipedia page found: {wiki['title']}")
            wiki_codes = self.wiki_parser.wiki_to_code(wiki, query_date)
        elif core_query != trimmed_text:
            # Try with full text if core entity search failed
            logger.debug(f"Core entity search failed. Trying full text: {trimmed_text}")
            wiki = self.wiki_matcher.query_wiki(
                query_term=trimmed_text,
                country=known_country,
                context=context,
                limit_term=search_limit_term
            )
            if wiki:
                wiki_codes = self.wiki_parser.wiki_to_code(wiki, query_date)
        elif ent_text:
            # Try again with just entity text if original lookup failed
            logger.debug(f"No wiki results. Trying with entity text: {ent_text}")
            wiki = self.wiki_matcher.query_wiki(
                query_term=ent_text,
                country=known_country,
                context=context,
                limit_term=search_limit_term
            )
            if wiki:
                wiki_codes = self.wiki_parser.wiki_to_code(wiki, query_date)
                
        # Combine all possible codes
        code_full_text_list = [code_full_text] if code_full_text else []
        all_codes = wiki_codes + code_full_text_list
        all_codes = [c for c in all_codes if c]
        
        logger.debug("--- ALL CODES ----")
        logger.debug(all_codes)
        
        # Extract unique codes for reference
        unique_code1s = list(set([c['code_1'] for c in all_codes if c.get('code_1')]))
        unique_code1s = [c for c in unique_code1s if c not in ["IGO"]]
        unique_code2s = list(set([c['code_2'] for c in all_codes if c.get('code_2')]))
        
        # Pick the best code
        best = self.code_selector.pick_best_code(all_codes, country)
        best = self.code_selector.clean_best(best)
        
        # Add unique codes lists for reference
        if best:
            best['all_code1s'] = unique_code1s
            best['all_code2s'] = unique_code2s
            
        # Cache and return result
        self.cache_manager.set(cache_key, best)
        return best

    def process(self, event_list, save_intermediate=False):
        """
        Process a list of events to resolve actor attributes.
        
        For each event, adds actor resolution information to the ACTOR and RECIP attributes.
        
        Args:
            event_list: List of event dictionaries
            save_intermediate: Whether to save intermediate results
            
        Returns:
            list: The same event list with actor resolution information added

        Examples:
            >>> input = [
                {"pub_date": "2007-07-01",
                 # attribute model output: one event per record, 'attributes' is a dict
                 "attributes": {
                    'actor': ['President Macron', 'Chancellor Angel Merkel'],
                    'recipient': ['N/A']
                 }},
            ]
            >>> ar.process(input)
            # each event gains top-level 'actor' and 'recipient' (coded)


        """
        # Don't modify the input list
        event_list = deepcopy(event_list)

        for event in track(event_list, description="Resolving actors..."):

            # Get the date from the event
            query_date = event.get('pub_date', "today")

            # 'attributes' is a single dict (one event per record). .get() guards
            # against a record with no attributes.
            attributes = event.get("attributes", {})

            # We need to go through both 'actor' and 'recipient'
            for attribute_key in ["actor", "recipient"]:
                # .get(): the model may omit a key entirely
                actor_list = attributes.get(attribute_key, [])

                # We put the resolved actor info at the top level of the event,
                # under the relevant key ('actor' or 'recipient'); so basically
                # moving one up from event["attributes"]
                event[attribute_key] = []

                # Resolve each actor
                for actor in actor_list:
                    # Malformed LLM output
                    exclude = ["N/A"]
                    if actor in exclude:
                        continue

                    # Pass the story text as context: it drives NER-based query
                    # expansion and the context-similarity features the wiki
                    # ranker was trained with. The demo and the ECAV evaluation
                    # already pass it; the pipeline path was silently omitting it.
                    #
                    # TODO: this re-parses the whole story with spaCy once per
                    # mention. The parse happens in WikiMatcher._expand_query,
                    # which is reached through query_wiki and has no way to
                    # accept a pre-parsed document, so fixing it means adding a
                    # doc argument there. A story with eight actor mentions is
                    # parsed eight times.
                    res = self.actor_to_code(actor,
                                             context=event.get("event_text", ""),
                                             query_date=query_date)
                    # actor_to_code can return None, this will break the code below
                    if res is None:
                        res = {}

                    # Normalize results
                    # TODO: not sure what the agent matcher outputs but need
                    # to sync it up with wiki matcher output
                    this_actor = {}

                    this_actor['wiki'] = res.get('wiki', "")
                    this_actor['actor_wiki_job'] = res.get('actor_wiki_job', "")

                    # Add code lists
                    this_actor['all_code1s'] = res.get('all_code1s', [])
                    this_actor['all_code2s'] = res.get('all_code2s', [])

                    # Add country and codes
                    this_actor['country'] = res.get('country', "")
                    this_actor['code_1'] = res.get('code_1', "")
                    this_actor['code_2'] = res.get('code_2', "")

                    # Add query and pattern information
                    this_actor['actor_role_query'] = res.get('query', "")
                    this_actor['actor_resolved_pattern'] = res.get('description', "")

                    # Add confidence and reason
                    this_actor['actor_pattern_conf'] = float(res.get('conf', 0))
                    this_actor['actor_resolution_reason'] = res.get('best_reason', "")

                    # Other stuff
                    this_actor['description'] = res.get('description', "")
                    this_actor['source'] = res.get('source', "")
                    this_actor['best_reason'] = res.get('best_reason', "")

                    # END normalize, now add it to the event actor/recipient list
                    event[attribute_key].append(this_actor)


        # Save intermediate results if requested
        if save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_actor_resolution_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)
                
        return event_list


#######################################################
# Main Entry Point
#######################################################

def main():
    """
    Main entry point for actor resolution.
    
    This function demonstrates how to use the ActorResolver.
    """
    import argparse

    from es_client import setup_es_client
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Resolve actors in events to PLOVER codes")
    parser.add_argument("input_file", help="Input JSONL file of events")
    parser.add_argument("output_file", help="Output JSONL file with actor resolution")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    args = parser.parse_args()
    
    # Load events from input file
    with jsonlines.open(args.input_file, "r") as f:
        events = list(f.iter())
    
    es_client = setup_es_client()

    # Create actor resolver
    resolver = ActorResolver(
        save_intermediate=args.save_intermediate,
        gpu=args.gpu,
        es_client=es_client,
    )
    
    # Process events
    processed_events = resolver.process(events)
    
    # Save results to output file
    with jsonlines.open(args.output_file, "w") as f:
        f.write_all(processed_events)
    
    print(f"Processed {len(processed_events)} events. Results saved to {args.output_file}")


if __name__ == "__main__":

    import logging

    from rich.logging import RichHandler

    # Use a real logger for debugging
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[RichHandler()]
    )
    
    main()

    exit(0)  # Exit cleanly after running main

    ## Testing junk to remove later

    from actor_resolution import (
        ActorResolver,
        AgentMatcher,
        CountryDetector,
        ModelManager,
        WikiMatcher,
        WikiParser,
    )
    from es_client import setup_es_client

    es_client = setup_es_client()

    event = {'event_text': 'Turkish forces and Turkish-backed militias battled with YPG militants in Syria.', 'id': '789_0', '_doc_position': 2, 'event_type': 'ASSAULT', 'event_mode': '', 'attributes': {'event_type': 'ASSAULT', 'anchor_quote': 'Turkish forces and Turkish-backed militias battled with YPG militants in Syria.', 'actor': ['Turkish forces', 'Turkish-backed militias'], 'recipient': ['YPG militants'], 'date': ['N/A'], 'location': ['Syria']}}
    agent_matcher = AgentMatcher()
    actor_match = agent_matcher.trf_agent_match("Chancellor", country="DEU")
    #{'pattern': 'chancellor', 'code_1': 'GOV', 'code_2': '', 'country': 'DEU', 'description': 'chancellor', 'query': 'Chancellor', 'conf': np.float64(0.9557092082997871)}
    
    resolver = ActorResolver(es_client=es_client)
    resolver.actor_to_code("German Chancellor")
    resolver.actor_to_code("Angela Merkel")
    resolver.actor_to_code("Angela Merkel",
                           query_date="2015-01-01")
    
    # Now demonstrate the wiki lookup functionality
    wiki = resolver.wiki_matcher.query_wiki("Merkel", context = "Angela Merkel is the Chancellor of Germany.")
    
    res = resolver.wiki_matcher.query_wiki("Obama", method="rules")
    resolver.wiki_matcher.query_wiki("Obama",
                                     context = "Michelle Obama is the former First Lady of the United States.",
                                     method = "rules")
    
    resolver.wiki_matcher.query_wiki("Obama",
                                     context = "Obama is the former First Lady of the United States.",
                                     method = "rules")
    
    resolver.wiki_matcher.query_wiki("Obama",
                                     context = "Obama is the former First Lady of the United States.",
                                     method = "neural")
