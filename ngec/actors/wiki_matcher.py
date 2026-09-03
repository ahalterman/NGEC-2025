"""
Functionality for matching a query term to a Wikipedia article. Relies on a
Elasticsearch index of Wikipedia data.
"""

import gzip
from importlib import resources
import json
import logging
import math
from pathlib import Path
import re
from typing import Literal

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import numpy as np
import pandas as pd
import pylcs
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from spacy.language import Language
from spacy.tokens import Doc
from textacy.preprocessing.remove import accents as remove_accents
import torch
from xgboost import XGBClassifier

from .common import CountryDetector, ModelManager, clean_query

# Constants
THRESHOLD_NEURAL_TITLE_MATCH = 0.9
THRESHOLD_COMBINED_SCORE = 9 

# Words of two or more letters, lowercased. Shared by the IDF table builder
# (setup/wiki/build_idf_table.py) and the runtime, so the two cannot drift.
TFIDF_TOKEN_PATTERN = re.compile(r"[a-z]{2,}")

# A capitalised word. Apostrophes and hyphens are kept ("Mugabe's", "Abu-Bakr").
CAPITALIZED_WORD = re.compile(r"[A-Z][A-Za-z'\u2019\-]{2,}")

# Punctuation that ends a sentence (or a bullet). A capitalised word right
# after one of these is capitalised by grammar, not because it is a name.
SENTENCE_BREAK = ".!?:;\u2022"

logger = logging.getLogger(__name__)

# Loaded once per process, on first use: see load_idf_table.
_idf_table = None
_country_detector = None


def load_idf_table(path: str | Path | None = None) -> dict:
    """
    Inverse document frequencies for Wikipedia intro paragraphs.

    A word like "footballer" says a lot about which article a mention belongs
    to; a word like "years" says almost nothing. The IDF table records how rare
    each word is across a 200,000-article sample of the index, so the ranker can
    weight the overlap between a news story and a candidate article accordingly.
    The table ships with the package (`ngec/assets/wiki_idf.json.gz`) and is
    built by `setup/wiki/build_idf_table.py`.

    The table is cached in a module-level variable, so the file is read at most
    once per process.
    """
    global _idf_table
    if _idf_table is None:
        if path is None:
            path = Path(str(resources.files("ngec"))) / "assets" / "wiki_idf.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            table = json.load(f)
        _idf_table = table["idf"]
        logger.debug(f"Loaded {len(_idf_table)} IDF weights from {path}")
    return _idf_table


def get_country_detector() -> CountryDetector:
    """A shared CountryDetector, built on first use (it compiles ~1,000 regexes)."""
    global _country_detector
    if _country_detector is None:
        _country_detector = CountryDetector()
    return _country_detector


def tfidf_vector(text: str, idf: dict) -> dict:
    """
    The L2-normalized TF-IDF vector of `text`, as a {word: weight} dict.

    Sublinear term frequency (1 + log(count)), so a word repeated ten times in a
    long news story does not swamp everything else. Words missing from the IDF
    table (stop words, typos, rare names) are dropped.
    """
    counts = {}
    for token in TFIDF_TOKEN_PATTERN.findall(text.lower()):
        if token in idf:
            counts[token] = counts.get(token, 0) + 1
    weights = {word: (1 + math.log(count)) * idf[word] for word, count in counts.items()}
    length = math.sqrt(sum(w * w for w in weights.values()))
    if length == 0:
        return {}
    return {word: weight / length for word, weight in weights.items()}


def tfidf_cosine(first: dict, second: dict) -> float:
    """Cosine similarity of two vectors from `tfidf_vector` (already normalized)."""
    if len(second) < len(first):
        first, second = second, first
    return float(sum(weight * second.get(word, 0.0) for word, weight in first.items()))


def capitalized_words(text: str) -> set:
    """
    The distinct capitalised words in `text`, skipping the first word of each
    sentence.

    Shared capitalised words are a cheap stand-in for "these two texts are about
    the same people and places": a news story about Ghana's election and the
    Wikipedia article on Ghana's electoral commission will share "Accra",
    "Mahama", "Ghana". Sentence-initial words are dropped because their capital
    letter carries no such information.
    """
    if not text:
        return set()
    words = set()
    for match in CAPITALIZED_WORD.finditer(text):
        before = text[max(0, match.start() - 3):match.start()].strip()
        if not before or before[-1] in SENTENCE_BREAK:
            continue
        words.add(match.group(0))
    return words


def country_phrase_variants(query_term: str, country: str) -> list[str]:
    """
    Title forms a country's institution is likely to be listed under.

    News stories name institutions the way a reader would ("the defense
    ministry"); Wikipedia names them the way a catalogue would ("Ministry of
    Defence (Ghana)"). This builds the country-qualified strings to search for.
    Returns an empty list if either argument is empty.

    With query_term="electoral commission" and country="Ghana":

    | rewrite                | result                              |
    |------------------------|-------------------------------------|
    | parenthetical          | electoral commission (Ghana)        |
    | "of" form              | electoral commission of Ghana       |
    | country first          | Ghana electoral commission          |

    Two head-noun rewrites are added for institutions, because the country
    forms above can never reach an inverted title on their own:

    | rewrite                | example                                       |
    |------------------------|-----------------------------------------------|
    | "X ministry"           | defense ministry -> Ministry of defense,       |
    |   / "X department"     |   Ministry of defense (Ghana),                 |
    |                        |   Ministry of defense of Ghana                 |
    | "central bank"         | central bank -> Bank of Ghana,                 |
    |                        |   Central Bank of Ghana                        |

    Finally, anything containing "defense"/"defence" is emitted in both
    spellings, since Wikipedia uses whichever the country itself uses.

    Capitalization does not matter: these strings are used in `match_phrase`
    queries against analyzed (lowercased) fields.
    """
    if not query_term or not country:
        return []

    variants = [f"{query_term} ({country})",
                f"{query_term} of {country}",
                f"{country} {query_term}"]

    # "defense ministry" -> "Ministry of defense" (+ its country forms)
    head_noun = re.match(r"(.+?)\s+(ministry|department)\s*$", query_term, flags=re.IGNORECASE)
    if head_noun:
        modifier, head = head_noun.group(1), head_noun.group(2).capitalize()
        inverted = f"{head} of {modifier}"
        variants += [inverted, f"{inverted} ({country})", f"{inverted} of {country}"]

    # "the central bank" is almost always titled after the country itself
    if re.search(r"\bcentral bank\b", query_term, flags=re.IGNORECASE):
        variants += [f"Bank of {country}", f"Central Bank of {country}"]

    # British vs. American spelling of "defence"
    for variant in list(variants):
        if re.search(r"defen[sc]e", variant, flags=re.IGNORECASE):
            variants.append(re.sub(r"defense", "defence", variant, flags=re.IGNORECASE))
            variants.append(re.sub(r"defence", "defense", variant, flags=re.IGNORECASE))

    # De-duplicate (case-insensitively) while keeping the order above
    seen = set()
    deduped = []
    for variant in variants:
        if variant.lower() not in seen:
            seen.add(variant.lower())
            deduped.append(variant)
    return deduped


def merge_ranked_results(primary: list[dict], alternate: list[dict], max_results: int) -> list[dict]:
    """
    Merge two ranked lists of Wikipedia articles by interleaving them.

    Takes the first article of each list, then the second of each, and so on,
    skipping any title already taken. Interleaving (rather than appending)
    keeps the merged list roughly sorted by "how highly did *some* query rank
    this", which is what the downstream trim and ranker expect.

    Args:
        primary: results for the main query term, in rank order
        alternate: results for the alternative query term, in rank order
        max_results: cap on the length of the merged list

    Returns:
        list: the merged, de-duplicated, capped list of articles
    """
    merged = []
    seen_titles = set()
    for rank in range(max(len(primary), len(alternate))):
        for results in (primary, alternate):
            if rank < len(results):
                article = results[rank]
                if article['title'] not in seen_titles:
                    seen_titles.add(article['title'])
                    merged.append(article)
    return merged[:max_results]



def title_concept_features(title: str, intro_length: int, doc_country: str,
                           detector: CountryDetector) -> tuple[int, int]:
    """
    Two flags for telling a *concept* page from an *institution* page.

    When a story mentions "the interior ministry", Wikipedia offers both the
    generic article `Interior ministry` (about the idea of interior ministries)
    and the specific one, `Ministry of the Interior (Ghana)`. The generic page is
    almost never the right answer, but it looks like a good match on every
    string feature, so the ranker needs a way to see the difference.

    Returns:
        tuple: (title_is_generic_concept, title_has_other_country)

        `title_is_generic_concept` is 1 when the title reads like a common noun:
        no disambiguating parenthetical, nothing title-cased after the first
        word, no country or demonym, and a short article behind it.

        `title_has_other_country` is 1 when the title names a country and it is
        not the country the story is about -- "Ministry of the Interior (Kenya)"
        in a story about Ghana.
    """
    has_parenthetical = "(" in title
    after_first_word = title.split(" ", 1)[1] if " " in title else ""
    nothing_capitalized_after_first = after_first_word == after_first_word.lower()

    country_in_title = detector.any_country_name.search(title)
    has_country_word = bool(country_in_title) or bool(detector.any_demonym.search(title))

    is_generic = int(not has_parenthetical
                     and nothing_capitalized_after_first
                     and not has_country_word
                     and intro_length < 900)
    has_other_country = int(bool(country_in_title) and bool(doc_country)
                            and doc_country not in title)
    return is_generic, has_other_country


#######################################################
# Wikipedia Client
#######################################################

class WikiClient:
    """
    Elasticsearch interface for Wikipedia data.
    
    This class provides methods for connecting to Elasticsearch
    and searching Wikipedia articles.
    
    Example:
        client = WikiClient()
        search = client.setup_es()
        client.check_wiki(search)
    """
    
    def __init__(self,
                 es_client: Elasticsearch,
                 ):
        """Initialize the Wikipedia client.

        es_config is a dict with es_host, es_port, es_user, es_password
        """
        try:
            es_client.ping()
            self.conn = Search(using=es_client, index="wiki")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Elasticsearch: {e}")
    
        self.check_wiki(self.conn)

    def check_wiki(self, conn):
        """
        Verify that the Wikipedia index is using the correct format.
        
        Args:
            conn: Elasticsearch search object
            
        Raises:
            ValueError: If Wikipedia index is outdated
        """
        query = {
            "multi_match": {
                "query": "Massachusetts",
                "fields": ['title^2', 'alternative_names'],
                "type": "phrase"
            }
        }
        
        try:
            res = conn.query(query)[0:1].execute()
            top = res['hits']['hits'][0].to_dict()['_source']
            if 'redirects' not in top.keys():
                raise ValueError("You seem to be using an outdated Wikipedia index that doesn't have a 'redirects' field. Please talk to Andy.")
        except Exception as e:
            raise ValueError(f"Error checking Wikipedia index: {e}")

    def run_wiki_search(self, query_term, limit_term="", max_results=200,
                        country="",
                        use_importance=False,
                        title_exact_boost=250,
                        title_and_boost=120,
                        title_fuzzy_boost=50,
                        redirects_exact_boost=100,
                        redirects_fuzzy_boost=50,
                        alternative_names_boost=100,
                        alternative_names_fuzzy_boost=20,
                        short_desc_boost=10,
                        intro_para_boost=5,
                        country_title_boost=200,
                        country_redirects_boost=150,
                        ):
        """
        Search the Wikipedia index for `query_term`.

        Args:
            query_term: the (already cleaned) string to search for
            limit_term: optional second term the article must also match
            max_results: how many articles to return
            country: the country *name* (e.g. "Ghana") the mention belongs to,
                if known. Used to add country-qualified title clauses; see
                `country_phrase_variants`. Passing "" is the old behavior.
            use_importance: also boost articles by redirect count and by having
                an infobox/short description
        """

        # Base matching clauses.
        #
        # The first four clauses are the "the query looks just like the title"
        # clauses. They used to be `term` queries, which do not work here:
        # `title`, `redirects` and `alternative_names` are analyzed `text`
        # fields with no `.keyword` sub-field, so a `term` query only matches
        # when the whole query is a single already-lowercase token. Measured on
        # the 1,966-row wiki gold set, they fired on 0.4% of queries, i.e. the
        # three highest boosts in the search were dead. `match_phrase` (all the
        # query's tokens, in order) and `match ... operator: "and"` (all the
        # tokens, any order) are the working equivalents.
        base_should_clauses = [
            {"match_phrase": {"title": {"query": query_term, "boost": title_exact_boost}}},
            {"match": {"title": {"query": query_term, "operator": "and", "boost": title_and_boost}}},
            {"match_phrase": {"redirects": {"query": query_term, "boost": redirects_exact_boost}}},
            {"match_phrase": {"alternative_names": {"query": query_term, "boost": alternative_names_boost}}},
            # Looser bag-of-words matches
            {"match": {"title": {"query": query_term, "boost": title_fuzzy_boost}}},
            # Folded (ASCII-normalized + stemmed) title match
            {"match": {"title.folded": {"query": query_term, "boost": title_fuzzy_boost}}},
            {"match": {"redirects": {"query": query_term, "boost": redirects_fuzzy_boost}}},
            # Folded redirects match (handles diacritics + plurals)
            {"match": {"redirects.folded": {"query": query_term, "boost": redirects_fuzzy_boost}}},
            {"match": {"alternative_names": {"query": query_term, "boost": alternative_names_fuzzy_boost}}},
            {"match": {"alternative_names.folded": {"query": query_term, "boost": alternative_names_fuzzy_boost}}},
            {"match": {"intro_para": {"query": query_term, "boost": intro_para_boost}}},
            {"match": {"short_desc": {"query": query_term, "boost": short_desc_boost}}}
        ]

        # Country-qualified title clauses. News text says "the electoral
        # commission"; Wikipedia says "Electoral Commission of Kenya". These
        # clauses look for the country-qualified titles the mention implies.
        # (Deliberately *not* a plain `match` of the country name against
        # intro_para/short_desc: measured, that is the slowest clause in the
        # set, buys no extra recall, and pads the candidate list for generic
        # mentions that should have had no candidates at all.)
        for phrase in country_phrase_variants(query_term, country):
            base_should_clauses += [
                {"match_phrase": {"title": {"query": phrase, "boost": country_title_boost}}},
                {"match_phrase": {"redirects": {"query": phrase, "boost": country_redirects_boost}}},
            ]

        if use_importance:
            # Boost articles by redirect count (proxy for Wikipedia importance)
            importance_clauses = [
                {"range": {"redirect_count": {"gte": 5, "boost": 5}}},
                {"range": {"redirect_count": {"gte": 10, "boost": 30}}},
                {"range": {"redirect_count": {"gte": 20, "boost": 100}}},

                # Boost articles with infoboxes
                {"exists": {"field": "infobox", "boost": 3}},

                # Boost articles with short descriptions
                {"exists": {"field": "short_desc", "boost": 2}}
            ]

            all_should_clauses = base_should_clauses + importance_clauses
        else:
            all_should_clauses = base_should_clauses

        if not limit_term:
            query = {
                "bool": {
                    "should": all_should_clauses,
                    "minimum_should_match": 1
                }
            }
        else:
            # Your existing limit_term logic
            limit_fields = [
                "title^100", "redirects^100", "alternative_names",
                "intro_para", "categories", "infobox"
            ]
            query = {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": base_should_clauses,
                                "minimum_should_match": 1
                            }
                        },
                        {
                            "multi_match": {
                                "query": limit_term,
                                "fields": limit_fields,
                                "type": "most_fields"
                            }
                        }
                    ]
                }
            }

        # Execute search
        res = self.conn.query(query)[0:max_results].execute()
        results = [hit.to_dict()['_source'] for hit in res['hits']['hits']]
        scores = [hit.to_dict()['_score'] for hit in res['hits']['hits']]

        for i, result in enumerate(results):
            result['raw_es_score'] = scores[i]

        logger.debug(f"Number of hits for Wiki query: {len(results)}")
        logger.debug(f"Titles of the first five results: {[result['title'] for result in results[0:5]]}")

        return results


#######################################################
# Wikipedia Searcher
#######################################################

class WikiSearcher:
    """
    Search and filter Wikipedia results.
    
    This class provides methods for searching Wikipedia and
    processing search results.
    
    Example:
        client = WikiClient()
        searcher = WikiSearcher(client)
        results = searcher.run_wiki_search("Barack Obama")
        filtered = searcher._trim_results(results)
    """
    
    def __init__(self, 
                 wiki_client: None | WikiClient=None, 
                 es_client: None | Elasticsearch=None):
        """
        Initialize the Wikipedia searcher.
        
        Args:
            wiki_client: WikiClient instance
            es_client: Elasticsearch client (used if wiki_client not provided)
        """
        match wiki_client, es_client:
            case None, None:
                raise ValueError("Must provide either wiki_client or es_client")
            case None, _:
                self.wiki_client = WikiClient(es_client=es_client)
            case _, _:
                self.wiki_client = wiki_client
    
    def search_wiki(self, query_term, limit_term="", max_results=200, country=""):
        """
        Search Wikipedia for a given query term.

        Args:
            query_term: Term to search for
            limit_term: Term to limit results by
            max_results: Maximum number of results to return
            country: Country *name* (e.g. "Ghana") the mention belongs to, if
                known. Adds country-qualified title clauses to the search.

        Returns:
            list: List of Wikipedia article dictionaries
        """
        # Clean query term
        query_term = clean_query(query_term)
        logger.debug(f"Using query term: '{query_term}'")

        # Perform search via client
        return self.wiki_client.run_wiki_search(
            query_term=query_term,
            limit_term=limit_term,
            max_results=max_results,
            country=country,
        )

    def text_ranker_features(self, matches, fields):
        """
        Extract and combine text from specified fields in Wiki matches.
        
        Args:
            matches: List of Wikipedia match dictionaries
            fields: List of fields to extract
            
        Returns:
            list: List of combined text strings
        """
        wiki_text = []
        
        for match in matches:
            combined_text = ""
            
            for field in fields:
                try:
                    field_value = match[field]
                    
                    # Handle different field types
                    if isinstance(field_value, str):
                        sentences = field_value.split("\n")
                        if sentences:
                            combined_text += " " + sentences[0]
                    elif isinstance(field_value, list):
                        combined_text += ", ".join(field_value)
                except KeyError:
                    logger.debug(f"Missing key {field} for {match['title']}")
                    continue
                    
            wiki_text.append(combined_text.strip())
            
        return wiki_text

    def _trim_results(self, results):
        """
        Remove bad Wikipedia articles from search results.
        
        Args:
            results: List of Wikipedia article dictionaries
            
        Returns:
            list: Filtered list of articles
        """
        # Early return if no results
        if not results:
            return []
            
        # Filter out articles without intro paragraph
        good_res = [r for r in results if 'intro_para' in r and r['intro_para']]
        
        # Filter out disambiguation and stub pages
        patterns_to_exclude = [
            (r"(stub|User|Wikipedia)\:", 'title'),
            (r"^Wikipedia\:", 'title'),
            (r"^Talk\:", 'title'),
            (r"disambiguation", 'title'),
            (r"^Template:", 'title'),
            (r"^Category:", 'title'),
            (r"^Portal:", 'title'),
            (r"Category\:", lambda r: r['intro_para'][0:50]),
            (r"is the name of", lambda r: r['intro_para'][0:50]),
            (r"may refer to", lambda r: r['intro_para'][0:50]),
            (r"is used as an abbreviation for", lambda r: r['intro_para'][0:40]),
            (r"can refer to", lambda r: r['intro_para'][0:50]),
            (r"most commonly refers to", lambda r: r['intro_para'][0:50]),
            (r"usually refers to", lambda r: r['intro_para'][0:80]),
            (r"may stand for", lambda r: r['intro_para'][0:80]),
            (r"is a surname", lambda r: r['intro_para'][0:50])
        ]
        
        # Apply each exclusion pattern
        for pattern, field_getter in patterns_to_exclude:
            if callable(field_getter):
                good_res = [r for r in good_res if not re.search(pattern, field_getter(r))]
            else:
                good_res = [r for r in good_res if field_getter not in r or not re.search(pattern, r[field_getter])]
        
        return good_res


#######################################################
# Wikipedia Matcher
#######################################################


def load_wiki_ranker_model(model_path: str | Path) -> tuple[XGBClassifier, XGBClassifier]:
    """
    Load the Wikipedia ranker models

    One model has context-related features and the other doesn't.
    (We need this to handle the case where the context is not provided)
    
    Args:
        model_dir: Directory containing the ranker model
        
    Returns:
        XGBoost models: Tuple of loaded ranker model
    """
    model_path = Path(model_path)
    
    wiki_ranker = XGBClassifier()
    wiki_ranker.load_model(model_path)
    logger.warning("Using context-based XGBoost model for *no context* ranking.")
    wiki_ranker_no_context =  XGBClassifier()
    wiki_ranker_no_context.load_model(model_path)
    
    return wiki_ranker, wiki_ranker_no_context


def load_actor_sim_model(model_dir: str | Path) -> SentenceTransformer:
    """
    Load the actor similarity model trained on Wikipedia redirects.
    
    This model helps identify if two names refer to the same entity.
    
    Args:
        model_dir: Directory containing the similarity model
        
    Returns:
        SentenceTransformer: Loaded similarity model
    """
    model_dir = Path(model_dir)
    return SentenceTransformer(str(model_dir))


class WikiMatcher:
    """
    Match entities to Wikipedia articles.
    
    This class provides methods for matching entities to the best
    Wikipedia article based on various criteria.
    
    Example:
        client = WikiClient()
        searcher = WikiSearcher(client)
        matcher = WikiMatcher(searcher)
        best_article = matcher.query_wiki("Barack Obama")
    """

    # TODO #26: allow overriding models; need to check this works correctly as implemented
    
    def __init__(self, 
                 es_client: Elasticsearch,
                 model_manager: ModelManager,
                 wiki_sort_method="neural",
                 trf_model=None, 
                 nlp=None,
                 actor_sim_model: None | str | Path=None, 
                 wiki_ranker_model: None | str | Path=None,
                 device=None,
                 ):
        """
        Initialize the Wikipedia matcher.
        
        Args:
            wiki_searcher: WikiSearcher instance
            trf_model: Sentence transformer model
            actor_sim_model: Actor similarity model
            device: Device to use for inference ('cuda' or None)
            wiki_sort_method: Method to use for sorting results
        """
        # Initialize components or use provided ones
        self.wiki_searcher = WikiSearcher(es_client=es_client)
            
        # Initialize models if not provided
        if trf_model is None:
            self.trf = model_manager.load_trf_model()
        else:
            self.trf = trf_model
        
        if nlp is None:
            self.nlp = model_manager.load_spacy_lg()
        else:
            self.nlp = nlp

        # Actor similarity model 
        if actor_sim_model is None:
            actor_sim_model = Path(str(resources.files("ngec"))) / "assets" / "actor_sim_model2"
        self.actor_sim = load_actor_sim_model(actor_sim_model)

        # Wiki Ranker models (xgboost)
        if wiki_ranker_model is None:
            wiki_ranker_model = Path(str(resources.files("ngec"))) / "assets" / 'xgb_model.json'
        self.wiki_ranker, self.wiki_ranker_no_context = load_wiki_ranker_model(wiki_ranker_model)

        self.wiki_sort_method = wiki_sort_method
            
    def _find_exact_title_matches(self, query_term, results, country=None):
        """
        Find exact matches between query term and Wikipedia article titles.
        
        Args:
            query_term: Query term to match
            results: List of Wikipedia article dictionaries
            country: Country to include in matching
            
        Returns:
            list: List of matching articles
        """
        query_country = f"{query_term} ({country})" if country else query_term
        exact_matches = []
        
        for result in results:
            # Check various forms of the title
            title = result['title']
            if (query_term == title or 
                query_country == title or
                query_term.upper() == title or 
                query_country.upper() == title or
                query_term == remove_accents(title) or 
                query_country == remove_accents(title) or
                query_term.title() == title or 
                query_country.title() == title):
                exact_matches.append(result)
                
        return exact_matches

    def _find_redirect_matches(self, query_term, results, country=None):
        """
        Find matches between query term and Wikipedia article redirects.
        
        Args:
            query_term: Query term to match
            results: List of Wikipedia article dictionaries
            country: Country to include in matching
            
        Returns:
            list: List of matching articles
        """
        query_country = f"{query_term} ({country})" if country else query_term
        redirect_matches = []
        
        for result in results:
            if 'redirects' not in result:
                continue
                
            redirects = result['redirects']
            if (query_term in redirects or 
                query_country in redirects or
                query_term.title() in redirects or 
                query_country.title() in redirects or
                query_term.upper() in redirects or 
                query_country.upper() in redirects):
                redirect_matches.append(result)
                
        return redirect_matches

    def _find_alt_name_matches(self, query_term, results):
        """
        Find matches between query term and Wikipedia article alternative names.
        
        Args:
            query_term: Query term to match
            results: List of Wikipedia article dictionaries
            
        Returns:
            list: List of matching articles
        """
        alt_matches = []
        
        for result in results:
            # Check alternative names
            if 'alternative_names' in result and (
                query_term in result['alternative_names'] or
                query_term.title() in result['alternative_names']):
                alt_matches.append(result)
                
            # Check infobox name
            elif ('infobox' in result and 
                  'name' in result['infobox'] and
                  query_term == result['infobox']['name']):
                alt_matches.append(result)
                
        return alt_matches


    def _check_titles_similarity(self, query_term, candidates):
        """
        Check similarity between query term and candidate titles using actor_sim model.
        
        Args:
            query_term: Query term to match
            candidates: List of candidate articles
            
        Returns:
            tuple: (best_match, similarity_score) or (None, 0)
        """
        if not candidates:
            return None, 0
            
        titles = [c['title'] for c in candidates[0:50]]  # Limit to first 50
        
        # Encode titles and query
        enc_titles = self.actor_sim.encode(titles, show_progress_bar=False)
        enc_query = self.actor_sim.encode(query_term, show_progress_bar=False)
        
        # Get similarity scores
        sims = cos_sim(enc_query, enc_titles)
        best_score = torch.max(sims)
        
        if best_score > THRESHOLD_NEURAL_TITLE_MATCH:
            best_idx = torch.argmax(sims)
            best_match = candidates[best_idx]
            best_match['wiki_reason'] = f"High neural similarity between query and Wiki title: {best_score}"
            return best_match, best_score
            
        return None, best_score
    
    def _edit_distance(self, articles, query_term):
        """
        Calculate simple edit distance between query term and titles.
        """
        if not articles:
            return None
        titles = [article['title'] for article in articles]
            
        # Use Levenshtein distance
        levenshtein = [pylcs.edit_distance(query_term, title) for title in titles]
        levenshtein_sim = []
        for n, i in enumerate(levenshtein):
            max_len = max(len(query_term), len(titles[n]))
            if max_len == 0:
                levenshtein_sim.append(0)
            else:
                levenshtein_sim.append(1 - i / max_len)
        longest_common_subseq = [pylcs.lcs_sequence_length(query_term, title) for title in titles]
        lcs_sim = []
        for n, i in enumerate(longest_common_subseq):
            max_len = max(len(query_term), len(titles[n]))
            if max_len == 0:
                lcs_sim.append(0)
            else:
                lcs_sim.append(i / max_len)
        best_subseq = [int(n == np.argmax(levenshtein_sim)) for n in range(len(lcs_sim))]
        best_lev = [int(n == np.argmax(levenshtein)) for n in range(len(levenshtein_sim))]
        output = {"levenshtein": levenshtein_sim, 
              "lcs": lcs_sim, 
              "best_subseq": best_subseq, 
              "best_lev": best_lev}
        return output

        

    def _create_scoring_dataframe(self, articles, query_term, context, actor_desc, country):
        """
        Create a pandas DataFrame with scores for each article, using batched computations.

        Args:
            articles: List of Wikipedia article dictionaries
            query_term: Query term to match
            context: Context text to help with disambiguation
            country: Country code to help with disambiguation
            actor_desc: Actor description to help with disambiguation

        Returns:
            pandas.DataFrame: DataFrame with article scores
        """
        logger.debug("Creating scoring rules df...")
        if len(articles) == 0:
            return pd.DataFrame()
        # Prepare data for DataFrame
        data = []

        # Things that depend on the document, not the candidate, so they are
        # computed once here rather than once per candidate article.
        detector = get_country_detector()
        # The country the *story* is about, which is often not the country of
        # the mention itself (and `country` below is frequently empty).
        doc_country = detector.most_frequent_country(context) if context else ""
        context_caps = capitalized_words(context)
        context_tfidf = tfidf_vector(context, load_idf_table()) if context else {}
        query_words_cased = set(re.findall(r"[A-Za-z'\u2019\-]{2,}", query_term))

        # Prepare basic matching scores (non-embedding based)

        for i, article in enumerate(articles):
            title = article['title']

            # Calculate various match scores
            exact_title_match = 1 if self._is_exact_title_match(query_term, article, country) else 0
            redirect_match = 1 if self._is_redirect_match(query_term, article, country) else 0
            alt_name_match = 1 if self._is_alt_name_match(query_term, article) else 0
            country_match = 0
            if country:
                if (re.search(country, article['intro_para']) or re.search(country, article['short_desc'])):
                    country_match = 1

            # Number of alternative names
            alt_names_count = len(article.get('alternative_names', []))
            redirect_names_count = len(article.get('redirects', []))
            results_count = len(articles)

            # Intro paragraph length
            intro_length = len(article.get('intro_para', ''))

            # Name coverage: fraction of query words that appear in the title
            # (+ redirects + alt names). Low coverage signals a false positive.
            query_words = set(query_term.lower().split())
            title_words = set(title.lower().split())
            all_name_words = title_words.copy()
            for r in article.get('redirects', []):
                all_name_words.update(r.lower().split())
            for a in article.get('alternative_names', []):
                all_name_words.update(a.lower().split())
            if query_words:
                name_coverage = len(query_words & all_name_words) / len(query_words)
            else:
                name_coverage = 0

            # Category overlap: how many category words appear in the document context
            cat_overlap = 0
            if context:
                context_lower = context.lower()
                for cat in article.get('categories', []):
                    cat_words = cat.lower().split()
                    if any(w in context_lower for w in cat_words if len(w) > 3):
                        cat_overlap += 1

            # Does the story's country show up in this candidate article? The
            # existing `country_match` above asks the same question of the
            # `country` the caller passed in, which in the pipeline is usually
            # empty; these ask it of the country detected from the story itself.
            categories = article.get('categories', [])
            intro_para = article.get('intro_para', '')
            short_desc = article.get('short_desc', '')
            cm_doc = int(bool(doc_country)
                         and (doc_country in intro_para or doc_country in short_desc))
            cm_title = int(bool(doc_country) and doc_country in title)
            cm_cat = int(bool(doc_country) and any(doc_country in c for c in categories))

            # Word-level overlap between the story and the candidate's intro:
            # rare words (TF-IDF) and shared names (capitalised words). Both are
            # 0 when there is no context to compare against.
            tfidf_ctx_intro = tfidf_cosine(context_tfidf,
                                           tfidf_vector(intro_para, load_idf_table())) if context_tfidf else 0.0
            intro_caps = capitalized_words(intro_para)
            shared_caps = (intro_caps & context_caps) - query_words_cased
            pn_overlap = len(shared_caps)
            pn_overlap_frac = pn_overlap / len(intro_caps) if intro_caps else 0.0

            is_generic_concept, has_other_country = title_concept_features(
                title, intro_length, doc_country, detector)

            data.append({
                'index': i,
                'title': title,
                'short_desc': article.get('short_desc', ''),
                'intro_para': article.get('intro_para', ''),
                'categories': article.get('categories', []),
                'infobox': article.get('infobox', {}),
                'query_term': query_term,
                'exact_title_match': exact_title_match,
                'redirect_match': redirect_match,
                'alt_name_match': alt_name_match,
                'alt_names_count': alt_names_count,
                'redirect_names_count': redirect_names_count,
                'num_es_results': results_count,
                'intro_length': intro_length,
                'country_match': country_match,
                'cm_doc': cm_doc,
                'cm_title': cm_title,
                'cm_cat': cm_cat,
                'tfidf_ctx_intro': tfidf_ctx_intro,
                'pn_overlap': pn_overlap,
                'pn_overlap_frac': pn_overlap_frac,
                'n_categories': len(categories),
                'title_is_generic_concept': is_generic_concept,
                'title_has_other_country': has_other_country,
                'from_alt_query': article.get('from_alt_query', 0),
                'name_coverage': name_coverage,
                'cat_overlap': cat_overlap,
                'raw_es_score': article.get('raw_es_score', 0),
                'log_es_score': np.log1p(article.get('raw_es_score', 0)),

                'title_sim': 0,  # Will be filled in later
                'context_sim_intro': 0,  # Will be filled in later
                'context_sim_short': 0,  # Will be filled in later
                'actor_desc_sim_intro': 0,  # Will be filled in later
                'actor_desc_sim_short': 0,  # Will be filled in later
                'combined_score': 0  # Will be calculated after all scores are in
            })

        # Create DataFrame with initial data
        df = pd.DataFrame(data)
        
        edit_distance = self._edit_distance(articles, query_term)
        if edit_distance is None:
            df['levenshtein'] = None
            df['lcs'] = None
            df['best_subseq'] = None 
            df['best_lev'] = None
        else:
            df['levenshtein'] =  edit_distance['levenshtein']
            df['lcs'] = edit_distance['lcs']
            df['best_subseq'] = edit_distance['best_subseq']
            df['best_lev'] = edit_distance['best_lev']

        # Batch compute title similarity
        if query_term:
            titles = [article['title'] for article in articles]
            # Encode query once
            query_embedding = self.actor_sim.encode(query_term, show_progress_bar=False)
            # Encode all titles in one batch
            title_embeddings = self.actor_sim.encode(titles, show_progress_bar=False)
            # Compute similarities
            title_sims = cos_sim(query_embedding.reshape(1, -1), title_embeddings)
            # Add to dataframe
            df['title_sim'] = title_sims[0].tolist()

        # Batch compute context similarity
        if context or actor_desc:
            intros = [article['intro_para'][0:600] for article in articles]
            short_descs = [article['short_desc'] for article in articles]
            # Encode context once
            intro_embeddings = self.trf.encode(intros, show_progress_bar=False)
            short_desc_embeddings = self.trf.encode(short_descs, show_progress_bar=False)
        
        if context:
            context_embedding = self.trf.encode(context, show_progress_bar=False)
            # Compute similarities
            context_sims = cos_sim(context_embedding.reshape(1, -1), intro_embeddings)
            short_desc_sims = cos_sim(context_embedding.reshape(1, -1), short_desc_embeddings)
            # Add to dataframe
            df['context_sim_intro'] = context_sims[0].tolist()
            df['context_sim_short'] = short_desc_sims[0].tolist()

        if actor_desc:
            # Encode actor description once
            desc_embedding = self.trf.encode(actor_desc, show_progress_bar=False)
            # Compute similarities
            desc_sims_intro = cos_sim(desc_embedding.reshape(1, -1), intro_embeddings)
            desc_sims_short = cos_sim(desc_embedding.reshape(1, -1), short_desc_embeddings)
            # Add to dataframe
            df['actor_desc_sim_intro'] = desc_sims_intro[0].tolist()
            df['actor_desc_sim_short'] = desc_sims_short[0].tolist()

        # The ranker expects an "empty text" feature, which lets it 
        # discount the context similarity columns when they're all 0.
        # Add that feature here
        df['text_is_empty'] = (context == "")

        # Normalize ES score within this candidate set
        es_max = df['raw_es_score'].max()
        es_min = df['raw_es_score'].min()
        df['es_score_norm'] = (df['raw_es_score'] - es_min) / (es_max - es_min) if es_max > es_min else 0

        # Normalize scores
        # The new country and overlap columns get `_norm` twins because their
        # existing counterparts (`country_match`, `context_sim_intro`) have them.
        for col in ['title_sim', 'context_sim_intro', 'context_sim_short',
                    'actor_desc_sim_intro', 'actor_desc_sim_short',
                    'lcs', 'levenshtein', 'country_match', 'exact_title_match',
                    'alt_name_match', 'redirect_match',
                    'cm_doc', 'cm_title', 'cm_cat',
                    'tfidf_ctx_intro', 'pn_overlap']:
            col_norm, normed = self._normalize_scores(df, col)
            df[col_norm] = normed

        # Calculate combined score with appropriate weighting
        df['combined_score'] = (
            df['exact_title_match'] * 10 +
            df['redirect_match'] * 5 +
            df['alt_name_match'] * 3 +
            df['title_sim'] * 1 +
            df['context_sim_intro'] * 2 +
            df['context_sim_short'] * 2 +
            df['actor_desc_sim_intro'] * 2 +
            df['actor_desc_sim_short'] * 2 +
            df['country_match'] * 2 +
            np.log1p(df['alt_names_count']) * 0.5 +
            np.log1p(df['intro_length']) * 0.1
        )

        return df
    
    def _normalize_scores(self, df, col):
        """
        Scale a column by its own maximum within this candidate set.

        When every candidate scores 0 the division gives NaN, which XGBoost
        treats as missing -- that is the intended behavior. It can also give
        +/- infinity: a column whose maximum is exactly 0.0 but which has
        negative entries below it (an encoder that maps an empty short
        description to the zero vector produces exactly this). XGBoost refuses
        infinite inputs outright, so those become NaN too.
        """
        col_norm = f"{col}_norm"
        normed = df[col] / df[col].max()
        return (col_norm, normed.replace([np.inf, -np.inf], np.nan))

    def _apply_selection_rules(self, df, articles, context):
        """
        Apply a series of prioritized rules to select the best article.

        Args:
            df: DataFrame with article scores
            articles: List of Wikipedia article dictionaries
            context: Context text to help with disambiguation

        Returns:
            dict or None: Selected article or None if no good match
        """
        # Rule 1: Single exact title match - highest priority
        exact_matches = df[df['exact_title_match'] == 1]
        logger.debug(f"Exact title matches found: {len(exact_matches)}")
        if len(exact_matches) == 1:
            selected = articles[exact_matches.iloc[0]['index']]
            logger.debug("Returning single exact title match")
            selected['wiki_reason'] = "Single exact title match"
            return selected
        logger.debug("No single exact title match found")

        # Rule 2: Multiple exact matches - use context if available
        if len(exact_matches) > 1 and context:
            logger.debug("Multiple exact title matches found, checking context...")
            best_context_match = exact_matches.sort_values('context_sim_intro', ascending=False).iloc[0]
            if best_context_match['context_sim_intro'] > 0.5:  # Threshold for good context match
                selected = articles[best_context_match['index']]
                logger.debug("Returning best context match among exact title matches")
                selected['wiki_reason'] = "Best context match among exact title matches"
                return selected

        # Rule 3: Multiple exact matches - use title similarity
        logger.debug("Multiple exact title matches found, checking title similarity...")
        if len(exact_matches) > 1:
            best_title_match = exact_matches.sort_values('title_sim', ascending=False).iloc[0]
            if best_title_match['title_sim'] > 0.7:  # Threshold for good title match
                selected = articles[best_title_match['index']]
                logger.debug("Returning best title match among exact title matches")
                selected['wiki_reason'] = "Best title similarity among exact matches"
                return selected

        # Rule 4: Single redirect match
        logger.debug("Checking for single redirect match...")
        redirect_matches = df[df['redirect_match'] == 1]
        if len(redirect_matches) == 1:
            selected = articles[redirect_matches.iloc[0]['index']]
            logger.debug("Returning single redirect match")
            selected['wiki_reason'] = "Single redirect match"
            return selected

        # Rule 5: Multiple redirect matches - use context
        if len(redirect_matches) > 1 and context:
            logger.debug("Multiple redirect matches found, checking context...")
            best_context_match = redirect_matches.sort_values('context_sim_intro', ascending=False).iloc[0]
            if best_context_match['context_sim_intro'] > 0.5:
                selected = articles[best_context_match['index']]
                logger.debug(f"Returning best context match among redirect matches. Title: {selected['title']}: {best_context_match['context_sim_intro']}")
                selected['wiki_reason'] = "Best context match among redirect matches"
                return selected

        # Rule 6: Multiple redirect matches - use title similarity
        if len(redirect_matches) > 1:
            logger.debug("Multiple redirect matches found, checking title similarity...")
            best_redirect = redirect_matches.sort_values('title_sim', ascending=False).iloc[0]
            if best_redirect['title_sim'] > 0.7:
                selected = articles[best_redirect['index']]
                logger.debug(f"Returning best title match among redirect matches. Title: {selected['title']}: {best_redirect['title_sim']}")
                selected['wiki_reason'] = "Best title similarity among redirect matches"
                return selected

        # Rule 7: Alternative name matches with good context
        alt_matches = df[df['alt_name_match'] == 1]
        if len(alt_matches) > 0 and context:
            logger.debug("Checking for alternative name matches with context...")
            best_alt_match = alt_matches.sort_values('context_sim_intro', ascending=False).iloc[0]
            if best_alt_match['context_sim_intro'] > 0.6:
                selected = articles[best_alt_match['index']]
                logger.debug(f"Returning best context match among alternative name matches. Title: {selected['title']}: {best_alt_match['context_sim_intro']}")
                selected['wiki_reason'] = "Best context match among alternative name matches"
                return selected

        # Rule 8: Fall back to combined score for any article with good context match
        logger.debug("Checking for any article with good context match...")
        if context:
            best_context = df.sort_values('context_sim_intro', ascending=False).iloc[0]
            if best_context['context_sim_intro'] > 0.6:  # Higher threshold for general context match
                selected = articles[best_context['index']]
                logger.debug(f"Returning best context match overall. Title: {selected['title']}: {best_context['context_sim_intro']}")
                selected['wiki_reason'] = "Best overall context match"
                return selected

        # Rule 9: Fall back to combined score as last resort
        logger.debug("No good matches found, checking combined score...")
        best_overall = df.sort_values('combined_score', ascending=False).iloc[0]
        if best_overall['combined_score'] > THRESHOLD_COMBINED_SCORE:  # Threshold for accepting combined score
            selected = articles[best_overall['index']]
            logger.debug(f"Returning best overall combined score: {selected['title']}: {best_overall['combined_score']}")
            selected['wiki_reason'] = "Best overall combined score"
            return selected

        # No good match found
        logger.debug("No good match found, returning None")
        return None

    
    def _is_exact_title_match(self, query_term, article, country):
        """Check if query_term exactly matches article title."""
        query_country = f"{query_term} ({country})" if country else query_term
        title = article['title']
        return (query_term == title or 
                query_country == title or
                query_term.upper() == title or 
                query_country.upper() == title or
                query_term == remove_accents(title) or 
                query_country == remove_accents(title) or
                query_term.title() == title or 
                query_country.title() == title)

    def _is_redirect_match(self, query_term, article, country):
        """Check if query_term matches any redirect."""
        if 'redirects' not in article:
            return False

        query_country = f"{query_term} ({country})" if country else query_term
        redirects = article['redirects']
        return (query_term in redirects or 
                query_country in redirects or
                query_term.title() in redirects or 
                query_country.title() in redirects or
                query_term.upper() in redirects or 
                query_country.upper() in redirects)

    def _is_alt_name_match(self, query_term, article):
        """Check if query_term matches any alternative name."""
        if 'alternative_names' in article and (
            query_term in article['alternative_names'] or
            query_term.title() in article['alternative_names'] or
            query_term.upper() in article['alternative_names'] or
            query_term in [remove_accents(name) for name in article['alternative_names']] or
            query_term.title() in [remove_accents(name) for name in article['alternative_names']] or
            query_term.upper() in [remove_accents(name) for name in article['alternative_names']] or
            query_term in [name.title() for name in article['alternative_names']]):
            return True

        if ('infobox' in article and 
            'name' in article['infobox'] and
            query_term == article['infobox']['name']):
            return True

        return False

    def _calculate_title_similarity(self, query_term, article):
        """Calculate similarity between query_term and article title."""
        # This could call into the existing _check_titles_similarity method
        # but return just the similarity score
        title = article['title']
        enc_query = self.actor_sim.encode(query_term, show_progress_bar=False)
        enc_title = self.actor_sim.encode(title, show_progress_bar=False)
        sims = cos_sim(enc_query, enc_title)
        return float(sims[0][0])

    def _calculate_context_similarity(self, context, article):
        """Calculate similarity between context and article intro."""
        intro = article['intro_para'][0:200]
        enc_context = self.trf.encode(context, show_progress_bar=False)
        enc_intro = self.trf.encode(intro, show_progress_bar=False)
        sims = cos_sim(enc_context, enc_intro)
        return float(sims[0][0])

    def _calculate_actor_desc_similarity(self, actor_desc, article):
        """Calculate similarity between actor description and article intro/categories."""
        if not actor_desc:
            return 0

        # Combine intro and categories for a rich description
        article_info = article['intro_para'][0:200]
        if 'categories' in article:
            article_info += " " + " ".join(article['categories'])

        enc_desc = self.trf.encode(actor_desc, show_progress_bar=False)
        enc_info = self.trf.encode(article_info, show_progress_bar=False)
        sims = cos_sim(enc_desc, enc_info)
        return float(sims[0][0])
    
    def _call_ranker(self, score_df, context):
        if context:
            X = score_df[self.wiki_ranker.feature_names_in_]
            y_proba = self.wiki_ranker.predict_proba(X)[:, 1]
        else:
            X = score_df[self.wiki_ranker_no_context.feature_names_in_]
            y_proba = self.wiki_ranker_no_context.predict_proba(X)[:, 1]
        score_df['ranker_score'] = y_proba
        score_df['is_max_for_task'] = (score_df['ranker_score'] == score_df['ranker_score'].max()).astype(int)
        score_df['is_predicted_match'] = score_df['is_max_for_task'] & (score_df['ranker_score'] > 0.1)
        pick = score_df[score_df['is_predicted_match'] == True]
        if not pick.empty:
            pick = pick.iloc[0].to_dict()
            logger.debug(f"Found an article with neural methods. Title: {pick['title']}")
        else:
            pick = None
        return pick


    def pick_best_wiki(self, 
                       query_term, 
                       results, 
                       context="", 
                       country="",
                       actor_desc="",
                       wiki_sort_method: Literal["rules", "neural"]='neural', 
                       rank_fields=None):
        """
        Select the best Wikipedia article from search results using a scoring matrix approach.

        Args:
            query_term: Query term to match
            results: List of Wikipedia article search results
            context: Context text to help with disambiguation
            country: Country code to help with disambiguation
            actor_desc: Actor description to help with disambiguation
            wiki_sort_method: Literal["rules", "neural"]: Method to use for sorting results
            rank_fields: Fields to use for ranking

        Returns:
            dict or None: Best matching Wikipedia article or None if no good match
        """
        logger.debug(f"Using wiki sort method {wiki_sort_method}")
        # Use instance method if not provided
        if wiki_sort_method is None:
            wiki_sort_method = self.wiki_sort_method

        # Set default rank fields if not provided
        if rank_fields is None:
            rank_fields = ['title', 'categories', 'alternative_names', 'redirects']

        # Clean query term
        query_term = clean_query(query_term)
        logger.debug(f"Using query term '{query_term}'")

        # Handle empty results
        if not results:
            logger.debug("No Wikipedia results. Returning None")
            return None

        # Remove disambiguation pages, stubs, etc.
        good_res = self.wiki_searcher._trim_results(results)
        if not good_res:
            logger.debug("No valid results after filtering disambiguation pages, etc.")
            return None

        logger.debug(f"Filtered down to {len(good_res)} valid results")

        # Early exit: If only one result, return it
        if len(good_res) == 1:
            best = good_res[0]
            best['wiki_reason'] = "Only one valid result"
            logger.debug(f"Only one valid result: {best['title']}")
            return best
        
        score_df = self._create_scoring_dataframe(good_res, query_term, context=context, 
                                                actor_desc=actor_desc, 
                                                country=country) 
        if wiki_sort_method == "rules":
            logger.debug("Using rules-based selection method")
            best = self._apply_selection_rules(score_df, articles=good_res, context=context)
            if best:
                logger.debug(f"Selected article using rules: {best['title']} (reason: {best['wiki_reason']})")
                return best
        else:
            logger.debug("Using neural selection method")
            # Create scoring dataframe
            

            # Apply prioritized selection rules
            logger.debug("Calling ranker...")
            selected = self._call_ranker(score_df, context=context)

            if selected is not None:
                selected['wiki_reason'] = f"XGBoost (score={selected['ranker_score']:.4f})"
                logger.debug(f"Selected article: {selected['title']} (reason={selected['wiki_reason']})")
                return selected
            else:
                logger.debug("No good match found")
                # print the best overall score
                best_overall = score_df.sort_values('combined_score', ascending=False).iloc[0].to_dict()
                logger.debug(f"Skipping best overall score: ({best_overall['title']}): {best_overall['combined_score']}")
                return None
        return None
        
    def _expand_query(self, query_term, context=None, doc=None):
        """
        Expand the query term using named entity recognition (NER) and acronyms.
        """
        if not context and not doc:
            logger.info("Either context or doc must be provided for NER expansion--returning original query term")
            return query_term 

        if context and not doc:
            context_doc = self.nlp(context)
        else:
            context_doc = doc

        if context_doc:
            logger.debug(f"Expanding query term '{query_term}' using NER and acronyms")
            # Find entities containing the query term, preferring:
            # 1. Word-boundary matches (query is a full word in the entity, not a substring)
            # 2. Entities that start with the query term
            # 3. The longest match (most informative)
            candidates = [i.text for i in context_doc.ents if query_term in i.text and len(i.text) > len(query_term)]
            if candidates:
                # Prefer word-boundary matches (e.g. "Xi" should match "Xi Jinping" not "Bo Xilai")
                boundary_pattern = r'(?:^|\s)' + re.escape(query_term) + r'(?:\s|$)'
                boundary_matches = [c for c in candidates if re.search(boundary_pattern, c)]
                if boundary_matches:
                    # Among boundary matches, prefer those starting with the query
                    starts_with = [c for c in boundary_matches if c.startswith(query_term)]
                    if starts_with:
                        expanded_query = max(starts_with, key=len)
                    else:
                        expanded_query = max(boundary_matches, key=len)
                else:
                    expanded_query = max(candidates, key=len)
                query_term = expanded_query
                logger.debug(f"Used NER to expand context: {expanded_query}")
        
            acronym_dict = make_acronym_dicts(doc=context_doc)
            # Check if query term is an acronym
            # and expand it if found in the acronym dictionary
            if query_term in acronym_dict:
                logger.debug(f"Using acronym expansion: {query_term} --> {acronym_dict[query_term]}")
                query_term = acronym_dict[query_term]
            if query_term in acronym_dict:
                query_term = acronym_dict[query_term]
                logger.debug(f"Using acronym expansion: {query_term}")
        return query_term

    def _pick_alt_query_term(self, query_term, alt_query_terms):
        """
        Pick the first alternative surface form worth a second search.

        "Worth a second search" means: it survives `clean_query` and, once
        cleaned, is not the same string as the primary query term (comparing
        the cleaned forms, since that is what actually gets sent to
        Elasticsearch). Returns "" when there is nothing to add.
        """
        if not alt_query_terms:
            return ""
        primary = clean_query(query_term)
        for term in alt_query_terms:
            cleaned = clean_query(term)
            if cleaned and cleaned != primary:
                return cleaned
        return ""

    def query_wiki(self,
                   query_term,
                   limit_term="",
                   country="",
                   context="",
                   actor_desc="",
                   method="neural",
                   max_results=200,
                   skip_expansion=False,
                   alt_query_terms: list[str] | None = None):
        """
        Search Wikipedia and return the best matching article.

        Args:
            query_term: Term to search for
            limit_term: Term to limit results by
            country: Country *name* (e.g. "Ghana") to help with disambiguation
            context: Context text to help with disambiguation
            actor_desc: Actor description (automatically parsed)
            max_results: Maximum results to return from search
            skip_expansion: If True, skip NER-based query expansion (use when
                the caller already extracted the core entity via NER)
            alt_query_terms: Other surface forms of the same mention (e.g. the
                raw span before country-stripping, or the span before NER
                expansion). The first one that differs from `query_term` is
                searched as well and the two candidate lists are merged. One
                extra Elasticsearch round trip buys about a point of recall,
                because whichever surface form the article is titled under is
                often not the one the pipeline settled on.

        Returns:
            dict or None: Best matching Wikipedia article or None if no good match
        """
        if method not in ["neural", "rules"]:
            raise ValueError(f"Wiki selection method must be 'neural' or 'rules'. You provided: {method}")
        # Strip possessive suffix before searching
        query_term = re.sub(r"['’]s\s*$", "", query_term).strip()
        # Do NER expansion unless caller already extracted a specific (multi-word) entity
        if context and not skip_expansion:
            logger.debug("Context present, so attempting NER expansion")
            query_term = self._expand_query(query_term, context)

        # A single search: run_wiki_search already combines exact (term) and
        # fuzzy (match / folded) clauses, so there is no separate fuzzy mode
        # to fall back to. An earlier version re-ran this identical search and
        # re-embedded every candidate whenever pick_best_wiki returned None,
        # doubling the cost of every failed lookup for no change in output.
        logger.debug("Searching Wikipedia")
        results = self.wiki_searcher.search_wiki(
            query_term,
            limit_term=limit_term,
            max_results=max_results,
            country=country,
        )
        for article in results:
            article['from_alt_query'] = 0

        # Optional second search over another surface form of the same mention.
        # The candidate's `raw_es_score` then comes from a different query and
        # is not on the same scale as the primary query's scores, so we flag
        # the candidates that came from it and let the ranker learn that.
        alt_term = self._pick_alt_query_term(query_term, alt_query_terms)
        if alt_term:
            logger.debug(f"Also searching Wikipedia for alternative form '{alt_term}'")
            alt_results = self.wiki_searcher.search_wiki(
                alt_term,
                limit_term=limit_term,
                max_results=max_results,
            )
            for article in alt_results:
                article['from_alt_query'] = 1
            results = merge_ranked_results(results, alt_results, max_results)

        best = self.pick_best_wiki(
            query_term, 
            results, 
            country=country, 
            context=context,
            actor_desc=actor_desc,
            wiki_sort_method=method
        )
        return best



def make_acronym_dicts(text: str | None=None, 
                       doc: Doc | None=None, 
                       nlp: None | Language=None) -> dict[str, str]:
    """
    Quick tool to identify acronyms (and their referents) in a doc.

    Args:
        text: string of text to process
        doc: spaCy doc object
    Returns:
        acronym_entities: dict of acronyms and their referents
    """
    match text, doc:
        case None, None:
            raise ValueError("Either text or doc must be provided.")
        case str(), None:
            if nlp is None:
                raise ValueError("nlp object must be provided if doc is not provided.")
            doc = nlp(text)
        case _, Doc():
            pass

    acronym_entities = {"U.N.": "United Nations", "UN": "United Nations"}
    for ent in doc.ents:    # type: ignore
        # skip cardinals
        if ent.label_ in ["CARDINAL", "DATE", "TIME", "ORDINAL", "QUANTITY"]:
            continue
        # only take non-acronyms
        if len(ent) > 1 and not ent.text.isupper():
            # strip out leading prepositions and articles
            ent_text = ''.join([i.text_with_ws for i in ent if i.pos_ != "DET" and i.pos_ != "ADP"]).strip()
            # only take title case names
            # The title case doesn't always work with some edge cases. E.g. "Ta'ang National Liberation Army".
            # Instead, we can check if the first letter of each word is uppercase.
            first_letters = [True if word[0].isupper() else False for word in ent_text.split()]
            if ent_text.istitle():
                acronym = ''.join([word[0].upper() for word in ent_text.split()])
                acronym_entities[acronym] = ent_text
            elif all(first_letters):
                # If the first letter of each word is uppercase, consider it as a potential acronym
                acronym = ''.join([word[0].upper() for word in ent_text.split()])
                acronym_entities[acronym] = ent_text
    return acronym_entities
    
