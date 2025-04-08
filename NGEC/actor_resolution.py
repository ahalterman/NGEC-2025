import os
import pandas as pd
import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import dateparser
import unidecode
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from collections import Counter
import logging
from textacy.preprocessing.remove import accents as remove_accents
from rich import print
from rich.progress import track
import time
import jsonlines
from scipy.spatial.distance import cdist
import pylcs
from sentence_transformers.util import cos_sim
import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Constants
#DEFAULT_MODEL_PATH = 'sentence-transformers/all-MiniLM-L12-v2'
DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v3"
DEFAULT_SIM_MODEL_PATH = 'actor_sim_model2'
DEFAULT_BASE_PATH = "./assets"

# Threshold constants
THRESHOLD_COSINE_SIMILARITY = 0.8
THRESHOLD_DOT_SIMILARITY = 45
THRESHOLD_NEURAL_TITLE_MATCH = 0.9
THRESHOLD_ALT_NAME_TITLE_MATCH = 0.8
THRESHOLD_CONTEXT_MATCH = 0.7
THRESHOLD_HIGH_CONFIDENCE = 0.90
THRESHOLD_VERY_HIGH_CONFIDENCE = 0.95


def setup_es():
    """Establish connection to Elasticsearch and return search object."""
    try:
        client = Elasticsearch()
        client.ping()
        conn = Search(using=client, index="wiki")
        return conn
    except Exception as e:
        raise ConnectionError(f"Could not connect to Elasticsearch: {e}")


def check_wiki(conn):
    """
    Verify that the Wikipedia index is using the correct format.
    
    We changed the Wikipedia format on 2022-04-21, so make sure we're using the
    right one.
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


def load_county_dict(base_path):
    """
    Construct a list of regular expressions to find countries by their name and nationality.
    
    Returns two lists of pattern tuples:
    1. Regular patterns for direct country mentions (e.g., Germany, German)
    2. Category patterns for indirect mentions (e.g., of German, in Germany)
    """
    file = os.path.join(base_path, "countries.csv")
    countries = pd.read_csv(file)
    
    # Direct country name/nationality patterns
    nat_list = []
    for _, row in countries.iterrows():
        # Handle nationalities
        nationalities = [nat.strip() for nat in row['Nationality'].split(",")]
        for nat in nationalities:
            pattern = (re.compile(nat + r"(?=[^a-z]|$)"), row['CCA3'])
            nat_list.append(pattern)
        
        # Handle country names
        pattern = (re.compile(row['Name']), row['CCA3'])
        nat_list.append(pattern)
    
    # Category patterns (for "of X" or "in X" constructions)
    nat_list_cat = []
    for prefix in ['of ', 'in ']: 
        for _, row in countries.iterrows():
            # Handle nationalities in categories
            nationalities = [nat.strip() for nat in row['Nationality'].split(",")]
            for nat in nationalities:
                pattern = (re.compile(prefix + nat), row['CCA3'])
                nat_list_cat.append(pattern)
            
            # Handle country names in categories
            pattern = (re.compile(prefix + row['Name']), row['CCA3'])
            nat_list_cat.append(pattern)
    
    return nat_list, nat_list_cat


def load_spacy_lg():
    """Load and return the spaCy language model."""
    return spacy.load("en_core_web_lg", disable=["pos", "dep"])


def load_trf_model(model_dir=DEFAULT_MODEL_PATH):
    """Load and return the sentence transformer model."""
    return SentenceTransformer(model_dir, trust_remote_code=True)


def load_actor_sim_model(base_path, model_dir=DEFAULT_SIM_MODEL_PATH):
    """
    Load the actor similarity model trained on Wikipedia redirects.
    
    This model helps identify if two names refer to the same entity.
    """
    combo_path = os.path.join(base_path, model_dir)
    return SentenceTransformer(combo_path)


class ActorResolver:
    # Actor type priority dictionary - used for sorting/ranking actor codes
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

    def __init__(self, 
                spacy_model=None,
                base_path=DEFAULT_BASE_PATH,
                save_intermediate=False,
                wiki_sort_method="neural",
                gpu=False):
        """
        Initialize the ActorResolver with the necessary models and data.
        
        Parameters
        ----------
        spacy_model : spaCy model, optional
            Pre-loaded spaCy model to use
        base_path : str, optional
            Path to the directory containing assets
        save_intermediate : bool, optional
            Whether to save intermediate results
        wiki_sort_method : str, optional
            Method to use for sorting Wikipedia results
        gpu : bool, optional
            Whether to use GPU for model inference
        """
        # Set device for model inference
        self.device = 'cuda' if gpu else None
        
        # Initialize Elasticsearch connection
        self.conn = setup_es()
        
        # Load models
        self.nlp = spacy_model if spacy_model else load_spacy_lg()
        self.trf = load_trf_model()
        self.actor_sim = load_actor_sim_model(base_path)
        
        # Load and preprocess data
        self.agents = self._load_and_clean_agents(base_path)
        self.trf_matrix = self._load_embeddings(base_path)
        self.nat_list, self.nat_list_cat = load_county_dict(base_path)
        
        # Store configuration
        self.base_path = base_path
        self.cache = {}
        self.save_intermediate = save_intermediate
        self.wiki_sort_method = wiki_sort_method

    def _load_and_clean_agents(self, base_path):
        """
        Load the PLOVER/CAMEO agents file and clean the data.
        
        This method replaces the previous clean_agents method.
        """
        file = os.path.join(base_path, "PLOVER_agents.txt")
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()

        # Remove curly braces content
        data = re.sub(r"\{.+?\}", "", data)
        
        # Split into lines and filter
        lines = [line for line in data.split("\n") if line and not line.startswith("#")]
        lines = [re.sub(r"#.+", "", line).strip() for line in lines if not line.startswith("!")]
        
        logger.debug(f"Total agents: {len(lines)}")
        
        # Parse patterns
        patterns = []
        for line in lines:
            try:
                # Extract code from square brackets
                code_match = re.findall(r"\[.+?\]", line)
                if not code_match:
                    continue
                    
                code = re.sub(r"[\[\]~]", "", code_match[0]).strip()
                
                # Extract pattern
                pattern = re.sub(r"(\[.+?\])", "", line)
                pattern = re.sub(r"_", " ", pattern).lower().strip()
                
                patterns.append({
                    "pattern": pattern,
                    "code_1": code[0:3],
                    "code_2": code[3:]
                })
            except Exception as e:
                logger.info(f"Error loading {line}: {e}")
        
        # Handle special pattern replacements
        cleaned_patterns = []
        for pattern in patterns:
            if 'code_1' not in pattern:
                continue
                
            # Handle !minist! placeholder
            if re.search("!minist!", pattern['pattern']):
                for replacement in ["Minister", "Ministers", "Ministry", "Ministries"]:
                    new_pattern = {
                        "code_1": pattern['code_1'],
                        "code_2": pattern['code_2'],
                        "pattern": re.sub(r"!minist!", replacement, pattern['pattern']).title()
                    }
                    cleaned_patterns.append(new_pattern)
            
            # Handle !person! placeholder
            elif re.search("!person!", pattern['pattern']):
                for replacement in ["person", "man", "woman", "men", "women"]:
                    new_pattern = {
                        "code_1": pattern['code_1'],
                        "code_2": pattern['code_2'],
                        "pattern": re.sub(r"!person!", replacement, pattern['pattern'])
                    }
                    cleaned_patterns.append(new_pattern)
            else:
                cleaned_patterns.append(pattern)
                
        return cleaned_patterns

    def _load_embeddings(self, base_path):
        """
        Load pre-computed embedding matrices or compute and save them if needed.
        
        This method checks if the agent embeddings are up-to-date and recomputes
        them if necessary.
        """
        # Check if the agents file and embedding matrix are mismatched
        hash_file = os.path.join(base_path, "PLOVER_agents.hash")
        try:
            with open(hash_file, "r") as f:
                existing_hash = f.read()
        except FileNotFoundError:
            existing_hash = ""
            
        # Get current hash of agents file
        agent_file = os.path.join(base_path, "PLOVER_agents.txt")
        with open(agent_file, "r", encoding="utf-8") as f:
            data = f.read() 
        current_hash = hash(data)
        
        # Recompute embeddings if hash mismatch
        if str(existing_hash) != str(current_hash):
            logger.info("Agents file and pre-computed matrix are mismatched. Recomputing...")
            patterns = [agent['pattern'] for agent in self.agents]
            trf_matrix = self.trf.encode(patterns, device=self.device)
            
            # Save new embeddings and hash
            file_bert = os.path.join(base_path, "bert_matrix.pkl")
            with open(file_bert, "wb") as f:
                pickle.dump(trf_matrix, f)
            with open(hash_file, "w") as f:
                f.write(str(current_hash))
        
        # Load embeddings
        logger.info("Reading in BERT matrix")
        file_bert = os.path.join(base_path, "bert_matrix.pkl")
        with open(file_bert, "rb") as f:
            return pickle.load(f)

    def clean_query(self, qt):
        """
        Clean and normalize a query string.
        
        Removes articles, ordinals, possessives, and other noise from text.
        """
        # Handle empty or simple cases
        qt = str(qt).strip()
        if qt in ['The', 'the', 'a', 'an', '']:
            return ""
            
        # Normalize whitespace
        qt = re.sub(' +', ' ', qt)  # remove multiple spaces
        qt = re.sub('\n+', ' ', qt)  # newline to space
        
        # Remove starting articles and ending prepositions
        qt = re.sub(r"^the ", "", qt, flags=re.IGNORECASE).strip()
        qt = re.sub(r"^an ", "", qt, flags=re.IGNORECASE).strip()
        qt = re.sub(r"^a ", "", qt, flags=re.IGNORECASE).strip()
        qt = re.sub(r" of$", "", qt).strip()
        qt = re.sub(r"^'s", "", qt).strip()
        
        # Remove ordinals
        qt = re.sub(r"(?<=\d\d)(st|nd|rd|th)\b", '', qt).strip()  # two-digit ordinals
        qt = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', qt).strip()    # one-digit ordinals
        
        # Remove leading numbers and possessives
        qt = re.sub(r"^\d+? ", "", qt).strip()
        qt = re.sub(r"'s$", "", qt).strip()
        
        # Return empty string if too short
        if len(qt) < 2:
            return ""
            
        return qt

    def trf_agent_match(self, text, country="", method="cosine", threshold=THRESHOLD_COSINE_SIMILARITY):
        """
        Compare input text to the agent file using sentence transformer embeddings.
        
        Parameters
        ----------
        text : str
            Text to match against agent patterns
        country : str, optional
            Country code to include in the result
        method : str, optional
            Similarity method to use ('cosine' or 'dot')
        threshold : float, optional
            Similarity threshold below which matches are ignored
            
        Returns
        -------
        dict or None
            Match information or None if no match above threshold
        """
        # Validate parameters
        if method not in ['cosine', 'dot']:
            raise ValueError("distance method must be one of ['cosine', 'dot']")
            
        # Adjust threshold based on method
        if method == "dot" and threshold < 1:
            threshold = THRESHOLD_DOT_SIMILARITY
            logger.info(f"Threshold is too low for dot product. Setting to {threshold}")
        if method == "cosine" and threshold > 2:
            threshold = 0.1
            logger.info(f"Threshold is too high for cosine. Setting to {threshold}")
            
        # Handle empty inputs
        if country is None:
            country = ""
        text = self.clean_query(text)
        if not text:
            return None
            
        # Compute similarity
        query_trf = self.trf.encode(text, show_progress_bar=False)
        if method == "dot":
            sims = np.dot(self.trf_matrix, query_trf.T)
        else:  # cosine
            sims = 1 - cdist(self.trf_matrix, np.expand_dims(query_trf.T, 0), metric="cosine")
            
        # Get best match
        best_idx = np.argmax(sims)
        max_sim = np.max(sims)
        
        # Return None if below threshold
        if max_sim < threshold:
            best_pattern = self.agents[best_idx]['pattern']
            logger.debug(f"Agent comparison. Closest result for '{text}' is '{best_pattern}' with confidence {max_sim}")
            return None
            
        # Create match object
        match = self.agents[best_idx].copy()
        match['country'] = country
        match['description'] = match['pattern']
        match['query'] = text
        match['conf'] = max_sim
        
        logger.debug(f"Match from trf_agent_match: {match}")
        return match

    def strip_ents(self, doc):
        """
        Strip out named entities from text, leaving only non-entity tokens.
        
        Parameters
        ----------
        doc : spaCy Doc
            Document to process
            
        Returns
        -------
        str
            Text with named entities removed
        """
        skip_list = ['a', 'and', 'the', "'s", "'", "s"]
        non_ent_tokens = [
            token.text_with_ws for token in doc 
            if token.ent_type_ == "" and token.text.lower() not in skip_list
        ]
        return ''.join(non_ent_tokens).strip()
    
    def get_noun_phrases(self, doc):
        """
        Extract non-entity noun phrases from a document.
        
        Parameters
        ----------
        doc : spaCy Doc
            Document to process
            
        Returns
        -------
        str
            Space-joined noun phrases
        """
        skip_list = ['a', 'and', 'the']
        skip_ent_types = ['CARDINAL', 'DATE', 'ORDINAL']
        
        # Get noun chunks that don't end with an entity
        noun_phrases = [chunk for chunk in doc.noun_chunks if chunk[-1].ent_type_ == ""]
        
        # Collect tokens from those chunks, skipping certain words and entity types
        phrase_tokens = []
        for chunk in noun_phrases:
            for token in chunk:
                if token.text not in skip_list and token.ent_type_ not in skip_ent_types:
                    phrase_tokens.append(token.text_with_ws.lower())
                    
        return ''.join(phrase_tokens).strip()

    def get_noun_phrases_list(self, doc):
        """
        Get a list of non-entity noun phrases from a document.
        
        Parameters
        ----------
        doc : spaCy Doc
            Document to process
            
        Returns
        -------
        list
            List of noun phrases
        """
        return [chunk for chunk in doc.noun_chunks if chunk[-1].ent_type_ == ""]

    def short_text_to_agent(self, text, strip_ents=False, threshold=THRESHOLD_COSINE_SIMILARITY):
        """
        Convert short text to an agent code, optionally stripping entities first.
        
        Parameters
        ----------
        text : str
            Text to convert
        strip_ents : bool, optional
            Whether to strip named entities before matching
        threshold : float, optional
            Similarity threshold for matching
            
        Returns
        -------
        dict or None
            Agent code information or None if no match
        """
        # Extract country and clean text
        country, trimmed_text = self.search_nat(text)
        trimmed_text = self.clean_query(trimmed_text)
        
        # Optionally strip entities
        if strip_ents:
            try:
                doc = self.nlp(text)
                trimmed_text = self.strip_ents(doc)
            except IndexError:
                # If NLP fails, continue with trimmed_text as-is
                pass
                
            if trimmed_text == "s":
                return None
                
        # Match against agent patterns
        return self.trf_agent_match(trimmed_text, country=country, threshold=threshold)

    def search_nat(self, text, method="longest", categories=False):
        """
        Search for country names/nationalities in text and return canonical form.
        
        Parameters
        ----------
        text : str
            Text to search for country mentions
        method : str, optional
            Method to use when multiple countries are found ('longest' or 'first')
        categories : bool, optional
            Whether to use category patterns (of X, in X)
            
        Returns
        -------
        tuple
            (country_code, trimmed_text) or (None, original_text) if no country found
        """
        if not text:
            return None, text
            
        # Normalize text for consistent matching
        text = unidecode.unidecode(text)
        found = []
        
        # Use appropriate pattern list based on categories flag
        patterns = self.nat_list_cat if categories else self.nat_list
        
        # Find all matching countries
        for pattern, country_code in patterns:
            match = re.search(pattern, text)
            if match:
                # Remove the matched country/nationality from text
                trimmed_text = re.sub(pattern, "", text).strip()
                trimmed_text = self.clean_query(trimmed_text)
                found.append((country_code, trimmed_text.strip(), match))
        
        # Return if no countries found
        if not found:
            return None, text
            
        # Return based on requested method
        if method == "longest":
            # Return the longest match to handle e.g. "Saudi", "Britain"
            found.sort(key=lambda x: len(x[1]))
            return found[0][0:2]
        elif method == "first":
            # Return the first occurrence in the text
            found.sort(key=lambda x: x[2].span()[0])
            return found[0][0:2]
        else:
            valid_methods = "['longest', 'first']"
            raise ValueError(f"search_nat sorting option must be one of {valid_methods}. You gave {method}")

    def _parse_office_term_dates(self, infobox, office_key, num=""):
        """
        Parse term start and end dates for an office from an infobox.
        
        Parameters
        ----------
        infobox : dict
            Wikipedia infobox data
        office_key : str
            Key for the office in the infobox
        num : str, optional
            Number suffix for additional offices
            
        Returns
        -------
        tuple
            (term_start, term_end) as datetime objects or None
        """
        # Try to get term end date
        term_end = None
        try:
            term_end = dateparser.parse(infobox[f"term_end{num}"])
        except KeyError:
            try:
                # Sometimes no underscore is used
                term_end = dateparser.parse(infobox[f"termend{num}"])
            except KeyError:
                pass
        
        # Try to get term start date
        term_start = None
        try:
            term_start = dateparser.parse(infobox[f"term_start{num}"])
        except KeyError:
            try:
                # Sometimes no underscore is used
                term_start = dateparser.parse(infobox[f"termstart{num}"])
            except KeyError:
                pass
                
        return term_start, term_end

    def parse_offices(self, infobox):
        """
        Extract office information from a Wikipedia infobox.
        
        Parameters
        ----------
        infobox : dict
            Wikipedia infobox data
            
        Returns
        -------
        list
            List of office information dictionaries
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
        
        Parameters
        ----------
        offices : list
            List of office dictionaries from parse_offices
        query_date : str or datetime
            Date to check against
            
        Returns
        -------
        tuple
            (active_offices, detected_countries)
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
            country, _ = self.search_nat(office['office'])
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

    def search_wiki(self, query_term, limit_term="", fuzziness="AUTO", max_results=200,
                   fields=['title^50', 'redirects^50', 'alternative_names'],
                   score_type="best_fields"):
        """
        Search Wikipedia for a given query term.
        
        Parameters
        ----------
        query_term : str
            Term to search for
        limit_term : str, optional
            Term to limit results by
        fuzziness : str, optional
            Elasticsearch fuzziness parameter
        max_results : int, optional
            Maximum number of results to return
        fields : list, optional
            Fields to search in
        score_type : str, optional
            Elasticsearch score type
            
        Returns
        -------
        list
            List of Wikipedia article dictionaries
        """
        # Clean query term
        query_term = self.clean_query(query_term)
        logger.debug(f"Using query term: '{query_term}'")
        
        # Construct query
        if not limit_term:
            #query = {
            #    "multi_match": {
            #        "query": query_term,
            #        "fields": fields,
            #        "type": score_type,
            #        "fuzziness": fuzziness,
            #        "operator": "and"
            #    }
            #}
            query = {
                "bool": {
                    "should": [
                        # Exact match on title (case-sensitive)
                        {"term": {"title": {"value": query_term, "boost": 50}}},

                        # Analyzed match on title (case-insensitive, tokenized)
                        {"match": {"title": {"query": query_term, "boost": 30}}},

                        # Exact match on redirects (for acronyms)
                        {"term": {"redirects": {"value": query_term, "boost": 150}}},

                        # Analyzed match on redirects
                        {"match": {"redirects": {"query": query_term, "boost": 50}}},

                        # Analyzed match on alternative names
                        {"match": {"alternative_names": {"query": query_term, "boost": 25}}}
                    ]
                }
            }
        else:
            # Include limit term in query
            limit_fields = [
                "title^100", "redirects^100", "alternative_names",
                "intro_para", "categories", "infobox"
            ]
            query = {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_term,
                                "fields": fields,
                                "type": score_type,
                                "fuzziness": fuzziness,
                                "operator": "and"
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
        logger.debug(f"Number of hits for Wiki query: {len(results)}")
        # log the titles of the results
        logger.debug(f"Titles of the results: {[result['title'] for result in results]}")
        
        return results

    def text_ranker_features(self, matches, fields):
        """
        Extract and combine text from specified fields in Wiki matches.
        
        Parameters
        ----------
        matches : list
            List of Wikipedia match dictionaries
        fields : list
            List of fields to extract
            
        Returns
        -------
        list
            List of combined text strings
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
        
        Filters out disambiguation pages, stub articles, and other non-useful pages.
        
        Parameters
        ----------
        results : list
            List of Wikipedia article dictionaries
            
        Returns
        -------
        list
            Filtered list of articles
        """
        # Early return if no results
        if not results:
            return []
            
        # Filter out articles without intro paragraph
        good_res = [r for r in results if 'intro_para' in r and r['intro_para']]
        
        # Filter out disambiguation and stub pages
        patterns_to_exclude = [
            (r"(stub|User|Wikipedia\:)", 'title'),
            (r"disambiguation", 'title'),
            (r"Category\:", lambda r: r['intro_para'][0:50]),
            (r"is the name of", lambda r: r['intro_para'][0:50]),
            (r"may refer to", lambda r: r['intro_para'][0:50]),
            (r"can refer to", lambda r: r['intro_para'][0:50]),
            (r"most commonly refers to", lambda r: r['intro_para'][0:50]),
            (r"usually refers to", lambda r: r['intro_para'][0:80]),
            (r"is a surname", lambda r: r['intro_para'][0:50])
        ]
        
        # Apply each exclusion pattern
        for pattern, field_getter in patterns_to_exclude:
            if callable(field_getter):
                good_res = [r for r in good_res if not re.search(pattern, field_getter(r))]
            else:
                good_res = [r for r in good_res if field_getter not in r or not re.search(pattern, r[field_getter])]
        
        # Filter out articles with short intro paragraphs
        good_res = [r for r in good_res if len(r['intro_para']) > 50 and r['intro_para'].strip()]
        
        return good_res

    def _find_exact_title_matches(self, query_term, results, country=None):
        """
        Find exact matches between query term and Wikipedia article titles.
        
        Parameters
        ----------
        query_term : str
            Query term to match
        results : list
            List of Wikipedia article dictionaries
        country : str, optional
            Country to include in matching
            
        Returns
        -------
        list
            List of matching articles
        """
        query_country = f"{query_term} ({country})" if country else query_term
        exact_matches = []

        
        for result in results:
            # TODO: construct two sets, one for query and one for title
            # and check intersection. E.g.:
            #query_expanded = [query_term, query_country, query_term.upper(), query_country.upper(), 
            #                  query_term.title(), query_country.title(), remove_accents(query_term), 
            #                  remove_accents(query_country)]
            #title_expanded = [result['title'], result['title'].upper(), result['title'].title(),
            #                    remove_accents(result['title'])]
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
        
        Parameters
        ----------
        query_term : str
            Query term to match
        results : list
            List of Wikipedia article dictionaries
        country : str, optional
            Country to include in matching
            
        Returns
        -------
        list
            List of matching articles
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
        
        Parameters
        ----------
        query_term : str
            Query term to match
        results : list
            List of Wikipedia article dictionaries
            
        Returns
        -------
        list
            List of matching articles
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

    def _rank_by_neural_similarity(self, candidates, query_term, context=None, fields=None):
        """
        Rank candidates by neural similarity to query term and/or context.
        
        Parameters
        ----------
        candidates : list
            List of candidate articles
        query_term : str
            Query term for comparison
        context : str, optional
            Context text for comparison
        fields : list, optional
            Fields to use for comparison
            
        Returns
        -------
        list
            Ranked list of candidates
        """
        if not candidates:
            return []
            
        if not fields:
            fields = ['title', 'categories', 'alternative_names', 'redirects']
            
        # Use neural similarity with context if provided
        if context:
            intro_paras = [c['intro_para'][0:200] for c in candidates[0:50]]
            logger.debug("Top 30 intro paras: " + str(intro_paras))
            encoded_intros = self.trf.encode(intro_paras, show_progress_bar=False, device=self.device)
            encoded_text = self.trf.encode(context, show_progress_bar=False, device=self.device)
            
            # Get similarity scores
            sims = cos_sim(encoded_text, encoded_intros)[0]
            logger.debug("Similarity scores: " + str(sims))
            
            # Sort candidates by similarity
            similarities = [(s, c) for s, c in zip(sims, candidates)]
            similarities.sort(reverse=True)
            
            best_score = similarities[0][0] if similarities else 0
            if best_score > THRESHOLD_CONTEXT_MATCH:
                best_candidate = similarities[0][1]
                best_candidate['wiki_reason'] = f"Neural similarity to context. Score = {best_score}"
                return [best_candidate]
        
        # If no context or low similarity, use title similarity
        try:
            wiki_info = self.text_ranker_features(candidates, fields)
            category_trf = self.trf.encode(wiki_info, show_progress_bar=False, device=self.device)
            query_trf = self.trf.encode(query_term, show_progress_bar=False, device=self.device)
            
            sims = 1 - cdist(category_trf, np.expand_dims(query_trf.T, 0), metric="cosine")
            ranked_candidates = [x for _, x in sorted(zip(sims.flatten(), candidates), reverse=True)]
            
            if ranked_candidates and max(sims) > THRESHOLD_NEURAL_TITLE_MATCH:
                ranked_candidates[0]['wiki_reason'] = f"Neural similarity to query. Score = {sims[0][0]}"
                return ranked_candidates[0]
        except Exception as e:
            logger.debug(f"Error in neural similarity ranking: {e}")
            logger.debug(f"Query term: {query_term}")
            return candidates

    def _check_titles_similarity(self, query_term, candidates):
        """
        Check similarity between query term and candidate titles using actor_sim model.
        
        Parameters
        ----------
        query_term : str
            Query term to match
        candidates : list
            List of candidate articles
            
        Returns
        -------
        tuple
            (best_match, similarity_score) or (None, 0)
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

    def pick_best_wiki(self, query_term, results, context="", country="",
                      wiki_sort_method="neural", rank_fields=None):
        """
        Select the best Wikipedia article from search results.
        
        This method applies various matching strategies in sequence to find the best match.
        
        Parameters
        ----------
        query_term : str
            Query term to match
        results : list
            List of Wikipedia article search results
        context : str, optional
            Context text to help with disambiguation
        country : str, optional
            Country code to help with disambiguation
        wiki_sort_method : str, optional
            Method to use for sorting results
        rank_fields : list, optional
            Fields to use for ranking
            
        Returns
        -------
        dict or None
            Best matching Wikipedia article or None if no good match
        """
        # Set default rank fields if not provided
        if rank_fields is None:
            rank_fields = ['title', 'categories', 'alternative_names', 'redirects']
            
        # Clean query term
        query_term = self.clean_query(query_term)
        logger.debug(f"Using query term '{query_term}'")
        
        # Construct country-qualified query if country provided
        query_country = f"{query_term} ({country})" if country else query_term
        
        # Handle empty results
        if not results:
            logger.debug("No Wikipedia results. Returning None")
            return None
            
        # Remove disambiguation pages, stubs, etc.
        good_res = self._trim_results(results)
        if not good_res:
            return None
            
        logger.debug(f"Filtered down to {len(good_res)} valid results")
        
        # Find exact title matches
        exact_matches = self._find_exact_title_matches(query_term, good_res, country)
        logger.debug(f"Found {len(exact_matches)} exact title matches")
        
        # Find redirect matches
        redirect_matches = self._find_redirect_matches(query_term, good_res, country)
        logger.debug(f"Found {len(redirect_matches)} redirect matches")
        
        # Find alternative name matches
        alt_matches = self._find_alt_name_matches(query_term, good_res)
        logger.debug(f"Found {len(alt_matches)} alternative name matches")
        
        # Handle single exact title match with no redirects
        if len(exact_matches) == 1 and not redirect_matches:
            best = exact_matches[0]
            best['wiki_reason'] = "Only one exact title match and no redirects"
            return best
        
            
        # Handle multiple exact matches
        if exact_matches:
            if wiki_sort_method == "alt_names":
                # Sort by number of alternative names
                exact_matches.sort(key=lambda x: -len(x.get('alternative_names', [])))
                if exact_matches:
                    best = exact_matches[0]
                    best['wiki_reason'] = "Multiple title exact matches: using longest alt names"
                    return best
            elif wiki_sort_method in ["neural", "lcs"]:
                # Try context-based similarity first
                if context:
                    combined_matches = exact_matches + redirect_matches
                    ranked = self._rank_by_neural_similarity(combined_matches, query_term, context)
                    if ranked and 'wiki_reason' in ranked[0]:
                        return ranked[0]
                
                # Fall back to query-based similarity
                ranked = self._rank_by_neural_similarity(exact_matches, query_term, fields=rank_fields)
                if ranked and len(ranked) > 0:
                    return ranked[0]
        
        # Handle single redirect match
        if len(redirect_matches) == 1:
            best = redirect_matches[0]
            best['wiki_reason'] = "Single redirect exact match"
            return best
            
        # Handle two redirect matches - pick one with longer intro
        if len(redirect_matches) == 2:
            if len(redirect_matches[0]['intro_para']) > len(redirect_matches[1]['intro_para']):
                best = redirect_matches[0]
            else:
                best = redirect_matches[1]
            best['wiki_reason'] = "Two exact redirect matches, returning page with longer intro"
            return best
            
        # Handle multiple redirect matches
        if redirect_matches:
            logger.debug(f"Multiple redirect matches. Using {wiki_sort_method} to select best.")
            
            if wiki_sort_method == "alt_names":
                redirect_matches.sort(key=lambda x: -len(x.get('alternative_names', [])))
                best = redirect_matches[0]
                best['wiki_reason'] = "Multiple redirect matches; picking by longest alt names"
                return best
            elif wiki_sort_method in ["neural", "lcs"]:
                # Try context-based similarity first
                if context:
                    ranked = self._rank_by_neural_similarity(redirect_matches, query_term, context)
                    if ranked and 'wiki_reason' in ranked[0]:
                        return ranked[0]
                
                # Fall back to query-based similarity
                query_context = query_term + (context or "")
                ranked = self._rank_by_neural_similarity(redirect_matches, query_context, fields=rank_fields)
                if ranked and len(ranked) > 0:
                    return ranked[0]
        
        # Handle single alternative name match
        #if len(alt_matches) == 1:
        #    best = alt_matches[0]
        #    best['wiki_reason'] = "Single alt name match"
        #    return best
            
        # Fall back to neural similarity between context and intro paragraphs
        if context:
            logger.debug("Falling back to text-intro neural similarity")
            ranked = self._rank_by_neural_similarity(good_res[0:50], query_term, context)
            if ranked and 'wiki_reason' in ranked[0]:
                return ranked[0]
                
        # Last resort: title similarity
        logger.debug("Falling back to title neural similarity")
        best_match, sim_score = self._check_titles_similarity(query_term, good_res)
        if best_match:
            return best_match
            
        # Check alt matches with title similarity
        if alt_matches:
            logger.debug("Checking alt name matches with title similarity")
            best_match, sim_score = self._check_titles_similarity(query_term, alt_matches)
            if best_match:
                return best_match
                
        logger.debug("No good match found")
        return None

    def query_wiki(self, query_term, limit_term="", country="", context="", max_results=200):
        """
        Search Wikipedia and return the best matching article.
        
        This method tries an exact search first, then falls back to fuzzy search.
        
        Parameters
        ----------
        query_term : str
            Term to search for
        limit_term : str, optional
            Term to limit results by
        country : str, optional
            Country code to help with disambiguation
        context : str, optional
            Context text to help with disambiguation
        max_results : int, optional
            Maximum results to return from search
            
        Returns
        -------
        dict or None
            Best matching Wikipedia article or None if no good match
        """
        # Do NER expansion by default
        if context:
            doc = self.nlp(context)
            if doc:
                expanded_query = [i.text for i in doc.ents if query_term in i.text]
                if expanded_query:
                    # take the longest match
                    expanded_query.sort(key=len, reverse=True)
                    # take the first one
                    expanded_query = expanded_query[0]
                    if len(expanded_query) > len(query_term):
                        query_term = expanded_query
                        logger.debug(f"Using NER to expand context: {expanded_query}")

        # Try exact search first
        results = self.search_wiki(
            query_term, 
            limit_term=limit_term, 
            fuzziness=0, 
            max_results=max_results,
        )
        best = self.pick_best_wiki(
            query_term, 
            results, 
            country=country, 
            context=context
        )
        if best:
            return best
            
        # Fall back to fuzzy search
        results = self.search_wiki(
            query_term, 
            limit_term=limit_term, 
            fuzziness=1, 
            max_results=max_results
        )
        best = self.pick_best_wiki(
            query_term, 
            results, 
            country=country, 
            context=context
        )
       
        return best

    def _process_wiki_short_description(self, wiki, countries):
        """
        Process the short description from a Wikipedia article to extract actor code.
        
        Parameters
        ----------
        wiki : dict
            Wikipedia article
        countries : list
            List to append detected countries to
            
        Returns
        -------
        list
            List containing SD code if found, otherwise empty list
        """
        if 'short_desc' not in wiki:
            return []
            
        # Extract country and text
        country, trimmed_text = self.search_nat(wiki['short_desc'])
        if country:
            countries.append(country)
            
        # Match text to agent pattern
        sd_code = self.trf_agent_match(trimmed_text, country=country)
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
        
        Parameters
        ----------
        wiki : dict
            Wikipedia article
        query_date : str or datetime
            Date to use for determining current offices
        countries : list
            List to append detected countries to
        office_countries : list
            List to append countries detected from offices to
            
        Returns
        -------
        tuple
            (box_codes, box_type_code, type_code)
        """
        if 'infobox' not in wiki:
            return [], None, None
            
        infobox = wiki['infobox']
        box_codes = []
        box_type_code = None
        type_code = None
        
        # Extract country from infobox
        if 'country' in infobox:
            country = self.search_nat(infobox['country'])[0]
            if country:
                countries.append(country)
                
        # Get box type code
        if 'box_type' in wiki:
            box_type_code = self.trf_agent_match(wiki['box_type'])
            if box_type_code:
                box_type_code['country'] = countries[0] if countries else None
                box_type_code['wiki'] = wiki['title']
                box_type_code['source'] = "Infobox Title"
                box_type_code['actor_wiki_job'] = wiki['box_type']
                
        # Get type code from infobox
        if 'type' in infobox:
            type_code = self.trf_agent_match(infobox['type'])
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
                office_text = self.clean_query(office['office'])
                code = self.short_text_to_agent(office_text)
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
        
        Parameters
        ----------
        offices : list
            List of office dictionaries
        wiki : dict
            Wikipedia article
        countries : list
            List of detected countries
            
        Returns
        -------
        list
            List of ELI codes if applicable
        """
        # Get codes from past offices
        old_codes_raw = [self.trf_agent_match(self.clean_query(o['office'])) for o in offices]
        old_codes = []
        for code in old_codes_raw:
            if not code or 'code_1' not in code:
                continue
            old_codes.append(code['code_1'])
            
        # Get countries from past offices
        old_countries = [self.search_nat(o['office'])[0] for o in offices]
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
        
        Parameters
        ----------
        wiki : dict
            Wikipedia article
        cat_countries : list
            List to append detected countries to
        """
        if 'categories' not in wiki:
            return
            
        for category in wiki['categories']:
            country, _ = self.search_nat(category, categories=True)
            if country:
                cat_countries.append(country)

    def wiki_to_code(self, wiki, query_date="today", country=""):
        """
        Convert a Wikipedia article to a PLOVER actor code.
        
        Parameters
        ----------
        wiki : dict
            Wikipedia article
        query_date : str or datetime, optional
            Date to use for determining current offices
        country : str, optional
            Country code if already known
            
        Returns
        -------
        list
            List of actor codes derived from the article
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
            first_sent_country, _ = self.search_nat(first_sent, method="first")
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

    def _get_actor_priority(self, code_1):
        """
        Get priority value for an actor code.
        
        Parameters
        ----------
        code_1 : str
            Actor code
            
        Returns
        -------
        int
            Priority value
        """
        return self.ACTOR_TYPE_PRIORITIES.get(code_1, 0)

    def pick_best_code(self, all_codes, country):
        """
        Select the best actor code from a list of candidates.
        
        Parameters
        ----------
        all_codes : list
            List of actor code dictionaries
        country : str, optional
            Country code if already known
            
        Returns
        -------
        dict or None
            Best actor code or None if no valid code
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
                
        # 4. Check for short description
        short_desc_codes = [c for c in code_sources if c['source'] == "Wiki short description"]
        if short_desc_codes:
            best = short_desc_codes[0]
            if best['code_1'] == "IGO":
                best['country'] = "IGO"
            else:
                best['country'] = best_country
            best['best_reason'] = "Picking Wiki short description"
            return best
            
        # 5. Check for pre-wiki lookup codes
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
        
        Parameters
        ----------
        best : dict
            Actor code dictionary
            
        Returns
        -------
        dict or None
            Cleaned actor code or None if input was None
        """
        if not best:
            return None
            
        # Handle special codes
        if best.get('code_1') == 'JNK':
            best['country'] = ""
            
        if best.get('code_1') == 'NON':
            best['country'] = ""
            
        # Handle actor types that should be treated as countries
        if best.get('code_1') in self.SPECIAL_ACTOR_TYPES:
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

    def agent_to_code(self, text, context="", query_date="today", known_country="", search_limit_term=""):
        """
        Resolve an actor mention to a code representing their role.
        
        Parameters
        ----------
        text : str
            Text mention of the actor to resolve
        context : str, optional
            Additional context to help with disambiguation
        query_date : str, optional
            Date to use when determining current offices
        known_country : str, optional
            Country code if already known
        search_limit_term : str, optional
            Term to limit Wikipedia search results
            
        Returns
        -------
        dict or None
            Actor code information or None if resolution fails
        """
        # Check cache first
        cache_key = text + "_" + str(query_date)
        if cache_key in self.cache:
            logger.debug("Returning from cache")
            return self.cache[cache_key]
            
        # Extract country from text
        country, trimmed_text = self.search_nat(text)
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
            self.cache[cache_key] = code_full_text
            return self.clean_best(code_full_text)
            
        # Parse entities in text
        try:
            doc = self.nlp(trimmed_text)
            non_ent_text = self.strip_ents(doc)
            ents = [i for i in doc.ents if i.label_ in ['EVENT', 'FAC', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON']]
            token_level_ents = [i.ent_type_ for i in doc]
            ent_text = ''.join([i.text_with_ws for i in doc if i.ent_type_ != ""])
            logger.debug(f"Found named entities: {ents}")
        except IndexError:
            # Usually caused by a mismatch between token and embedding
            logger.info(f"Token alignment error on {trimmed_text}")
            non_ent_text = trimmed_text
            token_level_ents = ['']
            ent_text = ""
            ents = []
            
        # Try direct matching first
        if trimmed_text:
            logger.debug(f"Trying direct matching on: {trimmed_text}")
            code_full_text = self.trf_agent_match(trimmed_text, country=country)
            
            if code_full_text:
                logger.debug(f"Direct match found: {code_full_text}")
                code_full_text['source'] = "BERT matching full text"
                code_full_text['wiki'] = ""
                code_full_text['actor_wiki_job'] = ""
                
                # Return without Wikipedia lookup in certain high-confidence cases
                if (code_full_text['conf'] > 0.6 and not ents or
                    code_full_text['conf'] > THRESHOLD_HIGH_CONFIDENCE and trimmed_text == trimmed_text.lower() or
                    code_full_text['conf'] > THRESHOLD_VERY_HIGH_CONFIDENCE):
                    logger.debug("High confidence match. Skipping Wikipedia lookup.")
                    code_full_text = self.clean_best(code_full_text)
                    self.cache[cache_key] = code_full_text
                    return code_full_text
            else:
                logger.debug(f"No direct match found for {trimmed_text}")
                
        # Try Wikipedia lookup for better resolution
        logger.debug(f"Trying Wikipedia lookup with: {trimmed_text}")
        wiki_codes = []
        wiki = self.query_wiki(
            query_term=trimmed_text, 
            country=known_country, 
            context=context,
            limit_term=search_limit_term
        )
        
        if wiki:
            logger.debug(f"Wikipedia page found: {wiki['title']}")
            wiki_codes = self.wiki_to_code(wiki, query_date)
        elif ent_text:
            # Try again with just entity text if original lookup failed
            logger.debug(f"No wiki results. Trying with entity text: {ent_text}")
            wiki = self.query_wiki(
                query_term=ent_text, 
                country=known_country, 
                context=context,
                limit_term=search_limit_term
            )
            if wiki:
                wiki_codes = self.wiki_to_code(wiki, query_date)
                
        # Combine all possible codes
        code_full_text_list = [code_full_text] if 'code_full_text' in locals() and code_full_text else []
        all_codes = wiki_codes + code_full_text_list
        all_codes = [c for c in all_codes if c]
        
        logger.debug("--- ALL CODES ----")
        logger.debug(all_codes)
        
        # Extract unique codes for reference
        unique_code1s = list(set([c['code_1'] for c in all_codes if c.get('code_1')]))
        unique_code1s = [c for c in unique_code1s if c not in ["IGO"]]
        unique_code2s = list(set([c['code_2'] for c in all_codes if c.get('code_2')]))
        
        # Pick the best code
        best = self.pick_best_code(all_codes, country)
        best = self.clean_best(best)
        
        # Add unique codes lists for reference
        if best:
            best['all_code1s'] = unique_code1s
            best['all_code2s'] = unique_code2s
            
        # Cache and return result
        self.cache[cache_key] = best
        return best

    def process(self, event_list):
        """
        Process a list of events to resolve actor attributes.
        
        For each event, adds actor resolution information to the ACTOR and RECIP attributes.
        
        Parameters
        ----------
        event_list : list
            List of event dictionaries
            
        Returns
        -------
        list
            The same event list with actor resolution information added
        """
        for event in track(event_list, description="Resolving actors..."):
            # Get the date from the event
            query_date = event.get('pub_date', "today")
            
            # Process each attribute block
            for attr_type, attr_block in event['attributes'].items():
                # Skip location and date attributes
                if attr_type in ["LOC", "DATE"]:
                    continue
                    
                # Process each attribute value
                for attr in attr_block:
                    # Get actor text
                    actor_text = attr['text']
                    if not isinstance(actor_text, str):
                        logger.warning(f"Non-string actor text: {actor_text}")
                        actor_text = actor_text[0]
                        
                    # Resolve actor to code
                    res = self.agent_to_code(actor_text, query_date=query_date)
                    
                    # Update attribute with resolution information
                    if res:
                        # Add wiki information
                        attr['wiki'] = res.get('wiki', "")
                        attr['actor_wiki_job'] = res.get('actor_wiki_job', "")
                        
                        # Add code lists
                        attr['all_code1s'] = res.get('all_code1s', [])
                        attr['all_code2s'] = res.get('all_code2s', [])
                        
                        # Add country and codes
                        attr['country'] = res.get('country', "")
                        attr['code_1'] = res.get('code_1', "")
                        attr['code_2'] = res.get('code_2', "")
                        
                        # Add query and pattern information
                        attr['actor_role_query'] = res.get('query', "")
                        attr['actor_resolved_pattern'] = res.get('description', "")
                        
                        # Add confidence and reason
                        attr['actor_pattern_conf'] = float(res.get('conf', 0))
                        attr['actor_resolution_reason'] = res.get('best_reason', "")
                    else:
                        # Set default values if resolution failed
                        attr['wiki'] = ""
                        attr['actor_wiki_job'] = ""
                        attr['country'] = ""
                        attr['code_1'] = ""
                        attr['code_2'] = ""
                        attr['actor_role_query'] = ""
                        attr['actor_resolved_pattern'] = ""
                        attr['actor_pattern_conf'] = ""
                        attr['actor_resolution_reason'] = ""

        # Save intermediate results if requested
        if self.save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_actor_resolution_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)
                
        return event_list



if __name__ == "__main__":
    import jsonlines

    ag = ActorResolver()
    with jsonlines.open("PLOVER_coding_201908_with_attr.jsonl", "r") as f:
        data = list(f.iter())

    out = ag.process(data)
    with jsonlines.open("PLOVER_coding_201908_with_actor.jsonl", "w") as f:
        f.write_all(out)

    """
    {'id': '20190801-2309-4e081644904c_COOPERATE_R',
 'date': '2019-08-01',
 'event_type': 'R',
 'event_mode': [],
 'event_text': 'Delegates of the Venezuelan president, Nicolas Maduro, and the leader objector Juan Guaidó resumed on Wednesday (31) conversations on the island of Barbados, sponsored by Norway, to seek a way out of the crisis in their country, announced the parties. "We started another round of sanctions under the mechanism of Oslo," indicated on Twitter Mr Stalin González, one of the envoys of Guaidó, parliamentary leader recognized as interim president by half hundred countries. The vice-president of Venezuela, Delcy Rodríguez, confirmed in a press conference that representatives of mature traveled to Barbados for the meetings with the opposition. Mature reaffirmed in a message to the nation that the government seeks to establish a "bureau for permanent dialog with the opposition, and called entrepreneurs and social movements to be added to the process. After exploratory approximations and a first face to face in Oslo in mid-May, the parties have transferred the dialog on 8 July for the caribbean island. The opposition search in the negotiations the output of mature and a new election, by considering that his second term, started last January, resulted from fraudulent elections, not recognized by almost 60 countries, among them the United States. ',
 'story_id': 'AFPPT00020190801ef81000jh:50066619',
 'publisher': 'translateme2-pt',
 'headline': '\nGoverno e oposição da Venezuela retomam diálogo em Barbados\n',
 'pub_date': '2019-08-01',
 'contexts': ['pro_democracy'],
 'version': 'NGEC_coder-Vers001-b1-Run-001',
 'attributes': {'ACTOR': {'text': 'Nicolas Maduro',
   'score': 0.23675884306430817,
   'wiki': 'Nicolás Maduro',
   'country': 'VEN',
   'code_1': 'ELI',
   'code_2': ''},
  'RECIP': {'text': 'Juan Guaidó',
   'score': 0.13248120248317719,
   'wiki': 'Juan Guaidó',
   'country': 'VEN',
   'code_1': 'REB',
   'code_2': ''},
  'LOC': {'text': 'Barbados', 'score': 0.4741457998752594}}}
    """ 