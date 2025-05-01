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
DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v3"
DEFAULT_SIM_MODEL_PATH = 'actor_sim_model2'
DEFAULT_BASE_PATH = "./assets"

# Threshold constants
THRESHOLD_COSINE_SIMILARITY = 0.8
THRESHOLD_DOT_SIMILARITY = 45
THRESHOLD_NEURAL_TITLE_MATCH = 0.9
THRESHOLD_ALT_NAME_TITLE_MATCH = 0.8
THRESHOLD_CONTEXT_MATCH = 0.6 # 0.7
THRESHOLD_HIGH_CONFIDENCE = 0.90
THRESHOLD_VERY_HIGH_CONFIDENCE = 0.95
THRESHOLD_COMBINED_SCORE = 9 

#######################################################
# Text Processing Utilities
#######################################################

class TextPreProcessor:
    """
    Utilities for cleaning and normalizing text.
    
    This class provides methods for text cleaning, entity extraction,
    and noun phrase identification.
    
    Example:
        processor = TextPreProcessor()
        clean_text = processor.clean_query("The United States Government")
        # Returns: "united states government"
    """
    
    def clean_query(self, qt):
        """
        Clean and normalize a query string.
        
        Removes articles, ordinals, possessives, and other noise from text.
        
        Args:
            qt: Text to clean
            
        Returns:
            str: Cleaned text
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
    
    def extract_entity_components(self, span_text, nlp, job_titles=None, job_title_embeddings=None, get_embedding_func=None):
        """
        Extracts core entity, role, and geographic information from a text span.

        Args:
            span_text: String containing the entity span
            job_titles: List of known job titles/roles (optional)
            job_title_embeddings: Dict mapping job titles to embeddings (optional)
            get_embedding_func: Function to get embedding for a new text (optional)

        Returns:
            Dict with core_entity, role, and geographic_info
        """
        doc = nlp(span_text)

        # Initialize results
        results = {
            'core_entity': None,
            'role': None,
            'geographic_info': None
        }

        # Step 1: Extract entities by type
        person_entities = []
        org_entities = []
        geo_entities = []

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                person_entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            elif ent.label_ == 'ORG':
                org_entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            elif ent.label_ in ['GPE', 'LOC', 'FAC', 'NORP']:
                geo_entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        # Step 1.5: Custom entity detection for abbreviations and special cases
        # Look for uppercase words that could be organization acronyms
        acronym_pattern = re.compile(r'\b([A-Z]{2,})\b')
        for match in acronym_pattern.finditer(span_text):
            acronym = match.group(1)
            # Check if it's not already detected
            already_detected = False
            for org in org_entities:
                if org['text'] == acronym:
                    already_detected = True
                    break
                
            if not already_detected:
                org_entities.append({
                    'text': acronym,
                    'start': match.start(),
                    'end': match.end()
                })

        # Step 2: Set geographic information
        if geo_entities:
            results['geographic_info'] = geo_entities[0]['text']

        # Step 3: Set core entity (prioritize PERSON over ORG)
        if person_entities:
            results['core_entity'] = person_entities[0]['text']
        elif org_entities:
            results['core_entity'] = org_entities[0]['text']

        # Step 4: Handle possessive patterns specially
        possessive_pattern = re.compile(r"([A-Za-z']+)['']s\s+([A-Za-z]+)")
        possessive_match = possessive_pattern.search(span_text)

        if possessive_match:
            possessor = possessive_match.group(1)
            possessed = possessive_match.group(2)

            # Check if possessor is a geo entity
            if results['geographic_info'] and possessor == results['geographic_info']:
                # Check if possessed is an org (or potential acronym)
                matches_org = False
                for org in org_entities:
                    if possessed in org['text']:
                        results['core_entity'] = org['text']
                        matches_org = True
                        break
                    
                # If not matched, check for acronyms
                if not matches_org and re.match(r'^[A-Z]{2,}$', possessed):
                    results['core_entity'] = possessed

        # Step 5: Extract role candidates
        role_candidates = []

        # 5.1: Look for appositives
        for token in doc:
            if token.dep_ == 'appos':
                appos_span = doc[token.left_edge.i:token.right_edge.i+1]

                # Check if this contains our core entity
                contains_core = False
                if results['core_entity'] and results['core_entity'] in appos_span.text:
                    contains_core = True

                if not contains_core:
                    role_candidates.append(appos_span.text)

        # 5.2: Extract parts not covered by core entity or geo info
        # Mark positions covered by entities
        covered = [False] * len(span_text)

        # Mark core entity
        if results['core_entity']:
            pattern = re.compile(r'\b' + re.escape(results['core_entity']) + r'\b')
            for match in pattern.finditer(span_text):
                start, end = match.span()
                for i in range(start, min(end, len(covered))):
                    covered[i] = True

        # Mark geographic info
        if results['geographic_info']:
            pattern = re.compile(r'\b' + re.escape(results['geographic_info']) + r'\b')
            for match in pattern.finditer(span_text):
                start, end = match.span()
                for i in range(start, min(end, len(covered))):
                    covered[i] = True

        # Extract uncovered segments
        uncovered_segments = []
        current = []

        for i, char in enumerate(span_text):
            if not covered[i]:
                current.append(char)
            elif current:
                segment = ''.join(current).strip(' ,')
                if segment and len(segment) > 1:
                    uncovered_segments.append(segment)
                current = []

        # Don't forget the last segment
        if current:
            segment = ''.join(current).strip(' ,')
            if segment and len(segment) > 1:
                uncovered_segments.append(segment)

        # Add these segments to role candidates
        role_candidates.extend(uncovered_segments)

        # 5.3: Special case for words before person entity
        if results['core_entity'] and person_entities:
            # Find the entity start position
            person_start = None
            for ent in person_entities:
                if ent['text'] == results['core_entity']:
                    person_start = ent['start']
                    break
                
            if person_start is not None and person_start > 0:
                # Check for text before person
                before_person = span_text[:person_start].strip()
                if before_person:
                    role_candidates.append(before_person)

        # 5.4: Special case for words after organization
        if results['core_entity'] and org_entities:
            # Find the entity end position
            org_end = None
            for ent in org_entities:
                if ent['text'] == results['core_entity']:
                    org_end = ent['end']
                    break
                
            if org_end is not None and org_end < len(span_text):
                # Get text after org entity
                after_org = span_text[org_end:].strip()
                if after_org:
                    # Clean up possessives in the after text
                    after_org = re.sub(r"^'s\s+", "", after_org)
                    if after_org:
                        role_candidates.append(after_org)

        # Step 6: Choose the best role candidate
        if role_candidates:
            # Clean up candidates
            cleaned_candidates = []
            for candidate in role_candidates:
                # Remove geographic entities from role description
                if results['geographic_info']:
                    candidate = re.sub(r'\b' + re.escape(results['geographic_info']) + r'\b', '', candidate)

                # Clean up whitespace, possessives and punctuation
                candidate = re.sub(r"['’]s\s+", " ", candidate)  # Remove possessives
                candidate = re.sub(r'[,.:;]+$', '', candidate)  # Remove trailing punctuation
                candidate = re.sub(r'\s+', ' ', candidate).strip()  # Clean whitespace
                # remove initial "'" or "’"
                candidate = re.sub(r"^[‘’]", '', candidate).strip()

                # Remove "of" without context
                candidate = re.sub(r'\bof\b\s*$', '', candidate).strip()

                if candidate:
                    cleaned_candidates.append(candidate)

            role_candidates = cleaned_candidates

            # Use embedding similarity if available
            if job_title_embeddings and get_embedding_func and role_candidates:
                best_match = None
                best_score = 0

                for candidate in role_candidates:
                    try:
                        candidate_emb = get_embedding_func(candidate)

                        for title, title_emb in job_title_embeddings.items():
                            sim = cos_sim([candidate_emb], [title_emb])[0][0]
                            if sim > best_score:
                                best_score = sim
                                best_match = candidate
                    except:
                        continue
                    
                if best_score > 0.5:
                    results['role'] = best_match
                    return results

            # Fallback heuristics if embedding matching doesn't work
            scored_candidates = []
            for candidate in role_candidates:
                score = 0

                # Check for role keywords
                role_keywords = ['official', 'president', 'mayor', 'secretary', 'minister', 
                                'member', 'council', 'general', 'party', 'service', 
                                'airport', 'police', 'attacker', 'right-wing', 'wing']

                for keyword in role_keywords:
                    if keyword in candidate.lower():
                        score += 5
                        break
                    
                # Favor multi-word candidates
                word_count = len(candidate.split())
                score += min(word_count, 3)

                # Favor candidates that appear at the beginning of the span
                if span_text.lower().startswith(candidate.lower()):
                    score += 2

                # Penalize very short candidates (less than 3 characters)
                if len(candidate) < 3:
                    score -= 2

                scored_candidates.append((candidate, score))

            if scored_candidates:
                results['role'] = max(scored_candidates, key=lambda x: x[1])[0]

        # Step 7: Final cleanup
        if results['role']:
            # Ensure descriptors like "right-wing party" are fully captured
            if 'party' in span_text.lower() and 'wing' in results['role'].lower() and 'party' not in results['role'].lower():
                results['role'] += ' party'

            # Ensure airport, service, etc. are included in role when appropriate
            for suffix in ['airport', 'service', 'council']:
                if suffix in span_text.lower() and suffix not in results['role'].lower():
                    if results['role'].strip() and suffix not in results['role'].lower():
                        results['role'] += f' {suffix}'
        if results['role']:
            logger.debug(f"Converted '{span_text}' to '{results['core_entity']} ({results['role']})'")
        else:
            logger.debug(f"Converted '{span_text}' to '{results['core_entity']}' (no role found)")
        return results

    def strip_ents(self, doc):
        """
        Strip out named entities from text, leaving only non-entity tokens.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            str: Text with named entities removed
        """
        skip_list = ['a', 'and', 'the', "'s", "'", "s"]
        non_ent_tokens = [
            token.text_with_ws for token in doc 
            if token.ent_type_ == "" and token.text.lower() not in skip_list
        ]
        return ''.join(non_ent_tokens).strip()
    
    def make_acronym_dicts(self, text=None, doc=None, nlp=None):
        """
        Quick tool to identify acronyms (and their referents) in a doc.

        Args:
            text: string of text to process
            doc: spaCy doc object
        Returns:
            acronym_entities: dict of acronyms and their referents
        """
        if text is None and doc is None:
            raise ValueError("Either text or doc must be provided.")
        if text is not None and doc is None:
            if nlp is None:
                raise ValueError("nlp object must be provided if doc is provided.")
            doc = nlp(text)

        acronym_entities = {}
        for ent in doc.ents:
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
    
    def get_noun_phrases(self, doc):
        """
        Extract non-entity noun phrases from a document.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            str: Space-joined noun phrases
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
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            list: List of noun phrases
        """
        return [chunk for chunk in doc.noun_chunks if chunk[-1].ent_type_ == ""]


#######################################################
# Country Detection
#######################################################

class CountryDetector:
    """
    Country detection and pattern matching utilities.
    
    This class provides methods for detecting countries and nationalities
    in text.
    
    Example:
        detector = CountryDetector("./assets")
        country, remaining_text = detector.search_nat("German Chancellor")
        # Returns: ("DEU", "Chancellor")
    """
    
    def __init__(self, base_path=DEFAULT_BASE_PATH):
        """
        Initialize the country detector.
        
        Args:
            base_path: Path to directory containing the countries.csv file
        """
        self.nat_list, self.nat_list_cat, self.nat_list_name, self.nat_list_name_cat = self._load_county_dict(base_path)
    
    def _load_county_dict(self, base_path):
        """
        Construct a list of regular expressions to find countries by their name and nationality.
        
        Args:
            base_path: Path to directory containing the countries.csv file
            
        Returns:
            tuple: Two lists of pattern tuples for direct and indirect country mentions
        """
        file = os.path.join(base_path, "countries.csv")
        countries = pd.read_csv(file)
        
        # Direct country name/nationality patterns
        nat_list = []
        nat_list_name = []
        for _, row in countries.iterrows():
            # Handle nationalities
            nationalities = [nat.strip() for nat in row['Nationality'].split(",")]
            for nat in nationalities:
                pattern = (re.compile(nat + r"(?=[^a-z]|$)"), row['CCA3'])
                pattern_name = (re.compile(nat + r"(?=[^a-z]|$)"), row['Name'])
                nat_list.append(pattern)
                nat_list_name.append(pattern_name)
            
            # Handle country names
            pattern = (re.compile(row['Name']), row['CCA3'])
            pattern_name = (re.compile(row['Name']), row['Name'])
            nat_list.append(pattern)
            nat_list_name.append(pattern_name)

        
        # Category patterns (for "of X" or "in X" constructions)
        nat_list_cat = []
        nat_list_name_cat = []
        for prefix in ['of ', 'in ']: 
            for _, row in countries.iterrows():
                # Handle nationalities in categories
                nationalities = [nat.strip() for nat in row['Nationality'].split(",")]
                for nat in nationalities:
                    pattern = (re.compile(prefix + nat), row['CCA3'])
                    pattern_name = (re.compile(prefix + nat), row['Name'])
                    nat_list_cat.append(pattern)
                    nat_list_name_cat.append(pattern_name)
                
                # Handle country names in categories
                pattern = (re.compile(prefix + row['Name']), row['CCA3'])
                pattern_name = (re.compile(prefix + row['Name']), row['Name'])
                nat_list_cat.append(pattern)
                nat_list_name_cat.append(pattern_name)
        
        return nat_list, nat_list_cat, nat_list_name, nat_list_name_cat

    def search_nat(self, text, method="longest", categories=False, use_name=False):
        """
        Search for country names/nationalities in text and return canonical form.
        
        Args:
            text: Text to search for country mentions
            method: Method to use when multiple countries are found ('longest' or 'first')
            categories: Whether to use category patterns (of X, in X)
            use_name: Whether to return the *name* of the country instead of the ISO code
            
        Returns:
            tuple: (country_code, trimmed_text) or (None, original_text) if no country found
        """
        if not text:
            return None, text
            
        # Normalize text for consistent matching
        text = unidecode.unidecode(text)
        found = []
        
        # Use appropriate pattern list based on categories flag
        if use_name:
            patterns = self.nat_list_name_cat if categories else self.nat_list_name
        else:
            patterns = self.nat_list_cat if categories else self.nat_list
        
        # Find all matching countries
        for pattern, country in patterns:
            match = re.search(pattern, text)
            if match:
                # Remove the matched country/nationality from text
                trimmed_text = re.sub(pattern, "", text).strip()
                trimmed_text = re.sub(r" +", " ", trimmed_text).strip()
                found.append((country, trimmed_text.strip(), match))
        
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


#######################################################
# Model Management
#######################################################

class ModelManager:
    """
    Model loading and management utilities.
    
    This class handles loading and caching of NLP models.
    
    Example:
        manager = ModelManager("./assets")
        nlp = manager.load_spacy_lg()
        trf = manager.load_trf_model()
    """
    
    def __init__(self, base_path=DEFAULT_BASE_PATH, device=None):
        """
        Initialize the model manager.
        
        Args:
            base_path: Path to directory containing model files
            device: Device to use for model inference ('cuda' or None)
        """
        self.base_path = base_path
        self.device = device
        self.models = {}  # Cache for loaded models
    
    def load_spacy_lg(self):
        """
        Load and return the spaCy language model.
        
        Returns:
            spaCy model: Loaded language model
        """
        if 'spacy' not in self.models:
            self.models['spacy'] = spacy.load("en_core_web_lg")
        return self.models['spacy']

    def load_trf_model(self, model_dir=DEFAULT_MODEL_PATH):
        """
        Load and return the sentence transformer model.
        
        Args:
            model_dir: Path or name of the transformer model
            
        Returns:
            SentenceTransformer: Loaded transformer model
        """
        if 'trf' not in self.models:
            self.models['trf'] = SentenceTransformer(model_dir, trust_remote_code=True)
        return self.models['trf']

    def load_actor_sim_model(self, model_dir=DEFAULT_SIM_MODEL_PATH):
        """
        Load the actor similarity model trained on Wikipedia redirects.
        
        This model helps identify if two names refer to the same entity.
        
        Args:
            model_dir: Directory containing the similarity model
            
        Returns:
            SentenceTransformer: Loaded similarity model
        """
        if 'actor_sim' not in self.models:
            combo_path = os.path.join(self.base_path, model_dir)
            self.models['actor_sim'] = SentenceTransformer(combo_path)
        return self.models['actor_sim']


#######################################################
# Cache Management
#######################################################

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
# Agent Matching
#######################################################

class AgentMatcher:
    """
    Match text to agent patterns using transformer models.
    
    This class provides methods for matching text to PLOVER agent patterns
    using transformer embeddings.
    
    Example:
        matcher = AgentMatcher(trf_model, "./assets")
        match = matcher.trf_agent_match("Chancellor", country="DEU")
    """
    
    def __init__(self, trf_model=None, base_path=DEFAULT_BASE_PATH, 
                 device=None, text_processor=None):
        """
        Initialize the agent matcher.
        
        Args:
            trf_model: Sentence transformer model
            base_path: Path to directory containing agent files
            device: Device to use for inference ('cuda' or None)
            text_processor: TextPreProcessor instance
        """
        self.base_path = base_path
        self.device = device
        
        # Initialize resources
        if trf_model is None:
            model_manager = ModelManager(base_path, device)
            self.trf = model_manager.load_trf_model()
        else:
            self.trf = trf_model
            
        if text_processor is None:
            self.text_processor = TextPreProcessor()
        else:
            self.text_processor = text_processor
            
        # Load agent data
        self.agents = self._load_and_clean_agents()
        self.trf_matrix = self._load_embeddings()
    
    def _load_and_clean_agents(self):
        """
        Load the PLOVER/CAMEO agents file and clean the data.
        
        Returns:
            list: Cleaned list of agent pattern dictionaries
        """
        file = os.path.join(self.base_path, "PLOVER_agents.txt")
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

    def _load_embeddings(self):
        """
        Load pre-computed embedding matrices or compute and save them if needed.
        
        Returns:
            numpy.ndarray: Matrix of agent pattern embeddings
        """
        # Check if the agents file and embedding matrix are mismatched
        hash_file = os.path.join(self.base_path, "PLOVER_agents.hash")
        try:
            with open(hash_file, "r") as f:
                existing_hash = f.read()
        except FileNotFoundError:
            existing_hash = ""
            
        # Get current hash of agents file
        agent_file = os.path.join(self.base_path, "PLOVER_agents.txt")
        with open(agent_file, "r", encoding="utf-8") as f:
            data = f.read() 
        current_hash = hash(data)
        
        # Recompute embeddings if hash mismatch
        if str(existing_hash) != str(current_hash):
            logger.info("Agents file and pre-computed matrix are mismatched. Recomputing...")
            patterns = [agent['pattern'] for agent in self.agents]
            trf_matrix = self.trf.encode(patterns, show_progress_bar=False, device=self.device)
            
            # Save new embeddings and hash
            file_bert = os.path.join(self.base_path, "bert_matrix.pkl")
            with open(file_bert, "wb") as f:
                pickle.dump(trf_matrix, f)
            with open(hash_file, "w") as f:
                f.write(str(current_hash))
        
        # Load embeddings
        logger.info("Reading in BERT matrix")
        file_bert = os.path.join(self.base_path, "bert_matrix.pkl")
        with open(file_bert, "rb") as f:
            return pickle.load(f)

    def trf_agent_match(self, text, country="", method="cosine", threshold=THRESHOLD_COSINE_SIMILARITY):
        """
        Compare input text to the agent file using sentence transformer embeddings.
        
        Args:
            text: Text to match against agent patterns
            country: Country code to include in the result
            method: Similarity method to use ('cosine' or 'dot')
            threshold: Similarity threshold below which matches are ignored
            
        Returns:
            dict or None: Match information or None if no match above threshold
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
        text = self.text_processor.clean_query(text)
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

    def short_text_to_agent(self, text, strip_ents=False, threshold=THRESHOLD_COSINE_SIMILARITY, 
                           country_detector=None):
        """
        Convert short text to an agent code, optionally stripping entities first.
        
        Args:
            text: Text to convert
            strip_ents: Whether to strip named entities before matching
            threshold: Similarity threshold for matching
            country_detector: CountryDetector instance (optional)
            
        Returns:
            dict or None: Agent code information or None if no match
        """
        # Use provided country detector or create a new one
        if country_detector is None:
            country_detector = CountryDetector(self.base_path)
            
        # Extract country and clean text
        country, trimmed_text = country_detector.search_nat(text)
        trimmed_text = self.text_processor.clean_query(trimmed_text)
        
        # Optionally strip entities
        if strip_ents:
            try:
                model_manager = ModelManager(self.base_path, self.device)
                doc = self.nlp(text)
                trimmed_text = self.text_processor.strip_ents(doc)
            except IndexError:
                # If NLP fails, continue with trimmed_text as-is
                pass
                
            if trimmed_text == "s":
                return None
                
        # Match against agent patterns
        return self.trf_agent_match(trimmed_text, country=country, threshold=threshold)


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
    
    def __init__(self):
        """Initialize the Wikipedia client."""
        self.conn = self.setup_es()
        self.check_wiki(self.conn)
    
    def setup_es(self):
        """
        Establish connection to Elasticsearch and return search object.
        
        Returns:
            Search: Elasticsearch search object
        
        Raises:
            ConnectionError: If Elasticsearch connection fails
        """
        try:
            client = Elasticsearch()
            client.ping()
            conn = Search(using=client, index="wiki")
            return conn
        except Exception as e:
            raise ConnectionError(f"Could not connect to Elasticsearch: {e}")

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

    def run_wiki_search(self, query_term, limit_term="", fuzziness="AUTO", max_results=200,
                   fields=['title^50', 'redirects^50', 'alternative_names'],
                   score_type="best_fields"):
        """
        Search Wikipedia for a given query term.
        
        Args:
            query_term: Term to search for
            limit_term: Term to limit results by
            fuzziness: Elasticsearch fuzziness parameter
            max_results: Maximum number of results to return
            fields: Fields to search in
            score_type: Elasticsearch score type
            
        Returns:
            list: List of Wikipedia article dictionaries
        """
        # Construct query
        if not limit_term:
            query = {
                "bool": {
                    "should": [
                        # Exact match on title (case-sensitive)
                        {"term": {"title": {"value": query_term, "boost": 150}}},
                        # Analyzed match on title (case-insensitive, tokenized)
                        {"match": {"title": {"query": query_term, "boost": 50}}},

                        # Exact match on redirects (for acronyms)
                        {"term": {"redirects": {"value": query_term, "boost": 150}}},
                        # Analyzed match on redirects
                        {"match": {"redirects": {"query": query_term, "boost": 50}}},

                        # Exact match on alternative names
                        {"term": {"alternative_names": {"value": query_term, "boost": 125}}},
                        # Analyzed match on alternative names
                        {"match": {"alternative_names": {"query": query_term, "boost": 25}}},

                        # Analyzed match on short description and intro para
                        {"match": {"intro_para": {"query": query_term, "boost": 5}}},
                        # Analyzed match on categories
                        {"match": {"short_desc": {"query": query_term, "boost": 10}}}
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
    
    def __init__(self, wiki_client=None, text_processor=None):
        """
        Initialize the Wikipedia searcher.
        
        Args:
            wiki_client: WikiClient instance
            text_processor: TextPreProcessor instance
        """
        if wiki_client is None:
            self.wiki_client = WikiClient()
        else:
            self.wiki_client = wiki_client
            
        if text_processor is None:
            self.text_processor = TextPreProcessor()
        else:
            self.text_processor = text_processor
    
    def search_wiki(self, query_term, limit_term="", fuzziness="AUTO", max_results=200, 
                   score_type="best_fields"):
        """
        Search Wikipedia for a given query term.
        
        Args:
            query_term: Term to search for
            limit_term: Term to limit results by
            fuzziness: Elasticsearch fuzziness parameter
            max_results: Maximum number of results to return
            fields: Fields to search in
            score_type: Elasticsearch score type
            
        Returns:
            list: List of Wikipedia article dictionaries
        """
        # Clean query term
        query_term = self.text_processor.clean_query(query_term)
        logger.debug(f"Using query term: '{query_term}'")
        
        # Perform search via client
        return self.wiki_client.run_wiki_search(
            query_term=query_term,
            limit_term=limit_term,
            fuzziness=fuzziness,
            max_results=max_results,
            score_type=score_type
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
        
        # Filter out articles with short intro paragraphs
        good_res = [r for r in good_res if len(r['intro_para']) > 50 and r['intro_para'].strip()]
        
        return good_res


#######################################################
# Wikipedia Matcher
#######################################################

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
    
    def __init__(self, wiki_searcher=None, text_processor=None, 
                trf_model=None, actor_sim_model=None, device=None,
                nlp=None,
                wiki_sort_method="neural"):
        """
        Initialize the Wikipedia matcher.
        
        Args:
            wiki_searcher: WikiSearcher instance
            text_processor: TextPreProcessor instance
            trf_model: Sentence transformer model
            actor_sim_model: Actor similarity model
            device: Device to use for inference ('cuda' or None)
            wiki_sort_method: Method to use for sorting results
        """
        # Initialize components or use provided ones
        if wiki_searcher is None:
            self.wiki_searcher = WikiSearcher()
        else:
            self.wiki_searcher = wiki_searcher
            
        if text_processor is None:
            self.text_processor = TextPreProcessor()
        else:
            self.text_processor = text_processor
            
        # Initialize models if not provided
        if trf_model is None or actor_sim_model is None:
            model_manager = ModelManager(device=device)
            self.trf = trf_model if trf_model else model_manager.load_trf_model()
            self.actor_sim = actor_sim_model if actor_sim_model else model_manager.load_actor_sim_model()
        else:
            self.trf = trf_model
            self.actor_sim = actor_sim_model
        
        if nlp is None:
            model_manager = ModelManager(device=device)
            self.nlp = model_manager.load_spacy_lg()
        else:
            self.nlp = nlp
            
        self.device = device
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
        if len(articles) == 0:
            return pd.DataFrame()
        # Prepare data for DataFrame
        data = []

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


            data.append({
                'index': i,
                'title': title,
                'exact_title_match': exact_title_match,
                'redirect_match': redirect_match,
                'alt_name_match': alt_name_match,
                'alt_names_count': alt_names_count,
                'redirect_names_count': redirect_names_count,
                'num_es_results': results_count,
                'intro_length': intro_length,
                'country_match': country_match,

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
            intros = [article['intro_para'][0:300] for article in articles]
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

        # Normalize scores
        for col in ['title_sim', 'context_sim_intro', 'context_sim_short',
                    'actor_desc_sim_intro', 'actor_desc_sim_short',
                    'lcs', 'levenshtein', 'country_match', 'exact_title_match',
                    'alt_name_match', 'redirect_match']:
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
        col_norm = f"{col}_norm"
        return (col_norm, df[col] / df[col].max())

    def _apply_selection_rules(self, df, articles, context):
        """
        Apply a series of prioritized rules to select the best article.

        Args:
            df: DataFrame with article scores
            articles: List of Wikipedia article dictionaries
            wiki_sort_method: Method to use for sorting results
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
                logger.debug(f"Returning best context match among redirect matches. Title: {selected['title']}: {best_context_match['context_sim']}")
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
                logger.debug(f"Returning best context match among alternative name matches. Title: {selected['title']}: {selected['context_sim']}")
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

    def pick_best_wiki(self, 
                       query_term, 
                       results, 
                       context="", 
                       country="",
                       actor_desc="",
                       wiki_sort_method=None, 
                       rank_fields=None):
        """
        Select the best Wikipedia article from search results using a scoring matrix approach.

        Args:
            query_term: Query term to match
            results: List of Wikipedia article search results
            context: Context text to help with disambiguation
            country: Country code to help with disambiguation
            actor_desc: Actor description to help with disambiguation
            wiki_sort_method: Method to use for sorting results
            rank_fields: Fields to use for ranking

        Returns:
            dict or None: Best matching Wikipedia article or None if no good match
        """
        import pandas as pd

        # Use instance method if not provided
        if wiki_sort_method is None:
            wiki_sort_method = self.wiki_sort_method

        # Set default rank fields if not provided
        if rank_fields is None:
            rank_fields = ['title', 'categories', 'alternative_names', 'redirects']

        # Clean query term
        query_term = self.text_processor.clean_query(query_term)
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

        # Create scoring dataframe
        df = self._create_scoring_dataframe(good_res, query_term, context=context, 
                                            actor_desc=actor_desc, 
                                            country=country)

        # Apply prioritized selection rules
        selected = self._apply_selection_rules(df, articles=good_res, context=context)

        if selected is not None:
            logger.debug(f"Selected article: {selected['title']} (reason: {selected['wiki_reason']})")
            return selected
        else:
            logger.debug("No good match found")
            # print the best overall score
            best_overall = df.sort_values('combined_score', ascending=False).iloc[0].to_dict()
            logger.debug(f"Skipping best overall score: ({best_overall['title']}): {best_overall['combined_score']}")
            return None
        
    def _expand_query(self, query_term, context):
        """
        Expand the query term using named entity recognition (NER) and acronyms.
        """
        context_doc = self.nlp(context)
        if context_doc:
            expanded_query = [i.text for i in context_doc.ents if query_term in i.text]
            if expanded_query:
                # take the first one
                expanded_query = expanded_query[0]
                if len(expanded_query) > len(query_term):
                    query_term = expanded_query
                    logger.debug(f"Using NER to expand context: {expanded_query}")
        
            acronym_dict = self.text_processor.make_acronym_dicts(doc=context_doc)
            # Check if query term is an acronym
            # and expand it if found in the acronym dictionary
            if query_term in acronym_dict:
                logger.debug(f"Using acronym expansion: {query_term} --> {acronym_dict[query_term]}")
                query_term = acronym_dict[query_term]
            if query_term in acronym_dict:
                query_term = acronym_dict[query_term]
                logger.debug(f"Using acronym expansion: {query_term}")
        return query_term

    def query_wiki(self, 
                   query_term, 
                   limit_term="", 
                   country="", 
                   context="", 
                   actor_desc="",
                   max_results=200):
        """
        Search Wikipedia and return the best matching article.
        
        Args:
            query_term: Term to search for
            limit_term: Term to limit results by
            country: Country code to help with disambiguation
            context: Context text to help with disambiguation
            actor_desc: Actor description (automatically parsed)
            max_results: Maximum results to return from search
            
        Returns:
            dict or None: Best matching Wikipedia article or None if no good match
        """
        # Do NER expansion by default
        if context:
            logger.debug("Context present, so attempting NER expansion")
            query_term = self._expand_query(query_term, context)

        # Try exact search first
        logger.debug("Starting with exact search")
        results = self.wiki_searcher.search_wiki(
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
        logger.debug("Falling back to fuzzy search")
        results = self.wiki_searcher.search_wiki(
            query_term, 
            limit_term=limit_term, 
            fuzziness=1, 
            max_results=max_results
        )
        best = self.pick_best_wiki(
            query_term, 
            results, 
            country=country, 
            context=context,
            actor_desc=actor_desc,
        )
       
        return best


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
    
    def __init__(self, country_detector=None, text_processor=None, agent_matcher=None,
                base_path=DEFAULT_BASE_PATH, device=None):
        """
        Initialize the Wikipedia parser.
        
        Args:
            country_detector: CountryDetector instance
            text_processor: TextPreProcessor instance
            agent_matcher: AgentMatcher instance
            base_path: Path to directory containing assets
            device: Device to use for inference ('cuda' or None)
        """
        # Initialize components or use provided ones
        if country_detector is None:
            self.country_detector = CountryDetector(base_path)
        else:
            self.country_detector = country_detector
            
        if text_processor is None:
            self.text_processor = TextPreProcessor()
        else:
            self.text_processor = text_processor
            
        if agent_matcher is None:
            model_manager = ModelManager(base_path, device)
            trf_model = model_manager.load_trf_model()
            self.agent_matcher = AgentMatcher(trf_model, base_path, device, self.text_processor)
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
                office_text = self.text_processor.clean_query(office['office'])
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
        old_codes_raw = [self.agent_matcher.trf_agent_match(self.text_processor.clean_query(o['office'])) for o in offices]
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
    
    def _get_actor_priority(self, code_1):
        """
        Get priority value for an actor code.
        
        Args:
            code_1: Actor code
            
        Returns:
            int: Priority value
        """
        return self.ACTOR_TYPE_PRIORITIES.get(code_1, 0)

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


#######################################################
# Event Processing
#######################################################

class EventProcessor:
    """
    Process event data with actor resolution.
    
    This class provides methods for processing event data and adding
    actor resolution information.
    
    Example:
        resolver = ActorResolver()
        processor = EventProcessor(resolver)
        processed_events = processor.process(events)
    """
    
    def __init__(self, actor_resolver):
        """
        Initialize the event processor.
        
        Args:
            actor_resolver: ActorResolver instance to use for resolution
        """
        self.actor_resolver = actor_resolver
    
    def process(self, event_list, save_intermediate=False):
        """
        Process a list of events to resolve actor attributes.
        
        For each event, adds actor resolution information to the ACTOR and RECIP attributes.
        
        Args:
            event_list: List of event dictionaries
            save_intermediate: Whether to save intermediate results
            
        Returns:
            list: The same event list with actor resolution information added
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
                    res = self.actor_resolver.agent_to_code(actor_text, query_date=query_date)
                    
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
        if save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_actor_resolution_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)
                
        return event_list


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
        code = resolver.agent_to_code("German Chancellor")
        processed_events = resolver.process(events)
    """
    
    def __init__(self, 
                spacy_model=None,
                base_path=DEFAULT_BASE_PATH,
                save_intermediate=False,
                wiki_sort_method="neural",
                gpu=False):
        """
        Initialize the ActorResolver with the necessary models and data.
        
        Args:
            spacy_model: Pre-loaded spaCy model to use
            base_path: Path to the directory containing assets
            save_intermediate: Whether to save intermediate results
            wiki_sort_method: Method to use for sorting Wikipedia results
            gpu: Whether to use GPU for model inference
        """
        # Set device for model inference
        self.device = 'cuda' if gpu else None
        
        # Initialize utility classes
        self.text_processor = TextPreProcessor()
        self.cache_manager = CacheManager()
        self.country_detector = CountryDetector(base_path)
        
        # Initialize model manager and load models
        self.model_manager = ModelManager(base_path, self.device)
        self.nlp = spacy_model if spacy_model else self.model_manager.load_spacy_lg()
        self.trf = self.model_manager.load_trf_model()
        self.actor_sim = self.model_manager.load_actor_sim_model()
        
        # Initialize agent matcher
        self.agent_matcher = AgentMatcher(
            self.trf, 
            base_path, 
            self.device, 
            self.text_processor
        )
        
        # Initialize Wikipedia components
        self.wiki_client = WikiClient()
        self.wiki_searcher = WikiSearcher(
            self.wiki_client, 
            self.text_processor
        )
        self.wiki_matcher = WikiMatcher(
            self.wiki_searcher, 
            self.text_processor, 
            self.trf, 
            self.actor_sim, 
            self.device, 
            wiki_sort_method
        )
        self.wiki_parser = WikiParser(
            self.country_detector,
            self.text_processor,
            self.agent_matcher,
            base_path,
            self.device
        )
        
        # Initialize code selector
        self.code_selector = CodeSelector()
        
        # Initialize event processor
        self.event_processor = EventProcessor(self)
        
        # Store configuration
        self.base_path = base_path
        self.save_intermediate = save_intermediate
        self.wiki_sort_method = wiki_sort_method

    def agent_to_code(self, text, doc=None, context="", query_date="today", known_country="", search_limit_term=""):
        """
        Resolve an actor mention to a code representing their role.
        
        Args:
            text: Text mention of the actor to resolve
            context: Additional context to help with disambiguation
            query_date: Date to use when determining current offices
            known_country: Country code if already known
            search_limit_term: Term to limit Wikipedia search results
            
        Returns:
            dict or None: Actor code information or None if resolution fails
        """
        # Check cache first
        cache_key = text + "_" + str(query_date)
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.debug("Returning from cache")
            return cached_result
        
        if doc is None:
            doc = self.nlp(text)

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
            self.cache_manager.set(cache_key, code_full_text)
            return self.code_selector.clean_best(code_full_text)
            
        # Parse entities in text
        # TODO: all of this probably goes away with the new entity splitter
        try:
            doc = self.nlp(trimmed_text)
            non_ent_text = self.text_processor.strip_ents(doc)
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
            code_full_text = self.agent_matcher.trf_agent_match(trimmed_text, country=country)
            
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
                    code_full_text = self.code_selector.clean_best(code_full_text)
                    self.cache_manager.set(cache_key, code_full_text)
                    return code_full_text
            else:
                logger.debug(f"No direct match found for {trimmed_text}")
                
        # Try Wikipedia lookup for better resolution
        logger.debug(f"Trying Wikipedia lookup with: {trimmed_text}")
        wiki_codes = []
        wiki = self.wiki_matcher.query_wiki(
            query_term=trimmed_text, 
            country=known_country, 
            context=context,
            limit_term=search_limit_term
        )
        
        if wiki:
            logger.debug(f"Wikipedia page found: {wiki['title']}")
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
        best = self.code_selector.pick_best_code(all_codes, country)
        best = self.code_selector.clean_best(best)
        
        # Add unique codes lists for reference
        if best:
            best['all_code1s'] = unique_code1s
            best['all_code2s'] = unique_code2s
            
        # Cache and return result
        self.cache_manager.set(cache_key, best)
        return best

    def process(self, event_list):
        """
        Process a list of events to resolve actor attributes.
        
        For each event, adds actor resolution information to the ACTOR and RECIP attributes.
        
        Args:
            event_list: List of event dictionaries
            
        Returns:
            list: The same event list with actor resolution information added
        """
        return self.event_processor.process(event_list, self.save_intermediate)


#######################################################
# Main Entry Point
#######################################################

def main():
    """
    Main entry point for actor resolution.
    
    This function demonstrates how to use the ActorResolver.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Resolve actors in events to PLOVER codes")
    parser.add_argument("input_file", help="Input JSONL file of events")
    parser.add_argument("output_file", help="Output JSONL file with actor resolution")
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH, help="Path to assets directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    args = parser.parse_args()
    
    # Load events from input file
    with jsonlines.open(args.input_file, "r") as f:
        events = list(f.iter())
    
    # Create actor resolver
    resolver = ActorResolver(
        base_path=args.base_path,
        save_intermediate=args.save_intermediate,
        gpu=args.gpu
    )
    
    # Process events
    processed_events = resolver.process(events)
    
    # Save results to output file
    with jsonlines.open(args.output_file, "w") as f:
        f.write_all(processed_events)
    
    print(f"Processed {len(processed_events)} events. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()