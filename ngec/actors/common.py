
from importlib import resources
import logging
from pathlib import Path
import re
from typing import Literal
import unidecode

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import spacy
from spacy.language import Language


# Constants
DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v3"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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

    # TODO #26: allow overriding models
    
    def __init__(self, device=None):
        """
        Initialize the model manager.
        
        Args:
            base_path: Path to directory containing model files
            device: Device to use for model inference ('cuda' or None)
        """
        self.device = device
        self.models = {}  # Cache for loaded models


    def load_spacy_lg(self) -> Language:
        """
        Load and return the spaCy language model.
        
        Returns:
            spaCy model: Loaded language model
        """
        if 'spacy' not in self.models:
            self.models['spacy'] = spacy.load("en_core_web_lg")
        return self.models['spacy']


    def load_trf_model(self, model_dir: None | str | Path=None) -> SentenceTransformer:
        """
        Load and return the sentence transformer model.
        
        Args:
            model_dir: Path or name of the transformer model
            
        Returns:
            SentenceTransformer: Loaded transformer model
        """
        if 'trf' not in self.models:
            if not model_dir:
                model_dir = Path(str(resources.files('ngec'))) / 'assets'
            self.models['trf'] = SentenceTransformer("jinaai/jina-embeddings-v3", 
                                                     trust_remote_code=True, 
                                                     model_kwargs={'use_flash_attn': False},
                                                     device=self.device)
        return self.models['trf']

    
    



#######################################################
# Text Processing Utilities
#######################################################


def clean_query(qt):
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

    # TODO: i think only strip_ents is used? Other methods can be deleted.

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
        >>> detector = CountryDetector()
        >>> detector.search_nat("German Chancellor")
        ("DEU", "Chancellor")
    """
    
    def __init__(self, country_csv_path: str | Path | None = None):
        """
        Initialize the country detector.
        
        Args:
            country_csv_path: Path to a countries.csv-like file; if None, uses 
            the built-in asset
        """
        self.nat_list, self.nat_list_cat, self.nat_list_name, self.nat_list_name_cat = self._load_country_dict(country_csv_path=country_csv_path)


    def _load_country_dict(self, country_csv_path: str | Path | None = None):
        """
        Construct a list of regular expressions to find countries by their name and nationality.
        
        Args:
            country_csv_path: Path to a countries.csv-like file; if None, uses 
            the built-in asset
            
        Returns:
            tuple: Two lists of pattern tuples for direct and indirect country mentions
        """
        if country_csv_path is None:
            # Load from package resources
            with (resources.files('ngec') / 'assets' / 'countries.csv').open('r') as f:
                countries = pd.read_csv(f)
        else:
            countries = pd.read_csv(country_csv_path)
        
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

    def search_nat(self, 
                   text: str, 
                   method: Literal["longest", "first"] = "longest",
                   categories: bool = False, 
                   use_name: bool = False
                   ) -> tuple[str | None, str]:
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
        if method not in ["longest", "first"]:
            raise ValueError(f"search_nat sorting option must be one of ['longest', 'first']. You gave {method}")

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
        elif method == "first":
            # Return the first occurrence in the text
            found.sort(key=lambda x: x[2].span()[0])
        else:
            pass  # This should not happen due to earlier check
        return found[0][0:2]
            

