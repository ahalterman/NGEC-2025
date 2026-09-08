
from importlib import resources
import logging
import os
from pathlib import Path
import re
from typing import Literal
import unidecode

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from spacy.language import Language

from ..models import load_spacy


# Constants
DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v3"

# Sentence-transformer encoders the Wikipedia matcher knows how to use. Each
# entry says how to load the model and what instruction, if any, to prepend on
# the *query* side (the news story), which is a convention of the model and not
# something we can guess. Pick one with `ModelManager(encoder_name=...)` or the
# NGEC_WIKI_ENCODER environment variable.
#
# The default is static-retrieval-mrl: it has no transformer in it at all, so it
# costs 0.026 CPU-seconds per query against jina's 28.3, while the ranker
# retrained on its features scores 84.4% [78.7, 89.4] on the held-out document
# split against jina's 85.2% [79.3, 90.4] -- three queries in 372, well inside a
# single interval. A ranker trained against each encoder ships as
# `ngec/assets/xgb_model_<name>.json`, and `xgb_model.json` (what WikiMatcher
# loads) is a copy of the default's. Changing this constant without also
# pointing the matcher at the matching ranker asset degrades linking silently,
# because the ranker's four similarity features would then come from a
# different model than the one it was fit on.
WIKI_ENCODERS = {
    # "ranker_asset" names the XGBoost ranker trained on this encoder's
    # features (train_NGEC_2026/train_wiki_model/03_train_ranker.py). The
    # ranker and the encoder must match: WikiMatcher picks the asset from here
    # unless given an explicit wiki_ranker_model. "ranker_threshold" is the
    # minimum ranker probability for the top candidate to be accepted, swept
    # per encoder on the gold document split and the institution probes
    # (train_wiki_model/06_threshold_sweep.py); it is encoder-specific.
    "jinaai/jina-embeddings-v3": {
        "load_kwargs": {"trust_remote_code": True,
                        "model_kwargs": {"use_flash_attn": False}},
        "query_prefix": "",
        "ranker_asset": "xgb_model_jina.json",
        "ranker_threshold": 0.1,   # not swept; the value the pipeline always shipped
    },
    "BAAI/bge-small-en-v1.5": {
        "load_kwargs": {},
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "ranker_asset": "xgb_model_bge-small.json",
        "ranker_threshold": 0.1,   # its institution probes fall off fast above this
    },
    "sentence-transformers/static-retrieval-mrl-en-v1": {
        "load_kwargs": {},
        "query_prefix": "",
        "ranker_asset": "xgb_model_static-mrl.json",
        "ranker_threshold": 0.3,   # 86.0% vs 84.4% gold top-1 at 0.1, -3 institution probes
    },
}
DEFAULT_ENCODER = "sentence-transformers/static-retrieval-mrl-en-v1"

# The agent matcher (PLOVER role patterns) uses its own encoder, chosen
# separately from the wiki encoder because its cosine threshold
# (agent_matcher.THRESHOLD_COSINE_SIMILARITY) is read on this model's
# similarity scale. bge-small replaced jina-embeddings-v3 here on 2026-09-03:
# its cosine scale coincides with jina's (mean best-match 0.795 vs 0.791), it
# agrees with jina's code on 72% of real spans, and ECAV actor categorization
# moved from 55.2% to 56.5% (gold spans) and 41.0% to 41.7% (model spans).
# A static encoder was worse (53.7%): PLOVER patterns are multi-word role
# phrases whose modifiers carry the code. Change this only together with a
# re-evaluation of actor categorization -- `ModelManager(agent_encoder_name=...)`
# or the NGEC_AGENT_ENCODER environment variable override it for that purpose;
# the original jina results reproduce with NGEC_AGENT_ENCODER=jinaai/jina-embeddings-v3
# and the threshold at 0.6.
AGENT_ENCODER = "BAAI/bge-small-en-v1.5"

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

    def __init__(self, device=None, encoder_name: str | None = None,
                 agent_encoder_name: str | None = None):
        """
        Initialize the model manager.

        Args:
            device: Device to use for model inference ('cuda' or None)
            encoder_name: Which sentence transformer to use for the Wikipedia
                matcher, as a Hugging Face model id. Defaults to the
                NGEC_WIKI_ENCODER environment variable, and then to
                DEFAULT_ENCODER. See WIKI_ENCODERS for the models with known
                settings; anything else is loaded with no extra arguments and
                no query prefix.
            agent_encoder_name: Which sentence transformer the *agent* matcher
                should use. Defaults to the NGEC_AGENT_ENCODER environment
                variable, and then to AGENT_ENCODER. Changing it changes the
                cosine scale the agent thresholds are read on, so it exists for
                evaluation rather than for everyday use.
        """
        self.device = device
        self.models = {}  # Cache for loaded models

        if agent_encoder_name is None:
            agent_encoder_name = os.environ.get("NGEC_AGENT_ENCODER", AGENT_ENCODER)
        self.agent_encoder_name = agent_encoder_name

        if encoder_name is None:
            encoder_name = os.environ.get("NGEC_WIKI_ENCODER", DEFAULT_ENCODER)
        self.encoder_name = encoder_name
        if encoder_name in WIKI_ENCODERS:
            self.encoder_settings = WIKI_ENCODERS[encoder_name]
        else:
            logger.warning(f"'{encoder_name}' is not in WIKI_ENCODERS. Loading it with no extra "
                           "arguments and no query prefix, which may not be what the model expects.")
            self.encoder_settings = {"load_kwargs": {}, "query_prefix": "",
                                     "ranker_asset": "xgb_model.json",
                                     "ranker_threshold": 0.1}
        # The instruction to prepend to the query side. WikiMatcher reads this.
        self.query_prefix = self.encoder_settings["query_prefix"]


    def load_spacy_lg(self) -> Language:
        """
        Load and return the spaCy language model.
        
        Returns:
            spaCy model: Loaded language model
        """
        if 'spacy' not in self.models:
            self.models['spacy'] = load_spacy("en_core_web_lg")
        return self.models['spacy']


    def _load_encoder(self, model_name: str) -> SentenceTransformer:
        """Load a sentence transformer once and keep it, keyed on its name."""
        if model_name not in self.models:
            load_kwargs = WIKI_ENCODERS.get(model_name, {}).get("load_kwargs", {})
            logger.info(f"Loading sentence transformer: {model_name}")
            self.models[model_name] = SentenceTransformer(model_name,
                                                          device=self.device,
                                                          **load_kwargs)
        return self.models[model_name]

    def load_trf_model(self, model_dir: None | str | Path=None) -> SentenceTransformer:
        """
        Load and return the agent matcher's sentence transformer: the one this
        ModelManager was configured with (agent_encoder_name /
        NGEC_AGENT_ENCODER / AGENT_ENCODER).

        Args:
            model_dir: Path or name of a transformer model to load instead of
                the configured agent encoder

        Returns:
            SentenceTransformer: Loaded transformer model
        """
        return self._load_encoder(str(model_dir) if model_dir else self.agent_encoder_name)

    def load_wiki_encoder(self) -> SentenceTransformer:
        """
        Load and return the Wikipedia matcher's sentence transformer: the one
        this ModelManager was configured with (encoder_name / NGEC_WIKI_ENCODER
        / DEFAULT_ENCODER). Shares the object with load_trf_model when the two
        names coincide.
        """
        return self._load_encoder(self.encoder_name)

    
    



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
    qt = re.sub(r"^['’]s", "", qt).strip()
    
    # Remove ordinals
    qt = re.sub(r"(?<=\d\d)(st|nd|rd|th)\b", '', qt).strip()  # two-digit ordinals
    qt = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', qt).strip()    # one-digit ordinals
    
    # Remove leading numbers and possessives
    qt = re.sub(r"^\d+? ", "", qt).strip()
    qt = re.sub(r"['’]s$", "", qt).strip()
    
    # Return empty string if too short
    if len(qt) < 2:
        return ""
        
    return qt


def strip_ents(doc) -> str:
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
    




#######################################################
# Country Detection
#######################################################

def _bounded(name: str) -> str:
    """Regex for a country name or nationality as a whole word."""
    return r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z0-9])"


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

        # The lists above mix country names and nationalities together, which is
        # what most callers want. Some features need to tell the two apart --
        # "a country name appears in this Wikipedia title" is a different test
        # from "a demonym appears in it" -- so keep them separately as well.
        self.demonyms = []
        for nationalities in self.countries['Nationality']:
            for nat in nationalities.split(","):
                nat = nat.strip()
                # Short demonyms ("Thai", "Lao") match too much inside titles
                if len(nat) > 3 and nat not in self.demonyms:
                    self.demonyms.append(nat)

        # One combined pattern each, for the common "does any country name (or
        # demonym) appear in this short string?" question -- asking it with 750
        # separate patterns per candidate article is too slow. Longest
        # alternative first, so a name is not shadowed by a shorter one that
        # happens to be a prefix of it.
        names = sorted(self.countries['Name'], key=len, reverse=True)
        self.any_country_name = re.compile(
            r"(?<![A-Za-z])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9])")
        self.any_demonym = re.compile(
            r"(?<![A-Za-z])(" + "|".join(re.escape(d) for d in sorted(self.demonyms, key=len, reverse=True)) + r")(?![A-Za-z0-9])")


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
        self.countries = countries

        # Direct country name/nationality patterns
        nat_list = []
        nat_list_name = []
        for _, row in countries.iterrows():
            # Handle nationalities
            nationalities = [nat.strip() for nat in row['Nationality'].split(",")]
            # Patterns are escaped and word-bounded. Unescaped, "U.S." matched
            # "UWSA" and "Mali" matched inside "al-Maliki"; unbounded, "UN"
            # matched inside "UNITA" and "UNRWA".
            for nat in nationalities:
                pattern = (re.compile(_bounded(nat)), row['CCA3'])
                pattern_name = (re.compile(_bounded(nat)), row['Name'])
                nat_list.append(pattern)
                nat_list_name.append(pattern_name)
            
            # Handle country names
            pattern = (re.compile(_bounded(row['Name'])), row['CCA3'])
            pattern_name = (re.compile(_bounded(row['Name'])), row['Name'])
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
                    pattern = (re.compile(prefix + _bounded(nat)), row['CCA3'])
                    pattern_name = (re.compile(prefix + _bounded(nat)), row['Name'])
                    nat_list_cat.append(pattern)
                    nat_list_name_cat.append(pattern_name)
                
                # Handle country names in categories
                pattern = (re.compile(prefix + _bounded(row['Name'])), row['CCA3'])
                pattern_name = (re.compile(prefix + _bounded(row['Name'])), row['Name'])
                nat_list_cat.append(pattern)
                nat_list_name_cat.append(pattern_name)
        
        return nat_list, nat_list_cat, nat_list_name, nat_list_name_cat

    def most_frequent_country(self, text: str) -> str:
        """
        The country named most often in `text`, as a country *name*.

        `search_nat` answers "which country does this short phrase belong to?".
        This answers the document-level question instead: over a whole news
        story, which country is this story about? Counting every mention (by
        name or by nationality) and taking the most frequent one is a crude but
        reliable answer, and it gives the Wikipedia ranker a country to check
        candidate articles against even when the mention itself carries none.

        Returns "" when no country is mentioned. Ties are broken by the order
        of the country list, so the answer is deterministic.

        Args:
            text: the text to scan, typically a whole document

        Returns:
            str: a country name (e.g. "Ghana"), or "" if none was found
        """
        if not text:
            return ""
        text = unidecode.unidecode(text)

        counts = {}
        for pattern, country in self.nat_list_name:
            hits = len(pattern.findall(text))
            if hits:
                counts[country] = counts.get(country, 0) + hits
        if not counts:
            return ""
        return max(counts, key=counts.get)

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
                # "Mexico's Zapatista rebel group" -> "'s Zapatista rebel group"
                trimmed_text = re.sub(r"^'s\b", "", trimmed_text).strip()
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
            

