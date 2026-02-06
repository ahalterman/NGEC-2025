from importlib import resources
import logging
import os
from pathlib import Path
import pickle
import re

import numpy as np
import platformdirs
from scipy.spatial.distance import cdist

from .common import ModelManager, TextPreProcessor, clean_query, CountryDetector


logger = logging.getLogger(__name__)


# Constants for decision making
THRESHOLD_COSINE_SIMILARITY = 0.6
THRESHOLD_DOT_SIMILARITY = 45
THRESHOLD_ALT_NAME_TITLE_MATCH = 0.8
THRESHOLD_CONTEXT_MATCH = 0.6 # 0.7

# TODO #26: allow overriding assets/models
# TODO not sure if strip_ents works correctly, and where the nlp model should be loaded if needed
# TODO get rid of base_path


def get_cache_path():
    """
    Get platform-appropriate cache directory for computed embeddings.

    Returns:
        Path: Cache directory path (e.g., ~/.cache/plover/ on Linux/Mac)
    """
    cache_dir = Path(platformdirs.user_cache_dir("plover", "ngec"))
    # Create embeddings subdirectory
    embeddings_cache = cache_dir / "embeddings"
    embeddings_cache.mkdir(parents=True, exist_ok=True)
    return embeddings_cache


#######################################################
# Agent Matching
#######################################################

class AgentMatcher:
    """
    Match text to agent patterns using transformer models.
    
    This class provides methods for matching text to PLOVER agent patterns
    using transformer embeddings and optional classifier-based matching.

    Parameters
    ----------
    trf_model : SentenceTransformer, optional
        Sentence transformer model for encoding text. If None, will be 
        loaded from ModelManager.
    agents_file : None | str | Path, default None
        Path to a PLOVER/CAMEO agents file containing agent patterns. The
        default None will use the PLOVER agents file included in the package.
    cache_path: None | str | Path, optional
        Path to cache directory for computed embeddings (default: platform cache dir)
    device : str, optional
        Device to use for model inference ('cuda' or None for CPU).
    text_processor : TextPreProcessor, optional
        TextPreProcessor instance for text cleaning. If None, 
        a new instance will be created.


    Attributes
    ----------
    device : str
        Device used for model inference.
    agents : list of dict
        Loaded and cleaned agent patterns.
    trf : SentenceTransformer
        Transformer model for encoding text.
    text_processor : TextPreProcessor
        Text preprocessing utilities.
    trf_matrix : numpy.ndarray
        Pre-computed embeddings for agent patterns.
    
        
    Examples
    --------
    >>> matcher = AgentMatcher()
    >>> match = matcher.trf_agent_match("Chancellor", country="DEU")
    >>> print(match['code_1'])
    'GOV'
    """
    
    def __init__(self,
                 trf_model=None,
                 agents_file: None | str | Path = None,
                 cache_path=None,
                 device=None,
                 text_processor=None,
                 ):

        self.device = device

        if agents_file is None:
            self.agents_file = str(resources.files("ngec.assets") / "PLOVER_agents.txt")
        else:
            self.agents_file = agents_file
        
        self.cache_path = cache_path if cache_path is not None else get_cache_path()

        # Initialize resources
        if trf_model is None:
            model_manager = ModelManager(device)
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

        with open(self.agents_file, "r", encoding="utf-8") as f:
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

        Embeddings are cached in a user-writable cache directory to avoid modifying
        installed package files. The cache key is based on the agents file content
        hash and filename, allowing multiple agent files to coexist.

        Returns:
            numpy.ndarray: Matrix of agent pattern embeddings
        """
        # Read agents file and compute hash for cache key
        with open(self.agents_file, "r", encoding="utf-8") as f:
            data = f.read()
        current_hash = hash(data)

        # Create cache directory based on hash and filename
        # This allows different agent files to have separate caches
        cache_key = f"{current_hash}_{Path(self.agents_file).stem}"
        cache_dir = Path(self.cache_path) / cache_key
        cache_file = cache_dir / "bert_matrix.pkl"

        # Check if cached embeddings exist and are valid
        if cache_file.exists():
            logger.info(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Cache miss - compute embeddings
        logger.info(f"Computing embeddings for {len(self.agents)} agent patterns from {self.agents_file}")
        logger.info("This is a one-time computation that will be cached for future use")
        patterns = [agent['pattern'] for agent in self.agents]
        trf_matrix = self.trf.encode(patterns, show_progress_bar=False, device=self.device)

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(trf_matrix, f)
        logger.info(f"Embeddings cached to {cache_file}")

        return trf_matrix

    def trf_agent_match(self, text, country="", method="cosine", threshold=THRESHOLD_COSINE_SIMILARITY):
        """
        Compare input text to the agent file using classifier or similarity matching.

        Args:
            text: Text to match against agent patterns
            country: Country code to include in the result
            method: Similarity method to use ('cosine' or 'dot') - only used if classifier unavailable
            threshold: Confidence threshold below which matches are ignored

        Returns:
            dict or None: Match information or None if no match above threshold
        """
        # Handle empty inputs
        if country is None:
            country = ""
        text = clean_query(text)
        if not text:
            return None

        # Get embedding (needed for both classifier and similarity)
        query_emb = self.trf.encode(text, show_progress_bar=False)

        # SIMILARITY APPROACH (original, used as fallback)
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

        # Compute similarity
        if method == "dot":
            sims = np.dot(self.trf_matrix, query_emb.T)
        else:  # cosine
            sims = 1 - cdist(self.trf_matrix, np.expand_dims(query_emb.T, 0), metric="cosine")

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
        match['source'] = 'similarity'

        logger.debug(f"Similarity match: '{text}' -> {match['code_1']} (conf={max_sim:.3f})")
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
            country_detector = CountryDetector()
            
        # Extract country and clean text
        country, trimmed_text = country_detector.search_nat(text)
        trimmed_text = clean_query(trimmed_text)
        
        # Optionally strip entities
        if strip_ents:
            try:
                model_manager = ModelManager(self.device)
                doc = self.nlp(text)
                trimmed_text = self.text_processor.strip_ents(doc)
            except IndexError:
                # If NLP fails, continue with trimmed_text as-is
                pass
                
            if trimmed_text == "s":
                return None
                
        # Match against agent patterns
        return self.trf_agent_match(trimmed_text, country=country, threshold=threshold)

