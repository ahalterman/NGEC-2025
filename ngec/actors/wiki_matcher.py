"""
Functionality for matching a query term to a Wikipedia article. Relies on a
Elasticsearch index of Wikipedia data.
"""

from importlib import resources
import logging
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

from .common import ModelManager, clean_query

# Constants
THRESHOLD_NEURAL_TITLE_MATCH = 0.9
THRESHOLD_COMBINED_SCORE = 9 

logger = logging.getLogger(__name__)

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
                        use_importance=False,
                        title_exact_boost=250,
                        title_fuzzy_boost=50,
                        redirects_exact_boost=100,
                        redirects_fuzzy_boost=50,
                        alternative_names_boost=100,
                        alternative_names_fuzzy_boost=20,
                        short_desc_boost=10,
                        intro_para_boost=5,
                        ):
        """
        Enhanced search with importance scoring compatible with ES 7.10.1
        """

        # Base matching clauses
        base_should_clauses = [
            # Exact matches (highest priority)
            {"term": {"title": {"value": query_term, "boost": title_exact_boost}}},
            {"match": {"title": {"query": query_term, "boost": title_fuzzy_boost}}},
            {"term": {"redirects": {"value": query_term, "boost": redirects_exact_boost}}},
            {"match": {"redirects": {"query": query_term, "boost": redirects_fuzzy_boost}}},
            {"term": {"alternative_names": {"value": query_term, "boost": alternative_names_boost}}},
            {"match": {"alternative_names": {"query": query_term, "boost": alternative_names_fuzzy_boost}}},
            {"match": {"intro_para": {"query": query_term, "boost": intro_para_boost}}},
            {"match": {"short_desc": {"query": query_term, "boost": short_desc_boost}}}
        ]

        if use_importance:
            # Add importance signals as range queries
            importance_clauses = [
                # Boost articles with many redirects (indicates importance)
                {"range": {"redirects": {"gte": ["5"], "boost": 5}}},
                {"range": {"redirects": {"gte": ["10"], "boost": 30}}},
                {"range": {"redirects": {"gte": ["20"], "boost": 100}}},

                # Boost articles with many categories
                {"range": {"categories": {"gte": ["5"], "boost": 3}}},
                {"range": {"categories": {"gte": ["10"], "boost": 50}}},

                # Boost articles with alternative names
                {"range": {"alternative_names": {"gte": ["2"], "boost": 2}}},
                {"range": {"alternative_names": {"gte": ["5"], "boost": 10}}},

                # Boost articles with infoboxes (if you have this field)
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
            text_processor: TextPreProcessor instance
        """
        if wiki_client is None:
            self.wiki_client = WikiClient(es_client=es_client)
        else:
            self.wiki_client = wiki_client
    
    def search_wiki(self, query_term, limit_term="", max_results=200):
        """
        Search Wikipedia for a given query term.
        
        Args:
            query_term: Term to search for
            limit_term: Term to limit results by
            max_results: Maximum number of results to return
            fields: Fields to search in
            
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

    TODO #26: allow overriding models; need to check this works correctly as implemented
    
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
        score_df['is_predicted_match'] = score_df['is_max_for_task'] & (score_df['ranker_score'] > 0.5)
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
            expanded_query = [i.text for i in context_doc.ents if query_term in i.text]
            if expanded_query:
                # take the first one
                expanded_query = expanded_query[0]
                if len(expanded_query) > len(query_term):
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

    def query_wiki(self, 
                   query_term, 
                   limit_term="", 
                   country="", 
                   context="", 
                   actor_desc="",
                   method="neural",
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
        if method not in ["neural", "rules"]:
            raise ValueError(f"Wiki selection method must be 'neural' or 'rules'. You provided: {method}")
        # Do NER expansion by default
        if context:
            logger.debug("Context present, so attempting NER expansion")
            query_term = self._expand_query(query_term, context)

        # Try exact search first
        logger.debug("Starting with exact search")
        results = self.wiki_searcher.search_wiki(
            query_term, 
            limit_term=limit_term, 
            max_results=max_results,
        )
        best = self.pick_best_wiki(
            query_term, 
            results, 
            country=country, 
            context=context,
            actor_desc=actor_desc,
            wiki_sort_method=method
        )
        if best:
            return best
            
        # Fall back to fuzzy search
        logger.debug("Falling back to fuzzy search")
        results = self.wiki_searcher.search_wiki(
            query_term, 
            limit_term=limit_term, 
            max_results=max_results
        )
        best = self.pick_best_wiki(
            query_term, 
            results, 
            country=country, 
            context=context,
            actor_desc=actor_desc,
            wiki_sort_method=method,
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
    