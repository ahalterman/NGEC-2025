
import logging

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

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
                 es_config=None):
        """Initialize the Wikipedia client.

        es_config is a dict with es_host, es_port, es_user, es_password
        """
        if not es_config:
            es_config = {
                "es_host": "localhost",
                "es_port": 9200,
                "es_user": None,
                "es_password": None
            }
        # fill in any missing keys with the default
        for key, default in [("es_host", "localhost"), ("es_port", 9200), ("es_user", None), ("es_password", None)]:
            if key not in es_config:
                es_config[key] = default

        self.conn = self.setup_es(
            es_host=es_config["es_host"],
            es_port=es_config["es_port"],
            es_user=es_config["es_user"],
            es_password=es_config["es_password"]
        )
        self.check_wiki(self.conn)

    def setup_es(self, 
                 es_host="localhost", 
                 es_port=9200, 
                 es_user=None, 
                 es_password=None):
        """
        Establish connection to Elasticsearch and return search object.
        
        Returns:
            Search: Elasticsearch search object
        
        Raises:
            ConnectionError: If Elasticsearch connection fails
        """
        try:
            client = Elasticsearch(hosts=[{'host': es_host,
                                           'port': es_port}],
                                   http_auth=(es_user, es_password) if es_user and es_password else None
                                   )
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

    #def run_wiki_search(self, query_term, limit_term="", fuzziness="AUTO", max_results=200,
    #               fields=['title^50', 'redirects^50', 'alternative_names'],
    #               score_type="best_fields"):
    #    """
    #    Search Wikipedia for a given query term.
    #    
    #    Args:
    #        query_term: Term to search for
    #        limit_term: Term to limit results by
    #        fuzziness: Elasticsearch fuzziness parameter
    #        max_results: Maximum number of results to return
    #        fields: Fields to search in
    #        score_type: Elasticsearch score type
    #        
    #    Returns:
    #        list: List of Wikipedia article dictionaries
    #    """
    #    # Construct query
    #    if not limit_term:
    #        query = {
    #            "bool": {
    #                "should": [
    #                    # Exact match on title (case-sensitive)
    #                    {"term": {"title": {"value": query_term, "boost": 150}}},
    #                    # Analyzed match on title (case-insensitive, tokenized)
    #                    {"match": {"title": {"query": query_term, "boost": 50}}},

    #                    # Exact match on redirects (for acronyms)
    #                    {"term": {"redirects": {"value": query_term, "boost": 150}}},
    #                    # Analyzed match on redirects
    #                    {"match": {"redirects": {"query": query_term, "boost": 50}}},

    #                    # Exact match on alternative names
    #                    {"term": {"alternative_names": {"value": query_term, "boost": 125}}},
    #                    # Analyzed match on alternative names
    #                    {"match": {"alternative_names": {"query": query_term, "boost": 50}}},

    #                    # Analyzed match on short description and intro para
    #                    {"match": {"intro_para": {"query": query_term, "boost": 5}}},
    #                    # Analyzed match on categories
    #                    {"match": {"short_desc": {"query": query_term, "boost": 10}}}
    #                ]
    #            }
    #        }
    #    else:
    #        # Include limit term in query
    #        limit_fields = [
    #            "title^100", "redirects^100", "alternative_names",
    #            "intro_para", "categories", "infobox"
    #        ]
    #        query = {
    #            "bool": {
    #                "must": [
    #                    {
    #                        "multi_match": {
    #                            "query": query_term,
    #                            "fields": fields,
    #                            "type": score_type,
    #                            "fuzziness": fuzziness,
    #                            "operator": "and"
    #                        }
    #                    },
    #                    {
    #                        "multi_match": {
    #                            "query": limit_term,
    #                            "fields": limit_fields,
    #                            "type": "most_fields"
    #                        }
    #                    }
    #                ]
    #            }
    #        }
    #    
    #    # Execute search
    #    res = self.conn.query(query)[0:max_results].execute()
    #    results = [hit.to_dict()['_source'] for hit in res['hits']['hits']]
    #    # Add the scores to the results to use for ranking?
    #    #scores = [hit.to_dict()['_score'] for hit in res['hits']['hits']]
    #    #for i, result in enumerate(results):
    #    #    result['raw_es_score'] = scores[i]
    #    logger.debug(f"Number of hits for Wiki query: {len(results)}")
    #    logger.debug(f"Titles of the first five results: {[result['title'] for result in results[0:5]]}")
    #    
    #    return results
