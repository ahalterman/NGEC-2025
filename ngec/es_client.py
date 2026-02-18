
from dataclasses import asdict, dataclass
from elasticsearch import Elasticsearch, TransportError
from typing import Any
from urllib3.exceptions import NewConnectionError


def setup_es_client(hosts: str | list[str] | None | Any = "localhost",
                    port: int | None = 9200,
                    **kwargs) -> Elasticsearch:
    """Helper class to setup a Elasticsearch client connection
    
    All arguments are passed on to elasticsearch.Elasticsearch() as is. 

    If you are using the default package setup instructions with a local 
    Elasticsearch instance, you likely do not need to change anything here.

    Otherwise, the keys here correspond to arguments that are passed to 
    elasticsearch.Elasticsearch(), as is. See the official documentation for 
    more details on the different ways setup and connect to ES: 
    
    https://www.elastic.co/docs/reference/elasticsearch/clients/python/connecting

    Note: to pass username and password for authentication, use the http_auth argument, e.g.
    `http_auth=(ES_USER, ES_PASSWORD)`
    """
    es = Elasticsearch(hosts=hosts, port=port, **kwargs)
    return es


@dataclass
class ESConfig():
    """
    Elasticsearch configuration dictionary. 

    If you are using the default package setup instructions with a local 
    Elasticsearch instance, you likely do not need to change anything here.

    Otherwise, the keys here correspond to arguments that are passed to 
    elasticsearch.Elasticsearch(), as is. See the official documentation for 
    more details on the different ways setup and connect to ES: 
    
    https://www.elastic.co/docs/reference/elasticsearch/clients/python/connecting
    """
    hosts: str | None | Any = "localhost"  # Any because ES accepts a wild range of inputs for this
    port: int | None = 9200


def es_is_available(es_config: ESConfig | dict) -> tuple[bool, str]:
    """Check if Elasticsearch is available with the given configuration.
    
    Returns:
        tuple[bool, str]: (is_available, message)
    """
    if isinstance(es_config, ESConfig):
        es_config = asdict(es_config)

    try:
        client = Elasticsearch(**es_config)
        
        # Test connection
        if not client.ping():
            return False, f"Ping failed to {es_config.get("hosts")}:{es_config.get("port")}"
        
        # Get cluster info
        info = client.info()
        cluster_name = info.get("name", "unknown")
        version = info.get("version", {}).get("number", "unknown")
        
        client.close()
        return True, f"Connected to '{cluster_name}' (v{version}) at {es_config.get("hosts")}:{es_config.get("port", 9200)}"
        
    except (NewConnectionError, ConnectionRefusedError) as e:
        return False, f"Connection failed to {es_config.get("hosts")}:{es_config.get("port", 9200)}: {str(e)}"
    except TransportError as e:
        return False, f"Authentication/transport error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
