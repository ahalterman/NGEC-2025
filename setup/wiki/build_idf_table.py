"""
Build `ngec/assets/wiki_idf.json.gz`, the inverse-document-frequency table the
Wikipedia ranker uses.

One of the ranker's features is the TF-IDF cosine between the news story and a
candidate article's intro paragraph. That needs to know how rare each word is
across Wikipedia, which is what this script measures: it draws a random sample
of articles from the Elasticsearch index, counts in how many of them each word
appears, and writes out the resulting weights.

The sample is drawn with Elasticsearch's `random_score` and a fixed seed, so
re-running this against the same index gives the same table.

Usage:

    python setup/wiki/build_idf_table.py
    python setup/wiki/build_idf_table.py --sample-size 50000 --output /tmp/idf.json.gz

Takes a few minutes for the default 200,000-article sample.
"""

import argparse
import gzip
import json
import logging
import math
from pathlib import Path

from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# The runtime and this script must tokenize the same way, so the pattern is
# imported from the module that uses the table rather than copied.
from ngec.actors.wiki_matcher import TFIDF_TOKEN_PATTERN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "ngec" / "assets" / "wiki_idf.json.gz"


def sample_intro_paragraphs(es: Elasticsearch, sample_size: int, seed: int,
                            page_size: int = 2000):
    """
    Yield `sample_size` intro paragraphs drawn at random from the wiki index.

    `random_score` with a fixed seed gives a reproducible random ordering; the
    scroll API then walks it. The `field` argument to `random_score` is what
    makes the ordering stable across runs (without it Elasticsearch reseeds per
    segment), and `_seq_no` is the recommended choice.
    """
    query = {
        "function_score": {
            "query": {"exists": {"field": "intro_para"}},
            "random_score": {"seed": seed, "field": "_seq_no"},
        }
    }
    response = es.search(index="wiki", query=query,
                         _source=["intro_para"], size=page_size, scroll="10m")
    scroll_id = response["_scroll_id"]
    seen = 0
    try:
        while response["hits"]["hits"] and seen < sample_size:
            for hit in response["hits"]["hits"]:
                intro = hit["_source"].get("intro_para", "")
                if intro:
                    yield intro
                seen += 1
                if seen >= sample_size:
                    return
            if seen % 50000 < page_size:
                logger.info(f"  sampled {seen} articles")
            response = es.scroll(scroll_id=scroll_id, scroll="10m")
            scroll_id = response["_scroll_id"]
    finally:
        es.clear_scroll(scroll_id=scroll_id, ignore=(404,))


def count_document_frequencies(intros) -> tuple[dict, int]:
    """
    How many sampled articles each word appears in.

    Words are lowercase and alphabetic (`TFIDF_TOKEN_PATTERN`); English stop
    words are dropped, since they appear everywhere and carry no signal.
    """
    document_frequency = {}
    n_documents = 0
    for intro in intros:
        n_documents += 1
        words = set(TFIDF_TOKEN_PATTERN.findall(intro.lower()))
        for word in words - ENGLISH_STOP_WORDS:
            document_frequency[word] = document_frequency.get(word, 0) + 1
    return document_frequency, n_documents


def build_idf(document_frequency: dict, n_documents: int,
              min_df: int = 2, max_terms: int = 100_000) -> dict:
    """
    Turn document frequencies into IDF weights, keeping the commonest terms.

    Words seen in only one article are dropped (they are mostly typos and
    one-off names, and there are millions of them), and the table is then capped
    at the `max_terms` most widely-seen words so the shipped asset stays small.
    The formula is scikit-learn's smoothed IDF, `ln((1 + n) / (1 + df)) + 1`.
    """
    kept = [(word, df) for word, df in document_frequency.items() if df >= min_df]
    kept.sort(key=lambda pair: (-pair[1], pair[0]))
    kept = kept[:max_terms]
    return {word: math.log((1 + n_documents) / (1 + df)) + 1 for word, df in kept}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--es-host", default="localhost")
    parser.add_argument("--es-port", type=int, default=9200)
    parser.add_argument("--sample-size", type=int, default=200_000,
                        help="how many articles to sample (default 200,000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random_score seed; changing it changes the sample")
    parser.add_argument("--min-df", type=int, default=2,
                        help="drop words seen in fewer than this many articles")
    parser.add_argument("--max-terms", type=int, default=100_000,
                        help="keep at most this many words, the most widely-seen ones")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    es = Elasticsearch(hosts=args.es_host, port=args.es_port, timeout=120)
    if not es.ping():
        raise ConnectionError(f"No Elasticsearch at {args.es_host}:{args.es_port}")

    logger.info(f"Sampling {args.sample_size} intro paragraphs (seed {args.seed})")
    intros = sample_intro_paragraphs(es, args.sample_size, args.seed)
    document_frequency, n_documents = count_document_frequencies(intros)
    logger.info(f"{n_documents} articles sampled, {len(document_frequency)} distinct words")

    idf = build_idf(document_frequency, n_documents,
                    min_df=args.min_df, max_terms=args.max_terms)
    logger.info(f"Keeping {len(idf)} words")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = {"idf": idf, "sample_size": n_documents, "seed": args.seed}
    with gzip.open(args.output, "wt", encoding="utf-8") as f:
        json.dump(table, f)
    logger.info(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
