"""
Build the offline Wikipedia Elasticsearch index used by the actor resolver.

This is the single entry point for the wiki index pipeline. It runs in three
stages, in order:

    1. build_links  -- parse the dump and collect every page redirect into a
                       pickle (data/redirect_dict.pkl).
    2. load_redis   -- load that redirect dict into Redis so stage 3 can attach
                       a "redirects" field to each article.
    3. load_es      -- parse the dump again and bulk-load formatted articles
                       into the "wiki" index.

Run each stage from this folder, e.g.:

    uv run --group es-build python load_wiki_es.py build_links data/enwiki-latest-pages-articles.xml.bz2
    uv run --group es-build python load_wiki_es.py load_redis  data/enwiki-latest-pages-articles.xml.bz2
    uv run --group es-build python load_wiki_es.py load_es     data/enwiki-latest-pages-articles.xml.bz2

The "wiki" index lives in the *same* Elasticsearch data directory as the
"geonames" index (a single ES node stores all indices together). The loader
only ever touches the "wiki" index, so the co-resident "geonames" index is left
intact. To *refresh* an existing wiki index, pass --drop to `load_es`: it
records the before-stats, deletes the old index, then creates and loads the new
one, so the before/after comparison in the log is meaningful. Without --drop,
new documents are merged into the existing index. See README.md.

Elasticsearch and Redis hosts can be overridden with the NGEC_ES_URL and
NGEC_REDIS_HOST environment variables (defaults: http://localhost:9200/ and
localhost). Redis is only used at build time -- the runtime does not need it.

Supersedes the older setup/wiki/load_wiki_es.py.
"""

import bz2
import datetime
import json
import logging
import multiprocessing
import os
import pickle
import re
import subprocess
import time

import elasticsearch
import mwparserfromhell
import plac
import redis
from elasticsearch import Elasticsearch, helpers
from lxml import etree
from textacy.preprocessing.remove import accents as remove_accents
from tqdm import tqdm

# Checkpoint cadence for long-running dump scans (build_links).
CHECKPOINT_EVERY = 1_000_000

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

es_logger = elasticsearch.logger
es_logger.setLevel(elasticsearch.logging.WARNING)

# Build-time service locations. Both default to the ports published by
# elasticsearch/compose-build.yml, but can be overridden for other setups.
ES_URL = os.environ.get("NGEC_ES_URL", "http://localhost:9200/")
REDIS_HOST = os.environ.get("NGEC_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("NGEC_REDIS_PORT", "6379"))

# Files written by build_links / read by load_redis live alongside the dump.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REDIRECT_DICT_PATH = os.path.join(DATA_DIR, "redirect_dict.pkl")
MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki_mapping.json")

# Approximate page count for English Wikipedia, used for the progress bar only.
ESTIMATED_PAGES = 25_700_000

REDIRECT_PATTERN = re.compile(r"#?(REDIRECT|redirect|Redirect)")

# Everything the indexer keeps from an article's body -- the intro paragraph,
# the bold alternative names, the short description and the infobox -- comes
# from the lead section, i.e. the text before the article's first heading.
# Handing mwparserfromhell the whole article to get it is where essentially all
# of load_es's CPU goes (see README, "Speed"), so cut the wikitext at the first
# heading and parse only that prefix. A heading is a line that starts with two
# or more "="; `re.M` lets it match at the very start of the text too.
#
# This is a plain text scan, so unlike mwparserfromhell it does not know about
# <!-- comments --> or <nowiki>: a "==heading==" line hidden inside one of those
# ends the lead here but would not in MediaWiki. On the 3,090-page benchmark
# slice that never happened; what did happen, five times, is the opposite --
# mwparserfromhell missed real headings and returned the *whole article* as the
# lead. See README.
LEAD_CUT_PATTERN = re.compile(r"^={2,}[^=]", re.M)

# Categories are the one field that lives below the lead, at the end of the
# article. Rather than parse the rest of the wikitext to reach them, find them
# in the raw text: [[Category:Name]], the lowercase [[category:Name]] spelling,
# and the [[Category:Name|sortkey]] form. The name is captured exactly as
# written, with no whitespace normalisation, because that is what the previous
# strip_code()-based extraction stored.
CATEGORY_PATTERN = re.compile(r"\[\[[Cc]ategory:([^\]|]*)")


# ---------------------------------------------------------------------------
# Dump parsing
# ---------------------------------------------------------------------------
def _open_dump_with_namespace(dump_file):
    """Open the dump (bz2 or plain) and detect its XML namespace from a head
    sample. Returns (file_obj, namespace_prefix), where namespace_prefix is
    already wrapped in "{...}" for use with etree tag lookups."""
    if dump_file.endswith(".bz2"):
        file_obj = bz2.BZ2File(dump_file, "rb")
        logger.info(f"Opened {dump_file} as BZ2 file")
    else:
        file_obj = open(dump_file, "rb")
        logger.info(f"Opened {dump_file} as regular file")

    logger.info("Detecting XML namespace...")
    sample = file_obj.read(10000)
    file_obj.seek(0)
    ns_match = re.search(rb'xmlns="(http://www.mediawiki.org/xml/export-[^"]+)"', sample)
    if ns_match:
        namespace = ns_match.group(1).decode("utf-8")
        logger.info(f"Detected namespace: {namespace}")
    else:
        namespace = "http://www.mediawiki.org/xml/export-0.10/"
        logger.info(f"No namespace detected, using default: {namespace}")

    return file_obj, "{" + namespace + "}"


def iterate_wiki_pages(dump_file):
    """
    A generator that efficiently parses a Wikipedia XML dump and yields
    (title, text) tuples.

    Args:
        dump_file (str): Path to the Wikipedia XML dump (.xml or .xml.bz2)

    Yields:
        tuple: (title, text) pairs for each page in the dump
    """
    file_obj, ns = _open_dump_with_namespace(dump_file)
    page_tag = f"{ns}page"
    title_tag = f"{ns}title"
    revision_tag = f"{ns}revision"
    text_tag = f"{ns}text"

    # iterparse keeps memory bounded by clearing elements as we go.
    logger.info("Starting to parse XML...")
    context = etree.iterparse(file_obj, events=("end",), tag=page_tag)

    for event, elem in context:
        try:
            title_elem = elem.find(f".//{title_tag}")
            if title_elem is None:
                continue
            title = title_elem.text

            revision = elem.find(f".//{revision_tag}")
            if revision is None:
                continue

            text_elem = revision.find(f".//{text_tag}")
            if text_elem is None:
                continue

            text = text_elem.text or ""
            yield (title, text)
        except Exception as e:
            current = title if "title" in locals() else "unknown"
            logger.warning(f"Error processing element for title '{current}': {e}")
        finally:
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    file_obj.close()
    logger.info("Finished parsing XML")


def iterate_page_redirects(dump_file, page_counter=None):
    """
    A generator that scans a Wikipedia XML dump and yields (source_title,
    target_title) for each page that is a redirect, reading the
    <redirect title="..."> attribute directly off the <page> element. Unlike
    iterate_wiki_pages, this never descends into <revision>/<text>, so it
    doesn't pay to decompress/parse every non-redirect page's full wikitext.

    This replaces an older approach that ran every page's wikitext through
    mwparserfromhell and matched a redirect regex. Benchmarking the two (plus
    mwxml's page.redirect) found the lxml-attribute read to be faster, to use
    far less memory, and to be more correct: the regex approach both missed
    genuine redirects and misclassified some non-redirect pages.

    If given, page_counter[0] is updated on every page scanned (redirect or
    not), so callers can report progress/checkpoints against the true page
    count rather than just the redirect count.
    """
    file_obj, ns = _open_dump_with_namespace(dump_file)
    page_tag = f"{ns}page"
    title_tag = f"{ns}title"
    redirect_tag = f"{ns}redirect"

    logger.info("Starting to parse XML for redirects...")
    context = etree.iterparse(file_obj, events=("end",), tag=page_tag)

    for i, (_, elem) in enumerate(context):
        if page_counter is not None:
            page_counter[0] = i + 1
        redirect_elem = elem.find(redirect_tag)
        if redirect_elem is not None:
            title_elem = elem.find(title_tag)
            target = redirect_elem.get("title")
            if title_elem is not None and target:
                yield title_elem.text, target
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    file_obj.close()
    logger.info("Finished parsing XML")


# ---------------------------------------------------------------------------
# Stage 1: build_links -- collect page redirects
# ---------------------------------------------------------------------------
def build_links(file):
    """Parse the dump and write all page redirects to REDIRECT_DICT_PATH."""
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("Building redirect link dictionary...")
    all_redirects = {}
    page_counter = [0]
    last_checkpoint = 0

    with tqdm(total=ESTIMATED_PAGES) as progress:
        for title, target in iterate_page_redirects(file, page_counter=page_counter):
            all_redirects.setdefault(target, set()).add(title)
            progress.update(page_counter[0] - progress.n)

            # Periodic checkpoint so a long run can be resumed/inspected.
            if page_counter[0] - last_checkpoint >= CHECKPOINT_EVERY:
                last_checkpoint = page_counter[0]
                with open(REDIRECT_DICT_PATH, "wb") as f:
                    pickle.dump({k: sorted(v) for k, v in all_redirects.items()}, f)
                logger.info(f"Checkpoint at {page_counter[0]} pages, {len(all_redirects)} redirect targets")

    result = {k: sorted(v) for k, v in all_redirects.items()}
    with open(REDIRECT_DICT_PATH, "wb") as f:
        pickle.dump(result, f)
    logger.info(
        f"Wrote {REDIRECT_DICT_PATH}: {page_counter[0]} pages, {len(result)} redirect targets"
    )


# ---------------------------------------------------------------------------
# Stage 2: load_redis -- load redirect dict into Redis
# ---------------------------------------------------------------------------
def read_clean_redirects():
    """Load the redirect dict and merge case variants into their standard form."""
    with open(REDIRECT_DICT_PATH, "rb") as f:
        redirect_dict = pickle.load(f)

    del_list = []
    for k in list(redirect_dict.keys()):
        if k.lower() in redirect_dict and k.lower() != k:
            # sorted(), not list(), so the dict written to Redis is reproducible.
            redirect_dict[k] = sorted(set(redirect_dict[k] + redirect_dict[k.lower()]))
            del_list.append(k.lower())
    for d in del_list:
        redirect_dict.pop(d, None)
    return redirect_dict


def load_redis():
    """Load the cleaned redirect dict into Redis for stage 3."""
    logger.info(f"Reading redirect dict from {REDIRECT_DICT_PATH}...")
    redirect_dict = read_clean_redirects()
    redis_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    pipe = redis_db.pipeline()
    for n, (k, v) in tqdm(enumerate(redirect_dict.items()), total=len(redirect_dict)):
        pipe.set(k, ";".join(v))
        if n % 1000 == 0:
            pipe.execute()
    pipe.execute()
    logger.info(f"Loaded {len(redirect_dict)} redirect targets into Redis")


# ---------------------------------------------------------------------------
# Stage 3: load_es -- parse and bulk-load articles into the "wiki" index
# ---------------------------------------------------------------------------
def clean_names(name_list):
    """Strip wiki markup from a list of names and add de-accented variants."""
    if not name_list:
        return []
    name_list = [re.sub(r"\|.+?\]\]", "", i).strip() for i in name_list]
    name_list = [re.sub(r"\[|\]", "", i).strip() for i in name_list]
    # Drop weird leftovers like "son:"
    name_list = [i for i in name_list if not i.endswith(":")]
    de_accent = [remove_accents(i) for i in name_list]
    # sorted(), not list(): Python randomises string hashing per process, so
    # list(set(...)) would order these differently on every run and two builds
    # of the same dump would never be byte-identical. See README.
    return sorted(set(name_list + de_accent))


# Titles for non-article namespaces / maintenance pages we never want to index.
_SKIP_TITLE_PREFIXES = (
    "Peer review/",
    "Requests for adminship/",
    "Featured list candidates/",
    "Sockpuppet investigations/",
)


def _should_skip_title(title):
    if title.endswith(".jpg") or title.endswith(".png"):
        return True
    if re.search(r"\-stub", title):
        return True
    if re.match(r"(User|Selected anniversaries)", title):
        return True
    if re.search(r"\([Dd]isambiguation\)", title):
        return True
    if re.search(r"Articles for deletion", title):
        return True
    if re.match(r"List ", title):
        return True
    if re.match(r"Portal ", title):
        return True
    if re.search(r"Today's featured article", title):
        return True
    if re.search(r"Featured article candidates", title):
        return True
    if re.match(r"Categories for", title):
        return True
    if title.startswith(_SKIP_TITLE_PREFIXES):
        return True
    return False


# One Redis client per process, built on first use.
#
# parse_wiki_article used to construct a redis.StrictRedis inside the function,
# so a full run built one throwaway client per article -- ~25 million of them.
# It runs inside a multiprocessing worker, so a client created at import time
# (i.e. before the fork) would be shared by every worker, which redis-py does
# not support. Creating it lazily gives each worker exactly one client and
# leaves the lookups themselves unchanged.
_redis_db = None


def _redis_client():
    global _redis_db
    if _redis_db is None:
        _redis_db = redis.StrictRedis(
            host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
        )
    return _redis_db


def parse_wiki_article(title=None, text=None, use_redis=True):
    """
    Format a single Wikipedia article into the document structure the actor
    resolver expects, pulling out:

    - title
    - short_desc (the Wikidata-style short description)
    - intro_para (first paragraph, markup stripped)
    - alternative_names (bold names in the intro + infobox name fields)
    - redirects (from Redis)
    - infobox / box_type
    - affiliated_people (infobox leaders / founders)
    - categories
    - redirect_count (how many redirects the page has)

    Returns None for pages that should be skipped (redirects, disambiguation
    pages, maintenance pages, etc.).

    Only the article's lead section is handed to mwparserfromhell; the
    categories are read out of the raw wikitext with a regex. See
    LEAD_CUT_PATTERN and CATEGORY_PATTERN.
    """
    if not title or not text:
        return None
    if _should_skip_title(title):
        logger.debug(f"Skipping non-article title: {title}")
        return None

    text = str(text)
    # Parse only the lead section (see LEAD_CUT_PATTERN). get_sections()[0] is
    # still what defines the lead, exactly as before -- it just now runs over a
    # few kilobytes instead of the whole article.
    cut = LEAD_CUT_PATTERN.search(text)
    lead_wikitext = text[: cut.start()] if cut else text

    wikicode = mwparserfromhell.parse(lead_wikitext)
    raw_intro = wikicode.get_sections()[0]
    intro_para = raw_intro.strip_code()
    # Remove stray links/thumbs/parentheses that slip through strip_code().
    intro_para = re.sub(r"(\[\[.+?\]\])", "", intro_para).strip()
    intro_para = re.sub(r"^thumb\|.+?\n", "", intro_para)
    intro_para = re.sub(r"^thumb\|.+?\n", "", intro_para)
    intro_para = re.sub(r"\(.+?\)", "", intro_para, count=1)

    if not intro_para:
        logger.debug(f"No intro para for {title}.")
        return None
    if re.match(REDIRECT_PATTERN, intro_para):
        logger.debug(f"Detected redirect in first para: {title}")
        return None
    if re.search(r"\*?\n?Category\:", intro_para) or intro_para.startswith("Category:"):
        logger.debug(f"Category page: {title}")
        return None
    if intro_para.startswith("<noinclude>"):
        logger.debug(f"Sneaky category? {title}")
        return None
    head = intro_para[0:100]
    if re.search(r"may refer to", head) or re.search(r"most often refers", head):
        logger.debug(f"Disambiguation-like intro: {title}")
        return None
    if re.search(r"most commonly refers", head) or re.search(r"[Pp]ortal\:", head):
        logger.debug(f"Disambiguation/portal intro: {title}")
        return None

    alternative_names = re.findall(r"'''(.+?)'''", str(raw_intro))

    redirects = []
    if use_redis:
        redirects = _redis_client().get(title)
        redirects = redirects.split(";") if redirects else []

    try:
        short_desc = re.findall(
            r"\{\{[Ss]hort description\|(.+?)\}\}", str(raw_intro)
        )[0].strip()
    except IndexError:
        short_desc = ""

    params = {
        "title": title,
        "short_desc": short_desc,
        "intro_para": intro_para.strip(),
        "alternative_names": clean_names(alternative_names),
        "redirects": clean_names(redirects),
        "affiliated_people": [],
        "box_type": None,
    }
    # How many pages redirect here, after clean_names() has deduplicated them.
    # A cheap proxy for how prominent an article is. Nothing reads it yet; it is
    # stored now so the wiki ranker's importance feature (use_importance in
    # ngec/actors/wiki_matcher.py) can be switched on without another rebuild.
    params["redirect_count"] = len(params["redirects"])

    for template in raw_intro.filter_templates():
        if re.search(r"[Ii]nfobox", template.name.strip()):
            params["infobox"] = {
                p.name.strip(): p.value.strip_code().strip() for p in template.params
            }
            params["box_type"] = re.sub(r"Infobox", "", str(template.name)).strip()
            break

    if "infobox" in params:
        for k in ["name", "native_name", "other_name", "alias", "birth_name", "nickname", "other_names"]:
            if k in params["infobox"]:
                newline_alt = [i.strip() for i in params["infobox"][k].split("\n") if i.strip()]
                new_alt = [j.strip() for i in newline_alt for j in i.split(",")]
                params["alternative_names"].extend(new_alt)

        affiliated_people = []
        for k in ["leaders", "founded_by", "founder"]:
            if k in params["infobox"]:
                aff = [i.strip() for i in params["infobox"][k].split("\n") if i.strip()]
                aff = [j.strip() for i in aff for j in i.split(",")]
                affiliated_people.extend(aff)
        params["affiliated_people"] = clean_names(affiliated_people)
        params["alternative_names"] = clean_names(params["alternative_names"])

        # The map blob is huge and never queried.
        params["infobox"].pop("map", None)

    # Categories, from the raw wikitext of the whole article (see
    # CATEGORY_PATTERN) -- the one thing we still need from below the lead.
    params["categories"] = CATEGORY_PATTERN.findall(text)
    params["update"] = datetime.date.today().isoformat()

    logger.debug(f"Good article: {title}")
    return params


def wrapper_loader(args):
    """Turn a (title, text) pair into an Elasticsearch bulk action, or None."""
    title, text = args
    res = parse_wiki_article(title, text)
    if not res:
        return None
    return {"_index": "wiki", "_id": res["title"], "_source": res}


def load_batch_es(page_batch, pool, es, chunk_size=500):
    """
    Parse a batch of pages in parallel and bulk-load them into the wiki index,
    backing off to smaller chunks / individual inserts if a bulk call fails.
    """
    args = [(title, text) for title, text in page_batch if title]
    results = pool.imap_unordered(wrapper_loader, args)
    actions = [r for r in tqdm(results, total=len(args), leave=False) if r]
    if not actions:
        return

    try:
        helpers.bulk(es, actions, chunk_size=chunk_size, raise_on_error=False)
    except Exception as e:
        logger.debug(f"Bulk exception: {e}")
        chunk_size = max(1, chunk_size // 2)
        logger.debug(f"Retrying with smaller chunk size: {chunk_size}")
        for i in range(0, len(actions), chunk_size):
            chunk = actions[i : i + chunk_size]
            try:
                helpers.bulk(es, chunk, chunk_size=chunk_size, raise_on_error=False)
            except Exception as e:
                logger.debug(f"Chunk exception: {e}")
                for doc in chunk:
                    try:
                        es.index(index="wiki", id=doc["_id"], body=doc["_source"])
                    except Exception as e:
                        logger.debug(f"Document exception for {doc['_id']}: {e}")


def wiki_index_stats(es):
    """
    Return (docs_count, store_size, health) for the wiki index, refreshing
    first so the numbers are current. Returns (None, None, None) if the
    index doesn't exist.
    """
    if not es.indices.exists(index="wiki"):
        return None, None, None
    es.indices.refresh(index="wiki")
    try:
        stats = es.cat.indices(index="wiki", format="json")[0]
    except elasticsearch.NotFoundError:
        return None, None, None
    return int(stats["docs.count"]), stats["store.size"], stats["health"]


def file_date(path):
    """The ISO date a file was last modified, or None if it isn't there.

    Used as the best available stand-in for "how old is this dump?". The
    Wikipedia XML has no generation timestamp in its header, and the canonical
    download is named "latest", so the download time is what we can actually
    know. If you fetched a dated dump instead (enwiki-20260801-...), the
    filename recorded alongside this carries the real answer.
    """
    try:
        return datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    except OSError:
        return None


def code_commit():
    """The git commit this loader was run from, or None outside a checkout.

    Two indices built from the same dump by different versions of this script
    are not interchangeable, so the commit belongs in the provenance alongside
    the dump date.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def stamp_index_meta(es, index, meta):
    """Record build provenance on the index itself.

    Elasticsearch stores mapping `_meta` verbatim and never interprets it, so
    it's a durable place to answer "how stale is this index?" without shipping
    a separate manifest that can drift. Read it back with:

        curl -s 'localhost:9200/<index>/_mapping' | python -m json.tool
    """
    meta = {k: v for k, v in meta.items() if v is not None}
    es.indices.put_mapping(index=index, body={"_meta": meta})
    logger.info(f"Stamped {index} _meta: {meta}")


def load_es(file, es_batch, threads, drop=False):
    """Parse the dump and bulk-load formatted articles into the wiki index."""
    logger.info("Loading Wikipedia into Elasticsearch")
    es = Elasticsearch(ES_URL, timeout=60, max_retries=3, retry_on_timeout=True)

    before_docs, before_size, before_health = wiki_index_stats(es)
    logger.info(
        f"wiki index before: {before_docs if before_docs is not None else 'no index'} docs, "
        f"{before_size if before_size is not None else 'n/a'}, "
        f"health={before_health if before_health is not None else 'n/a'}"
    )

    if drop and es.indices.exists(index="wiki"):
        logger.info("Dropping wiki index before reload (--drop)")
        es.indices.delete(index="wiki")

    t = time.time()

    es_batch = int(es_batch)
    logger.info(f"Using batch size of {es_batch}")

    # Create the index with our mapping if it doesn't already exist.
    # Pass --drop to delete the old index first (records before stats before deleting).
    if not es.indices.exists(index="wiki"):
        logger.info("Creating 'wiki' index in Elasticsearch")
        with open(MAPPING_PATH, "r") as f:
            es.indices.create(index="wiki", body=f.read())

    # Disable auto-refresh for the bulk load -- ES otherwise refreshes (opens
    # a new searchable segment) once a second by default, which adds real
    # overhead across millions of bulk requests. Restored in `finally` so a
    # failed/interrupted run doesn't leave the index stuck without periodic
    # refresh.
    logger.info("Disabling index refresh for the bulk load (refresh_interval=-1)")
    es.indices.put_settings(index="wiki", body={"index": {"refresh_interval": "-1"}})

    pool = multiprocessing.Pool(threads)
    try:
        page_batch = []
        n = 0
        for n, (title, text) in tqdm(enumerate(iterate_wiki_pages(file)), total=ESTIMATED_PAGES):
            page_batch.append((title, text))
            if len(page_batch) >= es_batch:
                load_batch_es(page_batch, pool, es, chunk_size=min(500, es_batch // 10))
                page_batch = []
        if page_batch:
            load_batch_es(page_batch, pool, es, chunk_size=min(500, es_batch // 10))
    finally:
        pool.close()
        pool.join()
        logger.info("Restoring index refresh_interval to 1s")
        es.indices.put_settings(index="wiki", body={"index": {"refresh_interval": "1s"}})
    es.indices.refresh(index="wiki")

    after_docs, after_size, after_health = wiki_index_stats(es)
    logger.info(
        f"wiki index after:  {after_docs if after_docs is not None else 'no index'} docs, "
        f"{after_size if after_size is not None else 'n/a'}, "
        f"health={after_health if after_health is not None else 'n/a'}"
    )
    if before_docs is not None and after_docs is not None:
        logger.info(f"wiki docs change: {after_docs - before_docs:+d}")

    stamp_index_meta(
        es,
        "wiki",
        {
            "dump_file": os.path.basename(file),
            "dump_date": file_date(file),
            "build_date": datetime.date.today().isoformat(),
            "code_commit": code_commit(),
            "doc_count": after_docs,
            "builder": "NGEC elasticsearch/es_wiki/load_wiki_es.py",
            # Which set of mapping sub-fields this index has. wiki_mapping.json
            # declares it at index-creation time, but put_mapping replaces
            # _meta wholesale, so the stamp at the end of the build would
            # otherwise drop it. Keep the two in step.
            "schema_version": 2,
        },
    )

    logger.info(f"Processed {n + 1} dump pages")
    logger.info(f"Elapsed minutes: {(time.time() - t) / 60:.1f}")
    logger.info("Done loading wiki index")


@plac.pos("process", "Which stage to run", choices=["build_links", "load_redis", "load_es"])
@plac.pos("file", "Wikipedia dump location")
@plac.opt("es_batch", "Elasticsearch batch size", type=int)
@plac.opt("threads", "Number of worker processes", type=int)
@plac.flg("drop", "Drop the wiki index before loading (records before-stats first; use for a full refresh)")
def main(process, file="data/enwiki-latest-pages-articles.xml.bz2", es_batch=5000, threads=10, drop=False):
    if process == "build_links":
        build_links(file)
    elif process == "load_redis":
        load_redis()
    elif process == "load_es":
        load_es(file, es_batch, threads, drop=drop)


if __name__ == "__main__":
    plac.call(main)
