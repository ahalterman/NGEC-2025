import multiprocessing
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import mwparserfromhell
import re
from tqdm import tqdm
from textacy.preprocessing.remove import accents as remove_accents
from bz2 import BZ2File as bzopen
import bz2
import pickle
import plac
import os
import redis
import json
import datetime
from lxml import etree

import logging

logger = logging.getLogger()
#handler = logging.FileHandler("wiki_es.log")
# use a console handler instead
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

es_logger = elasticsearch.logger
es_logger.setLevel(elasticsearch.logging.WARNING)



# Precompile regular expressions for better performance
REDIRECT_PATTERN = re.compile(r"#?(REDIRECT|redirect|Redirect)")
STUB_PATTERN = re.compile(r"\-stub")
USER_PORTAL_PATTERN = re.compile(r"^(User|Selected anniversaries|List |Portal )")
DISAMBIG_PATTERN = re.compile(r"\([Dd]isambiguation\)")
DELETION_PATTERN = re.compile(r"Articles for deletion")
CATEGORY_PATTERN = re.compile(r"\*?\n?Category\:")
PORTAL_PATTERN = re.compile(r"Portal\:")
INFOBOX_PATTERN = re.compile(r"[Ii]nfobox")
CATEGORY_EXTRACT_PATTERN = re.compile(r"Category:(.+?)\n")
LINK_PATTERN = re.compile(r"\[\[(.+?)\]\]")
NAME_PIPE_PATTERN = re.compile(r"\|.+?\]\]")
BRACKETS_PATTERN = re.compile(r"\[|\]")

# annoyingly, global variables are not shared between processes in Python
# so we need to reinitialize them in each worker
def init_worker():
    """Initialize worker processes with global regex patterns"""
    global REDIRECT_PATTERN, STUB_PATTERN, USER_PORTAL_PATTERN, DISAMBIG_PATTERN
    global DELETION_PATTERN, CATEGORY_PATTERN, PORTAL_PATTERN, INFOBOX_PATTERN
    global CATEGORY_EXTRACT_PATTERN, LINK_PATTERN, NAME_PIPE_PATTERN, BRACKETS_PATTERN
    
    # Precompile regular expressions for better performance
    REDIRECT_PATTERN = re.compile(r"#?(REDIRECT|redirect|Redirect)")
    STUB_PATTERN = re.compile(r"\-stub")
    USER_PORTAL_PATTERN = re.compile(r"^(User|Selected anniversaries|List |Portal )")
    DISAMBIG_PATTERN = re.compile(r"\([Dd]isambiguation\)")
    DELETION_PATTERN = re.compile(r"Articles for deletion")
    CATEGORY_PATTERN = re.compile(r"\*?\n?Category\:")
    PORTAL_PATTERN = re.compile(r"Portal\:")
    INFOBOX_PATTERN = re.compile(r"[Ii]nfobox")
    CATEGORY_EXTRACT_PATTERN = re.compile(r"Category:(.+?)\n")
    LINK_PATTERN = re.compile(r"\[\[(.+?)\]\]")
    NAME_PIPE_PATTERN = re.compile(r"\|.+?\]\]")
    BRACKETS_PATTERN = re.compile(r"\[|\]")


def get_redirect(page, title=None, text=None):
    if not title and not text and page:
        text = next(page).text
        if not page:
            logger.debug("not page")
            return None
        title = page.title
        if not page:
            return None
    if not text:
        return None

    wikicode = mwparserfromhell.parse(str(text))

    raw_intro = wikicode.get_sections()[0]
    intro_para = raw_intro.strip_code()
    if re.match("#?(REDIRECT|redirect)", intro_para):
        # skip/ignore redirects for now
        return None


def iterate_wiki_pages(dump_file):
    """
    A generator that efficiently parses a Wikipedia XML dump and yields (title, text) tuples.
    
    Args:
        dump_file (str): Path to the Wikipedia XML dump file (.xml or .xml.bz2)
        
    Yields:
        tuple: (title, text) pairs for each page in the dump
    """
    # Handle compressed files
    if dump_file.endswith('.bz2'):
        file_obj = bz2.BZ2File(dump_file, 'rb')
        logger.info(f"Opened {dump_file} as BZ2 file")
    else:
        file_obj = open(dump_file, 'rb')
        logger.info(f"Opened {dump_file} as regular file")
    
    # Try to detect the XML namespace from the file
    logger.info("Detecting XML namespace...")
    sample = file_obj.read(10000)  # Read a sample to detect namespace
    file_obj.seek(0)  # Reset to beginning
    
    # Look for namespace pattern
    ns_match = re.search(rb'xmlns="(http://www.mediawiki.org/xml/export-[^"]+)"', sample)
    if ns_match:
        namespace = ns_match.group(1).decode('utf-8')
        logger.info(f"Detected namespace: {namespace}")
    else:
        namespace = "http://www.mediawiki.org/xml/export-0.10/"
        logger.info(f"No namespace detected, using default: {namespace}")
    
    # Define XML tags with namespace
    ns = "{" + namespace + "}"
    page_tag = f"{ns}page"
    title_tag = f"{ns}title"
    revision_tag = f"{ns}revision"
    text_tag = f"{ns}text"
    
    # Use iterparse to avoid loading the whole file into memory
    logger.info("Starting to parse XML...")
    context = etree.iterparse(file_obj, events=('end',), tag=page_tag)
    
    for event, elem in context:
        try:
            # Extract title
            title_elem = elem.find(f".//{title_tag}")
            if title_elem is None:
                continue
            title = title_elem.text
            
            # Extract the latest revision text
            revision = elem.find(f".//{revision_tag}")
            if revision is None:
                continue
                
            text_elem = revision.find(f".//{text_tag}")
            if text_elem is None:
                continue
                
            text = text_elem.text or ""
            
            yield (title, text)
            
        except Exception as e:
            logger.warning(f"Error processing element for title '{title if 'title' in locals() else 'unknown'}': {e}")
        finally:
            # Clear the element to avoid memory issues
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    
    file_obj.close()
    logger.info("Finished parsing XML")


def get_page_redirect(page, title, text):
    """Returns (original page, new page to redirect to)"""
    wikicode = mwparserfromhell.parse(text)
    raw_intro = wikicode.get_sections()[0]
    if re.match(redirect_pattern, str(raw_intro)):
        new_page = re.findall(r"\[\[(.+?)\]\]", str(raw_intro))
        try:
            new_page = new_page[0]
        except:
            return None
        # Too many false positives come from this splitting. Keep as-is instead, even
        # if that means it won't get added to any articles.
        #new_page = new_page.split("#")[0]
        return (str(title), str(new_page))


def clean_names(name_list):
    if not name_list:
        return []
    name_list = [re.sub("\|.+?\]\]", "", i).strip() for i in name_list]
    name_list = [re.sub("\[|\]", "", i).strip() for i in name_list]
    # There are some weird entries here like "son:"
    name_list = [i for i in name_list if not i.endswith(":")]
    de_accent = [remove_accents(i) for i in name_list]
    name_list = name_list + de_accent
    name_list = list(set(name_list))
    return name_list



def process_redirect_chunk(args):
    """
    Process a chunk of pages to extract redirects.
    
    Args:
        args (list): List of (title, text) tuples
        
    Returns:
        dict: Dictionary mapping target pages to lists of redirecting pages
    """
    redirects = {}
    for title, text in args:
        redir = get_page_redirect(None, title, text)
        if redir:
            if redir[1] not in redirects:
                redirects[redir[1]] = [redir[0]]
            else:
                redirects[redir[1]] = list(set(redirects[redir[1]] + [redir[0]]))
    return redirects

def parse_wiki_article(title=None, text=None, use_redis=True):
    """
    Go through a Wikipedia dump and format the article so it's useful for us.

    Pull out the article's:
    - title
    - short desc: (new!) it's similar to the Wikidata short description
    - first para
    - redirects (from Redis)
    - alternative names (anything bold in the first para)
    - info box
    """
    # These had errors earlier: pull them out separately for inspection.
    if title in ['Kyle Rittenhouse', 'Dmitry Peskov', 'Warsaw', 'Brasília', 'Beirut', 'Muhammadu Buhari',
                'Anil Deshmukh', 'Viktor Orbán']:
        print(f"found article: {title}")
        with open(f"{title}.txt", "w") as f:
            f.write(text)
    if not title and not text:
            return None
    if not text:
        logger.debug(f"No text for {title}")
        return None

    # There are a whole bunch of article types that we want to skip
    if title.endswith(".jpg") or title.endswith(".png"):
        logger.debug(f"Skipping image: {title}")
        return None
    if re.search("\-stub", title):
        logger.debug(f"Skipping Stub: {title}")
        return None
    if re.match("(User|Selected anniversaries)", title):
        logger.debug(f"Skipping User: {title}")
        return None
    if re.search("\([Dd]isambiguation\)", title):
        logger.debug(f"Skipping Disambig: {title}")
        return None
    if re.search("Articles for deletion", title):
        logger.debug(f"Skipping For deletion: {title}")
        return None
    if re.match("List ", title):
        logger.debug(f"Skipping List: {title}")
        return None
    if re.match("Portal ", title):
        logger.debug(f"Skipping Portal: {title}")
        return None
    if re.search("Today's featured article", title):
        logger.debug(f"Skipping featured article: {title}")
        return None
    if re.search("Featured article candidates", title):
        logger.debug(f"Skipping featured article candidate: {title}")
        return None
    if title.startswith("Peer review/"):
        logger.debug(f"Skipping peer review article: {title}")
        return None
    if title.startswith("Requests for adminship/"):
        logger.debug(f"Skipping adminship: {title}")
        return None
    if title.startswith("Featured list candidates/"):
        logger.debug(f"Skipping list candidates: {title}")
        return None
    if title.startswith("Sockpuppet investigations/"):
        logger.debug(f"Skipping sockpuppt: {title}")
        return None
    # clean up intro para? [[File:Luhansk raions eng.svg|thumb|100px|Raions of Luhansk]]
    # also delete the leftover alt names parentheses? 
    # "[[File:Luhansk raions eng.svg|thumb|100px|Raions of Luhansk]]\nLuhansk,(, ; , , , ; , ), also known as Lugansk and formerly known as Voroshilovgrad (1935-1958)"    

    wikicode = mwparserfromhell.parse(str(text))

    raw_intro = wikicode.get_sections()[0]
    intro_para_raw = raw_intro.strip_code()
    # strip out the occasional stuff that slips through
    intro_para = re.sub("(\[\[.+?\]\])", "", intro_para_raw).strip()
    # delete thumbs (not removed by strip_code()):
    intro_para = re.sub("^thumb\|.+?\n", "", intro_para)
    # do it again, the lazy way
    intro_para = re.sub("^thumb\|.+?\n", "", intro_para)
    # delete the first set of paratheses
    intro_para = re.sub("\(.+?\)", "", intro_para, 1)
    if not intro_para:
        logger.debug(f"No intro para for {title}.")
        #logger.debug(f"{wikicode.get_sections()[:2]}")
        return None
    if re.match("#?(REDIRECT|redirect|Redirect)", intro_para):
        logger.debug(f"Detected redirect in first para: {title}")
        # skip/ignore redirects for now
        return None
    if re.search("\*?\n?Category\:", intro_para):
        logger.debug(f"Category: {title}")
        return None
    if intro_para.startswith("Category:"):
        logger.debug(f"Category: {title}")
        return None
    if intro_para.startswith("<noinclude>"):
        logger.debug(f"Sneaky category? {title}")
        return None
    if re.search("may refer to", intro_para[0:100]):
        logger.debug(f"may refer to: {title}")
        return None
    if re.search("most often refers", intro_para[0:100]):
        logger.debug(f"most often refers: {title}")
        return None
    if re.search("most commonly refers", intro_para[0:100]):
        logger.debug(f"most commonly refers: {title}")
        return None
    if re.search("[Pp]ortal\:", intro_para[0:100]):
        logger.debug(f"Portal: {title}")
        return None
    alternative_names = re.findall("'''(.+?)'''", str(raw_intro))

    redirects = []
    if use_redis:
        redis_db = redis.StrictRedis(host="localhost", port=6379, db=0, charset="utf-8", decode_responses=True)
        redirects = redis_db.get(title)
        if redirects:
            redirects = redirects.split(";")

    if re.match("Categories for", title):
        return None

    try:
        short_desc = re.findall("\{\{[Ss]hort description\|(.+?)\}\}", str(raw_intro))[0].strip()
    except:
        logger.debug(f"Error getting short desc for {title}")
        #title_mod = re.sub("/", "_", title)
        #with open(f"error_articles/short_desc/{title_mod}.txt", "w") as f:
        #    f.write(str(raw_intro))
        short_desc = ""


    params = {"title": title,
             "short_desc": short_desc,
             "intro_para": intro_para.strip(),
             "alternative_names": clean_names(alternative_names),
             "redirects": clean_names(redirects),
             "affiliated_people": [],
             "box_type": None}

    for template in wikicode.get_sections()[0].filter_templates():
        if re.search("[Ii]nfobox", template.name.strip()):
            # do it this way to prevent overwriting
            info_box = {p.name.strip(): p.value.strip_code().strip() for p in template.params}
            params['infobox'] = info_box
            params['box_type'] = re.sub("Infobox", "", str(template.name)).strip()
            break

    if 'infobox' in params.keys():
        for k in ['name', 'native_name', 'other_name', 'alias', 'birth_name', 'nickname', 'other_names']:
            if k in params['infobox'].keys():
                newline_alt = [i.strip() for i in params['infobox'][k].split("\n") if i.strip()]
                new_alt = [j.strip() for i in newline_alt for j in i.split(",")]
                params['alternative_names'].extend(new_alt)

        affiliated_people = []
        for k in ['leaders', 'founded_by', 'founder']:
            if k in params['infobox'].keys():
                aff_people = [i.strip() for i in params['infobox'][k].split("\n") if i.strip()]
                aff_people = [j.strip() for i in aff_people for j in i.split(",")]
                affiliated_people.extend(aff_people) 

        params['affiliated_people'] = clean_names(affiliated_people)
        params['alternative_names'] = clean_names(params['alternative_names'])


    raw_categories = wikicode.get_sections()[-1].strip_code()
    categories = re.findall("Category:(.+?)\n", raw_categories)
    params['categories'] = categories

    if 'infobox' in params.keys():
        for k in ['map']:
            if k in params['infobox'].keys():
                del params['infobox'][k]
    
    params['update'] = datetime.date.today().isoformat()
    logger.debug(f"Good article: {title}")

    if title in ['Kyle Rittenhouse', 'Dmitry Peskov', 'Warsaw', 'Brasília', 'Beirut', 'Muhammadu Buhari',
                'Anil Deshmukh', 'Viktor Orbán']:
        with open(f"{title}.json", "w") as f:
            json.dump(params, f)
    return params


def wrapper_loader(args):
    """
    Process a single article for Elasticsearch indexing.
    
    Args:
        args (tuple): (title, text) pair
        
    Returns:
        dict: Elasticsearch action dict or None if the article should be skipped
    """
    title, text = args
    res = parse_wiki_article(title, text)
    if not res:
        return None
    action = {"_index" : "wiki",
              "_id" : res['title'],
              "_source" : res}
    return action


#def load_batch_es(page_batch, p, es):
#    actions = [p.apply_async(wrapper_loader, (title, text)) for title, text in page_batch if title]
#    actions = [i.get() for i in tqdm(actions, leave=False) if i]
#    actions = [i for i in actions if i]
#    try:
#        helpers.bulk(es, actions, chunk_size=-1, raise_on_error=False)
#        logger.info("Bulk loading success")
#    except Exception as e:
#        logger.info(f"Error in loading Wiki batch!!: {e}. Loading stories individually...")
#        for i in actions:
#            try:
#                response = helpers.bulk(es, i, chunk_size=-1, raise_on_error=False)
#                if response[1]:
#                    logger.info(f"Error on loading story {i}: {response[1]}")
#            except Exception as e:
#                logger.info(f"Skipping single Wiki story {e}")


def load_batch_es(page_batch, p, es, chunk_size=500):
    """
    Load a batch of Wikipedia pages into Elasticsearch.
    
    Args:
        page_batch (list): List of (title, text) tuples
        p (multiprocessing.Pool): Process pool for parallel processing
        es (Elasticsearch): Elasticsearch client
        chunk_size (int): Size of chunks for bulk indexing
    """
    # Prepare arguments for the pool
    args = [(title, text) for title, text in page_batch if title]
    
    # Use imap_unordered for better performance with mixed processing times
    results = p.imap_unordered(wrapper_loader, args)
    actions = [result for result in tqdm(results, total=len(args), leave=False) if result]
    
    if not actions:
        return  # Skip if no valid actions
        
    # Use smaller chunks for better memory management
    try:
        helpers.bulk(es, actions, chunk_size=chunk_size, raise_on_error=False)
    except Exception as e:
        logger.debug(f"Bulk exception: {e}")
        # Split the batch into smaller chunks if bulk insert fails
        chunk_size = max(1, chunk_size // 2)
        logger.debug(f"Retrying with smaller chunk size: {chunk_size}")
        
        # Process in smaller chunks
        for i in range(0, len(actions), chunk_size):
            chunk = actions[i:i+chunk_size]
            try:
                helpers.bulk(es, chunk, chunk_size=chunk_size, raise_on_error=False)
            except Exception as e:
                logger.debug(f"Chunk exception: {e}")
                # Individual document insert as last resort
                for doc in chunk:
                    try:
                        es.index(index='wiki', id=doc['_id'], body=doc['_source'])
                    except Exception as e:
                        logger.debug(f"Document exception for {doc['_id']}: {e}")

def merge_redirect_dicts(dict_list):
    """
    Merge multiple redirect dictionaries.
    
    Args:
        dict_list (list): List of dictionaries to merge
        
    Returns:
        dict: Merged dictionary
    """
    result = {}
    for d in dict_list:
        for k, v in d.items():
            if k not in result:
                result[k] = v
            else:
                result[k] = list(set(result[k] + v))
    return result

def read_clean_redirects():
    """
    Read the latest redirect dictionary file and clean it.
    
    Returns:
        dict: Cleaned redirect dictionary
    """
    files = os.listdir()
    versions = [int(re.findall("dict_(\d+)\.", i)[0]) for i in files if re.match("redirect_dict", i)]

    with open(f"redirect_dict_{max(versions)}.0.pkl", "rb") as f:
        redirect_dict = pickle.load(f)

    # Merge lowercase entries into their standard case versions
    del_list = []
    for k in redirect_dict.keys():
        if k.lower() in redirect_dict.keys() and k.lower() != k:
            redirect_dict[k] = list(set(redirect_dict[k] + redirect_dict[k.lower()]))
            del_list.append(k.lower())

    for d in del_list:
        if d in redirect_dict.keys():
            del redirect_dict[d]
            
    return redirect_dict


@plac.pos('process', "Which process to run?", choices=['build_links', 'load_redis', 'load_es'])
@plac.pos('file', "Wikiepdia dump location")
@plac.pos('es_batch', "Elasticsearch batch size")
@plac.pos('threads', "number of threads to use")
def process(process, file="enwiki-latest-pages-articles.xml.bz2", es_batch=5000, threads=10):
    p = multiprocessing.Pool(threads)
    logger.info(f"Reading from {file}")
    if re.search("bz2", file):
        dump = mwxml.Dump.from_file(bzopen(file, "r"))
    else:
        dump = mwxml.Dump.from_file(file)

    #dump = mwxml.Dump.from_file(open("Wikipedia-protest-export.xml"))
    # 1 core   = 11.077 total
    # 5 cores  = 3.254 total
    # 10 cores = 3.075 total
    
    if process == "build_links":
        logger.info("Building redirect link dictionary...")
        page_batch = []
        all_redirects = {}
        
        # Set up a counter for progress tracking
        estimated_pages = 24488554# Approximate count for English Wikipedia
        
        # Use our iterator for pages
        for n, (title, text) in tqdm(enumerate(iterate_wiki_pages(file)), total=estimated_pages):
            if n % 100000 == 0 and n > 0:
                k = n / 100000
                # Save intermediate results
                with open(f"redirect_dict_{k}.pkl", "wb") as f:
                    pickle.dump(all_redirects, f)
                    logger.info(f"Saved redirect dictionary at {k} x 100,000 pages")
            
            # Add page to batch
            page_batch.append((title, text))
            
            if len(page_batch) >= 5000:
                # Split the batch into chunks for parallel processing
                chunk_size = 500  # Process 500 pages per worker
                chunks = [page_batch[i:i+chunk_size] for i in range(0, len(page_batch), chunk_size)]
                
                # Process chunks in parallel
                chunk_results = p.map(process_redirect_chunk, chunks)
                
                # Merge results into the main dictionary
                for result in chunk_results:
                    all_redirects = merge_redirect_dicts([all_redirects, result])
                
                page_batch = []
                logger.debug(f"Processed {n} pages, found {len(all_redirects)} redirect targets")
        
        # Process any remaining pages
        if page_batch:
            chunks = [page_batch[i:i+chunk_size] for i in range(0, len(page_batch), chunk_size)]
            chunk_results = p.map(process_redirect_chunk, chunks)
            for result in chunk_results:
                all_redirects = merge_redirect_dicts([all_redirects, result])
        
        # Save final results
        k = (n + 1) / 100000
        with open(f"redirect_dict_{k}.pkl", "wb") as f:
            pickle.dump(all_redirects, f)
            logger.info(f"Final dump at {k} x 100,000 pages with {len(all_redirects)} redirect targets")
        # total redirects: around 4,868,606


    elif process == "load_redis":
        logger.info("Reading redirect dict...")
        redirect_dict = read_clean_redirects()
        redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
        pipe = redis_db.pipeline()
        for n, item in tqdm(enumerate(redirect_dict.items()), total=len(redirect_dict)):
            k, v = item
            v_str = ";".join(v)
            pipe.set(k, v_str)
            if n % 1000 == 0:
                pipe.execute()
        # get the final batch
        pipe.execute()

    elif process == "load_es":
        logger.info("Loading Wikipedia into Elasticsearch")
        # Configure ES with better timeout and retry settings
        es = Elasticsearch(
            urls='http://localhost:9200/', 
            timeout=60,
            max_retries=3,
            retry_on_timeout=True
        )

        # Convert batch size to int (plac might pass it as string)
        es_batch = int(es_batch)
        logger.info(f"Using batch size of {es_batch}")
        
        # Check if the index exists, create if not
        if not es.indices.exists(index="wiki"):
            logger.info("Creating 'wiki' index in Elasticsearch")
            # use the mapping defined in `wiki_mapping.json`
            with open("wiki_mapping.json", "r") as f:
                mapping = f.read()
            es.indices.create(index="wiki", 
                                body=mapping)
        
        page_batch = []
        # Use our page iterator with a realistic total estimate
        estimated_pages = 24488554  # Approximate count for English Wikipedia
        
        for n, (title, text) in tqdm(enumerate(iterate_wiki_pages(file)), total=estimated_pages):
            # Add page to batch
            page_batch.append((title, text))
            
            if len(page_batch) >= es_batch:
                logger.debug(f"Processing batch {n//es_batch}, last title: {page_batch[-1][0]}")
                load_batch_es(page_batch, p, es, chunk_size=min(500, es_batch//10))
                page_batch = []
        
        # Load final batch
        if page_batch:
            logger.debug(f"Processing final batch, {len(page_batch)} items")
            load_batch_es(page_batch, p, es, chunk_size=min(500, es_batch//10))


if __name__ == '__main__':
    plac.call(process)


