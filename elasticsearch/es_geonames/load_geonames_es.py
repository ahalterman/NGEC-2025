"""
Build the offline GeoNames Elasticsearch index used by the location resolver.

Cross-platform, self-contained tool (runs the same on Linux, macOS, and
Windows via `uv run --group es-build python load_geonames_es.py ...`). It can:

    download  -- fetch the GeoNames gazetteer (allCountries + admin code files)
    recreate  -- delete ONLY the "geonames" index and recreate its mapping
    load      -- bulk-load the gazetteer into the "geonames" index
    all       -- download -> recreate -> load

The "geonames" index lives in the *same* Elasticsearch data directory as the
"wiki" index. This tool only ever touches the "geonames" index, so the
co-resident "wiki" index is left intact. Point it at the build stack defined in
elasticsearch/compose-build.yml, which mounts the shared live data dir.

    docker compose --project-directory . -f elasticsearch/compose-build.yml up -d es
    cd elasticsearch/es_geonames && uv run --group es-build python load_geonames_es.py all

The Elasticsearch URL can be overridden with NGEC_ES_URL
(default http://localhost:9200). See ../README.md.
"""

import csv
import os
import time
import zipfile
from datetime import date, datetime
from urllib.request import urlretrieve

import plac
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from textacy.preprocessing.remove import accents as remove_accents

csv.field_size_limit()

ES_URL = os.environ.get("NGEC_ES_URL", "http://localhost:9200")
es = Elasticsearch(ES_URL, request_timeout=60, max_retries=2)

GEONAMES_BASE = "https://download.geonames.org/export/dump"
GAZETTEER_FILES = ["allCountries.zip", "admin1CodesASCII.txt", "admin2Codes.txt"]
MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geonames_mapping.json")


def iso_convert(iso2c, bad_codes):
    """
    Takes a two character ISO country code and returns the corresponding 3
    character ISO country code.
    Parameters
    ----------
    iso2c: A two character ISO country code.
    bad_codes: a set that collects unrecognised codes (instead of printing).
    Returns
    -------
    iso3c: A three character ISO country code.
    """

    iso_dict = {"AD":"AND", "AE":"ARE", "AF":"AFG", "AG":"ATG", "AI":"AIA",
                "AL":"ALB", "AM":"ARM", "AO":"AGO", "AQ":"ATA", "AR":"ARG",
                "AS":"ASM", "AT":"AUT", "AU":"AUS", "AW":"ABW", "AX":"ALA",
                "AZ":"AZE", "BA":"BIH", "BB":"BRB", "BD":"BGD", "BE":"BEL",
                "BF":"BFA", "BG":"BGR", "BH":"BHR", "BI":"BDI", "BJ":"BEN",
                "BL":"BLM", "BM":"BMU", "BN":"BRN", "BO":"BOL", "BQ":"BES",
                "BR":"BRA", "BS":"BHS", "BT":"BTN", "BV":"BVT", "BW":"BWA",
                "BY":"BLR", "BZ":"BLZ", "CA":"CAN", "CC":"CCK", "CD":"COD",
                "CF":"CAF", "CG":"COG", "CH":"CHE", "CI":"CIV", "CK":"COK",
                "CL":"CHL", "CM":"CMR", "CN":"CHN", "CO":"COL", "CR":"CRI",
                "CU":"CUB", "CV":"CPV", "CW":"CUW", "CX":"CXR", "CY":"CYP",
                "CZ":"CZE", "DE":"DEU", "DJ":"DJI", "DK":"DNK", "DM":"DMA",
                "DO":"DOM", "DZ":"DZA", "EC":"ECU", "EE":"EST", "EG":"EGY",
                "EH":"ESH", "ER":"ERI", "ES":"ESP", "ET":"ETH", "FI":"FIN",
                "FJ":"FJI", "FK":"FLK", "FM":"FSM", "FO":"FRO", "FR":"FRA",
                "GA":"GAB", "GB":"GBR", "GD":"GRD", "GE":"GEO", "GF":"GUF",
                "GG":"GGY", "GH":"GHA", "GI":"GIB", "GL":"GRL", "GM":"GMB",
                "GN":"GIN", "GP":"GLP", "GQ":"GNQ", "GR":"GRC", "GS":"SGS",
                "GT":"GTM", "GU":"GUM", "GW":"GNB", "GY":"GUY", "HK":"HKG",
                "HM":"HMD", "HN":"HND", "HR":"HRV", "HT":"HTI", "HU":"HUN",
                "ID":"IDN", "IE":"IRL", "IL":"ISR", "IM":"IMN", "IN":"IND",
                "IO":"IOT", "IQ":"IRQ", "IR":"IRN", "IS":"ISL", "IT":"ITA",
                "JE":"JEY", "JM":"JAM", "JO":"JOR", "JP":"JPN", "KE":"KEN",
                "KG":"KGZ", "KH":"KHM", "KI":"KIR", "KM":"COM", "KN":"KNA",
                "KP":"PRK", "KR":"KOR", "XK":"XKX", "KW":"KWT", "KY":"CYM",
                "KZ":"KAZ", "LA":"LAO", "LB":"LBN", "LC":"LCA", "LI":"LIE",
                "LK":"LKA", "LR":"LBR", "LS":"LSO", "LT":"LTU", "LU":"LUX",
                "LV":"LVA", "LY":"LBY", "MA":"MAR", "MC":"MCO", "MD":"MDA",
                "ME":"MNE", "MF":"MAF", "MG":"MDG", "MH":"MHL", "MK":"MKD",
                "ML":"MLI", "MM":"MMR", "MN":"MNG", "MO":"MAC", "MP":"MNP",
                "MQ":"MTQ", "MR":"MRT", "MS":"MSR", "MT":"MLT", "MU":"MUS",
                "MV":"MDV", "MW":"MWI", "MX":"MEX", "MY":"MYS", "MZ":"MOZ",
                "NA":"NAM", "NC":"NCL", "NE":"NER", "NF":"NFK", "NG":"NGA",
                "NI":"NIC", "NL":"NLD", "NO":"NOR", "NP":"NPL", "NR":"NRU",
                "NU":"NIU", "NZ":"NZL", "OM":"OMN", "PA":"PAN", "PE":"PER",
                "PF":"PYF", "PG":"PNG", "PH":"PHL", "PK":"PAK", "PL":"POL",
                "PM":"SPM", "PN":"PCN", "PR":"PRI", "PS":"PSE", "PT":"PRT",
                "PW":"PLW", "PY":"PRY", "QA":"QAT", "RE":"REU", "RO":"ROU",
                "RS":"SRB", "RU":"RUS", "RW":"RWA", "SA":"SAU", "SB":"SLB",
                "SC":"SYC", "SD":"SDN", "SS":"SSD", "SE":"SWE", "SG":"SGP",
                "SH":"SHN", "SI":"SVN", "SJ":"SJM", "SK":"SVK", "SL":"SLE",
                "SM":"SMR", "SN":"SEN", "SO":"SOM", "SR":"SUR", "ST":"STP",
                "SV":"SLV", "SX":"SXM", "SY":"SYR", "SZ":"SWZ", "TC":"TCA",
                "TD":"TCD", "TF":"ATF", "TG":"TGO", "TH":"THA", "TJ":"TJK",
                "TK":"TKL", "TL":"TLS", "TM":"TKM", "TN":"TUN", "TO":"TON",
                "TR":"TUR", "TT":"TTO", "TV":"TUV", "TW":"TWN", "TZ":"TZA",
                "UA":"UKR", "UG":"UGA", "UM":"UMI", "US":"USA", "UY":"URY",
                "UZ":"UZB", "VA":"VAT", "VC":"VCT", "VE":"VEN", "VG":"VGB",
                "VI":"VIR", "VN":"VNM", "VU":"VUT", "WF":"WLF", "WS":"WSM",
                "YE":"YEM", "YT":"MYT", "ZA":"ZAF", "ZM":"ZMB", "ZW":"ZWE",
                "CS":"SCG", "AN":"ANT"}

    try:
        return iso_dict[iso2c]
    except KeyError:
        bad_codes.add(iso2c)
        return "NA"


def read_adm1(fn="admin1CodesASCII.txt"):
    adm1_dict = {}
    with open(fn, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            adm1_dict[row[0]] = row[1]
    return adm1_dict


def read_adm2(fn="admin2Codes.txt"):
    adm2_dict = {}
    with open(fn, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            adm2_dict[row[0]] = row[1]
    return adm2_dict

def documents(reader, adm1_dict, adm2_dict, expand_ascii=True, bad_codes=None):
    """
    Load Geonames entries provided by the `reader` into Elasticsearch.

    If `expand_ascii` = True, any alternative names with accents will have an
    accent-stripped form included in the alternative names list along with the original form.
    For example, the name "Ḩadīqat ash Shahbā" would generate a second entry
    "Hadiqat ash Shahba". This process does not affect non-Roman characters
    (e.g. "北京" remains "北京".)

    Parameters
    ----------
    reader: a CSV reader object
    adm1_dict: dict
      map from numeric ADM1 codes to names
    adm2_dict: dict
      map from numeric ADM2 codes to names
    expand_ascii: bool
      Include versions of names with accents stripped out.
    """
    if bad_codes is None:
        bad_codes = set()
    todays_date = datetime.today().strftime("%Y-%m-%d")
    # Two unrelated tallies, kept separate so the summary isn't alarming:
    #   adm2_missing -- benign: the row's admin2 code has no name in
    #                   admin2Codes.txt, so admin2_name is "" but the record is
    #                   still indexed.
    #   row_errors   -- real: the row failed to parse and was skipped (not indexed).
    adm2_missing = 0
    row_errors = 0
    for row in tqdm(reader, total=12237435): # approx
        try:
            coords = row[4] + "," + row[5]
            country_code3 = iso_convert(row[8], bad_codes)
            alt_names = list(set(row[3].split(",")))
            # so annoying...add "US" as an alt name for USA
            if str(row[0]) == "6252001":
                alt_names.append("US")
                alt_names.append("U.S.")
            if str(row[0]) == "239880":
                alt_names.append("C.A.R.")
            alt_name_length = len(alt_names)
            if expand_ascii:
                stripped = [remove_accents(i) for i in alt_names]
                both = alt_names + stripped
                both = list(set(both))
                alt_names = both
            # get ADM1 name
            if row[10]:
                country_admin1 = '.'.join([row[8], row[10]])
                try:
                    admin1_name = adm1_dict[country_admin1]
                except KeyError:
                    admin1_name = ""
            else:
                admin1_name = ""
            # Get ADM2 name
            if row[11]:
                country_admin2 = '.'.join([row[8], row[10], row[11]])
                try:
                    admin2_name = adm2_dict[country_admin2]
                except KeyError:
                    admin2_name = ""
                    adm2_missing += 1
            else:
                admin2_name = ""
            doc = {"geonameid" : row[0],
                    "name" : row[1],
                    "asciiname" : row[2],
                    "alternativenames" : alt_names,
                    "coordinates" : coords,  # 4, 5
                    "feature_class" : row[6],
                    "feature_code" : row[7],
                    "country_code3" : country_code3,
                    "admin1_code" : row[10],
                    "admin1_name": admin1_name,
                    "admin2_code" : row[11],
                    "admin2_name": admin2_name,
                    "admin3_code" : row[12],
                    "admin4_code" : row[13],
                    "population" : row[14],
                    "alt_name_length": alt_name_length,
                    "modification_date" : todays_date
                   }
            action = {"_index" : "geonames",
                      "_id" : doc['geonameid'],
                      "_source" : doc}
            yield action
        except Exception as e:
            print(e, row)
            row_errors += 1
    print(f"admin2 name not found (benign, record still indexed): {adm2_missing}")
    print(f"rows skipped due to errors: {row_errors}")



def download_gazetteer(data_dir):
    """Download and unpack the GeoNames gazetteer into `data_dir`."""
    os.makedirs(data_dir, exist_ok=True)
    for fn in GAZETTEER_FILES:
        dest = os.path.join(data_dir, fn)
        print(f"Downloading {fn} ...")
        urlretrieve(f"{GEONAMES_BASE}/{fn}", dest)
    print("Unpacking allCountries.zip ...")
    with zipfile.ZipFile(os.path.join(data_dir, "allCountries.zip")) as z:
        z.extractall(data_dir)
    print("Download complete.")


def index_count():
    """Return the number of docs in the geonames index, or None if it's absent."""
    if not es.indices.exists(index="geonames"):
        return None
    es.indices.refresh(index="geonames")
    return es.count(index="geonames")["count"]


def file_date(path):
    """The ISO date a file was last modified, or None if it isn't there.

    The GeoNames gazetteer carries no version stamp, so the download time is
    the best available answer to "how old is this data?".
    """
    try:
        return date.fromtimestamp(os.path.getmtime(path)).isoformat()
    except OSError:
        return None


def stamp_index_meta(index, meta):
    """Record build provenance on the index itself.

    Elasticsearch stores mapping `_meta` verbatim and never interprets it, so
    it's a durable place to answer "how stale is this index?" without shipping
    a separate manifest that can drift. Read it back with:

        curl -s 'localhost:9200/geonames/_mapping' | python -m json.tool
    """
    meta = {k: v for k, v in meta.items() if v is not None}
    es.indices.put_mapping(index=index, body={"_meta": meta})
    print(f"Stamped {index} _meta: {meta}")


def recreate_index():
    """Delete ONLY the geonames index and recreate it from the mapping.

    The co-resident "wiki" index is not touched.
    """
    if es.indices.exists(index="geonames"):
        print("Deleting existing 'geonames' index ...")
        es.indices.delete(index="geonames")

    print("Creating 'geonames' index from mapping ...")
    with open(MAPPING_PATH, "r") as f:
        es.indices.create(index="geonames", body=f.read())

    # geonames_mapping.json already sets number_of_replicas to 0 -- single-node
    # clusters can't allocate replicas, which leaves the index yellow forever.
    # Set it again here so a hand-edited mapping can't reintroduce that.
    es.indices.put_settings(index="geonames", body={"index": {"number_of_replicas": 0}})

    # Relax disk watermarks so a large bulk load isn't blocked on a full-ish disk.
    es.cluster.put_settings(
        body={
            "transient": {
                "cluster.routing.allocation.disk.watermark.low": "10gb",
                "cluster.routing.allocation.disk.watermark.high": "5gb",
                "cluster.routing.allocation.disk.watermark.flood_stage": "4gb",
                "cluster.info.update.interval": "1m",
            }
        }
    )


def load(data_dir):
    """Bulk-load the gazetteer files in `data_dir` into the geonames index."""
    t = time.time()
    bad_codes = set()
    adm1_dict = read_adm1(os.path.join(data_dir, "admin1CodesASCII.txt"))
    adm2_dict = read_adm2(os.path.join(data_dir, "admin2Codes.txt"))
    with open(os.path.join(data_dir, "allCountries.txt"), "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        actions = documents(reader, adm1_dict, adm2_dict, bad_codes=bad_codes)
        helpers.bulk(es, actions, chunk_size=500)
    es.indices.refresh(index="geonames")
    print("Elapsed minutes: ", (time.time() - t) / 60)
    if bad_codes:
        diag_path = os.path.join(data_dir, "bad_iso_codes.txt")
        with open(diag_path, "w") as f:
            for code in sorted(bad_codes):
                f.write(repr(code) + "\n")
        print(f"Unrecognised ISO codes ({len(bad_codes)} unique) written to {diag_path}")


@plac.pos("process", "Which stage to run", choices=["download", "recreate", "load", "reload", "all"])
@plac.opt("data_dir", "Directory for the gazetteer files")
def main(process="all", data_dir="geonames_data"):
    # Captured before `recreate` empties the index, so this reflects the OLD data.
    before = index_count()
    print(f"geonames records before: {before if before is not None else 'no index'}")

    if process in ("download", "all"):
        download_gazetteer(data_dir)
    if process in ("recreate", "reload", "all"):
        recreate_index()
    if process in ("load", "reload", "all"):
        load(data_dir)

    after = index_count()
    print(f"geonames records after:  {after if after is not None else 'no index'}")
    if before is not None and after is not None:
        print(f"geonames records change: {after - before:+d}")

    # Only after a load: `recreate` on its own rebuilds the mapping (dropping
    # any previous _meta) and leaves the index empty, which is nothing to stamp.
    if process in ("load", "reload", "all") and after is not None:
        stamp_index_meta(
            "geonames",
            {
                "source": GEONAMES_BASE,
                "gazetteer_file": "allCountries.txt",
                "dump_date": file_date(os.path.join(data_dir, "allCountries.txt")),
                "build_date": date.today().isoformat(),
                "doc_count": after,
                "builder": "NGEC elasticsearch/es_geonames/load_geonames_es.py",
            },
        )


if __name__ == "__main__":
    plac.call(main)
