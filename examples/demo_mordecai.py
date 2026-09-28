"""
Geoparsing a sentence with mordecai3, the geoparser NGEC uses.

NGEC's geolocation step (`ngec.geolocation.GeolocationModel`) is a thin
wrapper around mordecai3's `Geoparser`. This script calls the geoparser
directly: it finds the place names in a text and resolves each one to a
GeoNames entry with coordinates and a country code.

mordecai3 is installed with NGEC and ships its own model, so no model path is
needed. It does need Elasticsearch running with the `geonames` index. If your
Elasticsearch is not on localhost:9200, change the `setup_es_client` call
below (see demo_wiki_resolution.py for reading the settings from a .env file).

Run it from the top of the repo:

    uv run python examples/demo_mordecai.py
"""

from pprint import pprint

from mordecai3 import Geoparser

from ngec.es_client import setup_es_client

# Connect to Elasticsearch, which holds the GeoNames gazetteer.
es_client = setup_es_client()

# Create the geoparser. This loads spaCy's en_core_web_trf model, so it takes
# a little while.
geo = Geoparser(es_client=es_client)

text = ("The Mexican government sent 300 National Guard troopers to bolster the "
        "southern state of Guerrero on Tuesday, where a local police chief and 12 "
        "officers were shot dead in a brutal ambush the day before.")

output = geo.geoparse_doc(text)

# One entry per place name found, with where it is in the text (start_char,
# end_char), the GeoNames match, and the model's confidence in it (score).
pprint(output["geolocated_ents"])
