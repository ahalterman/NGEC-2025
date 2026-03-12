#
#   Demo app illustrating NGEC's Plover Coder end to end
#
#   To run:
#   
#   uv run --group demo-app streamlit run examples/ngec_app.py
#

import logging

from ngec.classifiers.plover_sklearn import PloverSklearnClassifier
from ngec import AttributeModel
from ngec import ActorResolver
from ngec import GeolocationModel
from ngec import Formatter
from ngec import utilities
from ngec.utilities import stories_to_events
from ngec.utilities import load_nlp as ngec_load_nlp
from ngec.es_client import setup_es_client
from ngec.logging import setup_logging


import streamlit as st

import spacy
import pandas as pd

# stuff that's just used to allow streamlit cacheing
#import preshed
#import cymem
#import spacy_transformers
#import thinc

setup_logging(level=logging.WARNING, quiet_third_party=True)
logger = logging.getLogger(__name__)

st.markdown("## NGEC test interface")

st.markdown("Put in some story text to see what NGEC produces.")
st.markdown("The event classifier step uses the open source models that are trained on synthetic documents. The accuracy is not as good as the proprietary models used to produce the POLECAT dataset. To manually override the event classification, set the event type (and mode) on the sidebar.")
st.markdown("Intermediate output is also returned but hidden by default.")

#@st.cache(allow_output_mutation = True)
#@st.cache_resource()
def load_nlp():
    return ngec_load_nlp()


nlp = load_nlp()

def es_client_external():
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        ES_HOST = os.getenv("ES_HOST", "localhost")
        ES_PORT = int(os.getenv("ES_PORT", "9200"))
        ES_USER = os.getenv("ES_USER", None)
        ES_PASSWORD = os.getenv("ES_PASSWORD", None)
        client = setup_es_client(hosts=[ES_HOST], port=ES_PORT, http_auth=(ES_USER, ES_PASSWORD))

        # Forece actual (non-lazy) connection so we catch if ES is not running at all
        client.info()

        return client
    except Exception as e:
        logger.error(f"Failed to set up external ES client: {e}")

es_client = es_client_external()

def format_output(cleaned_events):
    for event in cleaned_events:
        actors = recipients = actor_codes = recipient_codes = actor_wikis = recip_wikis = ""

        if event.get('actor'):
            actors = '; '.join([i['actor_role_query'] for i in event['actor']])
            actor_codes = '; '.join([f"{i['country']} {i['code_1']}" for i in event['actor']])
            actor_wikis = '; '.join([i['wiki'] for i in event['actor']])

        if event.get('recipient'):
            recipients = '; '.join([i['actor_role_query'] for i in event['recipient']])
            recipient_codes = '; '.join([f"{i['country']} {i['code_1']}" for i in event['recipient']])
            recip_wikis = '; '.join([i['wiki'] for i in event['recipient']])

        geo = event.get('event_location', {}).get('event_loc') or {}
        resolved_placename = geo.get('resolved_placename', '')
        adm1 = geo.get('admin1_name', '')
        country = geo.get('country_name', '')

        date_resolved = event.get('date_resolved', {})
        date = date_resolved.get('resolved_date', '') if isinstance(date_resolved, dict) else date_resolved

        d = {
            "Raw Actors": actors,
            "Actor Codes": actor_codes,
            "Actor Wikis": actor_wikis,
            "Event Type": event['event_type'],
            "Event Mode": event['event_mode'],
            "Raw Recipients": recipients,
            "Recipient Codes": recipient_codes,
            "Recipient Wikis": recip_wikis,
            "Resolved Placename": resolved_placename,
            "Admin1": adm1,
            "Country": country,
            "Date": date,
        }
        df = pd.DataFrame(d, index=[0]).transpose().reset_index()
        df.columns = ["Attribute", "Value"]
        df.index = [""] * len(df)
        st.table(df)

gpu = False
save_intermediate = False
expand_actors = False
base_path = "."

#@st.cache(allow_output_mutation = True)
@st.cache_resource()
def load_event_class():
    event_model = PloverSklearnClassifier(threshold=0.9)
    return event_model

pub_date = st.sidebar.text_input("Publication date", "today")
event_type = st.sidebar.text_input("Event type", "")
event_mode = st.sidebar.text_input("Mode type", "")
show_intermediate = st.sidebar.checkbox("Show intermediate output", False)
event_model = load_event_class()

#@st.cache(allow_output_mutation = True)
@st.cache_resource()
def load_geo(save_intermediate=save_intermediate, gpu=gpu):
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)
    return geolocation_model

#@st.cache(allow_output_mutation = True)
@st.cache_resource()
def load_attr():
    attribute_model = AttributeModel(silent=True, gpu=False, backend="transformers")
    return attribute_model


@st.cache_resource()
def load_resolution(nlp=nlp, base_path=base_path, save_intermediate=save_intermediate, gpu=gpu):
    actor_resolution_model = ActorResolver(spacy_model=nlp, es_client=es_client)
    return actor_resolution_model

@st.cache_resource()
def load_formatter():
    formatter = Formatter()
    return formatter

geolocation_model = load_geo()
attribute_model = load_attr()
actor_resolution_model = load_resolution()
formatter = load_formatter()

text = st.text_area("Input text", "Protesters were in the streets in Paris again today to protest against the government's austerity measures.")




if text:
    doc_list = [nlp(text)]

    story_list = [{"event_text": text, "id": "123", "event_type": [event_type], "event_mode": [event_mode], "pub_date": pub_date}]
    
    if not event_type:
        story_list = event_model.process(story_list)
        if show_intermediate:
            with st.expander("Show event class step output", expanded=False):
                st.write(story_list)
    if not story_list[0]['event_type']:
        st.error("No event type detected.")
        st.stop()
    story_list = geolocation_model.process(story_list, doc_list)

    event_list = utilities.stories_to_events(story_list, doc_list)

    if show_intermediate:
        with st.expander("Show geolocation step output", expanded=False):
            st.write(event_list)

    event_list = attribute_model.process(event_list)
    if show_intermediate:
        with st.expander("Show attribute step output", expanded=False):
            st.write(event_list)

    event_list = actor_resolution_model.process(event_list)
    if show_intermediate:
        with st.expander("Show actor resolution step output", expanded=False):
            st.write(event_list)

    st.markdown("### Final output")
    cleaned_events = formatter.process(event_list, return_raw=True)
    
    st.markdown(text)
    format_output(cleaned_events)

    with st.expander("Show raw final output", expanded=False):
        st.write(cleaned_events)
