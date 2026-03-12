#
#   Demo app for the attribute model
#
#   uv run --group demo-app streamlit run examples/attribute_model_app.py

import logging

import streamlit as st

from ngec import AttributeModel
from ngec.attribute_model import AttributeModelInput
from ngec.logging import setup_logging


PLOVER_EVENT_TYPES = [
    "ACCUSE",
    "AGREE",
    "AID",
    "ASSAULT",
    "COERCE",
    "CONCEDE",
    "CONSULT",
    "COOPERATE",
    "MOBILIZE",
    "PROTEST",
    "REJECT",
    "REQUEST",
    "RETREAT",
    "SANCTION",
    "SUPPORT",
    "THREATEN",
]


@st.cache_resource()
def load_attribute_model():
    attribute_model = AttributeModel(silent=True, gpu=False, backend="transformers")
    return attribute_model


setup_logging(level=logging.WARNING, quiet_third_party=True)
logger = logging.getLogger(__name__)


am = load_attribute_model()


st.markdown("## Attribute Model Demo")

st.markdown("The attribute model extracts key attributes from an event text and given event type. Sometimes this will be an expact span from the input text, but since the underlying model is generative, it can also rephrase or infer information that is not explicitly stated in the text.")

text = st.text_area("Text", "Protesters were in the streets in Paris again today to protest against the government's austerity measures.")
event_type = st.selectbox("Event type", PLOVER_EVENT_TYPES, index=9)



if st.button("Run Attribute Extraction"):

    input = [
        AttributeModelInput(
            event_text=text,
            event_type=event_type
        )
    ]

    result = am.process(input)

    st.write(dict(result[0])["attributes"])

