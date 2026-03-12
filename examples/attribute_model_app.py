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

st.markdown("Put in some story text to see what NGEC produces.")
st.markdown("The event classifier step uses the open source models that are trained on synthetic documents. The accuracy is not as good as the proprietary models used to produce the POLECAT dataset. To manually override the event classification, set the event type (and mode) on the sidebar.")
st.markdown("Intermediate output is also returned but hidden by default.")

text = st.text_area("Text", "Protesters were in the streets in Paris again today to protest against the government's austerity measures.")
event_type = st.selectbox("Event type", PLOVER_EVENT_TYPES)



if st.button("Run Attribute Extraction"):

    input = [
        AttributeModelInput(
            event_text=text,
            event_type=event_type
        )
    ]

    result = am.process(input)

    st.write(dict(result[0])["attributes"])

