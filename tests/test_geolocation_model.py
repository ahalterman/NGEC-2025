# Smoke test for geolocation model

import spacy

from ngec import GeolocationModel
from ngec.utilities import spacy_doc_setup


def test_geolocation_model():
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)

    assert geolocation_model is not None



# not working, 
def test_geolocation_model_process(es_client_external):
    # TODO this should be in the package somewhere
    def load_nlp():
        spacy_doc_setup()
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("token_tensors")
        return nlp

    nlp = load_nlp()    
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)

    story_list = [{"event_text": "President Macron and Chancellor Angel Merkel met in Brussels today to discuss EU debt relief plans", "event_type": ["CONSULT"], "pub_date": "2016-05-01"}]
    just_text = [i['event_text'] for i in story_list]
    doc_list = [doc for doc in nlp.pipe(just_text)]

    story_list = geolocation_model.process(story_list, doc_list)

    assert 'geolocated_ents' in story_list[0]
    assert len(story_list[0]['geolocated_ents']) > 0