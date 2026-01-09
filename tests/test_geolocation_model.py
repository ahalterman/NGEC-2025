# Smoke test for geolocation model

from ngec import GeolocationModel, load_nlp, setup_logging

def test_geolocation_model():
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)

    assert geolocation_model is not None



# not working, 
def test_geolocation_model_process(es_client_local):   
    setup_logging()

    nlp = load_nlp()    
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)

    story_list = [{"event_text": "President Macron and Chancellor Angel Merkel met in Brussels today to discuss EU debt relief plans", "event_type": ["CONSULT"], "pub_date": "2016-05-01"}]
    just_text = [i['event_text'] for i in story_list]
    doc_list = [doc for doc in nlp.pipe(just_text)]

    story_list = geolocation_model.process(story_list, doc_list)

    assert 'geolocated_ents' in story_list[0]
    assert len(story_list[0]['geolocated_ents']) > 0