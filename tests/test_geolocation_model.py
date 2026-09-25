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

def test_geolocation_batch_matches_one_at_a_time(es_client_local):
    """process() runs mordecai3's batched core; it must place every name the
    same way as geoparse_doc on one document at a time."""
    nlp = load_nlp()
    geolocation_model = GeolocationModel(nlp=nlp, es_client=es_client_local, quiet=True)
    texts = ["Protesters marched through central Nairobi on Tuesday, and police fired tear gas near Uhuru Park.",
             "Gunmen attacked a checkpoint in the Mexican state of Guerrero, killing two soldiers outside Chilpancingo.",
             "No places here at all."]
    docs = list(nlp.pipe(texts))
    one_at_a_time = [geolocation_model.geo.geoparse_doc(d)["geolocated_ents"] for d in docs]
    batched = geolocation_model.process([{"event_text": t} for t in texts], docs)
    from_texts = geolocation_model.process([{"event_text": t} for t in texts])

    def key(ents):
        return [(e["start_char"], e["end_char"], e.get("geonameid"), e.get("no_match")) for e in ents]

    for a, b, c in zip(one_at_a_time, batched, from_texts):
        assert key(a) == key(b["geolocated_ents"]) == key(c["geolocated_ents"])
