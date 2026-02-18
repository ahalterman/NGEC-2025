

from .attribute_model import AttributeModel
from .actors.actor_resolution import ActorResolver
from .geolocation import GeolocationModel
from .formatter import Formatter
from .utilities import load_nlp, stories_to_events


class PloverCoder:
    """Placeholder class for PloverCoder functionality.
    
    Aka where the end-to-end code should eventually end up (#18)"""

    def __init__(self, es_client):
        self.nlp = load_nlp()    

        # Instantiate components
        # TODO: doesn't work, Andy has not synced
        #event_model = EventClass()
        self.geolocation_model = GeolocationModel(geo_model=None, geo_path=None)
        self.attribute_model = AttributeModel(silent=True, gpu=False, backend="transformers")
        self.actor_resolution_model = ActorResolver(spacy_model=self.nlp, es_client=es_client)
        self.formatter = Formatter()


    def process(self, story_list: list[dict]) -> list[dict]: 
        """Process a list of input stories
        
        """
        just_text = [i['event_text'] for i in story_list]
        doc_list = [doc for doc in self.nlp.pipe(just_text)]

        # TODO event class not working
        #story_list = event_model.process(story_list)
        story_list = self.geolocation_model.process(story_list, doc_list)
        event_list = stories_to_events(story_list, doc_list)
        event_list = self.attribute_model.process(event_list)
        event_list = self.actor_resolution_model.process(event_list)
        cleaned_events = self.formatter.process(event_list, return_raw=True)
        return cleaned_events