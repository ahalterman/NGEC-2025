

from .attribute_model import AttributeModel
from .actors.actor_resolution import ActorResolver
from .geolocation import GeolocationModel
from .formatter import Formatter
from .utilities import load_nlp, stories_to_events
from .classifiers.plover_sklearn import PloverSklearnClassifier


class PloverCoder:
    """Placeholder class for PloverCoder functionality.

    Aka where the end-to-end code should eventually end up (#18)"""

    def __init__(self,
                 es_client,
                 event_threshold: float | None = None,
                 attribute_backend: str = "transformers",
                 gpu: bool = False,
                 save_intermediate: bool = False):
        """
        Run the whole PLOVER coding pipeline over a list of stories.

        Parameters
        ----------
        es_client: an Elasticsearch client (see ngec.es_client.setup_es_client).
            The wiki and geonames indices must be loaded.
        event_threshold: one confidence threshold applied to every event type.
            Higher means fewer, more certain events. Leave unset (the default) to
            use the per-type thresholds recorded with the models, which is
            usually what you want: the types are not calibrated alike, and their
            F1-maximizing thresholds range from roughly 0.15 to 0.8. The old
            default of 0.9 applied one number to all sixteen and, with the
            encoder mismatch fixed, would admit almost nothing.
        attribute_backend: inference backend for the attribute LLM. One of
            "transformers" (portable, works everywhere), "vllm" (much faster,
            Linux/CUDA), or "mlx" (macOS). See the README on installing these.
        gpu: run the attribute LLM on the GPU. "transformers" runs on CPU by
            default, which is fine for a handful of stories but slow for a
            corpus.
        save_intermediate: have each step write its output to a timestamped
            JSONL file, which is useful when debugging a specific step.
        """
        self.nlp = load_nlp()
        self.save_intermediate = save_intermediate

        # Instantiate components. Note that the event classifier has no
        # save_intermediate option, and the actor resolver takes it on
        # process() rather than here.
        self.event_model = PloverSklearnClassifier(threshold=event_threshold)
        # Hand the geoparser the spaCy model and ES connection we already have,
        # so it doesn't load a second copy of en_core_web_trf or assume ES is
        # on localhost.
        self.geolocation_model = GeolocationModel(nlp=self.nlp,
                                                  es_client=es_client,
                                                  save_intermediate=save_intermediate)
        self.attribute_model = AttributeModel(silent=True,
                                              gpu=gpu,
                                              backend=attribute_backend,
                                              save_intermediate=save_intermediate)
        self.actor_resolution_model = ActorResolver(spacy_model=self.nlp,
                                                    es_client=es_client)
        self.formatter = Formatter()


    def process(self, story_list: list[dict]) -> list[dict]:
        """Process a list of input stories

        """
        just_text = [i['event_text'] for i in story_list]
        doc_list = [doc for doc in self.nlp.pipe(just_text)]

        story_list = self.event_model.process(story_list)
        story_list = self.geolocation_model.process(story_list, doc_list)
        event_list = stories_to_events(story_list, doc_list)
        event_list = self.attribute_model.process(event_list)
        event_list = self.actor_resolution_model.process(event_list,
                                                         save_intermediate=self.save_intermediate)
        cleaned_events = self.formatter.process(event_list, return_raw=True)
        return cleaned_events