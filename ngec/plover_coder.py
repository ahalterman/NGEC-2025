import logging

from .attribute_model import AttributeModel
from .actors.actor_resolution import ActorResolver
from .geolocation import GeolocationModel
from .formatter import Formatter
from .utilities import load_nlp, stories_to_events
from .classifiers.plover_sklearn import PloverSklearnClassifier


logger = logging.getLogger(__name__)


class PloverCoder:
    """Placeholder class for PloverCoder functionality.

    Aka where the end-to-end code should eventually end up (#18)"""

    def __init__(self,
                 es_client,
                 event_threshold: float | None = None,
                 event_classifier=None,
                 attribute_backend: str = "auto",
                 attribute_model_name: str | None = None,
                 event_definitions_file: str | None = None,
                 agents_file: str | None = None,
                 priorities_file: str | None = None,
                 gpu: bool = False,
                 max_gpu_memory: float = 0.8,
                 save_intermediate: bool = False,
                 intermediate_dir: str | None = None):
        """
        Run the whole PLOVER coding pipeline over a list of stories.

        Every argument has a default that runs the standard PLOVER pipeline.
        The ones marked "customization" below swap in your own classifier,
        attribute model, event definitions, or actor ontology without having to
        assemble the six steps by hand.

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
            encoder mismatch fixed, would admit almost nothing. Only used by the
            default classifier; set the threshold on your own classifier
            yourself if you pass `event_classifier`.
        event_classifier: (customization) your own event classifier, already
            constructed. Any object with a `process(story_list)` method that adds
            'event_type' (list), 'event_type_confidence' (dict), and 'event_mode'
            (list) to each story will do; the module docstring of
            ngec/classifiers/plover_sklearn.py describes the contract. The
            default None uses PloverSklearnClassifier, the reference PLOVER
            classifier. A custom classifier that emits event types outside
            PLOVER also needs definitions for them in the attribute step; see
            `event_definitions_file`.
        attribute_backend: inference backend for the attribute LLM. One of
            "vllm" (Linux with an NVIDIA GPU; fastest), "llamacpp" (any CPU),
            "mlx" (a Mac with Apple Silicon), or "transformers" (deprecated;
            about three times slower than llamacpp on a CPU). Each needs its
            extra installed; see the README. The default "auto" uses vllm if it is installed
            and there is a CUDA GPU, mlx on an Apple Silicon Mac if it is
            installed, and llamacpp otherwise. llamacpp runs the model in this
            process unless NGEC_LLAMACPP_URL points it at a llama-server, and
            takes its thread count from NGEC_LLAMACPP_THREADS if set.
        attribute_model_name: (customization) the attribute LLM, as a Hugging
            Face model name or a path to a local model directory. The default
            None uses the NGEC_ATTRIBUTE_MODEL environment variable if set, and
            otherwise ngec.attribute_model.DEFAULT_MODEL. The model's prompt
            format is looked up from its name (see KNOWN_PROMPT_FORMATS in
            ngec/attribute_model.py).
        event_definitions_file: (customization) your own event definitions,
            which the attribute LLM reads to know what each event type means.
            For the default model, a JSON file in the format of
            ngec/assets/event_definitions_v6.json: its entries are added to the
            definitions the model was trained on, replacing any for the same
            event type and mode, so it only needs the types you add or reword.
            Needed for any event type your `event_classifier` emits that PLOVER
            does not have. Older models (the "legacy" and "v5" prompt formats)
            read a CSV in the format of
            ngec/assets/PLOVER_structured_codebook_updated.csv instead.
        agents_file: (customization) path to your own actor dictionary, in the
            format of ngec/assets/PLOVER_agents.txt. The default None uses the
            PLOVER agents. The same file codes both the mention text and the
            Wikipedia descriptions of matched actors. Pass `priorities_file`
            with it.
        priorities_file: (customization) path to an actor-code priority CSV, in
            the format of ngec/assets/PLOVER_priorities.csv, used to pick among
            several candidate codes for one actor. The default None uses the
            PLOVER priorities. Codes missing from this table all tie at 0, so
            with a custom `agents_file` and the default priorities the choice
            between your codes falls back to the order the evidence came in.
        gpu: run the "transformers" backend on the GPU. The other backends
            ignore it: vllm always runs on the GPU, llamacpp on the CPU, and
            mlx on the Mac's own GPU.
        max_gpu_memory: only used by the "vllm" backend, which reserves this
            fraction of the GPU's *total* memory up front. The default of 0.8
            fails on a GPU that is already partly in use (vLLM reports that free
            memory is less than the requested utilization); lower it to fit
            alongside whatever else is running.
        save_intermediate: have each step write its output to a timestamped
            JSONL file, which is useful when debugging a specific step.
        intermediate_dir: the directory those files go in. If None (the
            default), the current working directory.
        """
        if event_classifier is not None and event_threshold is not None:
            raise ValueError("event_threshold only applies to the default classifier. "
                             "Set the threshold on your own event_classifier instead.")
        if agents_file is not None and priorities_file is None:
            logger.warning("A custom agents_file was given without a priorities_file. "
                           "Codes missing from the PLOVER priorities all tie, so the "
                           "choice among several candidate codes falls back to the "
                           "order the evidence came in.")

        self.nlp = load_nlp()
        self.save_intermediate = save_intermediate

        # Instantiate components. Note that the event classifier has no
        # save_intermediate option.
        if event_classifier is None:
            event_classifier = PloverSklearnClassifier(threshold=event_threshold)
        self.event_model = event_classifier
        # Hand the geoparser the spaCy model and ES connection we already have,
        # so it doesn't load a second copy of en_core_web_trf or assume ES is
        # on localhost.
        self.geolocation_model = GeolocationModel(nlp=self.nlp,
                                                  es_client=es_client,
                                                  save_intermediate=save_intermediate,
                                                  intermediate_dir=intermediate_dir)
        self.attribute_model = AttributeModel(event_definitions_file=event_definitions_file,
                                              model_name=attribute_model_name,
                                              silent=True,
                                              gpu=gpu,
                                              max_gpu_memory=max_gpu_memory,
                                              backend=attribute_backend,
                                              save_intermediate=save_intermediate,
                                              intermediate_dir=intermediate_dir)
        self.actor_resolution_model = ActorResolver(spacy_model=self.nlp,
                                                    es_client=es_client,
                                                    agents_file=agents_file,
                                                    priorities_file=priorities_file,
                                                    save_intermediate=save_intermediate,
                                                    intermediate_dir=intermediate_dir)
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
        event_list = self.actor_resolution_model.process(event_list)
        cleaned_events = self.formatter.process(event_list, return_raw=True)
        return cleaned_events
