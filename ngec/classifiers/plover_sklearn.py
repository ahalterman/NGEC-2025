"""
Reference event classifier using sentence embeddings + sklearn.

This is the reference implementation for NGEC event classification.
To create your own classifier, copy this file and modify as needed.

Your classifier must have:
1. __init__() that loads your models
2. process(story_list) that adds 'event_type', 'event_type_confidence',
   and 'event_mode' to each story
3. Optionally: classify_one(text) for testing single documents

See the NGEC documentation for detailed guidance.
"""

import csv
import warnings
from pathlib import Path

import skops.io as sio
from sentence_transformers import SentenceTransformer



def load_ontology(codebook_path):
    """
    Load event type ontology from a codebook CSV file.

    Parses a PLOVER-style codebook CSV to extract event types and their
    associated modes.

    Parameters
    ----------
    codebook_path : str
        Path to the codebook CSV file. Expected to have columns:
        - 'event': event type names (e.g., AGREE, PROTEST, ASSAULT)
        - 'mode': mode names (may be empty for types without modes)

    Returns
    -------
    dict
        Dictionary mapping event types to lists of their modes.
        Event types without modes map to empty lists.

    Example
    -------
    ontology = load_ontology("ngec/assets/PLOVER_structured_codebook_updated.csv")
    ontology['AGREE']
    # []
    ontology['PROTEST']
    # ['demo', 'riot', 'strike', 'hunger', 'boycott', 'obstruct']
    """
    ontology = {}
    
    with open(codebook_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = row.get('event', '').strip()
            mode = row.get('mode', '').strip()

            if not event_type:
                continue

            if event_type not in ontology:
                ontology[event_type] = []

            if mode and mode not in ontology[event_type]:
                ontology[event_type].append(mode)

    return ontology


class PloverSklearnClassifier:
    """
    Event classifier using PLOVER ontology with sklearn binary classifiers.

    This is a reference implementation that users can copy and modify.
    It uses sentence embeddings from SentenceTransformer and per-event-type
    binary sklearn classifiers stored as .skops files.

    Parameters
    ----------
    codebook_path : str
        Path to PLOVER codebook CSV defining event types and modes.
    type_model_dir : str
        Directory containing .skops files for event type classifiers.
        Each file should be named {EVENT_TYPE}.skops (e.g., PROTEST.skops).
    mode_model_dir : str, optional
        Directory containing subdirectories for mode classifiers.
        Structure: {mode_model_dir}/{EVENT_TYPE}/{mode}.skops
    threshold : float
        Confidence threshold for predictions (default: 0.6).
    encoder_name : str
        Name of the SentenceTransformer model to use.
    progress_bar : bool
        Whether to show progress bar during batch encoding.

    Example
    -------
    clf = PloverSklearnClassifier(
        codebook_path="ngec/assets/PLOVER_structured_codebook_updated.csv",
        type_model_dir="ngec/assets/event_models/",
        threshold=0.6
    )
    stories = [{"event_text": "Protesters blocked the highway", "id": "1"}]
    results = clf.process(stories)
    print(results[0]['event_type'])
    # ['PROTEST']
    """

    def __init__(
        self,
        codebook_path,
        type_model_dir,
        mode_model_dir=None,
        threshold=0.6,
        encoder_name='paraphrase-mpnet-base-v2',
        progress_bar=False
    ):
        self.codebook_path = codebook_path
        self.type_model_dir = Path(type_model_dir)
        self.mode_model_dir = Path(mode_model_dir) if mode_model_dir else None
        self.threshold = threshold
        self.progress_bar = progress_bar

        # Load ontology (event types and their modes)
        self.ontology = load_ontology(codebook_path)

        # Load sentence encoder
        self.encoder = SentenceTransformer(f'sentence-transformers/{encoder_name}')

        # Load type classifiers
        self.type_models = self._load_type_models()

        # Load mode classifiers (optional)
        self.mode_models = self._load_mode_models() if mode_model_dir else {}

        print(
            "Event classification models loaded. NOTE: these models are not the "
            "production models used to produce the POLECAT dataset. Instead, these "
            "are demonstration models for the PLOVER ontology trained on synthetic "
            "text. If you are making custom event data, you'll need to train your "
            "own models. See the `setup` directory in the NGEC repo "
            "(github.com/ahalterman/NGEC)."
        )

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    def _load_type_models(self):
        """Load sklearn models for event type classification.

        Returns
        -------
        dict
            Dictionary mapping event_type -> sklearn model.
            Only includes event types with existing model files.
        """
        models = {}
        missing = []

        for event_type in self.ontology.keys():
            model_path = self.type_model_dir / f"{event_type}.skops"
            if model_path.exists():
                # Get untrusted types and trust them for loading sklearn models
                untrusted_types = sio.get_untrusted_types(file=model_path)
                models[event_type] = sio.load(model_path, trusted=untrusted_types)
            else:
                missing.append(event_type)

        if missing:
            warnings.warn(
                f"Model files not found for event types: {missing}. "
                f"These event types will not be classified.",
                UserWarning
            )

        return models

    def _load_mode_models(self):
        """Load sklearn models for mode classification.

        Returns
        -------
        dict
            Nested dictionary: event_type -> mode -> sklearn model.
        """
        if not self.mode_model_dir:
            return {}

        models = {}

        for event_type, modes in self.ontology.items():
            if not modes:
                continue

            event_mode_dir = self.mode_model_dir / event_type
            if not event_mode_dir.exists():
                continue

            models[event_type] = {}
            for mode in modes:
                model_path = event_mode_dir / f"{mode}.skops"
                if model_path.exists():
                    untrusted_types = sio.get_untrusted_types(file=model_path)
                    models[event_type][mode] = sio.load(model_path, trusted=untrusted_types)

        return models

    # -------------------------------------------------------------------------
    # Embedding Computation
    # -------------------------------------------------------------------------

    def _compute_embeddings(self, texts):
        """Compute sentence embeddings for a batch of texts.

        Parameters
        ----------
        texts : list of str
            Texts to encode.

        Returns
        -------
        numpy.ndarray
            Array of embeddings, shape (n_texts, embedding_dim).
        """
        show_progress = self.progress_bar and len(texts) > 100
        return self.encoder.encode(texts, show_progress_bar=show_progress)

    # -------------------------------------------------------------------------
    # Classification Methods
    # -------------------------------------------------------------------------

    def _classify_types(self, embeddings):
        """Classify embeddings into event types.

        Parameters
        ----------
        embeddings : numpy.ndarray
            Pre-computed embeddings.

        Returns
        -------
        list of list of dict
            For each text, a list of dicts with 'event_type' and 'confidence'.
            Example: [[{'event_type': 'PROTEST', 'confidence': 0.87}], ...]
        """
        results = []

        for emb in embeddings:
            preds = []
            for event_type, model in self.type_models.items():
                prob = model.predict_proba([emb])[0][1]
                if prob >= self.threshold:
                    preds.append({
                        'event_type': event_type,
                        'confidence': float(prob)
                    })
            results.append(preds)

        return results

    def _classify_modes(self, embeddings, type_results, texts_indices=None):
        """Classify embeddings into modes for detected event types.

        Parameters
        ----------
        embeddings : numpy.ndarray
            Pre-computed embeddings.
        type_results : list of list of dict
            Type classification results from _classify_types().
        texts_indices : list of int, optional
            If provided, maps embeddings back to original story indices.

        Returns
        -------
        list of list of str
            For each text, a list of "EVENT-mode" strings.
            Example: [['PROTEST-obstruct', 'PROTEST-demo'], ...]
        """
        if not self.mode_models:
            return [[] for _ in embeddings]

        results = [[] for _ in embeddings]

        for i, (emb, type_preds) in enumerate(zip(embeddings, type_results)):
            for type_pred in type_preds:
                event_type = type_pred['event_type']
                if event_type not in self.mode_models:
                    continue

                for mode, model in self.mode_models[event_type].items():
                    prob = model.predict_proba([emb])[0][1]
                    if prob >= self.threshold:
                        results[i].append(f"{event_type}-{mode}")

        return results

    # -------------------------------------------------------------------------
    # Main Interface
    # -------------------------------------------------------------------------

    def process(self, story_list):
        """
        Classify events in a list of stories.

        This is the main entry point for batch processing. It adds classification
        results directly to the input story dictionaries.

        Parameters
        ----------
        story_list : list of dict
            Each dict must have 'event_text' key containing the text to classify.
            Other keys are preserved.

        Returns
        -------
        list of dict
            Input stories with added keys:
            - 'event_type': list of detected event types (e.g., ['PROTEST', 'COERCE'])
            - 'event_type_confidence': dict mapping event_type -> confidence score
            - 'event_mode': list of "EVENT-mode" strings (e.g., ['PROTEST-obstruct'])

        Example
        -------
        >>> stories = [
        ...     {"event_text": "Protesters blocked the highway", "id": "1"},
        ...     {"event_text": "Countries signed a peace treaty", "id": "2"}
        ... ]
        >>> results = clf.process(stories)
        >>> results[0]['event_type']
        ['PROTEST']
        >>> results[0]['event_mode']
        ['PROTEST-obstruct']
        """
        if not story_list:
            return story_list

        # Extract texts and compute embeddings once
        texts = [s['event_text'] for s in story_list]
        embeddings = self._compute_embeddings(texts)

        # Classify event types
        type_results = self._classify_types(embeddings)

        # Classify modes (uses same embeddings)
        mode_results = self._classify_modes(embeddings, type_results)

        # Add results to stories
        for story, type_preds, modes in zip(story_list, type_results, mode_results):
            story['event_type'] = [t['event_type'] for t in type_preds]
            story['event_type_confidence'] = {
                t['event_type']: t['confidence'] for t in type_preds
            }
            story['event_mode'] = modes

        return story_list

    def classify_one(self, text):
        """
        Classify a single text. Useful for testing and debugging.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        tuple
            (event_types, event_modes) where:
            - event_types is a list of strings (e.g., ['PROTEST'])
            - event_modes is a list of "EVENT-mode" strings (e.g., ['PROTEST-obstruct'])

        Example
        -------
        >>> types, modes = clf.classify_one("Protesters blocked the highway")
        >>> print(types)
        ['PROTEST']
        >>> print(modes)
        ['PROTEST-obstruct']
        """
        result = self.process([{'event_text': text}])[0]
        return result['event_type'], result.get('event_mode', [])

if __name__ == "__main__":
    # Example usage
    clf = PloverSklearnClassifier(
        codebook_path="assets/PLOVER_structured_codebook_updated.csv",
        type_model_dir="assets/test_event_models/",
        threshold=0.9
    )

    test_text = "Protesters blocked the highway to demand better wages."
    event_types, event_modes = clf.classify_one(test_text)
    print("Event Types:", event_types)
    print("Event Modes:", event_modes)
