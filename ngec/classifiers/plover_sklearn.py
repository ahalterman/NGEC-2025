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

The model directory
-------------------------------
A directory of models should contain the following:

    {EVENT_TYPE}.skops        one binary classifier per event type
    modes/{EVENT_TYPE}/{mode}.skops   one per mode, conditional on its type
    tfidf_vectorizer.skops    the fitted vectorizer, if word features were used
    metadata.json             encoder name, per-class thresholds, training metrics

`metadata.json` includes things like the sentence transformer encoder name 
to avoid problems (that definitely never happened!) like using a different encoder
during inference from the one used during training.

This metadata.json also specifies per-event F1-maximizing thresholds.
This can be overridden by passing `threshold=` to override every
class at once.

Features are built by `features.py` and are shared with the training script:
documents are chunked into overlapping windows and mean-pooled rather than
truncated at the encoder's limit, optionally concatenated with TF-IDF word
features. 
"""

import csv
import json
import warnings
from pathlib import Path
from importlib import resources


import numpy as np
import skops.io as sio
from sentence_transformers import SentenceTransformer

from .features import combine_features, encode_documents

DEFAULT_ENCODER = "all-mpnet-base-v2"


class DemoModelWarning(UserWarning):
    """Warning for non-production/demo model usage."""



def load_ontology(codebook_path: str | None = None) -> dict:
    """
    Load event type ontology from a codebook CSV file.

    Parses a PLOVER-style codebook CSV to extract event types and their
    associated modes.

    Parameters
    ----------
    codebook_path : str | None
        Path to the codebook CSV file. Expected to have columns:
        - 'event': event type names (e.g., AGREE, PROTEST, ASSAULT)
        - 'mode': mode names (may be empty for types without modes)

        The default will use the PLOVER codebook included in NGEC. 

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
    if codebook_path is None:
        codebook_path = str(resources.files("ngec").joinpath("assets/PLOVER_structured_codebook_updated.csv"))
    
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
        Defaults to a `modes/` subdirectory of `type_model_dir` when one exists,
        so mode models trained alongside the type models are picked up.
    threshold : float, optional
        One confidence threshold applied to every class. Leave unset to use the
        per-class thresholds recorded in metadata.json, which is usually what you
        want.
    encoder_name : str, optional
        SentenceTransformer model to encode with. Leave unset to use the encoder
        recorded in metadata.json, which is the one the models were trained on.
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
        threshold: float | None = None,
        encoder_name: str | None = None,
        progress_bar: bool = False,
        codebook_path: str | None = None,
        type_model_dir: str | None = None,
        mode_model_dir: str | None = None,
    ):
        if type_model_dir is None:
            type_model_dir = str(resources.files("ngec").joinpath("assets/event_models_v2/"))
        self.type_model_dir = Path(type_model_dir)

        # Mode models live under the type model directory unless told otherwise,
        # so a caller who points at a model directory gets the modes that were
        # trained alongside those types without having to know a second path.
        if mode_model_dir is None:
            default_mode_dir = self.type_model_dir / "modes"
            mode_model_dir = str(default_mode_dir) if default_mode_dir.exists() else None
        self.mode_model_dir = Path(mode_model_dir) if mode_model_dir else None

        self.progress_bar = progress_bar

        # metadata.json records how the models in this directory were built.
        self.metadata = self._load_metadata()

        self.encoder_name = (
            encoder_name
            or self.metadata.get("encoder")
            or DEFAULT_ENCODER
        )

        # A threshold passed in explicitly applies to every class. Otherwise each
        # class uses the threshold that maximized its F1 at training time, which
        # varies a lot between types and is recorded in metadata.json.
        self.threshold = threshold
        self.thresholds = {
            name: entry["threshold"]
            for name, entry in self.metadata.get("metrics", {}).items()
            if "threshold" in entry
        }
        self.mode_thresholds = {
            name: entry["threshold"]
            for name, entry in self.metadata.get("mode_metrics", {}).items()
            if "threshold" in entry
        }

        # Load ontology (event types and their modes)
        self.ontology = load_ontology(codebook_path)

        # Load sentence encoder
        self.encoder = SentenceTransformer(f'sentence-transformers/{self.encoder_name}')

        # Word features, if these models were trained with them. Without the
        # exact vocabulary and IDF weights the vectorizer was fit with, the word
        # part of the feature vector cannot be rebuilt and the model's
        # coefficients would line up with the wrong terms.
        self.vectorizer = self._load_vectorizer()

        # Load type classifiers
        self.type_models = self._load_type_models()

        # Load mode classifiers (optional)
        self.mode_models = self._load_mode_models() if self.mode_model_dir else {}

        warnings.warn(
            "Event classification models loaded. NOTE: these models are not the "
            "production models used to produce the POLECAT dataset. They are "
            "demonstration models for the PLOVER ontology, trained on Voice of "
            "America news articles labeled by an LLM applying the PLOVER codebook. "
            "If you are making custom event data, you'll need to train your own "
            "models. See `setup/train_classifiers/codebook_llm/` in the NGEC repo "
            "(github.com/ahalterman/NGEC-2025).",
            DemoModelWarning,
            stacklevel=2
        )

    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------

    def _load_metadata(self):
        """Read metadata.json from the model directory, if the models ship one."""
        path = self.type_model_dir / "metadata.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_vectorizer(self):
        """Load the fitted TfidfVectorizer that goes with these models."""
        path = self.type_model_dir / "tfidf_vectorizer.skops"
        if not path.exists():
            return None
        untrusted_types = sio.get_untrusted_types(file=path)
        return sio.load(path, trusted=untrusted_types)

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
        """Compute the feature matrix for a batch of texts.

        Documents are split into overlapping windows that fit the encoder and the
        window embeddings are averaged, rather than the document being truncated
        at the encoder's 384 word-piece limit. News stories are routinely longer
        than that and we don't want to truncate them.  

        If the models were trained with word features, the TF-IDF block is
        appended here in exactly the way `train_classifiers.py` appended it.

        Parameters
        ----------
        texts : list of str
            Texts to encode.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            Feature matrix with one row per text.
        """
        show_progress = self.progress_bar and len(texts) > 100
        embeddings = encode_documents(self.encoder, texts,
                                      show_progress=show_progress)
        if self.vectorizer is None:
            return embeddings
        return combine_features(embeddings, self.vectorizer.transform(texts))

    def _threshold_for(self, name, is_mode=False):
        """The decision threshold for one class.

        An explicit `threshold=` argument overrides everything. Otherwise the
        per-class threshold chosen at training time is used, falling back to 0.5
        for a class that has no recorded threshold.
        """
        if self.threshold is not None:
            return self.threshold
        table = self.mode_thresholds if is_mode else self.thresholds
        return table.get(name, 0.5)

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
        n_texts = embeddings.shape[0]
        results = [[] for _ in range(n_texts)]

        # Score every document against a model at once. The old loop called
        # predict_proba once per (document, event type), which is 16 calls per
        # document on matrices of one row each.
        for event_type, model in self.type_models.items():
            probs = model.predict_proba(embeddings)[:, 1]
            threshold = self._threshold_for(event_type)
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    results[i].append({
                        'event_type': event_type,
                        'confidence': float(prob)
                    })

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
        n_texts = embeddings.shape[0]
        if not self.mode_models:
            return [[] for _ in range(n_texts)]

        results = [[] for _ in range(n_texts)]

        # Mode models are conditional on their parent type, matching how they were
        # trained: only documents where the type fired are scored for its modes.
        rows_by_type = {}
        for i, type_preds in enumerate(type_results):
            for type_pred in type_preds:
                rows_by_type.setdefault(type_pred['event_type'], []).append(i)

        for event_type, rows in rows_by_type.items():
            if event_type not in self.mode_models:
                continue
            subset = embeddings[rows]
            for mode, model in self.mode_models[event_type].items():
                probs = model.predict_proba(subset)[:, 1]
                threshold = self._threshold_for(f"{event_type}-{mode}", is_mode=True)
                for row, prob in zip(rows, probs):
                    if prob >= threshold:
                        results[row].append(f"{event_type}-{mode}")

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
