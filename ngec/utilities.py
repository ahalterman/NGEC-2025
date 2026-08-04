from copy import deepcopy
import logging

import numpy as np
import spacy
from spacy.tokens import Token
from spacy.language import Language

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_nlp():
    spacy_doc_setup()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp


def spacy_doc_setup():
    try:
        Token.set_extension('tensor', default=False)
    except ValueError:
        pass
    
    try:
        @Language.component("token_tensors")
        def token_tensors(doc):
            trf_data = doc._.trf_data
            
            # Check if we're using the new curated transformers (spaCy 3.7+)
            if hasattr(trf_data, 'last_hidden_layer_state'):
                # New spaCy 3.7+ with curated transformers
                # Get the last hidden layer state - this is a Ragged tensor
                hidden_states = trf_data.last_hidden_layer_state
                
                # Convert to numpy array - the data attribute contains the actual tensor
                if hasattr(hidden_states, 'data'):
                    flattened_hidden_states = hidden_states.data  # Shape: (total_pieces, embedding_dim)
                else:
                    flattened_hidden_states = hidden_states
                
                # Get piece-to-token alignment
                # In curated transformers, pieces are grouped by token in the Ragged tensor
                if hasattr(hidden_states, 'lengths'):
                    # Use the lengths to determine which pieces belong to which token
                    piece_lengths = hidden_states.lengths  # Array of how many pieces per token
                    
                    piece_idx = 0
                    for token_idx, token in enumerate(doc):
                        if token_idx < len(piece_lengths):
                            num_pieces = piece_lengths[token_idx]
                            
                            if num_pieces > 0:
                                # Get the pieces for this token
                                token_pieces = flattened_hidden_states[piece_idx:piece_idx + num_pieces]
                                # Average the embeddings of all pieces for this token
                                averaged_embedding = np.mean(token_pieces, axis=0)
                                token._.set('tensor', averaged_embedding)
                                piece_idx += num_pieces
                            else:
                                # Fallback: zero vector
                                embedding_dim = flattened_hidden_states.shape[-1]
                                token._.set('tensor', np.zeros(embedding_dim))
                        else:
                            # Fallback for tokens beyond the piece alignment
                            embedding_dim = flattened_hidden_states.shape[-1]
                            token._.set('tensor', np.zeros(embedding_dim))
                
            else:
                # Legacy spaCy 3.0-3.6 with spacy-transformers
                # This is your original code for older versions
                hidden_states = trf_data.tensors[0]
                num_chunks, wordpieces_per_chunk, embedding_dim = hidden_states.shape
                flattened_hidden_states = hidden_states.reshape(-1, embedding_dim)
                
                alignment = trf_data.align
                
                for token_idx, token in enumerate(doc):
                    wordpiece_indices = alignment[token_idx].data
                    valid_indices = [idx for idx in wordpiece_indices if 0 <= idx < flattened_hidden_states.shape[0]]
                    
                    if len(valid_indices) > 0:
                        token_embeddings = flattened_hidden_states[valid_indices]
                        averaged_embedding = np.mean(token_embeddings, axis=0)
                        token._.set('tensor', averaged_embedding)
                    else:
                        token._.set('tensor', np.zeros(embedding_dim))
            
            return doc
            
    except ValueError:
        pass


def stories_to_events(story_list, doc_list=None):
    if not doc_list:
        logger.warning("Missing doc list...")
    if doc_list:
        if len(doc_list) != len(story_list):
            raise ValueError("the story list and list of spaCy docs must be the same length")
        for n, story in enumerate(story_list):
            doc = doc_list[n]
            story['story_people'] = list(set([i.text for i in doc.ents if i.label_ == "PERSON"]))
            story['story_organizations'] = list(set([i.text for i in doc.ents if i.label_ == "ORG"]))
            story['story_places'] = list(set([i.text for i in doc.ents if i.label_ in ["GPE", "LOC", "FAC"]]))
            story['_doc_position'] = n
    # "lengthen" the story-level data to generate a separate element
    # for each event type
    event_list = []
    for n, ex in enumerate(story_list):
        # If there is no mode key, create one
        if "event_mode" not in ex.keys():
            modes = []
        else:
            # event modes are formatted ["ACCUSE-disapprove", "ACCUSE-allege", "CONSULT-third-party"]
            modes = [i.split("-") for i in ex['event_mode']]
        
        events_with_modes = list(set([i[0] if i else None for i in modes]))
        for event_type in ex['event_type']:
            if event_type not in events_with_modes:
                event_mode = ""
                d = ex.copy() # note: the copy is important!
                d['event_type'] = event_type
                d['orig_id'] = d['id']
                d['event_mode'] = event_mode
                d['id'] = d['id'] + "_" + event_type + "_" # generate a new ID
                event_list.append(d)
            else:
                for et, *event_mode in modes:
                    # annoyingly, the event and mode are separated by a hyphen, but
                    # there are also hyphens within certain mode names. Merge those back
                    # together
                    event_mode = '-'.join([*event_mode])
                    if et != event_type:
                        # skip modes that are attached to the wrong event type
                        continue
                    d = ex.copy() # note: the copy is important!
                    d['event_type'] = event_type
                    d['orig_id'] = d['id']
                    d['event_mode'] = event_mode
                    d['id'] = d['id'] + "_" + event_type + "_" + event_mode # generate a new ID
                    event_list.append(d)
    return event_list


def explode_events(event_list):
    """
    Expand each event's list of extracted sub-events into separate event records.

    The attribute model may extract zero, one, or several distinct events of the
    same type from a single document. This function "lengthens" the event list so
    that each extracted sub-event becomes its own event record with a single
    ``attributes`` dict -- mirroring the way ``stories_to_events`` lengthens a
    story into one record per event type.

    Events whose attribute extraction was empty (``attributes == []``, i.e. the
    model found no event) are dropped from the returned ``exploded`` list and
    returned separately in ``dropped`` so the caller can report and inspect them
    rather than emitting empty records into the main output.

    Each exploded event gets a unique id of the form ``<base id>_<index>`` (e.g.
    ``story1_PROTEST__0``), where ``index`` counts the sub-events within the
    record. ``orig_id`` (set by ``stories_to_events``) still links back to the
    original story.

    Parameters
    ----------
    event_list : list of dict
        Events whose ``attributes`` key holds a list of extracted sub-events
        (the intermediate output of the attribute model, before exploding).

    Returns
    -------
    exploded : list of dict
        One record per extracted sub-event, each with ``attributes`` as a single
        dict and a unique ``id``.
    dropped : list of dict
        Records that had no extracted sub-events, left unmodified.
    """
    exploded = []
    dropped = []
    for event in event_list:
        sub_events = event.get('attributes', [])
        if not sub_events:
            dropped.append(event)
            continue
        for idx, sub_event in enumerate(sub_events):
            # shallow copy, like stories_to_events; each record gets its own
            # single-dict `attributes` and a unique id.
            new_event = event.copy()
            new_event['attributes'] = sub_event
            new_event['id'] = f"{event['id']}_{idx}"
            exploded.append(new_event)
    return exploded, dropped

