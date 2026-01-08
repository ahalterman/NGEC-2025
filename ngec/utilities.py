import pandas as pd
from spacy.tokens import Token
from spacy.language import Language
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#def make_country_dict():
#    country = pd.read_csv("assets/wikipedia-iso-country-codes.txt")
#    country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
#    country_dict["CUW"] = len(country_dict)
#    country_dict["XKX"] = len(country_dict)
#    country_dict["SCG"] = len(country_dict)
#    country_dict["SSD"] = len(country_dict)
#    country_dict["BES"] = len(country_dict)
#    country_dict["NULL"] = len(country_dict)
#    country_dict["NA"] = len(country_dict)
#    return country_dict
#
#
#with open("assets/feature_code_dict.json", "r") as f:
#    feature_code_dict = json.load(f)
#

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
