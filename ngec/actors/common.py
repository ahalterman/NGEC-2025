
from importlib import resources
import logging
import os
from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import spacy
from xgboost import XGBClassifier

# Constants
DEFAULT_MODEL_PATH = "jinaai/jina-embeddings-v3"
DEFAULT_SIM_MODEL_PATH = 'actor_sim_model2'

# TODO remove this
DEFAULT_BASE_PATH = "./"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#######################################################
# Model Management
#######################################################

class ModelManager:
    """
    Model loading and management utilities.
    
    This class handles loading and caching of NLP models.
    
    Example:
        manager = ModelManager("./assets")
        nlp = manager.load_spacy_lg()
        trf = manager.load_trf_model()
    """
    
    def __init__(self, base_path=DEFAULT_BASE_PATH, device=None):
        """
        Initialize the model manager.
        
        Args:
            base_path: Path to directory containing model files
            device: Device to use for model inference ('cuda' or None)
        """
        self.base_path = base_path
        self.device = device
        self.models = {}  # Cache for loaded models
    
    def load_spacy_lg(self):
        """
        Load and return the spaCy language model.
        
        Returns:
            spaCy model: Loaded language model
        """
        if 'spacy' not in self.models:
            self.models['spacy'] = spacy.load("en_core_web_lg")
        return self.models['spacy']

    def load_trf_model(self, model_dir=DEFAULT_MODEL_PATH):
        """
        Load and return the sentence transformer model.
        
        Args:
            model_dir: Path or name of the transformer model
            
        Returns:
            SentenceTransformer: Loaded transformer model
        """
        if 'trf' not in self.models:
            self.models['trf'] = SentenceTransformer(model_dir, 
                                                     trust_remote_code=True, 
                                                     model_kwargs={'use_flash_attn': False},
                                                     device=self.device)
        return self.models['trf']

    def load_actor_sim_model(self, model_dir=DEFAULT_SIM_MODEL_PATH):
        """
        Load the actor similarity model trained on Wikipedia redirects.
        
        This model helps identify if two names refer to the same entity.
        
        Args:
            model_dir: Directory containing the similarity model
            
        Returns:
            SentenceTransformer: Loaded similarity model
        """
        if 'actor_sim' not in self.models:
            if self.base_path is not None:
                path = Path(self.base_path).joinpath('actor_sim_model2')
                self.models['actor_sim'] = SentenceTransformer(str(path))
            else:
                # TODO allow for custom model #26
                path = resources.files('ngec').joinpath('assets', 'actor_sim_model2')
                self.models['actor_sim'] = SentenceTransformer(str(path))
        return self.models['actor_sim']
    
    def load_wiki_ranker_model(self, model_dir=DEFAULT_MODEL_PATH):
        """
        Load the Wikipedia ranker models

        One model has context-related features and the other doesn't.
        (We need this to handle the case where the context is not provided)
        
        Args:
            model_dir: Directory containing the ranker model
            
        Returns:
            XGBoost models: Tuple of loaded ranker model
        """
        model_path = os.path.join(self.base_path, 'xgb_model.json')
        
        wiki_ranker = XGBClassifier()
        wiki_ranker.load_model(model_path)
        logger.warning("Using context-based XGBoost model for *no context* ranking.")
        wiki_ranker_no_context =  XGBClassifier()
        wiki_ranker_no_context.load_model(model_path)
        
        return wiki_ranker, wiki_ranker_no_context



#######################################################
# Text Processing Utilities
#######################################################

class TextPreProcessor:
    """
    Utilities for cleaning and normalizing text.
    
    This class provides methods for text cleaning, entity extraction,
    and noun phrase identification.
    
    Example:
        processor = TextPreProcessor()
        clean_text = processor.clean_query("The United States Government")
        # Returns: "united states government"
    """
    
    def clean_query(self, qt):
        """
        Clean and normalize a query string.
        
        Removes articles, ordinals, possessives, and other noise from text.
        
        Args:
            qt: Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Handle empty or simple cases
        qt = str(qt).strip()
        if qt in ['The', 'the', 'a', 'an', '']:
            return ""
            
        # Normalize whitespace
        qt = re.sub(' +', ' ', qt)  # remove multiple spaces
        qt = re.sub('\n+', ' ', qt)  # newline to space
        
        # Remove starting articles and ending prepositions
        qt = re.sub(r"^the ", "", qt, flags=re.IGNORECASE).strip()
        qt = re.sub(r"^an ", "", qt, flags=re.IGNORECASE).strip()
        qt = re.sub(r"^a ", "", qt, flags=re.IGNORECASE).strip()
        qt = re.sub(r" of$", "", qt).strip()
        qt = re.sub(r"^'s", "", qt).strip()
        
        # Remove ordinals
        qt = re.sub(r"(?<=\d\d)(st|nd|rd|th)\b", '', qt).strip()  # two-digit ordinals
        qt = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', qt).strip()    # one-digit ordinals
        
        # Remove leading numbers and possessives
        qt = re.sub(r"^\d+? ", "", qt).strip()
        qt = re.sub(r"'s$", "", qt).strip()
        
        # Return empty string if too short
        if len(qt) < 2:
            return ""
            
        return qt
    
    def extract_entity_components(self, 
                                  span_text, 
                                  nlp=None, 
                                  doc=None,
                                  job_titles=None, 
                                  job_title_embeddings=None, 
                                  get_embedding_func=None):
        """
        Extracts core entity, role, and geographic information from a text span.

        Args:
            span_text: String containing the entity span
            job_titles: List of known job titles/roles (optional)
            job_title_embeddings: Dict mapping job titles to embeddings (optional)
            get_embedding_func: Function to get embedding for a new text (optional)

        Returns:
            Dict with core_entity, role, and geographic_info
        """
        if doc is None:
            if nlp is None:
                raise ValueError("nlp object must be provided if pre-processed doc is not given.")
            doc = nlp(span_text)

        # Initialize results
        results = {
            'core_entity': None,
            'role': None,
            'geographic_info': None
        }

        # Step 1: Extract entities by type
        person_entities = []
        org_entities = []
        geo_entities = []

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                person_entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            elif ent.label_ == 'ORG':
                org_entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            elif ent.label_ in ['GPE', 'LOC', 'FAC', 'NORP']:
                geo_entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        # Step 1.5: Custom entity detection for abbreviations and special cases
        # Look for uppercase words that could be organization acronyms
        acronym_pattern = re.compile(r'\b([A-Z]{2,})\b')
        for match in acronym_pattern.finditer(span_text):
            acronym = match.group(1)
            # Check if it's not already detected
            already_detected = False
            for org in org_entities:
                if org['text'] == acronym:
                    already_detected = True
                    break
                
            if not already_detected:
                org_entities.append({
                    'text': acronym,
                    'start': match.start(),
                    'end': match.end()
                })

        # Step 2: Set geographic information
        if geo_entities:
            results['geographic_info'] = geo_entities[0]['text']

        # Step 3: Set core entity (prioritize PERSON over ORG)
        if person_entities:
            results['core_entity'] = person_entities[0]['text']
        elif org_entities:
            results['core_entity'] = org_entities[0]['text']

        # Step 4: Handle possessive patterns specially
        possessive_pattern = re.compile(r"([A-Za-z']+)['']s\s+([A-Za-z]+)")
        possessive_match = possessive_pattern.search(span_text)

        if possessive_match:
            possessor = possessive_match.group(1)
            possessed = possessive_match.group(2)

            # Check if possessor is a geo entity
            if results['geographic_info'] and possessor == results['geographic_info']:
                # Check if possessed is an org (or potential acronym)
                matches_org = False
                for org in org_entities:
                    if possessed in org['text']:
                        results['core_entity'] = org['text']
                        matches_org = True
                        break
                    
                # If not matched, check for acronyms
                if not matches_org and re.match(r'^[A-Z]{2,}$', possessed):
                    results['core_entity'] = possessed

        # Step 5: Extract role candidates
        role_candidates = []

        # 5.1: Look for appositives
        for token in doc:
            if token.dep_ == 'appos':
                appos_span = doc[token.left_edge.i:token.right_edge.i+1]

                # Check if this contains our core entity
                contains_core = False
                if results['core_entity'] and results['core_entity'] in appos_span.text:
                    contains_core = True

                if not contains_core:
                    role_candidates.append(appos_span.text)

        # 5.2: Extract parts not covered by core entity or geo info
        # Mark positions covered by entities
        covered = [False] * len(span_text)

        # Mark core entity
        if results['core_entity']:
            pattern = re.compile(r'\b' + re.escape(results['core_entity']) + r'\b')
            for match in pattern.finditer(span_text):
                start, end = match.span()
                for i in range(start, min(end, len(covered))):
                    covered[i] = True

        # Mark geographic info
        if results['geographic_info']:
            pattern = re.compile(r'\b' + re.escape(results['geographic_info']) + r'\b')
            for match in pattern.finditer(span_text):
                start, end = match.span()
                for i in range(start, min(end, len(covered))):
                    covered[i] = True

        # Extract uncovered segments
        uncovered_segments = []
        current = []

        for i, char in enumerate(span_text):
            if not covered[i]:
                current.append(char)
            elif current:
                segment = ''.join(current).strip(' ,')
                if segment and len(segment) > 1:
                    uncovered_segments.append(segment)
                current = []

        # Don't forget the last segment
        if current:
            segment = ''.join(current).strip(' ,')
            if segment and len(segment) > 1:
                uncovered_segments.append(segment)

        # Add these segments to role candidates
        role_candidates.extend(uncovered_segments)

        # 5.3: Special case for words before person entity
        if results['core_entity'] and person_entities:
            # Find the entity start position
            person_start = None
            for ent in person_entities:
                if ent['text'] == results['core_entity']:
                    person_start = ent['start']
                    break
                
            if person_start is not None and person_start > 0:
                # Check for text before person
                before_person = span_text[:person_start].strip()
                if before_person:
                    role_candidates.append(before_person)

        # 5.4: Special case for words after organization
        if results['core_entity'] and org_entities:
            # Find the entity end position
            org_end = None
            for ent in org_entities:
                if ent['text'] == results['core_entity']:
                    org_end = ent['end']
                    break
                
            if org_end is not None and org_end < len(span_text):
                # Get text after org entity
                after_org = span_text[org_end:].strip()
                if after_org:
                    # Clean up possessives in the after text
                    after_org = re.sub(r"^'s\s+", "", after_org)
                    if after_org:
                        role_candidates.append(after_org)

        # Step 6: Choose the best role candidate
        if role_candidates:
            # Clean up candidates
            cleaned_candidates = []
            for candidate in role_candidates:
                # Remove geographic entities from role description
                if results['geographic_info']:
                    candidate = re.sub(r'\b' + re.escape(results['geographic_info']) + r'\b', '', candidate)

                # Clean up whitespace, possessives and punctuation
                candidate = re.sub(r"['’]s\s+", " ", candidate)  # Remove possessives
                candidate = re.sub(r'[,.:;]+$', '', candidate)  # Remove trailing punctuation
                candidate = re.sub(r'\s+', ' ', candidate).strip()  # Clean whitespace
                # remove initial "'" or "’"
                candidate = re.sub(r"^[‘’]", '', candidate).strip()

                # Remove "of" without context
                candidate = re.sub(r'\bof\b\s*$', '', candidate).strip()

                if candidate:
                    cleaned_candidates.append(candidate)

            role_candidates = cleaned_candidates

            # Use embedding similarity if available
            if job_title_embeddings and get_embedding_func and role_candidates:
                best_match = None
                best_score = 0

                for candidate in role_candidates:
                    try:
                        candidate_emb = get_embedding_func(candidate)

                        for title, title_emb in job_title_embeddings.items():
                            sim = cos_sim([candidate_emb], [title_emb])[0][0]
                            if sim > best_score:
                                best_score = sim
                                best_match = candidate
                    except:
                        continue
                    
                if best_score > 0.5:
                    results['role'] = best_match
                    return results

            # Fallback heuristics if embedding matching doesn't work
            scored_candidates = []
            for candidate in role_candidates:
                score = 0

                # Check for role keywords
                role_keywords = ['official', 'president', 'mayor', 'secretary', 'minister', 
                                'member', 'council', 'general', 'party', 'service', 
                                'airport', 'police', 'attacker', 'right-wing', 'wing']

                for keyword in role_keywords:
                    if keyword in candidate.lower():
                        score += 5
                        break
                    
                # Favor multi-word candidates
                word_count = len(candidate.split())
                score += min(word_count, 3)

                # Favor candidates that appear at the beginning of the span
                if span_text.lower().startswith(candidate.lower()):
                    score += 2

                # Penalize very short candidates (less than 3 characters)
                if len(candidate) < 3:
                    score -= 2

                scored_candidates.append((candidate, score))

            if scored_candidates:
                results['role'] = max(scored_candidates, key=lambda x: x[1])[0]

        # Step 7: Final cleanup
        if results['role']:
            # Ensure descriptors like "right-wing party" are fully captured
            if 'party' in span_text.lower() and 'wing' in results['role'].lower() and 'party' not in results['role'].lower():
                results['role'] += ' party'

            # Ensure airport, service, etc. are included in role when appropriate
            for suffix in ['airport', 'service', 'council']:
                if suffix in span_text.lower() and suffix not in results['role'].lower():
                    if results['role'].strip() and suffix not in results['role'].lower():
                        results['role'] += f' {suffix}'
        if results['role']:
            logger.debug(f"Converted '{span_text}' to '{results['core_entity']} ({results['role']})'")
        else:
            logger.debug(f"Converted '{span_text}' to '{results['core_entity']}' (no role found)")
        return results

    def strip_ents(self, doc):
        """
        Strip out named entities from text, leaving only non-entity tokens.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            str: Text with named entities removed
        """
        skip_list = ['a', 'and', 'the', "'s", "'", "s"]
        non_ent_tokens = [
            token.text_with_ws for token in doc 
            if token.ent_type_ == "" and token.text.lower() not in skip_list
        ]
        return ''.join(non_ent_tokens).strip()
    
    def make_acronym_dicts(self, text=None, doc=None, nlp=None):
        """
        Quick tool to identify acronyms (and their referents) in a doc.

        Args:
            text: string of text to process
            doc: spaCy doc object
        Returns:
            acronym_entities: dict of acronyms and their referents
        """
        if text is None and doc is None:
            raise ValueError("Either text or doc must be provided.")
        if text is not None and doc is None:
            if nlp is None:
                raise ValueError("nlp object must be provided if doc is provided.")
            doc = nlp(text)

        acronym_entities = {"U.N.": "United Nations", "UN": "United Nations"}
        for ent in doc.ents:
            # skip cardinals
            if ent.label_ in ["CARDINAL", "DATE", "TIME", "ORDINAL", "QUANTITY"]:
                continue
            # only take non-acronyms
            if len(ent) > 1 and not ent.text.isupper():
                # strip out leading prepositions and articles
                ent_text = ''.join([i.text_with_ws for i in ent if i.pos_ != "DET" and i.pos_ != "ADP"]).strip()
                # only take title case names
                # The title case doesn't always work with some edge cases. E.g. "Ta'ang National Liberation Army".
                # Instead, we can check if the first letter of each word is uppercase.
                first_letters = [True if word[0].isupper() else False for word in ent_text.split()]
                if ent_text.istitle():
                    acronym = ''.join([word[0].upper() for word in ent_text.split()])
                    acronym_entities[acronym] = ent_text
                elif all(first_letters):
                    # If the first letter of each word is uppercase, consider it as a potential acronym
                    acronym = ''.join([word[0].upper() for word in ent_text.split()])
                    acronym_entities[acronym] = ent_text
        return acronym_entities
    
    def get_noun_phrases(self, doc):
        """
        Extract non-entity noun phrases from a document.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            str: Space-joined noun phrases
        """
        skip_list = ['a', 'and', 'the']
        skip_ent_types = ['CARDINAL', 'DATE', 'ORDINAL']
        
        # Get noun chunks that don't end with an entity
        noun_phrases = [chunk for chunk in doc.noun_chunks if chunk[-1].ent_type_ == ""]
        
        # Collect tokens from those chunks, skipping certain words and entity types
        phrase_tokens = []
        for chunk in noun_phrases:
            for token in chunk:
                if token.text not in skip_list and token.ent_type_ not in skip_ent_types:
                    phrase_tokens.append(token.text_with_ws.lower())
                    
        return ''.join(phrase_tokens).strip()

    def get_noun_phrases_list(self, doc):
        """
        Get a list of non-entity noun phrases from a document.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            list: List of noun phrases
        """
        return [chunk for chunk in doc.noun_chunks if chunk[-1].ent_type_ == ""]
