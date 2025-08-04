import pandas as pd
from datasets import Dataset
from transformers import pipeline
from rich.progress import track
import time
import jsonlines
import os
import numpy as np
from tqdm import tqdm
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _load_event_definitions(def_file="PLOVER_structured_codebook_updated.csv",
                            base_path="assets/"):
    """
    Load a CSV of event definitions (including special instructions for the model.)
    """
    event_definitions = pd.read_csv(os.path.join(base_path, def_file))
    if 'event' not in event_definitions.columns:
        raise ValueError(f"During loading of the event definitions file, 'event' column was not found.")
    if 'event_def' not in event_definitions.columns:
        raise ValueError(f"During loading of the event definitions file, 'event_def' column was not found.")
    if 'extraction_notes' not in event_definitions.columns:
        # raise a warning instead of an error
        logger.warning(f"No 'extraction_notes' column was found in {def_file}. Are you sure you don't want to add it?")
    if 'mode' not in event_definitions.columns:
        # raise a warning instead of an error
        logger.warning(f"No 'mode' column was found in {def_file}. Are you sure you don't want to add it?")
    return event_definitions


def _make_system_content_short():
    system_content_short = """Extract political events as JSON.

OUTPUT FORMAT:
[
  {
    "event_type": "EVENT_TYPE",
    "anchor_quote": "quote from text",
    "actor": "who performed action OR N/A",
    "recipient": "who was targeted OR N/A", 
    "date": "when occurred OR N/A",
    "location": "where occurred OR N/A"
  }
]

Return valid JSON only. Empty array [] if no events."""
    return system_content_short

def _load_sampling_params():
    """
    Load the sampling parameters for the model.
    """
    sampling_params = SamplingParams(
        temperature=0.5,       # Greedy decoding breaks Qwen
        top_p=0.8,             # Qwen3 non-thinking recommendation  
        top_k=20,              # Qwen3 recommendation
        presence_penalty=1.5,  # Recommended for quantized models
        min_p=0.0,
        #guided_decoding=guided_decoding_params, # Optionally, set a JSON schema for contrained decoding
        max_tokens=1024,
    )
    return sampling_params


class AttributeModel:
    def __init__(self, 
                 event_definitions_file=None,
                 silent=False, # whether to silence progress bars and logs
                 batch_size=8,
                 save_intermediate=False,
                 gpu=False,
                 base_path="assets/"
                 ):
        """
        Initialize the attribute model

        Parameters
        ---------
        """
        
        if gpu:
            self.device="cuda:0"
        else:
            self.device=-1
        logger.info(f"Device (-1 is CPU): {self.device}")
        print("Loading model")
        self.model = LLM(model="ahalt/event-attribute-extractor",
                        enable_prefix_caching=True,
                        max_model_len=8000,
                        gpu_memory_utilization=0.80)
        self.tokenizer = AutoTokenizer.from_pretrained("ahalt/event-attribute-extractor")
        self.sampling_params = _load_sampling_params()
        self.silent=silent
        self.batch_size=batch_size
        self.save_intermediate=save_intermediate
        self.system_prompt = _make_system_content_short()
        if event_definitions_file is None:
            event_definitions_file = "PLOVER_structured_codebook_updated.csv"
        self.event_definitions = _load_event_definitions(event_definitions_file, base_path)


    def _get_event_definitions(self,
                               doc,
                               event,
                               event_def,
                               mode_def=None,
                               extraction_notes=None):
        """
        Logic to get the event/mode definitions for a given event type.

        # Example format:
        '## Event: **REQUEST**: All requests, demands, and orders. Requests, demands, and orders are less forceful than threats and potentially carry less serious repercussions
         
        ## Specific Sub-Event: Make a request for changes in policy, government, or institutions
         
        ## Special Instructions: NOTE: Protests (including protests making requests) are coded under a separate PROTEST category. Protest DO NOT fall under this category.'
        """

        user_message = f"### Document:\n\n{doc}\n\n"
        user_message += f"### Event: **{event}**: {event_def}\n\n"
        if mode_def:
            user_message += f"### Specific Sub-Event: **{mode_def}**\n\n"
        if extraction_notes:
            user_message += f"### Special Instructions: {extraction_notes}\n\n"
        user_message += "Extract the attributes of the given event in JSON format."

    def event_to_message(self, event):
        """
        Convert an event dict to a message for the model.
        """
        doc = event['event_text']
        event_type = event['event_type']
        event_rows = event_definitions.loc[event_definitions['event'] == event_type]
        event_def = event_rows['event_def'].values[0]
        # Get mode definition and extraction notes if they exist
        if 'event_mode' in event:
            if 'mode' in event_definitions.columns and 'mode_def' in event_definitions.columns:
                mode_def = event_rows.loc[event_rows['mode'] == event['event_mode'], 'mode_def'].values[0]
        # TODO: Get extraction notes if they exist
        # TODO: add self. back in everywhere here
        
        return self._get_event_definitions(doc, event_type, event_def, mode_def, extraction_notes)

    def make_prompt(self, doc, event_type, event_def, event_specific_notes=None):

        messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"### Document:\n\n{doc}\n\n### Event type: {event_type}: {event_def}\n\n{f'### Special instructions: {event_specific_notes}' if event_specific_notes else ''}\n\nExtract the attributes of the given event in JSON format."}            ]
        prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
        return prompt
                

    def process(self,
                event_list, 
                show_progress=False):
        """
        Given event records from the previous steps in the NGEC pipeline,
        run the QA model to identify the spans of text corresponding with
        each of the event attributes (e.g. ACTOR, RECIP, LOC, DATE.)

        Parameters
        --------
        event_list: list of event dicts. 
          At a minimum, it should entries the following keys:
            - event_text
            - id (id for the event)
            - _doc_position (needed to link back to the nlped list)
            - event_type
            - mode
        doc_list: list of spaCy NLP docs
        expand: bool
          Expand the QA-returned answer to include appositives or compound words?
        show_progress: bool
            If True, show a tqdm progress bar.

        Returns
        -----
        event_list: list of dicts
          Adds 'attributes', which looks like: {'ACTOR': [{'text': 'Mario Abdo Benítez', 'score': 0.19762}], 
                                                'RECIP': [{'text': 'Fernando Lugo', 'score': 0.10433}], 
                                                'LOC': [{'text': 'Paraguay', 'score': 0.24138}]}
        """
        # Step 1: further lengthen the data to generate separate elements
        # for each attribute/question, so we have unique (ID, event_cat, attribute) 
        logger.debug("Starting attribute process")

        # Create a list of prompts
        prompts = []
        for i in event_list:
            event_type = i['event_type']
            event_def = self.event_definitions.loc[self.event_definitions['event'] == event_type, 'event_def'].values[0]
            event_specific_notes = self.event_definitions.loc[self.event_definitions['event'] == event_type, 'extraction_notes'].values[0] if 'extraction_notes' in self.event_definitions.columns else None
            
            # Create a prompt for each event
            prompt = self.make_prompt(i['event_text'], event_type, event_def, event_specific_notes)
            prompts.append(prompt)

        # Now, at the very end, put the results back into the event list.
        for i in event_list:
            i['attributes'] = final_attributes[i['id']]

        if self.save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_attribute_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)

        return event_list


if __name__ == "__main__":
    import jsonlines
    import utilities
    import spacy
    nlp = spacy.load("en_core_web_sm") 

    data = [
        {"event_text": "A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.",
        "id": 123,
        "_doc_position": 0,
        "event_type": "PROTEST",
        "event_mode": "riot"},
        {"event_text": "Turkish forces battled with YPG militants in Syria.",
        "id": 456,
        "_doc_position": 1,
        "event_type": "ASSAULT",
        "event_mode": ""},
        {"event_text": "Turkish forces and Turkish-backed militias battled with YPG militants in Syria.",
        "id": 789,
        "_doc_position": 2,
        "event_type": "ASSAULT",
        "event_mode": ""}
    ]

    doc_list = list(track(nlp.pipe([i['event_text'] for i in data])))

    event_list = utilities.stories_to_events(data, doc_list)
    qa_model = AttributeModel(model_dir = "NGEC/assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457",
                             base_path = "NGEC/assets",
                             silent=False)

    output = qa_model.process(event_list, doc_list)

    print(output)   