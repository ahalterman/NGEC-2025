import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
from tqdm import tqdm
from rich.progress import track
import time
import jsonlines
import re
import json
import os
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
                        gpu_memory_utilization=0.8)
        self.tokenizer = AutoTokenizer.from_pretrained("ahalt/event-attribute-extractor")
        self.sampling_params = _load_sampling_params()
        self.silent=silent
        self.batch_size=batch_size
        self.save_intermediate=save_intermediate
        self.system_prompt = _make_system_content_short()
        if event_definitions_file is None:
            event_definitions_file = "PLOVER_structured_codebook_updated.csv"
        self.event_definitions = _load_event_definitions(event_definitions_file, base_path)


    def _get_event_info(self, event):
        """
        Convert an event dict to a message for the model.
        """
        mode_def = None
        extraction_notes = None
        doc = event['event_text']
        event_type = event['event_type']
        event_rows = am.event_definitions.loc[am.event_definitions['event'] == event_type]
        event_def = event_rows['event_def'].values[0]
        # Get mode definition and extraction notes if they exist
        if 'event_mode' in event:
            if event['event_mode'] != "":
                if 'mode' in am.event_definitions.columns and 'mode_def' in am.event_definitions.columns:
                    mode_def = event_rows.loc[event_rows['mode'] == event['event_mode'], 'mode_def'].values[0]
                if 'extraction_notes' in am.event_definitions.columns:
                    extraction_notes = event_rows.loc[event_rows['mode'] == event['event_mode'], 'extraction_notes'].values[0]

        return doc, event_type, event_def, mode_def, extraction_notes
    
    def _make_user_message(self,
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
        user_message += f"### Event: **{event}**: {event_def}\n"
        if mode_def:
            user_message += f"### Specific Sub-Event: **{mode_def}**\n"
        if extraction_notes:
            if not pd.isna(extraction_notes):
                user_message += f"### Special Instructions: {extraction_notes}\n"
        user_message += "Extract the attributes of the given event in JSON format."
        return user_message

    def make_prompt(self, event):
        doc, event_type, event_def, mode_def, event_specific_notes = self._get_event_info(event)
        user_message = self._make_user_message(doc, 
                                               event_type, 
                                               event_def,
                                               mode_def,
                                               event_specific_notes)
        messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                    ]
        prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
        return prompt
    
    def call_llm_batch(self, prompts):
        if type(prompts) is not list:
            prompts = [prompts]
        outputs = self.model.generate(prompts, sampling_params=self.sampling_params)
        responses = [i.outputs[0].text.strip() for i in outputs]
        #print(responses)

        json_responses = []
        error_responses = []
        for response in responses:
            response = re.sub("<think>.*?</think>", "", response, flags=re.DOTALL)  # Remove <think> tags and content
            try:
                json_responses.append(json.loads(response))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                json_responses.append([])  # Append empty list on error
                error_responses.append(response)
        logger.info(f"Number of JSON decode errors: {len(error_responses)}")
        logger.debug(f"Error responses: {error_responses}")
        return json_responses
                

    def process(self,
                event_list):
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
        print("Making prompts...")
        prompts = [am.make_prompt(event) for event in tqdm(event_list, desc="Making prompts", disable=am.silent)]
        final_attributes = self.call_llm_batch(prompts)

        # Post-processing (split the ; separated attributes into lists)


        # Now, at the very end, put the results back into the event list.
        for n, i in enumerate(event_list):
            # split each attribute into a list (semicolon separated)
            attributes = final_attributes[n]
            # [{'actor': 'a group of Hindu nationalists; the VHP',
            #      'anchor_quote': 'A group of Hindu nationalists and the VHP rioted in '
            #                      'Dehli last week, burning Muslim shops.',
            #      'date': 'last week',
            #      'event_type': 'PROTEST:Violent riot',
            #      'location': 'Dehli',
            #      'recipient': 'Muslim shops'}]
            #i['attributes'] = final_attributes[n]
            for event in attributes:
                for key, value in event.items():
                    if key in ['actor', 'date', 'recipient', 'location']:
                        # If the value is a string, split it by semicolon and strip whitespace
                        if isinstance(value, str):
                            value = [v.strip() for v in value.split(';')]
                        # If the value is a list, ensure all items are stripped of whitespace
                        elif isinstance(value, list):
                            value = [v.strip() for v in value]
                        else:
                            continue
                        # Update the event with the cleaned value
                        event[key] = value
            event_list[n]['attributes'] = attributes

        if self.save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_attribute_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)

        return event_list


if __name__ == "__main__":
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

    am = AttributeModel(silent=False, gpu=True)
    prompt = am.make_prompt(data[0])
    print(prompt)
    output = am.call_llm_batch(prompt)

    all_prompts = [am.make_prompt(event) for event in data]
    all_attributes = am.call_llm_batch(all_prompts)

    all_outputs = am.process(data)

    #event_list[0]
    #{'event_text': 'A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.', 
    # 'id': 123, 
    # '_doc_position': 0, 
    # 'event_type': 'PROTEST', 
    # 'event_mode': 'riot', 
    # 'attributes': [{'event_type': 'PROTEST: Violent riot', 
    #               'anchor_quote': 'A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.', 
    #               'actor': ['a group of Hindu nationalists'], 
    #               'recipient': ['Muslim shops'], 
    #               'date': ['last week'], 
    #               'location': ['Dehli']}]}
