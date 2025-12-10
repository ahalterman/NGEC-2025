import os
import pandas as pd
import time
import jsonlines
import re
import json
import logging

from importlib import resources
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from typing import Any, cast, Literal, TypedDict, NotRequired

logger = logging.getLogger(__name__)

# The line below is only useful for debugging/timing
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Address vLLM multiprocessing method error
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


#
#   Types and classes for type annotations
#   ================================
#
#   These don't do anything at runtime, but make it easier to see what kind
#   of input the AttributeModel expects and what kind of output it produces.
#

BackendType = Literal["vllm", "transformers", "mlx"]

class AttributeModelInput(TypedDict):
    """
    Dictionary representing minimal input for AttributeModel processing.
    
    Required keys:
        event_text: The text describing the event
        event_type: The type/category of the event
    
    Optional keys:
        event_mode: The mode of the event (optional)
        
    Additional keys are allowed and will be preserved.
    """
    event_text: str  # Required
    event_type: str  # Required
    event_mode: NotRequired[str]  # Optional
    # Any other keys are allowed

class Attributes(TypedDict):
    """
    Dictionary representing extracted attributes for an event.
    """
    event_type: str
    anchor_quote: str
    actor: list[str]
    recipient: list[str]
    date: list[str]
    location: list[str]

class AttributeModelOutput(AttributeModelInput):
    """
    Dictionary representing output from AttributeModel processing.

    The input list of dicts is augmented with an 'attributes' key for each
    event.
    
    """
    attributes: list[Attributes]  



def _load_event_definitions(def_file="PLOVER_structured_codebook_updated.csv",
                            base_path=None):
    """
    Load a CSV of event definitions (including special instructions for the model.)
    """
    if base_path is None:
        # Use importlib.resources to access package data
        try:
            with resources.files("NGEC").joinpath("assets", def_file).open() as f:
                event_definitions = pd.read_csv(f)
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback to file-based approach for development
            current_dir = os.path.dirname(__file__)
            file_path = os.path.join(current_dir, "assets", def_file)
            event_definitions = pd.read_csv(file_path)
    else:
        # Use provided base_path
        file_path = os.path.join(base_path, def_file)
        event_definitions = pd.read_csv(file_path)

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

def _load_vllm_sampling_params():
    """
    Load the sampling parameters for the vLLM model.
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
                 base_path=None,
                 max_gpu_memory=0.8,
                 vllm_model=None,
                 backend: BackendType="vllm"
                 ):
        """
        Initialize the attribute model

        Parameters
        ----------
        event_definitions_file : str, optional
            Path to event definitions CSV file
        silent : bool, default=False
            Whether to silence progress bars and logs
        batch_size : int, default=8
            Batch size for processing
        save_intermediate : bool, default=False
            Whether to save intermediate results
        gpu : bool, default=False
            Whether to use GPU
        base_path : str, optional
            Base path for loading files
        max_gpu_memory : float, default=0.8
            GPU memory utilization for vLLM
        vllm_model : vllm.LLM, optional
            Pre-initialized vLLM model to use
        backend: BackendType="vllm"
            Which backend to use: "vllm", "mlx", or "transformers"
        """
        self.silent=silent
        self.backend = backend

        if gpu:
            self.device="cuda"
        else:
            self.device="cpu"
        if not self.silent:
            logger.info(f"Device: {self.device}")
            logger.info(f"Backend: {self.backend}")

        # Load model based on backend
        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                if not self.silent: logger.error("vLLM not available. Use another backend.")
                raise ImportError("vLLM is not installed. Please install it or use backend='transformers'")
            
            if not self.silent: logger.debug("Loading vLLM model")
            if vllm_model:
                self.model = vllm_model
            else:
                self.model = LLM(model="ahalt/event-attribute-extractor",
                                 enable_prefix_caching=True,
                                 max_model_len=8000,
                                 gpu_memory_utilization=max_gpu_memory)
            self.sampling_params = _load_vllm_sampling_params()
            self.tokenizer = AutoTokenizer.from_pretrained("ahalt/event-attribute-extractor")
        elif self.backend == "transformers":
            if not self.silent: logger.debug("Loading transformers model")
            # Use transformers pipeline
            device_id = 0 if self.device == "cuda" else -1
            self.model = pipeline(
                "text-generation",
                model="ahalt/event-attribute-extractor",
                device=device_id,
                torch_dtype="auto" if self.device == "cuda" else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("ahalt/event-attribute-extractor")
        elif self.backend == "mlx":
            try:
                from mlx_lm import load, generate
                from mlx_lm.sample_utils import make_sampler
            except ImportError:
                raise ImportError("mlx_lm is not installed. Please install it or use another backend.")
            
            if not self.silent: logger.debug("Loading MLX model")
            # MLX doesn't use device parameter the same way as PyTorch
            self.model, self.tokenizer = load("ahalt/event-attribute-extractor")
            # Store the generate function and create sampler
            self.mlx_generate = generate
            self.sampler = make_sampler(
                temp=0.5,           # temperature
                top_p=0.8,          # nucleus sampling
                top_k=20,           # top-k sampling
                min_p=0.0,          # minimum probability
                min_tokens_to_keep=1,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Must be 'vllm', 'transformers', or 'mlx'")

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
        event_rows = self.event_definitions.loc[self.event_definitions['event'] == event_type]
        event_def = event_rows['event_def'].values[0]
        # Get mode definition and extraction notes if they exist
        if 'event_mode' in event:
            if event['event_mode'] != "":
                if 'mode' in self.event_definitions.columns and 'mode_def' in self.event_definitions.columns:
                    mode_def = event_rows.loc[event_rows['mode'] == event['event_mode'], 'mode_def'].values[0]
                if 'extraction_notes' in self.event_definitions.columns:
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

        if self.backend == "vllm":
            # vLLM backend
            outputs = self.model.generate(prompts, sampling_params=self.sampling_params)
            responses = [i.outputs[0].text.strip() for i in outputs]
        elif self.backend == "transformers":
            # Transformers pipeline backend
            responses = []
            for prompt in prompts:
                output = self.model(
                    prompt,
                    max_new_tokens=1024,
                    temperature=0.5,
                    top_p=0.8,
                    top_k=20,
                    do_sample=True,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                # Extract the generated text
                generated_text = output[0]["generated_text"].strip()
                responses.append(generated_text)
        elif self.backend == "mlx":
            # MLX backend
            responses = []
            for prompt in prompts:
                # Generate with MLX
                output = self.mlx_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=1024,
                    sampler=self.sampler,
                    verbose=False,
                )
                # The output from mlx_lm.generate is a string
                responses.append(output.strip())
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

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
                event_list: list[AttributeModelInput] | list[dict[str, Any]]
                ) -> list[AttributeModelOutput] | list[dict[str, Any]]:
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
        if not self.silent: print("Making prompts...")
        prompts = [self.make_prompt(event) for event in tqdm(event_list, desc="Making prompts", disable=self.silent)]
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

        # The way this works currently, process() mutates the input lists and 
        # dicts, so technically there isn't really a need to return anything. 
        # An alternative would be to explicitly copy the input and return a new
        # list. Pros: probably more intuitive. Cons: more memory usage.
        # Or, just return a list of attributes, but add a shared key to both
        # the input list and output list to link them. 
        return cast(list[AttributeModelOutput], event_list)


if __name__ == "__main__":
    # add debug logging
    logging.basicConfig(level=logging.DEBUG)

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

    # Example: Use vLLM backend (default)
    am = AttributeModel(silent=False, gpu=True, backend="vllm")
    # Or use transformers backend:
    # am = AttributeModel(silent=False, gpu=True, backend="transformers")

    prompt = am.make_prompt(data[0])
    print(prompt)
    output = am.call_llm_batch(prompt)

    all_prompts = [am.make_prompt(event) for event in data]
    all_attributes = am.call_llm_batch(all_prompts)

    all_outputs = am.process(data)

    # clear the cuda cache
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()


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
