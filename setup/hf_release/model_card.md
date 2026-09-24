---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.5-0.8B
tags:
- event-data
- political-science
- computational-social-science
---

# qwen3.5-event-extraction-0.8b

This is the attribute-extraction model for
[NGEC](https://github.com/ahalterman/NGEC-2025) (Next Generation Event Coder),
a pipeline for generating custom political event datasets.  You give it a news
document a definition of a political event type reported in the document. It
returns every instance of that event type in the document as a JSON list. Each
record has an anchor quote, a sub-event (mode), and the actor, recipient, date,
and location as verbatim spans from the text. It can also return other attributes,
such as the number of people killed or injured in violent events. It returns `[]`
when the event type does not occur in the document.

It replaces [`ahalt/qwen3-event-extraction-exp5.1`](https://huggingface.co/ahalt/qwen3-event-extraction-exp5.1)
(the model in the submitted paper). On a 500-document test set of new VOA articles, its
mean F1 is 70.0, against 54.9 for exp5.1. See "Evaluation" below.

**It uses a different prompt and output format from exp5.1.** Multiple spans
for each attribute are now proper JSON lists instead of semi-colon separated, and the
specific prompt has also changed.

A Q8_0 GGUF for CPU inference with llama.cpp is at
[`ahalt/qwen3.5-event-extraction-0.8b-GGUF`](https://huggingface.co/ahalt/qwen3.5-event-extraction-0.8b-GGUF).

## Prompt format

The prompt consists of a system message and a user message, formatted using the existing `tokenizer`'s chat format.

**System message.** A fixed instruction. Optional attributes add a key to the record
schema, a sentence about verbatim spans, and a description block. `killed` and
`injured` are always on for ASSAULT, PROTEST and COERCE, and off for the other types.
`reporter` (the source the document attributes the report to) is off by default and
can be turned on for any type. The function below reproduces the exact training-time
text.

**User message.**

```
## Document: {document}

## Event Type: {definition}

Return the JSON list.
```

`{definition}` is the event type's definition from the codebook. It should not be
an event label on its own, since this is not a well defined task ([Halterman and Keith 2026](https://aclanthology.org/2026.acl-long.92/). 
The `definitions.json` in this repository has the exact definition strings
that the model saw in training from the PLOVER ontolgy. You can also write a definition for an event
type the model was *not* trained on, so you can code custom event types (see "Evaluation").

`{document}` is the article text.  The training documents were VOA articles of
up to about 1,000 words, with the headline first, followed by a period.

```python
import json
import re

CORE_SYSTEM = (
    "Extract every instance of the given political event type from the document "
    "as a JSON list, or [] if there is none.\n\n"
    "Each record:\n"
    '{{"event_type": "<type>", "mode": "<sub-event name or empty string>",\n'
    ' "anchor_quote": "<short verbatim passage identifying this instance>",\n'
    ' "actor": [...], "recipient": [...], "date": [...], "location": [...]{extra_keys}}}\n\n'
    "Every string in actor, recipient, date, and location must be an exact, "
    "verbatim substring of the document.{extra_verbatim}\n"
    "{extra_block}"
    "Return the JSON list only."
)

ATTRIBUTE_TEXT = {
    "killed": (
        "Every person or group the document reports as killed in this event, whether "
        "or not they were its target. Fill it whenever deaths are reported, even when "
        "the dead are also the recipient. One list element per distinct party, counts "
        "kept ('three officers'). Empty when no death is reported. Written like a "
        "recipient: the verbatim noun phrase, counts and quantifiers kept, leading "
        "articles dropped, one list element per distinct party."),
    "injured": (
        "Every person or group the document reports as injured in this event, whether "
        "or not they were its target. Fill it whenever injuries are reported, even when "
        "the injured are also the recipient. One list element per distinct party, "
        "counts kept ('three officers'). Empty when no injury is reported. Written like "
        "a recipient: the verbatim noun phrase, counts and quantifiers kept, leading "
        "articles dropped, one list element per distinct party."),
    "reporter": (
        "The source the document attributes this report to: a named outlet, an "
        "official statement, eyewitnesses, unnamed officials, a spokesperson. Verbatim "
        "span of the attributing phrase's subject ('the defense ministry', "
        "'witnesses', 'Reuters'). Empty when the report is in the publication's own "
        "voice. Written like a recipient: the verbatim noun phrase of the attributing "
        "subject, leading articles dropped, one list element per distinct source."),
}

CASUALTY_TYPES = {"ASSAULT", "PROTEST", "COERCE"}


def system_message(event_type, reporter=False):
    attrs = ["killed", "injured"] if event_type in CASUALTY_TYPES else []
    if reporter:
        attrs.append("reporter")
    if not attrs:
        return CORE_SYSTEM.format(extra_keys="", extra_verbatim="", extra_block="")
    extra_keys = "".join(f', "{a}": [...]' for a in attrs)
    names = attrs[0] if len(attrs) == 1 else ", ".join(attrs[:-1]) + " and " + attrs[-1]
    extra_block = ("ADDITIONAL ATTRIBUTES FOR THIS EVENT TYPE\n"
                   + "".join(f"- {a}: {ATTRIBUTE_TEXT[a]}\n" for a in attrs))
    return CORE_SYSTEM.format(extra_keys=extra_keys,
                              extra_verbatim=f" The same holds for {names}.",
                              extra_block=extra_block)


def user_message(document, definition):
    document = re.sub(r"\s+", " ", document).strip()
    return f"## Document: {document}\n\n## Event Type: {definition}\n\nReturn the JSON list."


definitions = {(d["event_type"], d["mode"]): d["definition"]
               for d in json.load(open("definitions.json"))}
```

## Example usage with vLLM

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "ahalt/qwen3.5-event-extraction-0.8b"
model = LLM(model=MODEL, enable_prefix_caching=True, max_model_len=8192)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Greedy decoding, which is how the model was evaluated
sampling_params = SamplingParams(temperature=0.0, max_tokens=768)


def make_prompt(document, event_type, mode="", reporter=False):
    messages = [
        {"role": "system", "content": system_message(event_type, reporter)},
        {"role": "user", "content": user_message(document, definitions[(event_type, mode)])},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)


text = """Sexual Violence a Continuous Threat in Haiti, Amnesty International Says. Amnesty International says sexual violence against women in Haiti is increasing one year after a deadly earthquake forced hundreds of thousands of people into makeshift shelters with little or no security. Amnesty said in a report Thursday the offenses are primarily committed by armed men roaming tent camps at night. The rights group says more than 250 rapes occurred in camps in the first 150 days after last January's earthquake. Amnesty is urging the newly elected government to include the topic of sexual violence in its plan to address the humanitarian crisis. The group says women should have input in developing an action plan. The rights group says immediate assistance should include security in the camps and help for police investigating cases."""

output = model.generate(make_prompt(text, "ASSAULT"), sampling_params=sampling_params)
records = json.loads(output[0].outputs[0].text)

# [{"event_type": "ASSAULT", "mode": "sexual",
#   "anchor_quote": "more than 250 rapes occurred in camps in the first 150 days after last January's earthquake",
#   "actor": ["armed men"], "recipient": ["women"],
#   "date": ["in the first 150 days after last January's earthquake"],
#   "location": ["in Haiti"], "killed": [], "injured": []}]
```

The output shown is the model's actual output for this document, produced by
the Q8_0 GGUF. The document is from the 500-document test set described below.

The model may return several records for one document, one per instance of the event.
On the 500 test documents, every output was valid JSON. 

## Evaluation

Scores are mean F1 over the four core roles (actor, recipient, date, location),
counting a predicted span as correct when it matches the answer key under the
project's span-matching rules. exp5.1, the previous model, was run with its
own prompt and output format; the two models' outputs were scored with the same code
on the same documents.

| Test set | exp5.1 | This model |
|---|---|---|
| 500 VOA documents (gold500_a) | 54.9 | 70.0 |
| Held-out 50 documents (earlier answer key; a rebuilt key is pending) | 65.2 | 72.5 |
| Original 100 human-coded ASSAULT documents, actor only (any annotator's span) | 73.6 | 88.5 |

On the 500 documents, the gain is 15 points, with a 95% interval of +11 to +20.
This model gets 54 articles entirely right (exp5.1: 5). It misses events on 26
articles (exp5.1: 170) and finds 69% of all events (exp5.1: 26%). It swaps the
actor and the recipient in 1.7% of events (exp5.1: 4.6%).

The answer keys for gold500_a and the held-out set were drafted by three models
and adjudicated by a fourth model and a human spot check of the adjudicated
keys. The original 100 documents were coded by three human annotators.

**Definitions of new event types.** In a test run, SANCTION was removed from the
training data, and the model was then given SANCTION's definition at inference time.
On the SANCTION events it reported, its F1 over the four roles was 74.6. A model trained
on SANCTION scored 74.0. The held-out type was harder to notice: the model said
"nothing here" on 38% of documents that did contain a SANCTION event, against 19% for
the model trained on it. Giving a bare type name instead of the definition costs 7 to
10 points.

**Speed and size.** On an Intel i9-12900K (8 threads), the Q8_0 GGUF takes 3.9
seconds per document on average, about 1,040 prompt tokens and 150 output tokens.
On a 100-document subset of gold500_a, the Q8_0 GGUF scored 74.3, against 73.6 for
these bf16 weights on a GPU, so quantization to 8 bits has no measurable effect. On one
GPU with vLLM, the model processes about 37 events per second.

## Training

The model was fine-tuned from `Qwen/Qwen3.5-0.8B` (full fine-tune, no LoRA) on 10,600 training units.
A unit is one document paired with one event type or sub-event, with either the
records to extract or `[]`. The units come from 1,000 VOA news articles (5,404 units)
and 1,008 synthetic news stories (5,196 units). 2,907 units are positive and 7,457
are negatives whose answer is `[]`. The other 236 are decoys: synthetic stories that
plant a similar event of a different sub-event.

The training labels were written by a teacher model, GLM 5.3 flash, prompted with the
consolidated PLOVER codebook of 2026-09-17 and the project's written labeling
decisions.

Training hyperparameters: two epochs, learning rate 2e-5 with cosine decay and
10% warmup, weight decay 0.01, effective batch size 12, maximum length 4,096
tokens, seed 1. Training took 42 minutes on one RTX 4090. Of the student models
tried, this one had the best trade-off between accuracy and CPU speed.
Qwen3-1.7B, trained the same way, scored 73.2 on gold500_a but takes 8 seconds
per document on a CPU.

## Limitations

- **Dates and locations include their preposition.** They come back as "in Haiti" or
  "on Monday", because that is how the training labels were written. Strip the
  preposition if your downstream code expects bare names.
- **Recipients are harder than actors.** The model only extracts recipients that the
  text names. It will not infer an unstated target (for example, "the government" as
  the target of a protest over a policy).
- **Multi-event articles.** In an article describing several events of the same type,
  the model returns each as a separate record. If you need one particular event, pass
  only the passage that describes it.
- **English news only.** Training and evaluation used English-language news.

## Citation

The article describing NGEC is under review. Until it is published, please cite the
repository: https://github.com/ahalterman/NGEC-2025
