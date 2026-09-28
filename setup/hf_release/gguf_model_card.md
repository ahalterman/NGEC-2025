---
license: apache-2.0
language:
- en
base_model: ahalt/qwen3.5-event-extraction-0.8b
base_model_relation: quantized
tags:
- event-data
- political-science
- computational-social-science
- gguf
- llama.cpp
---

# qwen3.5-event-extraction-0.8b-GGUF

Q8_0 GGUF of [`ahalt/qwen3.5-event-extraction-0.8b`](https://huggingface.co/ahalt/qwen3.5-event-extraction-0.8b),
the attribute-extraction model for [NGEC](https://github.com/ahalterman/NGEC-2025)
(Next Generation Event Coder), for running on a CPU with llama.cpp. See that model's card
for what the model does, its evaluation, the prompt format, and `definitions.json`.

| File | Size | SHA-256 |
|---|---|---|
| `qwen3.5-event-extraction-0.8b-Q8_0.gguf` | 834 MB | `523b30de33e68b703c72b772e1e1c1aa3a46c2c379a4153468587a93d480e11a` |

Built from the bf16 weights with llama.cpp commit `60081bb` (2026-09-18):
`convert_hf_to_gguf.py --outtype f16`, then `llama-quantize ... Q8_0`. No importance
matrix was used.

## Quality and speed

On a 100-document subset of the 500-document VOA test set, this file scored 74.3 mean
F1, against 73.6 for the unquantized model on a GPU. Quantizing to 8 bits costs
nothing measurable. On an Intel i9-12900K with 8 threads, it takes 3.9 seconds per
document on average (90th percentile 5.6 seconds) and uses about 2.5 GB of memory.
Set the thread count to the number of performance cores. On this CPU, letting the
library use every core made the model much slower.

## Usage with llama.cpp

```shell
llama-server -m qwen3.5-event-extraction-0.8b-Q8_0.gguf --port 8080 -c 8192 -t 8
```

The tested way to use it: render the prompt in Python with the base model's tokenizer,
and post the resulting string to llama-server's `/completion` endpoint with greedy
decoding. `system_message`, `user_message` and `definitions` are the helpers on the
base model's card.

```python
import json
import urllib.request
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ahalt/qwen3.5-event-extraction-0.8b")

messages = [
    {"role": "system", "content": system_message("ASSAULT")},
    {"role": "user", "content": user_message(text, definitions[("ASSAULT", "")])},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

request = urllib.request.Request(
    "http://127.0.0.1:8080/completion",
    data=json.dumps({"prompt": prompt, "n_predict": 768, "temperature": 0.0,
                     "top_k": 1, "stop": ["<|im_end|>"]}).encode(),
    headers={"Content-Type": "application/json"})
records = json.loads(json.load(urllib.request.urlopen(request))["content"])
```

The chat template is embedded in the GGUF, so llama-server's `/v1/chat/completions`
endpoint should also work if you pass `"chat_template_kwargs": {"enable_thinking":
false}`. That path has not been checked against the Python rendering. The numbers
above were produced with `/completion`.
