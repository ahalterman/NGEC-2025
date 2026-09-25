# Coding a corpus with the pretrained models

This is the most common request: "I have a folder/CSV of news articles, give me
event data." It needs the full install, including Elasticsearch
(`ngec guide setup`).

## Before the full run

1. **Get the input into shape.** One dict per story with `id`, `event_text`
   and `pub_date` (`YYYY-MM-DD`). Check for empty texts, duplicate ids and
   missing dates. Ask the user what their date column means if it is not
   obviously the publication date.
2. **Run 20 stories first** and time it. Extrapolate before starting the rest,
   and tell the user the estimate. The first run also downloads and compiles
   models, so time the second batch, not the first.
3. **Look at the output with the user** (see "Checking the output" below)
   before coding thousands of stories.

## The script

```python
import json

import pandas as pd

from ngec import events_to_table
from ngec.es_client import setup_es_client
from ngec.logging import quiet_third_party_loggers
from ngec.plover_coder import PloverCoder

quiet_third_party_loggers()

# Read the stories. Adjust the column names to the user's file.
stories = pd.read_csv("stories.csv")
story_list = [{"id": str(row.id),
               "event_text": row.text,
               "pub_date": str(row.date)[:10]}   # YYYY-MM-DD
              for row in stories.itertuples()]

es_client = setup_es_client(hosts=["localhost"], port=9200)

# On Linux with an NVIDIA GPU: attribute_backend="vllm", gpu=True.
# Elsewhere: attribute_backend="transformers" (add gpu=True if there is a GPU).
coder = PloverCoder(es_client=es_client, attribute_backend="vllm", gpu=True)

# Code in batches and write each batch out as it finishes, so a crash
# halfway through a long run does not lose the finished part.
all_events = []
batch_size = 500
with open("events.jsonl", "w", encoding="utf-8") as f:
    for start in range(0, len(story_list), batch_size):
        events = coder.process(story_list[start:start + batch_size])
        for event in events:
            # default=str: resolved dates are datetime objects
            f.write(json.dumps(event, default=str) + "\n")
        all_events.extend(events)
        done = min(start + batch_size, len(story_list))
        print(f"{done} of {len(story_list)} stories done, {len(all_events)} events so far")

# One row per event, for R, Stata or a spreadsheet.
events_to_table(all_events).to_csv("events.csv", index=False)
```

Load the models once (one `PloverCoder`) and reuse it for every batch; creating
it takes a minute or more. `vllm` reserves 80% of the GPU's memory when it
starts; if the GPU is shared, pass `max_gpu_memory=0.5` or similar.

To restart after a crash, skip the ids already in `events.jsonl`. Note that a
story with no events leaves no record, so "not in the file" also covers
stories that were coded and yielded nothing. Keep a separate list of the
batches that finished if that distinction matters.

## What comes out

One record per event (see `ngec guide` for the fields). `events_to_table`
keeps the columns most analyses need:

| Column | Meaning |
|---|---|
| `story_id` | the input `id` |
| `event_type`, `event_mode` | PLOVER event category and sub-type |
| `anchor_quote` | the passage the event was found in |
| `actor_text`, `actor_code`, `actor_wiki` | spans, codes such as `FRA GOV` or `CVL OPP`, and linked Wikipedia titles, `; `-separated and in the same order |
| `recipient_...` | the same for recipients |
| `date`, `date_granularity`, `date_type`, `date_text` | resolved date, its precision, and the span it came from |
| `location_name`, `location_country`, `lat`, `lon`, `geonameid`, `location_text` | geocoded location and its span |
| `killed_text`, `injured_text` | ASSAULT, PROTEST and COERCE only |

An empty code means the actor was found in the text but could not be
categorized; an empty `date` means the date span could not be resolved (or
there was none). The full records in `events.jsonl` say why (`reason` fields).

## Checking the output

The pretrained models were validated on other text, not the user's. Before the
user analyses counts, help them read a random sample:

```python
table = events_to_table(all_events)
sample = table.sample(20, random_state=1)
```

For each row, print the story text next to the event type, actor, recipient,
date and location, and ask the user whether each is right. Things to look for:

- event types the classifier over- or under-detects on this kind of text,
- actors coded with the wrong category, or linked to the wrong Wikipedia page,
- dates resolved to the publication date when the event happened earlier.

If one step is consistently wrong, `ngec guide customize` covers fixing it. For
debugging a specific step, `PloverCoder(save_intermediate=True,
intermediate_dir="debug/")` writes each step's output to its own JSONL file;
events the attribute model found nothing for go to a `*_dropped_events.jsonl`
file there.
