# Coding a corpus with the pretrained models

This is the most common request: "I have a folder/CSV of news articles, give me
event data." It needs the full install, including Elasticsearch
(`ngec guide setup`).

## Before the full run

1. **Get the input into shape** (next section). Check for empty texts,
   duplicate ids and missing dates. Ask the user what their date column means
   if it is not obviously the publication date.
2. **Run 20 stories first** and time it. Extrapolate before starting the rest,
   and tell the user the estimate. The first run also downloads models, so
   time the second batch, not the first. Almost all the time goes to attribute
   extraction; `ngec guide setup` has typical speeds per backend.
3. **Look at the output with the user** (see "Checking the output" below)
   before coding thousands of stories.

## Preparing the input

Each story needs `id`, `event_text` and `pub_date`. Three things go wrong
quietly:

- **Dates must be parsed, not cast to strings.** `str(row.date)` turns a blank
  cell into the string `"nan"`, and a date like `03/04/2019` is read month
  first whatever the user meant. Parse the column with an explicit format,
  write `YYYY-MM-DD`, and use `None` for missing dates.
- **There may be no id column.** Make one, and keep the user's own columns
  (URL, source, ...) in a separate table to merge back on `story_id` later.
- **Headline and body** are usually separate columns. Join them: the
  headline often names the event and its actors.

```python
import pandas as pd

articles = pd.read_csv("articles.csv")
# Ask the user which format the dates are in; this one is 03/15/2019.
articles["pub_date"] = pd.to_datetime(articles["date"], format="%m/%d/%Y", errors="coerce")
articles["story_id"] = [f"story{i}" for i in range(len(articles))]

story_list = []
for row in articles.itertuples():
    story_list.append({
        "id": row.story_id,
        "event_text": f"{row.headline}\n\n{row.body}",
        "pub_date": row.pub_date.strftime("%Y-%m-%d") if pd.notna(row.pub_date) else None,
    })

# Everything else about each article, to merge back onto the events later.
articles[["story_id", "url"]].to_csv("story_info.csv", index=False)
```

## The script

```python
import json

from ngec import events_to_table
from ngec.es_client import es_client_from_env
from ngec.logging import quiet_third_party_loggers
from ngec.plover_coder import PloverCoder

def main():
    quiet_third_party_loggers()

    es_client = es_client_from_env()   # reads ES_HOST etc. from .env, else localhost:9200

    # See `ngec guide setup` for choosing attribute_backend on this machine.
    coder = PloverCoder(es_client=es_client)

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

    # One row per event.
    table = events_to_table(all_events)
    table.to_csv("events.csv", index=False)
    table.to_stata("events.dta", write_index=False)   # for Stata


# The vllm backend starts a subprocess that re-imports this file. This line
# keeps it from running the pipeline a second time, and the function lets the
# models be freed so the script exits when it finishes.
if __name__ == "__main__":
    main()
```

Load the models once (one `PloverCoder`) and reuse it for every batch; creating
it takes a minute or more. With `attribute_backend="vllm"`, vLLM reserves 80% of
the GPU's memory when it starts; if the GPU is shared, pass
`max_gpu_memory=0.5` or similar.

To restart after a crash, skip the ids already in `events.jsonl`. Note that a
story with no events leaves no record, so "not in the file" also covers
stories that were coded and yielded nothing. Keep a separate list of the
batches that finished if that distinction matters.

## What comes out

One record per event (see `ngec guide` for the fields). `events_to_table`
keeps the columns most analyses need:

| Column | Meaning |
|---|---|
| `id`, `story_id`, `pub_date` | the event's id, the input `id`, the input date |
| `event_type`, `event_mode` | PLOVER event category and sub-type (`PROTEST`, `demo`) |
| `anchor_quote` | the passage the event was found in |
| `actor_text`, `actor_code`, `actor_wiki` | spans, codes such as `FRA GOV` or `CVL OPP`, and linked Wikipedia titles, `; `-separated and in the same order |
| `recipient_...` | the same for recipients |
| `date`, `date_granularity`, `date_type`, `date_text` | resolved date, its precision, whether it was resolved, and the span it came from |
| `location_name`, `location_country`, `location_admin1`, `lat`, `lon`, `geonameid`, `location_text` | geocoded location and its span |
| `killed_text`, `injured_text` | ASSAULT, PROTEST and COERCE only |

**Dates.** Filter on `date_type`, not on whether `date` is empty. When the
story gives no date, or one that cannot be resolved, `date` is the
publication date and `date_type` is `unresolved`; `date` is empty only when
there was no `pub_date`. `date` is always a full date, even when the text only
gave a month or a year: "in March" becomes a day in March with
`date_granularity` = `month`. Aggregate to periods no finer than the
granularity, or drop events less precise than the analysis needs.

**Actor codes.** `actor_code` is `country code_1 code_2`, e.g. `KEN COP`
(Kenyan police) or `CVL OPP` (civilian opposition, country unknown). The
country is often blank for a generic span ("teachers"), because it comes from
the span itself or the actor's Wikipedia article, not from the rest of the
story. An empty code means the actor was found in the text but could not be
categorized. The `code_1` values, with typical patterns from the agents file
that produce them (the PLOVER documentation has the formal definitions):

| Code | Actors | Code | Actors |
|---|---|---|---|
| `GOV` | government, ministries, officials | `MIL` | armed forces, officers |
| `COP` | police, border guards | `JUD` | courts, judges, lawyers |
| `LEG` | parliaments, assemblies | `PTY` | political parties, candidates |
| `ELI` | former officials, first families | `SPY` | intelligence services |
| `REB` | rebels, coup leaders | `PRM` | paramilitaries, private armies |
| `UAF` | armed groups not otherwise identified | `CRM` | criminals, kidnappers, traffickers |
| `CVL` | civilians, residents, crowds | `REF` | refugees, displaced people |
| `REL` | clergy, religious groups | `EDU` | students, academics, schools |
| `LAB` | workers, unions | `AGR` | farmers, herders |
| `BUS` | companies, businesspeople | `MED` | health workers, hospitals |
| `JRN` | journalists, media | `NGO` | non-governmental organizations |
| `SOC` | civil society, activists | `IGO` | international organizations, peacekeepers |
| `PRE` | unrecognized states | `UNK` | too generic to place ("group", "driver") |
| `JNK`, `NON` | not actors (disasters, places, dates); treat as errors | | |

`code_2` refines `code_1`; the most common is `OPP` (opposition), as in
`CVL OPP` or `PTY OPP`.

**Locations.** `location_*` is empty when the location span could not be
matched to a place the geoparser found in the story (`event_location["reason"]`
in the full record says why). To filter events to one country, don't rely on
`location_country` alone: also look at the actors' countries, and at the places
the geoparser found in the story (`geolocated_ents` in the full records).

## Counting events

Rows are not events in the ordinary sense. One story can yield several rows
for what a reader would call one event: once per mode the classifier detected
(`demo` and `obstruct` for one march), and the rows for different modes can
have different anchor quotes and actors. There is no automatic deduplication.
For counts, the most defensible unit is usually **story × event type** ("this
story reports at least one protest"):

```python
protest_stories = table[table["event_type"] == "PROTEST"].drop_duplicates(["story_id", "event_type"])
```

Discuss the unit with the user before they count anything, and say which one
was used in the write-up.

## Checking the output

The pretrained models were validated on other text, not the user's. Before the
user analyses counts, help them read a random sample:

```python
sample = table.sample(20, random_state=1)
texts = {s["id"]: s["event_text"] for s in story_list}
for row in sample.itertuples():
    print(texts[row.story_id][:600], "\n")
    print(f"  {row.event_type} ({row.event_mode}) | actor: {row.actor_text} [{row.actor_code}]"
          f" | recipient: {row.recipient_text} [{row.recipient_code}]")
    print(f"  date: {row.date} ({row.date_granularity}, {row.date_type}) | place: {row.location_name}\n")
```

Ask the user whether each is right. Things to look for:

- event types the classifier over- or under-detects on this kind of text,
- actors coded with the wrong category, or linked to the wrong Wikipedia page
  (a group named after a person linked to the person, a place name linked as
  an actor),
- dates resolved to the publication date when the event happened earlier.

If one step is consistently wrong, `ngec guide customize` covers fixing it and
validating the result. For debugging a specific step,
`PloverCoder(save_intermediate=True, intermediate_dir="debug/")` writes each
step's output to its own JSONL file; events the attribute model found nothing
for go to a `*_dropped_events.jsonl` file there.
