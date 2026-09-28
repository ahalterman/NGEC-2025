# NGEC -- the Next Generation Event Coder

<table border="0">
<tr>
<td width="300"><img src="docs/ngec_logo.jpg" alt="The NGEC logo, a stack of newspapers being fed into a lego machine with small boxes coming out on the other side, in a Modernist style" width="300"></td>
<td>

NGEC (Next Generation Event Coder) turns news stories into political event
data. 


For each story, NGEC:

1. **Detects event types** in news stories with a classifier.
2. **Extracts the event's attributes**: who did it (the actor), to whom (the
   recipient), when, where, and optionally things like how many people were
   killed. A small language model does this locally, with no API
   key or per-story cost.
3. **Categorizes the actors and recipients** into categories (government, military, civilians,
   ...). For named people and organizations, it looks them up in a local copy
   of Wikipedia to find out who they are.
4. **Resolves** the date to a calendar date and the place names to coordinates using
   the GeoNames gazetteer.

</td>
</tr>
</table>




<br clear="left">

A note on "ontologies": all event data projects begin with an ontology, which defines the event types to code,
how events are represented, and the way that actors and recipients should be categorized. By default, NGEC
uses a common event ontology ([PLOVER], similar to the older CAMEO ontolgy)(https://github.com/openeventdata/plover).
However, NGEC is designed to be used with custom ontologies or codebooks that define different types of events and political actors.

The event classifiers that come with NGEC are demonstration models; for
research data you will probably want to train your own (see [Before coding a
corpus](#before-coding-a-corpus)).

A much earlier version of this pipeline
produced the [POLECAT](https://dataverse.harvard.edu/dataverse/POLECAT) dataset.


**Table of Contents** 

- [Quickstart](#quickstart)
- [Letting a coding agent set it up](#setting-up-with-a-coding-agent)
- [Running NGEC on your own stories](#coding-your-own-stories)
- [Creating custom event data](#customizing-ngec)
- [Using single steps](#using-single-steps)
- [Before coding a corpus](#before-coding-a-corpus)
- [When something goes wrong](#when-something-goes-wrong)
- [FAQs](#faqs)
- [More documentation](#more-documentation)

## Quickstart

You need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs
  Python and NGEC. See [`docs/INSTALL.md`](docs/INSTALL.md#installing-with-pip) for
  instructions on installing with pip.
- [Docker](https://www.docker.com/get-started/), which runs Elasticsearch, the
  search engine that holds NGEC's local copy of Wikipedia and GeoNames (steps 3
  and 4 above).
- About 20 GB of free disk space and an hour, mostly waiting for downloads.

Then, in a terminal:

```shell
uv init --python 3.12 my-ngec          # make a new project folder
cd my-ngec
# download this repo and install
uv add "ngec[cpu,llamacpp] @ git+https://github.com/ahalterman/ngec-2025"
uv run ngec download-models            # download about 4 GB of models
uv run ngec download-index --start     # 11.6 GB index; starts Elasticsearch in Docker
uv run ngec doctor --smoke             # checks everything, then codes three stories
```

By default, `ngec[cpu,llamacpp]` installs a version optimized for
laptops without an Nvidia GPU. This is fine for testing, but for better
speed, you should plan to run on a computer with a GPU or on a recent Mac.
To install the versions for those systems, change the options in brackets 
using the table below:

| Your computer | Install |
|---|---|
| No NVIDIA GPU (most laptops, Windows or Linux) | `ngec[cpu,llamacpp]` |
| Mac with Apple Silicon (M1 or newer) | `ngec[mlx]` |
| Linux with an NVIDIA GPU | `ngec[cu12,vllm]` |
| Windows with an NVIDIA GPU | `ngec[cu12,llamacpp]` |

What the commands do:

- **`ngec download-models`** downloads the language models NGEC uses (spaCy,
  three sentence encoders, and the model that extracts event attributes) 
  so that NGEC starts up quickly when you first run it.
- **`ngec download-index --start`** downloads the Wikipedia and GeoNames index to
  `~/ngec-es-data`, checks it, unpacks it, and starts Elasticsearch on it on
  port 9200. It is the slowest step, so you can run it in a second terminal
  while `download-models` runs. If the download is interrupted, run it again and
  it carries on where it stopped. Elasticsearch keeps running in Docker after
  you close the terminal, and starts again with Docker after a reboot.
- **`ngec doctor --smoke`** checks the installation and then codes three news
  stories. It takes a few minutes on a laptop. If it prints events, then you've
  installed everything correctly!

**Not everything needs Elasticsearch.** The Elasticsearch indices are the heavy 
lift for setup, but only geocoding and Wikipedia-based actor
resolution use it. If you only need event types, to extract attribute text,
date resolution or actor categories for generic descriptions ("Kenyan police",
"protesters"), you can skip `download-index`. However, coding named actors ("Emmanuel Macron")
needs the Wikipedia index.

[`docs/INSTALL.md`](docs/INSTALL.md#do-you-need-all-of-it) has more details
on which downloads you'll need for each.

## Setting up with a coding agent

If you use Claude Code, Codex or another coding agent, it can do the installation
for you. Open it in an empty folder and ask it to:

> Install NGEC in this folder, following
> https://raw.githubusercontent.com/ahalterman/ngec-2025/main/ngec/assets/guide/setup.md

That guide is written for agents. It tells them to use `ngec doctor` to
find what is missing, to install only what your goal needs and to ask you before
running each fix. It'll still ask you to install system software such as Docker.

Once NGEC is installed, run this in your project folder:

```shell
uv run ngec guide --init
```

This adds a short section to the project's `AGENTS.md` (creating it if needed)
that tells any agent working there to read NGEC's guide first. The guide comes
with the package, so it always matches the version you have installed. It covers
installing, coding a corpus, using single steps, and using your own actor
categories, event definitions or classifier.

Codex and most other agents read `AGENTS.md` on their own. Claude Code reads it
when the folder has no `CLAUDE.md`. If yours has one, add a line to it that
says `@AGENTS.md`.

## Coding your own stories

Here's an example of how to code stories using the built-in PLOVER ontology. If you
want to code different kinds of events, see [Customizing NGEC](#customizing-ngec).

Each story needs an `id`, the `event_text`, and a `pub_date`, which is used to
resolve relative dates like "today" or "last Tuesday" and make sure that people
are assigned to the job they had when the event took place:

```python
from ngec import events_to_table
from ngec.es_client import es_client_from_env
from ngec.logging import quiet_third_party_loggers
from ngec.plover_coder import PloverCoder


def main():
    quiet_third_party_loggers()

    coder = PloverCoder(es_client=es_client_from_env())

    stories = [
        {"id": "story1",
         "event_text": "Protesters were in the streets in Paris again today to protest "
                       "against the Hollande government's austerity measures.",
         "pub_date": "2016-05-01"},
    ]
    events = coder.process(stories)

    # create a simplified table output
    table = events_to_table(events)
    table.to_csv("events.csv", index=False)


# On a GPU, NGEC starts a second process that re-reads this file. This line
# keeps that process from running the pipeline again.
if __name__ == "__main__":
    main()
```

Save the code above as a file in your project folder (e.g. `code_events.py`) and run it with
`uv run python code_events.py`. `events.csv` has one row per event.

```
id                story1_PROTEST_demo_0
story_id          story1
pub_date          2016-05-01
event_type        PROTEST
event_mode        demo
anchor_quote      Protesters were in the streets in Paris again today to protest against the Hollande government's austerity measures
date_text         today
date              2016-05-01
date_granularity  day
date_type         exact
location_text     Paris
killed_text
injured_text
location_name     Paris
location_country  FRA
location_admin1   Île-de-France
lat               48.85341
lon               2.3488
geonameid         2988507
actor_text        Protesters
actor_code        CVL OPP
actor_wiki
recipient_text    Hollande government
recipient_code    FRA GOV
recipient_wiki    François Holland
```

- **`event_type` and `event_mode`** are the PLOVER event type and its mode
  (here, `demo`, a demonstration, as opposed to e.g. a riot or a strike). A story can include
  several different event types, and more than one instance of each event type could be present. (For example,
  a news story could report two separate demonstrations). The attribute model
  will return separate records for each separate instance of an event type reported in a story.
- **`actor` and `recipient`** are who carried out the event and who it was
  directed at (the "source" and "target" in older CAMEO-coded data).
- **`anchor_quote`** is the passage the attribute model identified as the 
  best short span describing the event (though it can identify information
  from elsewhere in the story). `*_text` columns are the exact spans the model identified
  as reporting the actor, recipient, date, and location of the event.
- **`date`** is the resolved calendar date. `date_granularity` is how
  precise it is (day, week, month, quarter, year) and `date_type` whether it is exact,
  approximate or a range.
- **`location_`, `lat`, `lon` and `geonameid`** are the GeoNames information for
  the event's location. If the geoparser ([mordecai3](https://github.com/ahalterman/mordecai3/) is not confident, it will
  leave these blank.
- **`actor_code`/`recipient_code`** are the PLOVER actor categories. A code can
  have several parts: `CVL OPP` is civilians (`CVL`) in the opposition (`OPP`),
  and `GOV` is the government. 
- **`actor_wiki`/`recipient_wiki`** report the
  Wikipedia page when the actor is a named person or organization.

The `events` object itself is a list of Python dictionaries that have more detail than the
simplified table output, including the classifier's confidence, every place the geoparser found in
the story, and why each date and location was or wasn't resolved. We recommend
working with the full `events` object in production.  [`PIPELINE.md`](PIPELINE.md) documents
all of the fields.

## Customizing NGEC

One of the main objectives of NGEC is to make it easier for researchers to create
custom event data: changing the event types it codes, extracting other attributes
for events, and categorizing actors in different ways from the pre-built PLOVER ontology
it uses by default.

For example, say you're interested in coding legislative events, which aren't
part of the default PLOVER ontology. If you wanted to identify `INTRODUCE_BILL`
events, you would write new definitions for the event type, retrain the event
classifiers to detect your new event types (a few hundred labeled stories per
type), and run the attribute model using your new event types. The attribute
model can extract information for event types it didn't see during training, so
you can use it as-is with new event definitions.  In many cases, the attribute
model can also extract new *attributes* (e.g., `num_cosponsors`)  without
retraining, but this will require testing it on your corpus.

You can also change the *entity classification* step. To use the same example,
PLOVER's default `LEG`islative category isn't useful for researchers studying different
parties. Instead, you could define  `DEM` and `REP` entity categories by
writing a new file of example descriptions ("Republican", "GOP", "conservative" $\rightarrow$ REP)
and NGEC will use these to categorize actors using your new groupings.

[`ngec/assets/guide/customize.md`](ngec/assets/guide/customize.md) (also
`uv run ngec guide customize`) covers each of these with code, including the definition
format, a worked example of training a classifier from your own labels, the
agents and priorities file formats, and how to validate the customized output
against hand-coded stories. 

## Using single steps

If you don't need a full end-to-end event coding pipeline, you can also
use components of NGEC on their own.

For example:

- If you have a list of armed group names and want to link them to a canonical
  identifier, you can use the Wikipedia matcher to resolve different versions
of each name to their Wikipedia article title.
- If you have a set of politicians and want to extract information about their
  political history, you can run the Wikipedia matcher and infobox parser to
extract their past offices held.
- If you need to resolve raw dates to calendar dates, you can use just the date
  resolver.
- If you have an existing corpus of stories labeled with event types, you can
  enrich the dataset by extracting the events' attributes without needing to
  train an event detector. 

[`ngec/assets/guide/pieces.md`](ngec/assets/guide/pieces.md) (also
`uv run ngec guide pieces`) has a short example of each, and explains how to
chain a few steps without running the whole pipeline.

## Before coding a corpus

**The event classifiers are demonstration models.** These are not the models
that produced POLECAT. They were trained on Voice of America stories labeled by
an LLM applying the PLOVER codebook to generate some example training data. For
your own event data, you'll probably want to train your own; see
[`CLASSIFIERS.md`](CLASSIFIERS.md). [Customizing NGEC](#customizing-ngec)
describes how you can use your own event classifiers.

**Speed.** Most of the time it takes to run NGEC comes from the attribute extraction model, which runs
once for each event type and mode detected in a story. On a desktop CPU, that
takes about 4 seconds each, so a story with three event types takes about 14 seconds. On a
Linux machine with an NVIDIA GPU (`ngec[cu12,vllm]`), it is many times faster (around 20-40 prompts per second).
You should time a batch of 20 stories before planning a large run.
`ngec guide run` has a script for coding a whole corpus in batches, and
[`RUNNING.md`](RUNNING.md) describes what to expect at scale.

### Reproducibility


The coded output depends on the NGEC version, the attribute model, the event
classifiers, the Wikipedia/GeoNames index, and the backend that runs the model.
You should keep two files with your data to make it replicable: the `uv.lock`
in your project folder, which records the exact NGEC commit and every package
version, and the output of `uv run ngec doctor --json`, which records the
attribute model, your settings and the size of the index. NGEC logs which
backend it chose when it starts. `ngec update` tells you when newer models or a
newer index have been published but won't update unless you pass `--apply`.
Updating mid-project changes the output, so you may want to keep what you
started with.

## When something goes wrong

```shell
uv run ngec doctor
```

This checks the installation in a few seconds, without running the pipeline. It checks
the installed version, the settings NGEC reads, whether PyTorch can use your
GPU, and whether Elasticsearch is running with both indices. Anything it flags
is listed at the bottom with the command that fixes it. `--json` gives the same
report in a form to paste into a
[GitHub issue](https://github.com/ahalterman/ngec-2025/issues).

The most common problem is a PyTorch build that cannot use the GPU, which makes the pipeline (silently) 
fall back on the (much slower) CPU. The doctor will check that the pipeline is properly running on the GPU
if it's supposed to be.

## FAQs

1. How is this different from asking ChatGPT or Claude to code the stories?

A few answers:
- Replicability and privacy: NGEC's models can be re-run to produce identical
  output for replication, which is not possible with closed-weight LLMs. This
  is especially important when you're coding events over time and don't want
  sudden shifts in your coding pipeline as vendors deprecate models.
- Cost: the cost for coding thousands of stories: the paper reports a
  comparison with a closed-weight LLM that costs around $45 per 500 stories.
- LLMs can't do all of the steps reliably, including codes actors against
  Wikipedia as of the story's date, geocoding against GeoNames, and returning
  the same structured fields every time. 
- You can do both! The modular nature of NGEC means that if an LLM works as an event
  detector or attribute model for you, you can just swap it in for those
  individual steps.

2. How accurate is it?

[TODO with final replication]

3. Does it work on non-English text?

NGEC is currently configured to code English-language text only. Many of its
dependencies, including spaCy, the custom attribute model, and Wikipedia are
all English language only. Porting it to other languages is possible in theory,
but would involve a major effort.

4. Does NGEC de-duplicate events?

No. If an event is reported several times across different stories, NGEC will
generate a record for each. This is a challenging and still open research
problem, and different researchers will want to de-duplicate in different ways
depending on their needs. And some "events" can produce multiple events (e.g.,
a protest that involves both a demonstration and blocking traffic could
generate separate records for "PROTEST-demo" and "PROTEST-obstruct".  Use
caution in treating rows as synonymous with events.

5. Where do I get the news stories?

One of the major remaining challenges in producing custom event data is
obtaining news text. We cannot distribute copyrighted news text, so researchers
will be responsible for obtaining their own corpora of text to code.

6. Can I use it on text that isn't news?

Maybe! The models were trained and evaluated on news text, but they will run on
NGO or government reports, social media, or other text, but likely with
somewhat degraded accuracy. Getting good performance on "different" looking
text may require retraining some models.

## Citing NGEC

[TODO]

## More documentation

| File | What it covers |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | the install in detail: choosing extras, pip, the models, Elasticsearch on another host, updating, working from a clone |
| [`RUNNING.md`](RUNNING.md) | coding a large corpus: hardware, speed, what can go wrong |
| [`ngec/assets/guide/customize.md`](ngec/assets/guide/customize.md) | your own event definitions, classifier or actor categories, and validating the result |
| [`ngec/assets/guide/pieces.md`](ngec/assets/guide/pieces.md) | using one step of the pipeline on its own |
| [`PIPELINE.md`](PIPELINE.md) | each step of the pipeline and the fields it adds |
| [`CLASSIFIERS.md`](CLASSIFIERS.md) | where the event classifiers came from and their limits |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | speed measurements, and checking which PyTorch build you have |
| [`elasticsearch/SETUP.md`](elasticsearch/SETUP.md) | how to build the index by hand from a newer Wikipedia dump |
| [`DEVELOPING.md`](DEVELOPING.md) | changing NGEC itself: a clone, tests, retraining |
