# NGEC -- Next generation political event coder

This repository contains the code for the Next Generation Event Coder (NGEC), a
Python library for extracting event data from news text. The pipeline works out-of-the-box
to code events using the [PLOVER event ontology](https://osf.io/preprints/socarxiv/rm5dw/), but can 
be easily customized to produce events with a custom ontology.

It accompanies the working paper, ["Creating Custom Event Data Without Dictionaries: A Bag-of-Tricks"](https://arxiv.org/pdf/2304.01331.pdf).

## Overview

We break the problem of event extraction into six steps:

1. Event classification: identify the event described in a document (e.g., PROTEST, ASSAULT, AGREE,...) using a transformer classifier trained on new data.
2. Sub-event (``mode'') classification: identify a more specific event type (e.g., PROTEST-riot, ASSAULT-aerial), also using a transformer-based classifier.
3. Context classification: identify themes or topics in a document (e.g., "human rights", "environment") using a classifier.
4. Event attribute identification: identifying the spans of text that report who carried out the event, who it was directed against, where it occurred, etc. We do this with a fine-tuned question-answering model trained on newly annotated text.
5. Actor, location, and date resolution: we resolve extracted named actors and recipients to their Wikipedia page using an offline Wikipedia index and a custom neural similarity model.
6. Entity categorization: Finally, we map the actor to their country and their "sector" code as defined by the PLOVER ontology (e.g., "GOV", "MIL", etc.)

![](docs/pipeline_figure.png)


## Citing

The steps that this pipeline implements are described in more detail in the [paper](https://arxiv.org/pdf/2304.01331.pdf). If you use the pipeline or the techniques we introduce, please cite the following:

```
@article{halterman_et_al2023creating,
  title={Creating Custom Event Data Without Dictionaries: A Bag-of-Tricks},
  author={Andrew Halterman and Philip A. Schrodt and Andreas Beger and Benjamin E. Bagozzi and Grace I. Scarborough},
  journal={arXiv preprint arXiv:2304.01331},
  year={2023}
}
```

## Acknowledgements

This research was sponsored by the Political Instability Task Force (PITF). The PITF is funded by
the Central Intelligence Agency. The views expressed in this paper are the authors’ alone and do not
represent the views of the U.S. Government.