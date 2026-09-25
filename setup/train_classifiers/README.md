# Training classifiers for NGEC

**The event classifiers that ship with NGEC (`ngec/assets/event_models_v2/`) are built by the code in [`codebook_llm/`](codebook_llm/README.md).** Start there, both to see how the shipped models were made and to train classifiers for your own event types. `CLASSIFIERS.md` at the top of the repository explains why they replaced the approach below.

The rest of this directory is the earlier, synthetic-news approach, kept for reference. Its models scored well on synthetic text and poorly on real articles. The script that fit them (`fit_event_classifier.py`) has been removed: it wrote model directories without the `metadata.json` and features the current classifier expects, so its models loaded but classified with the wrong features.

Because of conditions imposed by our funder and the proprietary data used in the project, we cannot share the training data or the trained models used to produce the POLECAT event dataset. However, we can provide example code for training classifiers on your own data and demonstration pretrained models for the event categories used in the POLECAT dataset that draw on a corpus of pseudo-labeled synthetic news stories. These classifiers are not as accurate as the ones used in the POLECAT dataset, but work pretty well and could easily be improved with additional training data. For these demonstration classifiers, we use a synthetic data approach described in [Halterman (2023)](https://arxiv.org/abs/2303.16028), which prompts news articles with the desired event types by providing hand-written titles to elicit news from a language model. We then use the language model's predictions as pseudo-labels to train a classifier. 

Our primary objective with this pipeline is to encourage other researchers to develop custom event datasets for their own purposes. 
Most researchers will want to train custom classifiers using their own event ontologies, which requires generating new training data.

## Contents

- `generate_synthetic_news.py`: code to generate synthetic news stories using an offline Huggingface pretrained language model.
- `headlines_event_mode.csv`: a list of hand-written headlines used to prompt the language model to generate news stories.
- `gpt_synthetic_events_2023-04-06.jsonl.zip`: a zip file containing around 1,800 synthetic news stories with event and mode pseudo-labels for training the event classifier.

