"""Customising the event ontology — the answer to “why PLOVER?”"""

import io

import pandas as pd
import streamlit as st

from ngec_demo import examples as ex_mod
from ngec_demo import paper, pipeline, ui
from ngec_demo.theme import kv_table, note, rule

ui.setup()

ui.page_header(
    kicker="Make it yours",
    title="Your event types",
    standfirst=(
        "PLOVER is this pipeline's default, not its commitment. If your question needs "
        "different event types, step 1 is the component you replace — and everything "
        "downstream keeps working."
    ),
    paper_keys=("step1", "customizing"),
)

st.markdown(
    """
There is a fair objection here: if the argument against off-the-shelf event data is that
someone else's ontology may not fit your concepts, then shipping PLOVER as the default
reproduces the problem one level down. PLOVER's categories carry real coding choices, and a
researcher who inherits them uncritically faces the same measurement problem they were trying
to escape.

The argument holds only if replacing the ontology is genuinely tractable. Here is what it
takes.
"""
)

rule()

st.markdown("## 1 · Write the codebook")

st.markdown(
    "A codebook here is a label plus a real definition — the text a human coder would be "
    "given. The definition matters more than the label: a model asked “is this PROTEST?” "
    "answers from its own sense of the word, while a model given the definition and the "
    "coding rules answers from your codebook."
)

DEFAULT_CODEBOOK = """\
label,definition
ELECTORAL_VIOLENCE,"Physical violence or intimidation directed at candidates, voters, poll workers, or election infrastructure, occurring in the context of an election campaign, voting, or the announcement of results."
PROTEST_POLICING,"Actions by police or security forces to control, disperse, contain, or suppress a demonstration or crowd, including the use of tear gas, water cannon, batons, arrests at a protest, or cordons."
ELITE_DEFECTION,"A member of a governing party, cabinet, or security apparatus publicly leaving their position or switching allegiance to an opposition or rival faction."
HUMANITARIAN_ACCESS,"Negotiation, granting, restriction, or denial of access for aid organisations to a population in need, including corridors, convoys, and permissions."
"""

uploaded = st.file_uploader("Upload a codebook CSV (columns: label, definition)", type="csv")

if uploaded is not None:
    try:
        codebook_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that CSV: {exc}")
        codebook_df = pd.read_csv(io.StringIO(DEFAULT_CODEBOOK))
else:
    codebook_df = pd.read_csv(io.StringIO(DEFAULT_CODEBOOK))

codebook_df = st.data_editor(
    codebook_df,
    use_container_width=True,
    num_rows="dynamic",
    height=240,
    key="codebook_editor",
)

valid = (
    "label" in codebook_df.columns
    and "definition" in codebook_df.columns
    and len(codebook_df.dropna(subset=["label", "definition"])) > 0
)

if not valid:
    note(
        "The codebook needs a <code>label</code> column and a <code>definition</code> column, "
        "with at least one filled row.",
        title="Codebook not usable yet",
        tone="warn",
    )
    st.stop()

clean = codebook_df.dropna(subset=["label", "definition"])
definitions = [(str(r.label), str(r.definition)) for r in clean.itertuples()]

rule()

st.markdown("## 2 · Check it separates your corpus, before spending anything")

st.markdown(
    """
Annotation is the expensive part, so it is worth knowing early whether the categories are
separable at all. Encoding each definition and each document with the same sentence
transformer the classifiers use, and taking cosine similarity, costs a second on CPU and
answers that question cheaply.

**This is not a classifier.** It is the embedding half of the candidate seeder — the step
that finds documents worth annotating. Read it as "which category is this document nearest
to", not as a prediction.
"""
)

sample_docs = [(e.title, e.text) for e in ex_mod.EXAMPLES[:6]]

if st.button("Score the demo documents against this codebook", type="primary"):
    st.session_state["_scored_codebook"] = True

if st.session_state.get("_scored_codebook"):
    with st.spinner("Encoding definitions and documents…"):
        rows = pipeline.definition_similarity(definitions, sample_docs)

    sim_df = pd.DataFrame(rows).set_index("document")
    labels = list(sim_df.columns)

    shown = sim_df.copy()
    shown.insert(0, "nearest category", sim_df.idxmax(axis=1))
    st.dataframe(
        shown,
        use_container_width=True,
        column_config={
            label: st.column_config.ProgressColumn(
                label, min_value=0.0, max_value=float(sim_df.to_numpy().max()),
                format="%.3f",
            )
            for label in labels
        },
    )

    spread = sim_df.max(axis=1) - sim_df.min(axis=1)
    st.markdown(
        f'<div class="ngec-meta">Mean gap between a document\'s best and worst category: '
        f"<strong>{spread.mean():.3f}</strong>. A small gap across the board means the "
        f"categories are not distinguishable in embedding space, which usually means the "
        f"definitions are too close to each other or too abstract — worth fixing before "
        f"annotating anything.</div>",
        unsafe_allow_html=True,
    )

    note(
        "Definitions that pull apart cleanly here are not guaranteed to train well, and "
        "definitions that look muddled here can still work with enough labels. This is a "
        "cheap smoke test, not a prediction of classifier performance.",
        title="What this does and does not tell you",
    )

rule()

st.markdown("## 3 · Turn the codebook into labels")

st.markdown(
    """
This is the part that costs money and time, and it is the part the demo cannot do live in a
browser tab. The method that produced the classifiers shipping with this package is in
`setup/train_classifiers/codebook_llm/`, and it follows Halterman and Keith (2025) on using
an LLM as a codebook annotator.

**Screen, then verify.** Annotating 2,000 documents against 16 categories exhaustively is
32,000 judgements. Instead, a screening pass shows one document against all the definitions
at once and asks which categories it *might* contain — deliberately recall-oriented, so one
call yields a few candidates plus a lot of implied negatives. A verification pass then takes
one candidate at a time, with the full definition, its sub-types, and its most confusable
siblings, and asks for a careful judgement with a supporting quote.

Three details make the difference between labels you can train on and labels you cannot:
"""
)

st.markdown(
    kv_table([
        ("Every positive needs a verbatim quote",
         "A quote that is not in the article means the annotator summarised rather than "
         "quoted, so the evidence cannot be checked and the judgement is discarded. About 1% "
         "of screening candidates and 2.5% of verified positives fail this check."),
        ("The screen's error rate is measured, not assumed",
         "A random sample of implied negatives goes to verification anyway. The rate varies "
         "enormously by category — 2% for narrow, concrete ones and 36% for ACCUSE, which is "
         "defined broadly enough to cover most political news. Those rates become per-example "
         "weights on the negatives, not a filter."),
        ("Rare categories are hunted, not sampled",
         "A random sample of news contains no hunger strikes. Candidates are found with "
         "regexes written from each definition and with nearest-neighbours in embedding space "
         "— the same similarity computed above — and verification works rarest-first, so "
         "stopping early still leaves the scarce labels done."),
    ]),
    unsafe_allow_html=True,
)

note(
    "An earlier version of this pipeline dropped implied negatives for the high-error "
    "categories outright. That looked principled and was wrong: verification runs mostly on "
    "candidates, which are enriched for positives, so dropping the negatives left ACCUSE with "
    "992 positives against 39 negatives — and a model that predicted “yes” unconditionally, "
    "at an F1 of 0.99. The high score was measuring the class balance. Down-weighting keeps "
    "the negatives and their known unreliability.",
    title="A failure worth knowing about",
    tone="warn",
)

rule()

st.markdown("## 4 · Fit, evaluate, and drop the models in")

st.markdown(
    """
The models themselves are deliberately simple: one binary logistic regression per category,
on chunked mean-pooled sentence embeddings plus TF-IDF word features under an L1 penalty.
The chunking matters — sentence transformers truncate at 384 word pieces, and 76% of these
articles are longer than that, so an event described in the second half of a story was
invisible to the models this replaced. The lasso keeps a readable handful of decisive terms
per class, which you can print and inspect.

```shell
cd setup/train_classifiers/codebook_llm
python prepare_corpus.py --n 60000        # corpus -> year-stratified pool
python codebook.py                        # your codebook CSV -> annotation prompts
python embed_pool.py --device cuda        # cache chunked embeddings
python seed_candidates.py --min-hits 1    # find rare-category candidates
python make_batches.py screen --n 2000    # build screening prompts
#   ... annotate, then ...
python collect_annotations.py
python build_verify_candidates.py
python make_batches.py --batch-size 25 verify
#   ... annotate again, then ...
python collect_annotations.py
python train_classifiers.py --show-terms  # fit, and print the decisive terms
python evaluate.py                        # score on a never-seeded holdout
```

`train_classifiers.py` writes a directory with one `.skops` model per category and a
`metadata.json` recording the encoder name, the per-class thresholds, and the training
metrics. That directory is what the pipeline loads:

```python
from ngec.classifiers.plover_sklearn import PloverSklearnClassifier

classifier = PloverSklearnClassifier(type_model_dir="my_models/")
```

Because the models are self-describing, nothing else needs to change — the serving code
reads the encoder and the thresholds out of the directory rather than assuming defaults.
Feature construction lives in `ngec/classifiers/features.py`, which the training scripts
import too, so the two cannot drift.
"""
)

rule()

st.markdown("## What this actually costs")

st.markdown(
    kv_table([
        ("Annotation", "~2,000 documents, screened then verified. A few hours of LLM API "
                       "time and a modest API bill; the exact figure depends on your "
                       "document lengths and how many categories you screen against."),
        ("Embedding the pool", "About six minutes for 60,000 documents on a 4090; roughly "
                               "four hours on CPU."),
        ("Fitting", "Minutes. These are logistic regressions."),
        ("Human time", "The real cost. Writing definitions that an annotator — human or "
                       "model — can apply consistently is the hard part, and it is the part "
                       "that determines whether the resulting measure means what you think."),
        ("What you do not need",
         "Labelled training data of your own, a GPU for serving, or any change to steps 2 "
         "through 5. The attribute model extracts the four attributes for event types it was "
         "never trained on — that is what the ECAV validation tests."),
    ]),
    unsafe_allow_html=True,
)

note(
    "The demonstration classifiers reach a macro F1 of about 0.52 across sixteen PLOVER types "
    "on a random holdout, from 2,000 documents and one LLM annotator with no adjudication. "
    "That is a usable demonstration, not a production model, and a custom ontology built this "
    "way should expect something similar unless you invest more in annotation. Reporting that "
    "number rather than the enriched-split figure is the point — see "
    "<a href='step1'>the detection page</a>.",
    title="Set your expectations from the holdout, not the demo",
    tone="warn",
)

st.markdown(
    """
### And the answer to "why PLOVER?"

PLOVER is here because it is the ontology the pretrained attribute model was trained
alongside and the one POLECAT uses, which makes it a sensible default and a reproducible
baseline. It is not a claim that its categories fit your question. The framework's value is
that they are replaceable — and the honest version of the paper's argument is that
researchers *should* replace them when their concepts differ, rather than inheriting them
because they came in the box.
"""
)

paper.ref("step1", "customizing", "ecav")
