"""Step 1: which event type? Sixteen binary classifiers, sixteen thresholds."""

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import (demo_model_note, hbar_chart, lede, mode_badge,
                             running, timing_table)

# (button label, text). One text that fires a single type, one that fires
# several, and one that should fire nothing at all -- a classifier demo that
# only ever shows hits hides the half of the job that is saying "no".
EXAMPLES: list[tuple[str, str]] = [
    ("Protest",
     "Thousands of protesters blocked traffic in central Paris on Tuesday to "
     "demonstrate against the pension reform."),
    ("Protest + arrests",
     "Police in Nairobi arrested at least forty people on Saturday during a "
     "demonstration against a proposed finance bill. The interior ministry said "
     "the arrests were made to protect public order."),
    ("Not an event",
     "Heavy rain washed out the second day of the test match in Wellington, and "
     "forecasters expect the storm to clear by Thursday morning."),
]

st.title("1. Event Detection")

st.markdown("""The first step in our proposed event data pipeline is *event
detection*: identifying which event types are described in a document. Treating
this as a document classification task makes it easy to collect training data and fit models.

This is a step that almost anyone making custom event data would want to modify. The paper 
            has more details.""")

st.markdown("Every text is scored by all sixteen PLOVER classifiers; each fires at its "
     "own tuned threshold, so a story can be several events, or none.")

if "s1_text" not in st.session_state:
    st.session_state.s1_text = EXAMPLES[0][1]

cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES):
    if col.button(label, width="stretch", key=f"s1_ex_{label}"):
        st.session_state.s1_text = text
        st.session_state.pop("s1_result", None)
        st.rerun()

st.text_area("Text", key="s1_text", height=120)
if st.button("Classify", type="primary"):
    with running(R.current_mode(), "Scoring the text…"):
        _, notes = R.get_classifier(R.current_mode())
        st.session_state["model_notes"] = notes
        st.session_state["s1_result"] = steps.classify(st.session_state.s1_text)

result = st.session_state.get("s1_result")
if result:
    types = pd.DataFrame(result["types"])
    st.altair_chart(hbar_chart(types, "event_type", "probability", "threshold"),
                    width="stretch")
    fired = [row["event_type"] for row in result["types"] if row["fired"]]
    st.caption(f"{result['seconds']:.2f} s · {mode_badge(result['mode'])} · "
               "ticks are each class's threshold · "
               + (f"fired: {', '.join(fired)}" if fired else "nothing fired"))

    st.subheader("Modes")
    if not fired:
        st.caption("Nothing fired, so these are the modes of the top-scoring type — "
                   "mode models are trained conditional on their parent type.")
    for event_type, rows in result["modes"].items():
        st.markdown(f"**{event_type}**")
        modes = pd.DataFrame(rows)
        # A mode under a type that did not fire is drawn grey whatever its own
        # threshold says: the accent means "this fired", and the event did not.
        silent = event_type not in fired
        # No height override: the default gives every mode a full row, which a
        # short chart used to buy by dropping labels.
        st.altair_chart(hbar_chart(modes, "mode", "probability", "threshold",
                                   muted=silent),
                        width="stretch")
        if silent:
            st.caption("parent type did not fire")

    demo_model_note()

    with st.expander("Timing breakdown"):
        timing_table(result["timing"])

# The training scripts behind these steps are in
# setup/train_classifiers/codebook_llm/ (committed 2026-09-24), so this text could
# now point readers there instead of giving the recipe inline.
with st.expander("Customizing event classification"):
    st.markdown("""*Or, use conflibert*,

Nothing is fitted live here, but the recipe is short. To code your own ontology
rather than PLOVER:

**1. Write a codebook.** One short definition per event type, and one per mode if
your types have modes. This is human work, and it is most of the work.

**2. Label real documents.** The models shipped here were labelled by an LLM
applying the PLOVER codebook to Voice of America news articles, with a person
checking a sample of what it produced. Labelling real news this way worked better
than generating synthetic text for each type, an older approach whose scripts are
in `setup/train_classifiers/`.

**3. Fit one binary classifier per type**, on the features built by
`ngec.classifiers.features` — the training script and the classifier here import
the same module, so training and inference cannot drift apart. Pick a threshold
per class, then save the models into a directory with a `metadata.json` recording
the encoder name and those thresholds.

Point the classifier at that directory. Mode models are picked up from a `modes/`
subdirectory of it if one is there.
""")

    st.code("""from ngec.classifiers.plover_sklearn import PloverSklearnClassifier

clf = PloverSklearnClassifier(
    codebook_path="my_codebook.csv",  # your event types and their modes
    type_model_dir="my_models/",      # {TYPE}.skops, metadata.json, modes/
)

# PloverCoder has no argument for this yet, so set it after construction.
coder = PloverCoder(es_client)
coder.event_model = clf""", language="python")

    st.markdown(
        "A completely different classifier drops in the same way: anything with an "
        "`__init__` and a `process(list of dicts)` that adds `event_type`, "
        "`event_type_confidence`, and `event_mode` to each dict will serve.")

    # Counts from ngec/assets/event_models_v2/metadata.json (`n_labeled` runs
    # 2,104-2,145 across the sixteen types); encoding rate measured on this box's
    # GPU. Hard-coded rather than read at runtime to keep the page's imports small.
    st.caption("Scale: about 2,100 labelled documents per type went into the "
               "sixteen bundled models. Encoding runs at roughly 800 documents a "
               "second on a GPU and the sixteen fits take seconds, so the cost is "
               "the labelling, not the training. These remain demonstration "
               "models, not the production models behind POLECAT.")
