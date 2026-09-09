"""Many documents in, a file of coded events out."""

import json

import pandas as pd
import streamlit as st

from ngec_demo import resources as R
from ngec_demo import steps
from ngec_demo.style import lede, running

# Three real-looking rows, so a reviewer can download this, upload it straight
# back and see the whole thing work before deciding what to send through it.
SAMPLE = """id,text,publication_date
sample-1,"Thousands of protesters marched through central Paris on Tuesday against the government's pension reform. Riot police fired tear gas as the march reached the Place de la Republique.",2023-03-15
sample-2,"President Emmanuel Macron met Chancellor Olaf Scholz in Berlin on Monday to discuss European defense spending ahead of next month's summit.",2023-03-15
sample-3,"The Nigerian army said it had arrested twelve suspected militants in Borno state during an operation last week.",2023-03-15
"""

st.title("Bulk")
lede("Upload a CSV or JSONL with `text` and `publication_date` columns and "
     "download the coded events.")

mode = R.current_mode()
gpu = mode == "gpu"
if not gpu:
    st.info("Bulk coding is a GPU job — on the CPU path each event takes "
            "seconds, so the uploader is only enabled in GPU mode.")

st.download_button("Sample CSV (3 rows)", SAMPLE, file_name="ngec_sample.csv",
                   mime="text/csv")

upload = st.file_uploader("CSV or JSONL", type=["csv", "jsonl", "ndjson", "json"],
                          disabled=not gpu, key="bulk_upload")

# A new file should not leave the previous run's table sitting underneath it.
if upload is not None and st.session_state.get("bulk_source") != upload.name:
    st.session_state.pop("bulk_result", None)
    st.session_state["bulk_source"] = upload.name

docs: list[dict] = []
if upload is not None:
    try:
        docs, problems = steps.read_documents(upload.getvalue(), upload.name)
    except ValueError as exc:
        st.error(str(exc))
    else:
        if problems:
            # Skipped rows are worth showing but not worth scrolling past, and
            # the count is the part that matters when there are many.
            st.warning(f"{len(problems)} row(s) skipped or capped:\n\n"
                       + "\n".join(f"- {p}" for p in problems[:10])
                       + ("\n- …" if len(problems) > 10 else ""))
        st.caption(f"{len(docs)} document(s) ready. An `id` column is used when "
                   "present; otherwise ids are row numbers.")

label = f"Code {len(docs)} documents" if docs else "Code documents"
if st.button(label, type="primary", disabled=not (gpu and docs)):
    bar = st.progress(0.0, text="Starting…")

    def tick(stage: str, done: int, total: int) -> None:
        bar.progress(done / total, text=f"{stage} ({done}/{total})")

    with running(mode, f"Coding {len(docs)} documents…"):
        st.session_state["bulk_result"] = steps.run_pipeline_bulk(
            docs, mode=mode, progress=tick)
    bar.empty()

result = st.session_state.get("bulk_result")
if result:
    events = result["events"]
    counted, coded, declined = st.columns(3)
    counted.metric("Documents", result["n_docs"])
    coded.metric("Events", len(events))
    declined.metric("Dropped", len(result["dropped"]))
    st.caption(f"{result['seconds']:.1f} s total · "
               f"{result['per_doc_seconds']:.1f} s a document")
    st.caption("dropped = classified as an event type, but the extractor found "
               "no event of that type in the text")
    if result.get("error"):
        st.warning(result["error"])

    rows = steps.events_table(events)
    if rows:
        st.dataframe(pd.DataFrame(rows[:50]), hide_index=True, width="stretch")
        if len(rows) > 50:
            st.caption(f"First 50 of {len(rows)} events; the downloads have all "
                       "of them.")
        # The JSONL is the full record, the CSV the same columns as the table
        # above -- one for re-running analysis, one for opening in a spreadsheet.
        lines = "\n".join(json.dumps(event) for event in events)
        as_jsonl, as_csv = st.columns(2)
        as_jsonl.download_button("Events (JSONL)", lines,
                                 file_name="ngec_events.jsonl",
                                 mime="application/x-ndjson", width="stretch")
        as_csv.download_button("Events (CSV)",
                               pd.DataFrame(rows).to_csv(index=False),
                               file_name="ngec_events.csv", mime="text/csv",
                               width="stretch")
    else:
        st.info("No event cleared the classifier's thresholds, or the attribute "
                "model declined every candidate.")
