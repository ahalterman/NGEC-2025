"""Dump the exact prompt strings the demo sends to llama-server.

Uses AttributeModel's own prompt construction so the benchmark replays real
extraction prompts, not synthetic tokens. Output: ~/bench/prompts.json
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))

from ngec.attribute_model import AttributeModel

DOCS = [
    ("2023-03-15", "Thousands of protesters gathered in Paris on Tuesday to demonstrate against the government's proposed pension reforms, which would raise the retirement age by two years. President Emmanuel Macron said the reforms would proceed despite the demonstrations. Police used tear gas to disperse a crowd that had blocked traffic near the Place de la Republique. Union leaders said they would call further strikes if the government did not withdraw the bill."),
    ("2024-06-11", "Officials from Ethiopia and Eritrea signed a ceasefire agreement in Cairo on Monday, ending three weeks of fighting along their shared border. The agreement was mediated by Egypt and commits both sides to withdraw heavy weapons within thirty days. Ethiopian Prime Minister Abiy Ahmed described the agreement as a first step. A spokesperson for the Eritrean foreign ministry said monitors would be admitted to the border area next week."),
    ("2024-02-20", "Police in Nairobi arrested at least forty people on Saturday during a demonstration against a proposed finance bill. Protesters had gathered outside parliament since early morning, carrying placards and chanting. The interior ministry said in a statement that the arrests were made to protect public order and that those detained would appear in court on Monday. Opposition leaders condemned the arrests and called for a second demonstration next week."),
]

# The event types a document like these actually triggers in the demo.
TYPES = ["PROTEST", "ASSAULT", "COERCE", "ACCUSE", "REQUEST",
         "AGREE", "CONSULT", "THREATEN", "REJECT"]

am = AttributeModel(silent=True, backend="llamacpp")

out = []
for doc_i, (date, doc) in enumerate(DOCS):
    for et in TYPES:
        ev = {"id": f"d{doc_i}_{et}", "event_type": et, "event_mode": "",
              "event_text": doc, "date": date}
        try:
            out.append({"doc": doc_i, "event_type": et,
                        "prompt": am.make_prompt(ev)})
        except Exception as e:
            print(f"skip {et}: {e}", file=sys.stderr)

json.dump(out, open(os.path.join(HERE, "prompts.json"), "w"))
print(f"wrote {len(out)} prompts")
