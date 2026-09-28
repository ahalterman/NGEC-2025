"""
Assemble the annotation files the subagents wrote into one long table.

Annotations arrive as one JSON file per batch, in two shapes -- screening
(candidate types per document) and verification (a 1/0/NA judgment for one type).
This script reads all of them, checks them against the batch manifests, and emits
a single long table of labels:

    {"id", "event_type", "mode", "label", "source", "quote", "batch"}

`label` is 1, 0, or "NA". `mode` is null for a type-level label.

Two things worth understanding about where the labels come from.

**Most zeros are implied, not asked for.** The screening pass looks at a document
against all 16 definitions at once. A type it did not list is a zero for that
document -- one screening call yields roughly thirteen negatives alongside its two
or three candidates, which is what makes annotating at this scale affordable. Those
implied zeros are marked `source="screen_implied"` so they can be told apart from
zeros a verifier actually deliberated over (`source="verify"`).

**Implied zeros are only as good as the screen's recall**, so a sample of them is
sent to verification anyway as an audit. If the audit finds the screen missed real
positives at some rate, that rate is a measured property of the dataset rather
than a silent bias. `audit_report` prints it.

Usage:
    python collect_annotations.py --out data/annotations.jsonl
"""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from codebook import load_codebook

VALID_LABELS = {0, 1, "NA"}

# Typographic substitutions the annotators make when quoting. The articles use
# curly quotes and en dashes; a quote retyped with straight ASCII equivalents is
# the same quote and should not be treated as unsupported.
_CHAR_FIXES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})


def normalize_quote(text: str) -> str:
    """Fold away differences that do not change what was quoted."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_FIXES)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_supported(quote: str, document: str) -> bool:
    """
    Is this quote actually in the document?

    The check exists to catch invented evidence, so it has to be strict about
    content while forgiving about form. Two things it allows:

    - **Typography.** The corpus uses curly quotes and en dashes; annotators
      retype them as ASCII.
    - **Elision.** Annotators join two spans with "..." to skip an irrelevant
      clause, which is ordinary quoting practice. Each fragment must still appear
      in the document, in order.

    What it does not allow is paraphrase. A quote whose words are not in the text
    means the annotator summarized instead of quoting, and the evidence cannot be
    checked, so the judgment it supports is not trusted.
    """
    if not quote:
        return True

    document = normalize_quote(document)
    fragments = [f for f in re.split(r"\s*\.\.\.\s*|\s*…\s*", normalize_quote(quote)) if f]

    position = 0
    for fragment in fragments:
        found = document.find(fragment, position)
        if found == -1:
            return False
        position = found + len(fragment)
    return True


def _read_batch_outputs(batch_dir):
    """Yield (manifest_entry, parsed_json) for every completed batch."""
    manifest_path = Path(batch_dir) / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = {entry["batch"]: entry for entry in json.load(open(manifest_path))}

    for name, entry in sorted(manifest.items()):
        out_path = Path(batch_dir) / f"{name}.json"
        if not out_path.exists():
            continue
        try:
            data = json.load(open(out_path, encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  SKIP {name}: unparseable JSON ({exc})")
            continue
        if not isinstance(data, list):
            print(f"  SKIP {name}: expected a JSON array")
            continue
        yield entry, data


def collect_screening(batch_dir, event_types, pool_text):
    """
    Turn screening outputs into labels.

    Emits a positive-candidate marker for each listed type and an implied zero for
    every type that was not listed.
    """
    rows = []
    stats = Counter()

    for entry, data in _read_batch_outputs(batch_dir):
        expected = entry["doc_ids"]
        by_id = {}
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                by_id[str(item["id"])] = item

        missing = [d for d in expected if d not in by_id]
        if missing:
            stats["docs_missing_from_output"] += len(missing)

        for doc_id in expected:
            item = by_id.get(doc_id)
            if item is None:
                continue
            stats["docs"] += 1

            candidates = item.get("candidates") or []
            candidates = [c for c in candidates if c in event_types]
            quotes = item.get("quotes") or {}

            for event_type in event_types:
                if event_type in candidates:
                    quote = quotes.get(event_type, "") or ""
                    # A quote that is not actually in the article means the model
                    # invented its evidence; drop the candidate rather than
                    # promote it to verification.
                    if not quote_supported(quote, pool_text.get(doc_id, "")):
                        stats["candidates_dropped_bad_quote"] += 1
                        continue
                    stats["candidates"] += 1
                    rows.append({"id": doc_id, "event_type": event_type,
                                 "mode": None, "label": 1,
                                 "source": "screen_candidate",
                                 "quote": quote, "batch": entry["batch"]})
                else:
                    stats["implied_zeros"] += 1
                    rows.append({"id": doc_id, "event_type": event_type,
                                 "mode": None, "label": 0,
                                 "source": "screen_implied",
                                 "quote": "", "batch": entry["batch"]})

    return rows, stats


def collect_verification(batch_dir, book, pool_text):
    """Turn verification outputs into type labels and mode labels."""
    rows = []
    stats = Counter()

    for entry, data in _read_batch_outputs(batch_dir):
        event_type = entry.get("event_type")
        if event_type not in book:
            print(f"  SKIP {entry['batch']}: unknown event type {event_type!r}")
            continue
        known_modes = set(book[event_type]["modes"])
        expected = entry["doc_ids"]
        by_id = {str(i["id"]): i for i in data
                 if isinstance(i, dict) and i.get("id")}

        for doc_id in expected:
            item = by_id.get(doc_id)
            if item is None:
                stats["docs_missing_from_output"] += 1
                continue

            label = item.get("label")
            if label not in VALID_LABELS:
                stats["bad_label"] += 1
                continue

            quote = item.get("quote", "") or ""
            if label == 1 and not quote_supported(quote, pool_text.get(doc_id, "")):
                # Same rule as screening: unverifiable evidence, so treat the
                # judgment as undecidable rather than as a positive.
                stats["positives_downgraded_bad_quote"] += 1
                label = "NA"

            stats[f"label_{label}"] += 1
            rows.append({"id": doc_id, "event_type": event_type, "mode": None,
                         "label": label, "source": "verify",
                         "quote": quote, "batch": entry["batch"]})

            # Mode labels only mean anything when the type itself is present.
            if label != 1 or not known_modes:
                continue
            claimed = item.get("modes") or []
            claimed = [m for m in claimed if m in known_modes]
            for mode in known_modes:
                is_present = 1 if mode in claimed else 0
                stats[f"mode_{is_present}"] += 1
                rows.append({"id": doc_id, "event_type": event_type,
                             "mode": mode, "label": is_present,
                             "source": "verify", "quote": quote,
                             "batch": entry["batch"]})

    return rows, stats


def resolve(rows):
    """
    Collapse duplicate labels for the same (document, type, mode) cell.

    A cell can be labeled more than once: the screen implies a zero, and a
    verification batch then judges it directly. Verification wins, because it saw
    the full definition for that one type instead of all sixteen at once.

    When verification overrides a screening judgment, what the screen had said is
    kept on the surviving row as `screen_label`. That is what lets the training
    step measure the screen's error rate per event type -- and it matters, because
    the screen's implied zeros are used as negatives in bulk. For a broad category
    like ACCUSE the screen misses real positives often enough that feeding its
    implied zeros in as negatives would teach the model the wrong boundary.
    """
    priority = {"verify": 3, "screen_candidate": 2, "screen_implied": 1}
    best = {}
    for row in rows:
        key = (row["id"], row["event_type"], row["mode"])
        current = best.get(key)
        if current is None or priority[row["source"]] > priority[current["source"]]:
            if current is not None and current["source"].startswith("screen"):
                row = {**row, "screen_label": current["label"],
                       "screen_source": current["source"]}
            elif current is not None:
                row = {**row, **{k: current[k] for k in ("screen_label", "screen_source")
                                 if k in current}}
            best[key] = row
        elif row["source"].startswith("screen") and "screen_label" not in best[key]:
            best[key] = {**best[key], "screen_label": row["label"],
                         "screen_source": row["source"]}
    return list(best.values())


def audit_report(rows):
    """
    Measure how often verification overturned a screening judgment.

    The screen's false-negative rate is the number that matters: implied zeros
    are used as training negatives in bulk, so if the screen routinely misses
    positives the training data is quietly wrong.
    """
    by_cell = defaultdict(dict)
    for row in rows:
        if row["mode"] is None:
            by_cell[(row["id"], row["event_type"])][row["source"]] = row["label"]

    audited = fn = fp = 0
    for sources in by_cell.values():
        if "verify" not in sources:
            continue
        verdict = sources["verify"]
        if "screen_implied" in sources:
            audited += 1
            if verdict == 1:
                fn += 1
        elif "screen_candidate" in sources:
            if verdict == 0:
                fp += 1

    print("\nScreen-vs-verify agreement:")
    if audited:
        print(f"  audited implied zeros: {audited}")
        print(f"  screen missed a real positive: {fn} ({100 * fn / audited:.1f}%)")
    else:
        print("  no implied zeros were audited yet")
    print(f"  screen candidates rejected by verification: {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/voa_pool.jsonl")
    ap.add_argument("--screen-dir", default="data/batches/screen")
    ap.add_argument("--verify-dir", default="data/batches/verify")
    ap.add_argument("--out", default="data/annotations.jsonl")
    args = ap.parse_args()

    book = load_codebook()
    pool_text = {}
    for line in open(args.pool, encoding="utf-8"):
        rec = json.loads(line)
        pool_text[rec["id"]] = rec["event_text"]

    print("Reading screening batches ...")
    screen_rows, screen_stats = collect_screening(
        args.screen_dir, list(book), pool_text)
    for key, value in sorted(screen_stats.items()):
        print(f"  {key}: {value}")

    print("Reading verification batches ...")
    verify_rows, verify_stats = collect_verification(
        args.verify_dir, book, pool_text)
    for key, value in sorted(verify_stats.items()):
        print(f"  {key}: {value}")

    rows = resolve(screen_rows + verify_rows)
    audit_report(screen_rows + verify_rows)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    type_pos = Counter()
    mode_pos = Counter()
    for row in rows:
        if row["label"] != 1:
            continue
        if row["mode"] is None:
            type_pos[row["event_type"]] += 1
        else:
            mode_pos[(row["event_type"], row["mode"])] += 1

    print(f"\n{len(rows)} label cells -> {args.out}")
    print("\nPositives per event type:")
    for event_type in sorted(book):
        print(f"  {event_type:10s} {type_pos[event_type]:5d}")
    if mode_pos:
        print("\nPositives per mode:")
        for (event_type, mode), n in sorted(mode_pos.items()):
            print(f"  {event_type}-{mode:16s} {n:5d}")


if __name__ == "__main__":
    main()
