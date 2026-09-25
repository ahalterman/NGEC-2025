"""
Step 1: turn the PLOVER codebook CSV into prompts an LLM can follow.

Halterman and Keith (2025) find that an LLM asked to apply a social-science
codebook does much better when it is given the codebook's actual definition text,
restructured into instruction form, than when it is given only the label names
and left to fall back on its own priors about what a word like "PROTEST" means.
This module does that restructuring. Every definition, mode definition, and
extraction note below is copied verbatim out of
`ngec/assets/PLOVER_structured_codebook_updated.csv` -- nothing is paraphrased,
so changing the codebook changes the prompts.

The CSV has one row per (event, mode) pair with five columns. Two of them are
used in ways that are not obvious from the header:

- For the five event types that have no modes (AGREE, SUPPORT, CONCEDE,
  COOPERATE, AID), the `mode_def` column holds *exemplar sub-cases* -- concrete
  things that count as that event -- rather than mode definitions.
- `extraction_notes` holds boundary rules ("Protests DO NOT fall under this
  category", "Promises of future retreats do not apply"). These are the rules
  that separate confusable categories, so they are the most valuable text in the
  file and are surfaced prominently.

Two prompts are rendered:

- `render_screening_prompt` -- all 16 definitions at once, asking which types a
  document might contain. Cheap, recall-oriented, run over every pooled document.
- `render_verification_prompt` -- one type in full detail, plus its modes and the
  definitions of the types most likely to be confused with it, asking for a
  careful 1/0/NA judgment. Run only on candidates the screening pass surfaced.

The confusable "contrast" types are chosen by similarity between definition
texts, not by hand, so the prompts stay derived from the codebook.
"""

import csv
import json
from collections import OrderedDict
from importlib import resources
from pathlib import Path

DEFAULT_CODEBOOK = Path(__file__).resolve().parents[3] / "ngec" / "assets" / \
    "PLOVER_structured_codebook_updated.csv"

# How many confusable sibling types to show in a verification prompt.
N_CONTRAST_TYPES = 3


def load_codebook(path=None) -> "OrderedDict[str, dict]":
    """
    Parse the codebook CSV into one entry per event type.

    Returns
    -------
    OrderedDict mapping event type -> {
        "definition": str,           # the event_def, verbatim
        "modes": OrderedDict,        # mode name -> mode_def, verbatim
        "exemplars": list[str],      # sub-cases, for types with no modes
        "notes": list[str],          # deduplicated extraction_notes
    }
    """
    path = Path(path) if path else DEFAULT_CODEBOOK
    book: "OrderedDict[str, dict]" = OrderedDict()

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            event = row.get("event", "").strip()
            if not event:
                continue

            entry = book.setdefault(event, {
                "definition": row.get("event_def", "").strip(),
                "modes": OrderedDict(),
                "exemplars": [],
                "notes": [],
            })

            mode = row.get("mode", "").strip()
            mode_def = row.get("mode_def", "").strip()
            if mode:
                if mode not in entry["modes"] and mode_def:
                    entry["modes"][mode] = mode_def
            elif mode_def and mode_def not in entry["exemplars"]:
                # No mode name: this row is an exemplar sub-case of the type.
                entry["exemplars"].append(mode_def)

            note = row.get("extraction_notes", "").strip()
            # The same note is repeated across a type's rows; keep one copy.
            if note and note not in entry["notes"]:
                entry["notes"].append(note)

    return book


def _definition_contrasts(book, n=N_CONTRAST_TYPES):
    """
    For each type, find the `n` other types whose definitions are most similar.

    Uses TF-IDF cosine over the definition text. The point is to show a verifier
    the categories it is most likely to confuse the target with, so it can rule
    them out explicitly rather than accepting the first plausible label. Chosen
    from the codebook text rather than by hand so the prompts stay reproducible
    from the CSV alone.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    types = list(book)
    # Include exemplars and mode definitions: they carry much of what
    # distinguishes, say, COERCE from ASSAULT.
    docs = []
    for t in types:
        entry = book[t]
        parts = [entry["definition"], *entry["exemplars"], *entry["modes"].values()]
        docs.append(" ".join(parts))

    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    sim = cosine_similarity(tfidf)

    contrasts = {}
    for i, t in enumerate(types):
        order = sim[i].argsort()[::-1]
        contrasts[t] = [types[j] for j in order if j != i][:n]
    return contrasts


# -----------------------------------------------------------------------------
# Screening prompt: all 16 types, recall-oriented
# -----------------------------------------------------------------------------

SCREENING_HEADER = """\
You are applying the PLOVER political event ontology to news articles. PLOVER is \
a codebook used by political scientists to record what political actors did to \
each other. Your job in this pass is a wide first cut: for each article, list \
every event type that the article *might* contain, so a second, more careful pass \
can check them.

Be generous here. Missing a type at this stage means it is never checked, while \
a wrong guess is cheap because it gets verified later. But do not list a type \
with no support in the text at all.

An article reports an event type if it *describes that event happening* between \
political actors. Background, history, analysis, and speculation about what might \
happen do not count. The event does not have to be the article's main subject.

THE SIXTEEN EVENT TYPES
"""

SCREENING_FOOTER = """\

OUTPUT
Return a JSON array, one object per article, in the order the articles are given:

  [{{"id": "<article id>", "candidates": ["PROTEST", "COERCE"], "quotes": {{"PROTEST": "<=15 words quoted exactly from the article", "COERCE": "..."}}}}]

Rules:
- `candidates` uses the exact uppercase type names above. Use [] if none apply.
- Every candidate needs a `quotes` entry quoting the article *verbatim*. If you \
cannot find a literal supporting quote, drop that candidate.
- Return only the JSON array. No commentary, no markdown fences.

ARTICLES
"""


def render_screening_prompt(book, docs) -> str:
    """Build the screening prompt for a batch of documents."""
    lines = [SCREENING_HEADER]
    for event_type, entry in book.items():
        lines.append(f"\n{event_type}: {entry['definition']}")
        if entry["modes"]:
            lines.append(f"  Sub-types: {', '.join(entry['modes'])}")
        elif entry["exemplars"]:
            examples = "; ".join(entry["exemplars"][:4])
            lines.append(f"  For example: {examples}")
        for note in entry["notes"]:
            lines.append(f"  Note: {note}")

    lines.append(SCREENING_FOOTER)
    for doc in docs:
        lines.append(f"\n--- ARTICLE {doc['id']} ---\n{doc['event_text']}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Verification prompt: one type in full, with contrasts
# -----------------------------------------------------------------------------

VERIFICATION_HEADER = """\
You are applying the PLOVER political event ontology to news articles, as a \
political science research annotator. You are checking ONE event type against a \
batch of articles. Apply the definition below literally. Where the definition and \
your own intuition about the word disagree, the definition wins.

TARGET EVENT TYPE
"""

VERIFICATION_CONTRAST = """\

DO NOT CONFUSE IT WITH
These are separate types in the ontology. If what the article describes fits one \
of these better than the target, the target is 0, not 1.
"""

VERIFICATION_TASK = """\

YOUR JUDGMENT
For each article, return one of:
  1   - the article describes a {event_type} event as defined above, and you can \
quote the text that shows it.
  0   - it does not. This includes articles about the general topic that never \
report the event happening, and articles that fit a "do not confuse" type instead.
  NA  - genuinely undecidable: the definition is ambiguous for this case, or the \
text is too garbled or truncated to tell. Use this sparingly. "NA" means the \
example is thrown away, so do not use it for cases you merely find hard.

Judge the article, not its topic. An article *about* {event_type} that only \
discusses it in the abstract, or reports it as a possibility, is 0.
{mode_task}
OUTPUT
Return a JSON array, one object per article, in the order given:

  [{{"id": "<article id>", "label": 1, "quote": "<verbatim quote from the article>"{mode_field}}}]

Rules:
- `label` is 1, 0, or "NA" (1 and 0 as bare numbers, NA as the string "NA").
- `quote` is required when label is 1, copied *exactly* from the article, at most \
25 words. Use "" when the label is 0 or NA.
- Return only the JSON array. No commentary, no markdown fences.

ARTICLES
"""

MODE_TASK = """
SUB-TYPES (MODES)
When the label is 1, also say which sub-type(s) of {event_type} the article \
describes, using the definitions above. Use the exact lowercase names. An event \
can have more than one. Use [] if it is a {event_type} event but none of the \
sub-types fit.
"""


def render_verification_prompt(book, event_type, docs, contrasts=None) -> str:
    """Build the verification prompt for one event type over a batch of docs."""
    entry = book[event_type]
    lines = [VERIFICATION_HEADER, f"{event_type}: {entry['definition']}"]

    if entry["exemplars"]:
        lines.append(f"\nThings that count as {event_type}:")
        for exemplar in entry["exemplars"]:
            lines.append(f"  - {exemplar}")

    if entry["modes"]:
        lines.append(f"\nSub-types of {event_type}:")
        for mode, mode_def in entry["modes"].items():
            lines.append(f"  - {mode}: {mode_def}")

    if entry["notes"]:
        lines.append("\nCoding rules for this type:")
        for note in entry["notes"]:
            lines.append(f"  - {note}")

    if contrasts:
        lines.append(VERIFICATION_CONTRAST)
        for other in contrasts:
            lines.append(f"  {other}: {book[other]['definition']}")

    has_modes = bool(entry["modes"])
    lines.append(VERIFICATION_TASK.format(
        event_type=event_type,
        mode_task=MODE_TASK.format(event_type=event_type) if has_modes else "",
        mode_field=', "modes": ["<sub-type>"]' if has_modes else "",
    ))

    for doc in docs:
        lines.append(f"\n--- ARTICLE {doc['id']} ---\n{doc['event_text']}")
    return "\n".join(lines)


def main():
    """Write the rendered codebook to disk and show a sample prompt."""
    book = load_codebook()
    contrasts = _definition_contrasts(book)

    out = Path(__file__).parent / "data" / "codebook.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {t: {**e, "modes": dict(e["modes"]), "contrasts": contrasts[t]}
             for t, e in book.items()},
            f, indent=2,
        )

    n_modes = sum(len(e["modes"]) for e in book.values())
    print(f"{len(book)} event types, {n_modes} modes -> {out}")
    print("\nContrast types chosen from definition similarity:")
    for t in book:
        print(f"  {t:10s} vs {', '.join(contrasts[t])}")


if __name__ == "__main__":
    main()
