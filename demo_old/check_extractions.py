"""Measure extraction quality on the demo's curated examples.

Claims in the demo and its docs rest on numbers from this script, so this is the
script that produces them — otherwise they are assertions nobody can check.

It exists because of a mistake worth not repeating. A first pass compared spans
to the document with a plain string match and reported that the published model
got 38 of 41 anchor quotes verbatim while the retrained one got 27 of 27, and
that difference reached three documents before anyone looked at the three
"failures". All three were verbatim quotes whose apostrophes were curly where
the source's were straight. Both models are in fact 100%. The fix is
`_normalise_typography`, imported below rather than reimplemented, so that this
script and the live check on every page cannot drift apart.

    uv run --group demo-app python demo/check_extractions.py                      # the current cache
    uv run --group demo-app python demo/check_extractions.py a.json b.json        # compare two

What it measures, per cached run:

  anchor quote verbatim   Is `anchor_quote` a span that actually appears in the
                          document? The model is supposed to copy, not
                          paraphrase; a quote that is not in the text is the
                          known failure where it echoes the codebook definition
                          instead. This is the same check `theme.unmatched_spans`
                          makes live on every page.
  actor / recipient N/A   How often a slot came back "N/A". Not a defect on its
                          own — many event types genuinely have no recipient —
                          but a jump in it alongside non-verbatim quotes is the
                          signature of an extraction that has gone thin.
  records                 Total records. Worth watching: a model that declines
                          more of the classifier's false positives produces
                          *fewer* records, which is an improvement that looks
                          like a regression if you only count rows.

To compare two models, build a cache with each and pass both files:

    NGEC_ATTRIBUTE_MODEL=ahalt/event-attribute-extractor NGEC_DEMO_BACKEND=vllm \\
        uv run python demo/build_example_cache.py --out /tmp/published.json
    NGEC_DEMO_BACKEND=vllm uv run python demo/build_example_cache.py
    uv run --group demo-app python demo/check_extractions.py /tmp/published.json demo/data/example_runs.json

Eleven documents is far too small a sample to establish a quality difference.
It is big enough to catch a model or prompt format that has gone badly wrong,
which is what this is for.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The same typographic folding the pages use when they highlight a span, so this
# script and the live check cannot disagree about what "verbatim" means. Without
# it a curly apostrophe reads as a paraphrase: three of this demo's forty-one
# spans were being miscounted that way.
from ngec_demo.theme import _normalise_typography  # noqa: E402

DEFAULT = HERE / "data" / "example_runs.json"

# "N/A" is what the model writes for a slot it cannot fill; the rest are
# variants seen in practice.
EMPTY = {"N/A", "NA", "NONE", ""}


def normalise(text) -> str:
    """Lowercase, collapse whitespace, fold typographic variants."""
    return re.sub(r"\s+", " ", _normalise_typography(str(text))).strip().lower()


def is_empty(value) -> bool:
    values = value if isinstance(value, list) else [value]
    return not values or all(str(v).strip().upper() in EMPTY for v in values)


def measure(path: Path) -> dict:
    cache = json.loads(path.read_text(encoding="utf-8"))
    stats = {"records": 0, "quotes": 0, "verbatim": 0, "actor_na": 0,
             "recipient_na": 0, "paraphrased": []}

    for run in cache.values():
        document = normalise(run.get("text", ""))
        for record in run.get("final", []):
            attributes = record.get("attributes") or {}
            stats["records"] += 1
            quote = attributes.get("anchor_quote")
            if quote:
                stats["quotes"] += 1
                if normalise(quote) in document:
                    stats["verbatim"] += 1
                else:
                    stats["paraphrased"].append(
                        (record.get("event_type"), str(quote)[:70]))
            stats["actor_na"] += is_empty(attributes.get("actor"))
            stats["recipient_na"] += is_empty(attributes.get("recipient"))

    return stats


def report(path: Path) -> None:
    if not path.exists():
        print(f"{path}: not built — run demo/build_example_cache.py")
        return

    s = measure(path)

    def pct(n, d):
        return f"{100 * n / d:.0f}%" if d else "—"

    print(f"\n{path}")
    print(f"  records                {s['records']}")
    print(f"  anchor quote verbatim  {s['verbatim']}/{s['quotes']}"
          f"  ({pct(s['verbatim'], s['quotes'])})")
    print(f"  actor N/A              {s['actor_na']}/{s['records']}"
          f"  ({pct(s['actor_na'], s['records'])})")
    print(f"  recipient N/A          {s['recipient_na']}/{s['records']}"
          f"  ({pct(s['recipient_na'], s['records'])})")
    for event_type, quote in s["paraphrased"]:
        print(f"    not in the document — {event_type}: {quote!r}")


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or [DEFAULT]
    for path in paths:
        report(path)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
