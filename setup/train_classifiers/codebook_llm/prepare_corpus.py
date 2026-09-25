"""
Step 0: build the annotation pool from the VOA article corpus.

The raw corpus is ~319,000 JSON files, one article each, covering 2002-2024.
About 41% of them are from 2009, so a simple random sample would be a sample of
2009. This script filters the corpus down to usable English news articles and
then draws a year-stratified pool that the annotation stages work from.

Output is a single JSONL file whose records use NGEC's field names, so the same
records can be fed straight to `PloverCoder.process()` later:

    {"id": ..., "event_text": ..., "pub_date": ..., "year": ..., "section": ...}

Usage:
    python prepare_corpus.py --corpus ~/projects/eng_voanews/article \\
        --out data/voa_pool.jsonl --n 60000
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# VOA appends this to every article title; it carries no information and would
# otherwise show up in every document the classifier sees.
TITLE_SUFFIX = re.compile(r"\s*\|\s*Voice of America.*$", re.IGNORECASE)

MIN_CHARS = 300
MAX_CHARS = 6000
MIN_PARAGRAPHS = 2


def clean_title(title: str) -> str:
    return TITLE_SUFFIX.sub("", title or "").strip()


def build_event_text(doc: dict) -> str:
    """Title + body, matching how RUNNING.md feeds VOA into the pipeline."""
    title = clean_title(doc.get("title", ""))
    paragraphs = [p.strip() for p in doc.get("paragraphs", []) if p and p.strip()]
    if not paragraphs:
        return ""
    body = "\n".join(paragraphs)
    return f"{title}. {body}" if title else body


def is_usable(doc: dict, event_text: str) -> bool:
    """Filters that drop non-articles, stubs, and non-English text."""
    if not event_text:
        return False
    if not (MIN_CHARS <= len(event_text) <= MAX_CHARS):
        return False
    if doc.get("n_paragraphs", 0) < MIN_PARAGRAPHS:
        return False
    if doc.get("predicted_language") != "eng":
        return False
    # cld3 is occasionally confident that an English article is something else;
    # require it to agree rather than trusting predicted_language alone.
    cld3 = doc.get("cld3_detected_languages") or {}
    eng = cld3.get("eng") or {}
    if eng and not eng.get("is_reliable", False):
        return False
    if not doc.get("time_published"):
        return False
    return True


def dedup_key(event_text: str) -> str:
    """Hash of normalized text, to drop wire-copy reruns of the same story."""
    normalized = re.sub(r"\W+", " ", event_text.lower()).strip()
    return hashlib.sha1(normalized[:2000].encode("utf-8")).hexdigest()


def load_corpus(corpus_dir: Path):
    """Yield usable records. Malformed JSON files are skipped, not fatal."""
    seen = set()
    n_read = n_bad = n_dropped = n_dup = 0

    for path in sorted(corpus_dir.glob("*.json")):
        n_read += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            n_bad += 1
            continue

        event_text = build_event_text(doc)
        if not is_usable(doc, event_text):
            n_dropped += 1
            continue

        key = dedup_key(event_text)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)

        pub_date = str(doc["time_published"])[:10]
        yield {
            "id": doc.get("filename") or path.stem,
            "event_text": event_text,
            "title": clean_title(doc.get("title", "")),
            "pub_date": pub_date,
            "year": pub_date[:4],
            "section": doc.get("section") or "",
            "n_chars": len(event_text),
        }

    print(f"  read {n_read} files: {n_bad} unreadable, {n_dropped} filtered out, "
          f"{n_dup} duplicates")


def stratified_sample(records, n_target, seed=20260804):
    """
    Draw ~n_target records spread as evenly as possible across years.

    Years with fewer records than their share contribute everything they have;
    the leftover quota is redistributed over the years that still have surplus.
    This keeps 2009 from dominating without discarding thin years entirely.
    """
    rng = random.Random(seed)
    by_year = defaultdict(list)
    for rec in records:
        by_year[rec["year"]].append(rec)
    for year in by_year:
        rng.shuffle(by_year[year])

    years = sorted(by_year)
    remaining_years = set(years)
    quota = {}
    budget = min(n_target, sum(len(v) for v in by_year.values()))

    # Water-filling: repeatedly hand out an equal share, and give back whatever
    # the small years cannot use.
    while remaining_years and budget > 0:
        share = budget // len(remaining_years)
        if share == 0:
            break
        exhausted = set()
        for year in sorted(remaining_years):
            take = min(share, len(by_year[year]) - quota.get(year, 0))
            quota[year] = quota.get(year, 0) + take
            budget -= take
            if quota[year] >= len(by_year[year]):
                exhausted.add(year)
        if not exhausted:
            break
        remaining_years -= exhausted

    sample = []
    for year in years:
        sample.extend(by_year[year][: quota.get(year, 0)])
    rng.shuffle(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="~/projects/eng_voanews/article")
    ap.add_argument("--out", default="data/voa_pool.jsonl")
    ap.add_argument("--n", type=int, default=60000,
                    help="target pool size after year stratification")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    corpus_dir = Path(args.corpus).expanduser()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {corpus_dir} ...")
    records = list(load_corpus(corpus_dir))
    print(f"  {len(records)} usable articles")

    pool = stratified_sample(records, args.n, seed=args.seed)
    print(f"  sampled {len(pool)} into the pool")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in pool:
            f.write(json.dumps(rec) + "\n")

    year_counts = Counter(r["year"] for r in pool)
    print(f"\nWrote {out_path}")
    print("Year distribution:")
    for year in sorted(year_counts):
        print(f"  {year}  {year_counts[year]:6d}")


if __name__ == "__main__":
    main()
