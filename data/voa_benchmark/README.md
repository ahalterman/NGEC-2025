# VOA benchmark corpus

A small set of Voice of America news stories on international and conflict topics,
for benchmarking the NGEC pipeline.

## Contents

`voa_stories.jsonl` — one JSON object per line:

| field | meaning |
|---|---|
| `id` | VOA article ID |
| `url` | canonical article URL |
| `section` | VOA desk (Ukraine, Middle East, Africa, ...) |
| `title` | headline |
| `author` | VOA staff byline |
| `published` | publication timestamp (UTC) |
| `keywords` | VOA topic tags |
| `text` | full article body, whitespace-normalized |

```python
import json

with open("data/voa_benchmark/voa_stories.jsonl", encoding="utf-8") as f:
    stories = [json.loads(line) for line in f]
```

## Licensing

Voice of America is part of the US Agency for Global Media. Original text produced
exclusively by VOA is a **US government work and is in the public domain**, which is
why this corpus can be redistributed here. See
[USAGM's content terms](https://www.usagm.gov/work-with-us/content-requests/voa/).

Every story in this file was filtered to keep only VOA-authored material. Excluded:

- **Wire-service content** (AP, Reuters, AFP and others) that VOA republishes under
  licence — copyrighted by those agencies, not public domain.
- **Stories bylined `VOA News` / `Voice of America`**, which VOA states "may contain
  information from wire service reports."
- **USAGM grantee networks** (RFE/RL, Radio Free Asia, Alhurra/MBN, BenarNews). These
  are private non-profit corporations rather than federal agencies, so their output is
  copyrighted even though USAGM funds them.
- Stories that incorporate wire material, flagged by VOA's own footer ("Some
  information for this report came from / was provided by The Associated Press,
  Agence France-Presse and Reuters") or a wire reporting credit.

Stories that cite a wire in passing ("according to Reuters", "told AFP") are kept:
the prose is VOA's own and the underlying facts aren't copyrightable.

Article text is taken from the article paragraphs only; inline video-player and
image-caption boilerplate is stripped.

## Attribution

USAGM asks that reuse credit **voanews.com / Voice of America / VOA**. Each record
retains its source `url` for per-story attribution.

## Scope

International and conflict news only, matching NGEC's domain. Sport, entertainment,
arts and US domestic sections are excluded. Stories are at least 900 characters so
the set isn't padded with newsbrief stubs.

## Regenerating

This corpus is produced by a standalone scraper kept outside this repository
(`voa-scraper/`), so the public repo carries the data but not the collection code.
