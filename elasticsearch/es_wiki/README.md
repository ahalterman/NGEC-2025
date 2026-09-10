# Building / updating the Wikipedia index

Builds the `wiki` Elasticsearch index used by NGEC's actor resolution and entity
linking, from an English Wikipedia XML dump.

The whole flow is one cross-platform Python tool
([`load_wiki_es.py`](load_wiki_es.py)) plus Elasticsearch and Redis containers.
It runs identically on Linux, macOS, and Windows.

> **This updates only the `wiki` index.** It runs against the shared live data
> dir (which also holds the `geonames` index) and only ever touches `wiki`, so
> the `geonames` index is left intact. See [../README.md](../README.md) for the
> why.

Redis is used **only at build time** (to attach each article's redirects). The
NGEC runtime does not need Redis.

## Index format

Each indexed article is stored as:

- `title` — the title of the Wikipedia page (no underscores)
- `redirects` — every page that redirects to this page
- `alternative_names` — alternative names, from bold phrases in the first
  sentence and from infobox name fields
- `short_desc` — Wikipedia's "short description" of the article
- `categories` — the Wikipedia categories associated with this page
- `intro_para` — cleaned text of the first paragraph. Everything after the intro
  paragraph is discarded, for space reasons.
- `infobox` — the article's side infobox, if it has one
- `box_type` — infoboxes come in different formats, e.g. "legislature",
  "military unit", "settlement"
- `affiliated_people` — the 'leaders', 'founded_by', or 'founder' infobox fields
  if present (currently not used downstream)
- `update` — the date this document was indexed
- `redirect_count` — how many pages redirect here, as an integer. Wikipedia's
  cheapest importance signal, stored so it can be used at *retrieval* time
  (`len(redirects)` already serves the ranker).

Every one of these is now declared explicitly in `wiki_mapping.json`. Until the
2026-09 rebuild, `short_desc`, `affiliated_people` and `update` were left to
Elasticsearch's dynamic mapping; they are now written out with exactly the types
dynamic mapping assigned them (`text` + `.keyword(256)`, and `date`), so nothing
about them changes — the mapping file simply stops under-describing the
document.

**`folded` sub-fields, and why nothing queries them.** Indices built from
2026-09 on carry an `analysis` block with a `folded` analyzer (standard
tokenizer + `lowercase` + `asciifolding`) and a `lowercase_folded` normalizer,
plus `.folded` (accent-folded `text`) and `.keyword` (accent- and
case-normalized `keyword`) sub-fields on `title`, `redirects` and
`alternative_names`. They exist so that accent-insensitive matching — "Zoe" for
"Zoë", "Frederic" for "Frédéric" — is *possible* without another full rebuild.
**No query uses them today.** `wiki_matcher.py` once had three `match` clauses
against the `.folded` sub-fields, but no index has ever had the sub-field, so
they were inert and contributed nothing to any published number; they were
removed when the sub-fields were added, so this build is byte-identical in
retrieval to the last one. Switching them on roughly doubles the title-fuzzy
contribution and moves every `raw_es_score`, so it is a deliberate ranking
change to evaluate with `setup/train_wiki_model` first. The same goes for
`redirect_count`, which is only read by the `use_importance` clauses that no
caller sets. See
`docs/memos/2026-09-09-demo-review/wiki_index_future_proofing.md`.

The mapping also stamps `_meta.schema_version: 2` at index-creation time to
record which of these sub-fields an index has. Note the caveat below: `load_es`
re-stamps `_meta` wholesale at the end of a build, so `schema_version` survives
only if it is also in the loader's stamp.

The whole document is about 13 mapped fields, and it stays that small **only
because `infobox` is `flattened`** — one mapped field no matter how many
infobox keys a page has. Mapped as an `object` instead, the thousands of
distinct infobox keys across en-wiki would blow past the default
`index.mapping.total_fields.limit` of 1000 inside the first million pages. The
limit is a dynamic setting and can be raised at any time; the `flattened` type
is the thing to protect.

## How it works

`load_wiki_es.py` runs three stages, in order:

1. `build_links` — scan the dump and collect every page redirect into
   `data/redirect_dict.pkl`.
2. `load_redis` — load that redirect dict into Redis.
3. `load_es` — parse the dump again, format each article, and bulk-load it into
   the `wiki` index.

Stage 1 reads the `<redirect title="...">` attribute directly off each `<page>`
element rather than parsing every page's wikitext, which is why it is no longer
the overnight step it used to be.

### Only the lead section is parsed

Everything stage 3 stores about an article's *body* comes from its **lead
section** — the text above the first `==` heading. `intro_para` is the lead with
markup stripped, `alternative_names` are the `'''bold'''` phrases in it,
`short_desc` is its `{{Short description}}`, and the infobox (and with it
`box_type` and `affiliated_people`) is the first `{{Infobox ...}}` template in
it. None of that ever looked below the first heading, but the loader used to
hand mwparserfromhell the **whole article** to get it, which was essentially all
of stage 3's CPU time (see [Speed](#speed)).

So `parse_wiki_article` now cuts the wikitext at the first line starting with
`==` (`LEAD_CUT_PATTERN`) and parses only that prefix. Categories are the one
stored field that lives below the lead; they are read straight out of the raw
wikitext with a regex (`CATEGORY_PATTERN`) instead.

**What that covers.** `[[Category:Name]]` anywhere in the article, the lowercase
`[[category:Name]]` spelling, and the `[[Category:Name|sortkey]]` form. Category
names are stored exactly as written, with no whitespace normalisation, which is
what the old code did too.

**What it does not.** The cut is a plain text scan, so — unlike
mwparserfromhell — it does not know about `<!-- comments -->` or `<nowiki>`: a
`==heading==` line hidden inside one of those ends the lead here, but would not
in MediaWiki. (On the benchmark slice that never happened.) A category applied
by a *template* rather than a literal link is still invisible, as it was before,
and an inline `[[:Category:X]]` link is deliberately not treated as a
categorisation. An infobox below the first heading is still not picked up — it
never was.

**What changed in the output.** Measured on the 3,090-page slice, against the
same slice built with the old whole-article parse:

- **Categories gained 1,497 names across 1,138 articles.** 1,484 of them are
  the `[[Category:X|sortkey]]` form, which the old code silently dropped:
  `strip_code()` renders a piped wikilink as its *display text*, so
  `[[Category:Alabama| ]]` came out as `" "` and the `Category:` prefix the old
  regex looked for was gone. Twelve are plain category links that sat outside
  the article's last section (the old code only looked at the last section), and
  one is a lowercase `[[category:...]]` — the old regex was case-sensitive.
- **Two category names were lost, both junk**: a stray trailing space on one of
  *Blue crane*'s names, and a sentence of talk-page prose containing the word
  "Category:" that the old regex had captured on
  *Wikipedia:Building Wikipedia membership* (whose one real category is a
  sortkey form, and is now found).
- **Five articles are no longer truncated the wrong way round.** For these,
  mwparserfromhell failed to see real `==heading==` lines and returned the
  *entire article* as the lead: *Alcoholics Anonymous*, *Alex Lifeson*,
  *American Registry for Internet Numbers*, *Berthe Morisot* and
  *Brian Kernighan*. Their `intro_para` was 10–30× too long (51,690 characters
  for *Alcoholics Anonymous*), `alternative_names` picked up bold phrases from
  the body, and *Brian Kernighan* was dropped from the index altogether — the
  swallowed body text contained `Category:`, which the loader reads as "this is
  a category page, skip it". The slice indexes 2,081 documents now instead of
  2,080.
- Nothing else moved: `title`, `redirects`, `short_desc`, `infobox`, `box_type`
  and `affiliated_people` are identical for all 2,080 shared documents.

## Smoke-testing the loader without the full dump

A full build is an all-day commitment, which is a bad way to find out you have
the wrong Redis port. `Special:Export` hands back a handful of real articles in
exactly the same XML format as the dump, so you can exercise all three stages
end to end in about a minute:

```bash
cd elasticsearch/es_wiki
mkdir -p data
curl -s -X POST 'https://en.wikipedia.org/w/index.php?title=Special:Export&action=submit' \
  --data-urlencode 'pages=Massachusetts
Boston
Emmanuel Macron
Angela Merkel
Paris' --data 'curonly=1&wpDownload=1' -o data/fixture.xml
bzip2 -f data/fixture.xml            # optional; the loader reads plain .xml too

uv run --group es-build python load_wiki_es.py build_links data/fixture.xml.bz2
uv run --group es-build python load_wiki_es.py load_redis  data/fixture.xml.bz2
uv run --group es-build python load_wiki_es.py load_es     data/fixture.xml.bz2
```

Include **Massachusetts**. `WikiClient.check_wiki` (`ngec/actors/wiki_matcher.py`)
gate-checks the index at startup by searching for it and reading `redirects` off
the top hit; an index without it fails to open with a misleading "outdated
Wikipedia index" error.

Point this at a scratch node, not your live one — `NGEC_ES_URL` and
`NGEC_REDIS_PORT` are what you override. A fixture load with no `--drop` merges
into whatever index is already there.

For reference, on the reference box a 3,090-page slice of the real dump (126 MB
of XML) indexes 2,081 documents — the rest are redirects, disambiguation and
maintenance pages the loader filters out. Measured 2026-09-09, before and after
the switch to lead-only parsing:

| | `build_links` | `load_redis` | `load_es` | total |
|---|---|---|---|---|
| defaults, reading the `.bz2` | 8.8 s | 3.2 s | 19.0 s | **31.0 s** |
| defaults, `.bz2`, lead-only | 8.9 s | 3.2 s | **10.9 s** | **23.0 s** |
| plain `.xml`, `--threads 32` | 3.6 s | 3.2 s | 12.2 s | **19.1 s** |
| plain `.xml`, `--threads 32`, lead-only | 3.6 s | 3.2 s | **5.8 s** | **12.8 s** |

All four rows produce the same documents (2,080 before the lead-only change,
2,081 after — see [Only the lead section is
parsed](#only-the-lead-section-is-parsed)). Those rates do not extrapolate
honestly to the full dump — the low-numbered pages at the front of a dump are
its oldest and longest articles, so this slice averages 41 KB a page against
the dump's ~4 KB. See [Speed](#speed) for what the two knobs are and
[Steps](#steps) for the full-build estimate.

### Two builds of the same dump are byte-identical

They did not use to be. `clean_names()` ended in `list(set(...))`, and Python
randomises string hashing per process, so `alternative_names`, `redirects` and
`affiliated_people` came out in a different order on every run — 845 of 2,080
documents on this slice differed in list order between two runs of the *same*
loader, with no other difference. `clean_names()` and `read_clean_redirects()`
now use `sorted(set(...))`, and the one remaining stored list (`categories`) is
in document order, so two builds of one dump produce identical documents: two
consecutive slice builds, exported to canonical JSONL, now compare equal byte
for byte. A build can be verified with plain `diff`.

(Element order never affected retrieval — Elasticsearch puts a position gap
between array elements, so no phrase match spans two of them. The point of the
change is reproducibility, not scoring.)

## Prerequisites

The build-time Python deps (`lxml`, `mwparserfromhell`, `plac`, `redis`) live in
the `es-build` dependency group, which is **not** installed by default — most
users download a prebuilt index and never need them.

Every `uv run` below passes `--group es-build`, which installs them on demand,
so there is no separate setup step. Don't drop that flag: `uv sync` is exact and
prunes anything outside the groups you name, so an unrelated `uv sync` for
normal work silently removes these again. Pre-fetching with `uv sync --group
es-build` is fine, it just isn't durable on its own.

> ⚠️ **Pass your install extras too.** `uv sync` is exact about extras as well
> as groups, so `uv run --group es-build ...` on a machine installed with
> `--extra cu12 --extra vllm` re-resolves the environment *without* them and
> replaces your PyTorch build with the default PyPI (CUDA 13) one. On the
> reference box `uv sync --group es-build --dry-run` reports "would uninstall
> 146 packages". Repeat whichever extras you installed with, on every command:
>
> ```bash
> uv run --extra cu12 --extra vllm --group es-build python ...
> ```
>
> The extras (`cpu` / `cu12` / `cu13`, plus `models` and `vllm`) are the ones
> in the repo-root README's install section, and they are mutually exclusive.

## Speed

Two things dominate a full build, and neither is Elasticsearch:

**1. Decompressing the dump, twice.** `build_links` and `load_es` each read the
whole dump, and Python's `bz2` decompresses it at about 38 MB/s on one core —
roughly 45 minutes per pass for ~105 GB of XML. The loader reads a plain `.xml`
just as happily as a `.bz2`, so decompressing once up front and pointing both
stages at the plain file removes one of those passes outright. If you fetch the
**multistream** dump you can also do that decompression in parallel: it is a
concatenation of independent bz2 streams and its companion
`-index.txt.bz2` lists every stream's byte offset, so the file can be cut at
those offsets and the pieces decompressed by one process each.
`elasticsearch/tools/parallel_bunzip2.py` does exactly that (usage:
`python3 parallel_bunzip2.py DUMP.bz2 DUMP-index.txt.bz2 OUT.xml [workers]`,
standard library only) and verifies its output: as many `<page>` elements as
the index has lines, counted with a carry across read boundaries, and a
`<mediawiki>` root at the start and `</mediawiki>` at the end. Two bugs its
first version had are worth knowing about: counting tags in fixed-size reads
misses the handful that straddle a boundary (about 7 per 117 GB), and the
index's first offset is the first *page* stream, not byte 0, so a worker that
starts there drops the stream holding the root tag and `<siteinfo>` and
produces a headerless file. Cost: ~117 GB of disk for the 2026-09 dump.

**2. `mwparserfromhell`, on one core per worker.** Profiling `load_es` showed
essentially *all* of the per-article cost was the single
`mwparserfromhell.parse()` call — about 2 MB of wikitext per second per core.
Everything after it (the section splitting, the regexes, the infobox walk) was
noise by comparison.

That call now sees only the article's lead section rather than the whole article
(see [Only the lead section is
parsed](#only-the-lead-section-is-parsed)), which is a **13× cut in parse
time**: running `parse_wiki_article` over the whole slice on one core, best of
three, went **61.6 s → 4.7 s**. Because the parse was the whole cost, `load_es`
itself roughly halves — 12.2 s → 5.8 s on the slice at `--threads 32` — and what
is left of stage 3 (decompressing, walking the XML, bulk-indexing) is now what
dominates.

`--threads` still exists and still defaults to **10**; on a 32-core box pass
`--threads 32`. It just matters less than it did. Measured on the slice with the
old whole-article parse, `load_es` went 18.6 s → 17.0 s → 16.7 s → 16.5 s at
10 / 16 / 24 / 32 workers, because the dump-reading loop is serial and quickly
becomes the limit; with the parse 13× cheaper, that serial loop binds sooner
still.

Things that are *already* done and don't need doing again: the index is created
with `number_of_replicas: 0` (in `wiki_mapping.json`) and `load_es` sets
`refresh_interval: -1` for the duration of the load and restores it in a
`finally`. Elasticsearch bulk indexing is not a bottleneck — 0.6 s of the
slice's 11 s.

## Steps

This is a long process, though less long than it was. The old **5–7 hour**
estimate (reference box, 32 cores, 125 GB RAM, `--threads 32`, pre-decompressed
dump, scaling the slice by uncompressed bytes) was dominated by the
whole-article parse, which is now 13× cheaper; what is left is reading ~105 GB
of XML and bulk-indexing it. Scaling the same way gives roughly **3–4 hours**,
and half again as long at the defaults against the `.bz2` — but that
extrapolation is shakier than it looks, for the reason given under the slice
timings, so treat the `timings.tsv` of an actual full build as the real answer.
Run the fixture smoke test above first.

1. **Download the dump** into `data/`:

   ```bash
   cd elasticsearch/es_wiki
   mkdir -p data
   curl -L https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
     -o data/enwiki-latest-pages-articles.xml.bz2
   ```

   (`curl` ships with recent Windows, macOS, and Linux. There's no need to
   decompress — the loader reads `.bz2` directly.)

   Prefer a **dated, multistream** dump if you have the disk for it:

   ```bash
   D=20260901        # pick the newest complete date from https://dumps.wikimedia.org/enwiki/
   curl -L https://dumps.wikimedia.org/enwiki/$D/enwiki-$D-pages-articles-multistream.xml.bz2 \
     -o data/enwiki-$D-pages-articles-multistream.xml.bz2
   curl -L https://dumps.wikimedia.org/enwiki/$D/enwiki-$D-sha1sums.txt -o data/sha1sums.txt
   ( cd data && grep multistream.xml.bz2 sha1sums.txt | sha1sum -c - )
   ```

   The dated name is what makes `_meta.dump_date` mean something (see
   [Index metadata](#index-metadata)), the checksums let you verify a 25 GB
   download, and the multistream layout is what allows the parallel
   decompression described under [Speed](#speed).

2. **Back up** the live data dir, then **stop your normal Elasticsearch
   container** — the build stack mounts the same data dir, so the two ES nodes
   must never run at once (they'd corrupt the data and collide on port 9200):

   ```bash
   cd ../..                                    # back to the repo root
   cp -r "${NGEC_ES_DATA:-elasticsearch/data/wikigeo_index}" /tmp/wikigeo_index.bak
   docker stop <your-es-container>
   ```

3. **Start Elasticsearch + Redis** against the shared live data dir:

   ```bash
   docker compose --project-directory . -f elasticsearch/compose-build.yml up -d
   ```

   The path comes from `NGEC_ES_DATA` (repo-root `.env`), defaulting to
   `./elasticsearch/data/wikigeo_index`. `--project-directory .` is what makes
   compose read that `.env`; without it you get an empty data dir and no
   indices. See [../README.md](../README.md).

   Confirm both indices are present (so you don't clobber geonames):

   ```bash
   curl -s 'localhost:9200/_cat/indices?v'
   ```

4. **Build the redirect links:**

   ```bash
   cd elasticsearch/es_wiki
   DUMP=data/enwiki-latest-pages-articles.xml.bz2
   uv run --group es-build python load_wiki_es.py build_links $DUMP
   uv run --group es-build python load_wiki_es.py load_redis  $DUMP
   ```

5. **Load the new wiki index:**

   ```bash
   DUMP=data/enwiki-latest-pages-articles.xml.bz2
   uv run --group es-build python load_wiki_es.py load_es --drop $DUMP
   ```

   `--drop` records the before-stats, deletes the old `wiki` index, then
   creates it fresh from `wiki_mapping.json` and loads, so the before/after
   doc-count comparison in the log is meaningful. Omit `--drop` for a
   first-time build. Tunables: `--es-batch` (default 5000) and `--threads`
   (default 10 — **raise it to your core count**; see [Speed](#speed)).

6. **Verify** the `wiki` count changed, `geonames` is unchanged, and a sample
   document has the `redirects` field the runtime checks for:

   ```bash
   curl -s 'localhost:9200/_cat/indices?v'
   curl -s 'localhost:9200/wiki/_search?q=title:Massachusetts&size=1&pretty'
   ```

7. **Stop** the build stack and **bring your normal ES back** — the updated data
   is already in the shared data dir:

   ```bash
   cd ../..                                    # back to the repo root
   docker compose --project-directory . -f elasticsearch/compose-build.yml down
   docker start <your-es-container>
   ```

## Notes

- `build_links` always writes `redirect_dict.pkl` into `data/` *next to this
  script* — the path is derived from the script's own location and there is no
  override. On the full dump that file is large. If you want the intermediates
  somewhere else (off the checkout, on a bigger disk), make
  `elasticsearch/es_wiki/data` a symlink to wherever you want them; that is what
  the 2026-09 build does.
- Override service locations with `NGEC_ES_URL` (default
  `http://localhost:9200/`), `NGEC_REDIS_HOST` (default `localhost`), and
  `NGEC_REDIS_PORT` (default `6379`).
- Windows: run the same `uv run --group es-build python ...` commands in PowerShell. The loader
  uses Python's `bz2`/`lxml` parsing and Python's Redis/ES clients, so there are
  no shell-specific steps.

## Index metadata

`load_es` stamps build provenance onto the index when it finishes, in the
mapping's `_meta`. Elasticsearch stores it verbatim and never interprets it, so
it travels with the index — including into a copied data directory:

```bash
curl -s 'localhost:9200/wiki/_mapping' | python -m json.tool
```

```json
"_meta": {
  "dump_file": "enwiki-latest-pages-articles.xml.bz2",
  "dump_date": "2026-08-01",
  "build_date": "2026-08-12",
  "code_commit": "15608d51b097c7047cd91d20fc8834d01acab05d",
  "doc_count": 7854807,
  "builder": "NGEC elasticsearch/es_wiki/load_wiki_es.py"
}
```

`dump_date` is the dump file's modification time — in practice, when you
downloaded it. The Wikipedia XML carries no generation timestamp in its header
and the canonical download is named "latest", so that's the best available
answer. If you want the true dump date on record, fetch a dated dump
(`enwiki-20260801-pages-articles.xml.bz2`) and `dump_file` will carry it.

`--drop` (and any manual index delete) clears `_meta` along with the index. It's
re-stamped at the end of the next successful load.

⚠️ Every `load_es` re-stamps `_meta` with the file *it* just read, including a
run without `--drop`. So after merging a small fixture into a full index, the
provenance describes the fixture and undercounts nothing but describes the wrong
dump. Treat `_meta` as trustworthy only after a `--drop` build; if you have
merged into an index you intend to publish, re-stamp it by hand.

## Files

- `load_wiki_es.py` — the build_links / load_redis / load_es tool.
- `wiki_mapping.json` — field mappings for the `wiki` index (includes the
  `redirects` field the actor resolver validates).
