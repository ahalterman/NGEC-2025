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

Only `categories`, `intro_para`, `title`, `alternative_names`, `redirects`,
`box_type` and `infobox` are declared in `wiki_mapping.json`; `short_desc`,
`affiliated_people` and `update` get Elasticsearch's dynamic mapping. That is
what the shipped index has too, so a rebuild reproduces it exactly.

⚠️ **Known mismatch.** `ngec/actors/wiki_matcher.py` (lines 405, 408, 410) has
three `match` clauses against `title.folded`, `redirects.folded` and
`alternative_names.folded`. No `folded` sub-field or analyzer is defined here,
in the shipped index, or anywhere else in the repo, and a `match` on a field
that does not exist matches nothing without erroring. Those three clauses are
therefore inert, in both a freshly built index and the pre-built one. Adding a
`folded` sub-field would change retrieval and scoring, so it is a decision, not
a typo — see `docs/memos/2026-09-09-demo-review/es_index_verification.md`.

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

For reference, on the reference box a 3,090-page slice of the real dump took 9 s
for `build_links` (620 redirect targets) and 17 s for `load_es` (2,080 documents
indexed — the rest are redirects, disambiguation and maintenance pages the
loader filters out). Those rates do not extrapolate honestly to the full dump;
see the estimates below.

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

## Steps

This is a long process (the dump is tens of GB compressed). Nobody has timed a
full build with the current loader, so **many hours — about a day** is the only
honest estimate. Run the fixture smoke test above first.

1. **Download the dump** into `data/`:

   ```bash
   cd elasticsearch/es_wiki
   mkdir -p data
   curl -L https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
     -o data/enwiki-latest-pages-articles.xml.bz2
   ```

   (`curl` ships with recent Windows, macOS, and Linux. There's no need to
   decompress — the loader reads `.bz2` directly.)

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
   (default 10).

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
