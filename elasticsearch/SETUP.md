# Getting the `wiki` and `geonames` indices onto a machine

NGEC's actor resolution (step 5) and geolocation (step 2) both query one
Elasticsearch node holding two indices:

| Index      | Used by                          | Documents (published 2026-09 index) |
|------------|----------------------------------|-------------------------------------|
| `wiki`     | actor resolution, entity linking | 7,936,742 (Wikipedia dump of 2026-09-01) |
| `geonames` | location resolution (mordecai3)  | 13,472,152 (GeoNames of 2026-09-23)      |

A single Elasticsearch node keeps **all of its indices in one data directory**,
so the two travel together: the published index is simply that directory,
archived.

- **Path A — download the pre-built index.** One command, about half an hour,
  mostly the 11.6 GB download.
- **Path B — build both indices from source dumps.** Half an hour for GeoNames,
  most of a day for Wikipedia. Only for a newer dump or a changed index format.

`python3 setup/doctor/ngec_doctor.py` (before installing) or `ngec doctor`
(after) reports whether Elasticsearch is answering and whether both indices
are there, and prints the command for whatever is wrong.

---

## Path A — download the pre-built index

**You need:** Docker, and about 28 GB of free disk while it unpacks (11.6 GB
archive, 15 GB unpacked; the archive is deleted afterwards).

With NGEC installed:

```shell
ngec download-index --start
```

That downloads `wikigeo_index_2026-09.tar.gz` (resuming if interrupted), checks
it against its published SHA-256, unpacks it to `~/ngec-es-data/wikigeo_index`,
and starts Elasticsearch over it on port 9200, where NGEC looks by default.
`--dest` puts it elsewhere; without `--start` it prints the `docker run` instead
of running it.

Without NGEC installed, the same by hand:

```shell
mkdir -p ~/ngec-es-data && cd ~/ngec-es-data
curl -LO -C - https://andrewhalterman.com/files/wikigeo_index_2026-09.tar.gz
curl -LO https://andrewhalterman.com/files/wikigeo_index_2026-09.tar.gz.sha256
sha256sum -c wikigeo_index_2026-09.tar.gz.sha256      # macOS: shasum -a 256 -c ...
tar -xzf wikigeo_index_2026-09.tar.gz && rm wikigeo_index_2026-09.tar.gz

docker run -d --name ngec-es \
  --user "$(id -u):0" \
  -p 9200:9200 \
  -e discovery.type=single-node \
  --restart unless-stopped \
  -v "$HOME/ngec-es-data/wikigeo_index":/usr/share/elasticsearch/data \
  elasticsearch:7.10.1
```

Why these flags and no others:

| Flag | Why |
|---|---|
| `elasticsearch:7.10.1` | The version the index was built with. A 7.10 data directory opens only on 7.10.x. |
| `--user "$(id -u):0"` | The unpacked files belong to you; the image otherwise runs as uid 1000 and cannot write to them. It accepts any uid whose group is 0. Unpack as your ordinary user, not with `sudo`: Elasticsearch refuses to run as root. |
| `-e discovery.type=single-node` | Otherwise the node waits to form a cluster and never becomes available. |
| `-p 9200:9200` | NGEC connects to `localhost:9200` by default. |
| `--restart unless-stopped` | So a reboot does not leave Elasticsearch, and so the pipeline, down. |
| `-v …:/usr/share/elasticsearch/data` | The indices live in this directory, not in the image. Use an absolute path: Docker does not expand `~`. |

No memory flags: the image's default 1 GB heap serves both indices. **Never run
two Elasticsearch containers against one data directory** — they corrupt it.

### Check that both indices are there

It takes a minute to open them. Then:

```shell
curl -s 'localhost:9200/_cat/indices?v&h=health,index,docs.count'
```

```
health index    docs.count
green  geonames   13472152
green  wiki        7936742
```

**An empty list** means the volume path was wrong: Elasticsearch starts
happily on an empty directory, so this is the only place it shows. Fix the path
and re-create the container. Where the index came from and when it was built
travels inside it, in each index's `_meta` (see [Provenance](#provenance)).

---

## Point NGEC at it

Nothing to do if it is on `localhost:9200`. Otherwise put the host and port in
the repo-root `.env` (`ES_HOST`, `ES_PORT`, and `ES_USER` / `ES_PASSWORD` if the
cluster wants credentials), which is what the tests and the demo read, and pass
them to `ngec.es_client.setup_es_client` in your own code. Set `NGEC_ES_URL` to
the same cluster as well: the index-building tooling in `tools/` and
`elasticsearch/` reads that variable instead, and nothing keeps the two in step.

### Reusing the index elsewhere

The container from Path A is a plain Elasticsearch 7.10.1 node. Nothing about
it depends on NGEC, so any project that wants a searchable local Wikipedia can
use it: start the container, then query `localhost:9200/wiki` with `curl` or any
Elasticsearch client.

```shell
curl -s 'localhost:9200/wiki/_search?q=title:Massachusetts&size=1&pretty'
```

Each `wiki` document carries `title`, `redirects`, `alternative_names`,
`short_desc`, `categories`, `intro_para` (the first paragraph only — the rest of
the article is discarded), `infobox`, `box_type`, `affiliated_people` and
`update`. The `geonames` documents are the GeoNames gazetteer rows.
[`es_wiki/README.md`](es_wiki/README.md) documents the field semantics.

If you share a data directory between projects, share the *container* too: one
node at a time, always. Unpacking a separate copy of the archive for each
project avoids the problem entirely.

---

## Path B — build the indices yourself

Only when the pre-built index is stale, or you want a different Wikipedia
dump, a different gazetteer, or a changed index format.

Elasticsearch (and, for the wiki build, Redis) run in Docker via
`compose-build.yml`; the loaders are small Python CLIs that run on the host
under `uv`. The full procedure, including the tunables, the resumable
`tools/rebuild_index.sh` wrapper, and the publishing step, is in
[`README.md`](README.md), [`es_wiki/README.md`](es_wiki/README.md) and
[`es_geonames/README.md`](es_geonames/README.md). What follows is the sequence
in one place.

**You need:** Docker, and free disk for one of two options:

- **About 60 GB** for the steps below, which read the compressed dump
  directly: ~25 GB for the Wikipedia dump, 2 GB for the GeoNames gazetteer,
  ~16 GB for the built index and ~16 GB for the backup of the old one.
- **About 180 GB** for the faster option, `tools/rebuild_index.sh`'s default,
  which also decompresses the dump up front into 117 GB of plain XML. That
  saves roughly 90 minutes, because both wiki loader passes otherwise
  decompress the `.bz2` themselves. The script only takes this path when at
  least 130 GB is free, and `--no-parallel-bunzip` turns it off.

Only the ~16 GB index needs to be kept afterwards; the rest can be deleted.

Before anything else: **back up the data directory and stop your normal
Elasticsearch container.** The build stack mounts the same directory, and two
nodes on one directory corrupt it. (Or don't: see
[Building somewhere else entirely](README.md#building-somewhere-else-entirely)
for how to build into a scratch index on a side port instead, which is what you
want if the live node is serving something.)

```shell
cp -r "${NGEC_ES_DATA:-elasticsearch/data/wikigeo_index}" /tmp/wikigeo_index.bak
docker stop <your-es-container>
```

Set `NGEC_ES_DATA` in the repo-root `.env` to the **absolute** path of the data
directory first, and pass `--project-directory .` to every compose command —
that flag is what makes compose read the repo-root `.env`. Without it, compose
comes up against an empty directory and the loaders build into nothing.

### GeoNames — about half an hour

```shell
docker compose --project-directory . -f elasticsearch/compose-build.yml up -d es
curl -s 'localhost:9200/_cat/indices?v'        # both indices present before you start

cd elasticsearch/es_geonames
uv run --group es-build python load_geonames_es.py all
cd ../..
```

`all` runs three stages: `download` fetches `allCountries.zip`,
`admin1CodesASCII.txt` and `admin2Codes.txt` from
<https://download.geonames.org/export/dump> into `./geonames_data/` and unzips
them; `recreate` deletes **only** the `geonames` index and recreates its
mapping; `load` bulk-loads the gazetteer. `reload` is `recreate` + `load`, for
when the files are already on disk.

Measured 2026-09-09 on the reference box: **1 min 30 s to download** (421 MB
zip, 1.8 GB unpacked), **~23 minutes to load** 13.5M rows at 8,500–10,500
rows/second, giving ~13.3M documents and 2 GB on disk.

### Wikipedia — "many hours", "about a day"

```shell
docker compose --project-directory . -f elasticsearch/compose-build.yml up -d   # es + redis

cd elasticsearch/es_wiki
mkdir -p data
curl -L https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
  -o data/enwiki-latest-pages-articles.xml.bz2

DUMP=data/enwiki-latest-pages-articles.xml.bz2
uv run --group es-build python load_wiki_es.py build_links "$DUMP"
uv run --group es-build python load_wiki_es.py load_redis  "$DUMP"
uv run --group es-build python load_wiki_es.py load_es --drop "$DUMP"
cd ../..
```

The dump is tens of GB compressed (23 GB on the reference machine) and there is
no need to decompress it — the loader reads `.bz2` directly. Redis is used only
to attach redirects at build time; the NGEC runtime never touches it.

Nobody has timed a full wiki build with the current loader, so **many hours —
about a day** remains the only honest estimate. Before committing to it, run the
one-minute fixture smoke test in [`es_wiki/README.md`](es_wiki/README.md), which
exercises all three stages on eight real articles fetched from `Special:Export`.

### Afterwards

```shell
curl -s 'localhost:9200/_cat/indices?v'        # the rebuilt index changed, the other did not
docker compose --project-directory . -f elasticsearch/compose-build.yml down
docker start <your-es-container>
```

`tools/rebuild_index.sh wiki|geonames|both` automates all of the above with the
guards the manual procedure depends on you remembering, and is resumable with
`--resume`. `tools/publish_index.sh` packages and uploads the result. Read
[`README.md`](README.md) before using either — as of 2026-09-09 neither has been
run end to end (the 2026-09 release was packaged by hand, as below).

---

## Packaging an index to hand to someone else

The published index is the data directory, archived. The steps, which
`tools/publish_index.sh` automates for a node on port 9200:

1. **Unregister any snapshot repositories** (`curl localhost:9200/_cat/repositories`;
   `curl -XDELETE localhost:9200/_snapshot/<name>` leaves the snapshots on disk).
   Registrations live in the cluster state, inside the data directory, so they
   would ship with it and every recipient's Elasticsearch would log a stack
   trace per repository at start.
2. **Stop the container** (`docker stop`). A live data directory is not a
   consistent thing to copy; after a clean stop its logs end in `stopped` /
   `closed`, and everything is on disk.
3. **Archive it as `wikigeo_index/`, owned 1000:0:**
   `tar --owner=1000 --group=0 --numeric-owner -C <parent> -cf - wikigeo_index | pigz > wikigeo_index_YYYY-MM.tar.gz`,
   then `sha256sum` it into `<archive>.sha256`, and start the container again.
4. **Test the archive as a recipient would** — unpack it somewhere new as an
   ordinary user, `docker run --user "$(id -u):0"` over it on a spare port, and
   check both counts and a clean log — before uploading it.
5. **Upload** the archive and its `.sha256` next to each other, and point
   `INDEX_URL` in `ngec/index_download.py` and `PREBUILT_INDEX_URL` in
   `setup/doctor/ngec_doctor.py` at it (a test keeps the two equal). Archive
   names are dated, so an old URL keeps working.

The 2026-09 release was made this way from the build node (port 9201):
11,604,992,023 bytes, SHA-256
`2e0328fd50b48f76984ba90df27145cfe3c552b005b87f571b1bfbd73b9dc266`.

### Provenance

Both loaders stamp build provenance into the index mapping's `_meta`, and
`_meta` is part of the index, so **it travels inside the archive**. It should record:

```json
"_meta": {
  "dump_file":   "enwiki-20260801-pages-articles.xml.bz2",
  "dump_date":   "2026-08-01",
  "build_date":  "2026-08-12",
  "code_commit": "15608d51b097c7047cd91d20fc8834d01acab05d",
  "doc_count":   7854807,
  "builder":     "NGEC elasticsearch/es_wiki/load_wiki_es.py"
}
```

for `wiki`, and the equivalent with `source`/`gazetteer_file` for `geonames`,
where `dump_date` is the GeoNames download date (the gazetteer carries no
version stamp of its own).

`code_commit` was added 2026-09-09. Two indices built from the same dump by
different versions of a loader are not interchangeable, and until then nothing
recorded which loader ran. It is `git rev-parse HEAD` at build time, and is
omitted rather than guessed when the loader runs outside a checkout.

Read it back with:

```shell
curl -s 'localhost:9200/wiki/_mapping' | python3 -m json.tool
```

Caveat: `load_es` re-stamps `_meta` on *every* run, including one without
`--drop`, so an index that has had a small fixture merged into it will describe
itself as that fixture. Only trust `_meta` after a `--drop` build.
