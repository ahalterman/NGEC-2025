# Getting the `wiki` and `geonames` indices onto a machine

NGEC's actor resolution (step 5) and geolocation (step 2) both query one
Elasticsearch node holding two indices:

| Index      | Used by                          | Documents (reference box) |
|------------|----------------------------------|---------------------------|
| `wiki`     | actor resolution, entity linking | 7,601,204                 |
| `geonames` | location resolution (mordecai3)  | 13,250,817                |

A single Elasticsearch node keeps **all of its indices in one data directory**,
so both of these live together in whatever directory is mounted at
`/usr/share/elasticsearch/data`. That is the one fact everything else here
follows from.

There are two ways to get them:

- **Path A — mount a pre-built data directory.** Download the published
  archive, unpack it, and point Elasticsearch at it. Minutes, plus the
  download. Only works on Elasticsearch 7.10.x.
- **Path B — build both indices from source dumps.** Half an hour for GeoNames,
  most of a day for Wikipedia. For a newer Wikipedia dump, a different
  gazetteer, or a changed index format.

Path A stands on its own: nothing in it is NGEC-specific except the contents of
the indices, so it is also the recipe for reusing the `wiki` index in an
unrelated project. See
[Reusing the index elsewhere](#reusing-the-index-elsewhere).

To find out which of these you need on this machine:

```shell
python3 setup/doctor/ngec_doctor.py
```

The setup doctor reports whether Elasticsearch is answering, whether both
indices are there, and whether their document counts look like a complete load
or one that died part-way. It prints the right command for whichever is wrong.
See [`setup/doctor/README.md`](../setup/doctor/README.md).

> ⚠️ **There is currently no public download URL for the archive.** The
> address the root `README.md` used to give,
> `https://andrewhalterman.com/files/geonames_wiki_index_2023-03-02.tar.gz`,
> returns **HTTP 404** (checked 2026-09-09). `PREBUILT_INDEX_URL` in
> `setup/doctor/ngec_doctor.py` is the single constant `"TODO"`, and the setup
> console refuses to run any command containing it. Fill that constant in once
> an archive is published — the doctor, this document and `README.md` should
> then agree. Until then, get the archive from Andy directly, or build it
> (Path B).

---

## Path A — mount a pre-built data directory

The archive is a tar of an Elasticsearch **data directory**, which you mount
straight into the container. A 7.10 data directory only opens on
Elasticsearch 7.10.x, which is why the version is pinned below.

**You need:** Docker, and about 25 GB of free disk — roughly 10 GB for the
tarball and 13 GB for the unpacked data directory. Delete the tarball afterwards.

### 1. Download and unpack

```shell
mkdir -p ~/ngec-es-data
cd ~/ngec-es-data
curl -LO <URL of the archive>
tar -xzf geonames_wiki_index_*.tar.gz
mv geonames_index wikigeo_index          # the folder name says "geo"; it holds both
```

That is for the 2023 archive, whose directory name is historical. Renaming it
is optional but saves the next person from assuming it only has the gazetteer
in it. Newer archives, made by `tools/publish_index.sh`, are named
`wikigeo_index.tar.gz` and already unpack to `wikigeo_index/`, so there is
nothing to rename.

Note the **absolute** path of the result. Docker will not expand `~`: given
`~/…` it silently creates a directory named `~`.

### 2. Start Elasticsearch over it

```shell
docker run -d --name ngec-es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  --restart unless-stopped \
  -v /absolute/path/to/wikigeo_index:/usr/share/elasticsearch/data \
  elasticsearch:7.10.1
```

Flag by flag, and why these and not others — this is the exact configuration of
the container the reference machine has been serving from, read back out of
`docker inspect`:

| Flag | Why |
|---|---|
| `elasticsearch:7.10.1` | The version the index was built with. A 7.10 data directory will not open on Elasticsearch 8. |
| `-e discovery.type=single-node` | Otherwise the node waits to form a cluster and never becomes available. |
| `-p 9200:9200` | NGEC connects to `localhost:9200` by default. |
| `--restart unless-stopped` | Containers default to `restart=no`, which is why a reboot leaves Elasticsearch down and the pipeline mysteriously broken. `unless-stopped` rather than `always` so that stopping it deliberately sticks. |
| `-v …:/usr/share/elasticsearch/data` | The whole point: the indices live in this directory, not in the image. |

**No memory flags.** The reference container sets no `ES_JAVA_OPTS` and no
container memory limit; it runs on the image's default 1 GB heap and serves
these two indices fine. If you are on a memory-constrained host and want to pin
it, add `-e ES_JAVA_OPTS="-Xms2g -Xmx2g"` — but that is a tuning decision, not
part of the recipe.

**Never run two Elasticsearch containers against one data directory.** They
corrupt it, and they collide on port 9200. If you already have one running
(`docker ps`), stop it before starting another.

### 3. Check that both indices are actually there

```shell
curl -s 'localhost:9200/_cat/indices?v'
```

```
health status index    docs.count  store.size
green  open   wiki        7601204      10.1gb
yellow open   geonames   13250817         2gb
```

Two failure modes worth naming:

- **An empty list.** The volume path in step 2 was wrong. Elasticsearch starts
  perfectly happily against an empty data directory rather than failing, so
  this is the only place it shows up. Fix the path and re-create the container.
- **A count far below the numbers above.** A load that died part of the way
  through. Re-download, or rebuild that one index (Path B) — the other index
  shares the data directory and is left alone.

`yellow` health on a single-node cluster means it is trying, and failing, to
allocate replica shards. It is harmless — the shipped `geonames` index is yellow
because it was built with `number_of_replicas: 1`. To silence it permanently,
see [Cluster health is yellow](README.md#cluster-health-is-yellow).

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
run end to end.

---

## Packaging an index to hand to someone else

`tools/publish_index.sh` does this. It stops the Elasticsearch container
serving port 9200 first, because a live data directory is not a consistent
thing to copy: an archive taken while segments are being written can unpack
into a corrupt index. Then it tars the data directory as `wikigeo_index/`,
restarts the container, and writes into `elasticsearch/dist/`:

- `wikigeo_index.tar.gz` — the archive, unpacking to `wikigeo_index/`
- `wikigeo_index.tar.gz.sha256` — its checksum, for recipients to verify
- `manifest.json` — document counts, dump dates and build dates for both
  indices, read from each index's `_meta` (below)

It refuses to publish an index that has no `_meta`, and asks before uploading
anything. The upload goes to `NGEC_PUBLISH_DEST`, an rsync target such as
`user@host:/srv/www/ngec/index/`; `--no-upload` packages without uploading.

The recipient's side is [Path A](#path-a--mount-a-pre-built-data-directory)
above.

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
