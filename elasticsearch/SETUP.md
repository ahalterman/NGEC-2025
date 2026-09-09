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

There are three ways to get them, depending on what you were handed:

- **Path A — restore a snapshot archive.** The format this repo now publishes.
  Minutes. Portable across Elasticsearch versions, and verifiable.
- **Path B — mount a pre-built data directory.** The older archive format (the
  `geonames_wiki_index_*.tar.gz` that has been passed around). Also minutes, but
  only works on Elasticsearch 7.10.x.
- **Path C — build both indices from source dumps.** Half an hour for GeoNames,
  most of a day for Wikipedia. For a newer Wikipedia dump, a different
  gazetteer, or a changed index format.

Paths A and B stand on their own: nothing in them is NGEC-specific except the
contents of the indices, so either is also the recipe for reusing the `wiki`
index in an unrelated project. See
[Reusing the index elsewhere](#reusing-the-index-elsewhere).

To find out which of these you need on this machine:

```shell
python3 setup/doctor/ngec_doctor.py
```

The setup doctor reports whether Elasticsearch is answering, whether both
indices are there, and whether their document counts look like a complete load
or one that died part-way. It prints the right command for whichever is wrong.
See [`setup/doctor/README.md`](../setup/doctor/README.md).

> ⚠️ **There is currently no public download URL for either archive.** The
> address the root `README.md` used to give,
> `https://andrewhalterman.com/files/geonames_wiki_index_2023-03-02.tar.gz`,
> returns **HTTP 404** (checked 2026-09-09), and the copy committed at
> `setup/geonames_wiki_index_2023-03-02.tar.gz` is a truncated 14 MB fragment of
> a ~10 GB archive, not a usable index. `PREBUILT_INDEX_URL` in
> `setup/doctor/ngec_doctor.py` is the single constant `"TODO"`, and the setup
> console refuses to run any command containing it. Fill that constant in once
> an archive is published — the doctor, this document and `README.md` should
> then agree. Until then, get the archive from Andy directly, or build it
> (Path C).

---

## Path A — restore a snapshot archive

**You need:** Docker (<https://www.docker.com/get-started/>), and about 35 GB of
free disk. The archive is roughly 10 GB, the unpacked snapshot repository about
the same again, and the restored data directory about 13 GB — but the repository
can be deleted as soon as the restore finishes, so the peak is what to plan for,
not the resting size.

### 1. Unpack

```shell
mkdir -p ~/ngec-es
tar -xzf ngec-index-snapshot-YYYY-MM-DD.tar.gz -C ~/ngec-es
mkdir -p ~/ngec-es/data
```

You should now have `~/ngec-es/snapshots` (the repository) and an empty
`~/ngec-es/data` (where the restored indices will go). Note the **absolute**
path of both: Docker does not expand `~`, and given `~/…` it silently creates a
directory named `~`.

### 2. Start Elasticsearch with the snapshot repository mounted

```shell
docker run -d --name ngec-es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e path.repo=/snapshots \
  --restart unless-stopped \
  -v /absolute/path/to/ngec-es/data:/usr/share/elasticsearch/data \
  -v /absolute/path/to/ngec-es/snapshots:/snapshots \
  elasticsearch:7.10.1
```

`-e path.repo=/snapshots` is the one flag that is specific to this path:
Elasticsearch refuses to register a filesystem repository outside `path.repo`.
The other flags are explained in [Path B](#path-b--mount-a-pre-built-data-directory).

Wait for it to answer:

```shell
curl -s localhost:9200/_cluster/health
```

### 3. Register the repository and restore

```shell
curl -X PUT 'localhost:9200/_snapshot/ngec' \
  -H 'Content-Type: application/json' \
  -d '{"type":"fs","settings":{"location":"/snapshots"}}'

curl -s 'localhost:9200/_snapshot/ngec/_all' | python3 -m json.tool   # what's in it

curl -X POST 'localhost:9200/_snapshot/ngec/<snapshot-name>/_restore?wait_for_completion=true' \
  -H 'Content-Type: application/json' \
  -d '{"indices":"wiki,geonames","include_global_state":false}'
```

Restore fails if an index of the same name already exists and is open — that is
the intended safety, not a bug. Delete or close the old one first.

For the `wiki` index alone, pass `"indices":"wiki"`.

### 4. Verify

```shell
curl -s 'localhost:9200/_cat/indices?v'
curl -s 'localhost:9200/wiki/_mapping'     | python3 -m json.tool | head -20
curl -s 'localhost:9200/geonames/_mapping' | python3 -m json.tool | head -20
```

The counts should match the table at the top of this document, and each
mapping's `_meta` block should tell you what the index was built from and when
(see [Provenance](#provenance)). If `_meta` is missing, the index predates the
current loaders.

Once the restore is done, the `snapshots` mount is no longer needed. You can
delete the directory and re-create the container without the `-v …:/snapshots`
and `-e path.repo=…` flags.

---

## Path B — mount a pre-built data directory

This is the older archive format: a tar of an Elasticsearch **data directory**,
which you mount straight into the container. It works, but it is pinned to
Elasticsearch 7.10.x — see [Packaging](#packaging-an-index-to-hand-to-someone-else)
for why we no longer publish this way.

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

The directory name inside the archive is historical. Renaming it is optional
but saves the next person from assuming it only has the gazetteer in it.

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
  through. Re-download, or rebuild that one index (Path C) — the other index
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

The container in either path is a plain Elasticsearch 7.10.1 node. Nothing about
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
node at a time, always. Restoring from a snapshot (Path A) into each project's
own data directory avoids the problem entirely.

---

## Path C — build the indices yourself

Only when the pre-built index is stale, or you want a different Wikipedia
snapshot, a different gazetteer, or a changed index format.

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

**Publish a snapshot archive.** The recipe is below; the reasoning and the
measurements are after it.

### Making one

Run against the node that holds the finished indices. It can stay up — a
snapshot is taken from a running node, which is the first advantage over tarring
a data directory.

```shell
# 1. The node needs a repository path. If it wasn't started with one, re-create
#    the container adding:  -e path.repo=/snapshots -v /abs/path/snapshots:/snapshots
curl -X PUT 'localhost:9200/_snapshot/ngec' \
  -H 'Content-Type: application/json' \
  -d '{"type":"fs","settings":{"location":"/snapshots","compress":true}}'

# 2. Snapshot both indices. Skip the global state: it carries cluster settings
#    the recipient neither needs nor wants.
curl -X PUT "localhost:9200/_snapshot/ngec/ngec-$(date +%F)?wait_for_completion=true" \
  -H 'Content-Type: application/json' \
  -d '{"indices":"wiki,geonames","include_global_state":false}'

# 3. Tar the repository directory.
tar -C /abs/path -czf ngec-index-snapshot-$(date +%F).tar.gz snapshots
sha256sum ngec-index-snapshot-$(date +%F).tar.gz > ngec-index-snapshot-$(date +%F).tar.gz.sha256
```

The recipient's side is [Path A](#path-a--restore-a-snapshot-archive) above.

### Provenance

Both loaders stamp build provenance into the index mapping's `_meta`, and
**`_meta` travels inside the snapshot** — verified by restoring one into a
different Elasticsearch and reading it back. It should record:

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

### Why a snapshot rather than a tarred data directory

Both were tested on the scratch build described in
`docs/memos/2026-09-09-demo-review/es_index_verification.md`.

| | Tarred data directory | Snapshot archive |
|---|---|---|
| Node must be stopped to make it | **yes** — copying a live data directory can capture a torn index | no |
| Restores onto Elasticsearch 8 | no — 7.10 is not a direct-upgrade source for 8.x (you must pass through 7.17 first) | **yes, tested** (7.10.1 → 8.11.4, counts and `_meta` intact) |
| Restores onto Elasticsearch 9 | no | no — indices created in 7.x are outside 9.x's window; reindex first |
| Selective restore (`wiki` only) | no — it is all one directory | **yes** |
| Verifiable before shipping | only by starting a node on it | `_snapshot/_all` reports per-shard success |
| Merges into an existing node | no — it *is* the node's storage | **yes** — restores alongside indices already there |

The last two rows are the practical ones. A student who wants only the `wiki`
index for an unrelated project can restore just `wiki`, into a node that already
has their own data in it. With a data directory they have to take both indices
and give up the node.

The version story is the decisive one. The archive that has been passed around
so far is a 7.10 data directory, so every recipient is pinned to Elasticsearch
7.10.1 forever, with no upgrade path that does not involve rebuilding from
source. A snapshot taken from that same 7.10.1 node restores into 8.x today.

**Size and time are a wash.** Measured on a scratch node holding the full
13,250,817-document `geonames` index plus a small `wiki` index (2.2 GB of index
on disk):

| | Tarred data directory | Snapshot archive |
|---|---|---|
| Node downtime to produce | 83 s (1 s to stop + 82 s to tar) | **none** |
| Time to produce | 82 s | 54 s snapshot + 82 s tar = 136 s |
| Archive size | 1,515,791,693 bytes | 1,515,791,662 bytes |
| Recipient's unpack | untar | 14 s untar + 54 s restore |
| Result on disk | 2.2 GB | 2.2 GB |

The two archives came out within 31 bytes of each other. So the choice is
decided entirely by the rows above the size row — no downtime to produce,
restoring onto a newer Elasticsearch, and being able to hand over one index
rather than a whole node.

Scale those figures by about 6 for the real pair of indices (12 GB of index
rather than 2 GB).

### One trap

Never delete or move files inside a snapshot repository directory while it is
registered with a running cluster. Elasticsearch notices, disables the
repository, and every later call returns

> Could not read repository data because the contents of the repository do not
> match its expected state.

The recovery is the one the message describes and it does work: `DELETE
_snapshot/<name>`, clear the directory, then `PUT` the repository again — the
cluster rebuilds its view from the physical contents. Take the archive by
tarring the directory, not by pruning it.
