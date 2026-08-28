# Elasticsearch indices

NGEC's actor resolver and location resolver both query a single Elasticsearch
instance. It serves two indices:

| Index      | Used by                        | Built from                    |
|------------|--------------------------------|-------------------------------|
| `wiki`     | actor resolution, entity linking | an English Wikipedia XML dump |
| `geonames` | location resolution (mordecai3) | the GeoNames gazetteer        |

Most users should **not** build these. Download the pre-built index instead —
see the main [README.md](../README.md) for quick-start instructions. Building
the wiki index from scratch takes many hours and tens of GB of scratch space.

The rest of this directory is for the case where you do need to build your own.

## One data directory, two indices

The most important thing to understand: a single Elasticsearch node stores **all
of its indices in one data directory**. Both `wiki` and `geonames` live together
in whatever directory you mount at `/usr/share/elasticsearch/data`. The folder
name is historical — despite saying "geo", it contains both indices.

That directory is large (>10 GB) and is **not** committed to git.

### Where it lives

`NGEC_ES_DATA` in the repo-root `.env` sets the path; unset, it defaults to
`./elasticsearch/data/wikigeo_index`.

Set it to an **absolute** path — compose does not expand `~`, it would create a
directory literally named `~`. To find the path your current container uses:

```bash
docker inspect <container> --format '{{json .Mounts}}' | python -m json.tool
```

Every `docker compose` command below passes `--project-directory .` and is run
from the repo root. That flag is what makes compose read the repo-root `.env`
(and so `NGEC_ES_DATA`) and resolve the default correctly. Without it the
project directory is `elasticsearch/`, the `.env` is never read, and the build
stack silently comes up against an **empty** data directory with no indices in
it. If `_cat/indices` is empty after `up`, stop and check the path before
reloading anything.

If the directory is shared with another project, **never run two ES containers
against it at once** — two nodes on one data dir corrupts it.

## Scripted rebuild and publish

Two scripts automate everything below, and add the guards the manual procedure
depends on you remembering. Prefer them; the manual steps that follow are the
reference for what they do and the fallback when something needs doing by hand.

```bash
tools/rebuild_index.sh wiki        # or: geonames, both
tools/publish_index.sh             # package + upload, after you have read the report
```

`rebuild_index.sh` resolves the data directory, refuses to run if compose would
mount a different one, backs it up, stops whatever holds port 9200, brings the
build stack up, **checks that the index you are not rebuilding is actually
there**, runs the loaders, and prints a before/after report. It exits non-zero
if the result looks wrong — an emptied index, a >10% drop in documents, red
health — so a bad build announces itself instead of hiding in the numbers.

It is **resumable**. Progress is checkpointed per stage in
`.rebuild_state`, and `--resume` picks up from the last completed stage:

```bash
tools/rebuild_index.sh --resume wiki
```

Resume is safe because both loaders index with deterministic document IDs (the
page title for `wiki`, the geonameid for `geonames`), so every bulk load is an
idempotent upsert — re-running a stage rewrites documents rather than
duplicating them. What resume recovers is whole stages, not partial ones: the
dump download continues where it left off, but an interrupted `load_es`
restarts at the beginning of the dump. The result is correct either way; the
cost is time. On resume the script deliberately does **not** pass `--drop`
again, since the original run already dropped the index.

Publishing is deliberately a separate command. A wiki rebuild runs for most of
a day, and you want to read the report before the result reaches every user.
`publish_index.sh` refuses to publish an index that is empty, red, or missing
its `_meta` provenance, and always asks before uploading. Set the destination
in `NGEC_PUBLISH_DEST` (an rsync target); there is no default.

It writes three files to `elasticsearch/dist/`: the tarball, a `.sha256`, and a
`manifest.json` carrying both indices' doc counts and `_meta`. The manifest is
what lets a client answer "is my index stale?" by fetching a few hundred bytes
instead of the whole archive.

## Updating one index without dropping the other

Because both indices share one ES node, you can rebuild one while leaving the
other intact. **Do not** start from an empty data directory — that throws away
the index you're not rebuilding.

Docker runs only the datastores (Elasticsearch, plus Redis for the wiki build).
The loaders are small Python CLIs you run **on the host** with `uv`. The
procedure is identical on Linux, macOS, and Windows; the only OS-specific step
is downloading the source data, which uses `curl` and built-in Python downloads
available everywhere.

> ⚠️ **Stop your normal Elasticsearch container first.** The build stack mounts
> the *same* data dir, and two ES nodes must never run against one data
> directory at the same time — it corrupts the directory, and they collide on
> port 9200.

### One-time host setup

- Install Docker and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
- Point `NGEC_ES_DATA` at the data directory your existing Elasticsearch
  container uses, so the build stack writes to the same place. See
  [Where it lives](#where-it-lives) above.

### A note on the `es-build` dependency group

The build-time Python deps (`lxml`, `mwparserfromhell`, `plac`, `redis`) live in
the `es-build` dependency group, which is **not** installed by default — most
users download a prebuilt index and never need them.

Every `uv run` below passes `--group es-build`, which installs them on demand,
so there is no separate setup step. Don't drop that flag: `uv sync` is exact and
prunes anything outside the groups you name, so an unrelated `uv sync` for
normal work silently removes these again. Pre-fetching with `uv sync --group
es-build` is fine, it just isn't durable on its own.

### Update the GeoNames index

Run from the repo root. Takes >30 minutes for the full gazetteer.

```bash
# 0. back up, then stop your normal ES so the build stack can use the data dir
cp -r "${NGEC_ES_DATA:-elasticsearch/data/wikigeo_index}" /tmp/wikigeo_index.bak
docker stop <your-es-container>

# 1. start build ES against the shared live data dir. --project-directory . is
#    required: it is what makes compose read the repo-root .env (and therefore
#    NGEC_ES_DATA). Without it you get an empty data dir and no indices.
docker compose --project-directory . -f elasticsearch/compose-build.yml up -d es
curl -s 'localhost:9200/_cat/indices?v'        # sanity: geonames + wiki both present

# 2. download -> delete ONLY geonames -> reload
cd elasticsearch/es_geonames
uv run --group es-build python load_geonames_es.py all
curl -s 'localhost:9200/_cat/indices?v'        # geonames changed, wiki untouched
cd ../..

# 3. tear down the build stack and bring your normal ES back
docker compose --project-directory . -f elasticsearch/compose-build.yml down
docker start <your-es-container>
```

### Update the Wikipedia index

Same shape, but bring up Redis too. This is a long process (the dump is tens of
GB compressed).

```bash
# 0. back up, then stop your normal ES
cp -r "${NGEC_ES_DATA:-elasticsearch/data/wikigeo_index}" /tmp/wikigeo_index.bak
docker stop <your-es-container>

# 1. start build ES + Redis against the shared live data dir (see the note on
#    --project-directory . in the GeoNames flow above)
docker compose --project-directory . -f elasticsearch/compose-build.yml up -d
curl -s 'localhost:9200/_cat/indices?v'        # sanity: geonames + wiki both present

# 2. download the dump
cd elasticsearch/es_wiki
mkdir -p data
curl -L https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
  -o data/enwiki-latest-pages-articles.xml.bz2

# 3. build redirects, then reload ONLY the wiki index
DUMP=data/enwiki-latest-pages-articles.xml.bz2
uv run --group es-build python load_wiki_es.py build_links "$DUMP"
uv run --group es-build python load_wiki_es.py load_redis  "$DUMP"
uv run --group es-build python load_wiki_es.py load_es --drop "$DUMP"
curl -s 'localhost:9200/_cat/indices?v'        # wiki changed, geonames untouched
cd ../..

# 4. tear down the build stack and bring your normal ES back
docker compose --project-directory . -f elasticsearch/compose-build.yml down
docker start <your-es-container>
```

Per-index details (tunables, single-stage runs, env overrides):

- **Wikipedia:** [`es_wiki/README.md`](es_wiki/README.md)
- **GeoNames:** [`es_geonames/README.md`](es_geonames/README.md)

You'd rebuild an index when:

- the prebuilt one is stale,
- you want a different gazetteer or a different/-language Wikipedia, or
- you're changing the index format (requires editing the loader + mapping).

## Cluster health is yellow

```bash
curl -X GET "http://localhost:9200/_cluster/health?pretty"
```

may report `yellow`. That's usually a single-node cluster trying (and failing)
to allocate replica shards. Both mappings here set `number_of_replicas` to 0 and
the loaders set it again on the indices they create, so a freshly built index
should stay green. If you're seeing yellow on an index built elsewhere, drop
replicas to 0 (this persists):

```bash
curl -X PUT "http://localhost:9200/_settings" -H 'Content-Type: application/json' -d'
{ "index": { "number_of_replicas": 0 } }'
```
