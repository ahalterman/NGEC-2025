# Building / updating the GeoNames index

Builds the `geonames` Elasticsearch index used by location resolution
(mordecai3), from the [GeoNames](http://www.geonames.org/) gazetteer. Based on
[es-geonames](https://github.com/openeventdata/es-geonames).

The whole flow is one cross-platform Python tool
([`load_geonames_es.py`](load_geonames_es.py)) plus an Elasticsearch container.
It runs identically on Linux, macOS, and Windows.

> **This updates only the `geonames` index.** It runs against the shared live
> data dir (which also holds the `wiki` index) and deletes/recreates *only*
> `geonames`, so the `wiki` index is left intact. See
> [../README.md](../README.md) for the why.

## Prerequisites

The build-time Python deps (`lxml`, `mwparserfromhell`, `plac`, `redis`) live in
the `es-build` dependency group, which is **not** installed by default — most
users download a prebuilt index and never need them.

Every `uv run` below passes `--group es-build`, which installs them on demand,
so there is no separate setup step. Don't drop that flag: `uv sync` is exact and
prunes anything outside the groups you name, so an unrelated `uv sync` for
normal work silently removes these again. Pre-fetching with `uv sync --group
es-build` is fine, it just isn't durable on its own.

## Steps

Takes >30 minutes for the full gazetteer.

1. **Back up** the live data dir, then **stop your normal Elasticsearch
   container** — the build stack mounts the same data dir, so the two ES nodes
   must never run at once (they'd corrupt the data and collide on port 9200):

   ```bash
   cp -r "${NGEC_ES_DATA:-elasticsearch/data/wikigeo_index}" /tmp/wikigeo_index.bak
   docker stop <your-es-container>
   ```

2. **Start Elasticsearch** against the shared live data dir:

   ```bash
   docker compose --project-directory . -f elasticsearch/compose-build.yml up -d es
   ```

   The path comes from `NGEC_ES_DATA` (repo-root `.env`), defaulting to
   `./elasticsearch/data/wikigeo_index`. `--project-directory .` is what makes
   compose read that `.env`; without it you get an empty data dir and no
   indices. See [../README.md](../README.md).

   Wait until it's up (`curl -s localhost:9200/_cluster/health`). Confirm both
   indices are present and you're not about to lose wiki:

   ```bash
   curl -s 'localhost:9200/_cat/indices?v'
   ```

3. **Run the loader:**

   ```bash
   cd elasticsearch/es_geonames
   uv run --group es-build python load_geonames_es.py all
   ```

   `all` runs three stages in order:

   - `download` — fetch `allCountries.zip`, `admin1CodesASCII.txt`,
     `admin2Codes.txt` into `./geonames_data/` and unzip.
   - `recreate` — delete **only** the `geonames` index and recreate its mapping
     (also drops replicas to 0 and relaxes disk watermarks).
   - `load` — bulk-load the gazetteer into the `geonames` index.

   You can run a single stage instead, e.g. `... load_geonames_es.py load`, or
   point at a different download folder with `--data-dir`.

4. **Verify** the `geonames` count changed and `wiki` is unchanged:

   ```bash
   curl -s 'localhost:9200/_cat/indices?v'
   ```

5. **Stop** the build stack and **bring your normal ES back**. The updated data
   is already in the shared data dir:

   ```bash
   cd ../..                                    # back to the repo root
   docker compose --project-directory . -f elasticsearch/compose-build.yml down
   docker start <your-es-container>
   ```

## Notes

- The loader talks to `http://localhost:9200` by default; override with the
  `NGEC_ES_URL` environment variable.
- Unrecognised two-letter ISO country codes are collected during the load and
  written to `geonames_data/bad_iso_codes.txt` rather than printed. A short list
  there is normal.
- Windows: run the same `uv run --group es-build python ...` commands in PowerShell or a terminal
  of your choice. Downloads use Python's built-in HTTP client (no `wget`), and
  unzip uses Python's `zipfile`, so there are no shell-line-ending or
  missing-tool issues.

## Index metadata

A run that includes `load` stamps build provenance onto the index when it
finishes, in the mapping's `_meta`. Elasticsearch stores it verbatim and never
interprets it, so it travels with the index — including into a copied data
directory:

```bash
curl -s 'localhost:9200/geonames/_mapping' | python -m json.tool
```

```json
"_meta": {
  "source": "https://download.geonames.org/export/dump",
  "gazetteer_file": "allCountries.txt",
  "dump_date": "2026-08-12",
  "build_date": "2026-08-12",
  "doc_count": 12571784,
  "builder": "NGEC elasticsearch/es_geonames/load_geonames_es.py"
}
```

`dump_date` is the modification time of `allCountries.txt` — in practice, when
`download` fetched it. The gazetteer carries no version stamp of its own.

`recreate` on its own rebuilds the mapping, which clears `_meta` and leaves the
index empty; the stamp is written at the end of a run that actually loads data.

## Files

- `load_geonames_es.py` — the download/recreate/load tool.
- `geonames_mapping.json` — field mappings for the `geonames` index.
