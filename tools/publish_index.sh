#!/usr/bin/env bash
#
# Package the built Elasticsearch data directory and publish it for users.
#
# Deliberately separate from tools/rebuild_index.sh. A wiki rebuild runs for
# most of a day; welding the upload onto its end means a network failure at
# hour 10 costs you the run, and -- more importantly -- it removes the one
# moment where a human looks at the numbers before every downstream user pulls
# the result. Rebuild, read the report, then publish.
#
#   tools/publish_index.sh                 # package, then prompt before upload
#   tools/publish_index.sh --no-upload     # produce the artifacts locally only
#   tools/publish_index.sh --skip-tar      # re-use an existing tarball, upload it
#
# Produces, in elasticsearch/dist/:
#
#   wikigeo_index.tar.gz        the ES data directory, unpacking to wikigeo_index/
#   wikigeo_index.tar.gz.sha256 checksum, for users to verify
#   manifest.json               doc counts, dump dates and build dates for both
#                               indices, read from each index's mapping _meta
#
# The manifest is the point of the `_meta` stamping: it lets a client answer
# "is my index stale?" by fetching a few hundred bytes instead of 13 GB.
#
# The upload destination is NGEC_PUBLISH_DEST, an rsync target such as
# user@host:/srv/www/ngec/index/. There is no default -- publishing to the
# wrong host is not a mistake worth making convenient.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="elasticsearch/compose-build.yml"
COMPOSE=(docker compose --project-directory . -f "$COMPOSE_FILE")
RUN_LOADER=(uv run --group es-build python)
DIST_DIR="$REPO_ROOT/elasticsearch/dist"

COMPRESS="gzip"
DO_UPLOAD=1
SKIP_TAR=0
ASSUME_YES=0

if [ -t 1 ]; then
    B=$'\033[1m'; R=$'\033[31m'; Y=$'\033[33m'; G=$'\033[32m'; N=$'\033[0m'
else
    B=""; R=""; Y=""; G=""; N=""
fi
log()  { printf '%s[%s]%s %s\n' "$B" "$(date +%H:%M:%S)" "$N" "$*"; }
warn() { printf '%s[%s] WARNING:%s %s\n' "$Y" "$(date +%H:%M:%S)" "$N" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$R" "$(date +%H:%M:%S)" "$N" "$*" >&2; exit 1; }
rule() { printf '%s\n' "------------------------------------------------------------------------"; }

# Publishing is the one irreversible, outward-facing step here, so this never
# proceeds by default. On a non-terminal stdin it refuses rather than reading
# EOF as assent; unattended publishing has to be spelled out with --yes.
confirm() {
    local prompt="$1" reply
    [ "$ASSUME_YES" -eq 1 ] && return 0
    if [ ! -t 0 ]; then
        die "$prompt
       stdin is not a terminal, so this cannot be answered interactively.
       Re-run with --yes to publish unattended, or --no-upload to package only."
    fi
    printf '%s [y/N] ' "$prompt"
    read -r reply
    case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

usage() { sed -n '3,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; cat <<'EOF'

Options:
  --no-upload         package only; do not transfer anything
  --skip-tar          reuse the existing tarball in elasticsearch/dist/
  --compress MODE     gzip (default), zstd, or none. Lucene data is already
                      compressed, so heavy compression costs a lot of CPU for
                      very little size; gzip -1 is the default for that reason.
  --dest TARGET       rsync destination (overrides NGEC_PUBLISH_DEST)
  --yes               do not prompt before uploading
  -h, --help          show this message
EOF
exit 0; }

DEST="${NGEC_PUBLISH_DEST:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-upload) DO_UPLOAD=0; shift ;;
        --skip-tar)  SKIP_TAR=1; shift ;;
        --compress)  COMPRESS="${2:-}"; shift 2 ;;
        --dest)      DEST="${2:-}"; shift 2 ;;
        --yes|-y)    ASSUME_YES=1; shift ;;
        -h|--help)   usage ;;
        *)           die "unknown argument: $1 (try --help)" ;;
    esac
done

human() {
    awk -v b="$1" 'BEGIN{ s="B KB MB GB TB"; n=split(s,u," ");
        for(i=1;i<n && b>=1024;i++) b/=1024;
        printf (i==1 ? "%d %s" : "%.1f %s"), b, u[i] }'
}
file_size() { [ -e "$1" ] || { echo 0; return; }; stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0; }
sha256_of()  { if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1;
               else sha256sum "$1" | cut -d' ' -f1; fi; }

# ---------------------------------------------------------------------------
# Resolve the data directory (same rules as rebuild_index.sh)
# ---------------------------------------------------------------------------
# Shell environment wins over .env, matching compose's precedence.
_env_es_data="${NGEC_ES_DATA:-}"
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; . "$REPO_ROOT/.env"; set +a
fi
[ -n "$_env_es_data" ] && NGEC_ES_DATA="$_env_es_data"
ES_DATA="${NGEC_ES_DATA:-$REPO_ROOT/elasticsearch/data/wikigeo_index}"
case "$ES_DATA" in "~"*) die "NGEC_ES_DATA must be an absolute path (no '~')" ;; esac
[ "${ES_DATA#/}" != "$ES_DATA" ] || ES_DATA="$REPO_ROOT/$ES_DATA"
[ -d "$ES_DATA" ] || die "data directory not found: $ES_DATA"

# The leaf name becomes the tarball's top-level member, and the unpack
# instructions users follow assume exactly this name.
[ "$(basename "$ES_DATA")" = "wikigeo_index" ] || die \
    "$ES_DATA must end in /wikigeo_index -- that name becomes the tarball's
       top-level directory and users unpack it into place expecting it."

ES_PARENT="$(dirname "$ES_DATA")"
mkdir -p "$DIST_DIR"

case "$COMPRESS" in
    gzip) TARBALL="$DIST_DIR/wikigeo_index.tar.gz" ;;
    zstd) command -v zstd >/dev/null || die "zstd requested but not installed"
          TARBALL="$DIST_DIR/wikigeo_index.tar.zst" ;;
    none) TARBALL="$DIST_DIR/wikigeo_index.tar" ;;
    *)    die "--compress must be gzip, zstd, or none" ;;
esac
MANIFEST="$DIST_DIR/manifest.json"
SNAPSHOT="$DIST_DIR/.snapshot.json"

# ---------------------------------------------------------------------------
# Read the index metadata. This needs ES running, so it happens BEFORE the
# node is stopped for tarring.
# ---------------------------------------------------------------------------
STARTED_STACK=0
if ! curl -fsS "http://localhost:9200/_cluster/health" >/dev/null 2>&1; then
    log "Starting Elasticsearch briefly to read index metadata"
    "${COMPOSE[@]}" up -d es
    STARTED_STACK=1
    for i in $(seq 1 60); do
        curl -fsS "http://localhost:9200/_cluster/health" >/dev/null 2>&1 && break
        [ "$i" -eq 60 ] && die "Elasticsearch did not come up. Check: ${COMPOSE[*]} logs es"
        sleep 2
    done
else
    log "Using the Elasticsearch already listening on 9200"
fi

log "Reading index metadata"
NGEC_ES_DATA="$ES_DATA" "${RUN_LOADER[@]}" tools/index_stats.py \
    --no-redirect-count -o "$SNAPSHOT"
rule
"${RUN_LOADER[@]}" tools/index_stats.py --render "$SNAPSHOT"
rule

# Refuse to publish something obviously broken. This is the last gate before
# an index reaches every user.
"${RUN_LOADER[@]}" - "$SNAPSHOT" <<'PYEOF'
import json, sys
snap = json.load(open(sys.argv[1]))
problems = []
for name in ("wiki", "geonames"):
    d = snap["indices"].get(name) or {}
    if not d.get("exists"):
        problems.append(f"{name}: index is missing")
        continue
    if not d.get("doc_count"):
        problems.append(f"{name}: index is empty")
    if d.get("health") == "red":
        problems.append(f"{name}: health is red")
    if not (d.get("meta") or {}):
        problems.append(f"{name}: no _meta -- provenance would be unknown to users")
if problems:
    print("Refusing to publish:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Stop ES before tarring.
#
# A live Lucene data directory is not a consistent thing to copy: segments are
# being written and merged, and an archive taken mid-flight can restore into a
# corrupt index. This is the same constraint that makes ES snapshots
# attractive; see the note in elasticsearch/README.md.
# ---------------------------------------------------------------------------
RUNNING_ES="$(docker ps --filter publish=9200 --format '{{.Names}}' | head -1 || true)"
if [ -n "$RUNNING_ES" ]; then
    log "Stopping '$RUNNING_ES' so the data directory is quiescent for archiving"
    docker stop "$RUNNING_ES" >/dev/null
fi

if [ "$SKIP_TAR" -eq 1 ] && [ -f "$TARBALL" ]; then
    log "Reusing existing $TARBALL ($(human "$(file_size "$TARBALL")"))"
else
    log "Archiving $ES_DATA -> $TARBALL"
    log "(this is ~13 GB of already-compressed Lucene data; expect it to be slow"
    log " and to shrink very little)"
    rm -f "$TARBALL"
    case "$COMPRESS" in
        # -1: Lucene data barely compresses, so higher levels burn CPU for
        # almost no size reduction.
        gzip) tar -C "$ES_PARENT" -cf - wikigeo_index | gzip -1 > "$TARBALL" ;;
        zstd) tar -C "$ES_PARENT" -cf - wikigeo_index | zstd -1 -T0 -o "$TARBALL" -f ;;
        none) tar -C "$ES_PARENT" -cf "$TARBALL" wikigeo_index ;;
    esac
    log "Archive complete: $(human "$(file_size "$TARBALL")")"
fi

log "Computing checksum"
CHECKSUM="$(sha256_of "$TARBALL")"
printf '%s  %s\n' "$CHECKSUM" "$(basename "$TARBALL")" > "$TARBALL.sha256"

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
log "Writing $MANIFEST"
"${RUN_LOADER[@]}" - "$SNAPSHOT" "$MANIFEST" "$TARBALL" "$CHECKSUM" <<'PYEOF'
import datetime, json, os, sys

snapshot_path, manifest_path, tarball, checksum = sys.argv[1:5]
snap = json.load(open(snapshot_path))

manifest = {
    "published_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "archive": {
        "filename": os.path.basename(tarball),
        "size_bytes": os.path.getsize(tarball),
        "sha256": checksum,
        "unpacks_to": "wikigeo_index/",
        "note": "Elasticsearch 7.10.1 data directory. Mount at "
                "/usr/share/elasticsearch/data.",
    },
    "indices": {},
}
for name in ("wiki", "geonames"):
    d = snap["indices"].get(name) or {}
    manifest["indices"][name] = {
        "doc_count": d.get("doc_count"),
        "store_bytes": d.get("store_bytes"),
        # _meta is what the loaders stamped at build time: dump file, dump
        # date, build date, builder. It is the authoritative staleness answer.
        "meta": d.get("meta") or {},
    }

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PYEOF

# ---------------------------------------------------------------------------
# Restore whatever ES was running before
# ---------------------------------------------------------------------------
if [ "$STARTED_STACK" -eq 1 ]; then
    "${COMPOSE[@]}" down >/dev/null 2>&1 || true
fi
if [ -n "$RUNNING_ES" ] && [ "$RUNNING_ES" != "ngec-build-es" ]; then
    log "Restarting '$RUNNING_ES'"
    docker start "$RUNNING_ES" >/dev/null || warn "could not restart $RUNNING_ES"
fi

rule
printf '%sArtifacts ready in %s%s\n' "$G" "$DIST_DIR" "$N"
printf '  %-32s %s\n' "$(basename "$TARBALL")" "$(human "$(file_size "$TARBALL")")"
printf '  %-32s %s\n' "$(basename "$TARBALL").sha256" "$CHECKSUM"
printf '  %-32s\n' "manifest.json"
rule

if [ "$DO_UPLOAD" -eq 0 ]; then
    log "--no-upload given; stopping here."
    exit 0
fi

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
[ -n "$DEST" ] || die "no upload destination. Set NGEC_PUBLISH_DEST (an rsync
       target such as user@host:/srv/www/ngec/index/) or pass --dest, or
       rerun with --no-upload."

command -v rsync >/dev/null || die "rsync not found on PATH"

printf '\n%sAbout to publish to:%s %s\n' "$B" "$N" "$DEST"
printf 'This replaces the index every downstream user downloads.\n'
printf '  wiki     : %s docs\n' "$(grep -o '"doc_count": [0-9]*' "$MANIFEST" | head -1 | tr -d '"' | awk '{print $2}')"
printf '  archive  : %s (%s)\n' "$(human "$(file_size "$TARBALL")")" "$CHECKSUM"
confirm "Upload now?" || die "not uploaded. Artifacts remain in $DIST_DIR"

# --partial --append-verify so an interrupted 13 GB transfer resumes instead of
# restarting. The manifest and checksum go LAST: until they land, a client that
# reads the manifest still sees the previous, complete release.
log "Uploading archive (resumable; rerun this script to continue if interrupted)"
rsync -avh --progress --partial --append-verify "$TARBALL" "$DEST"
log "Uploading checksum and manifest"
rsync -avh "$TARBALL.sha256" "$MANIFEST" "$DEST"

rule
printf '%sPublished.%s\n' "$G" "$N"
printf 'Users can verify with:\n'
printf '  shasum -a 256 -c %s\n' "$(basename "$TARBALL").sha256"
rule
