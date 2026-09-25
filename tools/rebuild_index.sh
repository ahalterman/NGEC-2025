#!/usr/bin/env bash
#
# Rebuild the offline `wiki` and/or `geonames` Elasticsearch indices in place,
# against whatever data directory this checkout is configured to use.
#
# This automates the procedure in elasticsearch/README.md and adds the guards
# that procedure relies on you remembering: that the build stack must mount the
# LIVE data dir, that starting it against the wrong path silently produces an
# empty cluster, and that a rebuild which ends with fewer documents than it
# started with must not be published.
#
#   tools/rebuild_index.sh wiki
#   tools/rebuild_index.sh geonames
#   tools/rebuild_index.sh both
#   tools/rebuild_index.sh --resume wiki      # continue an interrupted run
#
# RESUMABILITY
#
# The run is checkpointed per stage in elasticsearch/.rebuild_state. --resume
# skips stages already recorded as done and restarts the one that was in
# flight. This is safe because both loaders index with deterministic document
# IDs (`_id` is the wiki page title / the geonameid), so every bulk load is an
# idempotent upsert: re-running a stage rewrites the same documents rather than
# duplicating them. Resume costs time, never correctness.
#
# What resume does and does not recover:
#
#   download     fully resumable -- curl -C - continues a partial file.
#   decompress   optional, and skipped unless it pays. A multistream dump can
#                be cut at its stream offsets and decompressed by every core at
#                once, so the two passes below read plain XML instead of
#                spending ~45 minutes each on single-threaded bz2. It costs
#                ~117 GB of disk, is skipped when the disk is too full, and any
#                failure falls back to reading the .bz2.
#   build_links  restarts from the beginning of the dump. The loader
#                checkpoints its pickle periodically, but only for inspection;
#                it has no resume offset, so this stage is re-done in full.
#   load_redis   cheap, always re-run.
#   load_es      restarts from the beginning of the dump. The documents already
#                indexed are simply rewritten. This is the expensive case: an
#                interruption at hour 9 of 10 costs ~9 hours of redundant
#                parsing, though the result is correct.
#
# So resume saves you whole stages, not partial ones. Recovering the tail of an
# interrupted load_es would need the loader to checkpoint its page offset; see
# the note at the end of this file.
#
# CRITICALLY: on resume, `--drop` is NOT passed to load_es again. The original
# run already dropped the index; dropping a second time would throw away the
# partial progress that makes resuming worthwhile.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

STATE_FILE="$REPO_ROOT/elasticsearch/.rebuild_state"
COMPOSE_FILE="elasticsearch/compose-build.yml"
COMPOSE=(docker compose --project-directory . -f "$COMPOSE_FILE")
RUN_LOADER=(uv run --group es-build python)

WIKI_DUMP_DIR="elasticsearch/es_wiki/data"
DUMP_URL_HOST="https://dumps.wikimedia.org/enwiki"
DUMP_DATE=""
DUMP_INDEX=""            # multistream stream-offset index, beside the dump
DUMP_XML=""              # the dump decompressed, when we take that path

# The parallel decompress writes the dump out as plain XML -- about 117 GB for
# enwiki, plus the part files it has not yet concatenated. Below this much free
# space the up-front decompress is skipped and the loader reads the .bz2, which
# is slower but needs no extra disk.
PARALLEL_BUNZIP=1
PARALLEL_BUNZIP_MIN_GB=130

TARGET=""
RESUME=0
DO_BACKUP=1
SKIP_DOWNLOAD=0
ASSUME_YES=0
DUMP=""

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    B=$'\033[1m'; R=$'\033[31m'; Y=$'\033[33m'; G=$'\033[32m'; N=$'\033[0m'
else
    B=""; R=""; Y=""; G=""; N=""
fi

log()  { printf '%s[%s]%s %s\n' "$B" "$(date +%H:%M:%S)" "$N" "$*"; }
warn() { printf '%s[%s] WARNING:%s %s\n' "$Y" "$(date +%H:%M:%S)" "$N" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$R" "$(date +%H:%M:%S)" "$N" "$*" >&2; exit 1; }
rule() { printf '%s\n' "------------------------------------------------------------------------"; }

# Ask a yes/no question. With --yes, or when stdin is not a terminal, do not
# block: a scheduled run has nobody to answer, and a `read` on a closed stdin
# either hangs forever or silently reads EOF. Requiring --yes makes the
# unattended case explicit rather than accidental.
confirm() {
    local prompt="$1" reply
    [ "$ASSUME_YES" -eq 1 ] && return 0
    if [ ! -t 0 ]; then
        die "$prompt
       stdin is not a terminal, so this cannot be answered interactively.
       Re-run with --yes if you intend this to proceed unattended."
    fi
    printf '%s [y/N] ' "$prompt"
    read -r reply
    case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

usage() {
    sed -n '3,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --resume            continue an interrupted run, skipping completed stages
  --dump PATH         use this dump file instead of downloading one
  --dump-date DATE    build from the dated Wikipedia dump for YYYYMMDD
                      (default: the newest dated dump whose articles job is
                      complete). Dated dumps are used rather than "latest" so
                      the published index records which snapshot it came from.
  --skip-download     use the dump / gazetteer already on disk
  --no-parallel-bunzip
                      do not decompress the dump up front. Saves ~117 GB of
                      disk and costs ~90 minutes, because the loader then
                      decompresses the .bz2 again on each of its two passes.
  --no-backup         skip the pre-run copy of the data dir (it is ~13 GB; the
                      copy is your only undo, so skip it only deliberately)
  --yes               do not prompt for confirmation
  -h, --help          show this message
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Portable helpers (macOS ships BSD stat and bash 3.2; no bash 4 features here)
# ---------------------------------------------------------------------------
file_size() {
    [ -e "$1" ] || { echo 0; return; }
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0
}
file_mtime() {
    [ -e "$1" ] || { echo 0; return; }
    stat -f%m "$1" 2>/dev/null || stat -c%Y "$1" 2>/dev/null || echo 0
}
human() {
    awk -v b="$1" 'BEGIN{ s="B KB MB GB TB"; n=split(s,u," ");
        for(i=1;i<n && b>=1024;i++) b/=1024;
        printf (i==1 ? "%d %s" : "%.1f %s"), b, u[i] }'
}
# Free GB on the filesystem holding $1, walking up to the nearest directory that
# exists (the dump dir is created later). -P forces the one-line POSIX format,
# which is the only thing BSD and GNU df agree on.
free_gb() {
    local d="$1"
    while [ ! -d "$d" ] && [ "$d" != "/" ]; do d="$(dirname "$d")"; done
    df -Pk "$d" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'
}

# State file is flat key=value so it can be read and repaired by hand.
state_get() {
    [ -f "$STATE_FILE" ] || return 0
    grep "^$1=" "$STATE_FILE" 2>/dev/null | tail -1 | cut -d= -f2- || true
}
state_set() {
    local key="$1" val="$2" tmp
    mkdir -p "$(dirname "$STATE_FILE")"
    touch "$STATE_FILE"
    tmp="$STATE_FILE.tmp.$$"
    grep -v "^$key=" "$STATE_FILE" > "$tmp" 2>/dev/null || true
    printf '%s=%s\n' "$key" "$val" >> "$tmp"
    mv "$tmp" "$STATE_FILE"
}
# Stage keys are namespaced per target. `both` runs two flows that each have a
# stage called "download"; without the prefix, geonames finishing its gazetteer
# download marks "download" done and the wiki flow then skips fetching the dump.
STAGE_PREFIX=""
stage_done()    { [ "$(state_get "stage_$STAGE_PREFIX$1")" = "done" ]; }
mark_running()  { state_set "stage_$STAGE_PREFIX$1" "running"; state_set "stage_$STAGE_PREFIX$1_started" "$(date +%s)"; }
mark_done()     { state_set "stage_$STAGE_PREFIX$1" "done";    state_set "stage_$STAGE_PREFIX$1_finished" "$(date +%s)"; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        wiki|geonames|both) TARGET="$1"; shift ;;
        --resume)        RESUME=1; shift ;;
        --dump)          DUMP="${2:-}"; [ -n "$DUMP" ] || die "--dump needs a path"; shift 2 ;;
        --dump-date)     DUMP_DATE="${2:-}"; [ -n "$DUMP_DATE" ] || die "--dump-date needs YYYYMMDD"
                         case "$DUMP_DATE" in [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) ;;
                             *) die "--dump-date must be YYYYMMDD, got '$DUMP_DATE'" ;; esac
                         shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --no-parallel-bunzip) PARALLEL_BUNZIP=0; shift ;;
        --no-backup)     DO_BACKUP=0; shift ;;
        --yes|-y)        ASSUME_YES=1; shift ;;
        -h|--help)       usage ;;
        *)               die "unknown argument: $1 (try --help)" ;;
    esac
done
[ -n "$TARGET" ] || die "say what to rebuild: wiki, geonames, or both (try --help)"

# ---------------------------------------------------------------------------
# Resolve the data directory
#
# This is the step the whole procedure hinges on. NGEC_ES_DATA lives in the
# repo-root .env, which docker compose only reads because every invocation here
# passes --project-directory . -- but this script also needs the value itself
# (to back it up, and to check it is not empty), so it sources .env directly.
# ---------------------------------------------------------------------------
# An NGEC_ES_DATA already exported in the shell wins over the one in .env,
# because that is compose's own precedence -- if this script disagreed with
# compose about which directory is live, it would back up one and rebuild the
# other.
_env_es_data="${NGEC_ES_DATA:-}"
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
fi
[ -n "$_env_es_data" ] && NGEC_ES_DATA="$_env_es_data"
ES_DATA="${NGEC_ES_DATA:-$REPO_ROOT/elasticsearch/data/wikigeo_index}"

case "$ES_DATA" in
    "~"*) die "NGEC_ES_DATA is '$ES_DATA'. Use an absolute path: docker compose does
       not expand '~', it would create a directory literally named '~'." ;;
esac
[ "${ES_DATA#/}" != "$ES_DATA" ] || ES_DATA="$REPO_ROOT/$ES_DATA"
[ "$(basename "$ES_DATA")" = "wikigeo_index" ] || warn \
    "data dir does not end in /wikigeo_index ($ES_DATA). Publishing expects that leaf name."

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v docker >/dev/null || die "docker not found on PATH"
command -v uv     >/dev/null || die "uv not found on PATH -- https://docs.astral.sh/uv/"
docker info >/dev/null 2>&1  || die "docker daemon is not running"
[ -f "$COMPOSE_FILE" ] || die "$COMPOSE_FILE not found (run this from anywhere; it locates the repo itself)"

# Confirm compose resolves to the same directory this script is about to back
# up. A mismatch means the .env was not picked up and the build would run
# against an empty directory -- the exact silent failure this guards.
COMPOSE_MOUNT="$("${COMPOSE[@]}" config 2>/dev/null \
    | awk '/source:/ {print $2; exit}')" || true
if [ -n "$COMPOSE_MOUNT" ] && [ "$COMPOSE_MOUNT" != "$ES_DATA" ]; then
    die "compose would mount a different directory than this script resolved:
       script  : $ES_DATA
       compose : $COMPOSE_MOUNT
       These must match. Check NGEC_ES_DATA in $REPO_ROOT/.env"
fi

# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------
RUN_ID="$(date +%Y%m%dT%H%M%S)"
SNAP_DIR="$REPO_ROOT/elasticsearch/.rebuild_snapshots"
mkdir -p "$SNAP_DIR"

if [ "$RESUME" -eq 1 ]; then
    [ -f "$STATE_FILE" ] || die "--resume given but no state file at $STATE_FILE"
    prev_target="$(state_get target)"
    prev_data="$(state_get es_data)"
    [ "$prev_target" = "$TARGET" ] || die \
        "state file is for target '$prev_target', you asked for '$TARGET'.
       Finish or delete that run first: rm $STATE_FILE"
    [ "$prev_data" = "$ES_DATA" ] || die \
        "the data directory changed since that run started:
       then: $prev_data
       now : $ES_DATA
       Resuming across a different index would mix two builds. Refusing."
    RUN_ID="$(state_get run_id)"
    BEFORE_SNAP="$(state_get before_snapshot)"
    log "Resuming run $RUN_ID"
    [ -n "$DUMP" ] || DUMP="$(state_get dump)"
else
    if [ -f "$STATE_FILE" ] && [ "$(state_get completed)" != "yes" ]; then
        warn "an unfinished run is recorded in $STATE_FILE:"
        sed 's/^/         /' "$STATE_FILE" >&2
        warn "starting fresh will redo its completed stages. Use --resume to continue it."
        confirm "Start a fresh run anyway?" || die "aborted -- rerun with --resume"
    fi
    # State is deliberately NOT written yet -- see "Commit to the run" below.
    # An aborted plan must leave no state file, or the next invocation reports
    # a phantom unfinished run.
    BEFORE_SNAP="$SNAP_DIR/$RUN_ID-before.json"
fi
AFTER_SNAP="$SNAP_DIR/$RUN_ID-after.json"

# ---------------------------------------------------------------------------
# Locate the dump (wiki only)
# ---------------------------------------------------------------------------
# Wikipedia keeps ~5 dated snapshots plus an undated "latest". Only a dated one
# can tell you afterwards which snapshot an index was built from, so resolve to
# the newest dated dump whose articles job actually finished -- the current
# month's directory exists while it is still being generated.
#
# The job to ask about is the one that produces the file we download, and those
# are not the same job. `articlesdump` writes the numbered part files;
# `articlesmultistreamdumprecombine` writes the single recombined multistream
# dump this script fetches, and finishes later. Asking the first one leaves a
# window where the status says done and the file does not exist yet.
resolve_dump_date() {
    local listing dates d status
    listing="$(curl -fsS --max-time 60 "$DUMP_URL_HOST/" 2>/dev/null)" \
        || die "could not list $DUMP_URL_HOST/ -- pass --dump-date or --dump"
    dates="$(printf '%s' "$listing" | grep -oE '2[0-9]{7}/' | tr -d '/' | sort -ru)"
    [ -n "$dates" ] || die "no dated dumps found at $DUMP_URL_HOST/"
    for d in $dates; do
        # dumpstatus.json is one JSON blob: strip whitespace, drop everything
        # before that job, then take the job's first "status".
        status="$(curl -fsS --max-time 60 "$DUMP_URL_HOST/$d/dumpstatus.json" 2>/dev/null \
            | tr -d ' \n' | sed 's/.*"articlesmultistreamdumprecombine"//' \
            | grep -o '"status":"[a-z]*"' | head -1 | cut -d'"' -f4)"
        if [ "$status" = "done" ]; then printf '%s' "$d"; return 0; fi
        log "skipping $d (multistream articles dump status: ${status:-unknown})" >&2
    done
    die "no dated dump has a completed articles job -- pass --dump-date explicitly"
}

if [ "$TARGET" != "geonames" ]; then
    if [ -n "$DUMP" ]; then
        case "$DUMP" in /*) ;; *) DUMP="$REPO_ROOT/$DUMP" ;; esac
    else
        if [ -z "$DUMP_DATE" ]; then
            [ "$RESUME" -eq 1 ] && DUMP_DATE="$(state_get dump_date)"
            if [ -z "$DUMP_DATE" ]; then
                log "Resolving newest completed Wikipedia dump..."
                DUMP_DATE="$(resolve_dump_date)"
                log "Using dump date $DUMP_DATE"
            fi
        fi
        DUMP="$REPO_ROOT/$WIKI_DUMP_DIR/enwiki-$DUMP_DATE-pages-articles-multistream.xml.bz2"
    fi
    # Multistream is the same articles, compressed as many small independent
    # streams instead of one big one, and Python's bz2 reads either without
    # noticing. It costs ~1 GB more to download and is the only form that can be
    # decompressed in parallel, so it is what we fetch; whether we then use that
    # property is decided per run, below.
    #
    # The stream-offset index sits beside such a dump under a matching name. A
    # dump that is not multistream has none, and the parallel path is then
    # simply unavailable.
    case "$DUMP" in
        *-multistream.xml.bz2) DUMP_INDEX="${DUMP%-multistream.xml.bz2}-multistream-index.txt.bz2" ;;
        *)                     DUMP_INDEX="" ;;
    esac
    # Keep the enwiki-YYYYMMDD- prefix: load_wiki_es.py reads the dump date off
    # this filename to stamp the index's _meta, and a generic name silently
    # stamps a null date.
    DUMP_XML="${DUMP%.bz2}"
fi

# What the plan will do about decompression. Same conditions as the stage
# itself, so the plan cannot promise something the run then skips.
decompress_plan() {
    local free
    [ "$PARALLEL_BUNZIP" -eq 1 ] || { echo "no (--no-parallel-bunzip)"; return; }
    [ -n "$DUMP_INDEX" ]         || { echo "no (dump is not multistream)"; return; }
    free="$(free_gb "$REPO_ROOT/$WIKI_DUMP_DIR")"
    if [ "$free" -lt "$PARALLEL_BUNZIP_MIN_GB" ]; then
        echo "no (${free} GB free, needs ~${PARALLEL_BUNZIP_MIN_GB} GB)"
    else
        echo "up front, in parallel (~117 GB, ${free} GB free)"
    fi
}

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
rule
printf '%sRebuild plan%s\n' "$B" "$N"
rule
printf '  target        : %s\n' "$TARGET"
printf '  run id        : %s%s\n' "$RUN_ID" "$([ "$RESUME" -eq 1 ] && echo '  (resuming)')"
printf '  data dir      : %s\n' "$ES_DATA"
if [ -d "$ES_DATA" ]; then
    printf '  current size  : %s\n' "$(du -sh "$ES_DATA" 2>/dev/null | cut -f1)"
else
    printf '  current size  : %s(does not exist yet -- this will be a first build)%s\n' "$Y" "$N"
fi
if [ "$TARGET" != "geonames" ]; then
    printf '  dump          : %s\n' "$DUMP"
    printf '  decompress    : %s\n' "$(decompress_plan)"
fi
printf '  backup        : %s\n' "$([ "$DO_BACKUP" -eq 1 ] && echo yes || echo 'NO (--no-backup)')"
if [ "$RESUME" -eq 1 ]; then
    printf '  stages done   : %s\n' \
        "$(grep -E '^stage_[a-z_]+=done$' "$STATE_FILE" 2>/dev/null \
           | sed 's/^stage_//; s/=done$//' | tr '\n' ' ')"
fi
rule

confirm "This stops any Elasticsearch on port 9200 and rewrites the index in place.
Proceed?" || die "aborted"

# ---------------------------------------------------------------------------
# Commit to the run
#
# Everything above this line is read-only: resolving paths, printing the plan,
# asking. The state file is created here, at the point where the run actually
# becomes real, so an aborted plan leaves nothing behind for --resume to trip
# over.
# ---------------------------------------------------------------------------
if [ "$RESUME" -eq 0 ]; then
    rm -f "$STATE_FILE"
    state_set run_id "$RUN_ID"
    state_set target "$TARGET"
    state_set es_data "$ES_DATA"
    state_set before_snapshot "$BEFORE_SNAP"
    state_set completed "no"
fi
if [ "$TARGET" != "geonames" ]; then
    state_set dump "$DUMP"
    [ -n "$DUMP_DATE" ] && state_set dump_date "$DUMP_DATE"
fi

# ---------------------------------------------------------------------------
# Stop whatever currently holds port 9200, remembering it so we can restore it
# ---------------------------------------------------------------------------
PREV_ES_CONTAINER="$(state_get prev_container)"
if [ -z "$PREV_ES_CONTAINER" ]; then
    PREV_ES_CONTAINER="$(docker ps --filter publish=9200 --format '{{.Names}}' \
        | grep -v '^ngec-build-es$' | head -1 || true)"
    # Publishing 9200 is not the only way to hold the index. A container on an
    # internal network mounts the same directory without publishing anything,
    # so the port filter misses it, the build stack starts anyway, and two ES
    # nodes write one data dir -- which corrupts it. Match on the mount too.
    if [ -z "$PREV_ES_CONTAINER" ]; then
        # The `if` matters: a bare `grep -qx ... && echo` leaves the loop's exit
        # status non-zero whenever the LAST container doesn't match, and under
        # `set -e` with `pipefail` that aborts the whole script. The trailing
        # `|| true` covers `head` closing the pipe early (SIGPIPE upstream).
        PREV_ES_CONTAINER="$(docker ps --format '{{.Names}}' \
            | grep -v '^ngec-build-es$' \
            | while read -r c; do
                  if docker inspect "$c" \
                          --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' 2>/dev/null \
                          | grep -qx "$ES_DATA"; then
                      echo "$c"
                  fi
              done | head -1 || true)"
    fi
    [ -n "$PREV_ES_CONTAINER" ] && state_set prev_container "$PREV_ES_CONTAINER"
fi
if [ -n "$PREV_ES_CONTAINER" ]; then
    log "Stopping '$PREV_ES_CONTAINER' (it holds port 9200; two ES nodes on one data dir corrupt it)"
    docker stop "$PREV_ES_CONTAINER" >/dev/null
fi

# ---------------------------------------------------------------------------
# Back up. Done after stopping ES so the copy is of a quiesced directory.
# ---------------------------------------------------------------------------
BACKUP_PATH="$(state_get backup_path)"
if [ "$DO_BACKUP" -eq 1 ] && [ -d "$ES_DATA" ] && [ -z "$BACKUP_PATH" ]; then
    BACKUP_PATH="${ES_DATA%/}.bak-$RUN_ID"
    log "Backing up $ES_DATA -> $BACKUP_PATH (this is your only undo)"
    cp -R "$ES_DATA" "$BACKUP_PATH"
    state_set backup_path "$BACKUP_PATH"
    log "Backup complete: $(du -sh "$BACKUP_PATH" 2>/dev/null | cut -f1)"
elif [ -n "$BACKUP_PATH" ]; then
    log "Reusing backup from this run: $BACKUP_PATH"
fi

# ---------------------------------------------------------------------------
# Bring the build stack up
# ---------------------------------------------------------------------------
if [ "$TARGET" = "geonames" ]; then
    log "Starting build Elasticsearch"
    "${COMPOSE[@]}" up -d es
else
    log "Starting build Elasticsearch + Redis"
    "${COMPOSE[@]}" up -d
fi

log "Waiting for Elasticsearch to accept connections..."
for i in $(seq 1 60); do
    if curl -fsS "http://localhost:9200/_cluster/health" >/dev/null 2>&1; then
        log "Elasticsearch is up"
        break
    fi
    [ "$i" -eq 60 ] && die "Elasticsearch did not come up within 120s. Check: ${COMPOSE[*]} logs es"
    sleep 2
done

# ---------------------------------------------------------------------------
# Guard: is this the live data directory, or an empty one?
#
# The failure this catches: a misresolved path brings ES up on an empty
# directory. Every check below still passes, the load runs for hours, and the
# real index is never touched.
# ---------------------------------------------------------------------------
INDICES_PRESENT="$(curl -fsS 'http://localhost:9200/_cat/indices?h=index' 2>/dev/null | tr -d ' ' | sort | tr '\n' ' ')"
log "Indices in this data directory: ${INDICES_PRESENT:-(none)}"

if [ -z "$INDICES_PRESENT" ]; then
    warn "This data directory contains NO indices."
    warn "If you expected an existing index here, the path is wrong -- stop now."
    warn "Resolved path: $ES_DATA"
    confirm "Continue as a first-time build?" || die "aborted -- check NGEC_ES_DATA in .env"
else
    # Rebuilding one index must not disturb the other, so the other must be here.
    case "$TARGET" in
        wiki)     echo "$INDICES_PRESENT" | grep -q geonames || warn \
                      "geonames index is absent -- expected alongside wiki. Continuing, but verify the path." ;;
        geonames) echo "$INDICES_PRESENT" | grep -q wiki || warn \
                      "wiki index is absent -- expected alongside geonames. Continuing, but verify the path." ;;
    esac
fi

# A previous run killed hard (SIGKILL, power loss) can leave the index with
# refresh disabled, which makes doc counts read as stale forever after.
for idx in wiki geonames; do
    ri="$(curl -fsS "http://localhost:9200/$idx/_settings" 2>/dev/null \
        | grep -o '"refresh_interval":"[^"]*"' | cut -d'"' -f4 || true)"
    if [ "$ri" = "-1" ]; then
        warn "$idx has refresh_interval=-1 from an interrupted run; restoring to 1s"
        curl -fsS -X PUT "http://localhost:9200/$idx/_settings" \
            -H 'Content-Type: application/json' \
            -d '{"index":{"refresh_interval":"1s"}}' >/dev/null || true
    fi
done

# ---------------------------------------------------------------------------
# BEFORE snapshot
# ---------------------------------------------------------------------------
if [ ! -f "$BEFORE_SNAP" ]; then
    log "Capturing BEFORE state -> $BEFORE_SNAP"
    NGEC_ES_DATA="$ES_DATA" "${RUN_LOADER[@]}" tools/index_stats.py -o "$BEFORE_SNAP"
    rule
    "${RUN_LOADER[@]}" tools/index_stats.py --render "$BEFORE_SNAP"
    rule
else
    log "Reusing BEFORE snapshot from this run: $BEFORE_SNAP"
fi

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
run_geonames() {
    STAGE_PREFIX="geonames_"
    cd "$REPO_ROOT/elasticsearch/es_geonames"
    if stage_done download || [ "$SKIP_DOWNLOAD" -eq 1 ]; then
        log "geonames: skipping download"
    else
        log "geonames: downloading gazetteer"
        mark_running download
        "${RUN_LOADER[@]}" load_geonames_es.py download
        mark_done download
    fi
    # `reload` is recreate+load: drop only the geonames index, then load it.
    # On resume this is re-run wholesale -- it is ~30 minutes, not worth
    # checkpointing more finely.
    if stage_done load; then
        log "geonames: recreate + load already done, skipping"
    else
        log "geonames: recreate + load"
        mark_running load
        "${RUN_LOADER[@]}" load_geonames_es.py reload
        mark_done load
    fi
    cd "$REPO_ROOT"
}

run_wiki() {
    STAGE_PREFIX="wiki_"
    cd "$REPO_ROOT/elasticsearch/es_wiki"
    mkdir -p data

    if stage_done download || [ "$SKIP_DOWNLOAD" -eq 1 ]; then
        log "wiki: skipping download"
    elif [ -n "${DUMP:-}" ] && [ -f "$DUMP" ]; then
        log "wiki: using supplied dump $DUMP"
        mark_done download
    else
        log "wiki: downloading dump (curl -C - resumes a partial file)"
        mark_running download
        # -C - continues rather than restarting, which is what makes an
        # interrupted 24 GB download survivable.
        [ -n "$DUMP_DATE" ] || die "no dump date resolved; pass --dump-date or --dump"
        curl -L -C - --retry 5 --retry-delay 10 \
            "$DUMP_URL_HOST/$DUMP_DATE/enwiki-$DUMP_DATE-pages-articles-multistream.xml.bz2" \
            -o "$DUMP"
        # The offset index is ~0.3 GB and is the whole reason for taking the
        # multistream variant. Failing to get it is not fatal: without it the
        # build just reads the .bz2, which is what it used to do anyway.
        if [ -n "$DUMP_INDEX" ]; then
            log "wiki: downloading the multistream offset index"
            curl -L -C - --retry 5 --retry-delay 10 \
                "$DUMP_URL_HOST/$DUMP_DATE/enwiki-$DUMP_DATE-pages-articles-multistream-index.txt.bz2" \
                -o "$DUMP_INDEX" \
                || warn "could not fetch the offset index -- the build will read the .bz2 directly"
        fi
        mark_done download
    fi
    [ -f "$DUMP" ] || die "dump not found: $DUMP"
    log "wiki: dump is $(human "$(file_size "$DUMP")")"

    # Refuse to build from a dump that changed mid-run: the redirect pickle and
    # the indexed articles would come from two different snapshots of Wikipedia.
    dump_sig="$(file_size "$DUMP"):$(file_mtime "$DUMP")"
    prev_sig="$(state_get dump_sig)"
    if [ -n "$prev_sig" ] && [ "$prev_sig" != "$dump_sig" ]; then
        die "the dump changed since this run started (size:mtime $prev_sig -> $dump_sig).
       Mixing two dumps produces an index whose redirects do not match its articles.
       Start a fresh run instead of resuming."
    fi
    state_set dump_sig "$dump_sig"

    # -----------------------------------------------------------------------
    # Decompress once in parallel, instead of twice single-threaded.
    #
    # build_links and load_es each read the whole dump, and Python's bz2 runs at
    # about 38 MB/s, so reading the .bz2 spends ~45 minutes per pass just
    # decompressing. A multistream dump can be cut at the stream offsets its
    # index lists and decompressed by every core at once: ~13 minutes, paid
    # once. The XML is identical either way, so this is purely an optimisation
    # and every failure path below falls back to handing the loader the .bz2.
    # -----------------------------------------------------------------------
    WIKI_INPUT="$DUMP"
    free="$(free_gb "$(dirname "$DUMP_XML")")"
    if [ "$PARALLEL_BUNZIP" -eq 0 ]; then
        log "wiki: up-front decompress disabled (--no-parallel-bunzip)"
    elif stage_done decompress && [ -s "$DUMP_XML" ]; then
        log "wiki: reusing $(basename "$DUMP_XML") from an earlier run"
        WIKI_INPUT="$DUMP_XML"
    elif [ -z "$DUMP_INDEX" ] || [ ! -f "$DUMP_INDEX" ]; then
        warn "no multistream offset index beside $(basename "$DUMP")."
        warn "         Reading the .bz2 directly -- correct, but ~45 min slower per pass."
    elif [ "$free" -lt "$PARALLEL_BUNZIP_MIN_GB" ]; then
        warn "only ${free} GB free, and decompressing up front needs ~${PARALLEL_BUNZIP_MIN_GB} GB."
        warn "         Reading the .bz2 directly -- slower, but no extra disk."
    else
        log "wiki: decompressing the dump in parallel (~13 min, ~117 GB)"
        mark_running decompress
        if "${RUN_LOADER[@]}" "$REPO_ROOT/elasticsearch/tools/parallel_bunzip2.py" \
                "$DUMP" "$DUMP_INDEX" "$DUMP_XML"; then
            mark_done decompress
            WIKI_INPUT="$DUMP_XML"
            log "wiki: decompressed to $(basename "$DUMP_XML") ($(human "$(file_size "$DUMP_XML")"))"
        else
            # A half-written .xml is worse than none: a later --resume would
            # find it and quietly build the index from a truncated dump.
            warn "parallel decompress failed its own page-count check -- discarding the output."
            warn "         Falling back to the .bz2. The index is unaffected, only slower to build."
            rm -f "$DUMP_XML"
            rm -rf "$DUMP_XML.parts"
            state_set "stage_${STAGE_PREFIX}decompress" "failed"
        fi
    fi
    log "wiki: loaders will read $(basename "$WIKI_INPUT")"

    if stage_done build_links; then
        log "wiki: skipping build_links (already done)"
    else
        log "wiki: build_links -- scanning dump for redirects (hours)"
        mark_running build_links
        "${RUN_LOADER[@]}" load_wiki_es.py build_links "$WIKI_INPUT"
        [ -s data/redirect_dict.pkl ] || die "build_links produced no redirect_dict.pkl"
        mark_done build_links
    fi

    # Always re-run, never skipped on resume. Redis holds the redirect map in
    # memory only, so any restart of the container empties it while the state
    # file still records the stage as done. Skipping it then produces a full
    # index whose documents all have empty `redirects` -- which looks like a
    # successful build and passes the anomaly gate. Reloading costs ~30s.
    log "wiki: load_redis (always re-run; Redis is in-memory)"
    mark_running load_redis
    "${RUN_LOADER[@]}" load_wiki_es.py load_redis "$WIKI_INPUT"
    mark_done load_redis

    # The --drop decision is the one place resume semantics really matter.
    # A fresh run drops the index so the rebuild is clean. A resumed run must
    # NOT drop: the original run already did, and dropping again would discard
    # exactly the partial progress we are resuming for.
    drop_flag="--drop"
    if [ "$(state_get "stage_${STAGE_PREFIX}load_es")" = "running" ]; then
        drop_flag=""
        log "wiki: load_es was interrupted -- resuming WITHOUT --drop"
        log "      (documents already indexed will be rewritten in place; _id is the"
        log "       page title, so this upserts rather than duplicating)"
    fi
    if stage_done load_es; then
        log "wiki: skipping load_es (already done)"
    else
        log "wiki: load_es -- parsing and indexing articles (many hours)"
        mark_running load_es
        # shellcheck disable=SC2086
        "${RUN_LOADER[@]}" load_wiki_es.py load_es $drop_flag "$WIKI_INPUT"
        mark_done load_es
    fi
    cd "$REPO_ROOT"
}

# On interrupt, leave the stack up: resume is then instant, and stopping ES
# mid-write buys nothing. Print how to continue rather than guessing.
on_interrupt() {
    printf '\n'
    warn "Interrupted. Progress is checkpointed in $STATE_FILE"
    warn "The build stack is still running (so --resume starts immediately)."
    warn ""
    warn "  continue :  tools/rebuild_index.sh --resume $TARGET"
    warn "  give up  :  ${COMPOSE[*]} down"
    [ -n "$PREV_ES_CONTAINER" ] && warn "              docker start $PREV_ES_CONTAINER"
    [ -n "$(state_get backup_path)" ] && warn "  restore  :  rm -rf '$ES_DATA' && mv '$(state_get backup_path)' '$ES_DATA'"
    exit 130
}
trap on_interrupt INT TERM

case "$TARGET" in
    wiki)     run_wiki ;;
    geonames) run_geonames ;;
    both)     run_geonames; run_wiki ;;
esac
trap - INT TERM

# ---------------------------------------------------------------------------
# AFTER snapshot + report
# ---------------------------------------------------------------------------
log "Capturing AFTER state -> $AFTER_SNAP"
NGEC_ES_DATA="$ES_DATA" "${RUN_LOADER[@]}" tools/index_stats.py -o "$AFTER_SNAP"

echo
REPORT_STATUS=0
NGEC_ES_DATA="$ES_DATA" "${RUN_LOADER[@]}" tools/index_stats.py \
    --diff "$BEFORE_SNAP" "$AFTER_SNAP" || REPORT_STATUS=$?
echo

state_set completed "yes"

# ---------------------------------------------------------------------------
# Tear down and restore
# ---------------------------------------------------------------------------
log "Stopping the build stack"
"${COMPOSE[@]}" down

if [ -n "$PREV_ES_CONTAINER" ]; then
    log "Restarting '$PREV_ES_CONTAINER'"
    docker start "$PREV_ES_CONTAINER" >/dev/null || warn "could not restart $PREV_ES_CONTAINER"
fi

rule
if [ "$REPORT_STATUS" -ne 0 ]; then
    printf '%sRebuild finished, but the report flagged anomalies above.%s\n' "$R" "$N"
    printf 'Do not publish this index until you understand them.\n'
    [ -n "$(state_get backup_path)" ] && printf 'Backup: %s\n' "$(state_get backup_path)"
else
    printf '%sRebuild complete.%s\n' "$G" "$N"
    printf 'Publish with: tools/publish_index.sh\n'
    if [ -n "$(state_get backup_path)" ]; then
        printf 'Backup (delete once satisfied): %s\n' "$(state_get backup_path)"
    fi
    if [ -n "$DUMP_XML" ] && [ -s "$DUMP_XML" ]; then
        printf 'Decompressed dump (delete to reclaim %s): %s\n' \
            "$(human "$(file_size "$DUMP_XML")")" "$DUMP_XML"
    fi
fi
printf 'Snapshots: %s\n           %s\n' "$BEFORE_SNAP" "$AFTER_SNAP"
rule
exit "$REPORT_STATUS"

# ---------------------------------------------------------------------------
# Possible improvement, deliberately not done here
#
# load_es restarts at the beginning of the dump, so an interruption late in the
# run wastes most of a day. Fixing that properly belongs in the loader, not
# this script: it would need load_es to persist its page counter alongside the
# index (the ES doc count is not a usable offset, because _should_skip_title
# discards many pages, so page index and document count diverge), plus a
# --skip-pages option to fast-forward the iterator without re-parsing wikitext.
# The XML is streamed from a bz2 file and cannot be seeked, but skipping the
# mwparserfromhell parse and the ES round trip is where nearly all the time
# goes, so a fast-forward would recover most of it.
# ---------------------------------------------------------------------------
