"""
Snapshot the state of the offline `wiki` and `geonames` Elasticsearch indices.

Used by tools/rebuild_index.sh to capture before/after pictures of a rebuild,
and by tools/publish_index.sh to write the manifest that ships beside a
published tarball. Also fine to run by hand:

    uv run --group es-build python tools/index_stats.py --text

Three modes:

    (default)          write a JSON snapshot to stdout
    --text             render that snapshot as a human-readable report
    --render A.json    render a previously saved snapshot as a report
    --diff A.json B.json   render two snapshots side by side with deltas

The snapshot covers, per index: existence, document count, store size, cluster
health, and the mapping `_meta` the loaders stamp on (dump file, dump date,
build date, builder). It also covers the wiki build's intermediate state --
the redirect pickle on disk and the Redis key count -- because a wiki index
that looks fine can still have been built from an empty or stale redirect set,
and that is invisible in the ES stats alone.

Elasticsearch and Redis locations come from NGEC_ES_URL, NGEC_REDIS_HOST and
NGEC_REDIS_PORT, matching the loaders.
"""

import argparse
import datetime
import json
import os
import pickle
import sys

from elasticsearch import Elasticsearch

ES_URL = os.environ.get("NGEC_ES_URL", "http://localhost:9200/")
REDIS_HOST = os.environ.get("NGEC_REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("NGEC_REDIS_PORT", "6379"))

REDIRECT_DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "elasticsearch", "es_wiki", "data", "redirect_dict.pkl",
)

INDICES = ("wiki", "geonames")


def human_bytes(n):
    """Format a byte count. Returns '-' for None so report rows stay aligned."""
    if n is None:
        return "-"
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= step


def human_int(n):
    return "-" if n is None else f"{n:,}"


def index_snapshot(es, index):
    """Stats plus stamped `_meta` for one index, or exists=False if absent."""
    try:
        if not es.indices.exists(index=index):
            return {"exists": False}
    except Exception as e:
        return {"exists": None, "error": str(e)}

    out = {"exists": True}
    try:
        es.indices.refresh(index=index)
        # bytes="b" so the number is diffable; the human form is rendered later.
        row = es.cat.indices(index=index, format="json", bytes="b")[0]
        out["doc_count"] = int(row["docs.count"])
        out["deleted_docs"] = int(row.get("docs.deleted") or 0)
        out["store_bytes"] = int(row["store.size"])
        out["health"] = row["health"]
    except Exception as e:
        out["error"] = str(e)

    try:
        mappings = es.indices.get_mapping(index=index)[index]["mappings"]
        out["meta"] = mappings.get("_meta", {})
        out["field_count"] = len(mappings.get("properties", {}))
    except Exception as e:
        out["meta"] = {}
        out["meta_error"] = str(e)

    return out


def redirect_snapshot(count_entries=True):
    """State of the wiki build's redirect pickle and its Redis load.

    The pickle maps each target page to the list of titles redirecting to it,
    so "targets" and "links" are different numbers and both are worth seeing:
    a plausible target count with a collapsed link count means a bad parse.
    """
    out = {"path": REDIRECT_DICT_PATH, "exists": os.path.exists(REDIRECT_DICT_PATH)}
    if out["exists"]:
        st = os.stat(REDIRECT_DICT_PATH)
        out["size_bytes"] = st.st_size
        out["modified"] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        if count_entries:
            try:
                with open(REDIRECT_DICT_PATH, "rb") as f:
                    d = pickle.load(f)
                out["redirect_targets"] = len(d)
                out["redirect_links"] = sum(len(v) for v in d.values())
            except Exception as e:
                out["error"] = str(e)

    # Redis holds the same dict during a build. It is emptied between runs in
    # practice (the build stack's redis has no persistence), so a zero here
    # next to a populated pickle just means load_redis hasn't run yet.
    try:
        import redis as redis_lib
        r = redis_lib.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_connect_timeout=3)
        out["redis_keys"] = r.dbsize()
    except Exception as e:
        out["redis_keys"] = None
        out["redis_error"] = str(e).split("\n")[0][:120]

    return out


def snapshot(count_entries=True):
    es = Elasticsearch(ES_URL, timeout=60, max_retries=2, retry_on_timeout=True)
    snap = {
        "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "es_url": ES_URL,
        "es_data_dir": os.environ.get("NGEC_ES_DATA"),
        "indices": {},
    }
    try:
        snap["cluster_health"] = es.cluster.health()["status"]
    except Exception as e:
        snap["cluster_health"] = None
        snap["cluster_error"] = str(e).split("\n")[0][:200]

    for index in INDICES:
        snap["indices"][index] = index_snapshot(es, index)
    snap["redirects"] = redirect_snapshot(count_entries=count_entries)
    return snap


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
META_FIELDS = ("dump_file", "dump_date", "build_date", "doc_count", "builder",
               "source", "gazetteer_file")


def render_text(snap, out=sys.stdout):
    p = lambda s="": print(s, file=out)
    p(f"Elasticsearch : {snap['es_url']}  (cluster: {snap.get('cluster_health') or 'UNREACHABLE'})")
    if snap.get("es_data_dir"):
        p(f"Data directory: {snap['es_data_dir']}")
    p(f"Captured      : {snap['captured_at']}")
    p()
    for index in INDICES:
        d = snap["indices"][index]
        p(f"[{index}]")
        if not d.get("exists"):
            p("  index does not exist")
            p()
            continue
        p(f"  documents  : {human_int(d.get('doc_count'))}"
          + (f"  (+{human_int(d['deleted_docs'])} deleted)" if d.get("deleted_docs") else ""))
        p(f"  store size : {human_bytes(d.get('store_bytes'))}")
        p(f"  health     : {d.get('health', '-')}   fields: {d.get('field_count', '-')}")
        meta = d.get("meta") or {}
        if meta:
            p("  _meta:")
            for k in META_FIELDS:
                if k in meta:
                    v = human_int(meta[k]) if k == "doc_count" else meta[k]
                    p(f"    {k:<15}: {v}")
            for k, v in meta.items():
                if k not in META_FIELDS:
                    p(f"    {k:<15}: {v}")
        else:
            p("  _meta      : (none -- built before stamping, or dropped since)")
        p()

    r = snap.get("redirects", {})
    p("[wiki redirect build state]")
    if r.get("exists"):
        p(f"  pickle     : {human_bytes(r.get('size_bytes'))}  modified {r.get('modified')}")
        if "redirect_targets" in r:
            p(f"  targets    : {human_int(r['redirect_targets'])}")
            p(f"  links      : {human_int(r['redirect_links'])}")
    else:
        p(f"  pickle     : not present ({r.get('path')})")
    p(f"  redis keys : {human_int(r.get('redis_keys'))}"
      + (f"   ({r['redis_error']})" if r.get("redis_error") else ""))


def _delta(before, after, fmt=human_int):
    """A 'before -> after (+delta)' cell, tolerant of missing values."""
    b_s, a_s = fmt(before), fmt(after)
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        d = after - before
        sign = f"{d:+,}" if fmt is human_int else f"{'+' if d >= 0 else '-'}{human_bytes(abs(d))}"
        pct = f"  {d / before * 100:+.1f}%" if before else ""
        return f"{b_s:>16}  ->  {a_s:<16}  {sign}{pct}"
    return f"{b_s:>16}  ->  {a_s:<16}"


def render_diff(before, after, out=sys.stdout):
    p = lambda s="": print(s, file=out)
    p("=" * 78)
    p("REBUILD REPORT")
    p("=" * 78)
    p(f"before: {before['captured_at']}")
    p(f"after : {after['captured_at']}")
    if after.get("es_data_dir"):
        p(f"data  : {after['es_data_dir']}")
    p()

    for index in INDICES:
        b = before["indices"].get(index, {})
        a = after["indices"].get(index, {})
        if not b.get("exists") and not a.get("exists"):
            continue
        p(f"[{index}]")
        p(f"  documents  {_delta(b.get('doc_count'), a.get('doc_count'))}")
        p(f"  store size {_delta(b.get('store_bytes'), a.get('store_bytes'), human_bytes)}")
        p(f"  health     {b.get('health', '-'):>16}  ->  {a.get('health', '-')}")

        bm, am = b.get("meta") or {}, a.get("meta") or {}
        changed = [k for k in sorted(set(bm) | set(am)) if bm.get(k) != am.get(k)]
        if changed:
            p("  _meta changes:")
            for k in changed:
                p(f"    {k:<15}: {bm.get(k, '(unset)')}  ->  {am.get(k, '(unset)')}")
        elif am:
            p("  _meta      : unchanged")
        p()

    br, ar = before.get("redirects", {}), after.get("redirects", {})
    if br or ar:
        p("[wiki redirect build state]")
        p(f"  targets    {_delta(br.get('redirect_targets'), ar.get('redirect_targets'))}")
        p(f"  links      {_delta(br.get('redirect_links'), ar.get('redirect_links'))}")
        p(f"  pickle     {_delta(br.get('size_bytes'), ar.get('size_bytes'), human_bytes)}")
        p(f"  redis keys {_delta(br.get('redis_keys'), ar.get('redis_keys'))}")
        p()

    # A rebuild that ends with fewer documents than it started with is the
    # signature of a truncated dump or a load that died partway, and it is the
    # one outcome you must not publish. Call it out rather than leaving it in
    # the numbers.
    warnings = []
    for index in INDICES:
        b_n = (before["indices"].get(index) or {}).get("doc_count")
        a_n = (after["indices"].get(index) or {}).get("doc_count")
        if isinstance(b_n, int) and isinstance(a_n, int):
            if a_n == 0 and b_n > 0:
                warnings.append(f"{index}: index is now EMPTY (was {b_n:,})")
            elif b_n and a_n < b_n * 0.9:
                warnings.append(f"{index}: lost {(1 - a_n / b_n) * 100:.1f}% of documents ({b_n:,} -> {a_n:,})")
        if (after["indices"].get(index) or {}).get("health") == "red":
            warnings.append(f"{index}: cluster health is RED")
    if warnings:
        p("!" * 78)
        for w in warnings:
            p(f"!! {w}")
        p("!! Do NOT publish this index until you understand why.")
        p("!" * 78)
    else:
        p("No anomalies detected in document counts or health.")
    return 1 if warnings else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="store_true", help="render a report instead of JSON")
    ap.add_argument("--render", metavar="SNAPSHOT",
                    help="render a saved JSON snapshot (does not contact Elasticsearch)")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="render two saved JSON snapshots side by side")
    ap.add_argument("--no-redirect-count", action="store_true",
                    help="skip unpickling the redirect dict (faster; omits target/link counts)")
    ap.add_argument("-o", "--output", help="write to this file instead of stdout")
    args = ap.parse_args()

    if args.render:
        with open(args.render) as f:
            render_text(json.load(f))
        return 0

    if args.diff:
        with open(args.diff[0]) as f:
            before = json.load(f)
        with open(args.diff[1]) as f:
            after = json.load(f)
        # Non-zero exit signals "anomalies found" so a caller can gate on it.
        return render_diff(before, after)

    snap = snapshot(count_entries=not args.no_redirect_count)
    stream = open(args.output, "w") if args.output else sys.stdout
    try:
        if args.text:
            render_text(snap, out=stream)
        else:
            json.dump(snap, stream, indent=2, sort_keys=True)
            stream.write("\n")
    finally:
        if args.output:
            stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
