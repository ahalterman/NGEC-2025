"""Decompress a Wikipedia *multistream* dump in parallel, using its index file.

Why this exists: the loader reads the .bz2 dump directly, but Python's bz2 is
single-threaded and decompresses this dump at about 38 MB/s, so each of the two
passes over it (build_links, then load_es) spends roughly 45 minutes just
decompressing. Doing it once, in parallel, and handing the loader the plain
.xml it already knows how to read costs disk (about 105 GB) and saves most of
that. Nothing about the loader or the documents changes: the XML is the same
bytes either way.

A "multistream" dump is a concatenation of independent bz2 streams, each
holding ~100 pages, and the companion -index.txt.bz2 lists the byte offset of
every stream. So the file can simply be cut at those offsets and the pieces
decompressed by separate processes.

    python3 parallel_bunzip2.py DUMP.bz2 INDEX.txt.bz2 OUT.xml [workers]

Exits non-zero if the result does not contain exactly as many <page> elements
as the index file has lines. The caller is expected to fall back to feeding the
loader the .bz2 directly if that happens -- this is an optimisation, never a
source of truth.
"""

import bz2
import multiprocessing
import os
import sys
import time


def stream_offsets(index_path):
    """Distinct bz2 stream start offsets, in order, and the total page count."""
    offsets = []
    pages = 0
    last = None
    with bz2.open(index_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            pages += 1
            off = int(line.split(":", 1)[0])
            if off != last:
                offsets.append(off)
                last = off
    return offsets, pages


def decompress_range(job):
    """Decompress the byte range [start, end) of the dump into its own part file."""
    dump, start, end, part_path = job
    with open(dump, "rb") as fin, open(part_path, "wb") as fout:
        fin.seek(start)
        remaining = end - start
        dec = bz2.BZ2Decompressor()
        while remaining > 0:
            chunk = fin.read(min(1 << 22, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            while chunk:
                fout.write(dec.decompress(chunk))
                if not dec.eof:
                    break
                # This stream ended mid-chunk; the rest belongs to the next one.
                chunk = dec.unused_data
                dec = bz2.BZ2Decompressor()
    return os.path.getsize(part_path)


def main():
    dump, index_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()

    t = time.time()
    offsets, expected_pages = stream_offsets(index_path)
    print(f"index: {len(offsets)} streams, {expected_pages} pages "
          f"({time.time() - t:.0f}s)", flush=True)

    size = os.path.getsize(dump)
    # Split the streams into `workers` contiguous runs, so the parts concatenate
    # back in order.
    per = max(1, len(offsets) // workers)
    bounds = offsets[::per][:workers]
    # The index's first offset is the first *page* stream, not byte 0. The bytes
    # before it are a separate stream holding the <mediawiki> root tag and
    # <siteinfo>; starting there keeps the output a well-formed document with
    # the namespace the loader detects. Starting at offsets[0] silently drops it.
    bounds[0] = 0
    ends = bounds[1:] + [size]
    part_dir = out_path + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    jobs = [
        (dump, s, e, os.path.join(part_dir, f"part.{i:04d}"))
        for i, (s, e) in enumerate(zip(bounds, ends))
    ]

    t = time.time()
    with multiprocessing.Pool(workers) as pool:
        sizes = pool.map(decompress_range, jobs)
    print(f"decompressed {sum(sizes) / 1e9:.1f} GB with {len(jobs)} workers "
          f"({time.time() - t:.0f}s)", flush=True)

    t = time.time()
    pages = 0
    tag = b"<page>"
    with open(out_path, "wb") as fout:
        # `carry` holds the last few bytes of the previous read and is prepended
        # to the next one, so a <page> tag lying across a read boundary is still
        # counted. Without it the count comes out a handful short on a file this
        # size -- about 7 misses expected over 117 GB -- which looks exactly like
        # a decompression fault and isn't.
        carry = b""
        for job in jobs:
            with open(job[3], "rb") as fin:
                while True:
                    chunk = fin.read(1 << 24)
                    if not chunk:
                        break
                    pages += (carry + chunk).count(tag)
                    carry = chunk[-(len(tag) - 1):]
                    fout.write(chunk)
            os.remove(job[3])
    os.rmdir(part_dir)
    print(f"concatenated to {out_path} ({time.time() - t:.0f}s), {pages} <page> elements",
          flush=True)

    with open(out_path, "rb") as fchk:
        head = fchk.read(4096)
        fchk.seek(max(0, os.path.getsize(out_path) - 4096))
        tail = fchk.read()
    if b"<mediawiki" not in head or b"</mediawiki>" not in tail:
        print("MISMATCH: output is missing the <mediawiki> root tag", flush=True)
        sys.exit(1)
    if pages != expected_pages:
        print(f"MISMATCH: {pages} pages in the output, {expected_pages} in the index",
              file=sys.stderr)
        sys.exit(1)
    print("page count matches the index", flush=True)


if __name__ == "__main__":
    main()
