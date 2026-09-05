"""Check the demo without a human eyeballing it.

Three things go wrong silently when the app is edited: a page raises only when a
visitor opens it, an internal link points at a page that has been renamed or
removed, and a new page file never gets registered in the navigation. All three
are cheap to check mechanically, so they should not be caught by a reviewer.

    uv run --group demo-app python demo/check_demo.py          # links + every page
    uv run --group demo-app python demo/check_demo.py --links  # links only, no models

The full run executes every page against the live services and the real models,
which is the point — an import is not an execution. It takes a couple of minutes
and needs whatever backend the environment selects, same as the app itself.

Pages that cannot reach a dependency still count as passing: degrading to a
message is the intended behaviour, and the run reports which pages did that.
"""

import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
sys.path.insert(0, str(HERE))

# page("home.py", "Start here", "home", ...) in app.py
PAGE_CALL = re.compile(r'page\(\s*"([^"]+)"\s*,\s*"[^"]*"\s*,\s*"([^"]+)"')
# href="step1" / href='coref' — internal links are bare url_paths, possibly with
# an ?ex= query string. Anything with a scheme or a slash is external.
HREF = re.compile(r"""href=['"](?!https?:|#|mailto:)([^'"]+)['"]""")
PAGE_LINK = re.compile(r"""st\.page_link\(\s*['"]([^'"]+)['"]""")


def registered() -> tuple[dict[str, str], set[str]]:
    """(url_path -> page file, set of registered page files) from app.py."""
    source = (HERE / "app.py").read_text(encoding="utf-8")
    pairs = PAGE_CALL.findall(source)
    return {url: filename for filename, url in pairs}, {f for f, _ in pairs}


def check_links() -> list[str]:
    urls, files = registered()
    problems = []

    for path in sorted(PAGES.glob("*.py")):
        if path.name not in files:
            problems.append(f"{path.name} is not registered in app.py")

        source = path.read_text(encoding="utf-8")
        for target in HREF.findall(source):
            # Links carry the selected example along: step3?ex=anniversary.
            base = target.split("?")[0].strip()
            if base and base not in urls:
                problems.append(f"{path.name}: href='{target}' matches no url_path")
        for target in PAGE_LINK.findall(source):
            if not (HERE / target).exists():
                problems.append(f"{path.name}: st.page_link('{target}') does not exist")

    for filename in sorted(files):
        if not (PAGES / filename).exists():
            problems.append(f"app.py registers pages/{filename}, which does not exist")

    return problems


def check_pages() -> list[str]:
    from streamlit.testing.v1 import AppTest

    problems = []
    for path in sorted(PAGES.glob("*.py")):
        # home.py uses st.page_link, which needs the st.navigation context and
        # raises KeyError: 'url_pathname' when run standalone. That is a harness
        # artefact, so drive it through app.py instead.
        target = HERE / ("app.py" if path.name == "home.py" else f"pages/{path.name}")
        started = time.time()
        at = AppTest.from_file(str(target), default_timeout=600)
        try:
            at.run()
            errors = [str(e.value)[:200] for e in at.exception]
        except Exception as exc:  # noqa: BLE001 - a crash here is the finding
            errors = [f"{type(exc).__name__}: {exc}"[:200]]

        degraded = any("Not everything the demo needs" in m.value for m in at.markdown)
        status = "FAIL" if errors else "ok  "
        print(f"  {status} {path.name:26s} {time.time() - started:6.1f}s"
              f"{'   [ran degraded]' if degraded else ''}", flush=True)
        for error in errors:
            print(f"       {error}", flush=True)
        problems += [f"{path.name}: {e}" for e in errors]

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", action="store_true",
                        help="check links and registrations only; do not load models")
    args = parser.parse_args()

    print("Links and page registrations")
    problems = check_links()
    for problem in problems:
        print(f"  FAIL {problem}")
    if not problems:
        print("  ok   every link resolves and every page is registered")

    if not args.links:
        print("\nEvery page executes")
        problems += check_pages()

    print()
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    print("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
