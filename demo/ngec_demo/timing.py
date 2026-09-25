"""Timing the pipeline's sub-components without putting timers in the library.

The demo wants to say where a document's seconds went -- how much was the
sentence encoder, how much the LLM, how much Elasticsearch -- and to compare
that between GPU and CPU. Rather than sprinkling timers through `ngec/`, the
loaders in `resources.py` wrap a handful of methods on the *cached instances*
with `instrument()`, and every step runs its body inside `collect()`. The
library stays free of timing code, and a method that gets renamed upstream
fails loudly at load time instead of quietly measuring nothing.

Nesting comes from an explicit stack of open labels, so a wrapped method called
inside `timed("actors")` is reported as "actors/wikipedia". A parent's seconds
is its own wall time, not the sum of its children; the difference is work the
parent did outside any wrapped call.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

MODES = ("gpu", "cpu")

# Streamlit runs each script in its own thread, and check_demo.py runs two modes
# one after another, so the open-label stack and the active collector are
# per-thread rather than module-level state.
_state = threading.local()


class Collector:
    """The timings recorded inside one `collect()` block."""

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._order: list[str] = []

    def add(self, path: str, seconds: float, calls: int = 1) -> None:
        if path not in self._totals:
            self._totals[path] = 0.0
            self._calls[path] = 0
            self._order.append(path)
        self._totals[path] += seconds
        self._calls[path] += calls

    def rows(self) -> list[dict]:
        """One row per path, in tree order, JSON-safe.

        [{"path": "attributes/generate", "label": "generate", "depth": 1,
          "seconds": 0.26, "calls": 3}]. `seconds` is the row's own wall time
        summed over its calls, so a parent is not the sum of its children --
        the UI shows the remainder.
        """
        # Sort by the order each path's *ancestors* were first seen, so a
        # child always follows its parent even if a sibling was recorded first.
        rank = {path: i for i, path in enumerate(self._order)}

        def key(path: str):
            parts = path.split("/")
            return tuple(rank.get("/".join(parts[: i + 1]), 0)
                         for i in range(len(parts)))

        return [{"path": path,
                 "label": path.split("/")[-1],
                 "depth": path.count("/"),
                 "seconds": round(self._totals[path], 4),
                 "calls": self._calls[path]}
                for path in sorted(self._totals, key=key)]


@contextmanager
def collect() -> Iterator[Collector]:
    """Start a fresh collector; `timed()` blocks inside it record into it.

    Re-entrant: a nested `collect()` gets its own collector and its own label
    stack, so a step called from inside another step does not pollute it.
    """
    previous_collector = getattr(_state, "collector", None)
    previous_stack = getattr(_state, "stack", None)
    collector = Collector()
    _state.collector = collector
    _state.stack = []
    try:
        yield collector
    finally:
        _state.collector = previous_collector
        _state.stack = previous_stack


@contextmanager
def timed(label: str):
    """Time a block under `label`, nested under any open `timed()` blocks.

    A no-op (beyond the stack push) when no collector is active, so the
    instrumented methods are safe to call outside a step.
    """
    stack = getattr(_state, "stack", None)
    if stack is None:
        stack = _state.stack = []
    stack.append(label)
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        stack.pop()
        collector = getattr(_state, "collector", None)
        if collector is not None:
            collector.add("/".join(stack + [label]), elapsed)


def record(label: str, seconds: float, calls: int = 1) -> None:
    """Add a measured value that did not come from a block.

    For numbers someone else measured for us -- the llama-server's own
    prompt_ms / predicted_ms -- filed under the currently open path.
    """
    collector = getattr(_state, "collector", None)
    if collector is None:
        return
    stack = getattr(_state, "stack", None) or []
    collector.add("/".join(list(stack) + [label]), seconds, calls)


def instrument(obj, methods: dict[str, str]) -> None:
    """Wrap bound methods on one instance so each call is `timed(label)`.

    `methods` is {method_name: label}; a name may be dotted
    ("wiki_matcher.query_wiki") to reach a method on an attribute. Idempotent,
    so a re-run of a loader does not stack wrappers, and a missing method
    raises AttributeError here rather than silently measuring nothing -- that
    is how `check_demo.py` catches a rename in `ngec/`.
    """
    for name, label in methods.items():
        *path, attribute = name.split(".")
        target = obj
        for part in path:
            target = getattr(target, part)
        original = getattr(target, attribute)  # AttributeError if renamed
        if getattr(original, "_ngec_timed", None):
            continue

        def wrapper(*args, _fn=original, _label=label, **kwargs):
            with timed(_label):
                return _fn(*args, **kwargs)

        wrapper._ngec_timed = label
        setattr(target, attribute, wrapper)
