"""Pre-code the demo's curated examples so the app serves them instantly.

Coding one document costs roughly a minute on CPU, almost all of it in the
attribute model. That is fine for text a visitor types in — the app says so and
shows a progress bar — but it is unusable for the curated examples, which are
what a reviewer actually clicks through.

This script runs the full pipeline over every example in `ngec_demo/examples.py`
and writes the traces to `demo/data/example_runs.json`. The app reads that file
and serves the pre-coded trace; a step page that only needs the first two steps
slices the same cached run rather than re-running.

    uv run python demo/build_example_cache.py

Re-run it whenever the examples or any model change. The output is a build
artefact, not source. Because the attribute model samples at temperature 0.5,
the cached run is one draw — which is also what a live run would give.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "data" / "example_runs.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", nargs="*", default=None,
                        help="example keys to (re)build; default is all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    try:
        from ngec.logging import setup_logging

        setup_logging(level=logging.ERROR, quiet_third_party=True)
    except Exception:  # noqa: BLE001
        pass

    from ngec_demo import examples as ex_mod
    from ngec_demo import pipeline

    wanted = [e for e in ex_mod.EXAMPLES
              if args.only is None or e.key in args.only]

    # The non-English page also codes a fixed English translation, and the
    # comparison page reuses the end-to-end examples; both come through the same
    # cache, so include any extra fixed texts here.
    extra: list[tuple[str, str, str]] = [
        (
            "spanish_translation",
            "Hundreds of demonstrators gathered on Tuesday in front of the presidential "
            "palace in Bogota to protest against the tax reform. Riot police used tear "
            "gas to disperse the crowd. The president declared that the reform would go "
            "ahead. The unions announced a general strike for next week.",
            "2024-07-22",
        ),
    ]

    jobs = [(e.key, e.text, e.pub_date) for e in wanted]
    if args.only is None:
        jobs += extra

    cache = {}
    if args.out.exists():
        try:
            cache = json.loads(args.out.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}

    print(f"Coding {len(jobs)} example(s)", flush=True)
    started = time.time()

    for i, (key, text, pub_date) in enumerate(jobs, 1):
        t0 = time.time()
        result = pipeline.run(text, pub_date)
        cache[pipeline.cache_key(text, pub_date)] = pipeline.run_to_dict(result)

        status = result.error or f"{len(result.final)} event(s)"
        print(f"  [{i}/{len(jobs)}] {key}: {time.time() - t0:.0f}s · {status}", flush=True)

        # Write after every example so an interrupted build still leaves a
        # usable cache behind.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cache, default=str), encoding="utf-8")

    print(f"\nWrote {args.out} ({len(cache)} entries, "
          f"{time.time() - started:.0f}s total)", flush=True)

    _shutdown_vllm()


def _shutdown_vllm() -> None:
    """Stop vllm's engine subprocess before exiting.

    vllm runs its engine core in a separate process. When this script exits that
    process does not always go with it, and an orphaned `VLLM::EngineCore` holds
    ~20 GB of the card — enough that the next thing to want the GPU (the
    sentence encoder in step 1, say) fails with an out-of-memory error that
    looks nothing like its cause. Shutting the client down explicitly avoids
    that. Best-effort: the private path differs across vllm versions, and
    failing to tear down cleanly must not fail the build.
    """
    from ngec_demo import resources

    try:
        model = getattr(resources.get_attribute_model(), "model", None)
        client = getattr(getattr(model, "llm_engine", None), "engine_core", None)
        if client is not None and hasattr(client, "shutdown"):
            client.shutdown()
            print("vllm engine shut down", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"note: could not shut vllm down cleanly ({exc}); "
              "check `nvidia-smi` for a leftover VLLM::EngineCore", flush=True)


if __name__ == "__main__":
    sys.exit(main())
