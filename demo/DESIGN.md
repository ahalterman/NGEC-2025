# The demo app: design, decisions, and status

This is the reasoning behind `demo/`. `README.md` next to it says how to run the
thing; this says why it is shaped the way it is, what is unfinished, and what is
still undecided.

> **Picking this up cold?** Read **[Status as of 2026-08-06](#status-as-of-2026-08-06)**
> first — it has what works, what is outstanding, the environment traps on this
> machine, and how the services are supervised. Then **[Open questions for the author](#open-questions-for-the-author)**,
> which are decisions for the author rather than work to pick up.
>
> The one-paragraph version: fourteen pages, all verified running; the attribute
> model was too slow to demo (60s/doc on CPU) and now is not — `llama.cpp` with a
> quantized copy gets it to ~12s, which is the deployment target, and vllm on a
> GPU gets it to ~1s; curated examples are pre-coded so clicking between them is
> instant. The demo runs the **2026 retraining** of the attribute model, not the
> one published to Hugging Face. It is built around coding one document at a
> time, which is what shows the stages.

## What it is for

Two audiences, deliberately not averaged together.

**A reviewer**, who wants evidence and is looking for places the paper
overclaims. They will not install anything, will not type into an empty box, and
will notice if every example flatters the system.

**A prospective user** after publication, who wants to know whether their project
is a week of work or a year of it.

The site serves both by keeping them on separate paths from the landing page,
and by being unusually willing to show its own failures — which is what the
reviewer needs and what saves the user time.

The organising constraint: **every panel should be citable from a response
letter.** That is what drove deep-linkable URLs and per-section links into the
manuscript, and it is worth preserving through any redesign.

## Structure

Four sections, fourteen pages, one claim per page.

| Section | Pages | Answers |
|---|---|---|
| See it work | Start here · Document → events | What does this produce? |
| Look inside | Steps 1–5, one page each | What does each component actually do? |
| Hard cases | Temporal echoes · Coreference · Non-English · Against a frontier LLM | Where does it break, and what about the obvious alternative? |
| Make it yours | Your event types · Your actor categories · Setup cost | What would it take for my project? |

The step pages follow the paper's **five-part framework**, not the code's six
steps — `stories_to_events` is an implementation detail, not a concept. Each
step page carries a rail linking to its siblings and a line linking into the
manuscript by section and PDF page.

## Decisions worth keeping

**Pasting your own text is the call to action.** Coding a document nobody
curated is the one thing that distinguishes this from a screenshot, so on the
pages whose step reads an article the box holding that article is editable and
sits directly under the header, above everything else. There is no "example *or*
your own text" mode switch: there is one box, and the examples fill it.

This went through two worse designs. First, the only way to code your own story
was to overwrite a curated one inside an "Edit the text" expander — hidden and
slow. Then the picker grew an explicit "✎ Your own document…" entry, which was
correct but was still the last option in a dropdown, and read as a footnote to
the examples rather than as the invitation. The examples are scaffolding.

The **Code this document** button matters: a bare text area re-runs the pipeline
on every pause in typing, because each new string is a new cache key. Submitting
text that still matches an example verbatim is recognised as that example, so
its pre-coded run is still a cache hit and the button is instant.

**No empty text boxes.** The box opens with a worked example already in it and
already coded. A reviewer with fifteen minutes will not compose a test document,
and a blank textarea is where demos go to die.

**One panel, not two.** The document box replaced a separate read-only rendering
of the article. Two panels showing the same text raised a question the visitor
should never have to ask — which of these is the thing being coded? — and cost
the vertical space that made the paste box feel like an afterthought.

Steps 3, 4 and 5 do **not** show the article. They are handed a few spans that
earlier steps pulled out — a name to look up, a mention to categorise, a date
expression and a publication date — and reprinting the article on those pages
implied they read it. `ui.stage_input` prints what the step actually receives
instead. The document box is still on those pages, collapsed under the examples,
and opens automatically when it holds text the visitor typed. (Geolocation is
the exception inside step 5, and the panel says so: mordecai3 does read the
whole document.)

**The examples explain themselves briefly, or not at all.** Each curated example
has a one-line blurb under the pills and a longer "what to notice" note. Only
the examples the pipeline gets *wrong* show that note unfolded — those are an
argument the demo is making. For the rest it is behind a click, because a page
that opens with two paragraphs about the example before any output has told the
visitor that the examples are the subject, and they are not.

Steps 3, 4 and 5 do **not** show it. They are handed a few spans that earlier
steps pulled out — a name to look up, a mention to categorise, a date expression
and a publication date — and reprinting the article on those pages implied they
read it. `ui.stage_input` prints what the step actually receives instead, with a
link back to the end-to-end page for the article. (Geolocation is the exception
inside step 5, and the panel says so: mordecai3 does read the whole document.)

**The extracted spans are a table, not a marked-up article.** Each attribute
gets a row listing the spans that filled it, colour-keyed to the same four hues
the legend uses, in the same table shape as the coded record below it. The
earlier version highlighted the spans inline in the article with small uppercase
tags after each. It demonstrated well on a two-sentence example and failed on a
real one: four slots scattered across six paragraphs cannot be collected by eye,
a span belonging to one of three records was indistinguishable from the others,
and the tags broke the line spacing of the prose they annotated. The one thing
that version gave for free — noticing a span that is not in the article — is
kept as an explicit check (`theme.unmatched_spans`), reported under the table.

**The example follows you.** The selected document lives in the URL query string
(`?ex=anniversary`), so the step rail carries it across pages and any panel is
directly linkable. This is what makes "see §4.2 of the paper and *this* page of
the demo" a workable sentence in a response letter.

**Degrade, never crash.** Every dependency is health-checked and reported in the
sidebar; a missing index yields a message naming what is unavailable rather than
a traceback. `showErrorDetails = "none"` in the config guarantees a visitor never
sees a stack trace. A reviewer hitting a Python traceback is worse than a
reviewer hitting a page that says "the Wikipedia index is down".

**Ship the failures.** Examples carrying `honest=True` in `examples.py` are
included *because* the pipeline handles them badly, and are labelled as such in
the UI. Three failure classes get their own pages. This is a rhetorical choice as
much as an honest one: a demo that volunteers its worst cases is far more
credible than one that has to be caught out.

**Show what was dropped.** When the classifier fires for an event type and the
attribute model returns nothing for it, no record is made. That is the pipeline's
most interesting self-correction — the extraction step declining a classifier
that over-fired — and it used to appear only as a count in a step note and a
`*_dropped_events.jsonl` file no visitor could see. The step 2 and end-to-end
pages now list the dropped type-mode pairs, with a pointer to the step 1 scores,
because a page where the over-firing is invisible would flatter the classifier.

**Preloaded examples are arguments, not placeholders.** The actor-categories page
opens with four codes — `CHF` customary authority, `VIG` vigilantes, `EMB`
election management bodies, `DSP` diaspora organisations — chosen under one
frame: each is an actor PLOVER already codes, into a category that erases the
distinction a researcher studying it is trying to measure. A chief matches
*supreme leader* and comes out `GOV`; a diaspora association matches
*international community* and comes out `NON`, the code for strings that are not
actors at all. The page states the score and the winning pattern for each, and
every one of those numbers was measured against the live matcher rather than
illustrated. The earlier default — football ultras, village militias — was a
grab-bag that demonstrated the mechanism without motivating it, and one of its
codes (`MIL`) was a PLOVER code already, so the "before" and "after" were not
actually different categories.

The electoral commission is kept as the page's counter-example: PLOVER's file
contains that exact phrase, so an added `EMB` pattern cannot outrank it at 0.98.
Adding a category does not override a phrase already in the file, and a page that
only showed successes would leave a reader to discover that on their own data.

**Three letters, enforced.** `AgentMatcher._load_and_clean_agents` slices the
bracketed code at position three unconditionally: the first block is `code_1`,
the remainder is `code_2`, which is how `[GOVMIL]` and `[CVLOPP]` work. A
four-letter code does not fail there — it silently becomes a three-letter
category plus a one-letter modifier nobody defined. The page's parser therefore
returns complaints alongside patterns and names the offending line, and the prose
explains the rule rather than asserting it.

**Let the visitor test the ontology claim, not just read it.** The paper's case
for a replaceable ontology rests on the attribute model reading a *definition*
rather than recognising a fixed set of labels — if that holds, a new ontology
costs a step-1 classifier and nothing else downstream. The step 2 page therefore
carries a panel that prompts the attribute model directly with an event label and
definition the visitor writes, on text they choose, with no classifier and no
codebook in between, and says whether the label they used is a PLOVER type. It
is preloaded with an invented category, and offers the fourteen ECAV definitions
(thirteen of which are not PLOVER labels) alongside PLOVER's own for comparison.

The panel deliberately shows the assembled prompt: it is the *whole* briefing the
model gets about a new event type, it is short, and seeing that is the clearest
possible statement of both why a definition is enough and why a vague one
extracts badly. The panel sits outside the "did the pipeline run" guard, because
the document a visitor with their own ontology brings is exactly the document the
demonstration classifiers fire on nothing for.

This needed one change in the package: `AttributeModel._get_event_info` now
honours an `event_def` on the record and only falls back to the codebook CSV,
and raises a KeyError naming the known types when neither is available (it used
to fail with an `IndexError` from `.values[0]` on an empty selection).

**Say what is not implemented.** The temporal-echo page computes two diagnostics
*on the page* and states plainly that neither is part of the pipeline. The ECAV
comparison table against an LLM is deliberately **empty**, with a note saying the
numbers have not been run. Nothing on the site is a mock-up presented as a
result.

**One hero interaction per page**, everything else in expanders. Coverage without
clutter comes from splitting pages, not from denser pages.

## Architecture

```
app.py                  st.navigation; the only place set_page_config is called
handoff/
  pocket-operator.css   the design handoff as authored; owns the tokens
ngec_demo/
  theme.py              loads the handoff CSS, adds this app's own components
  paper.py              section → PDF page map; every "in the paper" link
  examples.py           curated documents, including the deliberate failures
  resources.py          cached model loading + dependency health
  pipeline.py           runs the pipeline with a per-step trace; single-step helpers
  llm_baseline.py       the one-prompt frontier-LLM comparison
  ui.py                 page furniture, example picker, record rendering
pages/                  one file per page
```

Two things carry most of the weight:

**`pipeline.run()` returns a trace**, not just an answer: a `StepTrace` per step
with its timing, a note, and a JSON-safe snapshot of the record list after it
ran. The step pages read one field out of that; the end-to-end page reads all of
it. Nothing in `pages/` reimplements pipeline logic, so if a page shows
something, the pipeline really produced it. Snapshots are JSON round-trips
because records carry spaCy docs and numpy scalars that do not deep-copy.

**`stop_after`** lets a page run only the prefix it needs. The detection page
stops after classification and returns in under a second instead of spending a
minute in the attribute model.

Results are cached in `st.session_state["_runs"]` keyed by
`(text, pub_date, stop_after, override)`, bounded to 12 entries, and runs whose
dependencies were down are not cached — a visitor may be reloading precisely
because the index came back.

## The visual theme

The look is the **"Pocket Operator"** design handoff (option 3B): white paper,
black hairlines, grey `#F2F2F0` chips, and exactly one orange `#F0521E`. Nothing
is rounded, nothing has a shadow — depth comes from 1px rules. Space Grotesk for
anything written for a person; IBM Plex Mono at 10px uppercase with `0.2em`
tracking for anything the machine says about itself: section labels, step
numbers, timings, units, field names. **Body copy is never mono**, and there is
no serif in this design.

The stylesheet is split in two on purpose:

- `handoff/pocket-operator.css` is the handoff **as authored** and owns the
  tokens and the Streamlit widget overrides. Treat it as read-only: when a
  Streamlit upgrade moves a `data-testid`, re-point the selector there rather
  than restyling the component in Python. (One edit was made on receipt: the
  `@import` was moved above `:root`, because CSS drops an `@import` that follows
  a style rule and the two webfonts were silently never loading.)
- `theme._components_css()` is this app's own furniture — notes, record tables,
  classifier bars, the step rail — written entirely against the `--po-*` custom
  properties, so a token change propagates without touching Python.

`theme.PALETTE` mirrors the tokens for the few places that build an inline style
from Python. Note that the **status triad collapsed**: the design has one accent
and names multi-hue status pills as a failure mode, so `good` is now black and
`warn`/`bad` are both the accent. A page that needs to distinguish "degraded"
from "failed" has to say so in words.

The boxed numeral (`01`…`05`) is the most characteristic detail in the design,
and the step rail is what it is for.

### The four attribute colours

Identity is carried by a saturated 3px rule down the left edge of the row label,
plus the label in words. The cell itself sits on the neutral chip — the previous
theme tinted each cell with a pale wash of its hue, which under this design would
put four competing colour fields on the page. Confining the hue to the hairline
keeps the paper neutral and keeps the identity.

The set is slots 1, 4, 3 and 7 of the reference categorical palette. Recipient
moved from orange `#eb6834` to yellow `#eda100`, because the old hue was
essentially the new accent and a reader would fairly have read an orange rule as
"this is the row to look at". The four were then re-validated **as a set with the
accent included**, since an accented label and an attribute table share a screen:
worst all-pairs CVD ΔE 9.1 (protan), worst normal-vision ΔE 16.3.

Every other fourth hue fails once the accent is in the pairlist — red is ΔE 4.9
from it under deuteranopia, green 1.5 under protanopia, magenta 14.6 to normal
vision. Darkening the yellow to lift its contrast backfires for the same reason:
`#c98500` collapses to ΔE 1.5 from the accent, because brightness is precisely
what separates yellow from orange under CVD. So the bright step stands, and its
low contrast on the chip (1.93:1) is relieved the way it always was here — every
row is labelled in words.

The earlier history still applies: the obvious "muted academic" choice — blue,
terracotta, olive, teal — was written first and **failed**, with olive against
terracotta at ΔE 3.9 under protanopia. **Re-run the validator rather than
trusting your eye if these change, and include `#F0521E` in the set.**

**All of it goes in one `<style>` element, emitted starting at column 0.**
`st.markdown` runs `textwrap.dedent(...).strip()` on what it is given, and the
generated per-role rules have no leading indentation, so the dedent is a no-op
for the whole string and only `.strip()` saves the first line. A second `<style>`
further down therefore stayed indented, markdown read an indented line as a code
block, and the CSS was printed on the page as text.

## Status as of 2026-08-06

### The state of the machine, right now

Not documentation of intent — what is actually true of this box, which is the
part a fresh session cannot see and will otherwise rediscover the hard way.

- **Elasticsearch**: docker container `silly_shockley`, `restart=unless-stopped`
  (applied), `wiki` 7,854,807 docs / `geonames` 12,571,784 docs.
- **`llama-server`**: running from a shell, serving `attr-exp5.1-q8.gguf` on
  :8080. **Not** under systemd — the unit exists in `demo/deploy/` and has never
  been installed. It will not survive a reboot or a logout.
- **The demo**: nothing is running; start it with
  `cd demo && env -u LD_LIBRARY_PATH uv run --group demo-app streamlit run app.py`.
- **The GPU is free.** Note that a running demo *does* take about 1 GB of it even
  on the `llamacpp` backend — spaCy's transformer model and the sentence encoder
  go to the card if there is one. It is `NGEC_DEMO_BACKEND=vllm` that takes the
  large reservation, and that is only needed for building the example cache.
  Before starting anything that wants the card, check for an orphaned
  `VLLM::EngineCore`
  (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`) — one of those
  left running is what makes a later vllm load fail with an out-of-memory error
  that names no cause, and it breaks whatever the *user* is running too, not just
  the next script.
- **Attribute model**: `~/projects/train_NGEC_2026/qwen3-event-extraction-exp5.1`,
  loaded from that path because it is not on Hugging Face yet. The GGUF served by
  `llama-server` is built from the same weights. If either moves, the health
  panel will say so rather than running mismatched.

### Working

**All fourteen pages run.** Verified with Streamlit's `AppTest` against a live
Elasticsearch and the real models — executed, not merely imported. Internal
links and page registrations are checked in the same script:
`uv run --group demo-app python demo/check_demo.py`.

**Performance is no longer a constraint.** The demo is destined for a CPU-only
server, so `llamacpp` is the default: a quantized copy served through
`llama.cpp` at roughly **12s per document**, against 60s for plain transformers.
On a GPU the same pipeline runs at about **1s per document** (0.2–0.3s of that
being extraction) under vllm, which is what `build_example_cache.py` should be
run with. The page tells a visitor which of the two they are waiting for, so a
ten-second spinner does not get read as a property of the method.

**The demo runs the current attribute model, which is not the published one.**
The 2026 retraining (`exp5.1` in `~/projects/train_NGEC_2026`) is 18 points
better on exact actor match and 19 on location than the baseline that
`ahalt/event-attribute-extractor` predates. Until it is uploaded, the demo loads
it from that local directory and falls back to the published model if it is
missing. Both the step 2 page and the health panel say which one is running —
a reviewer comparing the demo against the paper needs to know. See "Swapping the
attribute model" below; the prompt format is the part that will bite.

**Example cache is built.** `data/example_runs.json`, 11 entries, 457 KB. It is
now a consistency device more than a speed one — a live run is fast enough to
serve directly — but it keeps every visitor seeing the same draw from a model
that samples at temperature 0.5.

**Elasticsearch survives a reboot; `llama-server` does not, yet.** The container
was created with `restart=no`, which is why a reboot left it stopped; it is now
`unless-stopped` and that is **applied** on this machine. The systemd user unit
for `llama-server` in `demo/deploy/` is **written but not installed** — installing
it is four commands (see `demo/deploy/README.md`, and do not miss
`loginctl enable-linger`). Until someone runs them, `llama-server` is still a
backgrounded shell command that dies with its session, which is exactly how this
was found in the first place.

**A missing service now announces itself.** Every page opens with a banner
naming what is down, which steps are unavailable because of it, and how to
bring it back; the sidebar status panel opens itself rather than staying
collapsed. Previously a stopped service looked like a pipeline that simply did
less.

### Outstanding, roughly in priority order

1. **The ECAV-vs-LLM comparison table is deliberately empty.** `pages/vs_llm.py`
   has a placeholder saying the numbers have not been run. Do not fill it with
   anything but real measurements. Running the LLM arm of the ECAV evaluation is
   the single highest-value thing for the paper.

2. **The temporal-status classifier is not built.** `pages/echoes.py` currently
   shows honest current behaviour plus two diagnostics computed *on the page*,
   both explicitly labelled as not part of the pipeline. If the classifier gets
   built, that page turns a concession into a contribution.

3. **A validation dashboard was scoped and never started.** ECAV, the held-out
   attribute numbers, and the classifier holdout table, browsable with per-class
   detail. The most obvious next page. Note that one of the three arms already
   has real numbers to show: `setup/train_classifiers/codebook_llm/data/holdout_metrics.json`
   carries per-class precision, recall, F1, average precision, threshold and base
   rate for all 16 event types and 55 modes, from the 500-document holdout. The
   attribute and ECAV arms do not, and the page should say so rather than leave
   a reader to assume the blank cells are zeros.

4. **The demo classifier still misfires, but it no longer shows.** It fires
   REJECT alongside PROTEST on the Paris example. With the published model that
   produced a visible bad record — `actor: ["N/A"]` and an `anchor_quote` of
   "All rejections and refusals.", the codebook definition rather than a quote
   from the article. The retrained model returns `[]` for the event type that is
   not there, so the spurious record is dropped before it reaches the page. On
   the eleven curated examples that shows up as records falling from 41 to 27,
   and the recipient slot coming back "N/A" on 37% of records rather than 54%.
   Those two are the defensible differences; see the correction below for one
   that was not.

   That is a real improvement but it is also a **mask**: the classifier is still
   wrong, and the extraction model is now covering for it. Two things keep it
   visible — the step 1 page shows the raw classifier scores, and the step 2 and
   end-to-end pages now list every type-mode pair the model declined, which is
   exactly the set the classifier got wrong. Anyone reporting classifier quality
   should read it there and not infer it from the coded records.

   **Correction, and a caution about this kind of number.** An earlier version of
   this section claimed the retrained model fixed the `anchor_quote` failure —
   "27/27 verbatim against 38/41". That was a measurement bug, not a finding: the
   three "failures" were verbatim quotes whose apostrophes were curly where the
   source text's were straight. Measured properly, **both models are 41/41 and
   27/27** — the failure simply does not occur on these eleven documents, for
   either model. The same bug was live in `theme._find_span`, so the pages were
   accusing the model of paraphrasing on three spans it had copied correctly;
   `_normalise_typography` now folds quote and dash variants (one character to
   one character only, so span offsets survive). `demo/check_extractions.py`
   shares that function, and exists so these numbers can be recomputed instead of
   taken on trust.

   Worked example, the Yemen story a reader tried: the classifier fires RETREAT,
   THREATEN and ASSAULT, with four modes. `stories_to_events` makes five records;
   the attribute model extracts from one or two of them and declines the rest.
   Before this was surfaced the page showed two or three apparently identical
   ASSAULT records and no sign of the three dropped ones.

5. **One event, several records.** A type detected under several modes produces
   one record per mode, each extracted separately, and when the modes describe
   the same underlying event those records are identical apart from `event_mode`.
   The decision taken was to **label the mode and not merge** — the duplication
   is a real cost that anyone counting events in a corpus has to deal with, so
   the demo names it (`ui.repeated_modes_note`) rather than tidying it away. If
   the ontology intends one record per type rather than per type-mode pair, that
   is a pipeline change and an author's call, not a demo one.

6. **The published Hugging Face model was out of date.** Addressed: `exp5.1` is
   now `AttributeModel`'s default (`ahalt/qwen3-event-extraction-exp5.1`, see
   `setup/hf_release/` for the upload process), `KNOWN_PROMPT_FORMATS` keys on
   the published name, and a fresh install now gets what the demo shows. On
   this box the demo still prefers the local checkpoint directory over
   downloading — same weights, just already on disk.

### Environment notes for whoever picks this up

- **Run everything with `env -u LD_LIBRARY_PATH`** on this machine. A stale
  system CUDA shadows the PyTorch wheels and produces
  `undefined symbol: __nvJitLinkGetErrorLogSize_12_9`.
- **Launch the app with `uv run`, never a bare `streamlit`.** The `streamlit` on
  this machine's `PATH` is Anaconda's, on Python 3.9, and the package needs 3.10+
  — it fails several imports deep on a `str | None` annotation, which does not
  look like a version problem. `app.py` now checks the version first and says so.
- **vllm holds the GPU for the life of the process** — about 22 GB of the 4090,
  and 24s to load with a warm `torch.compile` cache (52s cold). Two Streamlit
  processes therefore cannot both run: the second dies with
  `Free memory on device cuda:0 ... is less than desired GPU memory utilization`.
  Check `nvidia-smi` before starting a second one.
- **A killed vllm parent can leave `VLLM::EngineCore` behind** still holding the
  GPU. If a load fails for memory, look for orphans in
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before believing
  the number.
- **A standalone script that builds an `AttributeModel` needs an
  `if __name__ == "__main__":` guard.** vllm's V1 engine spawns EngineCore in a
  subprocess that re-imports the launching module; without the guard the script
  re-runs itself, tries to construct a second `LLM`, and hangs with the GPU idle.
  Streamlit is not affected (the page is not `__main__`).
- **This box is AVX2-only** (i9-12900K, 24 threads). No AVX-512, no AMX, no
  native bf16. That shaped every CPU performance conclusion below.
- **llama.cpp artefacts live in `~/ngec-llamacpp/`** — `attr-q8.gguf` (the one
  the CPU path uses), plus `attr-f16.gguf` and `attr-q4km.gguf` for comparison,
  and the built binaries in `bin/`. They were moved out of the session scratchpad
  because that is session-scoped and gets cleaned.
- **Do not run heavy jobs concurrently.** Several early measurements were wrong
  because a corpus build, a page-test sweep, and a profiling run were all fighting
  over 24 cores (load average hit 47). Check `uptime` before timing anything.
- `test_vllm_cpu` in `tests/backend_and_device/` **fails on unmodified code** —
  it is the vllm-on-CPU limitation itself, not a regression. Everything else
  passes: 61 passed, 6 skipped.
- **`test_attribute_model_minimal_input` is flaky**, and looks exactly like a
  regression when it fires. It asserts an exact attribute dict, but the model
  samples at temperature 0.5, so it fails on things like
  `'a group of Hindu nationalists'` against `'A group of Hindu nationalists'` —
  one capital letter. Seen once in five runs on 2026-08-06; re-run before
  believing it. Making it robust means comparing case-insensitively, or seeding
  the sampler, and is worth doing.

### Running it

Elasticsearch and (on a CPU host) `llama-server` should be under the units in
`demo/deploy/` rather than started by hand. What is left is the app:

```shell
cd ~/projects/NGEC-2025/demo
env -u LD_LIBRARY_PATH uv run --group demo-app streamlit run app.py \
    --server.port 8577 --server.address 0.0.0.0
```

That uses the default `vllm` backend on the GPU. On a machine without one, add
`NGEC_DEMO_BACKEND=llamacpp` and make sure `llama-server` is up.

Streamlit also advertises an **External URL** on the public IP. Whether that is
actually reachable depends on the router, but an unauthenticated app that runs
models on visitor-supplied text should not be internet-facing — bind to the LAN
address or put it behind auth before any public deployment.

### Checks worth re-running after changes

None of these need a human to eyeball the app. The first two are
`demo/check_demo.py`:

```shell
env -u LD_LIBRARY_PATH uv run --group demo-app python demo/check_demo.py
env -u LD_LIBRARY_PATH uv run --group demo-app python demo/check_demo.py --links
```

- **Every page executes:** each `pages/*.py` is driven through
  `streamlit.testing.v1.AppTest` and `at.exception` must be empty. `home.py` runs
  via `app.py` instead — `st.page_link` needs the `st.navigation` context and
  raises `KeyError: 'url_pathname'` standalone, which is a harness artefact
  rather than a bug. Pages that ran degraded are flagged, because a full pass
  against dead services proves less than it looks.
- **Internal links resolve:** every `href=` and `st.page_link` path in `pages/`
  matches a `url_path` or page file declared in `app.py`, and every file in
  `pages/` is registered. `--links` does this alone, in about a second and
  without loading a model — worth running after any page rename.
- **Colour still passes:** if `SPAN_COLORS` changes, re-run the palette
  validator on the four `hue` values **plus the accent `#F0521E`** with
  `--pairs all --surface "#F2F2F0"` (see The visual theme, below). Do not trust
  your eye — the first palette failed and looked fine.

## The performance problem, measured

### The short version

The demo defaults to `NGEC_DEMO_BACKEND=llamacpp`, because it is destined for a
CPU-only server. `vllm` is one environment variable away and worth about 10x on
the development machine's RTX 4090. Over the same three curated examples, live
(not cached) runs of the whole pipeline:

| | extraction | whole pipeline |
|---|---:|---:|
| vllm, RTX 4090 | **0.2–0.3s** | **~1s** |
| llama.cpp Q8_0, CPU | ~9s | 12.2s |
| transformers fp32, CPU | ~50s | 62.2s |

The vllm model takes 24s to load with a warm `torch.compile` cache (52s cold).
Its prefix caching is doing real work: one document costs one `generate()` call
per detected event type — four on average — and those calls share a ~1,300-token
document prefix.

**Reserve 20% of the card, not the package default of 80%.** vllm takes its
share up front for weights plus KV cache. 80% suits a corpus run where a large
KV cache buys throughput; here the model is 0.6B (~1.2 GB in fp16) and the demo
codes one document at a time, so the rest sits idle *and* starves the sentence
encoder that the classifier pages put on the same card — which then fails with a
CUDA out-of-memory error that does not mention vllm. `NGEC_DEMO_GPU_MEMORY`
defaults to 0.2, about 4.7 GB of a 24 GB card, and the cache build measured the
same at 0.2 as at 0.8.

**Shut vllm down explicitly.** Its engine runs in a subprocess that does not
reliably die with its parent, and an orphaned `VLLM::EngineCore` holds ~20 GB.
`build_example_cache.py` now tears the client down before exiting; if something
later fails for GPU memory, check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` for an orphan
before believing the number.

At ~1s per document on a GPU the pre-coding in `build_example_cache.py` is no
longer needed for speed. It is kept because the model samples at temperature 0.5,
so without it two visitors comparing notes on the same example would see different
records.

**Everything below is the CPU story**, which is what a user without a GPU faces
and is therefore still the honest answer to "what does this cost to run". It is
kept in full because the conclusions are not obvious and were expensive to
reach.

### Where the time goes on a CPU

All figures from the development machine (i9-12900K, 24 threads, **AVX2 only** —
no AVX-512, no AMX, so no native bfloat16), on three real Guardian articles,
machine otherwise idle.

One document costs **one `generate()` call per detected event type** — 4 on
average, 6 for a heavily-classified article. Each call is a ~1,300-token prompt
and a 1–345 token JSON output. Splitting a single call:

| phase | cost | share |
|---|---|---|
| prefill (1,151 tokens) | 2.1s @ 559 tok/s | ~15% |
| decode (~185 tokens) | ~9.5s @ 19.6 tok/s | ~85% |

**Decode dominates.** That single fact ruled out the intuitive fixes.

### What did not work

| change | result |
|---|---|
| Batching (`batch_size=4`) | **1.8x slower** — 50s/doc → 90s/doc |
| Greedy instead of sampled | no change (49.1s vs 50.3s) |
| Prefix caching (what vllm buys) | bounded by the 15% prefill share |
| vllm on CPU | not available — see below |

Batching is the instructive one. On synthetic prompts with uniform output
lengths it measured 1.7x *faster*, which is why it looked obviously right. On
real extractions, outputs range from 1 to 345 tokens and a batch runs until its
longest member finishes, so short sequences sit padded while the longest decodes.
`call_llm_batch` now carries a comment saying so, because the change is very
tempting to make again.

**vllm on CPU is not available here.** The installed vllm is a CUDA build and
reports `is_cuda: True` even with devices hidden; the CPU backend needs a source
build, officially wants AVX-512, and `pyproject.toml` already declares `cpu` and
`vllm` mutually exclusive extras.

### What did work

**float32 on CPU** — 62.2s/doc → 50.3s/doc (**1.24x**), one line. The checkpoint
is bf16 and the old code passed `torch_dtype=None` on CPU, meaning "keep the
checkpoint dtype", so CPU runs were silently emulating bf16 on hardware that
cannot do it. Applied.

**llama.cpp with a quantized model** — the fix on a CPU host. Since decode is
memory-bandwidth-bound, cutting the weights from 2.4 GB (fp32) to 604 MB (Q8_0)
is close to a proportional win:

| runtime | prefill tok/s | decode tok/s | per call |
|---|---:|---:|---:|
| transformers fp32 | 559 | 19.6 | 12.6s |
| llama.cpp F16 | 605 | 53.8 | 4.58s |
| **llama.cpp Q8_0** | 510 | **95.7** | **3.09s** |
| llama.cpp Q4_K_M | 1006 | 140.1 | 2.51s |

**Q8_0 is the chosen setting**, not the fastest one. Comparing quantization
levels *within* llama.cpp (same runtime, so the only variable is quantization),
against F16 as the near-lossless reference on 12 real prompts:

| variant | valid JSON | exact match with F16 |
|---|---:|---:|
| Q8_0 | 12/12 | 5/12 |
| Q4_K_M | 12/12 | 0/12 |

Greedy decoding is chaotic — one different token cascades — so "not identical"
is not the same as "worse", and 12 prompts cannot establish a quality
difference. But the ordering is real: **more quantization, more drift**. Q8_0
keeps most of the speed at a fraction of the drift, so it is the default.
**Anyone using this for data production rather than demonstration should
validate their quantization against the held-out annotations first.**

End to end through the real pipeline: **62.2s/doc → 12.2s/doc, a 5.1x
speedup.** Building llama.cpp CPU-only took about three minutes with cmake and
gcc already present; conversion and quantization took under a minute.

### How it is wired

`AttributeModel` gains a fourth backend, `llamacpp`, which talks to a running
`llama-server` over HTTP rather than loading a model in-process. That choice
avoids making `llama-cpp-python` (a source build) an install dependency, lets
the model stay loaded across Streamlit restarts, and gets the server's prompt
cache — which matters here because the several event types extracted from one
document share a long document prefix.

```shell
llama-server -m attr-q8.gguf --port 8080 -c 8192 -t 16
NGEC_DEMO_BACKEND=llamacpp streamlit run app.py
```

The demo's health check pings the server, so a stopped `llama-server` shows up
as a named unavailability rather than a hang.

The two backends differ in more than speed, and it is worth being explicit about
which one produced a given record. `llamacpp` serves a **Q8_0 quantization**;
`vllm` serves the checkpoint. On 12 real prompts Q8_0 produced valid JSON every
time but matched the F16 reference exactly on only 5, so the CPU path is not
bit-identical to the GPU one. Neither is wrong, but a number quoted from the
demo should say which backend it came from — and `data/example_runs.json` should
be rebuilt when the backend changes, or the cached records will be from one
model while live runs come from another.

## Swapping the attribute model

`AttributeModel` used to hardcode `ahalt/event-attribute-extractor` in five
places. It now takes `model_name` (or `NGEC_ATTRIBUTE_MODEL`), which may be a
Hugging Face name or a local directory, and defaults to
`ahalt/qwen3-event-extraction-exp5.1` — that is also what let the demo run the
2026 retraining before it was uploaded, by pointing the same argument at a local
checkpoint directory instead.

**The prompt format is the trap.** The two models were trained on different
prompts, and prompting one in the other's format does not raise: it returns
valid JSON with worse spans. The differences are small enough to look like
formatting noise —

| | legacy (`ahalt/event-attribute-extractor`) | v5 (`ahalt/qwen3-event-extraction-exp5.1`, 2026 retraining) |
|---|---|---|
| system prompt | terse, output format only | output format plus extraction rules |
| document | `### Document:\n\n{doc}` | `## Document: {doc}` |
| event | `### Event: **TYPE**: …` then `### Specific Sub-Event:` and `### Special Instructions:` on their own lines | `## Event Type:` followed by the whole definition inline |
| closing line | "Extract the attributes of the given event in JSON format." | none |
| max output tokens | 1024 | 2048 |

— so they are not two independent settings. `KNOWN_PROMPT_FORMATS` maps a model
name (or a local directory's basename) to its format, and an unrecognised model
falls back to `legacy` **with a warning**, because the alternative is silently
evaluating a model in a format it never saw. The v5 strings are transcribed from
`eval_unified.py` in `train_NGEC_2026`, which is the script that produced that
model's reported numbers; they are part of the measurement and should not be
edited for style.

Under the `llamacpp` backend the weights come from whatever GGUF `llama-server`
was started with, while the prompt format comes from the Python side's model
name. Those can disagree, so the health check reads `/v1/models` from the server
and compares; a mismatch is reported as an unavailability rather than run
anyway. Re-quantize when the model changes:

```shell
MODEL=~/projects/train_NGEC_2026/qwen3-event-extraction-exp5.1
PYTHONPATH=gguf-py python convert_hf_to_gguf.py "$MODEL" \
    --outfile ~/ngec-llamacpp/attr-exp5.1-bf16.gguf --outtype bf16
~/ngec-llamacpp/bin/llama-quantize ~/ngec-llamacpp/attr-exp5.1-bf16.gguf \
    ~/ngec-llamacpp/attr-exp5.1-q8.gguf Q8_0
systemctl --user restart ngec-llama-server
```

(bf16, not f16, as the intermediate — the model was trained in bf16, and an
f16 downcast is an avoidable rounding step before quantization; see
`setup/hf_release/README.md`.)

### The demo-side fix, independent of runtime

The curated examples are **pre-coded** by `build_example_cache.py` into
`data/example_runs.json`, and `ui.run_with_progress` serves those instantly; one
cached full run also serves every step page, since a page that stops early just
slices the trace. Live runs happen only for text a visitor typed.

This started as a fix for a 10s/doc attribute model, and on the GPU it is no
longer needed for that. It stays for a second reason that has not gone away: the
model samples at temperature 0.5, so the same example coded twice gives two
slightly different records. Pre-coding means the panel a response letter cites
is the panel a reviewer sees.

## Open questions for the author

These were put to the author and are still unanswered. Do not guess at them; the
demo is currently built so that each remains a live choice rather than being
quietly decided by default.

1. **The LLM comparison: live or precomputed?** Live needs `ANTHROPIC_API_KEY` on
   the host and rate-limiting if the URL is public. The page already handles both
   — it runs live when a key is present and explains itself when not — so this is
   a deployment decision, not a code one. The related question is whether the
   **ECAV arm gets run** so the empty table can be filled.
2. **The temporal-status classifier: build it or not?** Build it and the echoes
   page becomes a contribution; leave it and the page stays an honest account of
   a gap. Either is defensible; the page is written so that it does not
   over-claim under either outcome.
3. **A hosted, read-only Wikipedia index.** The single largest adoption barrier
   in the whole system. Publishing the endpoint the demo already uses would
   change the setup-cost story more than any documentation change, and would make
   Table 1's "edit a category mapping file" a true description of total effort
   for a large class of users. Real ops cost, so it is the author's call.
4. **Hosting hardware.** A GPU host makes this a non-question: vllm gives ~1s per
   document and there is no second service to supervise. The llama.cpp route
   works on plain AVX2 at ~12s per document, so a CPU host is viable but not
   free — and the GGUF must be rebuilt on (or copied to) whatever box serves it.
   The real constraint on a GPU host is that vllm holds ~22 GB for the life of
   the process, so the demo cannot share a card with anything else, including a
   second copy of itself.
5. **Paper pagination.** `paper.py` pins section → PDF page numbers extracted from
   the current manuscript. Recompiling with different pagination silently breaks
   every "in the paper" link. Re-extraction belongs in the pre-submission
   checklist — the section titles are stable, the page numbers are not.

## Enhancements worth considering

- **A validation dashboard** — ECAV, the held-out attribute numbers, and the
  classifier holdout table, browsable with per-class confusion.
- **A cost calculator** — corpus size in, estimated GPU-hours and API dollars
  out, for both approaches. The per-document measurements above give the NGEC
  side of it directly.
- **A model-service split.** If concurrency ever matters, move the models behind
  a small FastAPI service and make Streamlit a thin client: models load once
  rather than once per Streamlit process, the UI can restart without a reload,
  and there is one place to rate-limit.
- **Per-page "what to look at" for reviewers** — an optional overlay mapping
  reviewer comments to panels, if the response letter ends up citing the demo
  heavily.
