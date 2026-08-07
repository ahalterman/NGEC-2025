"""A frontier-LLM baseline for event attribute extraction.

This is deliberately the obvious thing: one prompt, one API call, structured
output. It is what a researcher would write instead of installing this package,
and the comparison page uses it to make that alternative concrete rather than
hypothetical.

Nothing here is tuned to lose. If the prompt can be improved, improve it — an
unfair baseline would make the comparison worthless.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import streamlit as st

MODEL = os.environ.get("NGEC_DEMO_LLM_MODEL", "claude-opus-5")

# Published per-million-token prices for the default model, used for the cost
# counter. Override if you point the demo at a different model.
PRICE_PER_MTOK_INPUT = float(os.environ.get("NGEC_DEMO_LLM_INPUT_PRICE", "5.00"))
PRICE_PER_MTOK_OUTPUT = float(os.environ.get("NGEC_DEMO_LLM_OUTPUT_PRICE", "25.00"))

PROMPT = """\
You are coding political event data from news text, following the PLOVER ontology.

Read the article and extract every political event it reports as having happened.
For each event, give:

- event_type: one of ACCUSE, AGREE, AID, ASSAULT, COERCE, CONCEDE, CONSULT,
  COOPERATE, MOBILIZE, PROTEST, REJECT, REQUEST, RETREAT, SANCTION, SUPPORT,
  THREATEN
- actor: who carried out the event
- recipient: who it was directed at, if anyone
- date: the date expression the article uses, resolved to YYYY-MM-DD against the
  publication date
- location: where it happened, as a place name
- anchor_quote: a verbatim sentence from the article that supports the extraction

Copy actor, recipient and location spans from the article rather than
paraphrasing them. Use null for a slot the article does not fill. Do not report
events that the article mentions only as historical background.

Publication date: {pub_date}

Article:
{text}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_type": {"type": "string"},
                    "actor": {"type": ["string", "null"]},
                    "recipient": {"type": ["string", "null"]},
                    "date": {"type": ["string", "null"]},
                    "location": {"type": ["string", "null"]},
                    "anchor_quote": {"type": ["string", "null"]},
                },
                "required": [
                    "event_type", "actor", "recipient",
                    "date", "location", "anchor_quote",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}


@dataclass
class LLMResult:
    events: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = MODEL
    error: str | None = None
    refused: bool = False

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * PRICE_PER_MTOK_INPUT
            + self.output_tokens / 1_000_000 * PRICE_PER_MTOK_OUTPUT
        )


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    import anthropic

    return anthropic.Anthropic()


def extract(text: str, pub_date: str, effort: str = "medium",
            nonce: int = 0) -> LLMResult:
    """One document, one API call. `nonce` only varies the cache key."""
    if not available():
        return LLMResult(error="No ANTHROPIC_API_KEY is configured on this server.")

    try:
        client = _client()
    except Exception as exc:  # noqa: BLE001
        return LLMResult(error=f"Could not create the API client: {exc}")

    request = {
        "model": MODEL,
        "max_tokens": 4096,
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        "messages": [
            {"role": "user", "content": PROMPT.format(pub_date=pub_date, text=text)}
        ],
    }

    t0 = time.time()
    try:
        # Safety classifiers can decline a request; server-side fallbacks re-run it
        # on another model rather than returning the refusal. Conflict reporting
        # is exactly the kind of benign-but-adjacent text that occasionally trips
        # them, so it is worth having here.
        try:
            response = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **request,
            )
        except TypeError:
            # Older SDK without the fallbacks parameter.
            response = client.messages.create(**request)
    except Exception as exc:  # noqa: BLE001
        return LLMResult(seconds=time.time() - t0, error=str(exc)[:300])

    elapsed = time.time() - t0
    usage = getattr(response, "usage", None)
    result = LLMResult(
        seconds=elapsed,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        model=getattr(response, "model", MODEL),
    )

    if getattr(response, "stop_reason", None) == "refusal":
        result.refused = True
        result.error = "The request was declined by the model's safety classifiers."
        return result

    text_blocks = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    if not text_blocks:
        result.error = "The model returned no text content."
        return result

    try:
        parsed = json.loads(text_blocks[0])
        result.events = parsed.get("events", [])
    except (ValueError, AttributeError) as exc:
        result.error = f"Could not parse the model's JSON: {exc}"

    return result


@st.cache_data(show_spinner=False, max_entries=64)
def cached_extract(text: str, pub_date: str, effort: str = "medium",
                   nonce: int = 0) -> LLMResult:
    return extract(text, pub_date, effort=effort, nonce=nonce)
