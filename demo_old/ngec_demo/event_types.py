"""Event type definitions the attribute page offers, PLOVER and otherwise.

The attribute model is not a classifier over PLOVER's sixteen types. Its prompt
carries a *definition*, and the label is only a handle for it — which is what
makes a new ontology cheap in steps 2 through 5, and is the claim the "your
event types" page rests on. This module holds the two sets of definitions the
demo uses to let a visitor check that claim:

* `plover()` reads the codebook that ships with the package, so the definitions
  shown are the ones the pipeline actually prompts with.
* `UNSEEN` is the fourteen event types the ECAV validation used. They come from
  a different project with a different ontology, and only PROTEST is also a
  PLOVER label, so prompting the model with them tests exactly what a visitor
  bringing their own codebook would be doing.
"""

from __future__ import annotations

import functools

# Copied verbatim from `eval_ecav_overlap.py::EVENT_TYPE_DICT` in
# train_NGEC_2026, which produced the ECAV numbers quoted on the page. These are
# part of a reported measurement: if they are edited, the page is no longer
# showing what was evaluated.
#
# ECAV (the Event Coding for Armed Violence data) is annotated under its own
# ontology. Thirteen of these fourteen labels are not PLOVER types, and none of
# these definition strings appeared in the model's training data.
UNSEEN: dict[str, str] = {
    "ATTACK": "Any violent attack including shooting, killing, bombing, or beating. Includes attacks on property. EXCLUDE arrests.\n\n## Extraction Note: The actor is the perpetrator of the violent attack, the recipient is the person or persons who were attacked/shot/beaten.",
    "BOMBING": "Any bombing or explosion incident.\n\n## Extraction Note: The actor is the person or entity who carried out the bombing, the recipient is the victims/target of the bombing.",
    "RAID": "Any seizure, search, or storming of a location.\n\n## Extraction Note: The actor is the perpetrator of the raid, the recipient is the person or persons who were raided/searched/stormed.",
    "ARREST": "Any arrest or detention.\n\n## Extraction Note: The recipient is the person or persons who were arrested or detained.",
    "STRIKE": "Any work stoppage or labor strike.\n\n## Extraction Note: The actor is the person or persons who were on strike.",
    "PROTEST": "Any protest, demonstration, march, rally, or boycott.\n\n## Extraction Note: The actor is the person or persons who were protesting. The recipient is the group, person, or entity that the protest was directed against.",
    "CLASH": "Any clash or violent confrontation, often between two groups. This includes general descriptions of violence, fighting breaking out, etc. EXCLUDE verbal arguments.\n\n## Extraction Note: The actor is one of the entities involved in the clash. The recipient is the other entity involved in the clash.",
    "ARSON": "Any intentional burning or setting fire to property, including ballots.\n\n## Extraction Note: The actor is the person or persons who set the fire, the recipient is the property or location that was burned.",
    "KILLING": "Any killing or lethal violence.\n\n## Extraction Note: The actor is the person or persons who carried out the killing, the recipient is the person or persons who were killed.",
    "SHOOTING": "Any shooting incident involving firearms.\n\n## Extraction Note: The actor is the person or persons who carried out the shooting, the recipient is the person or persons who were shot.",
    "RIOT": "Any violent public disorder involving a crowd.\n\n## Extraction Note: The actor is the rioters or crowd involved in the disorder.",
    "INTIMIDATION": "Any threats, warnings, or attempts to intimidate. Includes police harassing pollworkers, blocking roads, etc.\n\n## Extraction Note: The actor is the person or persons making the threats, the recipient is the person or persons being intimidated.",
    "BLOCKADE": "Any blocking of access or transportation routes.\n\n## Extraction Note: The actor is the person or persons carrying out the blockade, the recipient is who or what is being blocked.",
    "KIDNAPPING": "Any abduction or taking of hostages.\n\n## Extraction Note: The actor is the person or persons carrying out the kidnapping, the recipient is the person or persons who were kidnapped.",
}

# Written for this page rather than taken from any ontology: a category a
# political scientist might plausibly invent, which is neither PLOVER's nor
# ECAV's. It is the default in the editor so that the first thing a visitor sees
# is a definition nobody has ever trained or evaluated a model on.
INVENTED_LABEL = "ELECTORAL_VIOLENCE"
INVENTED_DEFINITION = (
    "Physical violence or intimidation directed at candidates, voters, poll workers, "
    "party agents, or election infrastructure, occurring in the context of a campaign, "
    "voting, counting, or the announcement of results."
    "\n\n## Extraction Note: The actor is whoever carried out the violence or "
    "intimidation; the recipient is the candidate, voter, official, or facility it was "
    "directed at."
)


@functools.lru_cache(maxsize=1)
def plover() -> dict[str, str]:
    """PLOVER's event types and definitions, from the codebook the pipeline uses.

    Read out of the same CSV `AttributeModel` loads, so the definition shown on
    the page is the definition the pipeline prompts with. Modes are ignored
    here: the page is about the event type, and `event_def` is constant across a
    type's mode rows.
    """
    from ngec.attribute_model import _load_event_definitions

    frame = _load_event_definitions()
    out: dict[str, str] = {}
    for event, event_def in zip(frame["event"], frame["event_def"]):
        out.setdefault(str(event), str(event_def))
    return dict(sorted(out.items()))


def is_plover(label: str) -> bool:
    """Is this label one of PLOVER's own event types?

    Compared on the label alone, which is the honest comparison: a visitor who
    reuses the name PROTEST for a different concept still gets a label the model
    saw thousands of times in training, and the page should say so.
    """
    return label.strip().upper() in plover()
