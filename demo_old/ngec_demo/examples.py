"""Curated documents used throughout the demo.

Every page opens with one of these already coded, so a visitor never faces an
empty text box. Several are chosen because the pipeline handles them *badly* —
those carry `honest=True` and are surfaced on the "Where it breaks" page. A demo
that only shows its best cases is not evidence.

The texts are written for the demo rather than copied from a wire service, so
they can be redistributed, but they follow the register and length of the Voice
of America articles the demo classifiers were trained on.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    key: str
    title: str
    blurb: str
    text: str
    pub_date: str
    # Which demo pages offer this example.
    pages: tuple[str, ...] = ()
    # What the visitor should look at. Rendered as the "what to notice" note.
    notice: str = ""
    # True when the example is included because the pipeline gets it wrong.
    honest: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


EXAMPLES: list[Example] = [
    Example(
        key="paris_protest",
        title="Protest with a named head of state",
        blurb="The canonical case: one clear event, a named actor, a resolvable city and date.",
        pub_date="2023-03-15",
        pages=("end_to_end", "step1", "step2", "step3", "step4", "step5"),
        notice=(
            "Every step has something to show here: the classifier fires on PROTEST, the "
            "attribute model finds all four spans, “Emmanuel Macron” resolves to a Wikipedia "
            "page and through it to a country and role code, and “Tuesday” resolves backwards "
            "from the publication date."
        ),
        text=(
            "Thousands of protesters gathered in Paris on Tuesday to demonstrate against the "
            "government's proposed pension reforms, which would raise the retirement age by "
            "two years. President Emmanuel Macron said the reforms would proceed despite the "
            "demonstrations. Police used tear gas to disperse a crowd that had blocked traffic "
            "near the Place de la Republique. Union leaders said they would call further "
            "strikes if the government did not withdraw the bill."
        ),
    ),
    Example(
        key="ceasefire",
        title="Agreement between two states",
        blurb="Two named state actors and a negotiated outcome — an AGREE event with a clear recipient.",
        pub_date="2024-06-11",
        pages=("end_to_end", "step1", "step2", "step3", "step4"),
        notice=(
            "The recipient slot matters here. Note also that the mediator (Egypt) is a third "
            "party the four-attribute schema has nowhere to put — a design limit of the "
            "ontology, not a bug in the extraction."
        ),
        text=(
            "Officials from Ethiopia and Eritrea signed a ceasefire agreement in Cairo on "
            "Monday, ending three weeks of fighting along their shared border. The agreement "
            "was mediated by Egypt and commits both sides to withdraw heavy weapons within "
            "thirty days. Ethiopian Prime Minister Abiy Ahmed described the agreement as a "
            "first step. A spokesperson for the Eritrean foreign ministry said monitors would "
            "be admitted to the border area next week."
        ),
    ),
    Example(
        key="multi_event",
        title="Several events in one story",
        blurb="One document containing an arrest, a protest, and a government statement.",
        pub_date="2024-02-20",
        pages=("end_to_end", "step2"),
        notice=(
            "This is what `explode_events` is for. The attribute model can return several "
            "events for a single document, and each becomes its own record with its own "
            "actor, recipient, date and location — the story-level record is not the unit of "
            "analysis."
        ),
        text=(
            "Police in Nairobi arrested at least forty people on Saturday during a "
            "demonstration against a proposed finance bill. Protesters had gathered outside "
            "parliament since early morning, carrying placards and chanting. The interior "
            "ministry said in a statement that the arrests were made to protect public order "
            "and that those detained would appear in court on Monday. Opposition leaders "
            "condemned the arrests and called for a second demonstration next week."
        ),
    ),
    Example(
        key="anniversary",
        title="Temporal echo: a commemoration",
        blurb="A story about a memorial service that reports a five-year-old massacre in the present tense.",
        pub_date="2024-08-12",
        pages=("end_to_end", "step5", "echoes"),
        honest=True,
        notice=(
            "There is a real, current event here (the commemoration) and a historical one (the "
            "killings). They are grammatically almost indistinguishable. Watch which one the "
            "classifier and the attribute model latch onto, and what the date resolver does "
            "with “five years ago”."
        ),
        text=(
            "Hundreds of people gathered in the town of Sinjar on Saturday to mark the fifth "
            "anniversary of the massacre in which more than a thousand villagers were killed. "
            "Survivors laid wreaths at a memorial on the edge of the town. Five years ago "
            "fighters swept through the district, killing men and abducting women and "
            "children. A regional official told the crowd that the search for mass graves "
            "would continue."
        ),
    ),
    Example(
        key="retrospective",
        title="Temporal echo: a date-displaced reference",
        blurb="A profile piece that reports a 2019 crackdown alongside this week's news.",
        pub_date="2025-01-30",
        pages=("step5", "echoes"),
        honest=True,
        notice=(
            "Unlike the commemoration, nothing here is stylistically commemorative — the "
            "historical event is reported in ordinary declarative prose. The only signal is "
            "the date, which is why the publication-date lag filter is the intervention that "
            "catches this class of case and a mode classifier is the one that catches the "
            "other."
        ),
        text=(
            "The president announced a new cabinet on Wednesday, naming three former governors "
            "to senior posts. His government has faced criticism since 2019, when security "
            "forces opened fire on demonstrators in the capital, killing at least sixty "
            "people. No officer has been prosecuted for those killings. The new interior "
            "minister said on Thursday that a review of policing would begin within a month."
        ),
    ),
    Example(
        key="coref",
        title="Coreference: surname, role, and pronoun",
        blurb="The same person referred to four different ways in five sentences.",
        pub_date="2024-09-05",
        pages=("step3", "coref"),
        honest=True,
        notice=(
            "The pipeline expands a bare surname to a full named entity found earlier in the "
            "document. It does not resolve “the president” or “he”. Both kinds of mention "
            "appear here, so you can see exactly where the heuristic stops."
        ),
        text=(
            "Turkish President Recep Tayyip Erdogan met European Union officials in Brussels on "
            "Thursday to discuss migration. Erdogan said the existing agreement needed to be "
            "renegotiated. The president warned that his government would not extend the "
            "current arrangement past December. He also criticised the pace of visa "
            "liberalisation talks. European officials said afterwards that discussions would "
            "continue."
        ),
    ),
    Example(
        key="ambiguous_place",
        title="An ambiguous place name",
        blurb="“Georgia” is a country and a US state; the geoparser has to choose.",
        pub_date="2024-05-02",
        pages=("step5",),
        notice=(
            "Geolocation is a ranking problem over a gazetteer, not a lookup. The surrounding "
            "context — Tbilisi, the parliament — is what pushes the country above the state."
        ),
        text=(
            "Demonstrators filled the streets of Tbilisi on Wednesday for a third night of "
            "protests against a proposed foreign agents law. Georgia's parliament gave the "
            "bill a second reading earlier in the day. Riot police used water cannon to clear "
            "the area outside the legislature. The opposition has called for daily protests "
            "until the bill is withdrawn."
        ),
    ),
    Example(
        key="generic_actors",
        title="Only generic actors",
        blurb="No named entities at all — nothing for Wikipedia to link, so categorisation carries the load.",
        pub_date="2024-11-18",
        pages=("step3", "step4"),
        notice=(
            "Entity linking has nothing to do here, and that is fine: step 4 still assigns "
            "category codes from the pattern file by semantic similarity. This is the case "
            "the paper's “edit a category mapping file” claim is really about."
        ),
        text=(
            "Farmers blocked three major highways on Tuesday to protest against falling grain "
            "prices. Local officials said the blockades had disrupted deliveries to nearby "
            "towns. Riot police were deployed to two of the sites but did not intervene. A "
            "spokesperson for the farmers' association said the protest would continue until "
            "the agriculture ministry agreed to meet them."
        ),
    ),
    Example(
        key="spanish",
        title="Non-English text",
        blurb="A Spanish-language story run through an English-only pipeline.",
        pub_date="2024-07-22",
        pages=("nonenglish",),
        honest=True,
        notice=(
            "Nothing in this pipeline is multilingual. This example exists so the failure is "
            "visible and specific rather than a sentence in the limitations section."
        ),
        text=(
            "Cientos de manifestantes se concentraron el martes frente al palacio presidencial "
            "en Bogota para protestar contra la reforma tributaria. La policia antidisturbios "
            "utilizo gases lacrimogenos para dispersar a la multitud. El presidente declaro "
            "que la reforma seguira adelante. Los sindicatos anunciaron una huelga general "
            "para la proxima semana."
        ),
    ),
    Example(
        key="hard_negative",
        title="A story with no event",
        blurb="Political text that reports no PLOVER event — the classifier should stay silent.",
        pub_date="2024-10-09",
        pages=("step1",),
        honest=True,
        notice=(
            "Precision is the thing to watch at this step. With sixteen binary classifiers "
            "each at its own threshold, the chance that at least one fires on innocuous "
            "political prose is much higher than any single classifier's false positive rate."
        ),
        text=(
            "The finance ministry published its quarterly economic outlook on Wednesday, "
            "revising its growth forecast for the year to 2.4 percent from 2.1 percent. The "
            "report attributed the revision to stronger than expected household consumption "
            "and a recovery in tourism. Inflation is projected to fall gradually through the "
            "end of the year. The ministry publishes the outlook four times a year."
        ),
    ),
]

BY_KEY = {e.key: e for e in EXAMPLES}


def for_page(page: str) -> list[Example]:
    """Examples offered on a given demo page, in declaration order."""
    return [e for e in EXAMPLES if page in e.pages]


def default_for(page: str) -> Example:
    got = for_page(page)
    return got[0] if got else EXAMPLES[0]


# --------------------------------------------------------------------------
# Date phrases for the resolver page
# --------------------------------------------------------------------------

# These exercise the cascade in ngec/formatter.py::_resolve_core without running
# any model. The last group is the set the resolver deliberately refuses.
DATE_PHRASES: dict[str, list[str]] = {
    "Absolute and near-absolute": [
        "March 15, 2023",
        "15 March",
        "(March 15)",
        "2023-03-15",
    ],
    "Relative to publication": [
        "Tuesday",
        "last Friday",
        "three days ago",
        "a week before last Friday",
        "the day before yesterday",
    ],
    "Ranges and spans": [
        "March 15-20",
        "over the weekend",
        "since Thursday",
        "between Monday and Wednesday",
    ],
    "Vague but resolvable": [
        "late Tuesday night",
        "early March",
        "mid-January",
        "the second week of April",
        "the first quarter",
        "the 1990s",
    ],
    "Refused by design (need world knowledge)": [
        "the anniversary of the coup",
        "last Ramadan",
        "the 2024 election",
        "three days of clashes",
    ],
}
