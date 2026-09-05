"""Inputs the demo opens with, so no page ever starts on an empty box.

Three full documents drive the main page and the end-to-end run; the small
per-step inputs below are the example buttons on the individual step pages.
The texts are written for the demo rather than copied from a wire service, so
they can be redistributed, but they follow the register and length of the Voice
of America stories the classifiers were trained on.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    key: str
    title: str
    pub_date: str
    text: str


DOCUMENTS: list[Document] = [
    Document(
        key="paris_protest",
        title="Paris protest",
        pub_date="2023-03-15",
        text=(
            "Thousands of protesters gathered in Paris on Tuesday to demonstrate against the "
            "government's proposed pension reforms, which would raise the retirement age by "
            "two years. President Emmanuel Macron said the reforms would proceed despite the "
            "demonstrations. Police used tear gas to disperse a crowd that had blocked traffic "
            "near the Place de la Republique. Union leaders said they would call further "
            "strikes if the government did not withdraw the bill."
        ),
    ),
    Document(
        key="ceasefire",
        title="Ceasefire in Cairo",
        pub_date="2024-06-11",
        text=(
            "Officials from Ethiopia and Eritrea signed a ceasefire agreement in Cairo on "
            "Monday, ending three weeks of fighting along their shared border. The agreement "
            "was mediated by Egypt and commits both sides to withdraw heavy weapons within "
            "thirty days. Ethiopian Prime Minister Abiy Ahmed described the agreement as a "
            "first step. A spokesperson for the Eritrean foreign ministry said monitors would "
            "be admitted to the border area next week."
        ),
    ),
    Document(
        key="multi_event",
        title="Arrests in Nairobi",
        pub_date="2024-02-20",
        text=(
            "Police in Nairobi arrested at least forty people on Saturday during a "
            "demonstration against a proposed finance bill. Protesters had gathered outside "
            "parliament since early morning, carrying placards and chanting. The interior "
            "ministry said in a statement that the arrests were made to protect public order "
            "and that those detained would appear in court on Monday. Opposition leaders "
            "condemned the arrests and called for a second demonstration next week."
        ),
    ),
]

DOCUMENTS_BY_KEY = {d.key: d for d in DOCUMENTS}


# --- per-step example inputs -------------------------------------------------

# Step 1 (classification) and step 2 (attributes): one or two sentences, which
# is what the classifier and the attribute model actually see in production.
SHORT_TEXTS: list[tuple[str, str]] = [
    ("Protest", "Thousands of protesters blocked traffic in central Paris on Tuesday to "
                "demonstrate against the pension reform."),
    ("Agreement", "Officials from Ethiopia and Eritrea signed a ceasefire agreement in Cairo "
                  "on Monday, ending three weeks of fighting."),
    ("Arrest", "Police in Nairobi arrested at least forty people on Saturday outside "
               "parliament."),
]

# Steps 3 and 4 (Wikipedia lookup, actor codes): a span plus the sentence it
# came from. Context is what separates "Macron the president" from a namesake.
ENTITY_SPANS: list[tuple[str, str]] = [
    ("Emmanuel Macron",
     "President Emmanuel Macron said the pension reforms would proceed despite the "
     "demonstrations in Paris."),
    ("Abiy Ahmed",
     "Ethiopian Prime Minister Abiy Ahmed described the ceasefire agreement as a first step."),
    ("the interior ministry",
     "The interior ministry said the arrests in Nairobi were made to protect public order."),
]

# Step 5 (dates and places): phrases the attribute model returns, resolved
# against a publication date.
DATE_PHRASES: list[tuple[str, str]] = [
    ("Tuesday", "2023-03-15"),
    ("three weeks ago", "2024-06-11"),
    ("since early 2015", "2024-02-20"),
]
