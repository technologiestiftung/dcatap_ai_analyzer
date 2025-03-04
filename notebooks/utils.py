import pandas as pd
import os
import re
from openai import OpenAI
from mistralai import Mistral
import instructor
from instructor import Mode
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


class MetadataAnalysis(BaseModel):
    dateninhalt: str = Field(
        ...,
        description="Analyse der Informationsqualität des Datensatzes mit spezifischen Mängeln oder Stärken",
    )
    methodik: str = Field(
        ...,
        description="Analyse der methodischen Qualität mit spezifischen Mängeln oder Stärken",
    )
    datenqualitaet: str = Field(
        ...,
        description="Analyse der Datenqualität mit spezifischen Mängeln oder Stärken",
    )
    geographie: str = Field(
        ...,
        description="Analyse der geographischen Informationsqualität mit spezifischen Mängeln oder Stärken",
    )
    tag_qualitaet: str = Field(
        ...,
        description="Analyse der Tag-Qualität mit spezifischen Mängeln oder Stärken",
    )
    referenz: str = Field(
        ...,
        description="Analyse der Referenzqualität mit spezifischen Mängeln oder Stärken",
    )

    dateninhalt_score: Literal[1, 2, 3] = Field(
        ..., description="Bewertung der Informationsqualität (1-3)"
    )
    methodik_score: Literal[1, 2, 3] = Field(
        ..., description="Bewertung der methodischen Qualität (1-3)"
    )
    datenqualitaet_score: Literal[1, 2, 3] = Field(
        ..., description="Bewertung der Datenqualität (1-3)"
    )
    geographie_score: Literal[1, 2, 3] = Field(
        ..., description="Bewertung der geographischen Informationsqualität (1-3)"
    )
    tag_qualitaet_score: Literal[1, 2, 3] = Field(
        ..., description="Bewertung der Tag-Qualität (1-3)"
    )
    referenz_score: Literal[1, 2, 3] = Field(
        ..., description="Bewertung der Referenzqualität (1-3)"
    )


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

instructor_openai_client = instructor.from_openai(openai_client)


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
model = "mistral-small-latest"
SYSTEM_MESSAGE = """"Du bist ein hilfreicher Assistent für ein Statistikamt. Du wirst gebeten, Metadaten für einen Datensatz zu analysieren. Bleibe stets wahrheitsgemäß und objektiv. Schreib nur das, was du anhand der vom Benutzer bereitgestellten Metadaten sicher feststellen kannst. Mache keine Annahmen. Schreibe einfach und klar. Schreibe immer in deutscher Sprache."""

instructor_mistral_client = instructor.from_mistral(
    client=mistral_client,
    model=model,
    mode=Mode.MISTRAL_TOOLS,
    max_tokens=4096,
)


# Using the currently most cheap but capable AI model GPT4o-Mini
def call_openai(
    prompt, modelId="gpt-4o-mini", temperature=0, max_tokens=4096, use_instructor=True
):
    try:
        if use_instructor:
            chat_response = instructor_openai_client.messages.create(
                model=modelId,
                response_model=MetadataAnalysis,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return chat_response
        else:
            completion = openai_client.chat.completions.create(
                model=modelId,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content

    except Exception as e:
        print(f"Error: {e}")
        return None


def call_mistral(prompt, model=model, use_instructor=True):
    try:
        if use_instructor:
            chat_response = instructor_mistral_client.messages.create(
                response_model=MetadataAnalysis,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return chat_response
        else:
            chat_response = mistral_client.complete(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return chat_response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None


PROMPT_ANOMALIES = """Du bist Experte für Datenqualität und sollst einen Datenkatalog überprüfen. Du erhältst eine kommagetrennte Liste von {feature} aller Datensätze im Katalog. Du sollst die Beschreibungen auf Anomalien prüfen. Hier ist die kommagetrennte Liste von {feature} der Datensätze:

<datensatz-{feature}>
{data}
</datensatz-{feature}>

Prüfe nun, ob es Anomalien oder sonstige Auffälligkeiten in den {feature} gibt. Bitte liste alle Anomalien auf, die du findest, zitiere passende Beispiele und kommentiere kurz, warum diese ein Problem sind."""


def check_text_properties(feature, data):
    """Check the text properties of the titles and descriptions of the datasets globally. Each will be provided to the LLM as a comma-separated list. The analysis will be done in one go, not per dataset."""
    return call_mistral(PROMPT_ANOMALIES.format(feature=feature, data=data))

with open('../data/few_shot_prompts.txt', 'r', encoding='utf-8') as file:
    few_shots = file.read()

PROMPT_ANALYSIS = """Du erhältst die Metadaten eines Datensatzes. Diese sollst du detailliert und genau analysieren. Die Metadaten bestehen aus einem Titel, einer Beschreibung, dem geographischem Bezug, der geographischen Granularität, einer Liste von Tags und der zuständigen Stelle. Du sollst die Metadaten analysieren und sicherstellen, dass diese aussagekräftig, vollständig und von hoher Qualität sind. 
Gib niemals die originalen Metadatenfelder wieder. Verwende stattdessen Formulierungen wie 'der Titel' oder 'die Beschreibung', anstatt deren Inhalt zu wiederholen.
Liefere ausschließlich deine kritische Bewertung der Metadaten. 

Bewerte die Metadaten anhand der folgenden Kriterien:
    1. Dateninhalt - Wie detailliert und eindeutig erklären Titel und Beschreibung, worum es in dem Datensatz geht?
    2. Methodik - Wie detailliert und eindeutig erklären Titel und Beschreibung, wie die Daten gemessen wurden und wofür? Wird die Quelle der Daten genannt und beschrieben?
    3. Datenqualität - Wie detailliert und eindeutig erklären Titel und Beschreibung, wie gut die Qualität der Daten ist? Wird erklärt, wie vollständig die Daten sind? Gibt es Änderungen in der Erhebung? 
    4. Geographie - Wie detailliert und eindeutig erklärt der Eintrag zur geographischen Granularität, wie die Daten geographisch zu verstehen sind? Wird klar, auf welche geographischen Orte oder Gebiete sich die Daten beziehen? 
    5. Tag-Qualität - Wie sinnvoll sind die verwendeten Tags? Sind sie klar und eindeutig? Ist die Anzahl der Tags angemessen?
    6. Referenz - Scheint die Angabe der veröffentlichten Stelle korrekt zu sein? Wurden Organisationsnamen ausgeschrieben und sind fehlerfrei? Gibt es im Namen der Stelle Tippfehler? Ist die Stelle eindeutig identifizierbar?

Prüfe bei jedem Punkt auch, ob es Auffälligkeiten oder Anomalien gibt und bennene diese kurz. Erwähne sie nur, wenn du dir sicher bist, dass sie auch vorliegen. Das können zum Beispiel sein: 
    - fehlende Informationen, 
    - schwer verständliche oder vieldeutige Begriffe,
    - unnötige und unklare Abkürzungen,
    - Rechtschreibfehler oder uneinheitliche Schreibweisen,
    - ungewöhnliche Zeichen, Symbole oder Formatierungen,
    - Tags, Markdown, HTML-Code oder andere Programmierzeichen,
    - unklare oder unvollständige Sätze oder Absätze,
    - keine geschlechtergerechte Sprache.

Vergib dann eine Bewertung für jedes der sechs Kriterien. 
Verwende die Skala von 1-3, wobei 1 die schlechteste und 3 die beste Bewertung ist.
Sei klar, hart und kritisch bei deiner Bewertung. 
Hier ist die Bewertungsskala:

1 Punkt - Keine oder wenige Informationen zu diesem Kriterium.
2 Punkte - Mittelmässige bis gute Informationen, einige Informationen sind vorhanden, einiges fehlt.
3 Punkte - Exzellente Informationen, alles ist sehr klar, vollständig und detailliert.


{few_shots}


Analysiere und bewerte jetzt die Metadaten des Datensatzes.

Hier sind die Metadaten des Datensatzes, den du analysieren sollst:
---------------------------------------------------------------------

Titel: {title}
Beschreibung: {description}
Geographischer Bezug: {geo_reference}
Geographischen Granularität: {geo_granularity}
Tags: {tags}
Zuständige Stelle: {author}
"""


def do_full_analysis(data, use_openai=True):
    """Check the titles and descriptions of each dataset individually."""
    if use_openai:
        return call_openai(
            PROMPT_ANALYSIS.format(
                title=data["title"],
                description=data["notes"],
                geo_reference=data["geographical_coverage"],
                geo_granularity=data["geographical_granularity"],
                tags=", ".join(data["formatted_tags"]),
                author=data["author"],
            )
        )
    else:
        return call_mistral(
            PROMPT_ANALYSIS.format(
                title=data["title"],
                description=data["notes"],
                geo_reference=data["geographical_coverage"],
                geo_granularity=data["geographical_granularity"],
                tags=", ".join(data["formatted_tags"]),
                author=data["author"],
            )
        )


def parse_instructor_results(results):
    """Parse the LLM instructor response given the MetadataAnalysis class. Extract the scores and the qualitative analysis.

    Args:
        results (MetadataAnalysis): The response from the LLM, already parsed into a MetadataAnalysis object.

    Returns:
        pd.DataFrame: A DataFrame with the scores and the qualitative analysis.
    """
    data_rows = []

    # Process each MetadataAnalysis object
    for results in results:
        data_rows.append(
            {
                "dateninhalt": results.dateninhalt,
                "dateninhalt_score": results.dateninhalt_score,
                "methodik": results.methodik,
                "methodik_score": results.methodik_score,
                "datenqualitaet": results.datenqualitaet,
                "datenqualitaet_score": results.datenqualitaet_score,
                "geographie": results.geographie,
                "geographie_score": results.geographie_score,
                "tag_qualitaet": results.tag_qualitaet,
                "tag_qualitaet_score": results.tag_qualitaet_score,
                "referenz": results.referenz,
                "referenz_score": results.referenz_score,
            }
        )

    # Create DataFrame from all results
    df = pd.DataFrame(data_rows)

    return df


def parse_analysis_results(results):
    """Parse the LLM response. Extract the scores and the qualitative analysis.

    Args:
        results (str): The response from the LLM.

    Returns:
        pd.DataFrame: A DataFrame with the scores and the qualitative analysis.

    """
    content = re.findall(r"<dateninhalt>(.*?)</dateninhalt>", results, re.DOTALL)[
        0
    ].strip()
    content_score = re.findall(
        r"<dateninhalt-score>(.*?)</dateninhalt-score>", results, re.DOTALL
    )[0].strip()
    context = re.findall(
        r"<methodik>(.*?)</methodik>",
        results,
        re.DOTALL,
    )[0].strip()
    context_score = re.findall(
        r"<methodik-score>(.*?)</methodik-score>",
        results,
        re.DOTALL,
    )[0].strip()
    quality = re.findall(r"<datenqualität>(.*?)</datenqualität>", results, re.DOTALL)[
        0
    ].strip()
    quality_score = re.findall(
        r"<datenqualität-score>(.*?)</datenqualität-score>",
        results,
        re.DOTALL,
    )[0].strip()
    spacial = re.findall(
        r"<geographie>(.*?)</geographie>",
        results,
        re.DOTALL,
    )[0].strip()
    spacial_score = re.findall(
        r"<geographie-score>(.*?)</geographie-score>",
        results,
        re.DOTALL,
    )[0].strip()
    tmp = pd.DataFrame(
        (
            content,
            content_score,
            context,
            context_score,
            quality,
            quality_score,
            spacial,
            spacial_score,
        )
    ).T
    tmp.columns = [
        "content",
        "content_score",
        "context",
        "context_score",
        "quality",
        "quality_score",
        "spacial",
        "spacial_score",
    ]
    return tmp


# Data derived from the DCAT-AP.de specification here:
# https://www.dcat-ap.de/def/dcatde/2.0/spec/#klasse-katalog
# Updated data derived from the DCAT-AP.de specification for the Dataset class

# Explanation:
#
# Obligation is mapped directly from the German column "Verbindlichkeit":
#     Pflicht → "M" (Mandatory)
#     Empfohlen → "R" (Recommended)
#     Optional → "O" (Optional)
# Cardinality (the fifth element in each tuple) matches the [x..y] in the table exactly, for example:
#     [1..*] → "1..*"
#     [*] → "*"
#     [0..1] → "0..1"

DCAT_CLASS_DATASET = [
    # ----------------------------------------------------------------
    # MANDATORY (Pflicht)
    # ----------------------------------------------------------------
    ("Title", "dct:title", "rdfs:Literal", "M", "1..*"),
    ("Description", "dct:description", "rdfs:Literal", "M", "1..*"),
    # ----------------------------------------------------------------
    # RECOMMENDED (Empfohlen)
    # ----------------------------------------------------------------
    ("Keyword", "dcat:keyword", "rdfs:Literal", "R", "*"),
    (
        "Political Geocoding Level URI",
        "dcatde:politicalGeocodingLevelURI",
        "rdfs:Resource",
        "R",
        "*",
    ),
    (
        "Political Geocoding URI",
        "dcatde:politicalGeocodingURI",
        "rdfs:Resource",
        "R",
        "*",
    ),
    ("Theme (Category)", "dcat:theme", "skos:Concept", "R", "*"),
    ("Contact Point", "dcat:contactPoint", "vcard:Kind", "R", "*"),
    ("Availability", "dcatap:availability", "skos:Concept", "R", "0..1"),
    ("Spatial Coverage", "dct:spatial", "dct:Location", "R", "*"),
    ("Temporal Coverage", "dct:temporal", "dct:PeriodOfTime", "R", "*"),
    ("Distribution", "dcat:distribution", "dcat:Distribution", "R", "*"),
    ("Publisher", "dct:publisher", "foaf:Agent", "R", "0..1"),
    # ----------------------------------------------------------------
    # OPTIONAL (Optional)
    # ----------------------------------------------------------------
    ("Contributor ID", "dcatde:contributorID", "rdfs:Resource", "O", "*"),
    ("Geocoding Description", "dcatde:geocodingDescription", "rdfs:Literal", "O", "*"),
    ("Identifier", "dct:identifier", "rdfs:Literal", "O", "*"),
    ("Other ID", "adms:identifier", "adms:Identifier", "O", "*"),
    ("Issued (Publication Date)", "dct:issued", "rdfs:Literal", "O", "0..1"),
    ("Modified (Update Date)", "dct:modified", "rdfs:Literal", "O", "0..1"),
    ("Version Info", "owl:versionInfo", "rdfs:Literal", "O", "0..1"),
    ("Version Notes", "adms:versionNotes", "rdfs:Literal", "O", "*"),
    ("Legal Basis", "dcatde:legalBasis", "rdfs:Literal", "O", "*"),
    ("Related Resource", "dct:relation", "rdfs:Resource", "O", "*"),
    ("Landing Page", "dcat:landingPage", "foaf:Document", "O", "*"),
    ("Documentation Page", "foaf:page", "foaf:Document", "O", "*"),
    ("Language", "dct:language", "dct:LinguisticSystem", "O", "*"),
    ("Conforms To (Standard)", "dct:conformsTo", "dct:Standard", "O", "*"),
    ("Access Rights Level", "dct:accessRights", "dct:RightsStatement", "O", "0..1"),
    ("Provenance", "dct:provenance", "dct:ProvenanceStatement", "O", "*"),
    ("Accrual Periodicity", "dct:accrualPeriodicity", "dct:Frequency", "O", "0..1"),
    ("Quality Process URI", "dcatde:qualityProcessURI", "rdfs:Resource", "O", "0..1"),
    ("Dataset Type", "dct:type", "skos:Concept", "O", "0..1"),
    ("Was Generated By", "prov:wasGeneratedBy", "prov:Activity", "O", "*"),
    (
        "Spatial Resolution in Meters",
        "dcat:spatialResolutionInMeters",
        "rdfs:Literal",
        "O",
        "*",
    ),
    ("Temporal Resolution", "dcat:temporalResolution", "rdfs:Literal", "O", "*"),
    (
        "Temporal Granularity (DEPRECATED)",
        "dcat:granularity",
        "skos:Concept",
        "O",
        "0..1",
    ),
    (
        "Qualified Attribution",
        "prov:qualifiedAttribution",
        "prov:Attribution",
        "O",
        "*",
    ),
    ("Qualified Relation", "dcat:qualifiedRelation", "dcat:Relationship", "O", "*"),
    ("Is Referenced By", "dct:isReferencedBy", "rdfs:Resource", "O", "*"),
    ("References", "dct:references", "rdfs:Resource", "O", "*"),
    ("Source Dataset", "dct:source", "dcat:Dataset", "O", "*"),
    ("Has Version", "dct:hasVersion", "dcat:Dataset", "O", "*"),
    ("Is Version Of", "dct:isVersionOf", "dcat:Dataset", "O", "*"),
    ("Sample Distribution", "adms:sample", "dcat:Distribution", "O", "*"),
    ("Creator (Author)", "dct:creator", "foaf:Agent", "O", "*"),
    ("Contributor (Editor)", "dct:contributor", "foaf:Agent", "O", "*"),
    ("Originator", "dcatde:originator", "foaf:Agent", "O", "*"),
    ("Maintainer", "dcatde:maintainer", "foaf:Agent", "O", "*"),
]


# https://www.dcat-ap.de/def/dcatde/2.0/spec/#klasse-distribution
DCAT_CLASS_DISTRIBUTION = data = [
    ("access URL", "dcat:accessURL", "rdfs:Resource", "M", "1..*"),
    ("title", "dct:title", "rdfs:Literal", "R", "*"),
    ("update/ modification date", "dct:modified", "rdfs:Literal", "R", "0..1"),
    ("license", "dct:license", "dct:LicenseDocument", "R", "0..1"),
    ("format", "dct:format", "dct:MediaType", "R", "0..1"),
    ("availability", "dcatap:availability", "skos:Concept", "R", "0..1"),
    (
        "planned availability (DEPRECATED)",
        "dcatde:plannedAvailability",
        "rdfs:Resource",
        "R",
        "0..1",
    ),
    (
        "license attribution by text",
        "dcatde:licenseAttributionByText",
        "rdfs:Literal",
        "O",
        "*",
    ),
    ("description", "dct:description", "rdfs:Literal", "O", "*"),
    ("byte size", "dcat:byteSize", "rdfs:Literal", "O", "0..1"),
    ("release date", "dct:issued", "rdfs:Literal", "O", "0..1"),
    ("download URL", "dcat:downloadURL", "rdfs:Resource", "O", "*"),
    ("language", "dct:language", "dct:LinguisticSystem", "O", "*"),
    ("documentation", "foaf:page", "foaf:Document", "O", "*"),
    ("rights", "dct:rights", "dct:RightsStatement", "O", "0..1"),
    ("conforms to", "dct:conformsTo", "dct:Standard", "O", "*"),
    ("media type", "dcat:mediaType", "dct:MediaType", "O", "0..1"),
    ("compression format", "dcat:compressFormat", "dct:MediaType", "O", "0..1"),
    ("package format", "dcat:packageFormat", "dct:MediaType", "O", "0..1"),
    ("policy", "odrl:hasPolicy", "odrl:Policy", "O", "0..1"),
    ("status", "adms:status", "skos:Concept", "O", "0..1"),
    ("access service", "dcat:accessService", "dcat:DataService", "O", "*"),
    ("checksum", "spdx:checksum", "spdx:Checksum", "O", "0..1"),
]

# Data derived from here:
# https://publications.europa.eu/resource/authority/data-theme
VOCAB_EU_THEME = [
    "http://publications.europa.eu/resource/authority/data-theme/TECH",
    "http://publications.europa.eu/resource/authority/data-theme/TRAN",
    "http://publications.europa.eu/resource/authority/data-theme/REGI",
    "http://publications.europa.eu/resource/authority/data-theme/SOCI",
    "http://publications.europa.eu/resource/authority/data-theme/AGRI",
    "http://publications.europa.eu/resource/authority/data-theme/ECON",
    "http://publications.europa.eu/resource/authority/data-theme/JUST",
    "http://publications.europa.eu/resource/authority/data-theme/OP_DATPRO",
    "http://publications.europa.eu/resource/authority/data-theme/HEAL",
    "http://publications.europa.eu/resource/authority/data-theme/INTR",
    "http://publications.europa.eu/resource/authority/data-theme/ENVI",
    "http://publications.europa.eu/resource/authority/data-theme/GOVE",
    "http://publications.europa.eu/resource/authority/data-theme/EDUC",
    "http://publications.europa.eu/resource/authority/data-theme/ENER",
]

# Data derived from here:
# https://publications.europa.eu/resource/authority/frequency
VOCAB_EU_FREQUENCY = [
    "http://publications.europa.eu/resource/authority/frequency/BIDECENNIAL",
    "http://publications.europa.eu/resource/authority/frequency/TRIDECENNIAL",
    "http://publications.europa.eu/resource/authority/frequency/BIHOURLY",
    "http://publications.europa.eu/resource/authority/frequency/TRIHOURLY",
    "http://publications.europa.eu/resource/authority/frequency/OTHER",
    "http://publications.europa.eu/resource/authority/frequency/WEEKLY",
    "http://publications.europa.eu/resource/authority/frequency/HOURLY",
    "http://publications.europa.eu/resource/authority/frequency/QUADRENNIAL",
    "http://publications.europa.eu/resource/authority/frequency/QUINQUENNIAL",
    "http://publications.europa.eu/resource/authority/frequency/DECENNIAL",
    "http://publications.europa.eu/resource/authority/frequency/WEEKLY_2",
    "http://publications.europa.eu/resource/authority/frequency/WEEKLY_3",
    "http://publications.europa.eu/resource/authority/frequency/UNKNOWN",
    "http://publications.europa.eu/resource/authority/frequency/UPDATE_CONT",
    "http://publications.europa.eu/resource/authority/frequency/QUARTERLY",
    "http://publications.europa.eu/resource/authority/frequency/TRIENNIAL",
    "http://publications.europa.eu/resource/authority/frequency/NEVER",
    "http://publications.europa.eu/resource/authority/frequency/OP_DATPRO",
    "http://publications.europa.eu/resource/authority/frequency/MONTHLY_2",
    "http://publications.europa.eu/resource/authority/frequency/MONTHLY_3",
    "http://publications.europa.eu/resource/authority/frequency/IRREG",
    "http://publications.europa.eu/resource/authority/frequency/MONTHLY",
    "http://publications.europa.eu/resource/authority/frequency/DAILY",
    "http://publications.europa.eu/resource/authority/frequency/DAILY_2",
    "http://publications.europa.eu/resource/authority/frequency/BIWEEKLY",
    "http://publications.europa.eu/resource/authority/frequency/CONT",
    "http://publications.europa.eu/resource/authority/frequency/BIENNIAL",
    "http://publications.europa.eu/resource/authority/frequency/BIMONTHLY",
    "http://publications.europa.eu/resource/authority/frequency/ANNUAL_2",
    "http://publications.europa.eu/resource/authority/frequency/ANNUAL_3",
    "http://publications.europa.eu/resource/authority/frequency/ANNUAL",
    "http://publications.europa.eu/resource/authority/frequency/NOT_PLANNED",
    "http://publications.europa.eu/resource/authority/frequency/AS_NEEDED",
]

# These are values for «hidden» nulls. These values satisfy the DCAT standard, because they are not empty. However, they are not useful or meaningful.
HIDDEN_NULLS = (
    "",
    "null",
    "[]",
    "{}",
    "nan",
    "none",
    "ohne angabe",
    "keine angabe",
    "nichts",
)
