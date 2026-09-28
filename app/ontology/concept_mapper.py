import json
import re

from pydantic import ValidationError
from rdflib import (
    Graph,
    Literal,
    RDF,
    RDFS,
    URIRef,
)
from rdflib.namespace import SKOS

from app.config.settings import (
    Settings,
    get_settings,
)
from app.llm.client import (
    LLMServiceError,
    _call_with_one_retry,
    create_openrouter_client,
)
from app.llm.prompts import (
    CONCEPT_EXTRACTION_SYSTEM_PROMPT,
)
from app.models.ontology import (
    ExtractedConcepts,
    MappedOntologyConcept,
    OntologyContext,
)
from app.ontology.ontology_loader import (
    load_ontology,
)
from app.ontology.ontology_queries import (
    MED,
)


class ConceptMappingError(RuntimeError):
    """
    Raised when concept extraction or deterministic ontology
    mapping cannot be completed safely.
    """


CLINICAL_CONCEPT_TYPES: set[URIRef] = {
    MED.Condition,
    MED.Treatment,
    MED.Drug,
    MED.DrugClass,
    MED.PatientGroup,
    MED.Investigation,
    MED.Guideline,
}


def normalize_concept_text(
    text: str,
) -> str:
    """
    Normalize text for deterministic lexical matching.

    This performs formatting normalization only.
    It does not perform clinical reasoning.
    """
    normalized = (
        text
        .strip()
        .lower()
    )

    normalized = re.sub(
        r"[_\-]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"[^\w\s]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def _local_name(
    uri: URIRef,
) -> str:
    """
    Return the final local component of an ontology URI.
    """
    text = str(
        uri
    )

    if "/" in text:
        return text.rsplit(
            "/",
            1,
        )[-1]

    if "#" in text:
        return text.rsplit(
            "#",
            1,
        )[-1]

    return text


def _canonical_label(
    graph: Graph,
    uri: URIRef,
) -> str:
    """
    Return the canonical rdfs:label for a resource.

    Fall back to the URI local name when no label exists.
    """
    label = graph.value(
        uri,
        RDFS.label,
    )

    if isinstance(
        label,
        Literal,
    ):
        return str(
            label
        )

    return _local_name(
        uri
    )


def _is_controlled_clinical_concept(
    graph: Graph,
    uri: URIRef,
) -> bool:
    """
    Return True when a resource is typed as one of the
    controlled clinical concept classes.
    """
    concept_types = {
        object_value
        for object_value
        in graph.objects(
            uri,
            RDF.type,
        )
        if isinstance(
            object_value,
            URIRef,
        )
    }

    return bool(
        concept_types
        & CLINICAL_CONCEPT_TYPES
    )


def build_ontology_lexicon(
    graph: Graph,
) -> dict[str, URIRef]:
    """
    Build a deterministic normalized-term → ontology-URI
    lookup table.

    Terms include:

    - canonical rdfs:label
    - SKOS alternative labels
    - ontology URI local name

    Only controlled clinical concept instances are indexed.
    """
    lexicon: dict[
        str,
        URIRef,
    ] = {}

    candidate_uris: set[
        URIRef
    ] = set()

    for subject in graph.subjects(
        RDF.type,
        None,
    ):
        if not isinstance(
            subject,
            URIRef,
        ):
            continue

        if _is_controlled_clinical_concept(
            graph,
            subject,
        ):
            candidate_uris.add(
                subject
            )

    for uri in candidate_uris:
        terms: list[str] = []

        canonical = graph.value(
            uri,
            RDFS.label,
        )

        if isinstance(
            canonical,
            Literal,
        ):
            terms.append(
                str(canonical)
            )

        for alias in graph.objects(
            uri,
            SKOS.altLabel,
        ):
            if isinstance(
                alias,
                Literal,
            ):
                terms.append(
                    str(alias)
                )

        terms.append(
            _local_name(
                uri
            )
        )

        for term in terms:
            normalized = normalize_concept_text(
                term
            )

            if not normalized:
                continue

            existing_uri = lexicon.get(
                normalized
            )

            if (
                existing_uri is not None
                and existing_uri != uri
            ):
                raise ConceptMappingError(
                    (
                        "Ontology lexical collision for "
                        f"normalized term {normalized!r}."
                    )
                )

            lexicon[
                normalized
            ] = uri

    return lexicon


def map_candidate_concepts(
    *,
    candidates: list[str],
    graph: Graph,
) -> tuple[
    list[MappedOntologyConcept],
    list[str],
]:
    """
    Deterministically resolve candidate phrases against the
    ontology lexicon.

    Unknown terms remain unknown. No URI guessing occurs.
    """
    lexicon = build_ontology_lexicon(
        graph
    )

    mapped: list[
        MappedOntologyConcept
    ] = []

    unknown: list[str] = []

    seen_uris: set[str] = set()
    seen_unknown: set[str] = set()

    for candidate in candidates:
        original = (
            candidate.strip()
        )

        if not original:
            continue

        normalized = normalize_concept_text(
            original
        )

        uri = lexicon.get(
            normalized
        )

        if uri is None:
            unknown_key = normalized

            if (
                unknown_key
                and unknown_key
                not in seen_unknown
            ):
                unknown.append(
                    original
                )

                seen_unknown.add(
                    unknown_key
                )

            continue

        uri_text = str(
            uri
        )

        if uri_text in seen_uris:
            continue

        mapped.append(
            MappedOntologyConcept(
                original_text=original,
                concept_name=_local_name(
                    uri
                ),
                label=_canonical_label(
                    graph,
                    uri,
                ),
                uri=uri_text,
            )
        )

        seen_uris.add(
            uri_text
        )

    return (
        mapped,
        unknown,
    )


def find_known_concepts_in_query(
    *,
    query: str,
    graph: Graph,
) -> list[MappedOntologyConcept]:
    """
    Deterministically identify ontology labels or aliases that
    explicitly occur in the user's query.

    This complements LLM extraction and requires no additional
    LLM call.
    """
    normalized_query = normalize_concept_text(
        query
    )

    if not normalized_query:
        return []

    lexicon = build_ontology_lexicon(
        graph
    )

    matches: list[
        MappedOntologyConcept
    ] = []

    seen_uris: set[str] = set()

    items = sorted(
        lexicon.items(),
        key=lambda item: len(
            item[0]
        ),
        reverse=True,
    )

    for normalized_label, uri in items:
        if not normalized_label:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(
                normalized_label
            )
            + r"(?!\w)"
        )

        if not re.search(
            pattern,
            normalized_query,
        ):
            continue

        uri_text = str(
            uri
        )

        if uri_text in seen_uris:
            continue

        matches.append(
            MappedOntologyConcept(
                original_text=normalized_label,
                concept_name=_local_name(
                    uri
                ),
                label=_canonical_label(
                    graph,
                    uri,
                ),
                uri=uri_text,
            )
        )

        seen_uris.add(
            uri_text
        )

    return matches


def _extract_json_object(
    content: str,
) -> dict:
    """
    Parse one JSON object returned by the development LLM.
    """
    cleaned = (
        content.strip()
    )

    if cleaned.startswith(
        "```"
    ):
        lines = (
            cleaned.splitlines()
        )

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

        if cleaned.lower().startswith(
            "json"
        ):
            cleaned = (
                cleaned[4:]
                .strip()
            )

    try:
        parsed = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:
        raise ConceptMappingError(
            (
                "Concept-extraction LLM returned invalid JSON. "
                f"Raw response: {content!r}"
            )
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):
        raise ConceptMappingError(
            (
                "Concept-extraction response was valid JSON "
                "but was not a JSON object."
            )
        )

    return parsed


def extract_candidate_concepts(
    query: str,
    *,
    settings: Settings | None = None,
) -> ExtractedConcepts:
    """
    Ask the LLM only for candidate phrases.

    At most one regeneration attempt is allowed if a successful
    model call returns malformed or schema-invalid output.
    """
    settings = (
        settings
        or get_settings()
    )

    normalized_query = (
        query.strip()
    )

    if not normalized_query:
        raise ValueError(
            "Concept extraction query cannot be empty."
        )

    client = create_openrouter_client(
        settings
    )

    total_attempts = 2
    last_error: Exception | None = None

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        try:
            completion = _call_with_one_retry(
                lambda: (
                    client
                    .chat
                    .completions
                    .create(
                        model=(
                            settings.openrouter_model
                        ),
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    CONCEPT_EXTRACTION_SYSTEM_PROMPT
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    normalized_query
                                ),
                            },
                        ],
                        max_tokens=min(
                            settings.llm_max_output_tokens,
                            500,
                        ),
                        response_format={
                            "type": "json_object",
                        },
                    )
                ),
                operation_name=(
                    "ontology concept extraction"
                ),
            )

            if not completion.choices:
                raise ConceptMappingError(
                    (
                        "Concept-extraction LLM "
                        "returned no choices."
                    )
                )

            content = (
                completion
                .choices[0]
                .message
                .content
            )

            if not content:
                raise ConceptMappingError(
                    (
                        "Concept-extraction LLM "
                        "returned an empty response."
                    )
                )

            raw_data = _extract_json_object(
                content
            )

            try:
                extracted = (
                    ExtractedConcepts
                    .model_validate(
                        raw_data
                    )
                )

            except ValidationError as exc:
                raise ConceptMappingError(
                    (
                        "Concept-extraction JSON did not "
                        "match the expected schema. "
                        f"Raw response: {content!r}"
                    )
                ) from exc

            cleaned_concepts: list[
                str
            ] = []

            seen: set[str] = set()

            for concept in (
                extracted.concepts
            ):
                cleaned = (
                    concept.strip()
                )

                normalized = normalize_concept_text(
                    cleaned
                )

                if (
                    not cleaned
                    or not normalized
                    or normalized in seen
                ):
                    continue

                cleaned_concepts.append(
                    cleaned
                )

                seen.add(
                    normalized
                )

            return ExtractedConcepts(
                concepts=cleaned_concepts
            )

        except (
            ConceptMappingError,
            LLMServiceError,
        ) as exc:
            last_error = exc

            if attempt == total_attempts:
                break

    raise ConceptMappingError(
        (
            "Concept extraction did not produce valid "
            "structured output after one bounded retry. "
            f"Last error: {last_error}"
        )
    ) from last_error


def build_ontology_context(
    query: str,
    *,
    graph: Graph | None = None,
    settings: Settings | None = None,
) -> OntologyContext:
    """
    Build the natural-language → ontology bridge.

    Sources of mapped concepts:

    1. LLM candidate extraction followed by deterministic mapping.
    2. Direct deterministic matching against ontology labels/aliases.

    Results are merged by ontology URI.
    """
    normalized_query = (
        query.strip()
    )

    if not normalized_query:
        raise ValueError(
            "Ontology query cannot be empty."
        )

    graph = (
        graph
        or load_ontology()
    )

    extracted = (
        extract_candidate_concepts(
            normalized_query,
            settings=settings,
        )
    )

    mapped_from_llm, unknown = (
        map_candidate_concepts(
            candidates=(
                extracted.concepts
            ),
            graph=graph,
        )
    )

    direct_matches = (
        find_known_concepts_in_query(
            query=normalized_query,
            graph=graph,
        )
    )

    merged: list[
        MappedOntologyConcept
    ] = []

    seen_uris: set[str] = set()

    for concept in (
        mapped_from_llm
        + direct_matches
    ):
        if concept.uri in seen_uris:
            continue

        merged.append(
            concept
        )

        seen_uris.add(
            concept.uri
        )

    return OntologyContext(
        query=normalized_query,
        extracted_concepts=(
            extracted.concepts
        ),
        mapped_concepts=merged,
        unknown_concepts=unknown,
        mapped_count=len(
            merged
        ),
        unknown_count=len(
            unknown
        ),
    )