from qdrant_client.http import models


def _source_document_condition(
    source_document_ids: list[str] | None,
) -> models.FieldCondition | None:
    """
    Build an optional exact-source Qdrant condition.
    """
    if not source_document_ids:
        return None

    unique_ids = sorted(
        {
            document_id
            for document_id in source_document_ids
            if document_id.strip()
        }
    )

    if not unique_ids:
        return None

    if len(unique_ids) == 1:
        return models.FieldCondition(
            key="document_id",
            match=models.MatchValue(
                value=unique_ids[0],
            ),
        )

    return models.FieldCondition(
        key="document_id",
        match=models.MatchAny(
            any=unique_ids,
        ),
    )


def approved_document_filter(
    source_document_ids: list[str] | None = None,
) -> models.Filter:
    """
    Restrict retrieval to APPROVED evidence.

    When source IDs are supplied, also restrict retrieval
    to those exact governed documents.
    """
    must_conditions: list[
        models.Condition
    ] = [
        models.FieldCondition(
            key="approval_status",
            match=models.MatchValue(
                value="APPROVED",
            ),
        )
    ]

    source_condition = (
        _source_document_condition(
            source_document_ids
        )
    )

    if source_condition is not None:
        must_conditions.append(
            source_condition
        )

    return models.Filter(
        must=must_conditions
    )


def ontology_concept_filter(
    concept_names: list[str],
    source_document_ids: list[str] | None = None,
) -> models.Filter:
    """
    Require APPROVED evidence and every specified ontology
    concept.

    When source IDs are supplied, also restrict retrieval
    to those exact governed documents.
    """
    base_filter = approved_document_filter(
        source_document_ids
    )

    must_conditions: list[
        models.Condition
    ] = list(
        base_filter.must
        or []
    )

    for concept_name in concept_names:
        must_conditions.append(
            models.FieldCondition(
                key="ontology_concepts",
                match=models.MatchValue(
                    value=concept_name,
                ),
            )
        )

    return models.Filter(
        must=must_conditions
    )
