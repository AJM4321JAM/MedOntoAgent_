from qdrant_client.http import models


def approved_document_filter() -> models.Filter:
    """
    Restrict retrieval to APPROVED evidence.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key="approval_status",
                match=models.MatchValue(
                    value="APPROVED",
                ),
            )
        ]
    )


def ontology_concept_filter(
    concept_names: list[str],
) -> models.Filter:
    """
    Require APPROVED evidence and every specified ontology
    concept to occur in the chunk ontology_concepts array.
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