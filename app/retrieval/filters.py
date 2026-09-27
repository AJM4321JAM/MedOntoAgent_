from qdrant_client.http import models


def approved_document_filter() -> models.Filter:
    """
    Restrict retrieval to documents explicitly approved
    for the active MedOntoAgent corpus.
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