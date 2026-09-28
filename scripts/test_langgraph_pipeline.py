from app.agent.graph import (
    medonto_graph,
)


query = (
    "Can Example Antihypertensive Class "
    "be used during pregnancy?"
)


result = medonto_graph.invoke(
    {
        "query": query,
    }
)


print(
    "STATUS:",
    result.get(
        "orchestration_status"
    ),
)

print(
    "RETRY COUNT:",
    result.get(
        "retry_count"
    ),
)

print(
    "ONTOLOGY FILTER:",
    result.get(
        "used_ontology_filter"
    ),
)

print(
    "RETRIEVAL RELAXED:",
    result.get(
        "retrieval_relaxed"
    ),
)


constraint_set = result.get(
    "constraint_set"
)

if constraint_set is not None:
    print(
        "CONSTRAINT COUNT:",
        constraint_set.constraint_count,
    )


retrieval = result.get(
    "retrieval_result"
)

if retrieval is not None:
    print(
        "RESULT COUNT:",
        retrieval.result_count,
    )

    for chunk in (
        retrieval.retrieved_chunks
    ):
        print(
            " -",
            chunk.title,
            round(
                chunk.score,
                4,
            ),
        )


draft = result.get(
    "draft_output"
)


if draft is not None:
    print()

    print(
        "GENERATION STATUS:",
        draft.status,
    )

    print()

    print(
        "ANSWER:"
    )

    print(
        draft.answer
    )

    print()

    print(
        "CLAIMS:"
    )

    if draft.claims:
        for index, claim in enumerate(
            draft.claims,
            start=1,
        ):
            print(
                f"{index}.",
                claim.claim_text,
            )

            print(
                "   citations:",
                claim.citation_chunk_ids,
            )

    else:
        print(
            "None"
        )

    print()

    print(
        "CONSTRAINTS REPORTED BY MODEL:"
    )

    if (
        draft
        .considered_constraints
    ):
        for constraint in (
            draft
            .considered_constraints
        ):
            print(
                "-",
                constraint.constraint_type,
                constraint.subject_name,
                constraint.predicate_name,
                constraint.object_name,
            )

    else:
        print(
            "None"
        )


abstention_reason = result.get(
    "abstention_reason"
)

if abstention_reason:
    print()

    print(
        "ABSTENTION REASON:"
    )

    print(
        abstention_reason
    )