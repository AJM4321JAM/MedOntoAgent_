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


citation_validation = result.get(
    "citation_validation"
)

if citation_validation is not None:
    print()

    print(
        "CITATION VALIDATION:"
    )

    if citation_validation.items:
        for item in (
            citation_validation.items
        ):
            print(
                "- Claim",
                item.claim_index + 1,
                item.verdict,
            )

            print(
                "  Reason:",
                item.reason,
            )

            print(
                "  Citations:",
                item.citation_chunk_ids,
            )

    else:
        print(
            "None"
        )

    print(
        "ALL CLAIMS SUPPORTED:",
        citation_validation
        .all_claims_supported,
    )


ontology_validation = result.get(
    "ontology_validation"
)

if ontology_validation is not None:
    print()

    print(
        "ONTOLOGY VALIDATION:"
    )

    if ontology_validation.items:
        for item in (
            ontology_validation.items
        ):
            print(
                "-",
                item.constraint_type,
                item.subject_name,
                item.predicate_name,
                item.object_name,
            )

            print(
                "  Verdict:",
                item.verdict,
            )

            print(
                "  Reason:",
                item.reason,
            )

    else:
        print(
            "None"
        )

    print(
        "HAS ONTOLOGY CONFLICT:",
        ontology_validation
        .has_conflict,
    )


response_validation = result.get(
    "response_validation"
)

if response_validation is not None:
    print()

    print(
        "FINAL VALIDATION:",
        response_validation.status,
    )

    if response_validation.feedback:
        print(
            "VALIDATION FEEDBACK:"
        )

        print(
            response_validation.feedback
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