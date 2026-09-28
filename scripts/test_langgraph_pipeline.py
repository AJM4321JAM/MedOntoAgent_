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


print()

print(
    result.get(
        "draft_answer"
    )
)