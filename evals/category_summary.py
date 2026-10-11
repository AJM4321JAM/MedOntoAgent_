import json
from collections import defaultdict
from pathlib import Path


DATASET_PATH = Path(
    "evals/golden_dataset.json"
)

COMPARISON_PATH = Path(
    "evals/results/comparison_final_v1_1.json"
)

OUTPUT_PATH = Path(
    "evals/results/category_summary_final_v1_1.json"
)


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def rate(
    values: list[bool],
) -> float | None:
    if not values:
        return None

    return round(
        sum(
            1
            for value in values
            if value
        )
        / len(values),
        4,
    )


def mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return round(
        sum(values)
        / len(values),
        4,
    )


def summarize_system(
    rows: list[dict],
    *,
    prefix: str,
) -> dict:
    behavior_matches = [
        row[f"{prefix}_behavior_match"]
        for row in rows
    ]

    statuses = [
        row[f"{prefix}_status"]
        for row in rows
    ]

    document_hits = [
        row[
            f"{prefix}_retrieved_expected_document"
        ]
        for row in rows
        if row[
            f"{prefix}_retrieved_expected_document"
        ] is not None
    ]

    chunk_hits = [
        row[
            f"{prefix}_retrieved_expected_chunk"
        ]
        for row in rows
        if row[
            f"{prefix}_retrieved_expected_chunk"
        ] is not None
    ]

    ontology_hits = [
        row[
            f"{prefix}_expected_ontology_relationships_found"
        ]
        for row in rows
        if row[
            f"{prefix}_expected_ontology_relationships_found"
        ] is not None
    ]

    latencies = [
        float(
            row[f"{prefix}_latency_ms"]
        )
        for row in rows
    ]

    return {
        "question_count": len(rows),
        "behavior_match_rate": (
            rate(
                behavior_matches
            )
        ),
        "error_rate": rate(
            [
                status == "ERROR"
                for status in statuses
            ]
        ),
        "abstention_rate": rate(
            [
                status == "ABSTAINED"
                for status in statuses
            ]
        ),
        "expected_document_retrieval_hit_rate": (
            rate(
                document_hits
            )
        ),
        "expected_chunk_retrieval_hit_rate": (
            rate(
                chunk_hits
            )
        ),
        "expected_ontology_relationship_found_rate": (
            rate(
                ontology_hits
            )
        ),
        "mean_latency_ms": (
            mean(
                latencies
            )
        ),
    }


def main() -> None:
    dataset = load_json(
        DATASET_PATH
    )

    comparison = load_json(
        COMPARISON_PATH
    )

    dataset_categories = {
        question["question_id"]:
        question["category"]
        for question
        in dataset["questions"]
    }

    grouped: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for row in comparison["questions"]:
        question_id = (
            row["question_id"]
        )

        category = (
            dataset_categories[
                question_id
            ]
        )

        grouped[
            category
        ].append(
            row
        )

    category_order = [
        "STRAIGHTFORWARD_EVIDENCE",
        "ONTOLOGY_CONSTRAINT",
        "PATIENT_GROUP",
        "GUIDELINE_VERSION",
        "NO_EVIDENCE",
        "DIFFICULT_AMBIGUOUS",
    ]

    categories = {}

    for category in category_order:
        rows = grouped.get(
            category,
            [],
        )

        categories[
            category
        ] = {
            "question_count": len(rows),
            "baseline": (
                summarize_system(
                    rows,
                    prefix="baseline",
                )
            ),
            "medonto": (
                summarize_system(
                    rows,
                    prefix="medonto",
                )
            ),
        }

    output = {
        "dataset_name": (
            comparison["dataset_name"]
        ),
        "dataset_version": (
            comparison[
                "dataset_version"
            ]
        ),
        "corpus_version": (
            comparison[
                "corpus_version"
            ]
        ),
        "baseline_run_id": (
            comparison[
                "baseline_run_id"
            ]
        ),
        "medonto_run_id": (
            comparison[
                "medonto_run_id"
            ]
        ),
        "categories": categories,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "CATEGORY DETERMINISTIC SUMMARY"
    )
    print()

    for category in category_order:
        result = categories[
            category
        ]

        baseline = result[
            "baseline"
        ]

        medonto = result[
            "medonto"
        ]

        print(
            category
        )
        print(
            "  Questions:",
            result[
                "question_count"
            ],
        )
        print(
            "  Behavior match:",
            baseline[
                "behavior_match_rate"
            ],
            "vs",
            medonto[
                "behavior_match_rate"
            ],
        )
        print(
            "  Error rate:",
            baseline[
                "error_rate"
            ],
            "vs",
            medonto[
                "error_rate"
            ],
        )
        print(
            "  Document hit:",
            baseline[
                "expected_document_retrieval_hit_rate"
            ],
            "vs",
            medonto[
                "expected_document_retrieval_hit_rate"
            ],
        )
        print(
            "  Chunk hit:",
            baseline[
                "expected_chunk_retrieval_hit_rate"
            ],
            "vs",
            medonto[
                "expected_chunk_retrieval_hit_rate"
            ],
        )
        print(
            "  Ontology relationship:",
            baseline[
                "expected_ontology_relationship_found_rate"
            ],
            "vs",
            medonto[
                "expected_ontology_relationship_found_rate"
            ],
        )
        print(
            "  Mean latency ms:",
            baseline[
                "mean_latency_ms"
            ],
            "vs",
            medonto[
                "mean_latency_ms"
            ],
        )
        print()

    print(
        "Result:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
