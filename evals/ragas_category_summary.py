import json
from collections import defaultdict
from pathlib import Path


DATASET_PATH = Path(
    "evals/golden_dataset.json"
)

BASELINE_PATH = Path(
    "evals/results/ragas_baseline_final_v1_1.json"
)

MEDONTO_PATH = Path(
    "evals/results/ragas_medonto_final_v1_1.json"
)

OUTPUT_PATH = Path(
    "evals/results/ragas_category_summary_final_v1_1.json"
)


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def summarize_scores(
    rows: list[dict],
) -> dict:
    faithfulness = [
        float(row["faithfulness"])
        for row in rows
        if row["faithfulness"] is not None
    ]

    context_precision = [
        float(row["context_precision"])
        for row in rows
        if row["context_precision"] is not None
    ]

    answer_relevancy = [
        float(row["answer_relevancy"])
        for row in rows
        if row["answer_relevancy"] is not None
    ]

    scored_question_ids = {
        row["question_id"]
        for row in rows
        if (
            row["faithfulness"] is not None
            or row["context_precision"] is not None
            or row["answer_relevancy"] is not None
        )
    }

    metric_error_count = sum(
        len(
            row.get(
                "metric_errors",
                [],
            )
        )
        for row in rows
    )

    return {
        "question_count": len(rows),
        "scored_records": len(
            scored_question_ids
        ),
        "faithfulness_mean": (
            mean(
                faithfulness
            )
        ),
        "context_precision_mean": (
            mean(
                context_precision
            )
        ),
        "answer_relevancy_mean": (
            mean(
                answer_relevancy
            )
        ),
        "metric_error_count": (
            metric_error_count
        ),
    }


def main() -> None:
    dataset = load_json(
        DATASET_PATH
    )

    baseline = load_json(
        BASELINE_PATH
    )

    medonto = load_json(
        MEDONTO_PATH
    )

    category_by_question = {
        question["question_id"]:
        question["category"]
        for question
        in dataset["questions"]
    }

    baseline_grouped: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    medonto_grouped: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for row in baseline["scores"]:
        category = (
            category_by_question[
                row["question_id"]
            ]
        )

        baseline_grouped[
            category
        ].append(
            row
        )

    for row in medonto["scores"]:
        category = (
            category_by_question[
                row["question_id"]
            ]
        )

        medonto_grouped[
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
        baseline_summary = (
            summarize_scores(
                baseline_grouped.get(
                    category,
                    [],
                )
            )
        )

        medonto_summary = (
            summarize_scores(
                medonto_grouped.get(
                    category,
                    [],
                )
            )
        )

        categories[
            category
        ] = {
            "question_count": (
                sum(
                    1
                    for question
                    in dataset["questions"]
                    if question[
                        "category"
                    ] == category
                )
            ),
            "baseline": (
                baseline_summary
            ),
            "medonto": (
                medonto_summary
            ),
        }

    output = {
        "dataset_name": (
            dataset[
                "dataset_name"
            ]
        ),
        "dataset_version": (
            dataset[
                "dataset_version"
            ]
        ),
        "corpus_version": (
            dataset[
                "corpus_version"
            ]
        ),
        "baseline_source_run_id": (
            baseline[
                "source_run_id"
            ]
        ),
        "medonto_source_run_id": (
            medonto[
                "source_run_id"
            ]
        ),
        "baseline_evaluator_model": (
            baseline[
                "evaluator_model"
            ]
        ),
        "medonto_evaluator_model": (
            medonto[
                "evaluator_model"
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
        "RAGAS CATEGORY SUMMARY"
    )
    print()

    for category in category_order:
        result = categories[
            category
        ]

        b = result[
            "baseline"
        ]

        m = result[
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
            "  Scored:",
            b[
                "scored_records"
            ],
            "vs",
            m[
                "scored_records"
            ],
        )
        print(
            "  Faithfulness:",
            b[
                "faithfulness_mean"
            ],
            "vs",
            m[
                "faithfulness_mean"
            ],
        )
        print(
            "  Context precision:",
            b[
                "context_precision_mean"
            ],
            "vs",
            m[
                "context_precision_mean"
            ],
        )
        print(
            "  Answer relevancy:",
            b[
                "answer_relevancy_mean"
            ],
            "vs",
            m[
                "answer_relevancy_mean"
            ],
        )
        print(
            "  Metric errors:",
            b[
                "metric_error_count"
            ],
            "vs",
            m[
                "metric_error_count"
            ],
        )
        print()

    print(
        "Result:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
