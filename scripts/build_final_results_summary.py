import json
from pathlib import Path


RESULTS = Path(
    "evals/results"
)

COMPARISON = (
    RESULTS
    / "comparison_final_v1_1.json"
)

RAGAS_BASELINE = (
    RESULTS
    / "ragas_baseline_final_v1_1.json"
)

RAGAS_MEDONTO = (
    RESULTS
    / "ragas_medonto_final_v1_1.json"
)

CATEGORY = (
    RESULTS
    / "category_summary_final_v1_1.json"
)

RAGAS_CATEGORY = (
    RESULTS
    / "ragas_category_summary_final_v1_1.json"
)

JSON_OUTPUT = (
    RESULTS
    / "final_summary_v1_1.json"
)

MARKDOWN_OUTPUT = (
    RESULTS
    / "final_summary_v1_1.md"
)


def load(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def fmt(
    value,
) -> str:
    if value is None:
        return "N/A"

    if isinstance(
        value,
        float,
    ):
        return f"{value:.4f}"

    return str(value)


def main() -> None:
    comparison = load(
        COMPARISON
    )

    ragas_baseline = load(
        RAGAS_BASELINE
    )

    ragas_medonto = load(
        RAGAS_MEDONTO
    )

    category = load(
        CATEGORY
    )

    ragas_category = load(
        RAGAS_CATEGORY
    )

    baseline_metrics = (
        comparison[
            "baseline_metrics"
        ]
    )

    medonto_metrics = (
        comparison[
            "medonto_metrics"
        ]
    )

    baseline_ragas = (
        ragas_baseline[
            "summary"
        ]
    )

    medonto_ragas = (
        ragas_medonto[
            "summary"
        ]
    )

    summary = {
        "experiment_version": (
            "MEDONTO-EVAL-V1.1"
        ),
        "dataset_name": (
            comparison[
                "dataset_name"
            ]
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
        "primary_execution_count": 50,
        "deterministic_metrics": {
            "baseline": (
                baseline_metrics
            ),
            "medonto": (
                medonto_metrics
            ),
        },
        "ragas_metrics": {
            "baseline": (
                baseline_ragas
            ),
            "medonto": (
                medonto_ragas
            ),
        },
        "category_deterministic": (
            category[
                "categories"
            ]
        ),
        "category_ragas": (
            ragas_category[
                "categories"
            ]
        ),
        "methodology_notes": [
            (
                "Primary Baseline and MedOnto outputs "
                "were generated once from the frozen "
                "golden dataset."
            ),
            (
                "Deterministic comparison and Ragas "
                "evaluation were derived from those "
                "saved execution artifacts."
            ),
            (
                "ABSTAINED and ERROR records were not "
                "semantically scored by Ragas."
            ),
            (
                "Ragas scores are LLM-assisted semantic "
                "evaluation signals and are not "
                "deterministic ground truth."
            ),
            (
                "Category-level semantic means may use "
                "small and unequal scored-record counts."
            ),
            (
                "The summary reports measurements only "
                "and does not declare either architecture "
                "the overall winner."
            ),
        ],
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# MedOntoAgent Final Evaluation Summary v1.1",
        "",
        "## Experiment",
        "",
        (
            f"- Dataset: "
            f"{summary['dataset_name']} "
            f"{summary['dataset_version']}"
        ),
        (
            f"- Corpus: "
            f"{summary['corpus_version']}"
        ),
        (
            f"- Baseline run: "
            f"{summary['baseline_run_id']}"
        ),
        (
            f"- MedOnto run: "
            f"{summary['medonto_run_id']}"
        ),
        (
            "- Primary system executions: 50"
        ),
        "",
        "## Overall Deterministic Metrics",
        "",
        "| Metric | Baseline | MedOnto |",
        "|---|---:|---:|",
    ]

    metric_rows = [
        (
            "Behavior match rate",
            "behavior_match_rate",
        ),
        (
            "Error rate",
            "error_rate",
        ),
        (
            "Abstention rate",
            "abstention_rate",
        ),
        (
            "Expected document retrieval hit",
            (
                "expected_document_"
                "retrieval_hit_rate"
            ),
        ),
        (
            "Expected chunk retrieval hit",
            (
                "expected_chunk_"
                "retrieval_hit_rate"
            ),
        ),
        (
            "Citation document precision",
            (
                "citation_document_"
                "precision"
            ),
        ),
        (
            "Citation document recall",
            (
                "citation_document_"
                "recall"
            ),
        ),
        (
            "Citation chunk precision",
            (
                "citation_chunk_"
                "precision"
            ),
        ),
        (
            "Citation chunk recall",
            (
                "citation_chunk_"
                "recall"
            ),
        ),
        (
            "Ontology relationship recall",
            (
                "ontology_relationship_"
                "recall"
            ),
        ),
        (
            "Ontology exact match",
            (
                "ontology_exact_match_"
                "rate"
            ),
        ),
        (
            "Retry rate",
            "retry_rate",
        ),
        (
            "Retrieval relaxation rate",
            (
                "retrieval_relaxation_"
                "rate"
            ),
        ),
    ]

    for label, key in metric_rows:
        lines.append(
            (
                f"| {label} | "
                f"{fmt(baseline_metrics.get(key))} | "
                f"{fmt(medonto_metrics.get(key))} |"
            )
        )

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Metric | Baseline | MedOnto |",
            "|---|---:|---:|",
            (
                "| Mean latency ms | "
                f"{fmt(baseline_metrics['latency']['mean_ms'])} | "
                f"{fmt(medonto_metrics['latency']['mean_ms'])} |"
            ),
            (
                "| Median latency ms | "
                f"{fmt(baseline_metrics['latency']['median_ms'])} | "
                f"{fmt(medonto_metrics['latency']['median_ms'])} |"
            ),
            (
                "| P95 latency ms | "
                f"{fmt(baseline_metrics['latency']['p95_ms'])} | "
                f"{fmt(medonto_metrics['latency']['p95_ms'])} |"
            ),
            "",
            "## Overall Ragas Metrics",
            "",
            "| Metric | Baseline | MedOnto |",
            "|---|---:|---:|",
            (
                "| Scored records | "
                f"{baseline_ragas['scored_records']} | "
                f"{medonto_ragas['scored_records']} |"
            ),
            (
                "| Faithfulness | "
                f"{fmt(baseline_ragas['faithfulness_mean'])} | "
                f"{fmt(medonto_ragas['faithfulness_mean'])} |"
            ),
            (
                "| Context precision | "
                f"{fmt(baseline_ragas['context_precision_mean'])} | "
                f"{fmt(medonto_ragas['context_precision_mean'])} |"
            ),
            (
                "| Answer relevancy | "
                f"{fmt(baseline_ragas['answer_relevancy_mean'])} | "
                f"{fmt(medonto_ragas['answer_relevancy_mean'])} |"
            ),
            "",
            "## Category Deterministic Results",
            "",
        ]
    )

    category_order = [
        "STRAIGHTFORWARD_EVIDENCE",
        "ONTOLOGY_CONSTRAINT",
        "PATIENT_GROUP",
        "GUIDELINE_VERSION",
        "NO_EVIDENCE",
        "DIFFICULT_AMBIGUOUS",
    ]

    for category_name in category_order:
        row = (
            category[
                "categories"
            ][
                category_name
            ]
        )

        b = row[
            "baseline"
        ]

        m = row[
            "medonto"
        ]

        lines.extend(
            [
                f"### {category_name}",
                "",
                (
                    f"- Questions: "
                    f"{row['question_count']}"
                ),
                (
                    "- Behavior match: "
                    f"{fmt(b['behavior_match_rate'])} "
                    "vs "
                    f"{fmt(m['behavior_match_rate'])}"
                ),
                (
                    "- Error rate: "
                    f"{fmt(b['error_rate'])} "
                    "vs "
                    f"{fmt(m['error_rate'])}"
                ),
                (
                    "- Expected document hit: "
                    f"{fmt(b['expected_document_retrieval_hit_rate'])} "
                    "vs "
                    f"{fmt(m['expected_document_retrieval_hit_rate'])}"
                ),
                (
                    "- Expected chunk hit: "
                    f"{fmt(b['expected_chunk_retrieval_hit_rate'])} "
                    "vs "
                    f"{fmt(m['expected_chunk_retrieval_hit_rate'])}"
                ),
                (
                    "- Ontology relationship found: "
                    f"{fmt(b['expected_ontology_relationship_found_rate'])} "
                    "vs "
                    f"{fmt(m['expected_ontology_relationship_found_rate'])}"
                ),
                (
                    "- Mean latency ms: "
                    f"{fmt(b['mean_latency_ms'])} "
                    "vs "
                    f"{fmt(m['mean_latency_ms'])}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Category Ragas Results",
            "",
        ]
    )

    for category_name in category_order:
        row = (
            ragas_category[
                "categories"
            ][
                category_name
            ]
        )

        b = row[
            "baseline"
        ]

        m = row[
            "medonto"
        ]

        lines.extend(
            [
                f"### {category_name}",
                "",
                (
                    "- Scored records: "
                    f"{b['scored_records']} "
                    "vs "
                    f"{m['scored_records']}"
                ),
                (
                    "- Faithfulness: "
                    f"{fmt(b['faithfulness_mean'])} "
                    "vs "
                    f"{fmt(m['faithfulness_mean'])}"
                ),
                (
                    "- Context precision: "
                    f"{fmt(b['context_precision_mean'])} "
                    "vs "
                    f"{fmt(m['context_precision_mean'])}"
                ),
                (
                    "- Answer relevancy: "
                    f"{fmt(b['answer_relevancy_mean'])} "
                    "vs "
                    f"{fmt(m['answer_relevancy_mean'])}"
                ),
                (
                    "- Metric errors: "
                    f"{b['metric_error_count']} "
                    "vs "
                    f"{m['metric_error_count']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Methodology Notes",
            "",
        ]
    )

    for note in summary[
        "methodology_notes"
    ]:
        lines.append(
            f"- {note}"
        )

    lines.append("")

    MARKDOWN_OUTPUT.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "FINAL RESULTS SUMMARY"
    )
    print()

    print(
        "JSON:",
        JSON_OUTPUT,
    )

    print(
        "Markdown:",
        MARKDOWN_OUTPUT,
    )


if __name__ == "__main__":
    main()
