# MedOntoAgent Evaluation Data

The evaluation framework separates development questions from final evaluation questions and provides repeatable execution and comparison tooling for conventional baseline RAG and MedOntoAgent.

## Development Dataset

`development_dataset.json`

Used for:

- implementation debugging,
- prompt refinement,
- ontology alias refinement,
- retrieval threshold tuning,
- validating abstention behavior,
- validating evaluation instrumentation,
- testing deterministic comparison metrics.

Development questions may contain explicitly synthetic software-test content.

They must not be included in the final reported benchmark.

## Final Golden Dataset

`golden_dataset.json`

The final dataset contains 25 questions distributed as:

- 7 straightforward evidence questions,
- 6 ontology/contraindication questions,
- 4 patient-group questions,
- 3 guideline/version questions,
- 3 no-evidence questions,
- 2 difficult/ambiguous questions.

The final question wording, reference answers, expected evidence, and ontology expectations must be grounded in the approved frozen clinical corpus.

Do not invent reference answers from model memory.

## Freeze Procedure

Before final evaluation:

1. freeze the source-document corpus,
2. record the corpus version,
3. freeze ontology assertions and provenance,
4. freeze chunking configuration,
5. freeze embedding model and dimensions,
6. freeze retrieval top-k and thresholds,
7. freeze prompts,
8. complete the final 25 questions,
9. review expected evidence manually,
10. set `frozen` to `true`,
11. increment the dataset version from the template version.

After the dataset is frozen, changes require a new dataset version.

## Evaluation Principle

The same final questions must be run through both:

- conventional baseline RAG,
- MedOntoAgent.

Both systems must use the same:

- corpus,
- chunks,
- embedding model,
- reference LLM,
- retrieval top-k configuration where applicable.

The intended experimental difference is the ontology-aware retrieval, generation context, and ontology-validation layer.

## Evaluation Runners

The project provides two independent evaluation runners.

Baseline:

```powershell
python -m evals.baseline_runner