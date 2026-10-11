# MedOntoAgent Evaluation Framework

MedOntoAgent evaluates conventional baseline RAG and ontology-aware MedOntoAgent as separate systems using the same controlled evaluation datasets.

The evaluation framework has two complementary layers:

1. project-controlled deterministic evaluation,
2. offline semantic evaluation using Ragas.

The deterministic layer remains authoritative for system behavior, evidence identifiers, ontology relationships, retries, retrieval relaxation, and latency.

Ragas adds semantic measurements for eligible answered responses.

## Development Dataset

`development_dataset.json`

The development dataset is used for:

- implementation debugging,
- prompt refinement,
- ontology alias refinement,
- retrieval tuning,
- validating abstention behavior,
- validating evaluation instrumentation,
- testing deterministic comparison metrics,
- testing semantic evaluation infrastructure.

Development questions may contain explicitly synthetic software-test content.

They must not be included in the final reported benchmark.

## Final Golden Dataset

`golden_dataset.json`

The final dataset template contains 25 questions distributed as:

- 7 straightforward evidence questions,
- 6 ontology or contraindication questions,
- 4 patient-group questions,
- 3 guideline or version questions,
- 3 no-evidence questions,
- 2 difficult or ambiguous questions.

The final question wording, reference answers, expected evidence, and ontology expectations must be grounded in the approved frozen clinical corpus.

Reference answers must not be invented from model memory.

## Freeze Procedure

Before formal final evaluation:

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

## Experimental Principle

The same final questions must be run through both:

- conventional baseline RAG,
- MedOntoAgent.

Both systems must use the same:

- approved corpus,
- underlying chunks,
- embedding model,
- embedding dimensions,
- configured generation model,
- retrieval top-k where applicable.

The intended experimental difference is the ontology-aware retrieval, ontology-controlled generation context, and ontology-validation layer.

MedOntoAgent's ontology filtering may legitimately reduce the number of eligible retrieved chunks below the configured top-k.

## System Evaluation Runners

Baseline:

```powershell
python -m evals.baseline_runner
```

MedOntoAgent:

```powershell
python -m evals.medonto_runner
```

Both default to:

```text
evals/development_dataset.json
```

Development datasets may run while unfrozen.

A dataset with:

```text
split = FINAL
```

is rejected unless final-dataset readiness checks pass.

## Standard Evaluation Artifact

Both system runners emit the same `EvaluationRunArtifact` structure.

Each per-question result records:

- question ID,
- category,
- original question,
- expected ANSWER or ABSTAIN behavior,
- system,
- terminal system status,
- behavior match,
- generated answer,
- citation chunk IDs,
- citation document IDs,
- retrieved chunk IDs,
- retrieved document IDs,
- exact retrieved context text in retrieval order,
- ontology constraints where applicable,
- ontology-filter usage where applicable,
- retrieval relaxation where applicable,
- retry count where applicable,
- end-to-end latency,
- safe error classification.

The exact `retrieved_contexts` list is persisted because semantic evaluation must operate on the evidence actually supplied to the system during that run.

Context order is preserved.

This is important because Context Precision evaluates ranked retrieved contexts.

## Baseline Result Behavior

Baseline terminal states include:

- `ANSWERED`
- `ABSTAINED`
- `ERROR`

Baseline-specific evaluation fields remain:

```text
ontology_constraints = []
used_ontology_filter = null
retrieval_relaxed = null
retry_count = null
```

because those controls are not part of conventional baseline RAG.

## MedOntoAgent Result Behavior

MedOntoAgent terminal states include:

- `VERIFIED`
- `ABSTAINED`
- `ERROR`

MedOntoAgent evaluation records can additionally contain:

- explicit ontology relationships,
- ontology-filter usage,
- retrieval relaxation,
- LangGraph retry count.

The MedOnto runner invokes the graph directly rather than going through the public HTTP API.

This allows evaluation to inspect experimental state without exposing that internal state through the public API.

## Deterministic Comparison

Phase 22 introduced deterministic baseline-versus-MedOnto comparison.

The deterministic metrics include:

- expected answer or abstention behavior,
- operational error rate,
- abstention rate,
- expected-document retrieval hit rate,
- expected-chunk retrieval hit rate,
- citation-document precision,
- citation-document recall,
- citation-chunk precision,
- citation-chunk recall,
- ontology relationship recall,
- ontology exact-match rate,
- retry rate,
- retrieval-relaxation rate,
- latency statistics.

Metrics without sufficient annotations are reported as not applicable rather than treated as failures.

The deterministic comparator verifies reproducibility-critical configuration before comparing paired runs.

## Semantic Evaluation With Ragas

Phase 23 adds an independent offline semantic-evaluation layer using Ragas 0.3.9.

Implemented semantic metrics are:

- Faithfulness,
- Context Precision with Reference,
- Answer Relevancy.

Ontology-specific evaluation remains project-controlled.

Ragas is not used to replace ontology relationship metrics.

### Faithfulness

Faithfulness evaluates whether the generated answer is supported by the retrieved contexts supplied to the system.

Inputs:

```text
question
answer
retrieved_contexts
```

### Context Precision

Context Precision with Reference evaluates whether useful evidence appears appropriately in the ranked retrieved contexts.

Inputs:

```text
question
reference_answer
retrieved_contexts
```

A reference answer is required.

The original context ranking is preserved.

### Answer Relevancy

Answer Relevancy evaluates how directly the generated answer addresses the question.

Inputs:

```text
question
answer
```

Local Sentence Transformers embeddings are used for the embedding component.

## Semantic-Evaluation Eligibility

Ragas scores are calculated only for successful answered outputs.

Baseline:

```text
ANSWERED
```

MedOntoAgent:

```text
VERIFIED
```

The following terminal states are not semantically scored:

```text
ABSTAINED
ERROR
```

They are reported as not applicable for Ragas semantic metrics.

Their behavior is evaluated by the deterministic evaluation layer instead.

This avoids assigning artificial semantic-quality scores to intentional abstentions or failed executions.

## Separate Semantic Artifact

Ragas evaluation produces a separate `RagasEvaluationArtifact`.

It does not modify the original `EvaluationRunArtifact`.

This preserves the system execution as an immutable experimental record.

The semantic artifact records:

- source run ID,
- system,
- dataset name,
- dataset version,
- corpus version,
- evaluator model,
- embedding model,
- per-question Ragas scores,
- safe per-metric error classifications,
- aggregate metric means.

Missing scores are represented as `null`.

They are not converted to zero.

## Running Ragas

Ragas is intentionally isolated from the main MedOntoAgent Python environment because Ragas 0.3.9 is not dependency-compatible with the OpenAI SDK version used by the application.

Main application:

```text
.venv
OpenAI SDK 3.16.2
```

Semantic evaluator:

```text
.ragas-venv
Ragas 0.3.9
OpenAI SDK 2.8.1
```

The evaluator dependencies are defined in:

```text
requirements-ragas.txt
```

Current evaluator dependency set:

```text
ragas==0.4.3
openai==2.8.1
sentence-transformers==6.1.0
pydantic-settings==2.15.0
langchain-community==0.3.31
```

The `langchain-community` compatibility pin is required by the installed Ragas 0.3.9 dependency stack.

The evaluator must be run from `.ragas-venv`.

Example:

```powershell
python -m evals.ragas_runner `
  --run evals/results/<system-run>.json `
  --dataset evals/development_dataset.json
```

## Evaluator Models

The semantic judge is separately configurable using:

```text
RAGAS_EVALUATOR_MODEL
```

The current zero-cost development default is:

```text
gemini-3.5-flash-lite
```

The evaluator reuses:

```text
GEMINI_API_KEY
GEMINI_BASE_URL
```

No separate paid evaluator API is required.

Answer Relevancy embeddings use the local model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

with CPU execution.

## Error Handling

One semantic metric failure does not terminate evaluation.

The artifact stores only a safe classification such as:

```text
faithfulness failed: InstructorRetryException
```

Raw provider responses, prompts, retrieved clinical text, API credentials, and provider stack traces are not persisted as metric-error text.

## Phase 23 Smoke Test

A real Ragas smoke test successfully initialized:

- Ragas 0.3.9,
- the Ragas metric classes,
- the OpenRouter-compatible asynchronous client,
- the local Sentence Transformers embedding model,
- the semantic-evaluation runner,
- semantic artifact persistence.

The test reached the live OpenRouter evaluator.

The configured free OpenRouter route returned HTTP 429 because the account's free daily request allowance had been exhausted.

The runner handled this correctly and produced:

```text
scored_records = 0
faithfulness_mean = null
context_precision_mean = null
answer_relevancy_mean = null
```

This is an operational provider limitation.

It is not evidence about baseline or MedOntoAgent answer quality.

Formal semantic scores must not be reported until eligible system runs are successfully evaluated.

## Development Results

Earlier development execution demonstrated the expected high-level behavior:

```text
DEV-001
Baseline -> ANSWERED
MedOnto -> VERIFIED

DEV-002
Baseline -> ANSWERED
MedOnto -> VERIFIED

DEV-003
Baseline -> ABSTAINED
MedOnto -> ABSTAINED
```

These are implementation checks using the development dataset.

They are not the formal research benchmark.

Recent OpenRouter free-route exhaustion has also produced operational `ERROR` runs.

Those runs are retained as operational observations and must not be interpreted as comparative quality evidence.

## Generated Results

Development result files are stored under:

```text
evals/results/
```

Generated JSON artifacts are ignored by Git.

The directory itself remains version controlled through:

```text
evals/results/.gitkeep
```

## Testing

Phase 23 adds tests covering:

- run-to-dataset compatibility,
- semantic-score calculation,
- context-order preservation,
- abstention exclusion,
- safe metric-error recording,
- semantic artifact persistence,
- correct aggregation when some metrics are not applicable.

At the Phase 23 implementation checkpoint:

```text
183 tests passed
0 failed
```

## Formal Evaluation Rule

The final benchmark must use:

```text
frozen approved clinical corpus
        +
frozen ontology
        +
frozen system configuration
        +
frozen final golden dataset
        |
        +----------------------+
        |                      |
        v                      v
   baseline run           MedOnto run
        |                      |
        v                      v
 execution artifact      execution artifact
        |                      |
        +----------+-----------+
                   |
          +--------+--------+
          |                 |
          v                 v
 deterministic         Ragas semantic
 comparison            evaluation
          |                 |
          +--------+--------+
                   |
                   v
             research analysis
```

Deterministic and semantic measurements remain separate.

No composite score is automatically produced.

## Current Status

Phase 23 semantic-evaluation infrastructure is implemented and tested.

Live semantic metric values remain pending a successful zero-cost evaluator run after provider quota availability returns.

## Next Phase

Phase 24 will focus on security hardening and security controls while preserving the evaluation framework established in Phases 20 through 23.
