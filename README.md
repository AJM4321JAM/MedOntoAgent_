# MedOntoAgent

MedOntoAgent is a domain-adaptable clinical intelligence agent that combines Retrieval-Augmented Generation (RAG) with a lightweight medical ontology.

The system is designed to:

- ingest approved clinical documents,
- retrieve evidence using semantic search,
- apply ontology-derived constraints,
- generate grounded answers using an LLM,
- validate claim-to-citation support,
- validate generated answers against explicit ontology constraints,
- retry generation once when validation identifies a correctable problem,
- abstain when sufficient validated evidence is unavailable.

## Project Status

Current phase: Versioned development and final evaluation datasets, golden-question schema, and evaluation freeze controls established.

The current implementation includes:

- document extraction and provenance models,
- deterministic chunking,
- local embeddings,
- Qdrant vector retrieval,
- conventional baseline RAG,
- RDF/Turtle ontology loading,
- deterministic ontology concept mapping,
- LLM-assisted candidate concept extraction,
- safe deterministic concept-mapping fallback,
- explicit ontology relationship discovery,
- ontology-derived retrieval constraints,
- ontology-tagged vector metadata,
- ontology-aware retrieval,
- LangGraph orchestration,
- structured grounded generation,
- claim-level citation validation,
- ontology-compliance validation,
- bounded regeneration,
- VERIFIED / ABSTAINED terminal behavior,
- a public FastAPI `/query` endpoint.

## Architecture

The current core stack includes:

- Python
- FastAPI
- LangGraph
- Pydantic
- RDF/Turtle
- RDFLib
- Qdrant Cloud
- OpenRouter
- Sentence Transformers
- pytest

Planned or later-stage components include:

- Ragas
- OpenTelemetry
- Docker
- AWS
- GitHub Actions
- Streamlit

At a high level:

User Query
    |
    v
FastAPI
    |
    v
LangGraph
    |
    v
Concept Mapping
    |
    v
RDF Ontology
    |
    v
Constraint Generation
    |
    v
Ontology-Aware Retrieval
    |
    v
Approved Evidence
    |
    +----------------------+
    |                      |
    v                      v
Ontology Constraints   Evidence Chunks
    |                      |
    +-----------+----------+
                |
                v
        Grounded Generation
                |
                v
        Structured DRAFT
                |
        +-------+-------+
        |               |
        v               v
 Citation Validator  Ontology Validator
        |               |
        +-------+-------+
                |
                v
        Response Policy
                |
       +--------+--------+
       |                 |
       v                 v
   VERIFIED          ABSTAINED

A failed validation may trigger one bounded regeneration attempt before the workflow abstains.

## Safety

MedOntoAgent is a research and engineering prototype.

It is not intended to autonomously diagnose, prescribe, or replace clinical judgment.

`VERIFIED` means that a response passed MedOntoAgent's implemented evidence-grounding and ontology-compliance checks against the controlled corpus and ontology.

It does not mean:

- medically guaranteed correct,
- regulator approved,
- safe for autonomous treatment.

Synthetic ontology assertions and synthetic documents are used during implementation testing and must not be interpreted as real clinical guidance.

## Configuration

Local development configuration is loaded from a private `.env` file.

A safe template is provided in `.env.example`.

Real API keys and credentials must never be committed to Git.

Development LLM access currently uses the configured OpenRouter model.

Embeddings are generated locally and therefore do not require a paid embedding API.

## Document Ingestion

MedOntoAgent provides a controlled ingestion pipeline for PDF, DOCX, and TXT source documents.

The pipeline performs:

1. source-file validation,
2. SHA-256 checksum calculation,
3. duplicate detection,
4. document metadata registration,
5. approval-status enforcement,
6. document extraction,
7. conservative cleaning,
8. deterministic chunking,
9. deterministic ontology tagging,
10. local embedding generation,
11. Qdrant upsert,
12. index-lifecycle recording.

Only `APPROVED` documents are eligible for vector indexing.

Registry index states include:

- `NOT_INDEXED`
- `INDEXING`
- `INDEXED`
- `FAILED`

Identical files already successfully indexed are not indexed again.

Files registered as `FAILED` or `NOT_INDEXED` can be retried without creating a second document record.

Each indexed chunk carries its source document's SHA-256 checksum so a vector point can be traced back to the exact source-file version.

Ingestion is currently exposed through the local administrative CLI:

`python -m scripts.ingest_document`

An administrative HTTP upload endpoint is intentionally deferred until authentication, authorization, upload limits, malware/file-security controls, and audit logging are implemented.

## Document Governance

Every source document is assigned validated provenance metadata before it can enter the active knowledge corpus.

The prototype tracks:

- document identity,
- SHA-256 checksum,
- publisher,
- clinical domain,
- source authority,
- version,
- publication/effective dates,
- approval status,
- source URL,
- supersession relationships.

Exact duplicate files are rejected using SHA-256 checksums.

Only documents with `APPROVED` status are eligible for the active corpus.

Older approved versions can be marked `SUPERSEDED` while remaining available for audit and historical analysis.

## Chunking

Approved source documents are cleaned conservatively and split using deterministic section-aware chunking.

Initial chunking parameters:

- maximum chunk size: 600 words
- overlap: 80 words

Every chunk preserves source provenance including:

- document ID,
- title,
- section,
- page where available,
- publisher,
- clinical domain,
- version,
- approval status,
- source URL.

The same underlying chunk corpus is used for baseline RAG and MedOntoAgent to preserve experimental fairness.

## Embeddings

MedOntoAgent uses the local Sentence Transformers model:

`sentence-transformers/all-MiniLM-L6-v2`

Embedding dimensionality:

- 384 dimensions

Embeddings are generated locally on CPU.

The same embedding model and configuration are used for both baseline RAG and MedOntoAgent.

Document chunks are embedded during indexing and stored in the vector database.

User queries are embedded using the same model before similarity search.

## Vector Database

MedOntoAgent uses Qdrant Cloud Free as its managed vector database.

Collection:

`medontoagent_clinical_documents`

Vector configuration:

- dense vectors
- 384 dimensions
- cosine distance

The vectors are generated locally using:

`sentence-transformers/all-MiniLM-L6-v2`

Each Qdrant point stores metadata including:

- deterministic point ID,
- 384-dimensional embedding,
- chunk text,
- document provenance,
- approval state,
- clinical domain,
- source authority,
- version metadata,
- ontology concept tags.

Metadata fields used for filtering are indexed where appropriate.

The application validates existing collection configuration and does not automatically delete or recreate an existing collection.

## Baseline Retrieval

The conventional baseline retrieval system:

1. validates the user query,
2. embeds the query locally,
3. produces a 384-dimensional query vector,
4. searches Qdrant using cosine similarity,
5. restricts retrieval to `APPROVED` source documents,
6. returns up to the configured top chunks with similarity scores and provenance.

The baseline does not use:

- ontology-derived constraints,
- ontology-aware filtering,
- ontology validation,
- ontology-driven query expansion,
- symbolic clinical relationships.

The same embedding model and underlying corpus are used for baseline RAG and MedOntoAgent so that later evaluation can better isolate the effect of the ontology control layer.

## Baseline RAG

The conventional baseline RAG pipeline is:

1. receive a user query,
2. embed the query locally,
3. retrieve approved chunks from Qdrant,
4. format the retrieved chunks as controlled evidence,
5. send only the question and retrieved evidence to the configured development LLM,
6. request structured JSON output,
7. validate the JSON locally with Pydantic,
8. require answered responses to cite retrieved chunk IDs,
9. abstain when no evidence is retrieved or supplied evidence is insufficient.

The baseline deliberately does not use:

- ontology-derived retrieval constraints,
- ontology validation,
- LangGraph orchestration.

The baseline remains available separately so it can later be compared experimentally with MedOntoAgent.

## Ontology

MedOntoAgent uses a lightweight RDF/Turtle ontology loaded locally using RDFLib.

The ontology models selected concept types including:

- conditions,
- treatments,
- drugs and drug classes,
- patient groups,
- investigations,
- guidelines.

It defines relationships including:

- treated by,
- recommended for,
- contraindicated in,
- not recommended for,
- requires investigation,
- supported by guideline.

The ontology is intentionally much smaller than a full medical knowledge graph.

Initial software-development assertions are explicitly marked as synthetic.

Clinical assertions used in formal evaluation must later be grounded in approved clinical source documents.

The ontology is a controlled application asset.

MedOntoAgent does not execute arbitrary user-supplied RDF or SPARQL.

## Concept Mapping

MedOntoAgent maps natural-language queries to ontology concepts using two complementary mechanisms:

1. LLM extraction of candidate phrases.
2. Deterministic matching against controlled RDF labels, SKOS aliases, and ontology resource names.

The LLM cannot:

- create ontology resources,
- select arbitrary ontology URIs,
- establish ontology relationships.

Candidate phrases extracted by the LLM are resolved deterministically against the RDF graph.

Candidates that cannot be safely resolved remain unknown rather than being mapped through semantic guessing.

Direct deterministic query scanning also identifies known ontology concepts that explicitly occur in the user query.

If optional LLM candidate extraction fails after its bounded retry, ontology-context construction safely falls back to deterministic ontology matching.

This fallback does not invent concepts, URIs, or relationships.

The resulting `OntologyContext` records:

- original query,
- extracted concepts,
- resolved ontology concepts,
- ontology URIs,
- canonical labels,
- unknown concepts,
- mapped and unknown counts.

## Ontology Constraints

Mapped query concepts are inspected for explicit relationships already present in the RDF ontology.

Supported initial constraint categories include:

- contraindication,
- not recommended,
- recommendation,
- treatment relationship,
- investigation requirement.

The LLM does not create ontology relationships.

Explicit ontology relationships are converted into structured application-controlled `OntologyConstraint` objects.

Constraints preserve relationship information such as:

- constraint type,
- subject,
- predicate,
- object,
- ontology provenance.

Ontology constraints can also generate retrieval hints and query expansion while preserving the original user query.

## Ontology-Aware Retrieval

MedOntoAgent tags indexed chunks deterministically with known ontology concepts.

When explicit ontology constraints exist, MedOntoAgent can apply ontology-aware Qdrant metadata filtering in addition to vector similarity.

The retrieval process keeps constant:

- source chunks,
- vector embeddings,
- embedding model,
- Qdrant collection.

This allows ontology-aware retrieval to be compared against the conventional baseline without changing the underlying vector representation.

When a strict ontology-aware filter returns no evidence, MedOntoAgent may perform one controlled fallback to approved-only retrieval using the same expanded query vector.

This behavior is recorded as `retrieval_relaxed`.

The conventional baseline retrieval path remains unchanged.

## LangGraph Orchestration

MedOntoAgent uses LangGraph to coordinate the application workflow.

The graph performs:

initialize
→ concept mapping
→ constraint generation
→ ontology-aware retrieval
→ grounded generation
→ validation
→ VERIFIED / RETRY / ABSTAINED

The workflow has a bounded retry policy.

A validation failure may result in one regeneration attempt using precise validator feedback.

There is no unbounded agent loop.

An explicit generation abstention terminates directly as `ABSTAINED` and does not consume the retry path.

## Grounded Generation

MedOntoAgent generation receives:

- the user question,
- approved retrieved evidence,
- explicit application-controlled ontology constraints,
- validation feedback when regenerating.

Generation returns structured output containing:

- answer,
- claims,
- retrieved citation chunk IDs,
- considered ontology constraints,
- DRAFT or ABSTAINED generation status.

A generated `DRAFT` is not automatically trusted.

Before final success, it must pass independent validation.

Retrieved document content is treated as untrusted data and cannot override application instructions.

## Citation Validation

Each structured claim is independently checked against only the retrieved chunks cited by that claim.

The citation validator returns:

- `SUPPORTED`
- `UNSUPPORTED`

Topical similarity alone is not considered sufficient support.

The validator judges only against supplied retrieved evidence and is instructed not to use outside medical knowledge.

Citation identity and citation support are separate controls.

Citation identity asks:

Did this chunk actually come from retrieval?

Citation support asks:

Does this cited evidence actually support the claim?

Citation identity is enforced deterministically.

Citation support is a semantic validation judgment.

## Ontology Validation

Generated answers and claims are independently checked against every explicit ontology constraint relevant to the query.

The ontology validator returns:

- `COMPLIES`
- `CONFLICTS`
- `NOT_APPLICABLE`

`COMPLIES` means the response addresses the relationship consistently.

`CONFLICTS` means the generated response contradicts or reverses the explicit ontology relationship.

`NOT_APPLICABLE` means the response did not make a relevant assertion about that ontology relationship.

Only `CONFLICTS` blocks verification.

The validator cannot create or modify ontology relationships.

## Response Validation

Application logic deterministically combines citation-validation and ontology-validation results.

A response becomes `VERIFIED` only when:

- every generated claim is supported by its cited retrieved evidence, and
- no generated statement conflicts with an explicit ontology constraint.

If validation fails and the retry budget remains, precise validation feedback is supplied to one regeneration attempt.

If validation still fails after the retry budget is exhausted, MedOntoAgent returns `ABSTAINED`.

The final terminal policy is deterministic Python logic.

The semantic judgments used by the citation and ontology validators are model-based.

A precise description is:

MedOntoAgent combines structured semantic validation with deterministic policy enforcement and graph routing.

## MedOntoAgent API

The primary MedOntoAgent endpoint is:

`POST /query`

Example request:

{
  "query": "Can Example Antihypertensive Class be used during pregnancy?"
}

Public terminal statuses are:

- `VERIFIED`
- `ABSTAINED`

Unexpected execution failures are returned as HTTP server errors without exposing internal exception details.

A `VERIFIED` response exposes:

- the validated answer,
- citations reconstructed from actual retrieved chunks,
- explicit ontology constraints involved in the workflow,
- a high-level retrieval summary.

Public citation metadata is reconstructed from the application's actual retrieval result.

The model may identify a retrieved chunk ID, but it does not supply trusted public citation metadata such as document title, page number, or source URL.

Internal information is not exposed through `/query`, including:

- system prompts,
- model messages,
- validation reasoning,
- retry feedback,
- LangGraph implementation details,
- embeddings,
- API credentials,
- provider stack traces.

## Abstention

`ABSTAINED` is a normal system outcome.

An abstained response may occur when:

- sufficient evidence cannot be retrieved,
- generation explicitly determines that supplied evidence is insufficient,
- citation validation fails after the retry budget,
- ontology validation detects a conflict that cannot be corrected within the retry budget.

For the public API:

HTTP 200
status = ABSTAINED
citations = []

Retrieved-but-insufficient chunks are deliberately not returned as validated citations.

## API Routes

The current development API includes:

GET  /
GET  /health

POST /llm-test
POST /query-baseline
POST /query

### Baseline

`POST /query-baseline`

Architecture:

query
→ vector retrieval
→ evidence-grounded LLM

### MedOntoAgent

`POST /query`

Architecture:

query
→ ontology mapping
→ explicit constraints
→ ontology-aware retrieval
→ grounded generation
→ citation validation
→ ontology validation
→ bounded retry / abstention

The baseline endpoint must remain available until formal comparative evaluation is complete.

`POST /llm-test` remains temporarily available as a local development helper for distinguishing model-provider connectivity failures from workflow failures.

## Public API Boundary

The application deliberately separates internal LangGraph state from the public API response.

Internal state may contain:

- validation results,
- validator reasons,
- retry counters,
- workflow status,
- intermediate ontology objects.

These are not automatically serialized to API consumers.

The public API exposes validated results rather than internal reasoning machinery.

## Current Experimental Focus

The formal evaluation focus is hypertension management with pregnancy-related treatment constraints and contraindications.

The project research question is:

Does adding a lightweight formal ontology around conventional RAG measurably improve retrieval and grounded answer quality without requiring a full graph-based clinical platform?

Baseline RAG and MedOntoAgent are maintained as separate execution paths so that the ontology control layer can be evaluated explicitly.

## Development Testing

The project uses `pytest` for automated testing.

Important implemented test areas include:

- document models,
- ingestion utilities,
- deterministic chunking,
- embedding generation,
- Qdrant integration,
- baseline retrieval,
- baseline RAG,
- ontology loading,
- concept mapping,
- deterministic concept fallback,
- ontology relationship queries,
- constraint generation,
- ontology-aware retrieval,
- grounded generation,
- citation validation,
- ontology validation,
- deterministic response validation,
- LangGraph routing and bounded retries,
- FastAPI public response behavior.

A phase is not considered complete until the relevant tests and the complete regression suite pass.

## Next Phase

Phase 19 will build the reusable ingestion pipeline tying together:

- document loading,
- checksum and registry governance,
- cleaning,
- chunking,
- ontology tagging,
- local embedding generation,
- Qdrant upsert,
- failure handling,
- ingestion status.

This will replace the current one-off synthetic indexing scripts with a reusable production-style ingestion path.

## Evaluation Dataset

MedOntoAgent separates development questions from final evaluation questions.

The development dataset may be used for implementation debugging, retrieval tuning, prompt refinement, and ontology-alias refinement.

The final golden dataset is a 25-question controlled benchmark consisting of:

- 7 straightforward evidence questions,
- 6 ontology/contraindication questions,
- 4 patient-group questions,
- 3 guideline/version questions,
- 3 no-evidence questions,
- 2 difficult/ambiguous questions.

The final clinical questions and reference answers are not invented during software development. They must be derived from the frozen approved corpus.

Formal evaluation cannot run until:

- the corpus version is frozen,
- placeholders are removed,
- reference answers are populated for answerable questions,
- the dataset is marked frozen.

The same frozen final dataset is used for both conventional baseline RAG and MedOntoAgent.

