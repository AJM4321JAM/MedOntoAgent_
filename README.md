# MedOntoAgent

MedOntoAgent is a domain-adaptable clinical intelligence agent that combines Retrieval-Augmented Generation (RAG) with a lightweight medical ontology.

The system is designed to:

- ingest approved clinical documents,
- retrieve evidence using semantic search,
- apply ontology-derived constraints,
- generate grounded answers using an LLM,
- validate citations and ontology relationships,
- abstain when sufficient evidence is unavailable.

## Project Status

Current phase: Explicit ontology relationship discovery and structured retrieval-constraint generation established.

## Architecture

The planned stack includes:

- Python
- FastAPI
- LangGraph
- Pydantic
- RDF/Turtle
- rdflib
- Qdrant Cloud
- OpenRouter
- Ragas
- pytest
- OpenTelemetry
- Docker
- AWS
- GitHub Actions
- Streamlit

## Safety

MedOntoAgent is a research and engineering prototype.

It is not intended to autonomously diagnose, prescribe, or replace clinical judgment.

## Configuration

Local development configuration is loaded from a private `.env` file.

A safe template is provided in `.env.example`.

Real API keys and credentials must never be committed to Git.

## Document Ingestion

The initial ingestion layer supports:

- PDF
- DOCX
- UTF-8 TXT

Source documents are converted into a common validated internal representation before cleaning, chunking, metadata generation, embedding, or retrieval.

Image-only PDFs are not silently accepted. OCR is outside the initial extraction path.

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

Only documents with `APPROVED` status are eligible for the active corpus. Older approved versions can be marked `SUPERSEDED` while remaining available for audit and historical analysis.

## Chunking

Approved source documents are cleaned conservatively and split using deterministic section-aware chunking.

Initial chunking parameters:

- maximum chunk size: 600 words
- overlap: 80 words

These are model-independent starting values and will be evaluated once the embedding model and real clinical corpus are frozen.

Every chunk preserves source provenance including document ID, title, section, page where available, publisher, clinical domain, version, approval status, and source URL.

The same chunking strategy will be used for baseline RAG and MedOntoAgent to preserve experimental fairness.

## Embeddings

MedOntoAgent uses the local Sentence Transformers model
`sentence-transformers/all-MiniLM-L6-v2` as the fixed embedding model.

Embedding dimensionality:

- 384 dimensions

Embeddings are generated locally on CPU and do not require an external
embedding API.

The same embedding model and configuration are used for both baseline
RAG and MedOntoAgent.

Document chunks are embedded during ingestion and stored in the vector
database. User queries are embedded using the same model before
similarity search.

## Vector Database

MedOntoAgent uses Qdrant Cloud Free as its managed vector database.

Collection:

`medontoagent_clinical_documents`

Vector configuration:

- dense vectors
- 384 dimensions
- cosine distance

The vectors are generated locally using
`sentence-transformers/all-MiniLM-L6-v2`.

Each Qdrant point stores:

- a deterministic point ID,
- a 384-dimensional embedding,
- chunk text,
- document provenance,
- approval state,
- clinical domain,
- source authority,
- version metadata.

Metadata fields used for filtering are indexed where appropriate.

The application validates existing collection configuration and never
automatically deletes or recreates an existing collection.

## Baseline Retrieval

The baseline retrieval system:

1. validates the user query,
2. embeds the query locally using
   `sentence-transformers/all-MiniLM-L6-v2`,
3. produces a 384-dimensional query vector,
4. searches the Qdrant vector database using cosine similarity,
5. restricts retrieval to `APPROVED` source documents,
6. returns up to the top 5 chunks with similarity scores and provenance.

The baseline does not use ontology constraints, reranking, query
rewriting, or symbolic clinical rules.

The same local embedding model is used for both baseline RAG and
MedOntoAgent so that the later comparison isolates the effect of the
ontology control layer.

## Baseline RAG

The conventional baseline RAG pipeline is:

1. receive a user query,
2. embed the query locally with
   `sentence-transformers/all-MiniLM-L6-v2`,
3. generate a 384-dimensional query vector,
4. retrieve up to five APPROVED chunks from Qdrant Cloud,
5. format the retrieved chunks as controlled evidence,
6. send only the question and retrieved evidence to the configured
   OpenRouter development model,
7. request a JSON response,
8. validate the JSON locally with Pydantic,
9. require ANSWERED responses to cite retrieved chunk IDs,
10. abstain when no evidence is retrieved or the supplied evidence
    is insufficient.

The baseline deliberately does not use ontology-derived retrieval
constraints, ontology validation, reranking, or LangGraph.

Citation identifiers are checked against the actual retrieval result.
Deeper claim-to-citation support validation is implemented separately
later in the project.

## Ontology

MedOntoAgent uses a lightweight RDF/Turtle ontology loaded
locally using RDFLib.

The initial ontology models selected concept types including:

- conditions
- treatments
- drugs and drug classes
- patient groups
- investigations
- guidelines

It defines relationships including:

- treated by
- recommended for
- contraindicated in
- not recommended for
- requires investigation
- supported by guideline

The ontology is intentionally much smaller than a full medical
knowledge graph.

Initial software-development assertions are explicitly marked as
synthetic. Clinical assertions used in formal evaluation must later
be grounded in approved clinical source documents.

The ontology is a controlled application asset. MedOntoAgent does
not execute arbitrary user-supplied RDF or SPARQL.

## Concept Mapping

MedOntoAgent maps natural-language clinical queries to ontology
concepts using two complementary mechanisms:

1. LLM extraction of candidate clinical phrases.
2. Deterministic matching against controlled RDF labels and
   SKOS aliases.

The LLM cannot create ontology resources, select ontology URIs,
or establish ontology relationships.

All candidate phrases are resolved deterministically against the
RDF graph.

Candidates that cannot be safely resolved remain unknown rather
than being mapped through semantic guessing.

The resulting OntologyContext records:

- the original query,
- extracted concepts,
- resolved concepts,
- ontology URIs,
- canonical labels,
- unknown concepts,
- mapped and unknown counts.

Direct deterministic query scanning complements the LLM so that
known ontology terms explicitly present in the query can still be
detected when the model misses them.

## Ontology Constraints

Mapped query concepts are inspected for explicit relationships
already present in the RDF ontology.

Supported v1 constraint categories include:

- contraindication
- not recommended
- recommendation
- treatment relationship
- investigation requirement

The LLM does not create ontology relationships.

Explicit ontology relationships are converted into structured
OntologyConstraint objects with provenance.

The same constraints may generate RetrievalHints while preserving
the original user query.

Ontology-derived Qdrant payload filtering is not yet performed,
because stored chunks have not yet been tagged with ontology
concept metadata.

The conventional baseline RAG pipeline remains unchanged.