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

Current phase: PDF, DOCX, and TXT document loading and extraction established.

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