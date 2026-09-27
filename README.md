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

Current phase: Minimal FastAPI backend established with automated health tests.

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