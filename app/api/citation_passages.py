"""Bounded, same-execution, governance-checked passages for admin demo queries.

This module does not provide arbitrary chunk lookup. Returned text is sourced
solely from retrieved chunks cited in the completed execution. All eligibility
checks fail closed. Do not log passage text.
"""
from app.ingestion.registry import DocumentRegistry, DEFAULT_REGISTRY_PATH, DocumentRegistryError
from app.models.documents import IndexStatus

MAX_PASSAGES = 5
MAX_PASSAGE_CHARS = 1200


def extract_cited_passages(*, execution, citations, status: str, registry=None) -> list[dict]:
    if status not in ("VERIFIED", "ANSWERED") or not citations:
        return []
    if registry is None:
        registry = DocumentRegistry(DEFAULT_REGISTRY_PATH)
    if execution.execution_mode == "MEDONTO":
        state = execution.medonto_state or {}
        retrieval = state.get("retrieval_result")
        chunks = retrieval.retrieved_chunks if retrieval is not None else []
    else:
        baseline = execution.baseline_response
        chunks = baseline.retrieved_chunks if baseline is not None else []

    by_pair = {(c.document_id, c.chunk_id): c for c in chunks}
    output = []
    seen = set()
    for citation in citations:
        key = (citation.document_id, citation.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        chunk = by_pair.get(key)
        if chunk is None or not isinstance(chunk.text, str) or not chunk.text.strip():
            continue
        try:
            if not registry.is_document_control_eligible(citation.document_id):
                continue
            document = registry.get_document(citation.document_id)
            if document.index_status != IndexStatus.INDEXED:
                continue
            if document.version != chunk.version:
                continue
        except (DocumentRegistryError, OSError, ValueError):
            continue
        # Include at most a short verbatim prefix, with explicit truncation.
        original = chunk.text
        text = original[:MAX_PASSAGE_CHARS]
        output.append({
            "chunk_id": citation.chunk_id,
            "document_id": citation.document_id,
            "text": text,
            "truncated": len(original) > MAX_PASSAGE_CHARS,
        })
        if len(output) >= MAX_PASSAGES:
            break
    return output
