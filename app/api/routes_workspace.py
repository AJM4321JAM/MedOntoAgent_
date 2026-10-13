"""Read-only, admin-protected data for the local demonstration workspace."""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from app.config.settings import get_settings
from app.ingestion.registry import DEFAULT_REGISTRY_PATH, DocumentRegistry, DocumentRegistryError
from app.security.auth import require_admin

router = APIRouter(prefix="/demo", tags=["demo workspace"], dependencies=[Depends(require_admin)])


class LibraryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    title: str
    publisher: str
    clinical_domain: str
    document_type: str
    source_authority: str
    version: str
    approval_status: str
    index_status: str
    publication_date: str | None = None
    effective_date: str | None = None
    registered_at: datetime
    indexed_at: datetime | None = None
    source_url: str | None = None
    supersedes_document_id: str | None = None
    superseded_by_document_id: str | None = None


class LibraryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registry_state: Literal["available", "missing"]
    total_documents: int
    documents: list[LibraryDocument]


class ComponentStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    state: Literal["available", "configured", "not_checked", "not_available"]
    explanation: str


class OperationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str
    environment: str
    status: Literal["running"]
    app_version: str
    demo_llm_timeout_seconds: float
    demo_provider_attempt_limit: int
    telemetry_configured: bool
    historical_metrics_available: bool
    components: list[ComponentStatus]


def _read_registry():
    registry = DocumentRegistry(DEFAULT_REGISTRY_PATH)
    exists = registry.registry_path.is_file()
    return exists, registry.list_documents()


@router.get("/library", response_model=LibraryResponse)
def read_library() -> LibraryResponse:
    """Return allowlisted governance metadata only, never source text or paths."""
    try:
        exists, documents = _read_registry()
    except (DocumentRegistryError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Document registry is unavailable.") from exc
    return LibraryResponse(
        registry_state="available" if exists else "missing",
        total_documents=len(documents),
        documents=[
            LibraryDocument(
                document_id=d.document_id, title=d.title, publisher=d.publisher,
                clinical_domain=d.clinical_domain, document_type=d.document_type,
                source_authority=str(d.source_authority), version=d.version,
                approval_status=str(d.approval_status), index_status=str(d.index_status),
                publication_date=str(d.publication_date) if d.publication_date else None,
                effective_date=str(d.effective_date) if d.effective_date else None,
                registered_at=d.registered_at, indexed_at=d.indexed_at,
                source_url=str(d.source_url) if d.source_url else None,
                supersedes_document_id=d.supersedes_document_id,
                superseded_by_document_id=d.superseded_by_document_id,
            ) for d in documents
        ],
    )


@router.get("/operations", response_model=OperationsResponse)
def read_operations() -> OperationsResponse:
    """Report local configuration; do not claim external dependency health."""
    settings = get_settings()
    # This route does not activate the demo scope: explicitly derive the demo policy
    # using the same context manager as /demo/query, without executing a query.
    from app.config.settings import activate_demo_settings
    with activate_demo_settings():
        scoped = get_settings()
        timeout = scoped.llm_timeout_seconds
        attempts = scoped.max_provider_requests_per_query
    return OperationsResponse(
        service=settings.app_name,
        environment=settings.app_env,
        status="running",
        app_version=settings.app_version,
        demo_llm_timeout_seconds=timeout,
        demo_provider_attempt_limit=attempts,
        telemetry_configured=bool(settings.otel_enabled),
        historical_metrics_available=False,
        components=[
            ComponentStatus(label="FastAPI", state="available", explanation="This authenticated request was served successfully."),
            ComponentStatus(label="Document registry", state="not_checked", explanation="View Knowledge Library for the registry's current read status."),
            ComponentStatus(label="Qdrant Cloud", state="not_checked", explanation="No external connectivity probe was performed."),
            ComponentStatus(label="Gemini", state="not_checked", explanation="No provider request was performed."),
            ComponentStatus(label="OpenTelemetry", state="configured" if settings.otel_enabled else "not_available", explanation="Export configuration only; historical metrics are not read by this endpoint."),
        ],
    )
