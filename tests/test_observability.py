from fastapi import FastAPI

from app.config.settings import Settings
from app.observability.metrics import (
    record_request,
    record_retrieval,
)
from app.observability.tracing import (
    configure_telemetry,
    current_trace_id,
    observed_span,
)


def test_telemetry_disabled_is_safe_noop():
    app = FastAPI()

    settings = Settings(
        _env_file=None,
        otel_enabled=False,
    )

    configure_telemetry(
        app=app,
        settings=settings,
    )

    with observed_span(
        "test.disabled",
        attributes={
            "safe.count": 1,
        },
    ):
        trace_id = current_trace_id()

    assert trace_id is None


def test_metrics_safe_when_telemetry_disabled():
    record_request(
        status="VERIFIED",
        latency_ms=10.0,
        retry_count=0,
    )

    record_retrieval(
        result_count=2,
        ontology_filter=True,
        relaxed=False,
    )


def test_observed_span_does_not_require_provider():
    with observed_span(
        "test.no_provider"
    ):
        pass


def test_telemetry_source_has_no_sensitive_attribute_names():
    import inspect

    from app.agent import nodes
    from app.api import routes_query
    from app.llm import medonto_generation
    from app.ontology import concept_mapper
    from app.retrieval import (
        embeddings,
        medonto_retriever,
        retriever,
    )
    from app.validation import (
        citation_validator,
        ontology_validator,
    )

    modules = [
        nodes,
        routes_query,
        medonto_generation,
        concept_mapper,
        embeddings,
        retriever,
        medonto_retriever,
        citation_validator,
        ontology_validator,
    ]

    source = "\n".join(
        inspect.getsource(module)
        for module in modules
    )

    forbidden_attribute_patterns = [
        '"medonto.query.text"',
        '"medonto.answer"',
        '"medonto.prompt"',
        '"medonto.evidence"',
        '"medonto.claim.text"',
        '"medonto.validation.feedback"',
        '"authorization"',
        '"api_key"',
    ]

    for pattern in forbidden_attribute_patterns:
        assert pattern not in source
