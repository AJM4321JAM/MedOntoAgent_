from __future__ import annotations

from contextlib import contextmanager
from typing import (
    Any,
    Iterator,
)

from fastapi import FastAPI
from opentelemetry import (
    metrics,
    trace,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import (
    FastAPIInstrumentor,
)
from opentelemetry.sdk.metrics import (
    MeterProvider,
)
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import (
    Resource,
)
from opentelemetry.sdk.trace import (
    TracerProvider,
)
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

from app.config.settings import (
    Settings,
)


_TRACER_NAME = "medontoagent"
_configured = False


def _otlp_endpoint(
    base_endpoint: str,
    signal_path: str,
) -> str:
    return (
        base_endpoint.rstrip("/")
        + signal_path
    )


def configure_telemetry(
    *,
    app: FastAPI,
    settings: Settings,
) -> None:
    """
    Configure OpenTelemetry traces and metrics.

    Telemetry remains disabled unless OTEL_ENABLED=true.

    Raw clinical questions, retrieved text, generated answers,
    prompts, credentials, and authorization headers must never
    be added as telemetry attributes.
    """
    global _configured

    if not settings.otel_enabled:
        return

    if _configured:
        return

    resource = Resource.create(
        {
            "service.name": (
                settings.otel_service_name
            ),
            "service.version": (
                settings.otel_service_version
            ),
        }
    )

    trace_provider = TracerProvider(
        resource=resource,
    )

    trace_exporter = OTLPSpanExporter(
        endpoint=_otlp_endpoint(
            settings.otel_exporter_otlp_endpoint,
            "/v1/traces",
        ),
    )

    trace_provider.add_span_processor(
        BatchSpanProcessor(
            trace_exporter
        )
    )

    trace.set_tracer_provider(
        trace_provider
    )

    metric_exporter = OTLPMetricExporter(
        endpoint=_otlp_endpoint(
            settings.otel_exporter_otlp_endpoint,
            "/v1/metrics",
        ),
    )

    metric_reader = (
        PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=(
                settings.otel_export_interval_ms
            ),
        )
    )

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            metric_reader
        ],
    )

    metrics.set_meter_provider(
        meter_provider
    )

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=trace_provider,
        meter_provider=meter_provider,
    )

    _configured = True


@contextmanager
def observed_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span]:
    """
    Create one application span.

    When telemetry is disabled, the OpenTelemetry API supplies
    a no-op span, so callers do not need conditional logic.
    """
    tracer = trace.get_tracer(
        _TRACER_NAME
    )

    with tracer.start_as_current_span(
        name,
        record_exception=False,
        set_status_on_exception=True,
    ) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(
                        key,
                        value,
                    )

        yield span


def current_trace_id() -> str | None:
    """
    Return the active trace ID as a 32-character hexadecimal
    string, or None when no valid trace context exists.
    """
    span = trace.get_current_span()
    context = span.get_span_context()

    if not context.is_valid:
        return None

    return format(
        context.trace_id,
        "032x",
    )
