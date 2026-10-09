from opentelemetry import (
    metrics,
)


_METER_NAME = "medontoagent"


def _meter():
    return metrics.get_meter(
        _METER_NAME
    )


def record_request(
    *,
    status: str,
    latency_ms: float,
    retry_count: int,
) -> None:
    """
    Record bounded operational request metrics.

    No query text, answer text, retrieved clinical text,
    prompts, credentials, or provider errors are recorded.
    """
    meter = _meter()

    request_counter = meter.create_counter(
        "medonto.requests",
        unit="1",
        description=(
            "Total MedOntoAgent query requests."
        ),
    )

    latency_histogram = meter.create_histogram(
        "medonto.request.duration",
        unit="ms",
        description=(
            "MedOntoAgent request duration."
        ),
    )

    retry_counter = meter.create_counter(
        "medonto.retries",
        unit="1",
        description=(
            "Total MedOntoAgent graph retries."
        ),
    )

    attributes = {
        "status": status,
    }

    request_counter.add(
        1,
        attributes=attributes,
    )

    latency_histogram.record(
        latency_ms,
        attributes=attributes,
    )

    if retry_count > 0:
        retry_counter.add(
            retry_count,
            attributes=attributes,
        )


def record_retrieval(
    *,
    result_count: int,
    ontology_filter: bool,
    relaxed: bool,
) -> None:
    """
    Record retrieval behavior using bounded metadata only.
    """
    meter = _meter()

    result_histogram = meter.create_histogram(
        "medonto.retrieval.results",
        unit="1",
        description=(
            "Number of retrieved results per retrieval."
        ),
    )

    result_histogram.record(
        result_count,
        attributes={
            "ontology_filter": (
                ontology_filter
            ),
            "relaxed": relaxed,
        },
    )
