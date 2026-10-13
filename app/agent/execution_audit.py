"""Bounded, content-free LangGraph execution audit."""

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any


ALLOWED_NODES = frozenset({
    "initialize", "map_concepts", "build_constraints",
    "retrieve", "baseline_routed", "generate",
    "validate", "retry", "verified", "abstain",
})

logger = logging.getLogger(__name__)

MAX_EVENTS = 64
TERMINAL_STATUSES = frozenset({
    "VERIFIED", "ABSTAINED", "BASELINE_ROUTED",
})


@dataclass(frozen=True)
class ExecutionEvent:
    sequence: int
    node: str
    elapsed_since_previous_update_ms: float


@dataclass(frozen=True)
class GraphAuditResult:
    state: dict[str, Any]
    events: tuple[ExecutionEvent, ...]


class GraphAuditInterrupted(RuntimeError):
    """A graph execution failed after zero or more completed nodes.

    Only safe, allow-listed node metadata is retained.
    The underlying exception is deliberately not included
    in the public-facing message.
    """

    def __init__(
        self,
        events: tuple[ExecutionEvent, ...],
    ) -> None:
        super().__init__("Graph execution interrupted.")
        self.events = events


def execute_graph_with_audit(
    graph: Any,
    query: str,
) -> GraphAuditResult:
    """Execute the graph and capture safe node-completion metadata.

    The returned state is internal and must never be serialized
    directly into a public API response.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    events: list[ExecutionEvent] = []
    final_state: dict[str, Any] | None = None
    last_update = perf_counter()

    iterator = iter(
        graph.stream(
            {"query": query},
            stream_mode=["updates", "values"],
        )
    )

    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            raise GraphAuditInterrupted(tuple(events)) from exc

        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError("Unexpected graph stream envelope.")

        mode, payload = item

        if mode == "values":
            if not isinstance(payload, dict):
                raise RuntimeError("Unexpected graph state snapshot.")
            final_state = payload
            continue

        if mode != "updates" or not isinstance(payload, dict):
            raise RuntimeError("Unexpected graph stream event.")

        for node, values in payload.items():
            if node not in ALLOWED_NODES:
                raise RuntimeError("Unexpected graph node.")
            if not isinstance(values, dict):
                raise RuntimeError("Unexpected graph node update.")
            if len(events) >= MAX_EVENTS:
                raise RuntimeError("Graph audit event limit exceeded.")

            now = perf_counter()
            events.append(
                ExecutionEvent(
                    sequence=len(events) + 1,
                    node=node,
                    elapsed_since_previous_update_ms=round(
                        max(0.0, (now - last_update) * 1000),
                        2,
                    ),
                )
            )
            last_update = now

            logger.info(
                "MedOnto audit stage complete: "
                "node=%s elapsed_ms=%.2f sequence=%d",
                node,
                events[-1].elapsed_since_previous_update_ms,
                events[-1].sequence,
            )

    if not events or final_state is None:
        raise RuntimeError("Graph produced no complete audit result.")

    if final_state.get("orchestration_status") not in TERMINAL_STATUSES:
        raise RuntimeError("Graph did not reach a recognized terminal state.")

    return GraphAuditResult(
        state=final_state,
        events=tuple(events),
    )
