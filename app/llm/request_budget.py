"""Evaluation-scoped request budgeting for LLM provider calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Callable


class EvaluationBudgetExhausted(Exception):
    """An evaluation has reached its allowed provider-request count."""


@dataclass
class RequestBudget:
    max_attempts: int
    min_interval_seconds: float = 0.0
    attempts: int = 0
    last_attempt_at: float | None = None
    clock: Callable[[], float] = field(
        default=time.monotonic,
        repr=False,
    )
    sleeper: Callable[[float], None] = field(
        default=time.sleep,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")

    def before_request(self) -> None:
        """Reserve one HTTP attempt, pacing it if necessary."""
        if self.attempts >= self.max_attempts:
            raise EvaluationBudgetExhausted(
                "Evaluation provider-request budget exhausted."
            )

        if self.last_attempt_at is not None:
            elapsed = self.clock() - self.last_attempt_at
            remaining = self.min_interval_seconds - elapsed

            if remaining > 0:
                self.sleeper(remaining)

        self.attempts += 1
        self.last_attempt_at = self.clock()


_active_budget: ContextVar[RequestBudget | None] = ContextVar(
    "medonto_evaluation_request_budget",
    default=None,
)


def get_active_budget() -> RequestBudget | None:
    return _active_budget.get()


@contextmanager
def activate_request_budget(budget: RequestBudget):
    """Temporarily activate a request budget in this context."""
    token = _active_budget.set(budget)

    try:
        yield budget
    finally:
        _active_budget.reset(token)
