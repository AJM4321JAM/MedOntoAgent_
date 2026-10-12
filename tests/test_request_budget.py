"""Offline tests for evaluation request budgeting."""

import pytest

from app.llm.request_budget import (
    EvaluationBudgetExhausted,
    RequestBudget,
)


def test_each_attempt_consumes_one_unit() -> None:
    budget = RequestBudget(
        max_attempts=3,
        min_interval_seconds=0,
    )

    budget.before_request()
    budget.before_request()

    assert budget.attempts == 2

    budget.before_request()

    assert budget.attempts == 3


def test_exhaustion_blocks_next_attempt() -> None:
    budget = RequestBudget(
        max_attempts=2,
        min_interval_seconds=0,
    )

    budget.before_request()
    budget.before_request()

    with pytest.raises(EvaluationBudgetExhausted):
        budget.before_request()

    assert budget.attempts == 2


def test_pacing_uses_simulated_clock() -> None:
    now = [100.0]
    sleeps = []

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    budget = RequestBudget(
        max_attempts=3,
        min_interval_seconds=6.0,
        clock=clock,
        sleeper=sleeper,
    )

    budget.before_request()

    assert sleeps == []
    assert budget.attempts == 1

    now[0] += 2.0

    budget.before_request()

    assert sleeps == pytest.approx([4.0])
    assert budget.attempts == 2

    budget.before_request()

    assert sleeps == pytest.approx([4.0, 6.0])
    assert budget.attempts == 3


def test_invalid_budget_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        RequestBudget(max_attempts=0)

    with pytest.raises(ValueError):
        RequestBudget(
            max_attempts=2,
            min_interval_seconds=-1,
        )



def test_request_budget_context_is_restored() -> None:
    from app.llm.request_budget import (
        activate_request_budget,
        get_active_budget,
    )

    budget = RequestBudget(max_attempts=2)

    assert get_active_budget() is None

    with activate_request_budget(budget):
        assert get_active_budget() is budget

    assert get_active_budget() is None


def test_nested_request_budgets_restore_previous_context() -> None:
    from app.llm.request_budget import (
        activate_request_budget,
        get_active_budget,
    )

    outer = RequestBudget(max_attempts=3)
    inner = RequestBudget(max_attempts=1)

    with activate_request_budget(outer):
        assert get_active_budget() is outer

        with activate_request_budget(inner):
            assert get_active_budget() is inner

        assert get_active_budget() is outer

    assert get_active_budget() is None


def test_request_budget_context_restored_after_exception() -> None:
    from app.llm.request_budget import (
        activate_request_budget,
        get_active_budget,
    )

    budget = RequestBudget(max_attempts=1)

    with pytest.raises(RuntimeError):
        with activate_request_budget(budget):
            assert get_active_budget() is budget
            raise RuntimeError("Synthetic failure")

    assert get_active_budget() is None



def test_llm_retry_helper_counts_actual_attempts() -> None:
    from app.llm.client import _call_with_one_retry
    from app.llm.request_budget import activate_request_budget

    calls = []

    def operation():
        calls.append("called")
        return "success"

    budget = RequestBudget(max_attempts=2)

    with activate_request_budget(budget):
        result = _call_with_one_retry(
            operation,
            operation_name="synthetic request",
        )

    assert result == "success"
    assert calls == ["called"]
    assert budget.attempts == 1


def test_exhausted_budget_blocks_provider_operation() -> None:
    from app.llm.client import _call_with_one_retry
    from app.llm.request_budget import activate_request_budget

    calls = []

    def operation():
        calls.append("called")
        return "success"

    budget = RequestBudget(max_attempts=1)

    with activate_request_budget(budget):
        assert _call_with_one_retry(
            operation,
            operation_name="first request",
        ) == "success"

        with pytest.raises(EvaluationBudgetExhausted):
            _call_with_one_retry(
                operation,
                operation_name="second request",
            )

    assert calls == ["called"]
    assert budget.attempts == 1



@pytest.mark.parametrize(
    "module",
    [
        "evals.baseline_runner",
        "evals.medonto_runner",
    ],
)
@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-provider-requests", "20"],
        ["--min-request-interval", "6"],
        [
            "--max-provider-requests", "0",
            "--min-request-interval", "6",
        ],
        [
            "--max-provider-requests", "20",
            "--min-request-interval", "-1",
        ],
        [
            "--max-provider-requests", "20",
            "--min-request-interval", "nan",
        ],
        [
            "--max-provider-requests", "20",
            "--min-request-interval", "inf",
        ],
    ],
)
def test_evaluation_cli_rejects_invalid_budget_arguments(
    module: str,
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import sys
    from unittest.mock import patch

    runner = importlib.import_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [module, *arguments],
    )

    with patch.object(
        runner,
        "load_evaluation_dataset",
        side_effect=AssertionError(
            "Invalid arguments reached dataset loading."
        ),
    ) as mock_load:
        with pytest.raises(SystemExit) as info:
            runner.main()

    assert info.value.code == 2
    mock_load.assert_not_called()

def test_blocked_rate_limit_retry_preserves_provider_cause() -> None:
    from unittest.mock import MagicMock, patch

    from openai import RateLimitError

    from app.llm.client import _call_with_one_retry
    from app.llm.error_classification import classify_error
    from app.llm.request_budget import activate_request_budget

    response = MagicMock()
    response.status_code = 429
    response.headers = {}

    provider_error = RateLimitError(
        "Synthetic HTTP 429",
        response=response,
        body=None,
    )

    calls = []

    def operation():
        calls.append("attempt")
        raise provider_error

    budget = RequestBudget(max_attempts=1)

    with patch("app.llm.client.time.sleep") as mock_sleep:
        with activate_request_budget(budget):
            with pytest.raises(EvaluationBudgetExhausted) as info:
                _call_with_one_retry(
                    operation,
                    operation_name="synthetic provider request",
                )

    assert calls == ["attempt"]
    assert budget.attempts == 1
    assert isinstance(info.value.__cause__, RateLimitError)
    assert classify_error(info.value) == "RATE_LIMIT"
    mock_sleep.assert_not_called()


def test_request_budget_isolated_across_threads() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from app.llm.request_budget import activate_request_budget, get_active_budget

    gate = Barrier(2)

    def worker():
        budget = RequestBudget(max_attempts=1)
        with activate_request_budget(budget):
            gate.wait(timeout=5)
            assert get_active_budget() is budget
            budget.before_request()
            return budget

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        budgets = [future.result(timeout=10) for future in futures]

    assert budgets[0] is not budgets[1]
    assert [budget.attempts for budget in budgets] == [1, 1]
    assert get_active_budget() is None
