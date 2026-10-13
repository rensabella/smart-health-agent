"""Local tests of evaluation admission accounting; no live model results."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from smarthealth.evaluation.live_llm.budget import EvaluationBudget
from smarthealth.evaluation.live_llm.config import EvalConfig
from smarthealth.runtime.errors import SkillError


def config(**overrides):
    return EvalConfig(max_completion_tokens=20, **overrides)


def test_known_usage_releases_unused_reservation_and_unknown_does_not():
    budget = EvaluationBudget(config(max_total_tokens=200))
    first = budget.reserve(100)
    with pytest.raises(SkillError, match="token reservation") as error:
        budget.reserve(100)
    assert error.value.code == "evaluation_token_budget_exceeded"
    budget.settle(first, input_tokens=30, output_tokens=10)
    second = budget.reserve(100)
    budget.settle(second)
    snapshot = budget.snapshot()
    assert snapshot["admitted_requests"] == snapshot["settled_requests"] == 2
    assert snapshot["known_usage_attempts"] == snapshot["unknown_usage_attempts"] == 1
    assert snapshot["known_total_tokens"] == 40
    assert snapshot["retained_unknown_tokens"] == 120
    assert snapshot["charged_tokens"] == 160
    assert snapshot["in_flight_requests"] == snapshot["in_flight_reserved_tokens"] == 0
    assert not snapshot["usage_complete"]


def test_output_cap_overrun_blocks_admission_even_when_total_fits_reservation():
    budget = EvaluationBudget(config())
    reservation = budget.reserve(1000)
    budget.settle(reservation, input_tokens=1, output_tokens=21)
    assert budget.snapshot()["charged_tokens"] == 22
    assert budget.snapshot()["token_reservation_overrun"]
    with pytest.raises(SkillError):
        budget.reserve(100)


async def test_concurrent_admission_cannot_overspend_shared_budget():
    budget = EvaluationBudget(config(max_total_tokens=360))
    release = asyncio.Event()
    errors = []

    async def call():
        try:
            reservation = budget.reserve(100)
        except SkillError as exc:
            errors.append(exc.code)
            return
        try:
            await release.wait()
        finally:
            budget.settle(reservation)

    tasks = [asyncio.create_task(call()) for _ in range(20)]
    await asyncio.sleep(0)
    snapshot = budget.snapshot()
    assert snapshot["admitted_requests"] == snapshot["in_flight_requests"] == 3
    assert snapshot["charged_tokens"] == snapshot["in_flight_reserved_tokens"] == 360
    assert errors == ["evaluation_token_budget_exceeded"] * 17
    release.set()
    await asyncio.gather(*tasks)
    assert budget.snapshot()["retained_unknown_tokens"] == 360


def test_shared_budget_admission_is_thread_safe():
    budget = EvaluationBudget(config(max_provider_requests=2))

    def attempt(_):
        try:
            return budget.reserve(100)
        except SkillError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(attempt, range(32)))
    assert results.count("evaluation_request_budget_exceeded") == 30
    assert budget.snapshot()["admitted_requests"] == 2


async def test_cancelled_attempt_retains_reserved_allowance():
    budget = EvaluationBudget(config())
    started = asyncio.Event()

    async def call():
        reservation = budget.reserve(100)
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            budget.settle(reservation)

    task = asyncio.create_task(call())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snapshot = budget.snapshot()
    assert snapshot["unknown_usage_attempts"] == 1
    assert snapshot["charged_tokens"] == 120
    assert snapshot["known_total_tokens"] == 0


def test_no_prices_means_cost_unavailable_not_zero():
    budget = EvaluationBudget(config())
    reservation = budget.reserve(100)
    budget.settle(reservation, 12, 3)
    snapshot = budget.snapshot()
    assert snapshot["usage_complete"]
    assert not snapshot["prices_available"]
    for key in (
        "estimated_known_cost",
        "estimated_charged_cost",
        "in_flight_estimated_cost",
        "retained_unknown_estimated_cost",
    ):
        assert snapshot[key] is None
    json.dumps(snapshot, allow_nan=False)


def test_operator_prices_enforce_estimated_cost_before_admission():
    budget = EvaluationBudget(
        config(input_price_per_million=2, output_price_per_million=10, max_estimated_cost=0.0004)
    )
    reservation = budget.reserve(100)  # 0.0002 input + 0.0002 output.
    with pytest.raises(SkillError) as error:
        budget.reserve(1)
    assert error.value.code == "evaluation_cost_budget_exceeded"
    budget.settle(reservation, 10, 1)
    other = budget.reserve(100 - 15)
    budget.settle(other)
    snapshot = budget.snapshot()
    assert snapshot["admitted_requests"] == 2
    assert snapshot["estimated_known_cost"] == pytest.approx(0.00003)
    assert snapshot["retained_unknown_estimated_cost"] == pytest.approx(0.00037)
    assert snapshot["estimated_charged_cost"] == pytest.approx(0.0004)


@pytest.mark.parametrize("missing", [None, -1, True, 1.5, "SECRET"])
def test_partial_or_invalid_usage_keeps_full_charge_and_reports_valid_lower_bound(missing):
    budget = EvaluationBudget(config(input_price_per_million=2, output_price_per_million=10))
    reservation = budget.reserve(100)
    budget.settle(reservation, 30, missing)
    snapshot = budget.snapshot()
    assert snapshot["known_input_tokens"] == 30
    assert snapshot["known_output_tokens"] == 0
    assert snapshot["unknown_usage_attempts"] == 1
    assert snapshot["charged_tokens"] == 120
    assert snapshot["retained_unknown_tokens"] == 90
    assert snapshot["estimated_known_cost"] == pytest.approx(0.00006)
    assert snapshot["estimated_charged_cost"] == pytest.approx(0.0004)
    assert snapshot["retained_unknown_estimated_cost"] == pytest.approx(0.00034)
    assert "SECRET" not in json.dumps(snapshot)


def test_reported_overrun_is_recorded_and_blocks_future_calls():
    budget = EvaluationBudget(config(max_total_tokens=1000))
    reservation = budget.reserve(100)
    budget.settle(reservation, 140, 10)
    snapshot = budget.snapshot()
    assert snapshot["overrun"] and snapshot["token_reservation_overrun"]
    assert snapshot["known_total_tokens"] == snapshot["charged_tokens"] == 150
    with pytest.raises(SkillError) as error:
        budget.reserve(1)
    assert error.value.code == "evaluation_token_budget_exceeded"
    assert budget.snapshot()["admitted_requests"] == 1


def test_partial_reported_overrun_is_not_hidden_by_unknown_usage():
    budget = EvaluationBudget(config(max_total_tokens=1000))
    reservation = budget.reserve(100)
    budget.settle(reservation, 140, None)
    snapshot = budget.snapshot()
    assert snapshot["overrun"]
    assert snapshot["known_total_tokens"] == snapshot["charged_tokens"] == 140
    assert snapshot["unknown_usage_attempts"] == 1
    assert not snapshot["usage_complete"]


def test_asymmetric_price_overrun_blocks_further_calls_even_with_lower_total_tokens():
    budget = EvaluationBudget(config(input_price_per_million=1, output_price_per_million=20))
    reservation = budget.reserve(100)
    budget.settle(reservation, 1, 30)
    snapshot = budget.snapshot()
    assert snapshot["cost_reservation_overrun"]
    # The expensive output also exceeds its own 20-token completion ceiling,
    # even though input+output is below the aggregate 120-token reservation.
    assert snapshot["token_reservation_overrun"]
    assert snapshot["estimated_known_cost"] == pytest.approx(0.000601)
    with pytest.raises(SkillError) as error:
        budget.reserve(1)
    assert error.value.code == "evaluation_token_budget_exceeded"


def test_double_and_foreign_settlement_cannot_release_or_recount_tokens():
    budget = EvaluationBudget(config())
    reservation = budget.reserve(100)
    foreign = EvaluationBudget(config()).reserve(100)
    for invalid in (foreign, replace(reservation), None):
        with pytest.raises(ValueError, match="does not belong"):
            budget.settle(invalid, 0, 0)
    assert budget.snapshot()["charged_tokens"] == 120
    budget.settle(reservation, 5, 2)
    with pytest.raises(ValueError, match="already been settled"):
        budget.settle(reservation, 0, 0)
    assert budget.snapshot()["charged_tokens"] == 7
    assert budget.snapshot()["known_total_tokens"] == 7


@pytest.mark.parametrize("bound", [0, -1, True, 1.5, "SECRET", None])
def test_invalid_input_bounds_never_admit_requests(bound):
    budget = EvaluationBudget(config())
    with pytest.raises(ValueError):
        budget.reserve(bound)
    assert budget.snapshot()["admitted_requests"] == 0


def test_zero_reported_usage_is_known_and_does_not_restore_request_allowance():
    budget = EvaluationBudget(config(max_provider_requests=1))
    reservation = budget.reserve(100)
    budget.settle(reservation, 0, 0)
    snapshot = budget.snapshot()
    assert snapshot["known_usage_attempts"] == 1 and snapshot["usage_complete"]
    assert snapshot["charged_tokens"] == 0
    with pytest.raises(SkillError) as error:
        budget.reserve(100)
    assert error.value.code == "evaluation_request_budget_exceeded"
