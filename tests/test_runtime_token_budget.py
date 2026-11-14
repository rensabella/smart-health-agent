"""Admission invariants, including contention and unobservable provider usage."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from smarthealth.runtime.errors import SkillError
from smarthealth.runtime.token_budget import RunTokenBudget


def test_exact_boundary_reserves_before_work_and_settles_once():
    budget = RunTokenBudget(100)
    ticket = budget.reserve(80, 20)
    assert budget.snapshot().available_tokens == 0
    assert budget.snapshot().in_flight_tokens == 100
    with pytest.raises(SkillError, match="Insufficient"):
        budget.reserve(1, 1)
    assert budget.snapshot().admitted_requests == 1
    budget.settle(ticket, 25, 5)
    result = budget.snapshot()
    assert (result.charged_tokens, result.available_tokens, result.in_flight_tokens) == (30, 70, 0)
    assert result.usage_complete and result.known_input_tokens == 25
    with pytest.raises(ValueError, match="already settled"):
        budget.settle(ticket, 0, 0)
    assert budget.snapshot() == result


@pytest.mark.parametrize("inputs,outputs", [(None, None), (15, None), (None, 3), (True, -1)])
def test_unknown_or_partial_usage_does_not_release_capacity(inputs, outputs):
    budget = RunTokenBudget(100)
    ticket = budget.reserve(80, 20)
    budget.settle(ticket, inputs, outputs)
    state = budget.snapshot()
    known = (inputs if type(inputs) is int and inputs >= 0 else 0) + (
        outputs if type(outputs) is int and outputs >= 0 else 0
    )
    assert state.charged_tokens == 100 and state.in_flight_tokens == 0
    assert state.retained_unknown_tokens == 100 - known
    assert state.unknown_usage_attempts == 1 and not state.usage_complete
    with pytest.raises(SkillError):
        budget.reserve(1, 1)


@pytest.mark.parametrize("inputs,outputs", [(81, 1), (1, 21), (150, None)])
def test_observed_bound_overrun_blocks_future_admission_even_with_balance(inputs, outputs):
    budget = RunTokenBudget(1000)
    ticket = budget.reserve(80, 20)
    budget.settle(ticket, inputs, outputs)
    assert budget.snapshot().bound_exceeded and budget.snapshot().available_tokens > 0
    with pytest.raises(SkillError):
        budget.reserve(1, 1)


def test_reservations_cannot_be_forged_or_settled_in_another_run():
    first, second = RunTokenBudget(100), RunTokenBudget(100)
    ticket = first.reserve(80, 20)
    second.reserve(80, 20)
    for ledger, other in ((second, ticket), (first, replace(ticket, output_bound=1))):
        with pytest.raises(ValueError):
            ledger.settle(other, 1, 1)
    assert first.snapshot().in_flight_tokens == second.snapshot().in_flight_tokens == 100


def test_concurrent_reservations_cannot_oversubscribe():
    budget = RunTokenBudget(100)

    def reserve(_):
        try:
            return budget.reserve(8, 2)
        except SkillError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = [r for r in pool.map(reserve, range(50)) if r is not None]
    assert len(admitted) == 10 and budget.snapshot().charged_tokens == 100
    for ticket in admitted:
        budget.settle(ticket, 8, 2)
    assert budget.snapshot().usage_complete and budget.snapshot().settled_requests == 10
