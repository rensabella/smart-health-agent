"""Conservative, shared admission limits for live evaluation requests.

The caller supplies the exact serialized request byte length as ``input_bound``.
One token per byte plus the configured completion cap is a conservative estimate,
not a provider tokenizer, billing guarantee, or substitute for provider spend limits.
Failed, cancelled, and unknown-usage attempts retain their entire reservation.
No retries or provider requests occur in this module.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal

from smarthealth.evaluation.live_llm.config import EvalConfig
from smarthealth.runtime.errors import SkillError


@dataclass(frozen=True)
class Reservation:
    request_id: int
    input_bound: int
    output_bound: int

    @property
    def tokens(self) -> int:
        return self.input_bound + self.output_bound


@dataclass
class _Attempt:
    reservation: Reservation
    settled: bool = False


def _counter(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class EvaluationBudget:
    """Atomic admission and exactly-once settlement across concurrent tasks.

    Reserve before awaiting a provider response and settle in ``finally``, even on
    cancellation. Only two valid usage counters release unused capacity. Partial
    valid counters remain visible as a lower bound; the rest remains reserved.
    An observed token or estimated-cost reservation overrun blocks all new calls.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._lock = threading.Lock()
        self._attempts: dict[int, _Attempt] = {}
        self._settled = self._known_attempts = self._unknown_attempts = 0
        self._input_tokens = self._output_tokens = 0
        self._charged_tokens = self._in_flight_tokens = self._unknown_tokens = 0
        self._token_overrun = self._cost_overrun = False
        self._known_cost = self._charged_cost = self._in_flight_cost = self._unknown_cost = Decimal(
            0
        )
        self._input_price = (
            Decimal(str(config.input_price_per_million)) / Decimal(1_000_000)
            if config.input_price_per_million is not None
            else None
        )
        self._output_price = (
            Decimal(str(config.output_price_per_million)) / Decimal(1_000_000)
            if config.output_price_per_million is not None
            else None
        )
        self._cost_cap = (
            Decimal(str(config.max_estimated_cost))
            if config.max_estimated_cost is not None
            else None
        )

    def _cost(self, inputs: int, outputs: int) -> Decimal:
        if self._input_price is None or self._output_price is None:
            return Decimal(0)
        return self._input_price * inputs + self._output_price * outputs

    def reserve(self, input_bound: int) -> Reservation:
        """Admit one potential provider attempt; rejected calls consume nothing."""
        if _counter(input_bound) is None or input_bound == 0:
            raise ValueError("input_bound must be a positive serialized payload byte length")
        tokens = input_bound + self.config.max_completion_tokens
        cost = self._cost(input_bound, self.config.max_completion_tokens)
        with self._lock:
            if self._token_overrun:
                raise SkillError(
                    "evaluation_token_budget_exceeded",
                    "A previous request exceeded its reservation.",
                )
            if self._cost_overrun:
                raise SkillError(
                    "evaluation_cost_budget_exceeded",
                    "A previous request exceeded its cost estimate.",
                )
            if len(self._attempts) >= self.config.max_provider_requests:
                raise SkillError(
                    "evaluation_request_budget_exceeded", "Evaluation request allowance exhausted."
                )
            if self._charged_tokens + tokens > self.config.max_total_tokens:
                raise SkillError(
                    "evaluation_token_budget_exceeded",
                    "Evaluation token reservation limit reached.",
                )
            if self._cost_cap is not None and self._charged_cost + cost > self._cost_cap:
                raise SkillError(
                    "evaluation_cost_budget_exceeded", "Evaluation spending estimate limit reached."
                )
            reservation = Reservation(
                request_id=len(self._attempts) + 1,
                input_bound=input_bound,
                output_bound=self.config.max_completion_tokens,
            )
            self._attempts[reservation.request_id] = _Attempt(reservation)
            self._charged_tokens += tokens
            self._in_flight_tokens += tokens
            self._charged_cost += cost
            self._in_flight_cost += cost
            return reservation

    def settle(
        self,
        reservation: Reservation,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Record usage once; unknown or malformed usage never releases capacity."""
        inputs, outputs = _counter(input_tokens), _counter(output_tokens)
        complete = inputs is not None and outputs is not None
        known_tokens = (inputs or 0) + (outputs or 0)
        known_cost = self._cost(inputs or 0, outputs or 0)
        with self._lock:
            attempt = (
                self._attempts.get(reservation.request_id)
                if isinstance(reservation, Reservation)
                else None
            )
            if attempt is None or attempt.reservation is not reservation:
                raise ValueError("Reservation does not belong to this budget")
            if attempt.settled:
                raise ValueError("Reservation has already been settled")
            reserved_cost = self._cost(reservation.input_bound, reservation.output_bound)
            charge = known_tokens if complete else max(known_tokens, reservation.tokens)
            cost_charge = known_cost if complete else max(known_cost, reserved_cost)
            self._token_overrun |= (
                known_tokens > reservation.tokens
                or (inputs is not None and inputs > reservation.input_bound)
                or (outputs is not None and outputs > reservation.output_bound)
            )
            self._cost_overrun |= known_cost > reserved_cost
            self._charged_tokens += charge - reservation.tokens
            self._in_flight_tokens -= reservation.tokens
            self._charged_cost += cost_charge - reserved_cost
            self._in_flight_cost -= reserved_cost
            self._input_tokens += inputs or 0
            self._output_tokens += outputs or 0
            self._known_cost += known_cost
            self._settled += 1
            if complete:
                self._known_attempts += 1
            else:
                self._unknown_attempts += 1
                self._unknown_tokens += charge - known_tokens
                self._unknown_cost += cost_charge - known_cost
            attempt.settled = True

    def snapshot(self) -> dict:
        """Return defensive aggregate counters without request or credential data.

        ``known_*`` may include partial reported usage. ``retained_unknown_*`` is
        the additional uncertainty above that lower bound. Together with in-flight
        reservations they equal the charged budget. Prices are operator estimates;
        costs are unavailable (null) when prices have not been supplied.
        """
        with self._lock:
            prices_available = self._input_price is not None
            return {
                "admitted_requests": len(self._attempts),
                "settled_requests": self._settled,
                "in_flight_requests": len(self._attempts) - self._settled,
                "known_usage_attempts": self._known_attempts,
                "unknown_usage_attempts": self._unknown_attempts,
                "known_input_tokens": self._input_tokens,
                "known_output_tokens": self._output_tokens,
                "known_total_tokens": self._input_tokens + self._output_tokens,
                "charged_tokens": self._charged_tokens,
                "in_flight_reserved_tokens": self._in_flight_tokens,
                "retained_unknown_tokens": self._unknown_tokens,
                "estimated_known_cost": float(self._known_cost) if prices_available else None,
                "estimated_charged_cost": float(self._charged_cost) if prices_available else None,
                "in_flight_estimated_cost": float(self._in_flight_cost)
                if prices_available
                else None,
                "retained_unknown_estimated_cost": float(self._unknown_cost)
                if prices_available
                else None,
                "prices_available": prices_available,
                "usage_complete": bool(self._attempts)
                and self._settled == len(self._attempts)
                and self._unknown_attempts == 0,
                "overrun": self._token_overrun or self._cost_overrun,
                "token_reservation_overrun": self._token_overrun,
                "cost_reservation_overrun": self._cost_overrun,
                "max_provider_requests": self.config.max_provider_requests,
                "max_total_tokens": self.config.max_total_tokens,
                "max_estimated_cost": self.config.max_estimated_cost,
                "reservation_method": "serialized_request_bytes_plus_completion_cap",
                "billing_guarantee": False,
            }
