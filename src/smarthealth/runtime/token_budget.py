"""Per-run admission and settlement; unknown provider usage retains allowance."""

from dataclasses import dataclass
from threading import Lock
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from smarthealth.runtime.errors import SkillError
from smarthealth.skills.base_skill import SkillModel


class TokenBudgetSnapshot(SkillModel):
    model_config = ConfigDict(frozen=True)

    limit: Annotated[int, Field(ge=1)]
    charged_tokens: Annotated[int, Field(ge=0)]
    available_tokens: Annotated[int, Field(ge=0)]
    in_flight_tokens: Annotated[int, Field(ge=0)]
    retained_unknown_tokens: Annotated[int, Field(ge=0)]
    known_input_tokens: Annotated[int, Field(ge=0)]
    known_output_tokens: Annotated[int, Field(ge=0)]
    admitted_requests: Annotated[int, Field(ge=0)]
    settled_requests: Annotated[int, Field(ge=0)]
    unknown_usage_attempts: Annotated[int, Field(ge=0)]
    usage_complete: bool
    bound_exceeded: bool
    reservation_method: Literal["serialized_request_bytes_plus_completion_cap"] = (
        "serialized_request_bytes_plus_completion_cap"
    )


class AdmissionDemand(SkillModel):
    """Payload-free measurements of a prepared request, including rejected requests."""

    request_bytes: Annotated[int, Field(ge=0)]
    input_bound: Annotated[int, Field(ge=0)]
    output_cap: Annotated[int, Field(ge=0)]
    requested_tokens: Annotated[int, Field(ge=0)]
    available_tokens: Annotated[int, Field(ge=0)]
    shortfall_tokens: Annotated[int, Field(ge=0)]


@dataclass(frozen=True)
class TokenReservation:
    request_id: int
    input_bound: int
    output_bound: int

    @property
    def tokens(self):
        return self.input_bound + self.output_bound


def _counter(value):
    return value if type(value) is int and value >= 0 else None


class RunTokenBudget:
    """Reserve before sending, settle once, and fail closed on observed overruns.

    Request bytes are an estimate, not a universal tokenizer or billing guarantee.
    Missing counters cannot release a reservation, including after cancellation.
    """

    def __init__(self, limit: int):
        if type(limit) is not int or limit < 1:
            raise ValueError("Token limit must be a positive integer")
        self.limit = limit
        self._lock = Lock()
        self._reservations: dict[int, tuple[TokenReservation, bool]] = {}
        self._charged = self._in_flight = self._retained = 0
        self._input = self._output = self._settled = self._unknown = 0
        self._overrun = False

    def reserve(self, input_bound: int, output_bound: int) -> TokenReservation:
        if any(type(v) is not int or v < 1 for v in (input_bound, output_bound)):
            raise ValueError("Reservation bounds must be positive integers")
        with self._lock:
            if self._overrun or self._charged + input_bound + output_bound > self.limit:
                raise SkillError(
                    "token_budget_exceeded", "Insufficient token allowance for this request."
                )
            ticket = TokenReservation(len(self._reservations) + 1, input_bound, output_bound)
            self._reservations[ticket.request_id] = (ticket, False)
            self._charged += ticket.tokens
            self._in_flight += ticket.tokens
            return ticket

    def settle(self, ticket, input_tokens=None, output_tokens=None):
        inputs, outputs = _counter(input_tokens), _counter(output_tokens)
        complete = inputs is not None and outputs is not None
        known = (inputs or 0) + (outputs or 0)
        with self._lock:
            attempt = (
                self._reservations.get(ticket.request_id)
                if isinstance(ticket, TokenReservation)
                else None
            )
            if attempt is None or attempt[0] is not ticket or attempt[1]:
                raise ValueError("Reservation is foreign or already settled")
            charge = known if complete else max(known, ticket.tokens)
            self._reservations[ticket.request_id] = (ticket, True)
            self._charged += charge - ticket.tokens
            self._in_flight -= ticket.tokens
            self._input += inputs or 0
            self._output += outputs or 0
            self._settled += 1
            if not complete:
                self._unknown += 1
                self._retained += charge - known
            self._overrun |= (
                (inputs is not None and inputs > ticket.input_bound)
                or (outputs is not None and outputs > ticket.output_bound)
                or self._charged > self.limit
            )

    def snapshot(self) -> TokenBudgetSnapshot:
        with self._lock:
            admitted = len(self._reservations)
            return TokenBudgetSnapshot(
                limit=self.limit,
                charged_tokens=self._charged,
                available_tokens=max(0, self.limit - self._charged),
                in_flight_tokens=self._in_flight,
                retained_unknown_tokens=self._retained,
                known_input_tokens=self._input,
                known_output_tokens=self._output,
                admitted_requests=admitted,
                settled_requests=self._settled,
                unknown_usage_attempts=self._unknown,
                usage_complete=admitted > 0 and self._settled == admitted and not self._unknown,
                bound_exceeded=self._overrun,
            )
