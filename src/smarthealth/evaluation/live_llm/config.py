"""Strict reproducible experiment settings. Credentials only come from Settings."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False
    )
    schema_version: Literal["1.0"] = "1.0"
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: str = Field(default="", max_length=128, pattern=r"^[A-Za-z0-9._:/-]*$")
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    max_completion_tokens: int = Field(default=2048, ge=1, le=2048)
    seed: int | None = Field(default=42, ge=0, le=2147483647)
    fixture_seed: int = Field(default=10042, ge=0, le=2147483647)
    max_tasks: int = Field(default=1, ge=1, le=20)
    repetitions: int = Field(default=1, ge=1, le=10)
    concurrency: int = Field(default=1, ge=1, le=4)
    max_total_tokens: int = Field(default=300000, ge=1, le=10000000)
    max_provider_requests: int = Field(default=32, ge=1, le=512)
    per_task_token_budget: int = Field(default=65536, ge=1, le=1000000)
    decision_timeout_seconds: float = Field(default=30, gt=0, le=120)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    max_estimated_cost: float | None = Field(default=None, gt=0)
    baseline: Literal["full_guard", "guard_off", "no_lineage", "no_budget"] = "full_guard"
    # Phase 3 comparison method. Baseline ablations mutate the gated system and
    # are therefore undefined for the ungated loops.
    method: Literal["full_guard", "b1_ungated_react", "b2_no_rag"] = "full_guard"
    # Phase 4 WESAD suite: packaged reviewed fixture plus a deployed bundle.
    suite: Literal["synthetic", "wesad"] = "synthetic"
    model_dir: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def prices_together(self):
        if (self.input_price_per_million is None) != (self.output_price_per_million is None):
            raise ValueError("Both operator-supplied token prices are required together")
        if self.max_estimated_cost is not None and self.input_price_per_million is None:
            raise ValueError("A spending estimate cap requires both token prices")
        return self

    @model_validator(mode="after")
    def wesad_suite_is_gated_full_guard(self):
        if self.suite == "wesad":
            if self.method != "full_guard" or self.baseline != "full_guard":
                raise ValueError("The WESAD suite is only defined for the full-guard method")
            if not self.model_dir:
                raise ValueError("The WESAD suite requires the deployed model bundle directory")
        return self

    @model_validator(mode="after")
    def baselines_require_full_guard_method(self):
        if self.method != "full_guard" and self.baseline != "full_guard":
            raise ValueError("Baseline ablations are only defined for the full-guard method")
        return self


def load_config(path: Path | None = None) -> EvalConfig:
    if path is None:
        return EvalConfig()
    if path.stat().st_size > 32768:
        raise ValueError("Evaluation config exceeds size limit")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalConfig.model_validate(content, strict=True)
