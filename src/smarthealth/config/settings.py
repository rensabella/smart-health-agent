"""Validated environment configuration; secrets and mutable state stay outside the package."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from smarthealth.config.resources import resource_path


class Settings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", hide_input_in_errors=True, allow_inf_nan=False
    )

    home: Path = Field(default_factory=Path.cwd)
    model_dir: Path | None = None
    api_token: SecretStr | None = Field(default=None, exclude=True, repr=False)
    agent_backend: Literal["disabled", "openai_compatible"] = "disabled"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = ""
    llm_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    harness_config_dir: Path = Field(default_factory=lambda: resource_path("configs"))
    agent_trace_dir: Path | None = None
    rag_mode: Literal["disabled", "local"] = "disabled"
    rag_model_dir: Path | None = None
    rag_index_dir: Path | None = None
    rag_min_score: float = Field(default=0.78, ge=-1, le=1)
    memory_mode: Literal["local", "ephemeral"] = "local"
    memory_dir: Path | None = None
    memory_session_ttl_seconds: float = Field(default=3600, gt=0)
    memory_retention_seconds: float = Field(default=2592000, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = "http://127.0.0.1:8000"

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_level(cls, value):
        return value.upper() if isinstance(value, str) else value

    @field_validator("api_token")
    @classmethod
    def nonempty_api_token(cls, value):
        if value is not None and not value.get_secret_value():
            raise ValueError("HEALTH_API_TOKEN must be omitted or nonempty")
        return value

    @model_validator(mode="after")
    def resolve_paths(self):
        home = self.home.expanduser().absolute()
        object.__setattr__(self, "home", home)
        defaults = {
            "model_dir": home / "models_saved" / "lightgbm_v1.0",
            "memory_dir": home / "data" / "memory" / "qdrant",
            "agent_trace_dir": home / "logs" / "agent_runs",
        }
        for name in (
            "model_dir",
            "memory_dir",
            "agent_trace_dir",
            "harness_config_dir",
            "rag_model_dir",
            "rag_index_dir",
        ):
            value = getattr(self, name) or defaults.get(name)
            if value is not None:
                # Preserve explicit symlink components for each storage adapter's
                # existing path checks; only trusted package resources are resolved.
                object.__setattr__(self, name, value.expanduser().absolute())
        return self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read a fresh snapshot. No .env loading, import-time cache or global mutation."""
        env = os.environ if environ is None else environ
        values = {
            name: env[f"HEALTH_{name.upper()}"]
            for name in cls.model_fields
            if f"HEALTH_{name.upper()}" in env
        }
        return cls.model_validate(values)
