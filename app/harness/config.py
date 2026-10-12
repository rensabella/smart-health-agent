"""Bounded YAML loading and immutable, explicitly authorized policy snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator
from yaml.events import AliasEvent

from app.agents.limits import AgentLimits
from skills.base_skill import SkillModel

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
IMPLEMENTED_SKILLS = frozenset(
    {
        "csv_analysis",
        "signal_processing",
        "feature_extraction",
        "anomaly_prediction",
        "feature_explanation",
        "visualization",
    }
)
REPORT_SKILLS = frozenset({"rag_retrieval", "report_generation", "safety_verification"})
ALL_SKILLS = IMPLEMENTED_SKILLS | REPORT_SKILLS
AGENT_ROLES = ("planner", "data_analysis", "rag", "report_writer", "safety_verification")
Version = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]


class FrozenPolicy(SkillModel):
    model_config = ConfigDict(frozen=True)


class RolePolicy(FrozenPolicy):
    # Missing permissions fail validation. An explicit [] grants nothing.
    allowed_skills: tuple[str, ...]
    allowed_delegates: tuple[
        Literal["data_analysis", "rag", "report_writer", "safety_verification"], ...
    ]

    @field_validator("allowed_skills", "allowed_delegates", mode="before")
    @classmethod
    def unique_permissions(cls, value):
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise ValueError("Permissions must be explicit string lists")
        if len(value) != len(set(value)):
            raise ValueError("Duplicate permission")
        return tuple(value)

    @field_validator("allowed_skills")
    @classmethod
    def known_skills(cls, value):
        if not set(value) <= ALL_SKILLS:
            raise ValueError("Unknown Skill permission")
        return value


class Roles(FrozenPolicy):
    planner: RolePolicy
    data_analysis: RolePolicy
    rag: RolePolicy | None = None
    report_writer: RolePolicy | None = None
    safety_verification: RolePolicy | None = None

    @model_validator(mode="after")
    def capability_floor(self):
        if self.planner.allowed_skills or self.data_analysis.allowed_delegates:
            raise ValueError("Planner cannot execute Skills; Data Agent cannot delegate")
        if not set(self.data_analysis.allowed_skills) <= IMPLEMENTED_SKILLS:
            raise ValueError("Data Agent cannot review or write reports")
        for role, skill in (
            ("rag", "rag_retrieval"),
            ("report_writer", "report_generation"),
            ("safety_verification", "safety_verification"),
        ):
            policy = getattr(self, role)
            if policy is not None and (
                policy.allowed_delegates or not set(policy.allowed_skills) <= {skill}
            ):
                raise ValueError("Report roles have separate, nondelegating responsibilities")
        return self


class AgentConfig(FrozenPolicy):
    schema_version: Literal["1.0"]
    version: Version
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    limits: AgentLimits
    roles: Roles


class SkillPolicy(FrozenPolicy):
    name: str
    enabled: bool
    version: Version
    timeout_seconds: Annotated[float, Field(gt=0, le=120)]


class SkillsConfig(FrozenPolicy):
    schema_version: Literal["1.0"]
    version: Version
    skills: tuple[SkillPolicy, ...]

    @field_validator("skills", mode="before")
    @classmethod
    def entries(cls, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Skills must be an explicit list")
        return tuple(value)

    @model_validator(mode="after")
    def exact_catalogue(self):
        names = [entry.name for entry in self.skills]
        if len(names) != len(set(names)) or set(names) not in (IMPLEMENTED_SKILLS, ALL_SKILLS):
            raise ValueError(
                "Declare the six data Skills, optionally the complete report capability set"
            )
        return self


class SafetyRules(FrozenPolicy):
    # These are non-negotiable minimums. There is no text-based policy interpreter.
    require_evidence: Literal[True]
    forbid_definitive_diagnosis: Literal[True]
    require_high_risk_recommendation: Literal[True]
    require_uncertainty: Literal[True]
    require_current_input: Literal[True]

    @field_validator("*", mode="before")
    @classmethod
    def real_boolean(cls, value):
        if value is not True:
            raise ValueError("Safety minimums must be boolean true")
        return value


class SafetyConfig(FrozenPolicy):
    schema_version: Literal["1.0"]
    version: Version
    publication_policy: Literal["tool_facts_only_v1", "evidence_report_v1"]
    medical_report_enabled: bool
    safety_rules: SafetyRules

    @field_validator("medical_report_enabled", mode="before")
    @classmethod
    def no_medical_report(cls, value):
        if type(value) is not bool:
            raise ValueError("Report capability flag must be a real boolean")
        return value

    @model_validator(mode="after")
    def policy_scope(self):
        if self.medical_report_enabled and self.publication_policy != "evidence_report_v1":
            raise ValueError("Reports require the evidence publication policy")
        return self


class HarnessConfig(FrozenPolicy):
    agent: AgentConfig
    skills: SkillsConfig
    safety: SafetyConfig

    @model_validator(mode="after")
    def matching_versions(self):
        if len({self.agent.version, self.skills.version, self.safety.version}) != 1:
            raise ValueError("Configuration files must belong to the same policy release")
        if self.safety.medical_report_enabled:
            if {s.name for s in self.skills.skills} != ALL_SKILLS:
                raise ValueError("Report policy requires all implemented report Skill contracts")
            if any(getattr(self.agent.roles, role) is None for role in AGENT_ROLES[2:]):
                raise ValueError("Report roles require explicit permission policies")
        return self

    @property
    def fingerprint(self):
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class _StrictLoader(yaml.SafeLoader):
    """Safe constructors plus no aliases, duplicate keys, merge keys or deep nesting."""

    def __init__(self, stream):
        super().__init__(stream)
        self._depth = 0

    def compose_node(self, parent, index):
        if self.check_event(AliasEvent):
            raise ValueError("YAML aliases are not allowed")
        self._depth += 1
        try:
            if self._depth > 16:
                raise ValueError("YAML nesting exceeds policy limit")
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key == "<<" or key in result:
                raise ValueError("YAML requires unique string keys without merges")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_config(directory: Path | str | None = None) -> HarnessConfig:
    directory = Path(
        directory
        if directory is not None
        else os.getenv("HEALTH_HARNESS_CONFIG_DIR", str(DEFAULT_CONFIG_DIR))
    )
    data = {}
    for name in ("agent", "skills", "safety"):
        try:
            path = directory / f"{name}.yaml"
            # Read bounds apply before parsing; errors never echo administrator file contents.
            with path.open("rb") as stream:
                content = stream.read(65537)
            if len(content) > 65536:
                raise ValueError("Configuration file exceeds 64 KiB")
            data[name] = yaml.load(content.decode("utf-8"), Loader=_StrictLoader)
        except (OSError, ValueError, yaml.YAMLError):
            raise ValueError(f"Invalid Harness configuration: {name}.yaml") from None
    try:
        return HarnessConfig.model_validate(data, strict=True)
    except ValueError:
        raise ValueError("Invalid Harness configuration schema or permissions") from None


def with_limits(config: HarnessConfig, limits: AgentLimits | None) -> HarnessConfig:
    """Explicit application overrides can tighten, never enlarge operator ceilings."""
    checked = HarnessConfig.model_validate(config.model_dump(), strict=True)
    if limits is None:
        return checked
    limits = AgentLimits.model_validate(limits.model_dump(), strict=True)
    merged = {
        key: min(value, getattr(limits, key))
        for key, value in checked.agent.limits.model_dump().items()
    }
    data = checked.model_dump()
    data["agent"]["limits"] = merged
    return HarnessConfig.model_validate(data, strict=True)
