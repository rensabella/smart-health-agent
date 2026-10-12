"""Exercise parser boundaries and the immutable policy used by the runtime."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.agents.limits import AgentLimits
from app.harness.config import IMPLEMENTED_SKILLS, load_config, with_limits

PROJECT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_NAMES = ("agent", "skills", "safety")


@pytest.fixture
def config_dir(tmp_path):
    for name in CONFIG_NAMES:
        (tmp_path / f"{name}.yaml").write_bytes((PROJECT_CONFIG_DIR / f"{name}.yaml").read_bytes())
    return tmp_path


def edit_config(directory, name, mutate):
    path = directory / f"{name}.yaml"
    value = yaml.safe_load(path.read_text())
    mutate(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_default_policy_loads_without_environment_override(monkeypatch):
    monkeypatch.delenv("HEALTH_HARNESS_CONFIG_DIR", raising=False)
    config = load_config()
    assert config.agent.name == "health_reasoning_agent"
    assert config.agent.version == config.skills.version == config.safety.version == "1.0"
    assert config.agent.roles.planner.allowed_skills == ()
    assert config.agent.roles.planner.allowed_delegates == ("data_analysis",)
    assert set(config.agent.roles.data_analysis.allowed_skills) == IMPLEMENTED_SKILLS
    assert config.safety.medical_report_enabled is False
    assert all(config.safety.safety_rules.model_dump().values())
    assert len(config.fingerprint) == 64


def test_environment_directory_is_explicit_and_invalid_override_does_not_fallback(
    config_dir, monkeypatch
):
    monkeypatch.setenv("HEALTH_HARNESS_CONFIG_DIR", str(config_dir))
    assert load_config() == load_config(config_dir)
    monkeypatch.setenv("HEALTH_HARNESS_CONFIG_DIR", str(config_dir / "missing"))
    with pytest.raises(ValueError, match="Invalid Harness configuration: agent.yaml"):
        load_config()
    assert load_config(config_dir).agent.version == "1.0"


@pytest.mark.parametrize(
    "malformed",
    [
        lambda original: 'version: "1.0"\n' + original,
        lambda original: original + '\n---\nversion: "1.0"\n',
        lambda original: original + "\nprivate: &private SECRET\ncopy: *private\n",
        lambda original: original + "\nmerged: {<<: {private: SECRET}}\n",
        lambda original: "nested: " + "[" * 18 + "0" + "]" * 18,
        lambda original: "1: SECRET\n",
        lambda original: "\tprivate: SECRET\n",
    ],
    ids=[
        "duplicate-key",
        "multiple-documents",
        "alias",
        "merge",
        "depth",
        "non-string-key",
        "syntax",
    ],
)
def test_yaml_parser_rejects_ambiguous_or_unbounded_structures(config_dir, malformed):
    path = config_dir / "agent.yaml"
    path.write_text(malformed(path.read_text()))
    with pytest.raises(ValueError) as error:
        load_config(config_dir)
    # A parser failure must happen before schema validation; an ignored alias or
    # duplicate key must not merely fail later due to an unrelated unknown field.
    assert str(error.value) == "Invalid Harness configuration: agent.yaml"
    assert "SECRET" not in str(error.value)


def test_unsafe_yaml_tags_never_execute_a_constructor(config_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "system", lambda command: calls.append(command))
    (config_dir / "agent.yaml").write_text('!!python/object/apply:os.system ["SECRET"]\n')
    with pytest.raises(ValueError, match="Invalid Harness configuration: agent.yaml"):
        load_config(config_dir)
    assert calls == []


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_size_limit_is_applied_to_every_file_before_parsing(config_dir, name):
    (config_dir / f"{name}.yaml").write_bytes(b"#" + b"x" * 65536)
    with pytest.raises(ValueError) as error:
        load_config(config_dir)
    assert str(error.value) == f"Invalid Harness configuration: {name}.yaml"


def test_invalid_encoding_and_missing_files_fail_without_leaking_paths(config_dir):
    path = config_dir / "safety.yaml"
    path.write_bytes(b"\xffSECRET")
    with pytest.raises(ValueError) as error:
        load_config(config_dir)
    assert str(error.value) == "Invalid Harness configuration: safety.yaml"
    path.unlink()
    with pytest.raises(ValueError) as error:
        load_config(config_dir)
    assert str(config_dir) not in str(error.value)


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("agent", lambda value: value.update(private="SECRET")),
        ("agent", lambda value: value["limits"].update(max_hidden_calls=5)),
        (
            "agent",
            lambda value: value["roles"].update(
                rogue={"allowed_skills": [], "allowed_delegates": []}
            ),
        ),
        ("agent", lambda value: value["roles"]["planner"].update(private="SECRET")),
        ("skills", lambda value: value["skills"][0].update(private="SECRET")),
        ("safety", lambda value: value.update(private="SECRET")),
        ("safety", lambda value: value["safety_rules"].update(allow_diagnosis=True)),
    ],
)
def test_unknown_schema_fields_and_roles_are_rejected(config_dir, name, mutate):
    edit_config(config_dir, name, mutate)
    with pytest.raises(ValueError) as error:
        load_config(config_dir)
    assert str(error.value) == "Invalid Harness configuration schema or permissions"
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "agent",
            lambda value: value["roles"]["data_analysis"]["allowed_skills"].append("rag_retrieval"),
        ),
        ("agent", lambda value: value["roles"]["planner"]["allowed_delegates"].append("safety")),
        (
            "agent",
            lambda value: value["roles"]["data_analysis"]["allowed_skills"].append("csv_analysis"),
        ),
        (
            "agent",
            lambda value: value["roles"]["planner"]["allowed_delegates"].append("data_analysis"),
        ),
        ("skills", lambda value: value["skills"].append(copy.deepcopy(value["skills"][0]))),
        ("skills", lambda value: value["skills"][0].update(name="unknown_skill")),
        ("skills", lambda value: value["skills"].pop()),
        ("agent", lambda value: value["roles"]["data_analysis"].pop("allowed_skills")),
        ("agent", lambda value: value["roles"]["planner"].pop("allowed_delegates")),
        ("agent", lambda value: value["roles"].pop("data_analysis")),
        ("agent", lambda value: value["roles"]["planner"].update(allowed_skills=["csv_analysis"])),
        (
            "agent",
            lambda value: value["roles"]["data_analysis"].update(
                allowed_delegates=["data_analysis"]
            ),
        ),
        (
            "agent",
            lambda value: value["roles"]["data_analysis"].update(allowed_skills="csv_analysis"),
        ),
        ("agent", lambda value: value["roles"]["data_analysis"].update(allowed_skills=[True])),
    ],
)
def test_permissions_require_known_unique_explicit_roles_and_skills(config_dir, name, mutate):
    edit_config(config_dir, name, mutate)
    with pytest.raises(ValueError, match="schema or permissions"):
        load_config(config_dir)


def test_empty_allowlists_preserve_denial_instead_of_adding_defaults(config_dir):
    def deny_all(value):
        for role in value["roles"].values():
            role["allowed_skills"] = []
            role["allowed_delegates"] = []

    edit_config(config_dir, "agent", deny_all)
    config = load_config(config_dir)
    for role in (config.agent.roles.planner, config.agent.roles.data_analysis):
        assert role.allowed_skills == ()
        assert role.allowed_delegates == ()
    # This stage checks the immutable input shape; runtime authorization is tested
    # independently in HarnessValidator tests.
    assert with_limits(config, AgentLimits()).agent.roles == config.agent.roles


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_policy_release_and_schema_versions_must_match(config_dir, name):
    edit_config(config_dir, name, lambda value: value.update(version="2.0"))
    with pytest.raises(ValueError, match="schema or permissions"):
        load_config(config_dir)
    edit_config(config_dir, name, lambda value: value.update(version="1.0", schema_version="2.0"))
    with pytest.raises(ValueError, match="schema or permissions"):
        load_config(config_dir)


@pytest.mark.parametrize(
    "field",
    [
        "require_evidence",
        "forbid_definitive_diagnosis",
        "require_high_risk_recommendation",
        "require_uncertainty",
        "require_current_input",
    ],
)
@pytest.mark.parametrize("value", [False, 0, 1, "true", None])
def test_safety_minimums_cannot_be_disabled_or_impersonated(config_dir, field, value):
    edit_config(config_dir, "safety", lambda config: config["safety_rules"].update({field: value}))
    with pytest.raises(ValueError, match="schema or permissions"):
        load_config(config_dir)


@pytest.mark.parametrize("value", [True, 0, "false", None])
def test_medical_report_gate_requires_real_boolean_false(config_dir, value):
    edit_config(config_dir, "safety", lambda config: config.update(medical_report_enabled=value))
    with pytest.raises(ValueError, match="schema or permissions"):
        load_config(config_dir)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_skill_enabled_is_a_strict_boolean(config_dir, value):
    edit_config(config_dir, "skills", lambda config: config["skills"][0].update(enabled=value))
    with pytest.raises(ValueError, match="schema or permissions"):
        load_config(config_dir)


def test_frozen_policy_is_immutable_through_nested_models_and_collections(config_dir):
    config = load_config(config_dir)
    fingerprint = config.fingerprint
    for target, field, replacement in (
        (config, "agent", config.agent),
        (config.agent, "name", "changed"),
        (config.agent.limits, "max_tool_calls", 1),
        (config.agent.roles.data_analysis, "allowed_skills", ()),
        (config.skills.skills[0], "enabled", False),
        (config.safety.safety_rules, "require_evidence", False),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(target, field, replacement)
    with pytest.raises(TypeError):
        config.skills.skills[0] = config.skills.skills[1]
    with pytest.raises(AttributeError):
        config.agent.roles.data_analysis.allowed_skills.append("rag_retrieval")
    detached = config.model_dump()
    detached["agent"]["limits"]["max_tool_calls"] = 1
    assert config.fingerprint == fingerprint


def reverse_mapping_order(value):
    if isinstance(value, dict):
        return {key: reverse_mapping_order(item) for key, item in reversed(value.items())}
    if isinstance(value, list):
        return [reverse_mapping_order(item) for item in value]
    return value


def test_fingerprint_is_canonical_across_comments_mapping_order_and_yaml_format(config_dir):
    first = load_config(config_dir)
    for name in CONFIG_NAMES:
        path = config_dir / f"{name}.yaml"
        reordered = reverse_mapping_order(yaml.safe_load(path.read_text()))
        path.write_text(
            "# Operator notes are not executable policy.\n"
            + yaml.safe_dump(reordered, sort_keys=False, default_flow_style=True)
        )
    second = load_config(config_dir)
    assert second == first
    assert second.fingerprint == first.fingerprint
    canonical = json.dumps(first.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert first.fingerprint == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("agent", lambda value: value["limits"].update(max_tool_calls=3)),
        ("agent", lambda value: value["roles"]["planner"].update(allowed_delegates=[])),
        ("skills", lambda value: value["skills"][0].update(enabled=False)),
        ("skills", lambda value: value["skills"][0].update(timeout_seconds=4.0)),
    ],
)
def test_effective_policy_changes_have_distinct_fingerprints(config_dir, name, mutate):
    before = load_config(config_dir).fingerprint
    edit_config(config_dir, name, mutate)
    assert load_config(config_dir).fingerprint != before


def test_with_limits_tightens_every_field_without_mutating_operator_snapshot(config_dir):
    config = load_config(config_dir)
    before = config.fingerprint
    overrides = AgentLimits(
        max_tool_calls=2,
        max_decisions=64,
        max_data_tool_calls=20,
        max_delegations=0,
        max_total_tokens=1000,
        run_timeout_seconds=600.0,
        decision_timeout_seconds=2.0,
        retention_seconds=30.0,
        simple_max_llm_requests=2,
    )
    effective = with_limits(config, overrides)
    for key, configured in config.agent.limits.model_dump().items():
        assert getattr(effective.agent.limits, key) == min(configured, getattr(overrides, key))
    assert effective.agent.limits.max_tool_calls == 2
    assert effective.agent.limits.max_decisions == config.agent.limits.max_decisions
    assert effective.agent.roles == config.agent.roles
    assert effective.skills == config.skills and effective.safety == config.safety
    assert config.fingerprint == before
    assert effective.fingerprint != before
    assert with_limits(config, None) == config


def test_with_limits_revalidates_unsafe_constructed_overrides(config_dir):
    config = load_config(config_dir)
    for malformed in (
        AgentLimits.model_construct(max_tool_calls=True),
        AgentLimits.model_construct(max_tool_calls=-1),
        AgentLimits.model_construct(run_timeout_seconds=float("inf")),
    ):
        with pytest.raises(ValidationError):
            with_limits(config, malformed)
