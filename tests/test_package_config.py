"""Package boundaries and settings behavior, including legacy import compatibility."""

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.evaluation.cases import load_cases
from smarthealth.runtime.harness.config import load_config


def test_defaults_work_outside_checkout_without_writing_package(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings.from_env({})
    assert settings.model_dir == tmp_path / "models_saved/lightgbm_v1.0"
    assert settings.memory_dir == tmp_path / "data/memory/qdrant"
    assert settings.agent_trace_dir == tmp_path / "logs/agent_runs"
    assert load_config(settings.harness_config_dir).agent.version == "1.0"
    assert load_config(resource_path("configs", "report")).skills is not None
    load_cases(resource_path("evaluation", "test_cases.json"), root=resource_path())
    assert not list(tmp_path.iterdir())


def test_settings_are_fresh_validated_and_secrets_are_not_exported(tmp_path):
    env = {
        "HEALTH_HOME": str(tmp_path),
        "HEALTH_LOG_LEVEL": "debug",
        "HEALTH_API_TOKEN": "private-token",
        "HEALTH_LLM_API_KEY": "private-provider-key",
        "HEALTH_MEMORY_RETENTION_SECONDS": "120",
        "HEALTH_API_PORT": "8010",
    }
    first = Settings.from_env(env)
    assert first.log_level == "DEBUG" and first.api_port == 8010
    assert first.memory_retention_seconds == 120
    assert first.api_token.get_secret_value() == "private-token"
    assert "private-" not in repr(first) + first.model_dump_json()
    env["HEALTH_API_PORT"] = "8020"
    assert Settings.from_env(env).api_port == 8020 and first.api_port == 8010
    with pytest.raises(ValidationError):
        first.api_port = 8020


@pytest.mark.parametrize(
    "name,value",
    [
        ("HEALTH_API_PORT", "0"),
        ("HEALTH_API_PORT", "invalid"),
        ("HEALTH_MEMORY_SESSION_TTL_SECONDS", "nan"),
        ("HEALTH_MEMORY_RETENTION_SECONDS", "-1"),
        ("HEALTH_RAG_MIN_SCORE", "inf"),
        ("HEALTH_LOG_LEVEL", "TRACE"),
        ("HEALTH_AGENT_BACKEND", "scripted"),
        ("HEALTH_API_TOKEN", ""),
    ],
)
def test_invalid_environment_fails_without_disclosing_inputs(name, value):
    with pytest.raises(ValidationError) as exc:
        Settings.from_env({name: value, "HEALTH_LLM_API_KEY": "private-provider-key"})
    assert "private-provider-key" not in str(exc.value)


@pytest.mark.parametrize(
    "old,new",
    [
        ("app.main", "smarthealth.api.application"),
        ("app.agents.runtime", "smarthealth.agents.executor"),
        ("app.core.context", "smarthealth.runtime.context"),
        ("skills.registry", "smarthealth.skills.registry"),
        ("frontend.client", "smarthealth.frontend.client"),
    ],
)
def test_legacy_imports_share_module_identity(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_bundled_defaults_match_reviewed_source_assets():
    # These root directories remain operator examples; changes must update wheel assets too.
    roots = [
        Path("configs"),
        Path("knowledge"),
        Path("data/sample/demo.csv"),
        Path("evaluation/test_cases.json"),
    ]
    for root in roots:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in files:
            assert resource_path(*path.parts).read_bytes() == path.read_bytes(), str(path)


def test_packaged_resources_under_symlinked_install_prefix(monkeypatch, tmp_path):
    from smarthealth.config import resources
    from smarthealth.retrieval.files import read_json

    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "example.json").write_text('{"resource": true}')
    alias = tmp_path / "prefix_alias"
    alias.symlink_to(installed, target_is_directory=True)
    monkeypatch.setattr(resources, "files", lambda package: alias)
    resolved = resources.resource_path("example.json")
    assert resolved == installed / "example.json"
    assert read_json(resolved, max_bytes=100) == {"resource": True}


def test_settings_do_not_bypass_external_index_symlink_checks(tmp_path):
    from smarthealth.retrieval.files import checked_path
    from smarthealth.runtime.errors import SkillError

    index = tmp_path / "index"
    index.mkdir()
    alias = tmp_path / "index_alias"
    alias.symlink_to(index, target_is_directory=True)
    settings = Settings.from_env({"HEALTH_RAG_INDEX_DIR": str(alias)})
    assert settings.rag_index_dir == alias
    with pytest.raises(SkillError):
        checked_path(settings.rag_index_dir, directory=True)
