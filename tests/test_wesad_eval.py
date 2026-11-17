"""WESAD suite evaluation wiring: tasks, config gating, fixture manifest, dry-run."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from smarthealth.config.resources import resource_path
from smarthealth.evaluation.live_llm.config import EvalConfig
from smarthealth.evaluation.live_llm.graders import _result_schema
from smarthealth.evaluation.live_llm.runner import load_wesad_fixture, run
from smarthealth.evaluation.live_llm.tasks import BenchmarkSuite, load_tasks
from smarthealth.runtime.errors import SkillError

VALID_COLUMNS = [
    "timestamp",
    "subject_id",
    "session_id",
    "hr",
    "eda",
    "emg",
    "temp",
    "resp",
    "acc_mag",
    "scr",
]


def test_wesad_suite_mirrors_the_synthetic_structure():
    wesad_suite, digest = load_tasks("wesad")
    synthetic, _ = load_tasks()
    assert wesad_suite.version == "live-wesad-v1" and len(wesad_suite.tasks) == 20
    assert len(digest) == 64
    assert {t.task_id for t in wesad_suite.tasks} == {t.task_id for t in synthetic.tasks}
    by_id = {t.task_id: t for t in wesad_suite.tasks}
    fixtures = [t.fixture for t in wesad_suite.tasks]
    assert fixtures.count("wesad_csv") == 15
    assert fixtures.count("invalid_csv") == 1 and fixtures.count("none") == 4
    # Context-free tasks stay byte-identical to the synthetic suite.
    for name in ("invalid_csv", "missing_upload", "ambiguous_task", "unsupported_capability"):
        original = next(t for t in synthetic.tasks if t.task_id == name)
        assert by_id[name] == original
    # The suite version and its fixtures must agree; drift is a contract failure.
    data = wesad_suite.model_dump()
    data["tasks"][1]["fixture"] = "synthetic_csv"
    with pytest.raises(ValidationError):
        BenchmarkSuite.model_validate(data)
    data = wesad_suite.model_dump()
    data["version"] = "live-synthetic-v1"
    with pytest.raises(ValidationError):
        BenchmarkSuite.model_validate(data)
    with pytest.raises(ValueError):
        load_tasks("unknown_suite")


@pytest.mark.parametrize(
    "values",
    [
        {"suite": "wesad"},
        {"suite": "wesad", "model_dir": "m", "method": "b1_ungated_react"},
        {"suite": "wesad", "model_dir": "m", "method": "b2_no_rag"},
        {"suite": "wesad", "model_dir": "m", "baseline": "guard_off"},
        {"suite": "wesad", "model_dir": "x" * 513},
    ],
)
def test_wesad_suite_requires_gated_full_guard_and_a_model_dir(values):
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(values)


def test_wesad_suite_config_accepts_the_pinned_bundle():
    config = EvalConfig.model_validate(
        {"suite": "wesad", "model_dir": "models_saved/wesad_v1/lightgbm_v1.0"}
    )
    assert config.suite == "wesad" and config.method == "full_guard"
    assert config.baseline == "full_guard"


def test_result_schema_score_semantics_follow_the_suite():
    wesad_result = {
        "kind": "score_definition",
        "summary": "Stress-model risk score explanation.",
        "source": {
            "type": "model_card",
            "score_semantics": "wesad_stress_model_score",
            "aggregation": "max",
            "name": "LightGBM",
            "version": "1.0",
        },
    }
    assert _result_schema(wesad_result, "wesad")
    assert not _result_schema(wesad_result)  # default suite is synthetic
    assert not _result_schema(wesad_result, "unknown_suite")
    synthetic_result = {
        **wesad_result,
        "source": {**wesad_result["source"], "score_semantics": "synthetic_anomaly_model_score"},
    }
    assert _result_schema(synthetic_result)
    assert not _result_schema(synthetic_result, "wesad")


def _write_packaged_fixture(directory: Path, **manifest_overrides) -> dict:
    directory = directory / "evaluation"
    directory.mkdir()
    csv_path = directory / "wesad_fixture.csv"
    csv_path.write_bytes(
        b"timestamp,subject_id,session_id,hr,eda,emg,temp,resp,acc_mag,scr\n"
        + b"2026-01-01T00:00:00Z,S17,protocol,62.0,4.1,0.05,36.4,0.5,1.0,0.0\n" * 3
    )
    manifest = {
        "schema_version": "1.0",
        "fixture": "wesad_fixture.csv",
        "label_column_absent": True,
        "dataset": {
            "name": "WESAD",
            "license": "CC-BY-4.0",
            "citation": "Schmidt, Philip, et al. 'Introducing WESAD' ICMI 2018.",
            "source_subject": "S17",
        },
        "derivation": {"profile": "wesad_physiology_v1"},
        "rows": 3,
        "columns": list(VALID_COLUMNS),
        "fixture_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    manifest.update(manifest_overrides)
    (directory / "wesad_fixture_manifest.json").write_text(json.dumps(manifest))
    return manifest


def _point_resource_path(monkeypatch, directory: Path):
    import smarthealth.evaluation.live_llm.runner as module

    monkeypatch.setattr(module, "resource_path", lambda *parts: directory.joinpath(*parts))


def test_load_wesad_fixture_verifies_the_reviewed_manifest(tmp_path, monkeypatch):
    manifest = _write_packaged_fixture(tmp_path)
    _point_resource_path(monkeypatch, tmp_path)
    content, provenance = load_wesad_fixture()
    packaged = tmp_path / "evaluation"
    assert content == (packaged / "wesad_fixture.csv").read_bytes()
    assert provenance["suite"] == "wesad"
    assert provenance["fixture_sha256"] == manifest["fixture_sha256"]
    assert (
        provenance["manifest_sha256"]
        == hashlib.sha256((packaged / "wesad_fixture_manifest.json").read_bytes()).hexdigest()
    )
    assert provenance["dataset"]["license"] == "CC-BY-4.0"
    assert provenance["dataset"]["source_subject"] == "S17"
    assert provenance["derivation"]["profile"] == "wesad_physiology_v1"
    assert provenance["label_column_absent"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "2.0"},
        {"fixture": "other_fixture.csv"},
        {"label_column_absent": False},
        {"rows": 0},
        {"rows": "600"},
        {
            "dataset": {
                "name": "OTHER",
                "license": "CC-BY-4.0",
                "citation": "x",
                "source_subject": "S17",
            }
        },
        {"dataset": {"name": "WESAD", "license": "MIT", "citation": "x", "source_subject": "S17"}},
        {
            "dataset": {
                "name": "WESAD",
                "license": "CC-BY-4.0",
                "citation": "  ",
                "source_subject": "S17",
            }
        },
        {"derivation": {"profile": "synthetic_physiology_v1"}},
        {"columns": [*VALID_COLUMNS, "label"]},
        {"columns": ["timestamp", "subject_id"]},
        {"columns": "not-a-list"},
        {"fixture_sha256": "0" * 64},
    ],
)
def test_load_wesad_fixture_rejects_manifest_drift(tmp_path, monkeypatch, overrides):
    _write_packaged_fixture(tmp_path, **overrides)
    _point_resource_path(monkeypatch, tmp_path)
    with pytest.raises(SkillError):
        load_wesad_fixture()


def test_load_wesad_fixture_rejects_tampered_bytes_and_missing_files(tmp_path, monkeypatch):
    _write_packaged_fixture(tmp_path)
    _point_resource_path(monkeypatch, tmp_path)
    (tmp_path / "evaluation" / "wesad_fixture.csv").write_bytes(b"tampered\n")
    with pytest.raises(SkillError):
        load_wesad_fixture()
    empty = tmp_path / "empty"
    empty.mkdir()
    _point_resource_path(monkeypatch, empty)
    with pytest.raises(SkillError) as exc_info:
        load_wesad_fixture()
    assert exc_info.value.code == "evaluation_configuration_error"


def _wesad_run_args(tmp_path, output_name="out"):
    config_path = tmp_path / "wesad.yaml"
    config_path.write_text(
        "suite: wesad\nmodel_dir: models_saved/wesad_v1/lightgbm_v1.0\n"
        "max_tasks: 20\nrepetitions: 1\n"
    )
    return Namespace(
        config=config_path,
        max_tasks=None,
        repetitions=None,
        concurrency=None,
        model=None,
        method=None,
        suite=None,
        live=False,
        task=None,
        output=tmp_path / output_name,
    )


async def test_wesad_dry_run_verifies_the_fixture_without_any_live_assets(tmp_path, monkeypatch):
    manifest = _write_packaged_fixture(tmp_path)
    _point_resource_path(monkeypatch, tmp_path)
    import smarthealth.evaluation.live_llm.runner as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not access providers, models or environment secrets")

    monkeypatch.setattr(module, "new_backend", forbidden)
    monkeypatch.setattr(module, "load_model", forbidden)
    monkeypatch.setattr(module.Settings, "from_env", forbidden)
    output = tmp_path / "dry"
    args = _wesad_run_args(tmp_path)
    args.output = output
    assert await run(args) == 0
    report = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert all(r["suite"] == "wesad" and r["mode"] == "dry_run" for r in rows)
    assert report["suite"] == "wesad"
    assert report["benchmark_version"] == "live-wesad-v1"
    assert report["fixture_provenance"]["fixture_sha256"] == manifest["fixture_sha256"]
    assert report["fixture_provenance"]["dataset"]["license"] == "CC-BY-4.0"
    assert "WESAD-derived fixture" in report["limitations"][0]
    assert report["budget"]["admitted_requests"] == 0
    assert "fixture_seed" in report  # synthetic metadata keys stay stable


async def test_wesad_dry_run_fails_closed_without_the_packaged_fixture(tmp_path, monkeypatch):
    empty = tmp_path / "missing"
    empty.mkdir()
    _point_resource_path(monkeypatch, empty)
    args = _wesad_run_args(tmp_path)
    assert await run(args) == 2
    output = args.output
    report = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert report["setup_error"] == "evaluation_setup_failed"
    assert len(rows) == 20
    assert all(
        r["status"] == "infrastructure_error" and r["error_code"] == "evaluation_setup_failed"
        for r in rows
    )
    assert "fixture_provenance" not in report


@pytest.mark.skipif(
    not Path(resource_path("evaluation", "wesad_fixture.csv")).is_file(),
    reason="Stage 5 packages the reviewed real-data fixture",
)
def test_packaged_fixture_matches_its_reviewed_manifest():
    content, provenance = load_wesad_fixture()
    assert provenance["dataset"]["license"] == "CC-BY-4.0"
    assert provenance["dataset"]["source_subject"].startswith("S")
    header = content.splitlines()[0].decode("utf-8")
    assert "label" not in header.split(",")
    for name in ("hr", "eda", "emg", "temp", "resp", "acc_mag", "scr"):
        assert name in header.split(",")
