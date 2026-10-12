"""Explicit startup configuration must be honored across API/runtime factories."""

from fastapi.testclient import TestClient

from smarthealth.api.application import create_app
from smarthealth.config.settings import Settings


def test_injected_settings_override_environment_across_factories(
    monkeypatch, tmp_path, predictor, csv_bytes
):
    settings = Settings.from_env(
        {
            "HEALTH_HOME": str(tmp_path),
            "HEALTH_MEMORY_MODE": "ephemeral",
            "HEALTH_API_TOKEN": "injected-test-token",
        }
    )
    monkeypatch.setenv("HEALTH_MEMORY_MODE", "invalid")
    monkeypatch.setenv("HEALTH_AGENT_BACKEND", "invalid")
    monkeypatch.setenv("HEALTH_HARNESS_CONFIG_DIR", str(tmp_path / "missing"))
    app = create_app(predictor=predictor, settings=settings)
    with TestClient(app) as client:
        assert app.state.settings is settings
        assert app.state.agent_runtime.backend.mode == "disabled"
        assert client.post("/predict", files={"file": ("demo.csv", csv_bytes)}).status_code == 401
        response = client.post(
            "/predict",
            files={"file": ("demo.csv", csv_bytes)},
            headers={"Authorization": "Bearer injected-test-token"},
        )
        assert response.status_code == 200
        assert response.json()["window_count"] == 4
