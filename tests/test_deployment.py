"""Static deployment invariants; these do not claim a Docker build or runtime test."""

from pathlib import Path

import yaml


def test_container_roles_keep_single_worker_and_private_state():
    config = yaml.safe_load(Path("docker-compose.yml").read_text())
    services = config["services"]
    api = services["api"]
    command = api["command"]
    assert command[command.index("--workers") + 1] == "1"
    assert "--no-access-log" in command and "--log-config" in command
    assert api["environment"]["HEALTH_API_TOKEN"].startswith("${HEALTH_API_TOKEN:?")
    assert api["environment"]["HEALTH_AGENT_BACKEND"].endswith(":-disabled}")
    assert services["frontend"]["environment"] == {"HEALTH_API_BASE_URL": "http://api:8000"}
    assert not services["frontend"].get("volumes")
    assert all(
        port.startswith("127.0.0.1:")
        for service in (api, services["frontend"])
        for port in service["ports"]
    )
    for service in services.values():
        assert service["user"] == "10001:10001" and service["read_only"]
        assert service["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in service["security_opt"]


def test_preparation_is_explicit_and_assets_are_shared_correctly():
    services = yaml.safe_load(Path("docker-compose.yml").read_text())["services"]
    for name in ("prepare-data", "train-models", "prepare-rag"):
        assert services[name]["profiles"] == ["prepare"]
        assert services[name]["healthcheck"]["disable"]
    volumes = services["api"]["volumes"]
    assert "model_bundles:/var/lib/smarthealth/models:ro" in volumes
    assert "e5_cache:/var/lib/smarthealth/embeddings:ro" in volumes
    assert "knowledge_index:/var/lib/smarthealth/knowledge" in volumes
    assert "--download-model" in services["prepare-rag"]["command"]


def test_image_context_excludes_models_credentials_and_logs():
    rules = Path(".dockerignore").read_text().splitlines()
    assert "**" in rules and "**/.env" in rules and "**/*.joblib" in rules and "**/*.onnx" in rules
    assert not any(
        line.startswith(("!logs", "!data/generated", "!models_saved", "!data/memory"))
        for line in rules
    )
    dockerfile = Path("Dockerfile").read_text()
    assert "libgomp1" in dockerfile and "USER 10001:10001" in dockerfile
    assert "COPY . " not in dockerfile
