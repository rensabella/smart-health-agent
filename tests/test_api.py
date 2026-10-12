import json

from fastapi.testclient import TestClient

from app.main import create_app


def test_api_prediction_analysis_and_protected_images(predictor, csv_bytes):
    with TestClient(create_app(predictor=predictor, api_token="test-token")) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        assert client.post("/predict", files={"file": ("demo.csv", csv_bytes)}).status_code == 401
        headers = {"Authorization": "Bearer test-token", "X-Request-ID": "request-123"}
        response = client.post("/predict", files={"file": ("demo.csv", csv_bytes)}, headers=headers)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["request_id"] == response.headers["X-Request-ID"] == "request-123"
        assert 0 <= result["risk_score"] <= 1
        assert result["model"]["name"] == "RandomForest"
        assert result["window_count"] == 4
        analyzed = client.post(
            "/analyze?include_visualizations=true",
            files={"file": ("demo.csv", csv_bytes)},
            headers=headers,
        )
        assert analyzed.status_code == 200, analyzed.text
        report = analyzed.json()
        assert report["prediction"]["risk_score"] == result["risk_score"]
        assert report["summary"] and report["explanation"]["local_evidence"]
        assert len(report["visualization"]) == 3
        image_url = report["visualization"][0]["url"]
        assert client.get(image_url).status_code == 401
        png = client.get(image_url, headers=headers)
        assert png.status_code == 200 and png.content.startswith(b"\x89PNG")
        assert (
            client.get(
                image_url.replace(report["session_id"], "another"), headers=headers
            ).status_code
            == 403
        )


def test_model_loaded_once_across_requests(bundles, csv_bytes):
    from app.models.model_loader import load_model

    loads = []

    def loader(path):
        loads.append(path)
        return load_model(path)

    with TestClient(create_app(model_dir=bundles["random_forest"], model_loader=loader)) as client:
        for _ in range(2):
            assert (
                client.post("/predict", files={"file": ("demo.csv", csv_bytes)}).status_code == 200
            )
    assert len(loads) == 1


def test_missing_model_readiness_and_inference(tmp_path):
    with TestClient(create_app(model_dir=tmp_path / "missing")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", files={"file": ("demo.csv", b"anything")})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"


def test_bad_requests_have_safe_error_schema(predictor, csv_bytes, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="smart_health")
    with TestClient(create_app(predictor=predictor)) as client:
        response = client.post(
            "/predict", files={"file": ("SECRET_FILENAME.csv", b"secret,column\n42,33")}
        )
        assert response.status_code == 422
        assert set(response.json()) == {"schema_version", "request_id", "error"}
        assert client.post("/predict").status_code == 422
        huge = client.post("/predict", files={"file": ("large.csv", b"x" * (5 * 1024 * 1024 + 1))})
        assert huge.status_code == 413
    logs = "\n".join(record.message for record in caplog.records if record.name == "smart_health")
    assert "SECRET_FILENAME" not in logs and "secret,column" not in logs
    events = [
        json.loads(record.message) for record in caplog.records if record.name == "smart_health"
    ]
    assert all(
        "request_id" in e and "latency_ms" in e for e in events if e["event"] == "http.completed"
    )


def test_http_log_links_to_skill_trace(predictor, csv_bytes, caplog):
    import logging

    from app.core.context import SkillContext

    caplog.set_level(logging.INFO, logger="smart_health")
    app = create_app(predictor=predictor)
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("demo.csv", csv_bytes)})
        assert response.status_code == 200
        body = response.json()
        event = next(
            json.loads(record.message)
            for record in caplog.records
            if record.name == "smart_health"
            and json.loads(record.message).get("request_id") == body["request_id"]
        )
        context = SkillContext(
            owner_id="local_demo",
            session_id=body["session_id"],
            input_version=body["input_version"],
            run_id=event["run_id"],
        )
        events = app.state.registry.trace_sink.events_for(context)
        assert len([e for e in events if e["event"] == "skill.completed"]) == 4
        assert event["prediction"] == body["prediction"]
        assert "subject_splits" not in body["model"]["metadata"]
        assert "normal_reference" not in body["model"]["metadata"]
