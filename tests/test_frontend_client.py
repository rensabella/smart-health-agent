import hashlib
import json
from pathlib import Path

import httpx
import pytest

from app.agents.contracts import RunSnapshot
from frontend.client import (
    ClientError,
    HealthApiClient,
    ProgressTracker,
    StreamMessage,
    approved_report,
    parse_sse,
)


@pytest.fixture
def reviewed_snapshot():
    return RunSnapshot.model_validate_json(Path("docs/report_example/run.json").read_text())


def message(run_id, index, event, data=None):
    return StreamMessage(
        event,
        index,
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "event_id": index,
            "timestamp": "2026-09-09T00:00:00+00:00",
            "event": event,
            "data": data or {},
        },
    )


def test_report_body_requires_completed_snapshot_and_matching_publication(reviewed_snapshot):
    assert approved_report(reviewed_snapshot)
    assert approved_report(reviewed_snapshot.model_copy(update={"status": "verifying"})) is None
    assert approved_report(reviewed_snapshot.model_copy(update={"result_available": False})) is None
    changed = reviewed_snapshot.model_copy(deep=True)
    changed.final_result["publication"]["review_ref"] = "art_" + "0" * 32
    with pytest.raises(ClientError, match="publication_not_verified"):
        approved_report(changed)


def test_sse_progress_never_displays_prepublication_text(reviewed_snapshot):
    report = approved_report(reviewed_snapshot)
    tracker = ProgressTracker(reviewed_snapshot.run_id)
    data = {
        "chunk_index": 0,
        "report_hash": hashlib.sha256(report.body.encode()).hexdigest(),
        "content_available": True,
        "text": report.body[:400],
    }
    tracker.apply(message(tracker.run_id, 1, "report.chunk", data))
    assert tracker.checked_prefix(report) == "" and not tracker.chunks
    tracker.apply(
        message(tracker.run_id, 2, "publication.checked", {"result_kind": "health_report"})
    )
    tracker.apply(message(tracker.run_id, 3, "report.chunk", data))
    assert tracker.checked_prefix(report) == report.body[:400]
    tracker.apply(message(tracker.run_id, 3, "report.chunk", data))
    assert tracker.cursor == 3 and len(tracker.chunks) == 1


def test_sse_hash_mismatch_and_expired_content_fail_closed(reviewed_snapshot):
    report = approved_report(reviewed_snapshot)
    tracker = ProgressTracker(reviewed_snapshot.run_id)
    tracker.apply(
        message(tracker.run_id, 1, "publication.checked", {"result_kind": "health_report"})
    )
    tracker.apply(
        message(
            tracker.run_id,
            2,
            "report.chunk",
            {
                "chunk_index": 0,
                "report_hash": "0" * 64,
                "content_available": True,
                "text": report.body[:400],
            },
        )
    )
    with pytest.raises(ClientError, match="report_chunk_mismatch"):
        tracker.checked_prefix(report)
    tracker.apply(
        message(
            tracker.run_id,
            3,
            "report.chunk",
            {
                "chunk_index": 1,
                "report_hash": "0" * 64,
                "content_available": False,
                "text": None,
            },
        )
    )
    assert not tracker.content_available and not tracker.chunks
    assert tracker.checked_prefix(report) == ""


def test_sse_gap_resets_publication_gate_and_cross_run_is_rejected():
    tracker = ProgressTracker("run1")
    tracker.apply(message("run1", 1, "publication.checked", {"result_kind": "health_report"}))
    tracker.apply(message("run1", 5, "run.completed"))
    assert tracker.replay_gap and not tracker.publication_checked
    with pytest.raises(ClientError, match="invalid_stream"):
        tracker.apply(message("other", 6, "run.completed"))


def test_sse_parser_does_not_commit_truncated_frame():
    assert list(parse_sse(iter(["event: run.completed", "id: 1", 'data: {"ok":true}']))) == []
    parsed = list(
        parse_sse(
            iter([": keepalive", "", "event: run.completed", "id: 1", 'data: {"ok":true}', ""])
        )
    )
    assert len(parsed) == 1 and parsed[0].event_id == 1


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/data",
        "https://user:secret@example.org",
        "http://localhost/x",
        "http://localhost?token=secret",
    ],
)
def test_api_address_rejects_embedded_secrets_or_untrusted_paths(url):
    with pytest.raises(ClientError) as error:
        HealthApiClient(url)
    assert "secret" not in str(error.value)


def test_client_never_follows_redirects_or_exposes_error_body():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer private-test-token"
        return httpx.Response(
            302, headers={"location": "https://external.example/leak"}, text="private response"
        )

    with HealthApiClient(
        "http://api:8000", "private-test-token", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ClientError) as error:
            client.health()
    assert len(seen) == 1 and "private" not in str(error.value)


def test_images_use_validated_refs_not_server_supplied_urls():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=b"\x89PNG\r\n\x1a\nfixture"
        )

    with HealthApiClient("http://api:8000", transport=httpx.MockTransport(handler)) as client:
        assert client.image("art_" + "a" * 32, "session1", "v1").startswith(b"\x89PNG")
        with pytest.raises(ClientError):
            client.image("https://external.example/file", "session1", "v1")
    assert len(seen) == 1 and seen[0].url.host == "api"
    assert seen[0].url.params["session_id"] == "session1"


def test_sse_reconnect_sends_only_last_complete_cursor():
    event = message("run1", 6, "run.completed")

    def handler(request):
        assert request.headers["last-event-id"] == "5"
        frame = f"id: 6\nevent: run.completed\ndata: {json.dumps(event.payload)}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=frame)

    with HealthApiClient("http://api:8000", transport=httpx.MockTransport(handler)) as client:
        batch = client.events("run1", "session1", after_id=5)
    assert len(batch.messages) == 1 and batch.messages[0].event_id == 6
