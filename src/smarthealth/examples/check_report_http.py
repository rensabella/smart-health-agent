"""Check reviewed report delivery and reconnect with an actual loopback HTTP server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx


def check(output=Path("data/generated/report_example/http_smoke.json")):
    headers = {"Authorization": "Bearer local-report-test", "X-Request-ID": "report-http-smoke"}
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=90, trust_env=False) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        uploaded = client.post(
            "/uploads",
            files={"file": ("demo.csv", Path("data/sample/demo.csv").read_bytes())},
            headers=headers,
        )
        uploaded.raise_for_status()
        upload = uploaded.json()
        body = {
            "query": "分析合成CSV并生成有来源的健康分析报告和图表",
            "session_id": upload["session_id"],
            "input_version": upload["input_version"],
            "upload_refs": [upload["upload_ref"]],
            "include_visualizations": True,
        }
        response = client.post(
            "/agent/runs", json=body, headers={**headers, "Idempotency-Key": "report-smoke"}
        )
        response.raise_for_status()
        task = response.json()
        assert task["backend_mode"] == "scripted"
        assert client.get(task["events_url"]).status_code == 401
        first_id = None
        with client.stream("GET", task["events_url"], headers=headers) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                if line.startswith("id: "):
                    first_id = int(line[4:])
                    break
        snapshot_url = f"/agent/runs/{task['run_id']}?session_id={task['session_id']}"
        disconnected_status = client.get(snapshot_url, headers=headers).json()["status"]
        events = []
        with client.stream(
            "GET", task["events_url"], headers={**headers, "Last-Event-ID": str(first_id)}
        ) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        assert events[-1]["event"] == "run.completed", events[-1]
        assert [e["event_id"] for e in events] == list(
            range(first_id + 1, events[-1]["event_id"] + 1)
        )
        final = client.get(snapshot_url, headers=headers).json()
        for _ in range(100):
            if final["memory_status"] != "pending":
                break
            time.sleep(0.05)
            final = client.get(snapshot_url, headers=headers).json()
        assert final["status"] == "completed" and final["memory_status"] == "stored"
        assert final["tool_calls"] == 9 and final["llm_requests"] == 0
        result = final["final_result"]
        report = result["report"]
        assert result["review_mode"] == "scripted_test" and report["safety"]["safe"]
        publication_id = next(e["event_id"] for e in events if e["event"] == "publication.checked")
        chunks = [e for e in events if e["event"] == "report.chunk"]
        assert chunks and all(e["event_id"] > publication_id for e in chunks)
        assert all(e["data"]["content_available"] for e in chunks)
        assert "".join(e["data"]["text"] for e in chunks) == report["body"]
        report_hash = hashlib.sha256(report["body"].encode()).hexdigest()
        assert all(e["data"]["report_hash"] == report_hash for e in chunks)
        assert all(c["sources"] for c in report["claims"])
        replay = client.get(
            task["events_url"],
            headers={**headers, "Last-Event-ID": str(chunks[0]["event_id"] - 1)},
        )
        replay.raise_for_status()
        replayed = [
            json.loads(line[6:]) for line in replay.text.splitlines() if line.startswith("data: ")
        ]
        assert [e for e in replayed if e["event"] == "report.chunk"] == chunks
        trace = client.get(
            f"/agent/runs/{task['run_id']}/trace?session_id={task['session_id']}", headers=headers
        )
        trace.raise_for_status()
        assert all("text" not in e["data"] for e in trace.json()["events"])
        assert all(c["text"] not in trace.text for c in report["claims"])
        roles = sorted({e["agent"] for e in events if e["event"] == "agent.started"})
        assert roles == ["data_analysis", "planner", "rag", "report_writer", "safety_verification"]
        repeated = client.post(
            "/agent/runs", json=body, headers={**headers, "Idempotency-Key": "report-smoke"}
        )
        repeated.raise_for_status()
        assert repeated.json()["run_id"] == task["run_id"]
        plots = result["analysis"]["visualization"]
        assert plots
        for plot in plots:
            image = client.get(plot["url"], headers=headers)
            assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
        harness_hash = final["harness_hash"]
        assert (
            report["safety"]["policy_hash"] == result["publication"]["harness_hash"] == harness_hash
        )
        record = {
            "mode": "scripted_roles_real_http_e5_qdrant_lightgbm",
            "external_llm_calls": 0,
            "status_after_disconnect": disconnected_status,
            "status": final["status"],
            "roles": roles,
            "decisions": final["decision_count"],
            "tool_calls": final["tool_calls"],
            "delegations": final["delegation_count"],
            "last_event_id": events[-1]["event_id"],
            "report_chunk_count": len(chunks),
            "report_hash": report_hash,
            "publication_precedes_body": True,
            "replay_contiguous": True,
            "report_replay_identical": True,
            "trace_excludes_body": True,
            "idempotency_verified": True,
            "png_download_count": len(plots),
            "memory_status": final["memory_status"],
            "review_mode": result["review_mode"],
            "harness_hash": harness_hash,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n")
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/generated/report_example/http_smoke.json")
    )
    print(json.dumps(check(parser.parse_args().output), indent=2))
