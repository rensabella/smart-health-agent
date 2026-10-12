"""Verify the explicit loopback fixture's SSE disconnect/replay behavior with real HTTP."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx


def check(output=Path("data/generated/http_smoke.json")):
    headers = {"Authorization": "Bearer local-sse-test", "X-Request-ID": "agent-sse-smoke"}
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=60, trust_env=False) as client:
        uploaded = client.post(
            "/uploads",
            files={"file": ("demo.csv", Path("data/sample/demo.csv").read_bytes())},
            headers=headers,
        )
        uploaded.raise_for_status()
        upload = uploaded.json()
        body = {
            "query": "Analyze this synthetic CSV data.",
            "session_id": upload["session_id"],
            "input_version": upload["input_version"],
            "upload_refs": [upload["upload_ref"]],
            "entry_agent": "planner",
            "include_visualizations": True,
        }
        initial = client.post(
            "/agent/runs", json=body, headers={**headers, "Idempotency-Key": "sse-smoke"}
        )
        initial.raise_for_status()
        task = initial.json()
        assert task["backend_mode"] == "scripted"
        assert client.get(task["events_url"]).status_code == 401
        first_id = None
        with client.stream("GET", task["events_url"], headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("id: "):
                    first_id = int(line[4:])
                    break
        snapshot_url = f"/agent/runs/{task['run_id']}?session_id={task['session_id']}"
        after_disconnect = client.get(snapshot_url, headers=headers).json()
        assert after_disconnect["status"] in {"queued", "running", "verifying", "completed"}
        events = []
        with client.stream(
            "GET", task["events_url"], headers={**headers, "Last-Event-ID": str(first_id)}
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        assert events and events[0]["event_id"] == first_id + 1
        assert events[-1]["event"] == "run.completed"
        assert [e["event_id"] for e in events] == list(
            range(first_id + 1, events[-1]["event_id"] + 1)
        )
        final = client.get(snapshot_url, headers=headers).json()
        for _ in range(100):
            if final["memory_status"] != "pending":
                break
            time.sleep(0.05)
            final = client.get(snapshot_url, headers=headers).json()
        assert (
            final["status"] == "completed"
            and final["tool_calls"] == 6
            and final["llm_requests"] == 0
        )
        image_url = final["final_result"]["analysis"]["visualization"][0]["url"]
        assert client.get(image_url, headers=headers).content.startswith(b"\x89PNG")
        repeated = client.post(
            "/agent/runs", json=body, headers={**headers, "Idempotency-Key": "sse-smoke"}
        ).json()
        assert repeated["run_id"] == task["run_id"]
        assert final["memory_status"] == "stored"
        harness = next(e for e in events if e["event"] == "harness.loaded")
        publication = final["final_result"]["publication"]
        assert (
            harness["data"]["harness_hash"] == final["harness_hash"] == publication["harness_hash"]
        )
        assert publication["medical_review"] == "not_performed"
        assert publication["evidence"] and publication["uncertainty"]
        memory_session = client.get(
            f"/memory/sessions/{task['session_id']}", headers=headers
        ).json()
        assert memory_session["current_result_run_id"] == task["run_id"]
        memory_hits = client.post(
            "/memory/search", json={"query": "risk 风险"}, headers=headers
        ).json()
        assert len(memory_hits) == 1 and memory_hits[0]["entry"]["source_run_id"] == task["run_id"]
        report = {
            "mode": "scripted_decisions_real_http_real_lightgbm",
            "external_llm_calls": 0,
            "status_after_disconnect": after_disconnect["status"],
            "status": final["status"],
            "first_event_id": first_id,
            "last_event_id": events[-1]["event_id"],
            "replay_contiguous": True,
            "idempotency_verified": True,
            "png_downloaded": True,
            "decisions": final["decision_count"],
            "tool_calls": final["tool_calls"],
            "delegations": final["delegation_count"],
            "workflow_type": final["route"]["workflow_type"],
            "route_traced": any(e["event"] == "route.selected" for e in events),
            "memory_status": final["memory_status"],
            "memory_loaded_traced": any(e["event"] == "memory.loaded" for e in events),
            "memory_search_hits": len(memory_hits),
            "harness_version": final["harness_version"],
            "harness_hash": final["harness_hash"],
            "publication_policy": publication["policy"],
            "publication_evidence_verified": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/generated/http_smoke.json"))
    print(json.dumps(check(parser.parse_args().output), indent=2))
