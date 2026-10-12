"""Integrity checks for the manually reviewed, attribution-preserving corpus release."""

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"


def test_knowledge_release_hashes_and_source_allowlist_are_consistent():
    raw = (KNOWLEDGE / "corpus.json").read_bytes()
    corpus = json.loads(raw)
    manifest = json.loads((KNOWLEDGE / "manifest.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == manifest["corpus_file_sha256"]
    assert corpus["schema_version"] == manifest["schema_version"] == "1.0"
    assert corpus["knowledge_version"] == manifest["knowledge_version"]
    documents = corpus["documents"]
    ids = [doc["source"]["id"] for doc in documents]
    urls = [doc["source"]["url"] for doc in documents]
    assert len(ids) == len(set(ids)) == 8
    assert len(urls) == len(set(urls))
    assert urls == corpus["allowed_source_urls"]
    hashes = {entry["source_id"]: entry for entry in manifest["sources"]}
    assert set(hashes) == set(ids)
    for doc in documents:
        source = doc["source"]
        assert re.fullmatch(r"[a-z][a-z0-9_]*", source["id"])
        parsed = urlsplit(source["url"])
        assert parsed.scheme == "https" and parsed.hostname
        assert (
            not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
        )
        assert date.fromisoformat(doc["accessed_at"]) == date(2026, 9, 8)
        content_hash = hashlib.sha256(doc["content"].encode("utf-8")).hexdigest()
        assert doc["content_hash"] == hashes[source["id"]]["content_hash"] == content_hash
        attribution = {key: doc[key] for key in ("source", "locator", "accessed_at", "rights")}
        canonical = json.dumps(
            attribution, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        assert (
            hashlib.sha256(canonical).hexdigest() == hashes[source["id"]]["source_metadata_sha256"]
        )


def test_curated_notes_cover_required_topics_and_distinguish_project_scope():
    corpus = json.loads((KNOWLEDGE / "corpus.json").read_text())
    documents = corpus["documents"]
    assert {doc["source"]["source_type"] for doc in documents} == {
        "signal_reference",
        "sensor_documentation",
        "guideline",
        "research_paper",
    }
    combined = " ".join(doc["content"] for doc in documents)
    assert all(
        token in combined
        for token in ("heart_rate", "EDA", "EEG_feature", "activity", "sleep_quality", "PPG")
    )
    for doc in documents:
        assert doc["content"].startswith("来源事实：")
        assert doc["content"].count("\n项目适用边界：") == 1
        assert 1 <= len(doc["content"]) <= 400
        assert doc["locator"] and doc["rights"]
    paper = next(doc for doc in documents if doc["source"]["source_type"] == "research_paper")
    assert (
        "Smets" in paper["rights"]
        and "https://creativecommons.org/licenses/by/4.0/" in paper["rights"]
    )
    assert "改写" in paper["rights"]
