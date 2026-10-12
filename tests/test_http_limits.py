import json

import pytest

from app.core.http_limits import UploadBodyLimitMiddleware


@pytest.mark.parametrize("content_length", [None, b"1", b"101"])
async def test_oversized_body_never_reaches_multipart_parser(content_length):
    reads, replies = [], []

    async def downstream(*_):
        pytest.fail("Multipart parser must not see an oversized body")

    async def receive():
        reads.append(True)
        return {"type": "http.request", "body": b"x" * 60, "more_body": True}

    async def send(message):
        replies.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [(b"content-length", content_length)] if content_length else [],
        "state": {"request_id": "bounded-request"},
    }
    await UploadBodyLimitMiddleware(downstream, max_bytes=100)(scope, receive, send)
    assert len(reads) == (0 if content_length == b"101" else 2)
    assert replies[0]["status"] == 413
    assert json.loads(replies[1]["body"])["request_id"] == "bounded-request"


async def test_bounded_body_replayed_without_modification():
    chunks = iter(
        [
            {"type": "http.request", "body": b"first", "more_body": True},
            {"type": "http.request", "body": b"last", "more_body": False},
        ]
    )
    observed = []

    async def receive():
        return next(chunks)

    async def downstream(scope, bounded_receive, send):
        observed.append(await bounded_receive())

    scope = {"type": "http", "method": "POST", "path": "/uploads", "headers": []}
    await UploadBodyLimitMiddleware(downstream, max_bytes=9)(scope, receive, None)
    assert b"".join(m["body"] for m in observed) == b"firstlast"
