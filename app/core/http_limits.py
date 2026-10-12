"""Bound request bodies before the framework can spool multipart uploads to disk."""

from starlette.responses import JSONResponse

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_MULTIPART_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
UPLOAD_PATHS = frozenset({"/predict", "/analyze", "/uploads"})
MAX_AGENT_JSON_BYTES = 32 * 1024
AGENT_JSON_PATHS = frozenset(
    {"/agent/runs", "/agent/route", "/memory/search", "/memory/preferences"}
)


class UploadBodyLimitMiddleware:
    def __init__(self, app, max_bytes: int = MAX_MULTIPART_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] not in {"POST", "PUT"}
            or scope["path"] not in UPLOAD_PATHS | AGENT_JSON_PATHS
        ):
            await self.app(scope, receive, send)
            return
        limit = MAX_AGENT_JSON_BYTES if scope["path"] in AGENT_JSON_PATHS else self.max_bytes

        async def reject(code, message, status):
            response = JSONResponse(
                status_code=status,
                content={
                    "schema_version": "1.0",
                    "request_id": scope.get("state", {}).get("request_id", ""),
                    "error": {"code": code, "message": message},
                },
            )
            await response(scope, receive, send)

        lengths = [v for k, v in scope["headers"] if k.lower() == b"content-length"]
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit():
                await reject("invalid_input", "Invalid Content-Length header.", 400)
                return
            # Compare digits first: do not parse unbounded client-supplied integers.
            length = lengths[0].lstrip(b"0") or b"0"
            if len(length) > len(str(limit)) or int(length) > limit:
                await reject("file_too_large", "Request body exceeds the endpoint limit.", 413)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                await reject("file_too_large", "Request body exceeds the endpoint limit.", 413)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def bounded_receive():
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)
