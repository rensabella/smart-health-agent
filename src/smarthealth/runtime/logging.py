"""Application-owned JSON logs without changing the host application's root logger."""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        try:
            payload = json.loads(record.getMessage())
        except (ValueError, TypeError):
            payload = {"message": record.getMessage()}
        if not isinstance(payload, dict):
            payload = {"message": record.getMessage()}
        payload.setdefault(
            "timestamp", datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        )
        payload.setdefault("level", record.levelname)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def configure_logging(level="INFO"):
    logger = logging.getLogger("smart_health")
    logger.setLevel(level)
    # Respect a process manager, uvicorn log_config, test capture, or embedding host.
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def log_event(event: str, **fields):
    logging.getLogger("smart_health").info(
        json.dumps({"event": event, **fields}, ensure_ascii=False)
    )
