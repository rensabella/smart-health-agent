"""Start the single-worker HTTP service with validated environment configuration."""

import argparse

from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings


def main():
    import uvicorn

    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    args = parser.parse_args()
    uvicorn.run(
        "smarthealth.api.application:app",
        host=args.host,
        port=args.port,
        workers=1,
        limit_concurrency=16,
        access_log=False,
        log_config=str(resource_path("configs", "logging.json")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
