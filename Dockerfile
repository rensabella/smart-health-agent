# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/smarthealth-cache \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    HEALTH_MODEL_DIR=/var/lib/smarthealth/models/lightgbm_v1.0 \
    HEALTH_MEMORY_DIR=/var/lib/smarthealth/memory/qdrant \
    HEALTH_AGENT_TRACE_DIR=/var/lib/smarthealth/traces/agent_runs

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 smarthealth \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin smarthealth

WORKDIR /opt/smarthealth
COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY skills/ ./skills/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/
COPY evaluation/ ./evaluation/
COPY examples/ ./examples/
COPY configs/ ./configs/
COPY .streamlit/ ./.streamlit/
COPY knowledge/ ./knowledge/
COPY train.py generate_demo_data.py ./
COPY data/sample/demo.csv ./data/sample/demo.csv

# Train and serve with this same dependency environment. The macOS snapshot is
# intentionally not used as a Linux lock file. No model download occurs here.
RUN python -m pip install --no-cache-dir '.[ui]' \
    && python -m pip check \
    && mkdir -p /var/lib/smarthealth/datasets /var/lib/smarthealth/models \
        /var/lib/smarthealth/embeddings /var/lib/smarthealth/knowledge \
        /var/lib/smarthealth/memory /var/lib/smarthealth/traces \
    && touch /var/lib/smarthealth/datasets/.volume-owner \
        /var/lib/smarthealth/models/.volume-owner \
        /var/lib/smarthealth/embeddings/.volume-owner \
        /var/lib/smarthealth/knowledge/.volume-owner \
        /var/lib/smarthealth/memory/.volume-owner \
        /var/lib/smarthealth/traces/.volume-owner \
    && chown -R 10001:10001 /var/lib/smarthealth \
    && chmod 0700 /var/lib/smarthealth/*

USER 10001:10001
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).read()"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16", "--no-access-log", "--log-config", "configs/logging.json"]
