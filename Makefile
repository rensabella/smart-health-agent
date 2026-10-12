PYTHON ?= python

.PHONY: install dev data train api ui test lint format check build package-check docker-build docker-up docker-down

install:
	$(PYTHON) -m pip install -r requirements.txt
dev:
	$(PYTHON) -m pip install -e '.[dev,ui]'
data:
	$(PYTHON) -m smarthealth.cli.generate_demo_data
train:
	$(PYTHON) -m smarthealth.cli.train
api:
	$(PYTHON) -m smarthealth.cli.serve
ui:
	$(PYTHON) -m smarthealth.cli.ui
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check src app skills frontend scripts evaluation examples tests train.py generate_demo_data.py
	$(PYTHON) -m ruff format --check src app skills frontend scripts evaluation examples tests train.py generate_demo_data.py
format:
	$(PYTHON) -m ruff format src app skills frontend scripts evaluation examples tests train.py generate_demo_data.py
check: lint test
	$(PYTHON) -m pip check
build:
	$(PYTHON) -m build --wheel
package-check: build
	$(PYTHON) scripts/check_wheel.py --wheel dist/smart_health_agent-0.2.0-py3-none-any.whl
docker-build:
	docker compose build
docker-up:
	docker compose up -d api frontend
docker-down:
	docker compose down
