SHELL := /bin/sh
UV ?= uv
NPM ?= npm

.PHONY: help install lint test build check compose-config

help:
	@echo "OpenLLMOps 常用命令"
	@echo "  make install         安装所有开发依赖"
	@echo "  make lint            执行 Python 静态与格式检查"
	@echo "  make test            执行全部单元/集成测试"
	@echo "  make build           构建 Vue 生产包"
	@echo "  make check           依次执行 lint、test、build"
	@echo "  make compose-config  校验 deploy/compose.yaml（需要 deploy/.env）"

install:
	cd backend && $(UV) sync --all-groups
	cd agent && $(UV) sync --extra test
	cd evaluation && $(UV) sync --extra dev
	cd workers/model_importer && $(UV) sync --extra dev
	cd workers/training_config && $(UV) sync --extra dev
	cd workers/training_runtime && $(UV) sync --extra dev
	cd workers/artifacts && $(UV) sync --extra dev
	cd frontend && $(NPM) ci

lint:
	cd backend && $(UV) run ruff check . && $(UV) run ruff format --check .
	cd agent && $(UV) run --with ruff==0.16.4 ruff check . && $(UV) run --with ruff==0.16.4 ruff format --check .
	cd evaluation && $(UV) run --extra dev ruff check . && $(UV) run --extra dev ruff format --check .
	cd workers/model_importer && $(UV) run --extra dev ruff check . && $(UV) run --extra dev ruff format --check .
	cd workers/training_config && $(UV) run --extra dev ruff check . && $(UV) run --extra dev ruff format --check .
	cd workers/training_runtime && $(UV) run --extra dev ruff check . && $(UV) run --extra dev ruff format --check .
	cd workers/artifacts && $(UV) run --extra dev ruff check . && $(UV) run --extra dev ruff format --check .
	$(UV) run --with ruff==0.16.4 ruff check scripts deploy/scripts/*.py
	$(UV) run --with ruff==0.16.4 ruff format --check scripts deploy/scripts/*.py

test:
	cd backend && $(UV) run pytest -q
	cd agent && $(UV) run --extra test pytest -q
	cd evaluation && $(UV) run --extra dev pytest -q
	cd workers/model_importer && $(UV) run --extra dev pytest -q
	cd workers/training_config && $(UV) run --extra dev pytest -q
	cd workers/training_runtime && $(UV) run --extra dev pytest -q
	cd workers/artifacts && $(UV) run --extra dev pytest -q
	cd frontend && $(NPM) run test:unit

build:
	cd frontend && $(NPM) run build

compose-config:
	docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet

check: lint test build
