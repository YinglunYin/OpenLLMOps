SHELL := /bin/sh

.PHONY: help install lint test build check

help:
	@echo "OpenLLMOps 常用命令"
	@echo "  make install  安装前后端依赖"
	@echo "  make lint     执行静态检查"
	@echo "  make test     执行自动化测试"
	@echo "  make build    构建前端"
	@echo "  make check    依次执行 lint、test、build"

install:
	python3 -m pip install -e "./backend[dev]"
	cd frontend && npm install

lint:
	python3 -m ruff check backend
	cd frontend && npm run lint

test:
	python3 -m pytest backend/tests
	cd frontend && npm run test:unit

build:
	cd frontend && npm run build

check: lint test build
