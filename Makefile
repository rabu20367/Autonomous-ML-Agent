# Makefile for Autonomous ML Agent

.PHONY: help install test lint format clean build docs

help: ## Show this help message
	@echo "Autonomous ML Agent - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=autonomous_ml --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 autonomous_ml/ tests/
	mypy autonomous_ml/

format: ## Format code
	black autonomous_ml/ tests/ examples/
	isort autonomous_ml/ tests/ examples/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

run-example: ## Run basic usage example
	python examples/basic_usage.py

run-advanced: ## Run advanced usage example
	python examples/advanced_usage.py

setup-env: ## Set up development environment
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make install-dev"

ci: lint test ## Run CI pipeline (lint + test)

all: clean install test lint format ## Run everything (clean, install, test, lint, format)
