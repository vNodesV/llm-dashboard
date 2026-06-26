.DEFAULT_GOAL := help
PYTHON        := python3
VENV          := .venv
PIP           := $(VENV)/bin/pip
LLM_WATCH     := $(VENV)/bin/python llm-watch.py

.PHONY: help venv install run lint clean

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

venv:          ## Create virtualenv
	$(PYTHON) -m venv $(VENV)

install: venv  ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install   ## Run llm-watch (auto-detect source)
	$(LLM_WATCH)

run-log: install  ## Run with explicit log file (LOG= required)
	$(LLM_WATCH) --log $(LOG)

run-prometheus: install  ## Force Prometheus mode
	$(LLM_WATCH) --host $(HOST) --port $(PORT)

lint:          ## Lint with ruff (if installed)
	$(VENV)/bin/ruff check . 2>/dev/null || echo "ruff not installed — pip install ruff"

clean:         ## Remove virtualenv and caches
	rm -rf $(VENV) __pycache__ *.pyc .ruff_cache

# Quick install without venv (system Python)
install-sys:   ## Install deps into system Python
	$(PYTHON) -m pip install -r requirements.txt

run-sys:       ## Run without venv (system Python)
	$(PYTHON) llm-watch.py
