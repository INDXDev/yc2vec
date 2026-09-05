# YC2Vec task aliases. Every target is a thin wrapper around the CLI, so
# anything here can also be run directly with `uv run yc2vec ...`.

.DEFAULT_GOAL := help
PROFILE ?= balanced
DATA_DIR ?= ./data
FIXTURE_DATA ?= /tmp/yc2vec-fixture

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# -- setup -------------------------------------------------------------------

.PHONY: install
install: ## Install the pipeline and the frontend
	uv sync --all-extras
	cd frontend && npm ci

.PHONY: doctor
doctor: ## Check Ollama, models and hardware against the profile
	uv run yc2vec doctor --profile $(PROFILE)

# -- pipeline ----------------------------------------------------------------

.PHONY: fetch
fetch: ## Fetch and normalise public company records
	uv run yc2vec fetch --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: discover
discover: ## Discover candidate semantic tags
	uv run yc2vec discover-tags --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: review
review: ## Adjudicate merges and activate candidate tags
	uv run yc2vec review-tags --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: assign
assign: ## Judge company/tag pairs with evidence
	uv run yc2vec assign-tags --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: embed
embed: ## Build vectors and precompute neighbours
	uv run yc2vec embed --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: project
project: ## Fit the UMAP projection and label clusters
	uv run yc2vec project --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: publish
publish: ## Write browser artifacts and CSV/Parquet exports
	uv run yc2vec publish-data --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: validate
validate: ## Run every release gate
	uv run yc2vec validate --profile $(PROFILE) --data-dir $(DATA_DIR)

.PHONY: run
run: ## Run the whole pipeline incrementally
	uv run yc2vec run --profile $(PROFILE) --data-dir $(DATA_DIR) --incremental

.PHONY: fixture
fixture: ## Run the whole pipeline on committed fixtures (no model needed)
	uv run yc2vec run --profile fixture --data-dir $(FIXTURE_DATA)
	uv run yc2vec validate --profile fixture --data-dir $(FIXTURE_DATA)

# -- frontend ----------------------------------------------------------------

.PHONY: site-data
site-data: ## Copy the current dataset into the frontend
	mkdir -p frontend/public/data
	rm -rf frontend/public/data/v1
	cp -r $(DATA_DIR)/public/v1 frontend/public/data/v1
	cp $(DATA_DIR)/quality.json frontend/public/data/v1/quality.json

.PHONY: dev
dev: ## Run the frontend dev server
	cd frontend && npm run dev

.PHONY: build
build: ## Production build for the GitHub Pages subpath
	cd frontend && VITE_BASE_PATH=/yc2vec/ npm run build

.PHONY: preview
preview: build ## Build and serve the production bundle locally
	cd frontend && npm run preview

# -- quality -----------------------------------------------------------------

.PHONY: lint
lint: ## Lint and type-check everything
	uv run ruff check pipeline tests
	uv run ruff format --check pipeline tests
	uv run mypy pipeline
	cd frontend && npm run typecheck

.PHONY: fmt
fmt: ## Auto-format
	uv run ruff format pipeline tests
	uv run ruff check --fix pipeline tests

.PHONY: test
test: ## Run all tests
	uv run pytest -q
	cd frontend && npm run test

.PHONY: schemas
schemas: ## Regenerate JSON Schema from the typed models
	uv run yc2vec schemas

.PHONY: check
check: lint test fixture ## Everything CI runs
