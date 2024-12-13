# Set the PATH env var used by cargo and uv
export PATH := ${HOME}/.local/bin:${HOME}/.cargo/bin:$(PATH)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

update: download process_results generate_leaderboards  ## Download and process the latest results

download:
	@scp -o ConnectTimeout=5 percival:/home/alex-admin/scandeval/scandeval_benchmark_results.jsonl percival_results.jsonl || true
	@scp -o ConnectTimeout=5 lancelot:/home/alex-admin/scandeval/scandeval_benchmark_results.jsonl lancelot_results.jsonl || true
	@scp -o ConnectTimeout=5 blackknight:/home/alex-admin/scandeval/scandeval_benchmark_results.jsonl blackknight_results.jsonl || true
	@touch results.jsonl
	@if [ -f percival_results.jsonl ]; then \
		cat percival_results.jsonl >> results/results.jsonl; \
		rm percival_results.jsonl; \
	fi
	@if [ -f lancelot_results.jsonl ]; then \
		cat lancelot_results.jsonl >> results/results.jsonl; \
		rm lancelot_results.jsonl; \
	fi
	@if [ -f blackknight_results.jsonl ]; then \
		cat blackknight_results.jsonl >> results/results.jsonl; \
		rm blackknight_results.jsonl; \
	fi

process_results:
	@uv run src/process_results.py results/results.jsonl

generate_leaderboards:
	@uv run src/generate_leaderboards.py configs/danish.yaml

install: ## Install dependencies
	@echo "Installing the 'leaderboards' project..."
	@$(MAKE) --quiet install-rust
	@$(MAKE) --quiet install-uv
	@$(MAKE) --quiet install-dependencies
	@echo "Installed the 'leaderboards' project. If you want to use pre-commit hooks, run 'make install-pre-commit'."

install-rust:
	@if [ "$(shell which rustup)" = "" ]; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo "Installed Rust."; \
	fi

install-uv:
	@if [ "$(shell which uv)" = "" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
        echo "Installed uv."; \
    else \
		echo "Updating uv..."; \
		uv self update; \
	fi

install-dependencies:
	@uv python install 3.11
	@uv sync --all-extras --python 3.11

install-pre-commit:  ## Install pre-commit hooks
	@uv run pre-commit install

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

type-check:
	@uv run mypy . \
		--install-types \
		--non-interactive \
		--ignore-missing-imports \
		--show-error-codes \
		--check-untyped-defs

check: lint format type-check  ## Lint, format, and type-check the code
