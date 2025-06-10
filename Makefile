# Robust Stats - Statistical Analysis Toolkit Makefile
# A traditional Makefile for Rust workspace management

.PHONY: help build build-release test test-unit test-integration \
        clean fmt lint check doc install uninstall run \
        watch watch-test dev prod setup audit outdated \
        bench coverage release test-crate build-crate \
        test-all test-workspace fmt-check verify package \
        doc-all doc-open examples size stats tree deps \
        publish publish-dry-run clean-all

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m

# Project information
WORKSPACE_NAME := robust-stats
CARGO_TARGET_DIR := target
BUILD_DIR := $(CARGO_TARGET_DIR)

# Crate names
CRATES := robust-core robust-quantile robust-spread robust-confidence \
          robust-modality robust-changepoint robust-stability robust-viz robust-polars

# Help target
help: ## Show this help message
	@printf "$(PURPLE)Robust Stats - Statistical Analysis Toolkit$(NC)\n"
	@printf "$(BLUE)Available targets:$(NC)\n"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} \
		/^[a-zA-Z_-]+:.*##/ { \
			printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2 \
		}' $(MAKEFILE_LIST)
	@printf "\n$(BLUE)Workspace crates:$(NC)\n"
	@for crate in $(CRATES); do \
		printf "  $(CYAN)%-20s$(NC) %s\n" "$$crate" "crates/$$crate"; \
	done
	@printf "\n"

# Development workflow
dev: fmt lint test build ## Run complete development workflow (format, lint, test, build)
	@printf "$(GREEN) Development workflow completed successfully!$(NC)\n"

prod: fmt-check lint test-all build-release ## Run production workflow (format check, lint, test all, release build)
	@printf "$(GREEN) Production workflow completed successfully!$(NC)\n"

quick: fmt build test-unit ## Quick development cycle (format, build, unit tests only)
	@printf "$(GREEN) Quick check passed!$(NC)\n"

# Build targets
build: ## Build all crates in debug mode
	@printf "$(BLUE) Building all crates (debug)...$(NC)\n"
	cargo build --workspace

build-release: ## Build all crates in release mode
	@printf "$(BLUE) Building all crates (release)...$(NC)\n"
	cargo build --workspace --release

build-crate: ## Build specific crate (usage: make build-crate CRATE=robust-core)
	@if [ -z "$(CRATE)" ]; then \
		printf "$(RED) Please specify CRATE=<crate-name>$(NC)\n"; \
		exit 1; \
	fi
	@printf "$(BLUE) Building $(CRATE)...$(NC)\n"
	cargo build -p $(CRATE)

# Test targets
test: ## Run all tests with default features
	@printf "$(BLUE) Running all tests...$(NC)\n"
	cargo test --workspace

test-all: ## Run all tests with all features
	@printf "$(BLUE) Running all tests with all features...$(NC)\n"
	cargo test --workspace --all-features

test-simd: ## Run all tests with SIMD features enabled
	@printf "$(BLUE) Running all tests with SIMD enabled...$(NC)\n"
	cargo test --workspace --features simd

test-unit: ## Run unit tests only
	@printf "$(BLUE) Running unit tests...$(NC)\n"
	cargo test --workspace --lib

test-integration: ## Run integration tests only
	@printf "$(BLUE) Running integration tests...$(NC)\n"
	cargo test --workspace --test '*'

test-crate: ## Test specific crate (usage: make test-crate CRATE=robust-core)
	@if [ -z "$(CRATE)" ]; then \
		printf "$(RED) Please specify CRATE=<crate-name>$(NC)\n"; \
		exit 1; \
	fi
	@printf "$(BLUE) Testing $(CRATE)...$(NC)\n"
	cargo test -p $(CRATE) --all-features

test-doc: ## Test documentation examples
	@printf "$(BLUE) Testing documentation examples...$(NC)\n"
	cargo test --workspace --doc

# Code quality targets
fmt: ## Format all code
	@printf "$(BLUE) Formatting code...$(NC)\n"
	cargo fmt --all

fmt-check: ## Check formatting without modifying
	@printf "$(BLUE) Checking code format...$(NC)\n"
	cargo fmt --all -- --check

lint: ## Run clippy linter on all crates
	@printf "$(BLUE) Running linter...$(NC)\n"
	cargo clippy --workspace --all-targets --all-features -- -D warnings

lint-fix: ## Run clippy and automatically fix issues
	@printf "$(BLUE) Running linter in fix mode...$(NC)\n"
	cargo clippy --workspace --fix --all-targets --all-features --allow-dirty --allow-staged -- -D warnings

check: ## Quick check without building
	@printf "$(BLUE) Quick check...$(NC)\n"
	cargo check --workspace --all-targets --all-features

# Documentation
doc: ## Generate documentation for all crates
	@printf "$(BLUE) Generating documentation...$(NC)\n"
	cargo doc --workspace --all-features --no-deps

doc-open: doc ## Generate and open documentation
	@printf "$(BLUE) Opening documentation...$(NC)\n"
	cargo doc --workspace --all-features --no-deps --open

doc-all: ## Generate documentation including dependencies
	@printf "$(BLUE) Generating complete documentation...$(NC)\n"
	cargo doc --workspace --all-features

# Examples
examples: ## Build all examples
	@printf "$(BLUE) Building examples...$(NC)\n"
	cargo build --examples --all-features

run-example: ## Run specific example (usage: make run-example EXAMPLE=basic)
	@if [ -z "$(EXAMPLE)" ]; then \
		printf "$(RED) Please specify EXAMPLE=<example-name>$(NC)\n"; \
		exit 1; \
	fi
	@printf "$(BLUE) Running example: $(EXAMPLE)...$(NC)\n"
	cargo run --example $(EXAMPLE) --all-features

# Development tools
watch: ## Watch for changes and rebuild
	@printf "$(BLUE) Watching for changes...$(NC)\n"
	@if command -v cargo-watch >/dev/null 2>&1; then \
		cargo watch -c -x "build --workspace"; \
	else \
		printf "$(RED) cargo-watch not installed. Run 'make setup' first.$(NC)\n"; \
	fi

watch-test: ## Watch for changes and run tests
	@printf "$(BLUE) Watching for changes and running tests...$(NC)\n"
	@if command -v cargo-watch >/dev/null 2>&1; then \
		cargo watch -c -x "test --workspace"; \
	else \
		printf "$(RED) cargo-watch not installed. Run 'make setup' first.$(NC)\n"; \
	fi

watch-crate: ## Watch specific crate (usage: make watch-crate CRATE=robust-core)
	@if [ -z "$(CRATE)" ]; then \
		printf "$(RED) Please specify CRATE=<crate-name>$(NC)\n"; \
		exit 1; \
	fi
	@printf "$(BLUE) Watching $(CRATE) for changes...$(NC)\n"
	@if command -v cargo-watch >/dev/null 2>&1; then \
		cargo watch -c -x "test -p $(CRATE)"; \
	else \
		printf "$(RED) cargo-watch not installed. Run 'make setup' first.$(NC)\n"; \
	fi

# Coverage and benchmarks
coverage: ## Generate test coverage report
	@printf "$(BLUE) Generating coverage report...$(NC)\n"
	@if command -v cargo-tarpaulin >/dev/null 2>&1; then \
		cargo tarpaulin --workspace --all-features --timeout 120 --out Html; \
		printf "$(GREEN) Coverage report generated: tarpaulin-report.html$(NC)\n"; \
	else \
		printf "$(RED) cargo-tarpaulin not installed. Run 'make setup' first.$(NC)\n"; \
	fi

bench: ## Run all benchmarks
	@printf "$(BLUE) Running benchmarks...$(NC)\n"
	cargo bench --workspace

bench-crate: ## Benchmark specific crate (usage: make bench-crate CRATE=robust-core)
	@if [ -z "$(CRATE)" ]; then \
		printf "$(RED) Please specify CRATE=<crate-name>$(NC)\n"; \
		exit 1; \
	fi
	@printf "$(BLUE) Benchmarking $(CRATE)...$(NC)\n"
	cargo bench -p $(CRATE)

bench-quantile: ## Run quantile estimator benchmarks
	@printf "$(BLUE) Benchmarking quantile estimators...$(NC)\n"
	cargo bench -p robust-quantile --bench quantile_benchmarks

# Security and maintenance
audit: ## Run security audit on dependencies
	@printf "$(BLUE) Running security audit...$(NC)\n"
	@if command -v cargo-audit >/dev/null 2>&1; then \
		cargo audit; \
	else \
		printf "$(RED) cargo-audit not installed. Run 'make setup' first.$(NC)\n"; \
	fi

outdated: ## Check for outdated dependencies
	@printf "$(BLUE) Checking for outdated dependencies...$(NC)\n"
	@if command -v cargo-outdated >/dev/null 2>&1; then \
		cargo outdated --workspace; \
	else \
		printf "$(RED) cargo-outdated not installed. Run 'make setup' first.$(NC)\n"; \
	fi

update: ## Update dependencies
	@printf "$(BLUE) Updating dependencies...$(NC)\n"
	cargo update

# Setup and cleanup
setup: ## Install development tools
	@printf "$(BLUE) Setting up development environment...$(NC)\n"
	rustup component add rustfmt clippy
	@printf "Installing development tools...\n"
	@for tool in cargo-watch cargo-edit cargo-audit cargo-outdated cargo-tarpaulin cargo-workspaces; do \
		if ! command -v $$tool >/dev/null 2>&1; then \
			printf "Installing $$tool...\n"; \
			cargo install $$tool || printf "$(YELLOW) Failed to install $$tool$(NC)\n"; \
		else \
			printf "$(GREEN) $$tool already installed$(NC)\n"; \
		fi; \
	done
	@printf "$(GREEN) Development environment ready!$(NC)\n"

clean: ## Clean build artifacts
	@printf "$(BLUE) Cleaning build artifacts...$(NC)\n"
	cargo clean

clean-all: clean ## Clean everything including Cargo.lock
	@printf "$(BLUE) Deep cleaning...$(NC)\n"
	rm -f Cargo.lock
	find . -name "*.orig" -type f -delete
	find . -name "*.rej" -type f -delete

# Release management
verify: ## Verify workspace is ready for release
	@printf "$(BLUE) Verifying workspace...$(NC)\n"
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings
	cargo test --workspace --all-features
	cargo doc --workspace --all-features --no-deps
	@printf "$(GREEN) Workspace verification complete!$(NC)\n"

package: ## Create packages for all crates
	@printf "$(BLUE) Creating packages...$(NC)\n"
	@for crate in $(CRATES); do \
		printf "Packaging $$crate...\n"; \
		(cd crates/$$crate && cargo package --allow-dirty) || exit 1; \
	done

publish-dry-run: verify ## Dry run of publishing all crates
	@printf "$(BLUE) Dry run publishing...$(NC)\n"
	@if command -v cargo-workspaces >/dev/null 2>&1; then \
		cargo workspaces publish --dry-run; \
	else \
		printf "$(RED) cargo-workspaces not installed. Run 'make setup' first.$(NC)\n"; \
	fi

publish: verify ## Publish all crates to crates.io
	@printf "$(BLUE) Publishing to crates.io...$(NC)\n"
	@printf "$(YELLOW) This will publish all crates. Continue? [y/N] $(NC)" && read ans && [ $${ans:-N} = y ]
	@if command -v cargo-workspaces >/dev/null 2>&1; then \
		cargo workspaces publish; \
	else \
		printf "$(RED) cargo-workspaces not installed. Run 'make setup' first.$(NC)\n"; \
	fi

release: prod ## Prepare for release
	@printf "$(BLUE) Preparing release...$(NC)\n"
	@printf "$(GREEN) Release preparation complete!$(NC)\n"
	@printf "$(BLUE)Next steps:$(NC)\n"
	@printf "  1. Update version numbers in all Cargo.toml files\n"
	@printf "  2. Update CHANGELOG.md\n"
	@printf "  3. Commit all changes\n"
	@printf "  4. Tag the release: git tag -a v0.1.0 -m 'Release v0.1.0'\n"
	@printf "  5. Push tags: git push origin main --tags\n"
	@printf "  6. Run 'make publish' to publish to crates.io\n"

# Utility targets
size: build build-release ## Show binary sizes for examples
	@printf "$(BLUE) Binary sizes:$(NC)\n"
	@find target -name "*.rlib" -type f | xargs ls -lh | grep -E "(debug|release)" | sort -k5 -h

deps: ## Show dependency tree for workspace
	@printf "$(BLUE) Dependency tree:$(NC)\n"
	cargo tree --workspace

tree: ## Show crate dependency graph
	@printf "$(BLUE) Crate dependency graph:$(NC)\n"
	cargo tree --workspace --edges no-dev

stats: ## Show project statistics
	@printf "$(BLUE) Project Statistics:$(NC)\n"
	@printf "\nLines of Rust code per crate:\n"
	@for crate in $(CRATES); do \
		if [ -d "crates/$$crate/src" ]; then \
			lines=$$(find crates/$$crate/src -name "*.rs" -type f | xargs wc -l | tail -1 | awk '{print $$1}'); \
			printf "  $(CYAN)%-20s$(NC) %6d lines\n" "$$crate:" "$$lines"; \
		fi; \
	done
	@printf "\nTotal lines of Rust code:\n"
	@find crates -name "*.rs" -type f | xargs wc -l | tail -1
	@printf "\nNumber of dependencies:\n"
	@cargo tree --workspace --edges no-dev | grep -E "^[a-zA-Z]" | wc -l
	@printf "\nWorkspace crates: $(words $(CRATES))\n"

# Crate-specific shortcuts
core: ## Work with robust-core
	@$(MAKE) test-crate CRATE=robust-core

quantile: ## Work with robust-quantile
	@$(MAKE) test-crate CRATE=robust-quantile

spread: ## Work with robust-spread
	@$(MAKE) test-crate CRATE=robust-spread

confidence: ## Work with robust-confidence
	@$(MAKE) test-crate CRATE=robust-confidence

modality: ## Work with robust-modality
	@$(MAKE) test-crate CRATE=robust-modality

changepoint: ## Work with robust-changepoint
	@$(MAKE) test-crate CRATE=robust-changepoint

stability: ## Work with robust-stability
	@$(MAKE) test-crate CRATE=robust-stability

viz: ## Work with robust-viz
	@$(MAKE) test-crate CRATE=robust-viz

polars: ## Work with robust-polars
	@$(MAKE) test-crate CRATE=robust-polars

# Installation helpers
install-modality: ## Install robust-modality binary
	@printf "$(BLUE) Installing robust-modality...$(NC)\n"
	cargo install --path crates/robust-modality

# CI/CD helpers
ci: fmt-check lint test-all ## Run CI checks
	@printf "$(GREEN) CI checks passed!$(NC)\n"

# Graph visualization
graph: ## Generate and open dependency graph
	@printf "$(BLUE) Generating dependency graph...$(NC)\n"
	@if command -v cargo-depgraph >/dev/null 2>&1; then \
		cargo depgraph --workspace | dot -Tpng > dependency-graph.png; \
		printf "$(GREEN) Graph saved to dependency-graph.png$(NC)\n"; \
	else \
		printf "$(RED) cargo-depgraph not installed. Install with: cargo install cargo-depgraph$(NC)\n"; \
	fi
