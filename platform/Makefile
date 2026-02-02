# Ultimate Autonomous Platform - Makefile
# Common commands for platform management
#
# Usage:
#   make help          - Show all available commands
#   make start         - Start the platform
#   make stop          - Stop the platform
#   make status        - Check component health
#   make test          - Run all tests
#   make benchmark     - Run performance benchmarks

.PHONY: help start stop status test benchmark demo clean logs

# Default shell
SHELL := /bin/bash

# Project paths
PROJECT_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts
SWARM_DIR := $(PROJECT_ROOT)/swarm
BRIDGES_DIR := $(PROJECT_ROOT)/bridges
DEMO_DIR := $(PROJECT_ROOT)/demo
DEPLOY_DIR := $(PROJECT_ROOT)/deploy

# Python command
PYTHON := python

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#==============================================================================
# HELP
#==============================================================================

help:
	@echo "============================================================"
	@echo "  Ultimate Autonomous Platform - Commands"
	@echo "============================================================"
	@echo ""
	@echo "  Infrastructure:"
	@echo "    make start         Start all Docker services"
	@echo "    make stop          Stop all Docker services"
	@echo "    make restart       Restart all services"
	@echo "    make logs          Follow service logs"
	@echo ""
	@echo "  Monitoring:"
	@echo "    make status        Check platform health"
	@echo "    make dashboard     Run status dashboard"
	@echo ""
	@echo "  Testing:"
	@echo "    make test          Run all tests"
	@echo "    make test-hooks    Run hook integration tests"
	@echo "    make test-swarm    Run swarm coordinator tests"
	@echo "    make test-parallel Run parallel execution tests"
	@echo ""
	@echo "  Performance:"
	@echo "    make benchmark     Run full benchmark suite"
	@echo "    make demo          Run E2E workflow demo"
	@echo ""
	@echo "  Development:"
	@echo "    make lint          Run linters"
	@echo "    make clean         Clean temporary files"
	@echo ""
	@echo "============================================================"

#==============================================================================
# INFRASTRUCTURE
#==============================================================================

start:
	@echo "$(GREEN)[>>] Starting platform services...$(NC)"
	cd $(DEPLOY_DIR) && docker-compose up -d
	@echo "$(GREEN)[OK] Services started. Run 'make status' to check health.$(NC)"

stop:
	@echo "$(YELLOW)[>>] Stopping platform services...$(NC)"
	cd $(DEPLOY_DIR) && docker-compose down
	@echo "$(GREEN)[OK] Services stopped.$(NC)"

restart: stop start

logs:
	cd $(DEPLOY_DIR) && docker-compose logs -f

logs-qdrant:
	cd $(DEPLOY_DIR) && docker-compose logs -f qdrant

logs-neo4j:
	cd $(DEPLOY_DIR) && docker-compose logs -f neo4j

logs-letta:
	cd $(DEPLOY_DIR) && docker-compose logs -f letta

#==============================================================================
# MONITORING
#==============================================================================

status:
	@echo "$(GREEN)[>>] Checking platform health...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/platform_orchestrator.py status

dashboard:
	@echo "$(GREEN)[>>] Running status dashboard...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/status_dashboard.py

#==============================================================================
# TESTING
#==============================================================================

test: test-hooks test-swarm test-parallel
	@echo "$(GREEN)[OK] All tests completed!$(NC)"

test-hooks:
	@echo "$(GREEN)[>>] Running hook integration tests...$(NC)"
	$(PYTHON) $(PROJECT_ROOT)/hooks/test_integration.py

test-swarm:
	@echo "$(GREEN)[>>] Running swarm coordinator tests...$(NC)"
	$(PYTHON) $(SWARM_DIR)/test_coordinator.py

test-parallel:
	@echo "$(GREEN)[>>] Running parallel execution tests...$(NC)"
	$(PYTHON) $(SWARM_DIR)/test_parallel_agents.py --workers 5 --tasks 25

test-qdrant:
	@echo "$(GREEN)[>>] Running live Qdrant tests...$(NC)"
	$(PYTHON) $(SWARM_DIR)/test_live_qdrant.py

#==============================================================================
# PERFORMANCE
#==============================================================================

benchmark:
	@echo "$(GREEN)[>>] Running performance benchmarks...$(NC)"
	$(PYTHON) $(SWARM_DIR)/benchmark_coordinator.py

benchmark-scaling:
	@echo "$(GREEN)[>>] Running scaling benchmark...$(NC)"
	$(PYTHON) $(SWARM_DIR)/test_parallel_agents.py --scaling

demo:
	@echo "$(GREEN)[>>] Running E2E workflow demo...$(NC)"
	$(PYTHON) $(DEMO_DIR)/e2e_workflow_demo.py --verbose

#==============================================================================
# DEVELOPMENT
#==============================================================================

lint:
	@echo "$(GREEN)[>>] Running linters...$(NC)"
	-ruff check $(PROJECT_ROOT)
	-mypy $(PROJECT_ROOT) --ignore-missing-imports

format:
	@echo "$(GREEN)[>>] Formatting code...$(NC)"
	-ruff format $(PROJECT_ROOT)

clean:
	@echo "$(YELLOW)[>>] Cleaning temporary files...$(NC)"
	find $(PROJECT_ROOT) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(PROJECT_ROOT) -type f -name "*.pyc" -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type f -name "*.pyo" -delete 2>/dev/null || true
	find $(PROJECT_ROOT) -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)[OK] Cleanup complete.$(NC)"

#==============================================================================
# INITIALIZATION
#==============================================================================

init:
	@echo "$(GREEN)[>>] Initializing platform...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/platform_orchestrator.py start
	@echo "$(GREEN)[OK] Platform initialized.$(NC)"

install-deps:
	@echo "$(GREEN)[>>] Installing Python dependencies...$(NC)"
	pip install httpx structlog qdrant-client neo4j rich

#==============================================================================
# DOCKER UTILITIES
#==============================================================================

docker-ps:
	docker ps --filter "label=com.uap.component"

docker-volumes:
	docker volume ls --filter "label=com.uap.volume"

docker-clean:
	@echo "$(RED)[!!] This will remove all platform volumes!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		docker-compose -f $(DEPLOY_DIR)/docker-compose.yml down -v || \
		echo "Cancelled."
