# MedRAX Makefile
# Convenient commands for Docker management

.PHONY: help build up down logs restart clean shell test health

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "MedRAX Docker Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Initial setup (create .env from example)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file. Please edit it and add your OPENAI_API_KEY"; \
	else \
		echo "⚠️  .env already exists"; \
	fi
	@mkdir -p temp logs
	@chmod 777 temp logs

build: ## Build Docker image
	docker-compose build

up: ## Start services in background
	docker-compose up -d

down: ## Stop and remove containers
	docker-compose down

logs: ## View logs (follow mode)
	docker-compose logs -f medrax

restart: ## Restart the service
	docker-compose restart

stop: ## Stop services without removing
	docker-compose stop

start: ## Start existing containers
	docker-compose start

ps: ## Show running containers
	docker-compose ps

shell: ## Open shell in the container
	docker-compose exec medrax bash

python: ## Open Python shell in the container
	docker-compose exec medrax python3

health: ## Check service health
	@echo "Checking MedRAX health..."
	@curl -f http://localhost:8585/ && echo "✅ Service is healthy" || echo "❌ Service is not responding"

gpu: ## Check GPU availability in container
	docker-compose exec medrax python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"

clean: ## Stop containers and remove volumes
	docker-compose down -v

clean-all: ## Remove containers, volumes, and images
	docker-compose down -v --rmi all

rebuild: ## Rebuild without cache
	docker-compose build --no-cache

update: ## Pull latest code and rebuild
	git pull
	docker-compose up -d --build

test-dicom: ## Test DICOM processing
	docker-compose exec medrax python3 test_dicom_fix.py

inspect: ## Inspect container configuration
	docker-compose exec medrax env

stats: ## Show container resource usage
	docker stats medrax-agent

prune: ## Clean up Docker system
	docker system prune -a --volumes

dev: ## Start in development mode (with volume mounts)
	docker-compose up

quick-start: setup build up logs ## Complete setup and start
