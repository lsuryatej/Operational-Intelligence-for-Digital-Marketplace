# Makefile for Marketplace Late Delivery Prediction

.PHONY: setup train evaluate test serve docker-build docker-run clean help

help:
	@echo "Available commands:"
	@echo "  make setup        - Install dependencies"
	@echo "  make train        - Train the model and baseline"
	@echo "  make evaluate     - Run evaluation and generate report"
	@echo "  make test         - Run API integration tests"
	@echo "  make serve        - Start the FastAPI server locally"
	@echo "  make docker-build - Build the Docker image"
	@echo "  make docker-run   - Run the Docker container"
	@echo "  make audit        - Run diagnostic audit experiments"
	@echo "  make clean        - Remove temporary files and artifacts"

setup:
	pip install -r requirements.txt

train:
	python src/train.py

evaluate:
	python src/evaluate.py

test:
	python -m pytest tests/test_api.py -v

serve:
	uvicorn src.serve:app --reload --port 8000

docker-build:
	docker build -t marketplace-ml .

docker-run:
	docker run -p 8000:8000 marketplace-ml

audit:
	python experiments/run_audit_experiments.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
