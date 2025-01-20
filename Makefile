.PHONY: dev


dev:
	@echo "Running development server..."
	uv run fastapi dev ./api/main.py 

serve:
	@echo "Running production server..."
	uv run -m uvicorn api.main:app --reload 

test:
	@echo "Running tests..."
	uv run pytest --no-header -vv

format:
	@echo "Formatting code..."
	uv run ruff format .
	@echo "Sorting imports..."
	uv run isort . 

typecheck:
	@echo "Type checking..."
	uv run mypy . 

lint:
	@echo "Linting code..."
	uv run ruff check --select I --fix 

sync:
	@echo "Linting code..."
	uv sync

add:
	@echo "Adding dependencies..."
	uv run ruff add 

langflow:
	@echo "Running langflow..."
	uv pip install langflow && uv run langflow run

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	uv run ruff clean

