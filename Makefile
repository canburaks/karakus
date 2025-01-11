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

purge:
	@echo "Purging code..."
	uv run ruff clean

add:
	@echo "Adding dependencies..."
	uv run ruff add 
