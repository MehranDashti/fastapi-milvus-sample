.PHONY: run lint format fix test

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 9090

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/

fix:
	ruff check --fix app/ tests/

test:
	pytest tests/ -v
