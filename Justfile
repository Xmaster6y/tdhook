install:
	uv run pre-commit install
	uv sync

checks:
	uv run pre-commit run --all-files

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

docs:
	cd docs && uv run --group docs make html
