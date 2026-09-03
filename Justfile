install:
	uv run pre-commit install
	uv sync

checks:
	uv run pre-commit run --all-files

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=50 -s -v

benchmark-smoke:
	uv run python -m benchmarks.run --mode smoke --device cpu

benchmark-full device="auto":
	uv run python -m benchmarks.run --mode full --device {{device}}

notebook-tests:
	uv run pytest tests/test_demo_notebooks.py -m integration -v

docs:
	cd docs && uv run --group docs make html

build:
	uv build

release-check: checks tests docs build

release-smoke wheel version:
	uv run --isolated --no-project --with "{{wheel}}" python scripts/release_smoke.py "{{version}}"
