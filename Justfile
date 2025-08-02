install:
	uv run pre-commit install
	uv sync

checks:
	uv run pre-commit run --all-files

test-assets:
	@echo "No test assets to resolve"

tests:
	uv run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=50 -s -v

wandb-sync:
	uv run --no-sync wandb sync --sync-all

launch cluster script *args:
    sbatch launch/{{cluster}}/{{script}}.sh {{args}}

run script *args:
    uv run -m scripts.{{script}} {{args}}

docs:
	cd docs && uv run --group docs make html
