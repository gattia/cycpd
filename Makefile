autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	flake8